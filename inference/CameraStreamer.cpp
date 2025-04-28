#include "CameraStreamer.hpp"

// Constructor
CameraStreamer::CameraStreamer(TensorRTInferencer& infer, double scale, const std::string& win_name, bool show_orig)
    : scale_factor(scale), window_name(win_name), inferencer(infer), show_original(show_orig) {

    std::string pipeline = "nvarguscamerasrc sensor-mode=4 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
    cap.open(pipeline, cv::CAP_GSTREAMER);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open CSI camera" << std::endl;
        exit(-1);
    }
}

void CameraStreamer::initOpenGL() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        exit(-1);
    }

    window_width = static_cast<int>(1280 * scale_factor);
    window_height = static_cast<int>(720 * scale_factor);

    window = glfwCreateWindow(window_width, window_height, window_name.c_str(), NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);
    glewInit();

    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // ⚡ Register the texture with CUDA-OpenGL interop
    cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_resource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Failed to register OpenGL texture with CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

/* void CameraStreamer::uploadFrameToTexture(const cv::Mat& frame) {
    cv::Mat frame_rgb;
    cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);

    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame_rgb.cols, frame_rgb.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb.data);
} */

void CameraStreamer::uploadFrameToTexture(const cv::cuda::GpuMat& gpuFrame) {
    // Map the OpenGL texture to CUDA
    cudaGraphicsMapResources(1, &cuda_resource, 0);

    // Get the CUDA array associated with the OpenGL texture
    cudaArray_t texture_ptr;
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_resource, 0, 0);

    // Copy directly from GpuMat to the texture array
    cudaMemcpy2DToArray(
        texture_ptr,
        0, 0,
        gpuFrame.ptr(),
        gpuFrame.step,
        gpuFrame.cols * gpuFrame.elemSize(),
        gpuFrame.rows,
        cudaMemcpyDeviceToDevice
    );

    // Unmap the resource so OpenGL can use it
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);
}

void CameraStreamer::renderTexture() {
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 1); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1, 1); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1, 0); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0, 0); glVertex2f(-1.0f, 1.0f);
    glEnd();

    glfwSwapBuffers(window);
    glfwPollEvents();
}

// Add to your class constructor or initialization function
void CameraStreamer::initUndistortMaps() {
    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage fs("camera_calibration.yml", cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // Create undistortion maps
    cv::Mat mapx, mapy;
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix,
                               cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
                               CV_32FC1, mapx, mapy);

    // Upload maps to GPU (store as class members)
    d_mapx.upload(mapx);
    d_mapy.upload(mapy);
}

/* void CameraStreamer::start() {
    initUndistortMaps(); // Initialize undistortion maps
    initOpenGL();        // Initialize OpenGL and GLFW

    cv::Mat frame;

    // Load camera calibration data once, not in the loop
    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage fs("camera_calibration.yml", cv::FileStorage::READ);
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    // Create CUDA stream for asynchronous operations
    cv::cuda::Stream stream;

    while (!glfwWindowShouldClose(window)) {
        cap >> frame;
        if (frame.empty())
            break;

        // Upload to GPU
        cv::cuda::GpuMat d_frame(frame);
        cv::cuda::GpuMat d_undistorted;

        // Undistort on GPU
        cv::cuda::remap(d_frame, d_undistorted, d_mapx, d_mapy, cv::INTER_LINEAR, 0, cv::Scalar(), stream);

        // Run inference on the frame
        cv::Mat undistorted;
        d_undistorted.download(undistorted, stream);
        cv::Mat prediction_mask = inferencer.makePrediction(undistorted);

        // Upload prediction mask to GPU
        cv::cuda::GpuMat d_prediction_mask(prediction_mask);
        cv::cuda::GpuMat d_visualization;

        // Convert to 8-bit on GPU
        d_prediction_mask.convertTo(d_visualization, CV_8U, 255.0, 0, stream);

        // Download for color mapping (if not available in CUDA)
        cv::Mat visualization;
        d_visualization.download(visualization, stream);

        // Apply color map
        cv::Mat colorized_mask;
        cv::applyColorMap(visualization, colorized_mask, cv::COLORMAP_JET);

        // Upload for resizing
        cv::cuda::GpuMat d_colorized_mask(colorized_mask);
        cv::cuda::GpuMat d_resized_mask;

        // Resize on GPU
        cv::cuda::resize(d_colorized_mask, d_resized_mask,
                         cv::Size(frame.cols * scale_factor, frame.rows * scale_factor),
                         0, 0, cv::INTER_LINEAR, stream);

        // Download for OpenGL rendering
        cv::Mat resized_mask;
        d_resized_mask.download(resized_mask, stream);

        if (show_original) {
            // Resize original on GPU too
            cv::cuda::GpuMat d_resized_frame;
            cv::cuda::resize(d_frame, d_resized_frame,
                             cv::Size(frame.cols * scale_factor, frame.rows * scale_factor),
                             0, 0, cv::INTER_LINEAR, stream);

            cv::Mat resized_frame;
            d_resized_frame.download(resized_frame, stream);

            // Make sure both images have the same type before concatenation
            if (resized_frame.type() != resized_mask.type()) {
                cv::cvtColor(resized_mask, resized_mask, cv::COLOR_BGR2BGRA);
                cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2BGRA);
            }

            // Concatenate frames
            cv::Mat combined;
            try {
                cv::hconcat(resized_frame, resized_mask, combined);
                uploadFrameToTexture(combined);  // Upload combined image to OpenGL texture
            } catch (const cv::Exception& e) {
                std::cerr << "Warning: Could not concatenate images: " << e.what() << std::endl;
                uploadFrameToTexture(resized_mask);  // Fallback
            }
        } else {
            uploadFrameToTexture(resized_mask);  // Upload only the prediction mask
        }

        renderTexture();  // Render the frame using OpenGL
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
} */

void CameraStreamer::start() {
    initUndistortMaps(); // Undistortion maps on GPU
    initOpenGL();        // OpenGL window setup

    cv::Mat frame;
    cv::cuda::Stream stream;

    while (!glfwWindowShouldClose(window)) {
        cap >> frame;
        if (frame.empty()) break;

        // Upload captured frame to GPU
        cv::cuda::GpuMat d_frame(frame);

        // Undistort directly on GPU
        cv::cuda::GpuMat d_undistorted;
        cv::cuda::remap(d_frame, d_undistorted, d_mapx, d_mapy, cv::INTER_LINEAR, 0, cv::Scalar(), stream);

        // Run inference — returns GpuMat directly
        cv::cuda::GpuMat d_prediction_mask = inferencer.makePrediction(d_undistorted);

        // Convert prediction to 8-bit
        cv::cuda::GpuMat d_visualization;
        d_prediction_mask.convertTo(d_visualization, CV_8U, 255.0, 0, stream);

        // Apply color map using CPU (download + CPU colormap + upload)
        cv::Mat visualization_cpu;
        d_visualization.download(visualization_cpu, stream);  // Download 8-bit mask to CPU

        stream.waitForCompletion();  // Ensure download is complete before CPU access

        cv::Mat colorized_mask_cpu;
        cv::applyColorMap(visualization_cpu, colorized_mask_cpu, cv::COLORMAP_JET);  // Apply colormap on CPU

        cv::cuda::GpuMat d_colorized_mask;
        d_colorized_mask.upload(colorized_mask_cpu);  // Upload back to GPU

        // Resize for display
        cv::cuda::GpuMat d_resized_mask;
        cv::cuda::resize(d_colorized_mask, d_resized_mask,
                         cv::Size(frame.cols * scale_factor, frame.rows * scale_factor),
                         0, 0, cv::INTER_LINEAR, stream);

        // Upload to OpenGL
        uploadFrameToTexture(d_resized_mask);  // Using your CUDA-OpenGL interop method

        // Display
        renderTexture();

        // Optional: limit to ~30 FPS
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();
}

// Destructor
CameraStreamer::~CameraStreamer() {
    cap.release();
    cv::destroyAllWindows();
}
