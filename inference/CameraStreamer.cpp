#include "CameraStreamer.hpp"

// Constructor: initializes camera capture, inference reference, and settings
CameraStreamer::CameraStreamer(TensorRTInferencer& infer, double scale, const std::string& win_name, bool show_orig)
    : scale_factor(scale), window_name(win_name), inferencer(infer), show_original(show_orig) {

    // Define GStreamer pipeline for CSI camera
    std::string pipeline = "nvarguscamerasrc sensor-mode=4 ! video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=60/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";

    cap.open(pipeline, cv::CAP_GSTREAMER); // Open camera stream with GStreamer

    if (!cap.isOpened()) {  // Check if camera opened successfully
        std::cerr << "Error: Could not open CSI camera" << std::endl;
        exit(-1);  // Terminate if failed
    }
}

// Destructor: clean up resources
CameraStreamer::~CameraStreamer() {
    cap.release();  // Release camera
    cv::destroyAllWindows();  // Close OpenCV windows

    if (cuda_resource) {
        cudaGraphicsUnregisterResource(cuda_resource);  // Unregister CUDA graphics resource
        cuda_resource = nullptr;
    }

    if (textureID) {
        glDeleteTextures(1, &textureID);  // Delete OpenGL texture
        textureID = 0;
    }

    if (window) {
        glfwDestroyWindow(window);  // Destroy OpenGL window
        window = nullptr;
    }

    glfwTerminate();  // Shutdown GLFW
}

// Initialize OpenGL context and prepare a texture for CUDA interop
void CameraStreamer::initOpenGL() {
    if (!glfwInit()) {  // Initialize GLFW library
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        exit(-1);
    }

    window_width = static_cast<int>(1280 * scale_factor);  // Calculate scaled window width
    window_height = static_cast<int>(720 * scale_factor);  // Calculate scaled window height

    window = glfwCreateWindow(window_width, window_height, window_name.c_str(), NULL, NULL);  // Create OpenGL window
    if (!window) {  // Check if window creation failed
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(window);  // Make window's OpenGL context current
    glewInit();  // Initialize GLEW (needed to manage OpenGL extensions)

    glGenTextures(1, &textureID);  // Generate OpenGL texture ID
    glBindTexture(GL_TEXTURE_2D, textureID);  // Bind the texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // Set texture minification filter
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  // Set texture magnification filter

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);  // Allocate empty texture (RGBA8 format)

    // Register the OpenGL texture with CUDA
    cudaError_t err = cudaGraphicsGLRegisterImage(&cuda_resource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {  // Check CUDA-OpenGL interop registration
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

// Upload a GpuMat frame (on GPU) directly into an OpenGL texture using CUDA interop
void CameraStreamer::uploadFrameToTexture(const cv::cuda::GpuMat& gpuFrame) {
    cv::cuda::GpuMat d_rgba_frame;
    if (gpuFrame.channels() == 3) {  // If input is BGR
        cv::cuda::cvtColor(gpuFrame, d_rgba_frame, cv::COLOR_BGR2RGBA);  // Convert BGR to RGBA
    } else if (gpuFrame.channels() == 1) {  // If grayscale
        cv::cuda::cvtColor(gpuFrame, d_rgba_frame, cv::COLOR_GRAY2RGBA);  // Convert grayscale to RGBA
    } else {
        d_rgba_frame = gpuFrame;  // Already RGBA
    }

    cudaGraphicsMapResources(1, &cuda_resource, 0);  // Map OpenGL texture for CUDA access

    cudaArray_t texture_ptr;
    cudaGraphicsSubResourceGetMappedArray(&texture_ptr, cuda_resource, 0, 0);  // Get CUDA array pointer linked to OpenGL texture

    cudaMemcpy2DToArray(
        texture_ptr,
        0, 0,
        d_rgba_frame.ptr(),      // Source pointer (GPU memory)
        d_rgba_frame.step,       // Source stride
        d_rgba_frame.cols * d_rgba_frame.elemSize(), // Width in bytes
        d_rgba_frame.rows,       // Height in rows
        cudaMemcpyDeviceToDevice // GPU-to-GPU memory copy
    );

    cudaGraphicsUnmapResources(1, &cuda_resource, 0);  // Unmap resource after copy
}

// Render the current OpenGL texture to the screen
void CameraStreamer::renderTexture() {
    glClear(GL_COLOR_BUFFER_BIT);  // Clear the screen
    glEnable(GL_TEXTURE_2D);  // Enable 2D texturing
    glBindTexture(GL_TEXTURE_2D, textureID);  // Bind the texture to use

    glBegin(GL_QUADS);  // Start drawing a rectangle
    glTexCoord2f(0, 1); glVertex2f(-1.0f, -1.0f);  // Bottom-left
    glTexCoord2f(1, 1); glVertex2f(1.0f, -1.0f);   // Bottom-right
    glTexCoord2f(1, 0); glVertex2f(1.0f, 1.0f);    // Top-right
    glTexCoord2f(0, 0); glVertex2f(-1.0f, 1.0f);   // Top-left
    glEnd();  // End drawing rectangle

    glfwSwapBuffers(window);  // Swap front and back buffers (double buffering)
    glfwPollEvents();  // Process window events
}

// Load camera calibration file and initialize undistortion maps (upload to GPU)
void CameraStreamer::initUndistortMaps() {
    cv::Mat cameraMatrix, distCoeffs;
    cv::FileStorage fs("camera_calibration.yml", cv::FileStorage::READ);  // Open calibration file
    fs["camera_matrix"] >> cameraMatrix;  // Read camera matrix
    fs["distortion_coefficients"] >> distCoeffs;  // Read distortion coefficients
    fs.release();  // Close file

    cv::Mat mapx, mapy;
    cv::initUndistortRectifyMap(
        cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix,
        cv::Size(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
        CV_32FC1, mapx, mapy
    );  // Compute undistortion mapping

    d_mapx.upload(mapx);  // Upload X map to GPU
    d_mapy.upload(mapy);  // Upload Y map to GPU
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

// Main loop: capture, undistort, predict, visualize and render frames
void CameraStreamer::start() {
    initUndistortMaps();  // Initialize camera undistortion maps
    initOpenGL();  // Initialize OpenGL and CUDA interop

    cv::Mat frame;
    cv::cuda::Stream stream;  // CUDA stream for asynchronous operations

    const int framesToSkip = 2;  // Skip frames to reduce processing load

    while (!glfwWindowShouldClose(window)) {  // Main loop until window closed
        for (int i = 0; i < framesToSkip; ++i) {
            cap.grab();  // Grab frames without decoding
        }
        cap >> frame;  // Read one frame (decoded)

        if (frame.empty()) break;  // Stop if frame is invalid

        cv::cuda::GpuMat d_frame(frame);  // Upload frame to GPU
        cv::cuda::GpuMat d_undistorted;
        cv::cuda::remap(d_frame, d_undistorted, d_mapx, d_mapy, cv::INTER_LINEAR, 0, cv::Scalar(), stream);  // Undistort frame

        cv::cuda::GpuMat d_prediction_mask = inferencer.makePrediction(d_undistorted);  // Run model inference
        cv::cuda::GpuMat d_visualization;
        d_prediction_mask.convertTo(d_visualization, CV_8U, 255.0, 0, stream);  // Normalize prediction mask

        cv::Mat visualization_cpu;
        d_visualization.download(visualization_cpu, stream);  // Download mask to CPU
        stream.waitForCompletion();  // Ensure async operations are complete

        cv::Mat colorized_mask_cpu;
        cv::applyColorMap(visualization_cpu, colorized_mask_cpu, cv::COLORMAP_JET);  // Apply color map

        cv::cuda::GpuMat d_colorized_mask;
        d_colorized_mask.upload(colorized_mask_cpu);  // Upload colored mask back to GPU

        cv::cuda::GpuMat d_resized_mask;
        cv::cuda::resize(d_colorized_mask, d_resized_mask,
                         cv::Size(frame.cols * scale_factor, frame.rows * scale_factor),
                         0, 0, cv::INTER_LINEAR, stream);  // Resize for display
        stream.waitForCompletion();  // Synchronize

        uploadFrameToTexture(d_resized_mask);  // Upload final result to OpenGL
        renderTexture();  // Render it

        std::this_thread::sleep_for(std::chrono::milliseconds(33));  // Frame delay (~30 FPS)
    }

    glfwDestroyWindow(window);  // Clean up window
    glfwTerminate();  // Terminate GLFW
}
