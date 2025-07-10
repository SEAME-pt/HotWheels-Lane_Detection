
#include "CameraStreamer.hpp"

/**
 * @brief Constructs and initializes the complete CameraStreamer vision pipeline
 * @param scale Frame scaling factor (0.0-1.0) for performance vs. quality balance
 * @details This constructor performs comprehensive system initialization including:
 *
 * 			1. **AI Model Loading**: Initializes both TensorRT segmentation engine for lane
 * 			   detection and YOLOv5 engine for object detection with pre-trained models
 * 			2. **Camera Pipeline Setup**: Configures hardware-accelerated GStreamer pipeline
 * 			   optimized for CSI camera with NVMM memory and format conversions
 * 			3. **Hardware Validation**: Verifies camera accessibility and terminates gracefully
 * 			   if hardware is unavailable
 *
 * 			**GStreamer Pipeline Breakdown**:
 * 			- `nvarguscamerasrc`: Jetson-optimized CSI camera source with sensor mode 4
 * 			- `video/x-raw(memory:NVMM)`: Hardware memory allocation for zero-copy operations
 * 			- `nvvidconv`: Hardware-accelerated color space conversion
 * 			- `videoconvert`: Software fallback for format compatibility
 * 			- `appsink`: Application sink with optimized buffering (drop=1, buffers=1)
 *
 * 			**Model Paths**:
 * 			- Segmentation: `/home/jetson/models/lane-detection/model.engine`
 * 			- Object Detection: `/home/jetson/models/object-detection/yolov5m_updated.engine`
 *
 * @note Terminates with exit(-1) if camera initialization fails - this ensures
 * 			system reliability in autonomous vehicle applications where vision is critical
 *
 * @warning Requires Jetson platform with CSI camera and pre-deployed TensorRT models
 */
CameraStreamer::CameraStreamer (double scale)
    : scale_factor (scale), m_publisherFrameObject (nullptr), m_running (true),
      m_polyfitter (std::make_unique<Polyfitter> ()) {

	segmentationInferencer = std::make_shared<TensorRTInferencer>("/home/jetson/models/lane-detection/model.engine");
	yoloInferencer = std::make_shared<YOLOv5TRT>("/home/jetson/models/object-detection/yolov5m_updated.engine", "/home/jetson/models/object-detection/labels.txt");

	// Define GStreamer pipeline for CSI camera - EGL-free version for headless operation
	std::string pipeline = "nvarguscamerasrc sensor-mode=4 ! "
			"video/x-raw(memory:NVMM), width=1280, height=720, "
			"format=(string)NV12, framerate=30/1 ! "
			"nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! "
			"videoconvert ! video/x-raw, format=(string)BGR ! "
			"appsink drop=1 buffers=1 sync=false";

	std::cout << "[CameraStreamer] Using GStreamer pipeline: " << pipeline << std::endl;

	// Try the NVMM pipeline first
	cap.open(pipeline, cv::CAP_GSTREAMER);

	// If NVMM pipeline fails, fallback to basic USB camera
	if (!cap.isOpened()) {
		std::cout << "[CameraStreamer] NVMM pipeline failed, trying USB camera fallback..." << std::endl;
		pipeline = "v4l2src device=/dev/video0 ! "
				"video/x-raw, width=640, height=480, framerate=30/1 ! "
				"videoconvert ! video/x-raw, format=(string)BGR ! "
				"appsink drop=1 buffers=1 sync=false";
		std::cout << "[CameraStreamer] Using fallback pipeline: " << pipeline << std::endl;
		cap.open(pipeline, cv::CAP_GSTREAMER);
	}

	// Final fallback to direct camera access
	if (!cap.isOpened()) {
		std::cout << "[CameraStreamer] GStreamer failed, trying direct camera access..." << std::endl;
		cap.open(0); // Try direct camera access
	}

	std::cout << "[CameraStreamer] Camera opened." << std::endl;

	if (!cap.isOpened()) {  // Check if camera opened successfully
		std::cerr << "Error: Could not open CSI camera" << std::endl;
		exit(-1);  // Terminate if failed
	}
}

CameraStreamer::~CameraStreamer() {
	stop();  // Stop the camera stream

	// Join all threads safely
	if (captureThread.joinable()) captureThread.join();
	if (segmentationThread.joinable()) segmentationThread.join();
	if (detectionThread.joinable()) detectionThread.join();

	if (cap.isOpened()) {
		cap.release(); // Release camera
	}

	cudaDeviceSynchronize();  // Ensure all CUDA operations are complete

	if (cuda_resource) {
		cudaGraphicsUnregisterResource(cuda_resource);  // Unregister CUDA graphics resource
		cuda_resource = nullptr;
	}

	delete m_publisherFrameObject;
	m_publisherFrameObject = nullptr;

	std::cout << "[~CameraStreamer] Destructor done." << std::endl;
}

void CameraStreamer::segmentationWorker() {
	while (m_running) {
		cv::Mat frame;
		if (segmentationBuffer.getFrame(frame)) {
			// Executa a inferência, que já faz o upload para GPU internamente
			segmentationInferencer->doInference(frame);

			// Obtém a máscara já na GPU (sem duplicar processamento)
			cv::cuda::GpuMat maskResult = segmentationInferencer->getOutputMaskGpu();

			// Pós-processamento encapsulado no Polyfitter
			LaneInfo laneInfo = m_polyfitter->processMask(maskResult);
			if (m_mpcCallback && laneInfo.isValid) {
				m_mpcCallback(laneInfo);
			}
		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
}

std::string CameraStreamer::serializeLaneInfo(const LaneInfo &laneInfo) {
	std::ostringstream oss;
	oss << laneInfo.left_boundary << "," << laneInfo.right_boundary << "," << laneInfo.center_line
	    << "," << laneInfo.lateral_offset << "," << laneInfo.yaw_error << ","
	    << (laneInfo.isValid ? 1 : 0);
	return oss.str();
}

void CameraStreamer::detectionWorker() {
	while (m_running) {
		cv::Mat frame;
		if (detectionBuffer.getFrame(frame)) {
			yoloInferencer->process_image(frame);
		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
}

void CameraStreamer::start() {
	m_running = true;

	captureThread = std::thread(&CameraStreamer::captureLoop, this);
	segmentationThread = std::thread(&CameraStreamer::segmentationWorker, this);
	detectionThread = std::thread(&CameraStreamer::detectionWorker, this);
}

void CameraStreamer::captureLoop() {
	auto start_time = std::chrono::high_resolution_clock::now();
	int frame_count = 0;
	const int framesToSkip = 1;  // Skip frames to reduce processing load
	cv::Mat frame;

	while (m_running) {
		auto frame_start = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < framesToSkip; ++i) {
			cap.grab();  // Grab frames without decoding
		}
		cap >> frame;  // Read one frame (decoded)

		if (frame.empty()) {
			std::cerr << "Empty frame, exiting" << std::endl;
			break;
		}

		segmentationBuffer.update(frame);
		detectionBuffer.update(frame);

		frame_count++;
		auto now = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

		if (elapsed >= 1) {
			std::cout << "Average FPS: " << frame_count / static_cast<double>(elapsed) << std::endl;
			start_time = now;
			frame_count = 0;
		}
	}
}

void CameraStreamer::stop() {
	if (!m_running) return;
	m_running = false;

	try {
		cudaDeviceSynchronize();
	} catch (const std::exception& e) {
		std::cerr << "CUDA sync error in stop(): " << e.what() << std::endl;
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	std::cout << "[CameraStreamer] Shutdown complete." << std::endl;
}
