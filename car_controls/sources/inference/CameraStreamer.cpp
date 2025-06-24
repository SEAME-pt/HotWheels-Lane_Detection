#include "../../includes/inference/CameraStreamer.hpp"

// Constructor: initializes camera capture, inference reference, and settings
CameraStreamer::CameraStreamer(double scale, bool use_video, const std::string &video_path)
    : scale_factor(scale), m_publisherFrameObject(nullptr), m_running(true),
      m_rawFramePublisher(nullptr), m_useVideo(use_video), m_videoPath(video_path),
      m_videoLoop(true), m_currentFrame(0), m_totalFrames(0) {

	segmentationInferencer =
	    std::make_shared<TensorRTInferencer>("/home/jetson/models/lane-detection/model.engine");
	// yoloInferencer =
	//     std::make_shared<YOLOv5TRT>("/home/jetson/models/object-detection/yolov5m_updated.engine",
	//                                 "/home/jetson/models/object-detection/labels.txt");

	if(m_useVideo && !m_videoPath.empty()) {
		// Use video file
		std::cout << "[CameraStreamer] Using video file: " << m_videoPath << std::endl;
		cap.open(m_videoPath);

		if(!cap.isOpened()) {
			std::cerr << "Error: Could not open video file: " << m_videoPath << std::endl;
			exit(-1);
		}

		// Get video properties
		m_totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
		double fps = cap.get(cv::CAP_PROP_FPS);
		int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
		int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

		std::cout << "[CameraStreamer] Video properties: " << width << "x" << height << " @ " << fps
		          << " FPS, " << m_totalFrames << " frames" << std::endl;
	} else {
		// Use camera (original code)
		std::string pipeline = "nvarguscamerasrc sensor-mode=4 ! "
		                       "video/x-raw(memory:NVMM), width=1280, height=720, "
		                       "format=(string)NV12, framerate=30/1 ! "
		                       "nvvidconv ! video/x-raw, format=(string)BGRx ! "
		                       "videoconvert ! video/x-raw, format=(string)BGR ! "
		                       "appsink drop=1 buffers=1";

		std::cout << "[CameraStreamer] Using GStreamer pipeline: " << pipeline << std::endl;
		cap.open(pipeline, cv::CAP_GSTREAMER);

		if(!cap.isOpened()) {
			std::cerr << "Error: Could not open CSI camera" << std::endl;
			exit(-1);
		}
	}

	// Initialize ZeroMQ publishers (singletons - just store raw pointers)
	try {
		// Get publisher instances - they are singletons managed internally
		m_rawFramePublisher = Publisher::instance(5558);
		// m_inferencePublisher removed - now handled by TensorRTInferencer directly

		std::cout << "[CameraStreamer] ZeroMQ raw frame publisher initialized" << std::endl;
	} catch(const std::exception &e) {
		std::cerr << "[CameraStreamer] ZeroMQ setup error: " << e.what() << std::endl;
		m_rawFramePublisher = nullptr;
	}
}

// Destructor: clean up resources
CameraStreamer::~CameraStreamer() {
	stop(); // Stop the camera stream

	// Join all threads safely
	if(captureThread.joinable())
		captureThread.join();
	if(segmentationThread.joinable())
		segmentationThread.join();
	// if(detectionThread.joinable())
	// 	detectionThread.join();

	if(cap.isOpened()) {
		cap.release(); // Release camera
	}

	cudaDeviceSynchronize(); // Ensure all CUDA operations are complete

	if(cuda_resource) {
		cudaGraphicsUnregisterResource(cuda_resource); // Unregister CUDA graphics resource
		cuda_resource = nullptr;
	}

	delete m_publisherFrameObject;
	m_publisherFrameObject = nullptr;

	// Don't delete singletons - just null the pointer
	m_rawFramePublisher = nullptr;

	std::cout << "[~CameraStreamer] Destructor done." << std::endl;
}

void CameraStreamer::segmentationWorker() {
	while(m_running) {
		cv::Mat frame;
		if(segmentationBuffer.getFrame(frame)) {
			segmentationInferencer->doInference(frame);

			// Publishing is now handled directly by TensorRTInferencer::doInference()
			// No need to publish here anymore
		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}
}

// void CameraStreamer::detectionWorker() {
//     while(m_running) {
//         cv::Mat frame;
//         if(detectionBuffer.getFrame(frame)) {
//             yoloInferencer->process_image(frame);
//         } else {
//             std::this_thread::sleep_for(std::chrono::milliseconds(1));
//         }
//     }
// }

// Main loop: capture, undistort, predict, visualize and render frames
void CameraStreamer::start() {
	m_running = true;

	captureThread = std::thread(&CameraStreamer::captureLoop, this);
	segmentationThread = std::thread(&CameraStreamer::segmentationWorker, this);
	// detectionThread = std::thread(&CameraStreamer::detectionWorker, this); // YOLO paused
}

void CameraStreamer::captureLoop() {
	auto start_time = std::chrono::high_resolution_clock::now();
	int frame_count = 0;
	const int framesToSkip = m_useVideo ? 0 : 1; // Don't skip frames for video
	cv::Mat frame;

	while(m_running) {
		auto frame_start = std::chrono::high_resolution_clock::now();

		if(m_useVideo) {
			// Video playback logic
			cap >> frame;

			if(frame.empty()) {
				if(m_videoLoop && m_totalFrames > 0) {
					// Reset to beginning for loop
					cap.set(cv::CAP_PROP_POS_FRAMES, 0);
					m_currentFrame = 0;
					cap >> frame;
					std::cout << "[CameraStreamer] Video looped back to start" << std::endl;
				}

				if(frame.empty()) {
					std::cerr << "Empty frame, exiting" << std::endl;
					break;
				}
			}

			m_currentFrame++;

			std::this_thread::sleep_for(
			    std::chrono::milliseconds(33)); // 33ms = 30 FPS and 60 fps =

		} else {
			// Camera capture logic (original)
			for(int i = 0; i < framesToSkip; ++i) {
				cap.grab(); // Grab frames without decoding
			}
			cap >> frame; // Read one frame (decoded)

			if(frame.empty()) {
				std::cerr << "Empty frame, exiting" << std::endl;
				break;
			}
		}

		// Publish raw camera frame for testing/debugging
		try {
			if(m_rawFramePublisher &&
			   frame_count % 5 == 0) { // Publish every 5th frame to reduce bandwidth
				std::vector<uchar> buffer;
				cv::imencode(".jpg", frame, buffer, {cv::IMWRITE_JPEG_QUALITY, 70});
				std::string encoded_frame(buffer.begin(), buffer.end());
				m_rawFramePublisher->publish("camera_frame", encoded_frame);
			}
		} catch(const std::exception &e) {
			std::cerr << "[CameraStreamer] Raw frame publish error: " << e.what() << std::endl;
		}

		// Update buffers for inference threads INSIDE the loop
		// segmentationBuffer.update(frame);
		// detectionBuffer.update(frame);

		// Chamada direta da inferência para lane detection
		if(segmentationInferencer) {
			try {
				// Removido debug de frame
				segmentationInferencer->doInference(frame);
			} catch(const std::exception &e) {
				std::cerr << "[CameraStreamer] Lane detection error: " << e.what() << std::endl;
			}
		}

		frame_count++;
		auto now = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

		if(elapsed >= 10) { // Log every 10 seconds instead of every second
			if(m_useVideo) {
				std::cout << "[CameraStreamer] Video playback: Frame " << m_currentFrame << "/"
				          << m_totalFrames
				          << " (FPS: " << frame_count / static_cast<double>(elapsed) << ")"
				          << std::endl;
			} else {
				std::cout << "[CameraStreamer] Average FPS: "
				          << frame_count / static_cast<double>(elapsed) << std::endl;
			}
			start_time = now;
			frame_count = 0;
		}
	}
}

void CameraStreamer::stop() {
	if(!m_running)
		return;
	m_running = false;

	// Wait for any CUDA operations to finish
	try {
		cudaDeviceSynchronize();
	} catch(const std::exception &e) {
		std::cerr << "CUDA sync error in stop(): " << e.what() << std::endl;
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	std::cout << "[CameraStreamer] Shutdown complete." << std::endl;
}
