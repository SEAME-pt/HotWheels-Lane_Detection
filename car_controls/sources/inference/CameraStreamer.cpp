
#include "CameraStreamer.hpp"

CameraStreamer::CameraStreamer(double scale)
    : scale_factor(scale), m_zeromq_enabled(true), m_running(true),
      m_polyfitter(std::make_unique<Polyfitter>()) {

	segmentationInferencer =
	    std::make_shared<TensorRTInferencer>("/home/jetson/models/lane-detection/model.engine");
	yoloInferencer =
	    std::make_shared<YOLOv5TRT>("/home/jetson/models/object-detection/yolov5m_updated.engine",
	                                "/home/jetson/models/object-detection/labels.txt");

	std::string pipeline = "nvarguscamerasrc sensor-mode=4 ! "
	                       "video/x-raw(memory:NVMM), width=1280, height=720, "
	                       "format=(string)NV12, framerate=30/1 ! "
	                       "nvvidconv ! video/x-raw, format=(string)BGRx ! "
	                       "videoconvert ! video/x-raw, format=(string)BGR ! "
	                       "appsink drop=1 buffers=1";

	std::cout << "[CameraStreamer] Using GStreamer pipeline: " << pipeline << std::endl;

	cap.open(pipeline, cv::CAP_GSTREAMER);

	std::cout << "[CameraStreamer] Camera opened." << std::endl;

	if(!cap.isOpened()) {
		ERROR_LOG("CameraStreamer", "Error: Could not open CSI camera");
		exit(-1);
	}
}

CameraStreamer::~CameraStreamer() {
	stop();

	if(captureThread.joinable())
		captureThread.join();
	if(segmentationThread.joinable())
		segmentationThread.join();
	if(detectionThread.joinable())
		detectionThread.join();

	if(cap.isOpened()) {
		cap.release();
	}

	cudaDeviceSynchronize();

	std::cout << "[~CameraStreamer] Destructor done." << std::endl;
}

void CameraStreamer::segmentationWorker() {
	while(m_running) {
		cv::Mat frame;
		if(segmentationBuffer.getFrame(frame)) {
			// 1. Upload frame to GPU
			cv::cuda::GpuMat gpuFrame;
			gpuFrame.upload(frame);

			// 2. Execute TensorRT inference
			cv::cuda::GpuMat maskResult = segmentationInferencer->makePrediction(gpuFrame);

			// 3. Download mask for Polyfitter processing
			cv::Mat binaryMask;
			maskResult.download(binaryMask);

			// 4. Convert to binary if needed
			if(binaryMask.type() == CV_32F) {
				cv::threshold(binaryMask, binaryMask, 0.5, 255, cv::THRESH_BINARY);
				binaryMask.convertTo(binaryMask, CV_8U);
			}

			// 5. Process with Polyfitter to extract lane information
			LaneInfo laneInfo = m_polyfitter->processFrame(binaryMask);

			// === FLUXO DIRETO: Send directly to ControlsManager ===
			if(m_mpcCallback && laneInfo.isValid) {
				m_mpcCallback(laneInfo);
			}

			// === NOTA: ZeroMQ publishing será feito pelo Polyfitter/MPC ===

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
	while(m_running) {
		cv::Mat frame;
		if(detectionBuffer.getFrame(frame)) {
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
	const int framesToSkip = 1;
	cv::Mat frame;

	while(m_running) {
		auto frame_start = std::chrono::high_resolution_clock::now();

		for(int i = 0; i < framesToSkip; ++i) {
			cap.grab();
		}
		cap >> frame;

		if(frame.empty()) {
			ERROR_LOG("CameraStreamer", "Empty frame, exiting");
			break;
		}

		segmentationBuffer.update(frame);
		detectionBuffer.update(frame);

		frame_count++;
		auto now = std::chrono::high_resolution_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

		if(elapsed >= 1) {
			std::cout << "Average FPS: " << frame_count / static_cast<double>(elapsed) << std::endl;
			start_time = now;
			frame_count = 0;
		}
	}
}

void CameraStreamer::stop() {
	if(!m_running)
		return;
	m_running = false;

	try {
		cudaDeviceSynchronize();
	} catch(const std::exception &e) {
		ERROR_STREAM("CameraStreamer") << "CUDA sync error in stop(): " << e.what();
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	std::cout << "[CameraStreamer] Shutdown complete." << std::endl;
}
