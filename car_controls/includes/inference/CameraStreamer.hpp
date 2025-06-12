#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>

#include "TensorRTInferencer.hpp"
#include "../../../ZeroMQ/Subscriber.hpp"
#include "../../../ZeroMQ/Publisher.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <condition_variable>
#include <mutex>
#include <queue>

#include "IInferencer.hpp"
#include "../objectDetection/YOLOv5TRT.hpp"

class FrameBufferSegmentation {
public:
	void update(const cv::Mat& frame) {
		std::lock_guard<std::mutex> lock(mutex_);
		frame_ = frame.clone(); // deep copy
		has_new_frame_ = true;
	}

	bool getFrame(cv::Mat& out) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (!has_new_frame_) return false;
		out = frame_.clone();
		has_new_frame_ = false;
		return true;
	}

private:
	cv::Mat frame_;
	bool has_new_frame_ = false;
	std::mutex mutex_;
};

class FrameBufferDetection {
public:
	void update(const cv::Mat& frame) {
		std::lock_guard<std::mutex> lock(mutex_);
		frame_ = frame.clone(); // deep copy
		has_new_frame_ = true;
	}

	bool getFrame(cv::Mat& out) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (!has_new_frame_) return false;
		out = frame_.clone();
		has_new_frame_ = false;
		return true;
	}

private:
	cv::Mat frame_;
	bool has_new_frame_ = false;
	std::mutex mutex_;
};

class CameraStreamer {
public:
	CameraStreamer(double scale = 0.5);
	~CameraStreamer();

	void start();
	void stop();

private:
	cv::VideoCapture cap;
	double scale_factor;

	cudaGraphicsResource* cuda_resource;

	bool m_running;

	Publisher *m_publisherFrameObject;

	std::shared_ptr<TensorRTInferencer> segmentationInferencer;
	std::shared_ptr<YOLOv5TRT> yoloInferencer;

	FrameBufferSegmentation segmentationBuffer;
	FrameBufferDetection detectionBuffer;

	void segmentationWorker();
	void detectionWorker();
	void captureLoop();

	std::thread captureThread;
	std::thread segmentationThread;
	std::thread detectionThread;
};

#endif // CAMERA_STREAMER_HPP
