#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include "CommonTypes.hpp"
#include "Debugger.hpp"
#include "IInferencer.hpp"
#include "Polyfitter.hpp"
#include "Publisher.hpp"
#include "Subscriber.hpp"
#include "TensorRTInferencer.hpp"
#include "YOLOv5TRT.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <condition_variable>
#include <cuda_gl_interop.h>
#include <iostream>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <thread>

class FrameBufferSegmentation {
	public:
		void update(const cv::Mat &frame) {
			std::lock_guard<std::mutex> lock(mutex_);
			frame_ = frame.clone();
			has_new_frame_ = true;
		}

		bool getFrame(cv::Mat &out) {
			std::lock_guard<std::mutex> lock(mutex_);
			if(!has_new_frame_)
				return false;
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
		void update(const cv::Mat &frame) {
			std::lock_guard<std::mutex> lock(mutex_);
			frame_ = frame.clone();
			has_new_frame_ = true;
		}
		bool getFrame(cv::Mat &out) {
			std::lock_guard<std::mutex> lock(mutex_);
			if(!has_new_frame_)
				return false;
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
		void setMPCCallback(std::function<void(const LaneInfo &)> callback) {
			m_mpcCallback = callback;
		}
		void enableZeroMQPublishing(bool enable = true) {
			m_zeromq_enabled = enable;
		}
		std::string serializeLaneInfo(const LaneInfo &laneInfo);

	private:
		cv::VideoCapture cap;
		double scale_factor;
		bool m_zeromq_enabled = true;
		bool m_running;
		std::unique_ptr<Polyfitter> m_polyfitter;
		std::function<void(const LaneInfo &)> m_mpcCallback;
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

#endif