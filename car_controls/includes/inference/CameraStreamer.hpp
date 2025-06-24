#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "../../../ZeroMQ/Publisher.hpp"
#include "../../../ZeroMQ/Subscriber.hpp"
#include "ONNXInferencer.hpp"
#include "TensorRTInferencer.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include <condition_variable>
#include <mutex>
#include <queue>

#include "IInferencer.hpp"
#include "YOLOv5TRT.hpp"

class FrameBufferSegmentation {
	public:
		void update(const cv::Mat &frame) {
			std::lock_guard<std::mutex> lock(mutex_);
			frame_ = frame.clone(); // deep copy
		}

		bool getFrame(cv::Mat &out) {
			std::lock_guard<std::mutex> lock(mutex_);
			if(frame_.empty())
				return false;
			out = frame_.clone();
			return true;
		}

	private:
		cv::Mat frame_;
		std::mutex mutex_;
};

class FrameBufferDetection {
	public:
		void update(const cv::Mat &frame) {
			std::lock_guard<std::mutex> lock(mutex_);
			frame_ = frame.clone(); // deep copy
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
		CameraStreamer(double scale = 0.5, bool use_video = false,
		               const std::string &video_path = "");
		~CameraStreamer();

		void start();
		void stop();

	private:
		cv::VideoCapture cap;
		double scale_factor;

		// Video playback control
		bool m_useVideo;
		std::string m_videoPath;
		bool m_videoLoop;
		int m_currentFrame;
		int m_totalFrames;

		cudaGraphicsResource *cuda_resource;

		bool m_running;

		Publisher *m_publisherFrameObject;

		// ZeroMQ Publishers (raw pointers to singletons - we don't own them)
		Publisher *m_rawFramePublisher;
		// Removed: m_inferencePublisher - now handled by TensorRTInferencer directly

		std::shared_ptr<TensorRTInferencer> segmentationInferencer;
		// std::shared_ptr<ONNXInferencer> segmentationInferencer;

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
