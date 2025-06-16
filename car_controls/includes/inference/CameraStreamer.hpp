#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
// #include <opencv2/cudawarping.hpp>  // Not available in this OpenCV build

#include "../../ZeroMQ/Publisher.hpp"
#include "../../ZeroMQ/Subscriber.hpp"
#include "ONNXInferencer.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
// #include <cuda_gl_interop.h>  // Not available without CUDA

#include <condition_variable>
#include <mutex>
#include <queue>

#include "IInferencer.hpp"
// #include "objectDetection/YOLOv5TRT.hpp"  // Temporarily disabled due to
// TensorRT dependency

class FrameBufferSegmentation {
public:
  void update(const cv::Mat &frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    frame_ = frame.clone(); // deep copy
    has_new_frame_ = true;
  }

  bool getFrame(cv::Mat &out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_new_frame_)
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
    frame_ = frame.clone(); // deep copy
    has_new_frame_ = true;
  }

  bool getFrame(cv::Mat &out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_new_frame_)
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

private:
  cv::VideoCapture cap;
  double scale_factor;

  // cudaGraphicsResource* cuda_resource;  // Removed for compatibility

  bool m_running;

  Publisher *m_publisherFrameObject;

  std::shared_ptr<ONNXInferencer> segmentationInferencer;
  // std::shared_ptr<YOLOv5TRT> yoloInferencer;  // TODO: Implementar versão
  // ONNX

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
