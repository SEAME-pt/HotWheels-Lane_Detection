/**
 * @file CameraStreamer.hpp
 * @brief Header for CameraStreamer - Real-Time Multi-Threaded AI Vision System
 *
 * @details This header defines the CameraStreamer class and supporting infrastructure
 * 			for a high-performance computer vision pipeline designed for autonomous vehicle
 * 			perception systems. The architecture combines hardware-accelerated camera capture
 * 			with parallel AI inference engines for real-time lane detection and object detection.
 *
 * 			**Core Components Defined**:
 * 			- `FrameBufferSegmentation`: Thread-safe buffer for lane detection frames
 * 			- `FrameBufferDetection`: Thread-safe buffer for object detection frames
 * 			- `CameraStreamer`: Main orchestration class for the vision pipeline
 *
 * 			**Key Design Principles**:
 * 			1. **Thread Safety**: All shared data structures use mutex-based synchronization
 * 			2. **Performance**: Optimized for real-time processing with minimal latency
 * 			3. **Modularity**: Clean separation between capture, processing, and communication
 * 			4. **Reliability**: Comprehensive error handling and resource management
 * 			5. **Scalability**: Designed to handle varying processing loads gracefully
 *
 * 			**Target Platform**: Jetson Nano/Xavier series with CSI camera support
 * 			**Dependencies**: OpenCV, CUDA, TensorRT, GStreamer, ZeroMQ, OpenGL/GLFW
 * 			**Performance**: 30 FPS capture with real-time AI inference capabilities
 *
 * 			**Integration Points**:
 * 			- Inherits from IInferencer interface for standardized AI processing
 * 			- Uses ZeroMQ Publisher/Subscriber for distributed communication
 * 			- Integrates with TensorRT engines for optimized GPU inference
 * 			- Supports CUDA graphics interop for zero-copy operations
 *
 * @note This file requires CUDA-capable hardware and TensorRT runtime
 * @note All class implementations are in CameraStreamer.cpp
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef CAMERA_STREAMER_HPP
#define CAMERA_STREAMER_HPP

#include "Debugger.hpp"
#include "IInferencer.hpp"
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

/**
 * @brief Thread-safe frame buffer specifically designed for segmentation processing
 * @details This class implements a single-frame buffer with thread-safe access patterns
 * 			optimized for real-time segmentation workflows. It provides atomic frame updates
 * 			and retrievals using mutex synchronization, ensuring data consistency across
 * 			multiple threads without blocking operations.
 *
 * 			Key features:
 * 			- Single frame storage with immediate replacement (no queue)
 * 			- Deep copy operations to prevent memory corruption
 * 			- Non-blocking retrieval with availability status
 * 			- Automatic frame freshness tracking
 *
 * @note This buffer is designed for high-frequency updates where only the most
 * 			recent frame is relevant for segmentation processing.
 */
class FrameBufferSegmentation {
	public:
		/**
		 * @brief Updates the buffer with a new frame for segmentation processing
		 * @param frame The input frame to store (will be deep-copied)
		 * @details Atomically replaces the current frame with the new one using
		 * 			a deep copy operation. This ensures thread safety and prevents memory
		 * 			corruption when multiple threads access the buffer simultaneously.
		 * 			The operation is blocking until the mutex is acquired.
		 */
		void update(const cv::Mat &frame) {
			std::lock_guard<std::mutex> lock(mutex_);
			frame_ = frame.clone(); // deep copy
			has_new_frame_ = true;
		}

		/**
		 * @brief Retrieves the most recent frame if available
		 * @param out Reference to output cv::Mat where the frame will be copied
		 * @return true if a new frame was retrieved, false if no new frame available
		 * @details Performs a non-destructive read of the current frame if one is
		 * 			available. The frame is deep-copied to the output parameter and the
		 * 			"new frame" flag is reset. This ensures each frame is processed only once.
		 */
		bool getFrame(cv::Mat &out) {
			std::lock_guard<std::mutex> lock(mutex_);
			if(!has_new_frame_)
				return false;
			out = frame_.clone();
			has_new_frame_ = false;
			return true;
		}

	private:
		cv::Mat frame_;              ///< Current frame storage (deep copy)
		bool has_new_frame_ = false; ///< Flag indicating fresh frame availability
		std::mutex mutex_;           ///< Mutex for thread-safe operations
};

/**
 * @brief Thread-safe frame buffer specifically designed for object detection processing
 * @details Similar to FrameBufferSegmentation but optimized for object detection workflows.
 * 			Implements the same thread-safe single-frame buffer pattern with atomic operations
 * 			for high-performance real-time detection scenarios.
 *
 * 			Key features:
 * 			- Identical API to FrameBufferSegmentation for consistency
 * 			- Optimized for YOLO and other detection algorithm requirements
 * 			- Thread-safe frame replacement without queuing overhead
 * 			- Memory-efficient deep copy operations
 *
 * @note Could be template-unified with FrameBufferSegmentation in future refactoring
 */
class FrameBufferDetection {
	public:
		/**
		 * @brief Updates the buffer with a new frame for object detection processing
		 * @param frame The input frame to store (will be deep-copied)
		 * @details Identical behavior to FrameBufferSegmentation::update().
		 * 			Atomically replaces the current frame with thread-safe deep copy operation.
		 */
		void update(const cv::Mat &frame) {
			std::lock_guard<std::mutex> lock(mutex_);
			frame_ = frame.clone(); // deep copy
			has_new_frame_ = true;
		}

		/**
		 * @brief Retrieves the most recent frame for detection processing
		 * @param out Reference to output cv::Mat where the frame will be copied
		 * @return true if a new frame was retrieved, false if no new frame available
		 * @details Identical behavior to FrameBufferSegmentation::getFrame().
		 * 			Provides non-destructive frame access with automatic freshness tracking.
		 */
		bool getFrame(cv::Mat &out) {
			std::lock_guard<std::mutex> lock(mutex_);
			if(!has_new_frame_)
				return false;
			out = frame_.clone();
			has_new_frame_ = false;
			return true;
		}

	private:
		// Current frame storage for detection
		cv::Mat frame_;
		// Fresh frame availability flag
		bool has_new_frame_ = false;
		// Thread synchronization mutex
		std::mutex mutex_;
};

/**
 * @brief High-performance multi-threaded camera streaming and AI inference engine
 * @details CameraStreamer is a sophisticated real-time video processing system that
 * 			orchestrates camera capture, AI inference, and result publishing in a multi-threaded
 * 			architecture. It's specifically designed for autonomous vehicle applications requiring
 * 			simultaneous lane detection (segmentation) and object detection with minimal latency.
 *
 * 			Architecture Overview:
 * 			- Main Thread: Coordinates startup, shutdown, and resource management
 * 			- Capture Thread: Handles high-frequency camera frame acquisition via GStreamer
 * 			- Segmentation Thread: Processes frames for lane detection using TensorRT
 * 			- Detection Thread: Performs object detection using optimized YOLO models
 *
 * 			Key Technical Features:
 * 			- CSI camera integration with hardware-accelerated GStreamer pipeline
 * 			- Dual TensorRT inference engines for parallel processing
 * 			- CUDA-accelerated image processing and memory management
 * 			- Thread-safe frame buffers with zero-copy where possible
 * 			- ZeroMQ-based result publishing for distributed systems
 * 			- Frame skipping and rate control for performance optimization
 *
 * 			Performance Characteristics:
 * 			- Designed for 30 FPS camera input with real-time inference
 * 			- Automatic frame dropping to prevent buffer overflow
 * 			- GPU memory management with CUDA graphics interop
 * 			- Configurable scaling for different resolution requirements
 *
 * @note This class is the core component of the autonomous driving vision pipeline
 */
class CameraStreamer {
	public:
		/**
		 * @brief Constructs CameraStreamer with configurable frame scaling
		 * @param scale Frame scaling factor (0.0-1.0) for resolution adjustment
		 * @details Initializes the complete vision pipeline including camera setup,
		 * 			AI model loading, and threading infrastructure preparation
		 */
		CameraStreamer(double scale = 0.5);

		/**
		 * @brief Destructor ensuring clean shutdown of all resources
		 * @details Orchestrates proper cleanup of threads, CUDA resources,
		 * 			camera hardware, and AI inference engines
		 */
		~CameraStreamer();

		/**
		 * @brief Initiates the multi-threaded vision processing pipeline
		 * @details Spawns and coordinates all worker threads for camera capture,
		 * 			segmentation processing, and object detection
		 */
		void start();

		/**
		 * @brief Gracefully stops all processing threads and releases resources
		 * @details Signals shutdown to all worker threads and waits for clean termination
		 */
		void stop();

	private:
		// === Core Hardware Interfaces ===
		// OpenCV camera capture interface
		cv::VideoCapture cap;
		// Frame scaling factor for performance tuning
		double scale_factor;
		// CUDA graphics resource for GPU interop
		cudaGraphicsResource *cuda_resource;

		// === Threading Control ===
		// Master control flag for all threads
		bool m_running;

		// === Communication Infrastructure ===
		// ZeroMQ publisher for inference results
		Publisher *m_publisherFrameObject;

		// === AI Inference Engines ===
		// Lane detection engine
		std::shared_ptr<TensorRTInferencer> segmentationInferencer;
		// Object detection engine
		std::shared_ptr<YOLOv5TRT> yoloInferencer;

		// === Thread-Safe Frame Management ===
		// Buffer for lane detection frames
		FrameBufferSegmentation segmentationBuffer;
		// Buffer for object detection frames
		FrameBufferDetection detectionBuffer;

		// === Worker Thread Functions ===
		/**
		 * @brief Worker thread function for lane segmentation processing
		 * @details Continuously processes frames from segmentationBuffer using TensorRT
		 * 			inference for lane detection. Implements intelligent frame dropping and
		 * 			performance monitoring.
		 */
		void segmentationWorker();

		/**
		 * @brief Worker thread function for object detection processing
		 * @details Continuously processes frames from detectionBuffer using YOLO
		 * 			inference for object detection. Handles result publishing and performance
		 * 			tracking.
		 */
		void detectionWorker();

		/**
		 * @brief Main camera capture loop running in dedicated thread
		 * @details Manages high-frequency frame acquisition from CSI camera,
		 * 			implements frame skipping for performance, and distributes frames
		 * 			to both processing buffers. Includes FPS monitoring and error handling.
		 */
		void captureLoop();

		// === Thread Management ===
		// Camera capture thread handle
		std::thread captureThread;
		// Lane detection processing thread handle
		std::thread segmentationThread;
		// Object detection processing thread handle
		std::thread detectionThread;
};

#endif // CAMERA_STREAMER_HPP