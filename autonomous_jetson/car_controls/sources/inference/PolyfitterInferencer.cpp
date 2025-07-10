#include "PolyfitterInferencer.hpp"
#include "Debugger.hpp"
#include <chrono>

PolyfitterInferencer::PolyfitterInferencer(const std::string &enginePath)
    : TensorRTInferencer(enginePath), m_polyfitter(std::make_unique<Polyfitter>()),
      m_currentLaneInfo(0.0, 0.0) {
	DEBUG_LOG("PolyfitterInferencer", "Initialized with integrated polynomial fitting");
}

PolyfitterInferencer::~PolyfitterInferencer() = default;

cv::cuda::GpuMat PolyfitterInferencer::makePrediction(const cv::cuda::GpuMat &gpuImage) {
	auto start_time = std::chrono::steady_clock::now();

	// Call parent implementation for standard TensorRT inference
	cv::cuda::GpuMat result = TensorRTInferencer::makePrediction(gpuImage);

	// Download mask to CPU for Polyfitter processing
	cv::Mat binaryMask;
	result.download(binaryMask);

	// Convert to binary if needed
	if(binaryMask.type() == CV_32F) {
		cv::threshold(binaryMask, binaryMask, 0.5, 255, cv::THRESH_BINARY);
		binaryMask.convertTo(binaryMask, CV_8U);
	}

	// Extract MPC-optimized data directly
	extractMPCData(binaryMask);

	auto end_time = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	DEBUG_STREAM("PolyfitterInferencer")
	    << "Processing completed in " << duration.count() / 1000.0 << "ms, "
	    << "trajectory points: " << m_currentTrajectory.size()
	    << ", poly coeffs: " << m_currentPolyCoeffs.size();

	return result;
}

void PolyfitterInferencer::doInference(const cv::Mat &frame) {
	// Call parent implementation
	TensorRTInferencer::doInference(frame);

	// Extract enhanced MPC data from the last processed mask
	cv::Mat lastMask = getLastMask();
	if(!lastMask.empty()) {
		extractMPCData(lastMask);
	}
}

void PolyfitterInferencer::extractMPCData(const cv::Mat &binaryMask) {
	if(binaryMask.empty()) {
		return;
	}

	// Check cache validity to avoid unnecessary processing
	if(isCacheValid() && cv::norm(binaryMask, m_lastProcessedMask, cv::NORM_L1) < 1000) {
		return; // Use cached data
	}

	try {
		// Reset current data
		m_currentPolyCoeffs.clear();
		m_currentTrajectory.clear();

		// Use Polyfitter to extract lane information
		auto lanes = m_polyfitter->fitLanesInImage(binaryMask);

		if(!lanes.empty()) {
			// Compute virtual centerline using Polyfitter
			auto centerlineResult =
			    m_polyfitter->computeVirtualCenterline(lanes, binaryMask.cols, binaryMask.rows);

			if(centerlineResult.valid && !centerlineResult.blend.empty()) {
				// Store trajectory points
				m_currentTrajectory = centerlineResult.blend;

				// Calculate polynomial coefficients for MPC
				m_currentPolyCoeffs = m_polyfitter->getPolynomialCoeffs(m_currentTrajectory);

				// Update lane info for MPC
				if(m_currentTrajectory.size() >= 3) {
					// Calculate curvature from first few points
					double dx1 = m_currentTrajectory[1].x - m_currentTrajectory[0].x;
					double dy1 = m_currentTrajectory[1].y - m_currentTrajectory[0].y;
					double dx2 = m_currentTrajectory[2].x - m_currentTrajectory[1].x;
					double dy2 = m_currentTrajectory[2].y - m_currentTrajectory[1].y;

					double curvature =
					    std::abs((dx1 * dy2 - dy1 * dx2) / std::pow(dx1 * dx1 + dy1 * dy1, 1.5));

					m_currentLaneInfo =
					    LaneInfo(curvature, 0.0); // width can be calculated if needed
				}
			}
		}

		// Update cache
		m_lastProcessedMask = binaryMask.clone();
		m_lastProcessTime = std::chrono::steady_clock::now();

	} catch(const std::exception &e) {
		ERROR_STREAM("PolyfitterInferencer") << "Error in extractMPCData: " << e.what();
		// Keep last valid data on error
	}
}

bool PolyfitterInferencer::isCacheValid() const {
	auto now = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastProcessTime);
	return elapsed.count() < CACHE_VALIDITY_MS;
}

double PolyfitterInferencer::calculateCTE(double vehicle_x, double vehicle_y) const {
	if(m_currentPolyCoeffs.empty()) {
		return 0.0;
	}

	return m_polyfitter->calculateCTE(m_currentPolyCoeffs, vehicle_x, vehicle_y);
}

double PolyfitterInferencer::calculateEPSI(double vehicle_x, double vehicle_psi) const {
	if(m_currentPolyCoeffs.empty()) {
		return 0.0;
	}

	return m_polyfitter->calculateEPSI(m_currentPolyCoeffs, vehicle_x, vehicle_psi);
}

double PolyfitterInferencer::getLastProcessingTimeMs() const {
	auto now = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastProcessTime);
	return elapsed.count();
}
