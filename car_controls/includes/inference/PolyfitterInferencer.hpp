#pragma once

#include "TensorRTInferencer.hpp"
#include "../Polyfitter.hpp"

/**
 * @brief Enhanced TensorRT inferencer with integrated polynomial fitting for MPC
 *
 * This class extends TensorRTInferencer to provide direct lane polynomial coefficients
 * and trajectory data optimized for MPC consumption, eliminating the need for
 * asynchronous communication via ZeroMQ for critical path planning data.
 */
class PolyfitterInferencer : public TensorRTInferencer {
	private:
		std::unique_ptr<Polyfitter> m_polyfitter;
		std::vector<double> m_currentPolyCoeffs;
		std::vector<Point2D> m_currentTrajectory;
		LaneInfo m_currentLaneInfo;

		// Cache for performance optimization
		cv::Mat m_lastProcessedMask;
		std::chrono::steady_clock::time_point m_lastProcessTime;
		static constexpr double CACHE_VALIDITY_MS = 50.0; // 20 FPS max processing rate

		// Direct MPC data extraction
		void extractMPCData(const cv::Mat &binaryMask);
		bool isCacheValid() const;

	public:
		PolyfitterInferencer(const std::string &enginePath);
		~PolyfitterInferencer() override;

		// Override to provide enhanced processing
		cv::cuda::GpuMat makePrediction(const cv::cuda::GpuMat &gpuImage) override;
		void doInference(const cv::Mat &frame) override;

		// Direct MPC data access (eliminates ZeroMQ dependency for critical path)
		const std::vector<double> &getCurrentPolyCoeffs() const {
			return m_currentPolyCoeffs;
		}
		const std::vector<Point2D> &getCurrentTrajectory() const {
			return m_currentTrajectory;
		}
		const LaneInfo &getCurrentLaneInfo() const {
			return m_currentLaneInfo;
		}

		// CTE and EPSI calculation for current vehicle position
		double calculateCTE(double vehicle_x, double vehicle_y) const;
		double calculateEPSI(double vehicle_x, double vehicle_psi) const;

		// Validity checks
		bool hasValidTrajectory() const {
			return !m_currentTrajectory.empty();
		}
		bool hasValidPolyCoeffs() const {
			return !m_currentPolyCoeffs.empty();
		}

		// Performance monitoring
		double getLastProcessingTimeMs() const;
};
