#ifndef MPCPLANNER_HPP
#define MPCPLANNER_HPP

#include "MPCOptimizer.hpp"
#include "Polyfitter.hpp"
#include "inference/PolyfitterInferencer.hpp"
#include <Eigen/Dense>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <vector>

class MPCPlanner {

	private:
		std::vector<Eigen::Vector2d>
		_prepareReference (const VehicleState &state,
		                   const std::vector<Eigen::Vector2d> &global_waypoints) const;

		MPCOptimizer _optimizer;

		// Direct integration with enhanced inferencer
		std::shared_ptr<PolyfitterInferencer> m_polyfitterInferencer;
		bool m_useDirectInference;
		struct CachedVisionData {
				std::vector<Point2D> waypoints;
				LaneInfo lane_info{0.0, 0.0};
				std::chrono::steady_clock::time_point timestamp;
				bool valid = false;
				std::mutex mutex;
		} m_cachedVisionData;
		// State estimation components
		struct StateEstimator {
				VehicleState m_estimatedState;
				std::mutex m_stateMutex;
				std::chrono::steady_clock::time_point m_lastUpdate;
				bool m_initialized = false;
				std::atomic<bool> m_useRealSensors{false};
				std::atomic<double> m_realVelocity{0.0};
				std::atomic<double> m_realYawRate{0.0};
		} m_stateEstimator;

		// Last applied controls for state estimation
		std::atomic<double> m_lastThrottle{0.0};
		std::atomic<double> m_lastSteering{0.0};

		// Polyfitter for vision processing
		std::shared_ptr<Polyfitter> m_polyfitter;

		// Adicionar método para mapear comandos para hardware
		ControlCommand _mapCommandsToHardware (double throttle, double steer) const;

	public:
		static constexpr double DATA_TIMEOUT_MS = 200.0;
		MPCPlanner (void);
		MPCPlanner (const MPCPlanner &orign);
		MPCPlanner &operator= (const MPCPlanner &orign);
		~MPCPlanner (void);

		MPCPlanner (const MPCOptimizer &optimizer);
		std::vector<Point2D> extractWaypointsFromLaneInfo (const LaneInfo &lane_info);
		ControlCommand applySmoothSteering (const ControlCommand &control);
		// Enhanced constructor with direct inferencer integration
		MPCPlanner (std::shared_ptr<PolyfitterInferencer> inferencer);
		VehicleState getCurrentVehicleState ();
		VehicleState getEnhancedVehicleState ();
		void updateVehicleStateEstimation (double applied_throttle, double applied_steering,
		                                   double dt);
		void integrateRealSensorData ();
		LaneInfo generateStraightTrajectory ();
		void showVisionDebug ();
		std::vector<Point2D> getCachedWaypoints ();
		cv::Mat deserializeMask (const std::string &data);

		// Set direct inference mode
		void setPolyfitterInferencer (std::shared_ptr<PolyfitterInferencer> inferencer);
		void enableDirectInference (bool enable = true) {
			m_useDirectInference = enable;
		}

		ControlCommand plan (const VehicleState &current_state,
		                     const std::vector<Point2D> &global_waypoints,
		                     const LaneInfo *lane_info = NULL);

		// Enhanced planning method with direct inference
		ControlCommand planWithDirectInference (const VehicleState &current_state);

		// Autonomous step method
		ControlCommand runAutonomousStep (const VehicleState &state, const LaneInfo &lane_info);

		std::vector<Point2D> convertImagePointsToWorld (const std::vector<int> &center_x,
		                                                const std::vector<int> &center_y,
		                                                const VehicleTransform &vehicle_transform,
		                                                int img_width, int img_height) const;

		// Novo método para acessar a trajetória prevista do MPC
		const std::vector<Point2D> &getPredictedTrajectory () const {
			return _optimizer.getPredictedTrajectory ();
		}

		// Access to direct inference data
		bool hasValidTrajectoryData () const;
		const std::vector<Point2D> &getCurrentTrajectory () const;
};

#endif /* !MPCPlanner */