/*!
 * @file ControlsManager.hpp
 * @brief File containing the ControlsManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the declaration of the ControlsManager class, which
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef CONTROLSMANAGER_HPP
#define CONTROLSMANAGER_HPP

#include "EngineController.hpp"
#include "JoysticksController.hpp"
#include "MPCPlanner.hpp"
#include "Polyfitter.hpp"
#include "Publisher.hpp"
#include "Subscriber.hpp"
#include "inference/CameraStreamer.hpp"
#include <QObject>
#include <QProcess>
#include <QThread>
#include <chrono>
#include <condition_variable>
#include <fcntl.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

// === CONFIGURATION MACROS FOR EASY ADJUSTMENT ===
#ifndef DEFAULT_CONSTANT_SPEED_KMH
#define DEFAULT_CONSTANT_SPEED_KMH 2.0 // km/h - Velocidade normal dos motores
#endif
#ifndef DEFAULT_CONSTANT_SPEED
#define DEFAULT_CONSTANT_SPEED (DEFAULT_CONSTANT_SPEED_KMH / 3.6) // Auto conversion to m/s
#endif
#ifndef DEFAULT_CONSTANT_THROTTLE
#define DEFAULT_CONSTANT_THROTTLE 0.15 // Throttle normal (15%) - motores são robustos
#endif

/*!
 * @brief The ControlsManager class.
 * @details This class is responsible for managing the controls of the car.
 */
class ControlsManager : public QObject {
		Q_OBJECT

	private:
		// Core controllers
		EngineController m_engineController;
		JoysticksController *m_manualController;
		DrivingMode m_currentMode;

		// Subscriber objects
		Subscriber *m_subscriberJoystickObject;
		CameraStreamer *m_cameraStreamerObject;

		// Thread management
		std::atomic<bool> m_running;
		QThread *m_cameraStreamerThread;
		QThread *m_manualControllerThread;
		QThread *m_joystickControlThread;
		QThread *m_subscriberJoystickThread;
		QThread *m_autonomousControlThread;

		// MPC components
		MPCPlanner *m_mpcPlanner;
		Polyfitter *m_polyfitter;
		std::atomic<bool> m_autonomousMode;

		// Track applied controls for state estimation
		std::atomic<double> m_lastThrottle{0.0};
		std::atomic<double> m_lastSteering{0.0};

		// === NEW: Enhanced vehicle state estimation ===
		struct VehicleStateEstimator {
				// External data integration (vision-based measurements, manual input, etc.)
				std::atomic<bool> m_useRealSensors{false};
				std::atomic<double> m_realVelocity{0.0}; // External velocity measurement
				std::atomic<double> m_realYawRate{0.0};  // External yaw rate measurement

				// Kalman filter state
				VehicleState m_estimatedState{0.0, 0.0, 0.0, 0.0};
				std::chrono::steady_clock::time_point m_lastUpdate;

				// State covariance and noise parameters
				static constexpr double PROCESS_NOISE_POS = 0.1;
				static constexpr double PROCESS_NOISE_VEL = 0.05;
				static constexpr double PROCESS_NOISE_YAW = 0.02;
				static constexpr double MEASUREMENT_NOISE_VEL = 0.1;

				std::mutex m_stateMutex;
				bool m_initialized = false;

				VehicleStateEstimator() {
					m_lastUpdate = std::chrono::steady_clock::now();
				}
		} m_stateEstimator;

		// === NEW: Persistent ZMQ connections and data caching ===
		// Persistent ZMQ subscribers to avoid repeated connection overhead
		std::unique_ptr<Subscriber> m_visionSubscriber;
		std::unique_ptr<Subscriber> m_obstacleSubscriber;
		QThread *m_visionDataThread;
		QThread *m_obstacleDataThread;

		// Cached data structures with thread-safe access
		struct CachedVisionData {
				std::vector<Point2D> waypoints;
				LaneInfo lane_info{0.0, 0.0}; // Explicit initialization to avoid ambiguity
				std::chrono::steady_clock::time_point timestamp;
				bool valid = false;
				std::mutex mutex;
		} m_cachedVisionData;

		// === Direct MPC Integration (NOVO) ===
		struct DirectMPCData {
				LaneInfo current_lane_info;
				std::chrono::steady_clock::time_point timestamp;
				bool valid = false;
				std::mutex mutex;
		} m_directMPCData;

		struct CachedObstacleData {
				bool emergency_stop = false;
				std::chrono::steady_clock::time_point timestamp;
				bool valid = false;
				std::mutex mutex;
		} m_cachedObstacleData;

		// Control loop timing
		static constexpr double CONTROL_RATE = 20.0;       // Hz
		static constexpr double DATA_TIMEOUT_MS = 200.0;   // Max age for cached data
		static constexpr double VISION_UPDATE_RATE = 10.0; // Hz - Lower rate for vision processing
		static constexpr double OBSTACLE_UPDATE_RATE = 20.0; // Hz - Higher rate for safety

		// === NEW: Constant speed control ===
		bool m_constantSpeedMode = false;
		double m_targetConstantSpeed = DEFAULT_CONSTANT_SPEED; // Use macro for easy adjustment
		double m_constantThrottle = DEFAULT_CONSTANT_THROTTLE; // Use macro for easy adjustment

		void receiveLaneDataDirect(const LaneInfo &lane_info);

		// === NEW: Emergency stop system ===
		std::atomic<bool> m_emergencyStop{false};

		// === NEW: Smooth acceleration system (Soft Start) ===
		struct SoftStartConfig {
				bool enabled = true;
				double max_throttle_change_per_step =
				    0.01; // 1% por iteração - aceleração normal do motor
				double initial_throttle_limit = 0.05; // 5% máximo durante aquecimento
				double warmup_duration_seconds = 3.0; // 3 segundos de aquecimento
				double current_throttle_output = 0.0; // Current actual throttle being applied
				std::chrono::steady_clock::time_point start_time; // When autonomous mode started
		} m_softStart;

	public:
		explicit ControlsManager(int argc, char **argv, QObject *parent = nullptr);
		~ControlsManager();

		void setMode(DrivingMode mode);
		void readJoystickEnable();
		bool isProcessRunning(const QString &processName);
		void startAutonomousControl();
		void stopAutonomousControl();
		void autonomousControlLoop();
		void showVisionDebug();
		VehicleState getCurrentVehicleState();

		// === NEW: Enhanced sensor integration interface ===
		void enableRealSensors(bool enable = true);
		void updateRealVelocity(
		    double velocity); // For external velocity measurements (vision-based, etc.)
		void updateRealYawRate(
		    double yaw_rate); // For external yaw rate measurements (vision-based, etc.)
		VehicleState getVehicleStateWithDiagnostics();
		void resetVehicleState(const VehicleState &initial_state = {0.0, 0.0, 0.0, 0.0});

		// === NEW: Direct control methods for enhanced MPC ===
		void applyControlCommand(const ControlCommand &command);
		void applyThrottle(double throttle);
		void applySteering(double steering);
		DrivingMode getCurrentMode() const {
			return m_currentMode;
		}

		// === NEW: Constant speed control and emergency stop ===
		void setConstantSpeedMode(bool enable, double target_speed = DEFAULT_CONSTANT_SPEED,
		                          double throttle = DEFAULT_CONSTANT_THROTTLE);
		void emergencyMotorStop();
		void emergencyStop();      // Critical emergency stop method
		void resetEmergencyStop(); // Reset emergency flag when safe
		bool isEmergencyStopActive() const {
			return m_emergencyStop.load();
		}

		// === NEW: Constant speed access methods ===
		bool isConstantSpeedMode() const {
			return m_constantSpeedMode;
		}
		double getTargetConstantSpeed() const {
			return m_targetConstantSpeed;
		}

		// === NEW: Soft start configuration ===
		void setSoftStartEnabled(bool enabled) {
			m_softStart.enabled = enabled;
			std::cout << "[SOFT START] " << (enabled ? "ENABLED" : "DISABLED") << std::endl;
		}
		bool isSoftStartEnabled() const {
			return m_softStart.enabled;
		}
		void setSoftStartParameters(double max_change_per_step, double initial_limit,
		                            double warmup_duration) {
			m_softStart.max_throttle_change_per_step = max_change_per_step;
			m_softStart.initial_throttle_limit = initial_limit;
			m_softStart.warmup_duration_seconds = warmup_duration;
			std::cout << "[SOFT START] Parameters updated: Max change="
			          << (max_change_per_step * 100) << "%, Initial limit=" << (initial_limit * 100)
			          << "%, Warmup=" << warmup_duration << "s" << std::endl;
		}

		// === NEW: Soft start system for gradual acceleration ===
		double applySoftStart(double target_throttle);

		// Control flow selection
		bool m_useDirectFlow = true;  // Priorizar fluxo direto
		bool m_maintainZeroMQ = true; // Manter ZeroMQ para compatibilidade
	public:
		// === Flow Control Methods ===
		void setDirectFlowEnabled(bool enable) {
			m_useDirectFlow = enable;
			INFO_STREAM("ControlsManager") << "Direct flow " << (enable ? "ENABLED" : "DISABLED");
		}

		void setZeroMQMaintained(bool maintain) {
			m_maintainZeroMQ = maintain;
			INFO_STREAM("ControlsManager")
			    << "ZeroMQ compatibility " << (maintain ? "MAINTAINED" : "DISABLED");
		}

		// Get current flow status
		bool isUsingDirectFlow() const {
			return m_useDirectFlow;
		}
		bool isZeroMQMaintained() const {
			return m_maintainZeroMQ;
		}

	private:
		// === NEW: Thread-safe data access methods ===
		std::vector<Point2D> getCachedWaypoints();
		LaneInfo getCachedLaneInfo();
		bool getCachedEmergencyStop();

		// === NEW: Background data update threads ===
		void visionDataUpdateLoop();
		void obstacleDataUpdateLoop();

		// === NEW: Enhanced vehicle state estimation ===
		void updateVehicleStateEstimation(double applied_throttle, double applied_steering,
		                                  double dt);
		void integrateRealSensorData(); // For future sensor integration
		VehicleState getEnhancedVehicleState();

		// === REFACTORED: Direct ZMQ methods (now used only by background threads) ===
		std::vector<Point2D> getWaypointsFromVision();
		LaneInfo getLaneInfoFromVision();
		bool checkEmergencyObstacles();

		// Utility methods
		std::string serializeMask(const cv::Mat &mask);
		cv::Mat deserializeMask(const std::string &data);

	signals:
		void emergencyStopSignal();
		void modeChanged(DrivingMode mode);
};

#endif // CONTROLSMANAGER_HPP
