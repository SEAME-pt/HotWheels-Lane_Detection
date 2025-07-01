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

#ifndef DEFAULT_CONSTANT_SPEED_KMH
#define DEFAULT_CONSTANT_SPEED_KMH 2.0
#endif
#ifndef DEFAULT_CONSTANT_SPEED
#define DEFAULT_CONSTANT_SPEED (DEFAULT_CONSTANT_SPEED_KMH / 3.6)
#endif
#ifndef DEFAULT_CONSTANT_THROTTLE
#define DEFAULT_CONSTANT_THROTTLE 0.15
#endif

class ControlsManager : public QObject {
		Q_OBJECT
	private:
		EngineController m_engineController;
		DrivingMode m_currentMode;
		std::atomic<bool> m_running;
		std::atomic<bool> m_autonomousMode;
		std::atomic<double> m_lastThrottle{0.0};
		std::atomic<double> m_lastSteering{0.0};

		// === Thread Management ===
		std::atomic<bool> m_autonomousMode;
		std::atomic<double> m_lastThrottle{0.0};
		std::atomic<double> m_lastSteering{0.0};

		// === Thread Management ===
		QThread *m_cameraStreamerThread;
		QThread *m_manualControllerThread;
		QThread *m_subscriberJoystickThread;
		QThread *m_autonomousControlThread;
		QThread *m_visionDataThread;
		QThread *m_obstacleDataThread;
		QThread *m_visionDataThread;
		QThread *m_obstacleDataThread;

		// === Object Pointers ===
		CameraStreamer *m_cameraStreamerObject;
		JoysticksController *m_manualController;
		Subscriber *m_subscriberJoystickObject;
		// === Object Pointers ===
		CameraStreamer *m_cameraStreamerObject;
		JoysticksController *m_manualController;
		Subscriber *m_subscriberJoystickObject;
		MPCPlanner *m_mpcPlanner;
		Polyfitter *m_polyfitter;

		struct VehicleStateEstimator {
				std::atomic<bool> m_useRealSensors{false};
				std::atomic<double> m_realVelocity{0.0};
				std::atomic<double> m_realYawRate{0.0};
				VehicleState m_estimatedState{0.0, 0.0, 0.0, 0.0};
				std::chrono::steady_clock::time_point m_lastUpdate;
				static constexpr double PROCESS_NOISE_POS = 0.1;
				static constexpr double PROCESS_NOISE_VEL = 0.05;
				static constexpr double PROCESS_NOISE_YAW = 0.02;
				static constexpr double MEASUREMENT_NOISE_VEL = 0.1;
				std::mutex m_stateMutex;
				bool m_initialized = false;
				VehicleStateEstimator () {
					m_lastUpdate = std::chrono::steady_clock::now ();
				}
		} m_stateEstimator;

		std::unique_ptr<Subscriber> m_visionSubscriber;
		std::unique_ptr<Subscriber> m_obstacleSubscriber;
		struct CachedVisionData {
				std::vector<Point2D> waypoints;
				LaneInfo lane_info{0.0, 0.0};
				LaneInfo lane_info{0.0, 0.0};
				std::chrono::steady_clock::time_point timestamp;
				bool valid = false;
				std::mutex mutex;
		} m_cachedVisionData;
		struct DirectMPCData {
				LaneInfo current_lane_info;
				std::chrono::steady_clock::time_point timestamp;
				bool valid = false;
				std::mutex mutex;
		} m_directMPCData;
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
		static constexpr double CONTROL_RATE = 20.0;
		static constexpr double DATA_TIMEOUT_MS = 200.0;
		static constexpr double VISION_UPDATE_RATE = 10.0;
		static constexpr double OBSTACLE_UPDATE_RATE = 20.0;
		static constexpr double CONTROL_RATE = 20.0;
		static constexpr double DATA_TIMEOUT_MS = 200.0;
		static constexpr double VISION_UPDATE_RATE = 10.0;
		static constexpr double OBSTACLE_UPDATE_RATE = 20.0;
		bool m_constantSpeedMode = false;
		double m_targetConstantSpeed = DEFAULT_CONSTANT_SPEED;
		double m_constantThrottle = DEFAULT_CONSTANT_THROTTLE;
		void receiveLaneDataDirect (const LaneInfo &lane_info);
		std::atomic<bool> m_emergencyStop{false};
		struct SoftStartConfig {
				bool enabled = true;
				double max_throttle_change_per_step = 0.01;
				double initial_throttle_limit = 0.05;
				double warmup_duration_seconds = 3.0;
				double current_throttle_output = 0.0;
				std::chrono::steady_clock::time_point start_time;
				double max_throttle_change_per_step = 0.01;
				double initial_throttle_limit = 0.05;
				double warmup_duration_seconds = 3.0;
				double current_throttle_output = 0.0;
				std::chrono::steady_clock::time_point start_time;
		} m_softStart;

	public:
		explicit ControlsManager (int argc, char **argv, QObject *parent = nullptr);
		~ControlsManager ();
		void setMode (DrivingMode mode);
		void readJoystickEnable ();
		bool isProcessRunning (const QString &processName);
		void startAutonomousControl ();
		void stopAutonomousControl ();
		void autonomousControlLoop ();
		void enableRealSensors (bool enable = true);
		void updateRealVelocity (double velocity);
		void updateRealYawRate (double yaw_rate);
		VehicleState getVehicleStateWithDiagnostics ();
		void resetVehicleState (const VehicleState &initial_state = {0.0, 0.0, 0.0, 0.0});
		void applyControlCommand (const ControlCommand &command);
		void applyThrottle (double throttle);
		void applySteering (double steering);
		DrivingMode getCurrentMode () const {
			return m_currentMode;
		}
		void setConstantSpeedMode (bool enable, double target_speed = DEFAULT_CONSTANT_SPEED,
		                           double throttle = DEFAULT_CONSTANT_THROTTLE);
		void emergencyMotorStop ();
		void emergencyStop ();
		void resetEmergencyStop ();
		bool isEmergencyStopActive () const {
			return m_emergencyStop.load ();
		}
		bool isConstantSpeedMode () const {
			return m_constantSpeedMode;
		}
		double getTargetConstantSpeed () const {
			return m_targetConstantSpeed;
		}
		void setSoftStartEnabled (bool enabled) {
			m_softStart.enabled = enabled;
			std::cout << "[SOFT START] " << (enabled ? "ENABLED" : "DISABLED") << std::endl;
		}
		bool isSoftStartEnabled () const {
			return m_softStart.enabled;
		}
		void setSoftStartParameters (double max_change_per_step, double initial_limit,
		                             double warmup_duration) {
			m_softStart.max_throttle_change_per_step = max_change_per_step;
			m_softStart.initial_throttle_limit = initial_limit;
			m_softStart.warmup_duration_seconds = warmup_duration;
			std::cout << "[SOFT START] Parameters updated: Max change="
			          << "%, Warmup=" << warmup_duration << "s" << std::endl;
		}
		double applySoftStart (double target_throttle);
		bool m_useDirectFlow = true;
		bool m_maintainZeroMQ = true;

	public:
		void setDirectFlowEnabled (bool enable) {
			m_useDirectFlow = enable;
			INFO_STREAM ("ControlsManager") << "Direct flow " << (enable ? "ENABLED" : "DISABLED");
		}
		void setZeroMQMaintained (bool maintain) {
			m_maintainZeroMQ = maintain;
			INFO_STREAM ("ControlsManager")
			    << "ZeroMQ compatibility " << (maintain ? "MAINTAINED" : "DISABLED");
		}
		bool isUsingDirectFlow () const {
			return m_useDirectFlow;
		}
		bool isZeroMQMaintained () const {
			return m_maintainZeroMQ;
		}
		void stopAutonomousControlMotor ();
		bool getDirectLaneData (LaneInfo &lane_info);
		bool getZeroMQLaneData (LaneInfo &lane_info);
		void applyControlsWithSafety (const ControlCommand &control, int control_counter);
		cv::Mat deserializeMask (const std::string &data);

	private:
		void initializeHardwareControllers ();
		void initializeMPCComponents ();
		void initializeVisionPipeline (int argc, char **argv);
		void initializeCommunication (int argc, char **argv);
		void initializeDataThreads ();
		LaneInfo getCachedLaneInfo ();
		bool getCachedEmergencyStop ();
		void visionDataUpdateLoop ();
		void obstacleDataUpdateLoop ();
		std::vector<Point2D> getWaypointsFromVision ();
		LaneInfo getLaneInfoFromVision ();
		bool checkEmergencyObstacles ();
		std::string serializeMask (const cv::Mat &mask);
	signals:
		void emergencyStopSignal ();
		void modeChanged (DrivingMode mode);
};

#endif
