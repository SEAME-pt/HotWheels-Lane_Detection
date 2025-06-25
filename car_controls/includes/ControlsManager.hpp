/*!
 * @file ControlsManager.hpp
 * @brief File containing the ControlsManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the declaration of the ControlsManager class,
 * which
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#ifndef CONTROLSMANAGER_HPP
#define CONTROLSMANAGER_HPP

#include "CameraStreamer.hpp"
#include "EngineController.hpp"
#include "JoysticksController.hpp"
#include "MPCPlanner.hpp"
#include "Polyfitter.hpp"
#include "Publisher.hpp"
#include "Subscriber.hpp"
#include <QObject>
#include <QProcess>
#include <QThread>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

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

		// === NEW: Persistent ZMQ connections and data caching ===
		// Persistent ZMQ subscribers to avoid repeated connection overhead
		std::unique_ptr<Subscriber> m_visionSubscriber;
		std::unique_ptr<Subscriber> m_obstacleSubscriber;
		QThread *m_visionDataThread;
		QThread *m_obstacleDataThread;

		// Cached data structures with thread-safe access
		struct CachedVisionData {
				std::vector<Point2D> waypoints;
				LaneInfo lane_info;
				std::chrono::steady_clock::time_point timestamp;
				bool valid = false;
				std::mutex mutex;
		} m_cachedVisionData;

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

	public:
		explicit ControlsManager(int argc, char **argv, QObject *parent = nullptr);
		~ControlsManager();

		void setMode(DrivingMode mode);
		void readJoystickEnable();
		bool isProcessRunning(const QString &processName);
		void startAutonomousControl();
		void stopAutonomousControl();
		void autonomousControlLoop();
		// Exibe a imagem da câmera com as lanes e centerline desenhadas
		void showVisionDebug();

		// Make this public so main.cpp can access it
		VehicleState getCurrentVehicleState();

	private:
		// === NEW: Thread-safe data access methods ===
		std::vector<Point2D> getCachedWaypoints();
		LaneInfo getCachedLaneInfo();
		bool getCachedEmergencyStop();

		// === NEW: Background data update threads ===
		void visionDataUpdateLoop();
		void obstacleDataUpdateLoop();

		// === REFACTORED: Direct ZMQ methods (now used only by background threads) ===
		std::vector<Point2D> getWaypointsFromVision();
		LaneInfo getLaneInfoFromVision();
		bool checkEmergencyObstacles();

		// Utility methods
		std::string serializeMask(const cv::Mat &mask);
		cv::Mat deserializeMask(const std::string &data);

	signals:
		void emergencyStop();
		void modeChanged(DrivingMode mode);
};

#endif // CONTROLSMANAGER_HPP
