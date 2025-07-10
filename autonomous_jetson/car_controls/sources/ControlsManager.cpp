/*!
 * @file ControlsManager.cpp
 * @brief Implementation of the ControlsManager class.
 * @version 0.2
 * @date 2025-07-07
 * @details This file contains the implementation of the ControlsManager class,
 * which is responsible for managing the different controllers and worker threads
 * for the car controls with hybrid MPC architecture.
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "ControlsManager.hpp"
#include "Debugger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

/*!
 * @brief Constructs a ControlsManager object with hybrid MPC architecture.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param parent The parent QObject for this ControlsManager.
 * @details Initializes the engine controller, joystick controller, and various
 * worker threads for managing car controls with optimized direct MPC flow.
 */
ControlsManager::ControlsManager (int argc, char **argv, QObject *parent)
    : QObject (parent),
      // === Hardware Initialization ===
      m_engineController (0x40, 0x60, this), m_manualController (nullptr),

      // === State Initialization ===
      m_currentMode (DrivingMode::Manual), m_running (true), m_autonomousMode (false),

      // === Thread Pointers ===
      m_cameraStreamerThread (nullptr), m_manualControllerThread (nullptr),
      m_subscriberJoystickThread (nullptr), m_autonomousControlThread (nullptr),
      m_visionDataThread (nullptr), m_obstacleDataThread (nullptr),

      // === Object Pointers ===
      m_cameraStreamerObject (nullptr), m_subscriberJoystickObject (nullptr),
      m_mpcPlanner (nullptr), m_polyfitter (nullptr),

      // === Hybrid Flow Control ===
      m_useDirectFlow (true), m_maintainZeroMQ (true) {

	INFO_LOG ("ControlsManager", "Initializing ControlsManager with hybrid MPC architecture");

	// === ETAPA 1: Hardware Controllers ===
	initializeHardwareControllers ();

	// === ETAPA 2: MPC Components ===
	initializeMPCComponents ();

	// === ETAPA 3: Vision Pipeline ===
	initializeVisionPipeline (argc, argv);

	// === ETAPA 4: Communication Layers ===
	initializeCommunication (argc, argv);

	// === ETAPA 5: Data Processing Threads ===
	initializeDataThreads ();

	INFO_LOG ("ControlsManager", "ControlsManager initialization complete");
}

/*!
 * @brief Initialize hardware controllers (joystick and engine)
 */
void ControlsManager::initializeHardwareControllers () {
	// === Joystick Controller Setup ===
	m_manualController = new JoysticksController (
	    [this] (int steering) {
		    if (m_currentMode == DrivingMode::Manual) {
			    m_engineController.set_steering (steering);
		    }
	    },
	    [this] (int speed) {
		    if (m_currentMode == DrivingMode::Manual) {
			    m_engineController.set_speed (speed);
		    }
	    });

	if (!m_manualController->init ()) {
		ERROR_LOG ("ControlsManager", "Failed to initialize joystick controller");
		throw std::runtime_error ("Joystick initialization failed");
	}

	// === Joystick Thread Setup ===
	m_manualControllerThread = new QThread (this);
	m_manualController->moveToThread (m_manualControllerThread);

	connect (m_manualControllerThread, &QThread::started, m_manualController,
	         &JoysticksController::processInput);
	connect (m_manualController, &JoysticksController::finished, m_manualControllerThread,
	         &QThread::quit);

	m_manualControllerThread->start ();

	INFO_LOG ("ControlsManager", "Hardware controllers initialized");
}

/*!
 * @brief Initialize MPC components (Polyfitter and MPCPlanner)
 */
void ControlsManager::initializeMPCComponents () {
	// === Polyfitter for Lane Processing ===
	m_polyfitter = new Polyfitter ();

	// Configure Polyfitter for ZeroMQ publishing if needed
	if (m_maintainZeroMQ) {
		m_polyfitter->enableZeroMQPublishing (true);
		INFO_LOG ("ControlsManager", "Polyfitter ZeroMQ publishing enabled for external apps");
	}

	// === MPC Planner (lazy initialization) ===
	m_mpcPlanner = nullptr; // Will be created when autonomous mode starts

	INFO_LOG ("ControlsManager", "MPC components initialized");
}

/*!
 * @brief Initialize vision pipeline with direct MPC integration
 */
void ControlsManager::initializeVisionPipeline (int argc, char **argv) {
	// === Camera Streamer with Direct MPC Integration ===
	m_cameraStreamerThread = QThread::create ([this, argc, argv] () {
		try {
			m_cameraStreamerObject = new CameraStreamer (0.5);

			// === FLUXO DIRETO: Direct MPC callback ===
			if (m_useDirectFlow) {
				m_cameraStreamerObject->setMPCCallback (
				    [this] (const LaneInfo &lane_info) { receiveLaneDataDirect (lane_info); });
				INFO_LOG ("CameraStreamer", "Direct MPC flow enabled");
			}

			m_cameraStreamerObject->start ();

		} catch (const std::exception &e) {
			ERROR_STREAM ("ControlsManager") << "Vision pipeline error: " << e.what ();
		}
	});

	m_cameraStreamerThread->start ();
	INFO_LOG ("ControlsManager", "Vision pipeline initialized");
}

/*!
 * @brief Initialize communication layers (ZeroMQ subscribers)
 */
void ControlsManager::initializeCommunication (int argc, char **argv) {
	// === ZeroMQ Joystick Subscriber ===
	m_subscriberJoystickObject = new Subscriber ();
	m_subscriberJoystickThread = QThread::create ([this, argc, argv] () {
		m_subscriberJoystickObject->connect ("tcp://localhost:5555");
		m_subscriberJoystickObject->subscribe ("joystick_value");

		while (m_running) {
			try {
				zmq::pollitem_t items[] = {
				    {static_cast<void *> (m_subscriberJoystickObject->getSocket ()), 0, ZMQ_POLLIN,
				     0}};

				zmq::poll (items, 1, 100);

				if (items[0].revents & ZMQ_POLLIN) {
					zmq::message_t message;
					if (m_subscriberJoystickObject->getSocket ().recv (message,
					                                                   zmq::recv_flags::dontwait)) {
						std::string received_msg (static_cast<char *> (message.data ()),
						                          message.size ());

						if (received_msg.find ("joystick_value") == 0) {
							std::string value =
							    received_msg.substr (std::string ("joystick_value ").length ());
							if (value == "true") {
								setMode (DrivingMode::Manual);
							} else if (value == "false") {
								setMode (DrivingMode::Automatic);
							}
						}
					}
				}
			} catch (const zmq::error_t &e) {
				ERROR_STREAM ("ControlsManager") << "ZMQ communication error: " << e.what ();
				break;
			}
		}
	});

	m_subscriberJoystickThread->start ();
	INFO_LOG ("ControlsManager", "Communication layer initialized");
}

/*!
 * @brief Initialize data processing threads for ZeroMQ fallback
 */
void ControlsManager::initializeDataThreads () {
	if (m_maintainZeroMQ) {
		// === Vision Data Thread ===
		m_visionSubscriber = std::make_unique<Subscriber> ();
		m_visionDataThread = QThread::create ([this] () { visionDataUpdateLoop (); });
		m_visionDataThread->start ();

		// === Obstacle Data Thread ===
		m_obstacleSubscriber = std::make_unique<Subscriber> ();
		m_obstacleDataThread = QThread::create ([this] () { obstacleDataUpdateLoop (); });
		m_obstacleDataThread->start ();

		INFO_LOG ("ControlsManager", "ZeroMQ data threads initialized");
	}
}

/*!
 * @brief Destructor for the ControlsManager class.
 */
ControlsManager::~ControlsManager () {
	std::cout << "[~ControlsManager] CRITICAL SAFETY: Stopping all motors during cleanup"
	          << std::endl;

	// CRITICAL SAFETY: Stop motors immediately during destruction
	try {
		m_engineController.emergencyHardwareStop ();
		std::cout << "[~ControlsManager] Motors stopped successfully" << std::endl;
	} catch (const std::exception &e) {
		ERROR_STREAM ("ControlsManager")
		    << "[~ControlsManager] Error stopping motors: " << e.what ();
		try {
			m_engineController.set_speed (0);
			m_engineController.set_steering (0);
		} catch (...) {
			std::cerr << "[~ControlsManager] CRITICAL: Failed to stop motors during cleanup!"
			          << std::endl;
		}
	}

	m_running = false;
	stopAutonomousControl ();

	// === Stop data update threads first ===
	if (m_visionDataThread) {
		m_visionDataThread->quit ();
		if (!m_visionDataThread->wait (2000)) {
			m_visionDataThread->terminate ();
			m_visionDataThread->wait (1000);
		}
		delete m_visionDataThread;
		m_visionDataThread = nullptr;
	}

	if (m_obstacleDataThread) {
		m_obstacleDataThread->quit ();
		if (!m_obstacleDataThread->wait (2000)) {
			m_obstacleDataThread->terminate ();
			m_obstacleDataThread->wait (1000);
		}
		delete m_obstacleDataThread;
		m_obstacleDataThread = nullptr;
	}

	// Clean up persistent ZMQ connections
	m_visionSubscriber.reset ();
	m_obstacleSubscriber.reset ();

	// Stop threads safely
	if (m_subscriberJoystickThread) {
		if (m_subscriberJoystickObject) {
			m_subscriberJoystickObject->stop ();
		}
		m_subscriberJoystickThread->quit ();
		if (!m_subscriberJoystickThread->wait (3000)) {
			m_subscriberJoystickThread->terminate ();
			m_subscriberJoystickThread->wait (1000);
		}
		delete m_subscriberJoystickThread;
		m_subscriberJoystickThread = nullptr;
	}

	if (m_manualControllerThread) {
		if (m_manualController) m_manualController->requestStop ();
		m_manualControllerThread->quit ();
		if (!m_manualControllerThread->wait (3000)) {
			m_manualControllerThread->terminate ();
			m_manualControllerThread->wait (1000);
		}
		delete m_manualControllerThread;
		m_manualControllerThread = nullptr;
	}

	if (m_cameraStreamerThread) {
		if (m_cameraStreamerObject) m_cameraStreamerObject->stop ();
		m_cameraStreamerThread->quit ();
		if (!m_cameraStreamerThread->wait (3000)) {
			m_cameraStreamerThread->terminate ();
			m_cameraStreamerThread->wait (1000);
		}
		delete m_cameraStreamerThread;
		m_cameraStreamerThread = nullptr;
	}

	// Clean up objects
	delete m_cameraStreamerObject;
	m_cameraStreamerObject = nullptr;
	delete m_manualController;
	m_manualController = nullptr;
	delete m_subscriberJoystickObject;
	m_subscriberJoystickObject = nullptr;
	if (m_mpcPlanner) {
		delete m_mpcPlanner;
		m_mpcPlanner = nullptr;
	}
	if (m_polyfitter) {
		delete m_polyfitter;
		m_polyfitter = nullptr;
	}
}

/*!
 * @brief Sets the driving mode.
 */
void ControlsManager::setMode (DrivingMode mode) {
	if (m_currentMode == mode) return;

	m_currentMode = mode;

	if (m_currentMode == DrivingMode::Automatic) startAutonomousControl ();
	else
		stopAutonomousControlMotor ();
}

/*!
 * @brief Stop autonomous control motors
 */
void ControlsManager::stopAutonomousControlMotor () {
	m_engineController.set_speed (0);
	m_engineController.set_steering (0);

	if (m_currentMode == DrivingMode::Manual) {
		std::cout << "[STOP] Autonomous control stopped, motors set to zero" << std::endl;
	} else {
		std::cout << "[STOP] Autonomous control stopped, but still in automatic mode" << std::endl;
	}
}

/*!
 * @brief Get lane data from direct flow (primary source)
 */
bool ControlsManager::getDirectLaneData (LaneInfo &lane_info) {
	std::lock_guard<std::mutex> lock (m_directMPCData.mutex);
	auto now = std::chrono::steady_clock::now ();
	auto age_ms =
	    std::chrono::duration_cast<std::chrono::milliseconds> (now - m_directMPCData.timestamp)
	        .count ();

	if (m_directMPCData.valid && age_ms < 100) { // 100ms timeout
		lane_info = m_directMPCData.current_lane_info;
		return true;
	}
	return false;
}

/*!
 * @brief Get lane data from ZeroMQ fallback
 */
bool ControlsManager::getZeroMQLaneData (LaneInfo &lane_info) {
	if (!m_maintainZeroMQ) return false;

	std::lock_guard<std::mutex> lock (m_cachedVisionData.mutex);
	auto now = std::chrono::steady_clock::now ();
	auto age_ms =
	    std::chrono::duration_cast<std::chrono::milliseconds> (now - m_cachedVisionData.timestamp)
	        .count ();

	if (m_cachedVisionData.valid && age_ms < 200) { // 200ms timeout
		lane_info = m_cachedVisionData.lane_info;
		return true;
	}
	return false;
}

/*!
 * @brief Apply controls with safety limits and servo protection
 */
void ControlsManager::applyControlsWithSafety (const ControlCommand &control, int control_counter) {
	// === MPC FULL AUTHORITY MODE ===
	// Servo test confirmed proper command transmission - giving MPC full control
	// Physical safety switch is the primary safety mechanism

	// Convert to hardware values with EXPANDED limits for MPC authority
	int throttle_pct =
	    static_cast<int> (std::clamp (control.throttle * 100, 0.0, 25.0)); // Increased to 25%

	// MPC FULL STEERING AUTHORITY: Use hardware limits (±45° as confirmed by test)
	int steer_angle = static_cast<int> (
	    std::clamp (control.steer * 45.0, -45.0, 45.0)); // Full ±45° range for MPC

	// REMOVED rate limiting - MPC optimization handles smoothness
	// Trust MPC's internal optimization for smooth control
	static int last_servo_angle = 0;
	last_servo_angle = steer_angle; // Track but don't limit

	// Apply soft start to throttle
	double target_throttle = throttle_pct / 100.0;
	double final_throttle = applySoftStart (target_throttle);
	int final_throttle_pct = static_cast<int> (final_throttle * 100);

	// Store applied controls for state estimation
	m_lastThrottle = final_throttle;
	m_lastSteering = steer_angle * M_PI / 180.0;

	// Enhanced logging for MPC full control mode
	if (control_counter % 40 == 0) {
		std::cout << "[MPC FULL CONTROL] Target=" << throttle_pct
		          << "%, Final=" << final_throttle_pct << "%, Steering=" << steer_angle
		          << "° (FULL ±45° range enabled)" << std::endl;
	}

	// Apply to hardware (inverted speed for motor cross-connection fix)
	m_engineController.set_speed (-final_throttle_pct);
	m_engineController.set_steering (steer_angle);
}

/*!
 * @brief Start autonomous control with MPC
 */
void ControlsManager::startAutonomousControl () {
	if (m_currentMode != DrivingMode::Automatic) return;

	m_autonomousMode = true;

	// Create MPC planner if not already created
	if (!m_mpcPlanner) {
		m_mpcPlanner = new MPCPlanner ();
	}

	// Initialize soft start system
	m_softStart.current_throttle_output = 0.0;
	m_softStart.start_time = std::chrono::steady_clock::now ();

	std::cout << "[SOFT START] Autonomous mode activated with gradual acceleration" << std::endl;
	std::cout << "[SOFT START] Warmup period: " << m_softStart.warmup_duration_seconds << " seconds"
	          << std::endl;
	std::cout << "[SOFT START] Max throttle change per step: "
	          << (m_softStart.max_throttle_change_per_step * 100) << "%" << std::endl;

	m_autonomousControlThread = QThread::create ([this] () { autonomousControlLoop (); });
	m_autonomousControlThread->start ();
}

/*!
 * @brief Receive lane data directly from CameraStreamer (direct flow)
 */
void ControlsManager::receiveLaneDataDirect (const LaneInfo &lane_info) {
	std::lock_guard<std::mutex> lock (m_directMPCData.mutex);
	m_directMPCData.current_lane_info = lane_info;
	m_directMPCData.timestamp = std::chrono::steady_clock::now ();
	m_directMPCData.valid = true;
}

/*!
 * @brief Stop autonomous control
 */
void ControlsManager::stopAutonomousControl () {
	if (!m_autonomousMode) return;

	INFO_LOG ("ControlsManager", "Stopping autonomous control...");
	m_autonomousMode = false;

	if (m_autonomousControlThread) {
		m_autonomousControlThread->quit ();
		if (!m_autonomousControlThread->wait (2000)) {
			WARNING_LOG ("ControlsManager", "Autonomous thread did not finish gracefully");
			m_autonomousControlThread->terminate ();
			m_autonomousControlThread->wait (1000);
		}
		delete m_autonomousControlThread;
		m_autonomousControlThread = nullptr;
	}

	m_engineController.set_speed (0);
	m_engineController.set_steering (0);
	INFO_LOG ("ControlsManager", "Autonomous control stopped successfully");
}

/*!
 * @brief Main autonomous control loop with optimized hybrid architecture
 */
void ControlsManager::autonomousControlLoop () {
	const double CONTROL_PERIOD = 1.0 / CONTROL_RATE;
	auto last_control_time = std::chrono::steady_clock::now ();
	extern std::atomic<bool> g_running;

	INFO_LOG ("ControlsManager", "Autonomous control loop started (MPC delegated)");

	while (m_autonomousMode && m_running && g_running.load ()) {
		// 1. Segurança: parada de emergência
		if (m_emergencyStop.load ()) {
			m_engineController.set_speed (0);
			m_engineController.set_steering (0);
			INFO_LOG ("ControlsManager", "Emergency stop is active - motors stopped");
			std::this_thread::sleep_for (std::chrono::milliseconds (50));
			continue;
		}

		// 2. Controle de período (tempo real)
		auto now = std::chrono::steady_clock::now ();
		auto elapsed = std::chrono::duration<double> (now - last_control_time).count ();
		if (elapsed < CONTROL_PERIOD) {
			std::this_thread::sleep_for (std::chrono::microseconds (
			    static_cast<int> ((CONTROL_PERIOD - elapsed) * 1e6 * 0.8)));
			continue;
		}
		last_control_time = now;

		static int control_counter = 0;
		control_counter++;

		try {
			// 3. Estado do veículo
			VehicleState current_state = getVehicleStateWithDiagnostics ();

			// 4. Dados de faixa (LaneInfo) - fluxo direto ou fallback
			LaneInfo lane_info;
			bool has_valid_data = getDirectLaneData (lane_info);
			if (!has_valid_data && m_maintainZeroMQ) {
				has_valid_data = getZeroMQLaneData (lane_info);
				if (control_counter % 40 == 0) {
					DEBUG_LOG ("ControlsManager", "Using ZeroMQ fallback");
				}
			}
			if (!has_valid_data) {
				lane_info = m_mpcPlanner->generateStraightTrajectory ();
				if (control_counter % 40 == 0) {
					DEBUG_LOG ("ControlsManager", "Using straight fallback");
				}
			}

			// 5. Parada de emergência por obstáculos
			if (getCachedEmergencyStop ()) {
				m_engineController.set_speed (0);
				if (control_counter % 40 == 0) {
					INFO_LOG ("ControlsManager", "Emergency stop activated!");
				}
				continue;
			}

			// 6. Delegação para o MPC: cálculo do comando ótimo
			ControlCommand command = m_mpcPlanner->runAutonomousStep (current_state, lane_info);

			// 7. Modo de velocidade constante (opcional para diagnóstico)
			if (m_constantSpeedMode) {
				command.throttle = m_constantThrottle;
				command = m_mpcPlanner->applySmoothSteering (command);
			}

			// 8. Aplicação dos comandos ao hardware com proteção
			applyControlsWithSafety (command, control_counter);

		} catch (const std::exception &e) {
			ERROR_STREAM ("ControlsManager") << "Autonomous control error: " << e.what ();
			m_engineController.set_speed (0); // Segurança
		}
	}

	INFO_LOG ("ControlsManager", "Autonomous control loop ended");
}

// === Vehicle State Estimation Methods ===

VehicleState ControlsManager::getVehicleStateWithDiagnostics () {
	VehicleState state = m_mpcPlanner->getEnhancedVehicleState ();

	static int diagnostic_counter = 0;
	diagnostic_counter++;

	if (diagnostic_counter % 50 == 0) {
		INFO_LOG ("ControlsManager", "Vehicle State Diagnostics:");
		INFO_STREAM ("ControlsManager") << " Position: (" << state.x << ", " << state.y << ")";
		INFO_STREAM ("ControlsManager") << " Velocity: " << state.velocity << " m/s";
		INFO_STREAM ("ControlsManager") << " Yaw: " << state.yaw * 180.0 / M_PI << " degrees";
		INFO_STREAM ("ControlsManager") << " Applied throttle: " << m_lastThrottle.load ();
		INFO_STREAM ("ControlsManager")
		    << " Applied steering: " << m_lastSteering.load () * 180.0 / M_PI << " degrees";
	}

	return state;
}

// === ZeroMQ Methods (Fallback Support) ===

std::vector<Point2D> ControlsManager::getWaypointsFromVision () {
	std::vector<Point2D> waypoints;

	try {
		zmq::pollitem_t items[] = {
		    {static_cast<void *> (m_visionSubscriber->getSocket ()), 0, ZMQ_POLLIN, 0}};
		zmq::poll (items, 1, 50);

		if (items[0].revents & ZMQ_POLLIN) {
			zmq::message_t message;
			if (m_visionSubscriber->getSocket ().recv (message, zmq::recv_flags::dontwait)) {
				std::string received_msg (static_cast<char *> (message.data ()), message.size ());
				const std::string topic = "binary_mask ";

				if (received_msg.find (topic) == 0) {
					std::string mask_data = received_msg.substr (topic.size ());
					cv::Mat binary_mask = deserializeMask (mask_data);

					auto lanes = m_polyfitter->fitLanesInImage (binary_mask);
					CenterlineResult result = m_polyfitter->computeVirtualCenterline (
					    lanes, binary_mask.cols, binary_mask.rows);

					if (result.valid) {
						waypoints = result.blend;
					}
				}
			}
		}
	} catch (const zmq::error_t &e) {
		ERROR_STREAM ("ControlsManager") << "ZMQ error: " << e.what ();
	}

	if (waypoints.empty ()) {
		for (int i = 1; i <= 10; ++i) {
			waypoints.emplace_back (i * 2.0, 0.0);
		}
	}

	return waypoints;
}

LaneInfo ControlsManager::getLaneInfoFromVision () {
	try {
		zmq::pollitem_t items[] = {
		    {static_cast<void *> (m_visionSubscriber->getSocket ()), 0, ZMQ_POLLIN, 0}};
		zmq::poll (items, 1, 50);

		if (items[0].revents & ZMQ_POLLIN) {
			zmq::message_t message;
			if (m_visionSubscriber->getSocket ().recv (message, zmq::recv_flags::dontwait)) {
				std::string received_msg (static_cast<char *> (message.data ()), message.size ());
				const std::string topic = "binary_mask ";

				if (received_msg.find (topic) == 0) {
					std::string mask_data = received_msg.substr (topic.size ());
					cv::Mat binary_mask = deserializeMask (mask_data);

					auto lanes = m_polyfitter->fitLanesInImage (binary_mask);
					auto centerline = m_polyfitter->computeVirtualCenterline (
					    lanes, binary_mask.cols, binary_mask.rows);

					// Convert to LaneInfo
					LaneInfo lane_info;
					if (centerline.valid && !lanes.empty ()) {
						// Extract lane boundaries from detected lanes
						if (lanes.size () >= 2) {
							lane_info.left_boundary = lanes[0].centroids.front ().x;
							lane_info.right_boundary = lanes[1].centroids.front ().x;
							lane_info.center_line =
							    (lane_info.left_boundary + lane_info.right_boundary) / 2.0;
							lane_info.lateral_offset = 0.0;
							lane_info.yaw_error = 0.0;
							lane_info.isValid = true;
						}
					}
					return lane_info;
				}
			}
		}
	} catch (const zmq::error_t &e) {
		ERROR_STREAM ("ControlsManager") << "ZMQ error: " << e.what ();
	}

	return LaneInfo (0.0, 0.0);
}

bool ControlsManager::checkEmergencyObstacles () {
	try {
		zmq::pollitem_t items[] = {
		    {static_cast<void *> (m_obstacleSubscriber->getSocket ()), 0, ZMQ_POLLIN, 0}};
		zmq::poll (items, 1, 50);

		if (items[0].revents & ZMQ_POLLIN) {
			zmq::message_t message;
			if (m_obstacleSubscriber->getSocket ().recv (message, zmq::recv_flags::dontwait)) {
				std::string received_msg (static_cast<char *> (message.data ()), message.size ());
				const std::string topic = "emergency_stop ";

				if (received_msg.find (topic) == 0) {
					std::string obstacle_data = received_msg.substr (topic.size ());
					return (obstacle_data == "true");
				}
			}
		}
	} catch (const zmq::error_t &e) {
		ERROR_STREAM ("ControlsManager") << "ZMQ error: " << e.what ();
	}

	return false;
}

LaneInfo ControlsManager::getCachedLaneInfo () {
	std::lock_guard<std::mutex> lock (m_cachedVisionData.mutex);
	auto now = std::chrono::steady_clock::now ();
	auto age_ms =
	    std::chrono::duration_cast<std::chrono::milliseconds> (now - m_cachedVisionData.timestamp)
	        .count ();

	if (m_cachedVisionData.valid && age_ms < DATA_TIMEOUT_MS) {
		return m_cachedVisionData.lane_info;
	}

	return LaneInfo (0.0, 0.0);
}

bool ControlsManager::getCachedEmergencyStop () {
	std::lock_guard<std::mutex> lock (m_cachedObstacleData.mutex);
	auto now = std::chrono::steady_clock::now ();
	auto age_ms =
	    std::chrono::duration_cast<std::chrono::milliseconds> (now - m_cachedObstacleData.timestamp)
	        .count ();

	if (m_cachedObstacleData.valid && age_ms < DATA_TIMEOUT_MS) {
		return m_cachedObstacleData.emergency_stop;
	}

	return false;
}

// === Background Data Update Threads ===

void ControlsManager::visionDataUpdateLoop () {
	const double UPDATE_PERIOD = 1.0 / VISION_UPDATE_RATE;
	auto last_update_time = std::chrono::steady_clock::now ();
	extern std::atomic<bool> g_running;

	m_visionSubscriber->connect ("tcp://localhost:5556");
	m_visionSubscriber->subscribe ("binary_mask");
	INFO_LOG ("ControlsManager", "Vision data update thread started");

	while (m_running && g_running.load ()) {
		auto now = std::chrono::steady_clock::now ();
		auto elapsed = std::chrono::duration<double> (now - last_update_time).count ();

		if (elapsed < UPDATE_PERIOD) {
			std::this_thread::sleep_for (std::chrono::milliseconds (10));
			continue;
		}

		last_update_time = now;

		try {
			std::vector<Point2D> waypoints = getWaypointsFromVision ();
			LaneInfo lane_info = getLaneInfoFromVision ();

			{
				std::lock_guard<std::mutex> lock (m_cachedVisionData.mutex);
				m_cachedVisionData.waypoints = std::move (waypoints);
				m_cachedVisionData.lane_info = lane_info;
				m_cachedVisionData.timestamp = now;
				m_cachedVisionData.valid = true;
			}
		} catch (const std::exception &e) {
			ERROR_STREAM ("ControlsManager") << "Vision data update error: " << e.what ();
			{
				std::lock_guard<std::mutex> lock (m_cachedVisionData.mutex);
				m_cachedVisionData.valid = false;
			}
		}
	}

	INFO_LOG ("ControlsManager", "Vision data update thread ended");
}

void ControlsManager::obstacleDataUpdateLoop () {
	const double UPDATE_PERIOD = 1.0 / OBSTACLE_UPDATE_RATE;
	auto last_update_time = std::chrono::steady_clock::now ();
	extern std::atomic<bool> g_running;

	m_obstacleSubscriber->connect ("tcp://localhost:5557");
	m_obstacleSubscriber->subscribe ("emergency_stop");
	INFO_LOG ("ControlsManager", "Obstacle data update thread started");

	while (m_running && g_running.load ()) {
		auto now = std::chrono::steady_clock::now ();
		auto elapsed = std::chrono::duration<double> (now - last_update_time).count ();

		if (elapsed < UPDATE_PERIOD) {
			std::this_thread::sleep_for (std::chrono::milliseconds (5));
			continue;
		}

		last_update_time = now;

		try {
			bool emergency_stop = checkEmergencyObstacles ();

			{
				std::lock_guard<std::mutex> lock (m_cachedObstacleData.mutex);
				m_cachedObstacleData.emergency_stop = emergency_stop;
				m_cachedObstacleData.timestamp = now;
				m_cachedObstacleData.valid = true;
			}
		} catch (const std::exception &e) {
			ERROR_STREAM ("ControlsManager") << "Obstacle data update error: " << e.what ();
			{
				std::lock_guard<std::mutex> lock (m_cachedObstacleData.mutex);
				m_cachedObstacleData.valid = false;
			}
		}
	}

	INFO_LOG ("ControlsManager", "Obstacle data update thread ended");
}

// === Utility Methods ===

std::string ControlsManager::serializeMask (const cv::Mat &mask) {
	std::vector<uchar> buffer;
	cv::imencode (".png", mask, buffer);
	return std::string (buffer.begin (), buffer.end ());
}

// === Safety and Control Methods ===

void ControlsManager::setConstantSpeedMode (bool enable, double target_speed, double throttle) {
	m_constantSpeedMode = enable;
	m_targetConstantSpeed = target_speed;
	m_constantThrottle = throttle;

	if (enable) {
		std::cout << "[ControlsManager] CONSTANT SPEED MODE ENABLED:" << std::endl;
		std::cout << " Target speed: " << target_speed << " m/s" << std::endl;
		std::cout << " Fixed throttle: " << throttle << std::endl;
	} else {
		std::cout << "[ControlsManager] Constant speed mode DISABLED" << std::endl;
	}
}

void ControlsManager::emergencyMotorStop () {
	std::cout << "\n*** EMERGENCY MOTOR STOP ACTIVATED ***" << std::endl;

	try {
		m_engineController.set_speed (0);
		m_engineController.emergencyHardwareStop ();
	} catch (const std::exception &e) {
		ERROR_STREAM ("ControlsManager") << "Emergency stop error: " << e.what ();
		try {
			m_engineController.forcedMotorStop ();
		} catch (...) {
			ERROR_LOG ("ControlsManager", "CRITICAL: All motor stop methods failed!");
		}
	}

	try {
		m_engineController.set_steering (0);
	} catch (...) {
		ERROR_LOG ("ControlsManager", "Warning: Could not center steering");
	}

	m_constantSpeedMode = false;
	m_emergencyStop = true;
	std::cout << "*** MOTORS STOPPED - SYSTEM SAFE ***" << std::endl;
}

void ControlsManager::emergencyStop () {
	std::cout << "\n*** CRITICAL EMERGENCY STOP ACTIVATED ***" << std::endl;

	m_emergencyStop = true;
	m_constantSpeedMode = false;
	m_currentMode = DrivingMode::Manual;

	try {
		m_engineController.emergencyHardwareStop ();
		std::cout << "*** PRIMARY EMERGENCY STOP COMPLETED ***" << std::endl;
	} catch (const std::exception &e) {
		ERROR_STREAM ("ControlsManager") << "Emergency stop error: " << e.what ();
		try {
			m_engineController.forcedMotorStop ();
			std::cout << "*** FALLBACK FORCED STOP COMPLETED ***" << std::endl;
		} catch (const std::exception &e2) {
			ERROR_STREAM ("ControlsManager") << "Forced stop error: " << e2.what ();
			try {
				for (int i = 0; i < 5; ++i) {
					m_engineController.set_speed (0);
					std::this_thread::sleep_for (std::chrono::milliseconds (10));
				}
				std::cout << "*** BASIC STOP FALLBACK COMPLETED ***" << std::endl;
			} catch (...) {
				ERROR_LOG ("ControlsManager", "CRITICAL: ALL MOTOR STOP METHODS FAILED!");
			}
		}
	}

	try {
		m_engineController.set_steering (0);
	} catch (...) {
		ERROR_LOG ("ControlsManager", "Warning: Could not center steering");
	}

	std::cout << "*** EMERGENCY STOP COMPLETE - ALL SYSTEMS HALTED ***" << std::endl;
}

void ControlsManager::resetEmergencyStop () {
	std::cout << "[SAFETY] Resetting emergency stop flag..." << std::endl;
	m_emergencyStop = false;
	std::cout << "[SAFETY] Emergency stop flag cleared - system ready for operation" << std::endl;
}

double ControlsManager::applySoftStart (double target_throttle) {
	if (!m_softStart.enabled) {
		return target_throttle;
	}

	auto now = std::chrono::steady_clock::now ();
	auto elapsed_seconds = std::chrono::duration<double> (now - m_softStart.start_time).count ();

	double max_allowed_throttle;
	if (elapsed_seconds < m_softStart.warmup_duration_seconds) {
		double warmup_progress = elapsed_seconds / m_softStart.warmup_duration_seconds;
		max_allowed_throttle =
		    m_softStart.initial_throttle_limit +
		    (target_throttle - m_softStart.initial_throttle_limit) * warmup_progress;
		max_allowed_throttle = std::min (max_allowed_throttle, m_softStart.initial_throttle_limit +
		                                                           (0.3 * warmup_progress));
	} else {
		max_allowed_throttle = target_throttle;
	}

	double throttle_change = target_throttle - m_softStart.current_throttle_output;
	double max_change = m_softStart.max_throttle_change_per_step;

	if (std::abs (throttle_change) > max_change) {
		if (throttle_change > 0) {
			m_softStart.current_throttle_output += max_change;
		} else {
			m_softStart.current_throttle_output -= max_change;
		}
	} else {
		m_softStart.current_throttle_output = target_throttle;
	}

	m_softStart.current_throttle_output =
	    std::min (m_softStart.current_throttle_output, max_allowed_throttle);
	m_softStart.current_throttle_output =
	    std::clamp (m_softStart.current_throttle_output, 0.0, 1.0);

	static int log_counter = 0;
	if (++log_counter % 40 == 0 && elapsed_seconds < m_softStart.warmup_duration_seconds) {
		std::cout << "[SOFT START] Elapsed: " << std::fixed << std::setprecision (1)
		          << elapsed_seconds << "s, Target: " << std::setprecision (2)
		          << (target_throttle * 100)
		          << "%, Limited: " << (m_softStart.current_throttle_output * 100) << "%"
		          << std::endl;
	}

	return m_softStart.current_throttle_output;
}

// === Sensor Integration Methods ===

void ControlsManager::enableRealSensors (bool enable) {
	m_stateEstimator.m_useRealSensors.store (enable);
	INFO_STREAM ("ControlsManager") << "Real sensors " << (enable ? "enabled" : "disabled");
}

void ControlsManager::updateRealVelocity (double velocity) {
	m_stateEstimator.m_realVelocity.store (velocity);

	if (m_stateEstimator.m_useRealSensors.load ()) {
		std::lock_guard<std::mutex> lock (m_stateEstimator.m_stateMutex);
		const double sensor_weight = 0.3;
		m_stateEstimator.m_estimatedState.velocity =
		    sensor_weight * velocity +
		    (1.0 - sensor_weight) * m_stateEstimator.m_estimatedState.velocity;
	}
}

void ControlsManager::updateRealYawRate (double yaw_rate) {
	m_stateEstimator.m_realYawRate.store (yaw_rate);
}

void ControlsManager::resetVehicleState (const VehicleState &initial_state) {
	std::lock_guard<std::mutex> lock (m_stateEstimator.m_stateMutex);
	m_stateEstimator.m_estimatedState = initial_state;
	m_stateEstimator.m_lastUpdate = std::chrono::steady_clock::now ();

	qDebug () << "Vehicle state reset to: (" << initial_state.x << ", " << initial_state.y << ", "
	          << initial_state.yaw * 180.0 / M_PI << "°, " << initial_state.velocity << " m/s)";
}

// === Direct Control Methods ===

void ControlsManager::applyControlCommand (const ControlCommand &command) {
	applyThrottle (command.throttle);
	applySteering (command.steer);
}

void ControlsManager::applyThrottle (double throttle) {
	throttle = std::max (-1.0, std::min (1.0, throttle));
	int throttle_pwm = static_cast<int> (throttle * 100);
	m_engineController.set_speed (throttle_pwm);
	m_lastThrottle.store (throttle);
}

void ControlsManager::applySteering (double steering) {
	steering = std::max (-1.0, std::min (1.0, steering));
	int steering_pwm = static_cast<int> (steering * 100);
	m_engineController.set_steering (steering_pwm);
	m_lastSteering.store (steering);
}

cv::Mat ControlsManager::deserializeMask (const std::string &data) {
	std::vector<uchar> buffer (data.begin (), data.end ());
	return cv::imdecode (buffer, cv::IMREAD_GRAYSCALE);
}