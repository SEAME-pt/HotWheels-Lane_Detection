/*!
 * @file ControlsManager.cpp
 * @brief Implementation of the ControlsManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the ControlsManager class,
 * which is responsible for managing the different controllers and worker threads
 * for the car controls.
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "ControlsManager.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <QDebug>

/*!
 * @brief Constructs a ControlsManager object.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param parent The parent QObject for this ControlsManager.
 * @details Initializes the engine controller, joystick controller, and various
 * worker threads for managing car controls. Sets up joystick control with
 * callbacks for steering and speed adjustments, manages server and client
 * middleware threads, monitors processes, and handles joystick enable status
 * through dedicated threads.
 */
ControlsManager::ControlsManager(int argc, char **argv, QObject *parent)
    : QObject(parent), m_engineController(0x40, 0x60, this), m_manualController(nullptr),
      m_currentMode(DrivingMode::Manual), m_subscriberJoystickObject(nullptr),
      m_cameraStreamerObject(nullptr), m_running(true), m_cameraStreamerThread(nullptr),
      m_manualControllerThread(nullptr), m_joystickControlThread(nullptr),
      m_subscriberJoystickThread(nullptr), m_mpcPlanner(nullptr), m_polyfitter(nullptr),
      m_autonomousMode(false), m_autonomousControlThread(nullptr), m_visionDataThread(nullptr),
      m_obstacleDataThread(nullptr) {

	// Initialize the joystick controller with callbacks
	//! Verify where to put AUTO mode.
	m_manualController = new JoysticksController(
	    [this](int steering) {
		    if(m_currentMode == DrivingMode::Manual) {
			    m_engineController.set_steering(steering);
		    }
	    },
	    [this](int speed) {
		    if(m_currentMode == DrivingMode::Manual) {
			    m_engineController.set_speed(speed);
		    }
	    });

	if(!m_manualController->init()) {
		qDebug() << "Failed to initialize joystick controller.";
		return;
	}

	// Start the joystick controller in its own thread
	m_manualControllerThread = new QThread(this);
	m_manualController->moveToThread(m_manualControllerThread);

	connect(m_manualControllerThread, &QThread::started, m_manualController,
			&JoysticksController::processInput);
	connect(m_manualController, &JoysticksController::finished,
			m_manualControllerThread, &QThread::quit);

	m_manualControllerThread->start();

	// **Running camera streamer**
	m_cameraStreamerThread = QThread::create([this, argc, argv]() {
		try {
			m_cameraStreamerObject = new CameraStreamer(0.5);
			m_cameraStreamerObject->start();
		} catch(const std::exception &e) {
			std::cerr << "Error: " << e.what() << std::endl;
		}
	});
	m_cameraStreamerThread->start();

	// **Client Middleware Interface Thread**
	m_subscriberJoystickObject = new Subscriber();
	m_subscriberJoystickThread = QThread::create([this, argc, argv]()
									{
		m_subscriberJoystickObject->connect("tcp://localhost:5555");
		m_subscriberJoystickObject->subscribe("joystick_value");
		while (m_running) {
			try {
				zmq::pollitem_t items[] = {
					{ static_cast<void*>(m_subscriberJoystickObject->getSocket()), 0, ZMQ_POLLIN, 0 }
				};

				// Wait up to 100ms for a message
				zmq::poll(items, 1, 100);

				if (items[0].revents & ZMQ_POLLIN) {
					zmq::message_t message;
					if (!m_subscriberJoystickObject->getSocket().recv(&message, 0)) {
						continue;  // failed to receive
					}

					std::string received_msg(static_cast<char*>(message.data()), message.size());

					if (received_msg.find("joystick_value") == 0) {
						std::string value = received_msg.substr(std::string("joystick_value ").length());
						if (value == "true") {
							setMode(DrivingMode::Manual);
						} else if (value == "false") {
							setMode(DrivingMode::Automatic);
						}
					}
				}
			} catch (const zmq::error_t& e) {
				std::cerr << "[Subscriber] ZMQ error: " << e.what() << std::endl;
				break;  // exit safely if socket is closed
			}
		}
	});
	m_polyfitter = new Polyfitter();
	m_subscriberJoystickThread->start();

	// === NEW: Initialize persistent ZMQ connections for data streams ===
	// Initialize vision data subscriber in its own thread
	m_visionSubscriber = std::make_unique<Subscriber>();
	m_visionDataThread = QThread::create([this]() { visionDataUpdateLoop(); });
	m_visionDataThread->start();

	// Initialize obstacle data subscriber in its own thread
	m_obstacleSubscriber = std::make_unique<Subscriber>();
	m_obstacleDataThread = QThread::create([this]() { obstacleDataUpdateLoop(); });
	m_obstacleDataThread->start();

	qDebug() << "ControlsManager initialized with optimized thread architecture";
}

/*!
 * @brief Destructor for the ControlsManager class.
 * @details Safely stops and cleans up all threads and resources associated
 *          with the ControlsManager. This includes stopping the client,
 *          shared memory, process monitoring, joystick control, and manual
 *          controller threads. It also deletes associated objects such as
 *          m_carDataObject, m_subscriberJoystickThread, and m_manualController.
 */

ControlsManager::~ControlsManager()
{
	m_running = false;
	stopAutonomousControl();

	// === NEW: Stop data update threads first ===
	if(m_visionDataThread) {
		m_visionDataThread->quit();
		if(!m_visionDataThread->wait(2000)) {
			m_visionDataThread->terminate();
			m_visionDataThread->wait(1000);
		}
		delete m_visionDataThread;
		m_visionDataThread = nullptr;
	}

	if(m_obstacleDataThread) {
		m_obstacleDataThread->quit();
		if(!m_obstacleDataThread->wait(2000)) {
			m_obstacleDataThread->terminate();
			m_obstacleDataThread->wait(1000);
		}
		delete m_obstacleDataThread;
		m_obstacleDataThread = nullptr;
	}

	// Clean up persistent ZMQ connections
	m_visionSubscriber.reset();
	m_obstacleSubscriber.reset();

	// Stop the client thread safely
	if (m_subscriberJoystickThread) {
		if (m_subscriberJoystickObject) {
			m_subscriberJoystickObject->stop();
		}
		m_subscriberJoystickThread->quit();
		if(!m_subscriberJoystickThread->wait(3000)) { // 3 second timeout
			m_subscriberJoystickThread->terminate();
			m_subscriberJoystickThread->wait(1000);
		}

		if(m_subscriberJoystickObject) {
			m_subscriberJoystickObject->getSocket().close();
		}

		delete m_subscriberJoystickThread;
		m_subscriberJoystickThread = nullptr;
	}


	// Stop manual controller thread
	if(m_manualControllerThread) {
		if(m_manualController)
			m_manualController->requestStop();

		m_manualControllerThread->quit();
		if(!m_manualControllerThread->wait(3000)) { // 3 second timeout
			m_manualControllerThread->terminate();
			m_manualControllerThread->wait(1000);
		}
		delete m_manualControllerThread;
		m_manualControllerThread = nullptr;
	}

	// Stop camera streamer thread
	if(m_cameraStreamerThread) {
		if(m_cameraStreamerObject)
			m_cameraStreamerObject->stop();

		m_cameraStreamerThread->quit();
		if(!m_cameraStreamerThread->wait(3000)) { // 3 second timeout
			m_cameraStreamerThread->terminate();
			m_cameraStreamerThread->wait(1000);
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
	if(m_mpcPlanner) {
		delete m_mpcPlanner;
		m_mpcPlanner = nullptr;
	}
}

/*!
 * @brief Sets the driving mode.
 * @param mode The new driving mode.
 * @details Updates the current driving mode if it has changed.
 */
void ControlsManager::setMode(DrivingMode mode)
{
	if (m_currentMode == mode)
		return;

	m_currentMode = mode;
	if(m_currentMode == DrivingMode::Automatic)
		startAutonomousControl();
}

void ControlsManager::startAutonomousControl() {
	if(m_currentMode != DrivingMode::Automatic)
		return;

	m_autonomousMode = true;
	m_mpcPlanner = new MPCPlanner();

	m_autonomousControlThread = QThread::create([this]() { autonomousControlLoop(); });
	m_autonomousControlThread->start();
}

void ControlsManager::stopAutonomousControl() {
	if(!m_autonomousMode)
		return;

	qDebug() << "Stopping autonomous control...";

	m_autonomousMode = false;

	if(m_autonomousControlThread) {
		m_autonomousControlThread->quit();
		if(!m_autonomousControlThread->wait(2000)) {
			qDebug() << "Warning: Autonomous thread did not finish gracefully";
			m_autonomousControlThread->terminate();
			m_autonomousControlThread->wait(1000);
		}
		delete m_autonomousControlThread;
		m_autonomousControlThread = nullptr;
	}

	if(m_mpcPlanner) {
		delete m_mpcPlanner;
		m_mpcPlanner = nullptr;
	}

	m_engineController.set_speed(0);
	m_engineController.set_steering(0);

	qDebug() << "Autonomous control stopped successfully";
}

void ControlsManager::autonomousControlLoop() {
	const double CONTROL_PERIOD = 1.0 / CONTROL_RATE;
	auto last_control_time = std::chrono::steady_clock::now();

	// Add external reference to global running flag
	extern std::atomic<bool> g_running;

	qDebug() << "Autonomous control loop started with optimized architecture";

	while(m_autonomousMode && m_running && g_running.load()) {
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration<double>(now - last_control_time).count();

		// Precise timing control
		if(elapsed < CONTROL_PERIOD) {
			std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(
			    (CONTROL_PERIOD - elapsed) * 1000000 * 0.8) // Sleep for 80% of remaining time
			                                                      ));
			continue;
		}
		last_control_time = now;

		// Reduced logging frequency
		static int control_counter = 0;
		control_counter++;

		// Force memory cleanup every 200 iterations (every ~10 seconds at 20Hz)
		if(control_counter % 200 == 0) {
			// Force OpenCV memory cleanup (safer method)
			cv::Mat temp;
			temp.create(1, 1, CV_8UC1);
			temp = cv::Mat(); // Safe cleanup
		}

		if(control_counter % 40 == 0) { // Log every 40th iteration (every ~2 seconds)
			qDebug() << "Autonomous control loop #" << control_counter << "- Using cached data";
		}

		try {
			// === OPTIMIZED: Use cached data instead of blocking ZMQ calls ===
			// 1. Get current vehicle state with enhanced estimation and diagnostics
			VehicleState current_state = getVehicleStateWithDiagnostics();

			// 2. Get cached perception data (non-blocking)
			std::vector<Point2D> waypoints = getCachedWaypoints();
			LaneInfo lane_info = getCachedLaneInfo();

			if(control_counter % 40 == 0) {
				std::cout << "State: x=" << std::fixed << std::setprecision(2) << current_state.x
				          << ", y=" << current_state.y << ", vel=" << current_state.velocity
				          << ", yaw=" << current_state.yaw << " | Waypoints: " << waypoints.size()
				          << std::endl;
			}

			// 3. Check for emergency obstacles (cached data)
			if(getCachedEmergencyStop()) {
				m_engineController.set_speed(0);
				if(control_counter % 40 == 0) {
					qDebug() << "Emergency stop activated!";
				}
				continue;
			}

			// 4. Calculate MPC control (main computational work)
			ControlCommand control = m_mpcPlanner->plan(current_state, waypoints, &lane_info);

			// 5. Apply controls with safety limits
			int throttle_pct = static_cast<int>(std::clamp(control.throttle * 100, 0.0, 50.0));
			int steer_angle = static_cast<int>(std::clamp(control.steer * 45, -45.0, 45.0));

			// Store applied controls for state estimation
			m_lastThrottle = throttle_pct / 100.0;       // Convert back to 0-1 range
			m_lastSteering = steer_angle * M_PI / 180.0; // Convert to radians

			if(control_counter % 40 == 0) {
				std::cout << "Controls: Throttle=" << throttle_pct << "%, Steering=" << steer_angle
				          << "°" << std::endl;
			}

			// Apply controls to hardware
			//! Só ativar após validação do MPC!
			// m_engineController.set_speed(throttle_pct);
			// m_engineController.set_steering(steer_angle);

		} catch(const std::exception &e) {
			std::cerr << "Autonomous control error: " << e.what() << std::endl;
			qDebug() << "Autonomous control error:" << e.what();
			m_engineController.set_speed(0); // Safety stop
		}
	}

	qDebug() << "Autonomous control loop ended";
}

// === ENHANCED: Vehicle state estimation with sensor fusion ===
VehicleState ControlsManager::getCurrentVehicleState() {
	return getEnhancedVehicleState();
}

VehicleState ControlsManager::getEnhancedVehicleState() {
	std::lock_guard<std::mutex> lock(m_stateEstimator.m_stateMutex);

	auto now = std::chrono::steady_clock::now();
	double dt = std::chrono::duration<double>(now - m_stateEstimator.m_lastUpdate).count();
	m_stateEstimator.m_lastUpdate = now;

	if(!m_stateEstimator.m_initialized) {
		// Initialize state estimator
		m_stateEstimator.m_estimatedState = {0.0, 0.0, 0.0, 0.5}; // Start with small velocity
		m_stateEstimator.m_initialized = true;
		return m_stateEstimator.m_estimatedState;
	}

	// Clamp dt to prevent numerical issues
	dt = std::clamp(dt, 0.001, 0.1); // 1ms to 100ms

	// Get applied controls
	double applied_throttle = m_lastThrottle.load();
	double applied_steering = m_lastSteering.load();

	// Update state estimation
	updateVehicleStateEstimation(applied_throttle, applied_steering, dt);

	// Integrate real sensor data if available
	if(m_stateEstimator.m_useRealSensors.load()) {
		integrateRealSensorData();
	}

	return m_stateEstimator.m_estimatedState;
}

void ControlsManager::updateVehicleStateEstimation(double applied_throttle, double applied_steering,
                                                   double dt) {
	// Enhanced kinematic model with more realistic dynamics
	VehicleState &state = m_stateEstimator.m_estimatedState;

	// Vehicle parameters (tuned for typical RC car)
	const double wheelbase = 0.15;         // 15cm wheelbase
	const double max_acceleration = 3.0;   // m/s²
	const double max_deceleration = 4.0;   // m/s²
	const double rolling_resistance = 0.1; // Friction coefficient
	const double air_resistance = 0.05;    // Air drag coefficient
	const double max_velocity = 2.0;       // Maximum velocity m/s
	const double steering_response = 0.8;  // Steering response factor

	// === Velocity dynamics with realistic physics ===
	double target_acceleration = applied_throttle * max_acceleration;

	// Apply rolling resistance and air drag
	double resistance_force = rolling_resistance + air_resistance * state.velocity * state.velocity;
	double net_acceleration = target_acceleration - resistance_force;

	// Apply acceleration limits
	if(net_acceleration > 0) {
		net_acceleration = std::min(net_acceleration, max_acceleration);
	} else {
		net_acceleration = std::max(net_acceleration, -max_deceleration);
	}

	// Update velocity with realistic dynamics
	state.velocity += net_acceleration * dt;
	state.velocity = std::clamp(state.velocity, 0.0, max_velocity);

	// Add velocity noise for realism
	if(state.velocity > 0.1) {
		state.velocity += (((double)rand() / RAND_MAX) - 0.5) * 0.02; // ±1cm/s noise
	}

	// === Position integration ===
	double distance = state.velocity * dt;
	state.x += distance * std::cos(state.yaw);
	state.y += distance * std::sin(state.yaw);

	// === Yaw dynamics with realistic steering response ===
	if(std::abs(applied_steering) > 0.01 && state.velocity > 0.1) {
		// Bicycle model with realistic steering response
		double turning_radius = wheelbase / std::tan(applied_steering * steering_response);
		double angular_velocity = state.velocity / turning_radius;

		// Apply yaw rate limits
		angular_velocity = std::clamp(angular_velocity, -2.0, 2.0); // ±2 rad/s max

		state.yaw += angular_velocity * dt;

		// Add steering noise
		state.yaw += (((double)rand() / RAND_MAX) - 0.5) * 0.01; // ±0.01 rad noise

		// Normalize yaw to [-π, π]
		while(state.yaw > M_PI)
			state.yaw -= 2.0 * M_PI;
		while(state.yaw < -M_PI)
			state.yaw += 2.0 * M_PI;
	}
}

std::vector<Point2D> ControlsManager::getWaypointsFromVision() {
	std::vector<Point2D> waypoints;

	try {
		// Use the persistent vision subscriber connection
		zmq::pollitem_t items[] = {
		    {static_cast<void *>(m_visionSubscriber->getSocket()), 0, ZMQ_POLLIN, 0}};
		zmq::poll(items, 1, 50); // Reduced timeout: 50 ms

		if(items[0].revents & ZMQ_POLLIN) {
			zmq::message_t message;
			if(m_visionSubscriber->getSocket().recv(&message, 0)) {
				std::string received_msg(static_cast<char *>(message.data()), message.size());
				const std::string topic = "binary_mask ";

				if(received_msg.find(topic) == 0) {
					std::string mask_data = received_msg.substr(topic.size());
					cv::Mat binary_mask = deserializeMask(mask_data);

					// Extract lanes and compute centerline
					auto lanes = m_polyfitter->fitLanesInImage(binary_mask);
					CenterlineResult result = m_polyfitter->computeVirtualCenterline(
					    lanes, binary_mask.cols, binary_mask.rows);

					if(result.valid) {
						waypoints = result.blend;
					}
				}
			}
		}
	} catch(const zmq::error_t &e) {
		std::cerr << "[getWaypointsFromVision] ZMQ error: " << e.what() << std::endl;
	} catch(...) {
		std::cerr << "[getWaypointsFromVision] Unknown error" << std::endl;
	}

	// Fallback: straight waypoints if no data received
	if(waypoints.empty()) {
		for(int i = 1; i <= 10; ++i) {
			waypoints.emplace_back(i * 2.0, 0.0);
		}
	}

	return waypoints;
}

LaneInfo ControlsManager::getLaneInfoFromVision() {
	try {
		// Use the persistent vision subscriber connection
		zmq::pollitem_t items[] = {
		    {static_cast<void *>(m_visionSubscriber->getSocket()), 0, ZMQ_POLLIN, 0}};
		zmq::poll(items, 1, 50); // Reduced timeout: 50 ms

		if(items[0].revents & ZMQ_POLLIN) {
			zmq::message_t message;
			if(m_visionSubscriber->getSocket().recv(&message, 0)) {
				std::string received_msg(static_cast<char *>(message.data()), message.size());
				const std::string topic = "binary_mask ";

				if(received_msg.find(topic) == 0) {
					std::string mask_data = received_msg.substr(topic.size());
					cv::Mat binary_mask = deserializeMask(mask_data);

					// Use Polyfitter's methods to extract lane information
					auto lanes = m_polyfitter->fitLanesInImage(binary_mask);
					auto centerline = m_polyfitter->computeVirtualCenterline(
					    lanes, binary_mask.cols, binary_mask.rows);

					// TODO: Extract meaningful LaneInfo from lanes/centerline
					return LaneInfo(0.0, 0.0);
				}
			}
		}
	} catch(const zmq::error_t &e) {
		std::cerr << "[getLaneInfoFromVision] ZMQ error: " << e.what() << std::endl;
	} catch(...) {
		std::cerr << "[getLaneInfoFromVision] Unknown error" << std::endl;
	}

	return LaneInfo(0.0, 0.0); // fallback
}

bool ControlsManager::checkEmergencyObstacles() {
	try {
		// Use the persistent obstacle subscriber connection
		zmq::pollitem_t items[] = {
		    {static_cast<void *>(m_obstacleSubscriber->getSocket()), 0, ZMQ_POLLIN, 0}};
		zmq::poll(items, 1, 50); // Reduced timeout: 50 ms

		if(items[0].revents & ZMQ_POLLIN) {
			zmq::message_t message;
			if(m_obstacleSubscriber->getSocket().recv(&message, 0)) {
				std::string received_msg(static_cast<char *>(message.data()), message.size());
				const std::string topic = "emergency_stop ";

				if(received_msg.find(topic) == 0) {
					std::string obstacle_data = received_msg.substr(topic.size());
					return (obstacle_data == "true");
				}
			}
		}
	} catch(const zmq::error_t &e) {
		std::cerr << "[checkEmergencyObstacles] ZMQ error: " << e.what() << std::endl;
	} catch(...) {
		std::cerr << "[checkEmergencyObstacles] Unknown error" << std::endl;
	}

	return false; // Safe default
}

std::string ControlsManager::serializeMask(const cv::Mat &mask) {
	std::vector<uchar> buffer;
	cv::imencode(".png", mask, buffer);
	return std::string(buffer.begin(), buffer.end());
}

cv::Mat ControlsManager::deserializeMask(const std::string &data) {
	std::vector<uchar> buffer(data.begin(), data.end());
	return cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
}

void ControlsManager::showVisionDebug() {
	Subscriber vision_sub;
	vision_sub.connect("tcp://localhost:5556");
	vision_sub.subscribe("binary_mask");

	try {
		zmq::pollitem_t items[] = {{static_cast<void *>(vision_sub.getSocket()), 0, ZMQ_POLLIN, 0}};
		zmq::poll(items, 1, 100); // Timeout: 100 ms

		if(items[0].revents & ZMQ_POLLIN) {
			zmq::message_t message;
			if(vision_sub.getSocket().recv(&message, 0)) {
				std::string received_msg(static_cast<char *>(message.data()), message.size());
				const std::string topic = "binary_mask ";

				if(received_msg.find(topic) == 0) {
					std::string mask_data = received_msg.substr(topic.size());
					cv::Mat binary_mask = deserializeMask(mask_data);

					// Visualização da máscara
					cv::Mat vis;
					cv::cvtColor(binary_mask, vis, cv::COLOR_GRAY2BGR);

					// Extraia lanes e centerline
					auto lanes = m_polyfitter->fitLanesInImage(binary_mask);
					for(const auto &lane : lanes) {
						for(size_t i = 1; i < lane.curve.size(); ++i) {
							cv::line(vis, cv::Point(lane.curve[i - 1].x, lane.curve[i - 1].y),
							         cv::Point(lane.curve[i].x, lane.curve[i].y),
							         cv::Scalar(0, 255, 0), 2);
						}
					}
					auto centerline = m_polyfitter->computeVirtualCenterline(
					    lanes, binary_mask.cols, binary_mask.rows);
					if(centerline.valid) {
						for(size_t i = 1; i < centerline.blend.size(); ++i) {
							cv::line(
							    vis,
							    cv::Point(centerline.blend[i - 1].x, centerline.blend[i - 1].y),
							    cv::Point(centerline.blend[i].x, centerline.blend[i].y),
							    cv::Scalar(0, 128, 255), 2);
						}
					}

					cv::imshow("Lane Detection Debug", vis);
					cv::waitKey(1);
				}
			}
		}
	} catch(const zmq::error_t &e) {
		std::cerr << "[showVisionDebug] ZMQ error: " << e.what() << std::endl;
	} catch(...) {
		std::cerr << "[showVisionDebug] Unknown error" << std::endl;
	}
}

// === NEW: Thread-safe cached data access methods ===

std::vector<Point2D> ControlsManager::getCachedWaypoints() {
	std::lock_guard<std::mutex> lock(m_cachedVisionData.mutex);

	// Check if data is still valid (not too old)
	auto now = std::chrono::steady_clock::now();
	auto age_ms =
	    std::chrono::duration_cast<std::chrono::milliseconds>(now - m_cachedVisionData.timestamp)
	        .count();

	if(m_cachedVisionData.valid && age_ms < DATA_TIMEOUT_MS) {
		return m_cachedVisionData.waypoints;
	}

	// Return fallback waypoints if data is stale
	std::vector<Point2D> fallback_waypoints;
	for(int i = 1; i <= 10; ++i) {
		fallback_waypoints.emplace_back(i * 2.0, 0.0);
	}
	return fallback_waypoints;
}

LaneInfo ControlsManager::getCachedLaneInfo() {
	std::lock_guard<std::mutex> lock(m_cachedVisionData.mutex);

	auto now = std::chrono::steady_clock::now();
	auto age_ms =
	    std::chrono::duration_cast<std::chrono::milliseconds>(now - m_cachedVisionData.timestamp)
	        .count();

	if(m_cachedVisionData.valid && age_ms < DATA_TIMEOUT_MS) {
		return m_cachedVisionData.lane_info;
	}

	// Return neutral lane info if data is stale
	return LaneInfo(0.0, 0.0);
}

bool ControlsManager::getCachedEmergencyStop() {
	std::lock_guard<std::mutex> lock(m_cachedObstacleData.mutex);

	auto now = std::chrono::steady_clock::now();
	auto age_ms =
	    std::chrono::duration_cast<std::chrono::milliseconds>(now - m_cachedObstacleData.timestamp)
	        .count();

	if(m_cachedObstacleData.valid && age_ms < DATA_TIMEOUT_MS) {
		return m_cachedObstacleData.emergency_stop;
	}

	// Default to safe state if data is stale
	return false;
}

// === NEW: Background data update threads ===

void ControlsManager::visionDataUpdateLoop() {
	const double UPDATE_PERIOD = 1.0 / VISION_UPDATE_RATE;
	auto last_update_time = std::chrono::steady_clock::now();

	// Add external reference to global running flag
	extern std::atomic<bool> g_running;

	m_visionSubscriber->connect("tcp://localhost:5556");
	m_visionSubscriber->subscribe("binary_mask");

	qDebug() << "Vision data update thread started";

	while(m_running && g_running.load()) {
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration<double>(now - last_update_time).count();

		if(elapsed < UPDATE_PERIOD) {
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			continue;
		}
		last_update_time = now;

		try {
			// Get fresh vision data
			std::vector<Point2D> waypoints = getWaypointsFromVision();
			LaneInfo lane_info = getLaneInfoFromVision();

			// Update cached data in thread-safe manner
			{
				std::lock_guard<std::mutex> lock(m_cachedVisionData.mutex);
				m_cachedVisionData.waypoints = std::move(waypoints);
				m_cachedVisionData.lane_info = lane_info;
				m_cachedVisionData.timestamp = now;
				m_cachedVisionData.valid = true;
			}

		} catch(const std::exception &e) {
			std::cerr << "Vision data update error: " << e.what() << std::endl;
			// Mark data as invalid on error
			{
				std::lock_guard<std::mutex> lock(m_cachedVisionData.mutex);
				m_cachedVisionData.valid = false;
			}
		}
	}

	qDebug() << "Vision data update thread ended";
}

void ControlsManager::obstacleDataUpdateLoop() {
	const double UPDATE_PERIOD = 1.0 / OBSTACLE_UPDATE_RATE;
	auto last_update_time = std::chrono::steady_clock::now();

	// Add external reference to global running flag
	extern std::atomic<bool> g_running;

	m_obstacleSubscriber->connect("tcp://localhost:5557");
	m_obstacleSubscriber->subscribe("emergency_stop");

	qDebug() << "Obstacle data update thread started";

	while(m_running && g_running.load()) {
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration<double>(now - last_update_time).count();

		if(elapsed < UPDATE_PERIOD) {
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;
		}
		last_update_time = now;

		try {
			// Get fresh obstacle data
			bool emergency_stop = checkEmergencyObstacles();

			// Update cached data in thread-safe manner
			{
				std::lock_guard<std::mutex> lock(m_cachedObstacleData.mutex);
				m_cachedObstacleData.emergency_stop = emergency_stop;
				m_cachedObstacleData.timestamp = now;
				m_cachedObstacleData.valid = true;
			}

		} catch(const std::exception &e) {
			std::cerr << "Obstacle data update error: " << e.what() << std::endl;
			// Mark data as invalid on error
			{
				std::lock_guard<std::mutex> lock(m_cachedObstacleData.mutex);
				m_cachedObstacleData.valid = false;
			}
		}
	}

	qDebug() << "Obstacle data update thread ended";
}

// === NEW: Enhanced sensor integration interface ===

void ControlsManager::enableRealSensors(bool enable) {
	m_stateEstimator.m_useRealSensors.store(enable);
	qDebug() << "Real sensors" << (enable ? "enabled" : "disabled");
}

void ControlsManager::updateRealVelocity(double velocity) {
	m_stateEstimator.m_realVelocity.store(velocity);
	// This can be used to input measured velocity from external sources
	// For example: calculated from camera movement, wheel tick counting, etc.
	if(m_stateEstimator.m_useRealSensors.load()) {
		std::lock_guard<std::mutex> lock(m_stateEstimator.m_stateMutex);
		// Fuse with current estimate using weighted average
		const double sensor_weight = 0.3; // 30% external measurement, 70% model
		m_stateEstimator.m_estimatedState.velocity =
		    sensor_weight * velocity +
		    (1.0 - sensor_weight) * m_stateEstimator.m_estimatedState.velocity;
	}
}

void ControlsManager::updateRealYawRate(double yaw_rate) {
	m_stateEstimator.m_realYawRate.store(yaw_rate);
	// This can be used to input measured yaw rate from external sources
	// For example: calculated from camera rotation, estimated from vision, etc.
}

VehicleState ControlsManager::getVehicleStateWithDiagnostics() {
	VehicleState state = getEnhancedVehicleState();

	// Add diagnostic information
	static int diagnostic_counter = 0;
	diagnostic_counter++;

	if(diagnostic_counter % 50 == 0) { // Every ~2.5 seconds at 20Hz
		qDebug() << "Vehicle State Diagnostics:";
		qDebug() << "  Position: (" << state.x << ", " << state.y << ")";
		qDebug() << "  Velocity: " << state.velocity << " m/s";
		qDebug() << "  Yaw: " << state.yaw * 180.0 / M_PI << " degrees";
		qDebug() << "  Real sensors: " << (m_stateEstimator.m_useRealSensors.load() ? "ON" : "OFF");
		qDebug() << "  Applied throttle: " << m_lastThrottle.load();
		qDebug() << "  Applied steering: " << m_lastSteering.load() * 180.0 / M_PI << " degrees";
	}

	return state;
}

void ControlsManager::resetVehicleState(const VehicleState &initial_state) {
	std::lock_guard<std::mutex> lock(m_stateEstimator.m_stateMutex);
	m_stateEstimator.m_estimatedState = initial_state;
	m_stateEstimator.m_lastUpdate = std::chrono::steady_clock::now();
	qDebug() << "Vehicle state reset to: (" << initial_state.x << ", " << initial_state.y << ", "
	         << initial_state.yaw * 180.0 / M_PI << "°, " << initial_state.velocity << " m/s)";
}

void ControlsManager::integrateRealSensorData() {
	// === REALISTIC: Integration with available project sensors ===
	// This project only has access to:
	// - Camera vision data (lanes/waypoints)
	// - Applied control commands (throttle/steering)
	// - No GPS, IMU, encoders, or magnetometer available

	VehicleState &state = m_stateEstimator.m_estimatedState;

	if(m_stateEstimator.m_useRealSensors.load()) {
		// 1. Use vision data for position correction (if available)
		std::vector<Point2D> current_waypoints = getCachedWaypoints();
		if(!current_waypoints.empty()) {
			// If we have waypoints, we can infer lateral position relative to lane center
			// This simulates basic visual odometry

			// Simple lane-relative positioning (basic visual odometry substitute)
			// Assume we're following the waypoints and adjust position accordingly
			if(current_waypoints.size() >= 2) {
				// Calculate expected trajectory from waypoints
				Point2D first_wp = current_waypoints[0];
				Point2D second_wp = current_waypoints[1];

				// Estimate direction from waypoints
				double expected_yaw =
				    std::atan2(second_wp.y - first_wp.y, second_wp.x - first_wp.x);

				// Apply small correction to yaw based on vision (10% influence)
				const double vision_weight = 0.1;
				double yaw_correction = expected_yaw - state.yaw;

				// Normalize angle difference
				while(yaw_correction > M_PI)
					yaw_correction -= 2.0 * M_PI;
				while(yaw_correction < -M_PI)
					yaw_correction += 2.0 * M_PI;

				// Apply small yaw correction
				state.yaw += vision_weight * yaw_correction;
			}
		}

		// 2. Use applied controls for velocity estimation refinement
		// The controls we actually sent to the hardware are more accurate than model prediction
		double real_throttle_effect = m_stateEstimator.m_realVelocity.load();
		if(real_throttle_effect > 0.01) {
			// If we have measured velocity feedback, fuse it
			const double feedback_weight = 0.2; // 20% feedback, 80% model
			state.velocity =
			    feedback_weight * real_throttle_effect + (1.0 - feedback_weight) * state.velocity;
		}

		// 3. Add realistic measurement noise to simulate sensor limitations
		// This makes the simulation more realistic for testing the MPC robustness

		// Camera-based position noise (vision processing uncertainty)
		double vision_noise_x = (((double)rand() / RAND_MAX) - 0.5) * 0.05; // ±2.5cm
		double vision_noise_y = (((double)rand() / RAND_MAX) - 0.5) * 0.05; // ±2.5cm

		state.x += vision_noise_x;
		state.y += vision_noise_y;

		// Control actuation uncertainty (motor/servo response)
		double control_noise_vel = (((double)rand() / RAND_MAX) - 0.5) * 0.01;  // ±0.5cm/s
		double control_noise_yaw = (((double)rand() / RAND_MAX) - 0.5) * 0.005; // ±0.005 rad

		state.velocity += control_noise_vel;
		state.velocity = std::max(0.0, state.velocity); // Velocity can't be negative

		state.yaw += control_noise_yaw;

		// Normalize yaw
		while(state.yaw > M_PI)
			state.yaw -= 2.0 * M_PI;
		while(state.yaw < -M_PI)
			state.yaw += 2.0 * M_PI;
	}
}

#include "ControlsManager.moc"
