/*!
 * @file ControlsManager.cpp
 * @brief Implementation of the ControlsManager class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the ControlsManager class,
 * which is responsible for managing the different controllers and worker
 * threads for the car controls.
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "ControlsManager.hpp"
#include <QDebug>
#include <chrono>
#include <condition_variable>
#include <fcntl.h>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

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
      m_manualControllerThread(nullptr), m_joystickControlThread(nullptr),
      m_subscriberJoystickThread(nullptr), m_cameraStreamerThread(nullptr), m_running(true),
      mpcPlanner(nullptr), m_polyfitter(nullptr), m_autonomousMode(false),
      m_autonomousControlThread(nullptr), m_visionDataThread(nullptr),
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
	connect(m_manualController, &JoysticksController::finished, m_manualControllerThread,
	        &QThread::quit);

	m_manualControllerThread->start();

	// **Running camera streamer**
	m_cameraStreamerThread = QThread::create([this, argc, argv]() {
		try {
			// Use video file instead of camera for testing
			bool use_video = true;
			std::string video_path = "/home/jetson/Videos/output_1803ok.avi";

			m_cameraStreamerObject = new CameraStreamer(0.5, use_video, video_path);
			m_cameraStreamerObject->start();
		} catch(const std::exception &e) {
			std::cerr << "Error: " << e.what() << std::endl;
		}
	});
	m_cameraStreamerThread->start();

	// **Client Middleware Interface Thread**
	m_subscriberJoystickObject = new Subscriber();
	m_subscriberJoystickThread = QThread::create([this, argc, argv]() {
		// Add external reference to global running flag
		extern std::atomic<bool> g_running;

		m_subscriberJoystickObject->connect("tcp://localhost:5555");
		m_subscriberJoystickObject->subscribe("joystick_value");
		while(m_running && g_running.load()) {
			try {
				zmq::pollitem_t items[] = {
				    {static_cast<void *>(m_subscriberJoystickObject->getSocket()), 0, ZMQ_POLLIN,
				     0}};

				// Wait up to 100ms for a message
				zmq::poll(items, 1, 100);

				if(items[0].revents & ZMQ_POLLIN) {
					zmq::message_t message;
					if(!m_subscriberJoystickObject->getSocket().recv(&message, 0)) {
						continue; // failed to receive
					}

					std::string received_msg(static_cast<char *>(message.data()), message.size());

					if(received_msg.find("joystick_value") == 0) {
						std::string value =
						    received_msg.substr(std::string("joystick_value ").length());
						if(value == "true") {
							setMode(DrivingMode::Manual);
						} else if(value == "false") {
							setMode(DrivingMode::Automatic);
						}
					}
				}
			} catch(const zmq::error_t &e) {
				std::cerr << "[Subscriber] ZMQ error: " << e.what() << std::endl;
				break; // exit safely if socket is closed
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

ControlsManager::~ControlsManager() {
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
	if(m_subscriberJoystickThread) {
		if(m_subscriberJoystickObject) {
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
void ControlsManager::setMode(DrivingMode mode) {
	if(m_currentMode == mode)
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
			cv::Mat().copyTo(cv::Mat()); // Force OpenCV memory cleanup
		}

		if(control_counter % 40 == 0) { // Log every 40th iteration (every ~2 seconds)
			qDebug() << "Autonomous control loop #" << control_counter << "- Using cached data";
		}

		try {
			// === OPTIMIZED: Use cached data instead of blocking ZMQ calls ===
			// 1. Get current vehicle state (no blocking operations)
			VehicleState current_state = getCurrentVehicleState();

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
			m_engineController.set_speed(throttle_pct);
			m_engineController.set_steering(steer_angle);

		} catch(const std::exception &e) {
			std::cerr << "Autonomous control error: " << e.what() << std::endl;
			qDebug() << "Autonomous control error:" << e.what();
			m_engineController.set_speed(0); // Safety stop
		}
	}

	qDebug() << "Autonomous control loop ended";
}

// Adicionar ao ControlsManager
VehicleState ControlsManager::getCurrentVehicleState() {
	static VehicleState state{0.0, 0.0, 0.0, 0.0};
	static bool initialized = false;

	// Implementação básica - pode ser melhorada com odometria real
	static auto last_time = std::chrono::steady_clock::now();
	auto now = std::chrono::steady_clock::now();
	double dt = std::chrono::duration<double>(now - last_time).count();
	last_time = now;

	if(!initialized) {
		// Initialize with some starting values
		state.x = 0.0;
		state.y = 0.0;
		state.yaw = 0.0;
		state.velocity = 0.5; // Start with some velocity
		initialized = true;
	}

	// Get applied controls
	double throttle = m_lastThrottle.load();
	double steering = m_lastSteering.load();

	// Simple kinematic model based on actual applied controls
	double wheelbase = 0.15; // 15cm wheelbase for typical RC car

	// Update velocity based on throttle
	state.velocity += throttle * dt * 2.0;                 // Max acceleration ~2 m/s²
	state.velocity = std::clamp(state.velocity, 0.1, 1.5); // Reasonable velocity limits

	// Update position and orientation
	state.x += state.velocity * std::cos(state.yaw) * dt;
	state.y += state.velocity * std::sin(state.yaw) * dt;

	// Update yaw based on steering (bicycle model)
	if(std::abs(steering) > 0.01) { // Only update if significant steering
		state.yaw += (state.velocity / wheelbase) * std::tan(steering) * dt;

		// Normalize yaw to [-π, π]
		while(state.yaw > M_PI)
			state.yaw -= 2.0 * M_PI;
		while(state.yaw < -M_PI)
			state.yaw += 2.0 * M_PI;
	}

	return state;
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

#include "ControlsManager.moc"
