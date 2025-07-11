#include "AutonomousMode.hpp"
#include "Debugger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

AutonomousMode::AutonomousMode(EngineController* engineController, QObject *parent)
    : QObject(parent), m_autonomousMode(false), m_visionDataThread(nullptr), 
      m_obstacleDataThread(nullptr), m_autonomousControlThread(nullptr),
      m_mpcPlanner(nullptr), m_polyfitter(nullptr), m_useDirectFlow(true),
      m_maintainZeroMQ(true), m_engineController(engineController) {
	
	// Initialize MPC components
	initializeMPCComponents();
	
	// Initialize data processing threads
	initializeDataThreads();
}

AutonomousMode::AutonomousMode (void)
    : m_autonomousMode (false), m_visionDataThread (nullptr), m_obstacleDataThread (nullptr),
      m_mpcPlanner (nullptr), m_polyfitter (nullptr), m_useDirectFlow (true),
      m_maintainZeroMQ (true) {}

AutonomousMode::AutonomousMode (const AutonomousMode &origin) {
	*this = origin;
}

AutonomousMode &AutonomousMode::operator= (const AutonomousMode &origin) {
	if (this != &origin) *this = origin;
	return *this;
}

AutonomousMode::~AutonomousMode (void) {
	stopAutonomousControl();
	
	// Clean up MPC objects
	if (m_mpcPlanner) {
		delete m_mpcPlanner;
		m_mpcPlanner = nullptr;
	}
	if (m_polyfitter) {
		delete m_polyfitter;
		m_polyfitter = nullptr;
	}
}

void AutonomousMode::initializeMPCComponents () {
	// === Polyfitter for Lane Processing ===
	m_polyfitter = new Polyfitter ();

	// Configure Polyfitter for ZeroMQ publishing if needed
	if (m_maintainZeroMQ) {
		m_polyfitter->enableZeroMQPublishing (true);
		INFO_LOG ("AutonomousMode", "Polyfitter ZeroMQ publishing enabled for external apps");
	}

	// === MPC Planner (lazy initialization) ===
	m_mpcPlanner = nullptr; // Will be created when autonomous mode starts

	INFO_LOG ("AutonomousMode", "MPC components initialized");
}

void AutonomousMode::initializeDataThreads() {
	if (m_maintainZeroMQ) {
		// === Vision Data Thread ===
		m_visionSubscriber = std::make_unique<Subscriber>();
		m_visionDataThread = QThread::create([this]() { visionDataUpdateLoop(); });
		m_visionDataThread->start();

		// === Obstacle Data Thread ===
		m_obstacleSubscriber = std::make_unique<Subscriber>();
		m_obstacleDataThread = QThread::create([this]() { obstacleDataUpdateLoop(); });
		m_obstacleDataThread->start();

		INFO_LOG("AutonomousMode", "ZeroMQ data threads initialized");
	}
}

void AutonomousMode::startAutonomousControl() {
	if (m_autonomousMode.load()) return;

	m_autonomousMode = true;

	// Create MPC planner if not already created
	if (!m_mpcPlanner) {
		m_mpcPlanner = new MPCPlanner();
	}

	// Initialize soft start system
	m_softStart.current_throttle_output = 0.0;
	m_softStart.start_time = std::chrono::steady_clock::now();

	std::cout << "[SOFT START] Autonomous mode activated with gradual acceleration" << std::endl;

	m_autonomousControlThread = QThread::create([this]() { autonomousControlLoop(); });
	m_autonomousControlThread->start();
}

void AutonomousMode::stopAutonomousControl() {
	if (!m_autonomousMode.load()) return;

	INFO_LOG("AutonomousMode", "Stopping autonomous control...");
	m_autonomousMode = false;

	if (m_autonomousControlThread) {
		m_autonomousControlThread->quit();
		if (!m_autonomousControlThread->wait(2000)) {
			WARNING_LOG("AutonomousMode", "Autonomous thread did not finish gracefully");
			m_autonomousControlThread->terminate();
			m_autonomousControlThread->wait(1000);
		}
		delete m_autonomousControlThread;
		m_autonomousControlThread = nullptr;
	}

	if (m_engineController) {
		m_engineController->set_speed(0);
		m_engineController->set_steering(0);
	}
	INFO_LOG("AutonomousMode", "Autonomous control stopped successfully");
}

void AutonomousMode::receiveLaneDataDirect(const LaneInfo &lane_info) {
	std::lock_guard<std::mutex> lock(m_directMPCData.mutex);
	m_directMPCData.current_lane_info = lane_info;
	m_directMPCData.timestamp = std::chrono::steady_clock::now();
	m_directMPCData.valid = true;
}

bool AutonomousMode::getDirectLaneData(LaneInfo &lane_info) {
	std::lock_guard<std::mutex> lock(m_directMPCData.mutex);
	auto now = std::chrono::steady_clock::now();
	auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_directMPCData.timestamp).count();

	if (m_directMPCData.valid && age_ms < 100) { // 100ms timeout
		lane_info = m_directMPCData.current_lane_info;
		return true;
	}
	return false;
}

void AutonomousMode::autonomousControlLoop() {
	const double CONTROL_PERIOD = 1.0 / CONTROL_RATE;
	auto last_control_time = std::chrono::steady_clock::now();
	extern std::atomic<bool> g_running;

	INFO_LOG("AutonomousMode", "Autonomous control loop started (MPC delegated)");

	while (m_autonomousMode.load() && g_running.load()) {
		// Emergency stop check
		if (m_emergencyStop.load()) {
			if (m_engineController) {
				m_engineController->set_speed(0);
				m_engineController->set_steering(0);
			}
			INFO_LOG("AutonomousMode", "Emergency stop is active - motors stopped");
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
			continue;
		}

		// Control period timing
		auto now = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration<double>(now - last_control_time).count();
		if (elapsed < CONTROL_PERIOD) {
			std::this_thread::sleep_for(std::chrono::microseconds(
				static_cast<int>((CONTROL_PERIOD - elapsed) * 1e6 * 0.8)));
			continue;
		}
		last_control_time = now;

		static int control_counter = 0;
		control_counter++;

		try {
			// Get vehicle state
			VehicleState current_state = getVehicleStateWithDiagnostics();

			// Get lane data
			LaneInfo lane_info;
			bool has_valid_data = getDirectLaneData(lane_info);
			if (!has_valid_data && m_maintainZeroMQ) {
				has_valid_data = getZeroMQLaneData(lane_info);
				if (control_counter % 40 == 0) {
					DEBUG_LOG("AutonomousMode", "Using ZeroMQ fallback");
				}
			}
			if (!has_valid_data) {
				lane_info = m_mpcPlanner->generateStraightTrajectory();
				if (control_counter % 40 == 0) {
					DEBUG_LOG("AutonomousMode", "Using straight fallback");
				}
			}

			// Check emergency obstacles
			if (getCachedEmergencyStop()) {
				if (m_engineController) {
					m_engineController->set_speed(0);
				}
				if (control_counter % 40 == 0) {
					INFO_LOG("AutonomousMode", "Emergency stop activated!");
				}
				continue;
			}

			// MPC control calculation
			ControlCommand command = m_mpcPlanner->runAutonomousStep(current_state, lane_info);

			// Constant speed mode override
			if (m_constantSpeedMode) {
				command.throttle = m_constantThrottle;
				command = m_mpcPlanner->applySmoothSteering(command);
			}

			// Apply controls with safety
			applyControlsWithSafety(command, control_counter);

		} catch (const std::exception &e) {
			ERROR_STREAM("AutonomousMode") << "Autonomous control error: " << e.what();
			if (m_engineController) {
				m_engineController->set_speed(0); // Safety
			}
		}
	}

	INFO_LOG("AutonomousMode", "Autonomous control loop ended");
}

void AutonomousMode::applyControlsWithSafety(const ControlCommand &control, int control_counter) {
	if (!m_engineController) return;

	// Convert to hardware values with safety limits
	int throttle_pct = static_cast<int>(std::clamp(control.throttle * 100, 0.0, 25.0));
	int steer_angle = static_cast<int>(std::clamp(control.steer * 45.0, -45.0, 45.0));

	// Apply soft start to throttle
	double target_throttle = throttle_pct / 100.0;
	double final_throttle = applySoftStart(target_throttle);
	int final_throttle_pct = static_cast<int>(final_throttle * 100);

	// Store applied controls for state estimation
	m_lastThrottle = final_throttle;
	m_lastSteering = steer_angle * M_PI / 180.0;

	// Enhanced logging for MPC control mode
	if (control_counter % 40 == 0) {
		std::cout << "[MPC CONTROL] Target=" << throttle_pct
				  << "%, Final=" << final_throttle_pct << "%, Steering=" << steer_angle
				  << "°" << std::endl;
	}

	// Apply to hardware
	m_engineController->set_speed(-final_throttle_pct); // Inverted for motor cross-connection
	m_engineController->set_steering(steer_angle);
}

// Add implementations for other MPC-related methods moved from ControlsManager...
// (applySoftStart, getVehicleStateWithDiagnostics, visionDataUpdateLoop, etc.)

void AutonomousMode::setDirectFlowEnabled(bool enable) {
	m_useDirectFlow = enable;
	INFO_STREAM("AutonomousMode") << "Direct flow " << (enable ? "ENABLED" : "DISABLED");
}

void AutonomousMode::setZeroMQMaintained(bool maintain) {
	m_maintainZeroMQ = maintain;
	INFO_STREAM("AutonomousMode") << "ZeroMQ compatibility " << (maintain ? "MAINTAINED" : "DISABLED");
}

// Stub implementations for methods that need to be moved from ControlsManager
double AutonomousMode::applySoftStart(double target_throttle) {
	// Implementation from ControlsManager
	return target_throttle; // Simplified for now
}

VehicleState AutonomousMode::getVehicleStateWithDiagnostics() {
	// Implementation from ControlsManager
	return VehicleState{0.0, 0.0, 0.0, 0.0}; // Simplified for now
}

void AutonomousMode::visionDataUpdateLoop() {
	// Implementation from ControlsManager
}

void AutonomousMode::obstacleDataUpdateLoop() {
	// Implementation from ControlsManager
}

// Additional method stubs that need full implementations...
