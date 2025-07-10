#include "Debugger.hpp"
#include <chrono>
#include <experimental/filesystem>

// DebugStream implementation
DebugStream::DebugStream(Debugger *dbg, LogLevel lvl, const std::string &comp)
    : debugger(dbg), level(lvl), component(comp) {}

DebugStream::~DebugStream() {
	// When the DebugStream object is destroyed, log the accumulated message
	if(debugger) {
		debugger->log(level, component, stream.str());
	}
}

// Static member definitions
std::unique_ptr<Debugger> Debugger::instance = nullptr;
std::mutex Debugger::mutex_;

Debugger::Debugger()
    : console_output_enabled(false), file_output_enabled(true), min_log_level(LogLevel::DEBUG) {

	base_output_dir = "outputs";

	// Create outputs directory if it doesn't exist
	std::experimental::filesystem::create_directories(base_output_dir);

	startNewSession();
}

Debugger::~Debugger() {
	endSession();
}

Debugger *Debugger::getInstance() {
	std::lock_guard<std::mutex> lock(mutex_);
	if(instance == nullptr) {
		instance = std::unique_ptr<Debugger>(new Debugger());
	}
	return instance.get();
}

void Debugger::startNewSession() {
	// Generate session ID with timestamp
	auto now = std::time(nullptr);
	auto tm = *std::localtime(&now);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
	session_id = oss.str();

	// Close any existing files
	if(debug_file.is_open())
		debug_file.close();
	if(mpc_file.is_open())
		mpc_file.close();
	if(vision_file.is_open())
		vision_file.close();
	if(control_file.is_open())
		control_file.close();

	// Open new debug files
	if(file_output_enabled) {
		debug_file.open(getDebugFilePath(), std::ios::out | std::ios::trunc);
		mpc_file.open(getMPCFilePath(), std::ios::out | std::ios::trunc);
		vision_file.open(getVisionFilePath(), std::ios::out | std::ios::trunc);
		control_file.open(getControlFilePath(), std::ios::out | std::ios::trunc);

		// Write headers
		if(debug_file.is_open()) {
			debug_file << "=== JETRACER DEBUG SESSION ===" << std::endl;
			debug_file << "Session ID: " << session_id << std::endl;
			debug_file << "Started: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;
			debug_file << "Vehicle: Jetracer (wheelbase=150mm, max_steer=±22.5°)" << std::endl;
			debug_file << "=============================" << std::endl;
			debug_file.flush();
		}

		if(mpc_file.is_open()) {
			mpc_file << "=== MPC CONTROLLER DEBUG ===" << std::endl;
			mpc_file << "Session: " << session_id << std::endl;
			mpc_file << "Columns: Timestamp,Level,Message" << std::endl;
			mpc_file << "============================" << std::endl;
			mpc_file.flush();
		}

		if(vision_file.is_open()) {
			vision_file << "=== VISION SYSTEM DEBUG ===" << std::endl;
			vision_file << "Session: " << session_id << std::endl;
			vision_file << "Columns: Timestamp,Level,Message" << std::endl;
			vision_file << "===========================" << std::endl;
			vision_file.flush();
		}

		if(control_file.is_open()) {
			control_file << "=== CONTROL SYSTEM DEBUG ===" << std::endl;
			control_file << "Session: " << session_id << std::endl;
			control_file << "Columns: Timestamp,Level,Message" << std::endl;
			control_file << "=============================" << std::endl;
			control_file.flush();
		}
	}

	log(LogLevel::INFO, "Debugger", "New debug session started: " + session_id);
}

void Debugger::endSession() {
	log(LogLevel::INFO, "Debugger", "Ending debug session: " + session_id);

	if(debug_file.is_open()) {
		debug_file << "=== SESSION ENDED ===" << std::endl;
		debug_file.close();
	}
	if(mpc_file.is_open()) {
		mpc_file << "=== SESSION ENDED ===" << std::endl;
		mpc_file.close();
	}
	if(vision_file.is_open()) {
		vision_file << "=== SESSION ENDED ===" << std::endl;
		vision_file.close();
	}
	if(control_file.is_open()) {
		control_file << "=== SESSION ENDED ===" << std::endl;
		control_file.close();
	}
}

std::string Debugger::getCurrentTimestamp() const {
	auto now = std::time(nullptr);
	auto tm = *std::localtime(&now);
	std::ostringstream oss;
	oss << std::put_time(&tm, "%H:%M:%S");
	return oss.str();
}

std::string Debugger::getLogLevelString(LogLevel level) const {
	switch(level) {
	case LogLevel::DEBUG:
		return "DEBUG";
	case LogLevel::INFO:
		return "INFO";
	case LogLevel::WARNING:
		return "WARN";
	case LogLevel::ERROR:
		return "ERROR";
	case LogLevel::CRITICAL:
		return "CRIT";
	default:
		return "UNKNOWN";
	}
}

void Debugger::writeToFile(std::ofstream &file, const std::string &message) {
	if(file.is_open()) {
		file << message << std::endl;
		file.flush();
	}
}

void Debugger::log(LogLevel level, const std::string &component, const std::string &message) {
	if(level < min_log_level)
		return;

	std::string timestamp = getCurrentTimestamp();
	std::string level_str = getLogLevelString(level);

	std::ostringstream formatted_msg;
	formatted_msg << "[" << timestamp << "] [" << level_str << "] [" << component << "] "
	              << message;

	if(console_output_enabled) {
		std::cout << formatted_msg.str() << std::endl;
	}

	if(file_output_enabled) {
		writeToFile(debug_file, formatted_msg.str());
	}
}

void Debugger::logMPC(LogLevel level, const std::string &message) {
	if(level < min_log_level)
		return;

	std::string timestamp = getCurrentTimestamp();
	std::string level_str = getLogLevelString(level);

	std::ostringstream formatted_msg;
	formatted_msg << "[" << timestamp << "] [" << level_str << "] " << message;

	if(console_output_enabled) {
		std::cout << "[MPC] " << formatted_msg.str() << std::endl;
	}

	if(file_output_enabled) {
		writeToFile(mpc_file, formatted_msg.str());
	}
}

void Debugger::logVision(LogLevel level, const std::string &message) {
	if(level < min_log_level)
		return;

	std::string timestamp = getCurrentTimestamp();
	std::string level_str = getLogLevelString(level);

	std::ostringstream formatted_msg;
	formatted_msg << "[" << timestamp << "] [" << level_str << "] " << message;

	if(console_output_enabled) {
		std::cout << "[VISION] " << formatted_msg.str() << std::endl;
	}

	if(file_output_enabled) {
		writeToFile(vision_file, formatted_msg.str());
	}
}

void Debugger::logControl(LogLevel level, const std::string &message) {
	if(level < min_log_level)
		return;

	std::string timestamp = getCurrentTimestamp();
	std::string level_str = getLogLevelString(level);

	std::ostringstream formatted_msg;
	formatted_msg << "[" << timestamp << "] [" << level_str << "] " << message;

	if(console_output_enabled) {
		std::cout << "[CONTROL] " << formatted_msg.str() << std::endl;
	}

	if(file_output_enabled) {
		writeToFile(control_file, formatted_msg.str());
	}
}

// Convenience methods
void Debugger::debug(const std::string &component, const std::string &message) {
	log(LogLevel::DEBUG, component, message);
}

void Debugger::info(const std::string &component, const std::string &message) {
	log(LogLevel::INFO, component, message);
}

void Debugger::warning(const std::string &component, const std::string &message) {
	log(LogLevel::WARNING, component, message);
}

void Debugger::error(const std::string &component, const std::string &message) {
	log(LogLevel::ERROR, component, message);
}

void Debugger::critical(const std::string &component, const std::string &message) {
	log(LogLevel::CRITICAL, component, message);
}

// Stream-style logging methods
DebugStream Debugger::debug_stream(const std::string &component) {
	return DebugStream(this, LogLevel::DEBUG, component);
}

DebugStream Debugger::info_stream(const std::string &component) {
	return DebugStream(this, LogLevel::INFO, component);
}

DebugStream Debugger::warning_stream(const std::string &component) {
	return DebugStream(this, LogLevel::WARNING, component);
}

DebugStream Debugger::error_stream(const std::string &component) {
	return DebugStream(this, LogLevel::ERROR, component);
}

DebugStream Debugger::critical_stream(const std::string &component) {
	return DebugStream(this, LogLevel::CRITICAL, component);
}

// Specialized MPC logging
void Debugger::logMPCState(double x, double y, double yaw, double vel, double cte, double epsi) {
	std::ostringstream msg;
	msg << "STATE x=" << std::fixed << std::setprecision(3) << x << " y=" << y << " yaw=" << yaw
	    << " vel=" << vel << " cte=" << cte << " epsi=" << epsi;
	logMPC(LogLevel::DEBUG, msg.str());
}

void Debugger::logMPCControls(double throttle, double steering, double cost) {
	std::ostringstream msg;
	msg << "CONTROLS throttle=" << std::fixed << std::setprecision(3) << throttle
	    << " steering=" << steering;
	if(cost >= 0) {
		msg << " cost=" << cost;
	}
	logMPC(LogLevel::INFO, msg.str());
}

void Debugger::logMPCTrajectory(const std::vector<std::pair<double, double>> &trajectory) {
	std::ostringstream msg;
	msg << "TRAJECTORY " << trajectory.size() << " points: ";
	for(size_t i = 0; i < std::min(size_t(5), trajectory.size()); ++i) {
		msg << "(" << std::fixed << std::setprecision(2) << trajectory[i].first << ","
		    << trajectory[i].second << ") ";
	}
	if(trajectory.size() > 5) {
		msg << "...";
	}
	logMPC(LogLevel::DEBUG, msg.str());
}

void Debugger::logMPCReferences(const std::vector<std::pair<double, double>> &references) {
	std::ostringstream msg;
	msg << "REFERENCES " << references.size() << " points: ";
	for(size_t i = 0; i < std::min(size_t(3), references.size()); ++i) {
		msg << "(" << std::fixed << std::setprecision(2) << references[i].first << ","
		    << references[i].second << ") ";
	}
	if(references.size() > 3) {
		msg << "...";
	}
	logMPC(LogLevel::DEBUG, msg.str());
}

// Specialized Vision logging
void Debugger::logVisionFrame(int frame_count, double fps, int detected_objects) {
	std::ostringstream msg;
	msg << "FRAME " << frame_count << " fps=" << std::fixed << std::setprecision(1) << fps;
	if(detected_objects >= 0) {
		msg << " objects=" << detected_objects;
	}
	logVision(LogLevel::DEBUG, msg.str());
}

void Debugger::logLaneDetection(int lane_points, double curvature) {
	std::ostringstream msg;
	msg << "LANE_DETECTION points=" << lane_points;
	if(curvature > -999.0) {
		msg << " curvature=" << std::fixed << std::setprecision(4) << curvature;
	}
	logVision(LogLevel::INFO, msg.str());
}

// Specialized Control logging
void Debugger::logControlCommand(double throttle_pct, double steering_angle,
                                 const std::string &mode) {
	std::ostringstream msg;
	msg << "COMMAND throttle=" << std::fixed << std::setprecision(1) << throttle_pct
	    << "% steering=" << std::setprecision(1) << steering_angle << "° mode=" << mode;
	logControl(LogLevel::INFO, msg.str());
}

void Debugger::logServoProtection(const std::string &reason, int angle_requested,
                                  int angle_applied) {
	std::ostringstream msg;
	msg << "SERVO_PROTECTION " << reason << " requested=" << angle_requested
	    << "° applied=" << angle_applied << "°";
	logControl(LogLevel::WARNING, msg.str());
}

// Performance monitoring
void Debugger::logPerformance(const std::string &component, double execution_time_ms) {
	std::ostringstream msg;
	msg << "PERFORMANCE " << component << " took " << std::fixed << std::setprecision(2)
	    << execution_time_ms << "ms";
	log(LogLevel::DEBUG, "Performance", msg.str());
}
