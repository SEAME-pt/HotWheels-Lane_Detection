/**
 * @file Debugger.hpp
 * @author Michel Batista (michel_fab@outlook.com)
 * @brief Centralized debugging and logging system for the car controls project.
 * @version 0.1
 * @date 2025-06-27
 *
 * @copyright Copyright (c) 2025
 *
 */

#ifndef DEBUGGER_HPP
#define DEBUGGER_HPP

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief Enumeration for log levels.
 */
enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, CRITICAL = 4 };

/**
 * @brief Forward declaration of Debugger class.
 *
 * This is used to allow the DebugStream class to reference Debugger without
 * needing the full definition at this point.
 */
class Debugger;

/**
 * @brief Helper class for stream-like logging.
 *
 * This class allows for more natural C++-style logging using the stream operator
 * (<<) while still integrating with the Debugger's logging system.
 */
class DebugStream {
	private:
		Debugger *debugger;
		LogLevel level;
		std::string component;
		std::ostringstream stream;

	public:
		DebugStream(Debugger *dbg, LogLevel lvl, const std::string &comp);
		~DebugStream();

		// Template operator<< to handle any type that can be streamed
		template <typename T> DebugStream &operator<<(const T &value) {
			stream << value;
			return *this;
		}

		// Special handling for manipulators like std::endl
		DebugStream &operator<<(std::ostream &(*manip)(std::ostream &)) {
			stream << manip;
			return *this;
		}
};

/**
 * @brief Centralized debugging and logging system for the car controls project.
 *
 * This class provides a unified interface for logging messages from different
 * components of the system, including console output, file output, and custom
 * log levels.
 */
class Debugger {
	private:
		static std::unique_ptr<Debugger> instance;
		static std::mutex mutex_;

		std::ofstream debug_file;
		std::ofstream mpc_file;
		std::ofstream vision_file;
		std::ofstream control_file;

		bool console_output_enabled;
		bool file_output_enabled;
		LogLevel min_log_level;

		std::string session_id;
		std::string base_output_dir;

		// Private constructor for singleton
		Debugger();

		std::string getCurrentTimestamp() const;
		std::string getLogLevelString(LogLevel level) const;
		void writeToFile(std::ofstream &file, const std::string &message);

	public:
		// Singleton access
		static Debugger *getInstance();

		// Destructor
		~Debugger();

		// Configuration methods
		void enableConsoleOutput(bool enable = true) {
			console_output_enabled = enable;
		}
		void enableFileOutput(bool enable = true) {
			file_output_enabled = enable;
		}
		void setMinLogLevel(LogLevel level) {
			min_log_level = level;
		}

		// Main logging methods
		void log(LogLevel level, const std::string &component, const std::string &message);
		void logMPC(LogLevel level, const std::string &message);
		void logVision(LogLevel level, const std::string &message);
		void logControl(LogLevel level, const std::string &message);

		// Convenience methods
		void debug(const std::string &component, const std::string &message);
		void info(const std::string &component, const std::string &message);
		void warning(const std::string &component, const std::string &message);
		void error(const std::string &component, const std::string &message);
		void critical(const std::string &component, const std::string &message);

		// Specialized logging for MPC data
		void logMPCState(double x, double y, double yaw, double vel, double cte, double epsi);
		void logMPCControls(double throttle, double steering, double cost = -1.0);
		void logMPCTrajectory(const std::vector<std::pair<double, double>> &trajectory);
		void logMPCReferences(const std::vector<std::pair<double, double>> &references);

		// Specialized logging for Vision data
		void logVisionFrame(int frame_count, double fps, int detected_objects = -1);
		void logLaneDetection(int lane_points, double curvature = -999.0);

		// Specialized logging for Control data
		void logControlCommand(double throttle_pct, double steering_angle, const std::string &mode);
		void logServoProtection(const std::string &reason, int angle_requested, int angle_applied);

		// Performance monitoring
		void logPerformance(const std::string &component, double execution_time_ms);

		// Stream-style logging methods
		DebugStream debug_stream(const std::string &component);
		DebugStream info_stream(const std::string &component);
		DebugStream warning_stream(const std::string &component);
		DebugStream error_stream(const std::string &component);
		DebugStream critical_stream(const std::string &component);

		// Get file paths for external monitoring
		std::string getSessionId() const {
			return session_id;
		}
		std::string getDebugFilePath() const {
			return base_output_dir + "/debug_" + session_id + ".log";
		}
		std::string getMPCFilePath() const {
			return base_output_dir + "/mpc_" + session_id + ".log";
		}
		std::string getVisionFilePath() const {
			return base_output_dir + "/vision_" + session_id + ".log";
		}
		std::string getControlFilePath() const {
			return base_output_dir + "/control_" + session_id + ".log";
		}

		// Session management
		void startNewSession();
		void endSession();

		// Delete copy constructor and assignment operator
		Debugger(const Debugger &) = delete;
		Debugger &operator=(const Debugger &) = delete;
};

/**
 * @brief Convenience macros for easy usage.
 * This section defines macros for logging at different levels and for different components.
 * It also includes specialized macros for MPC, Vision, and Control subsystems
 */
#define DEBUG_LOG(component, message) Debugger::getInstance()->debug(component, message)
#define INFO_LOG(component, message) Debugger::getInstance()->info(component, message)
#define WARNING_LOG(component, message) Debugger::getInstance()->warning(component, message)
#define ERROR_LOG(component, message) Debugger::getInstance()->error(component, message)
#define CRITICAL_LOG(component, message) Debugger::getInstance()->critical(component, message)

// Stream-style macros for natural C++ logging
#define DEBUG_STREAM(component) Debugger::getInstance()->debug_stream(component)
#define INFO_STREAM(component) Debugger::getInstance()->info_stream(component)
#define WARNING_STREAM(component) Debugger::getInstance()->warning_stream(component)
#define ERROR_STREAM(component) Debugger::getInstance()->error_stream(component)
#define CRITICAL_STREAM(component) Debugger::getInstance()->critical_stream(component)

// MPC specialized macros
#define MPC_DEBUG(message) Debugger::getInstance()->logMPC(LogLevel::DEBUG, message)
#define MPC_INFO(message) Debugger::getInstance()->logMPC(LogLevel::INFO, message)
#define MPC_DEBUG_STREAM() Debugger::getInstance()->debug_stream("MPC")
#define MPC_INFO_STREAM() Debugger::getInstance()->info_stream("MPC")

// Vision specialized macros
#define VISION_DEBUG(message) Debugger::getInstance()->logVision(LogLevel::DEBUG, message)
#define VISION_INFO(message) Debugger::getInstance()->logVision(LogLevel::INFO, message)
#define VISION_DEBUG_STREAM() Debugger::getInstance()->debug_stream("VISION")
#define VISION_INFO_STREAM() Debugger::getInstance()->info_stream("VISION")

// Control specialized macros
#define CONTROL_DEBUG(message) Debugger::getInstance()->logControl(LogLevel::DEBUG, message)
#define CONTROL_INFO(message) Debugger::getInstance()->logControl(LogLevel::INFO, message)
#define CONTROL_DEBUG_STREAM() Debugger::getInstance()->debug_stream("CONTROL")
#define CONTROL_INFO_STREAM() Debugger::getInstance()->info_stream("CONTROL")

#endif // DEBUGGER_HPP
