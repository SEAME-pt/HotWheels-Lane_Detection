/*!
 * @file EngineController.cpp
 * @brief Implementation of the EngineController class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the EngineController class,
 * which is responsible for controlling the car's engine and steering.
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "EngineController.hpp"
#include "Debugger.hpp"
#include "PeripheralController.hpp"
#include <QDebug>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <thread>
#include <unistd.h>

/*!
 * @brief Clamps a value to a given range.
 *
 * @param value Value to be clamped.
 * @param min_val Minimum value of the range.
 * @param max_val Maximum value of the range.
 * @return The clamped value, or the original value if it is within the range.
 */
template <typename T> T clamp (T value, T min_val, T max_val) {
template <typename T> T clamp (T value, T min_val, T max_val) {
	return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

/*!
 * @brief Default constructor for the EngineController class.
 */
EngineController::EngineController () {}
EngineController::EngineController () {}

/*!
 * @brief Constructs an EngineController object, initializing motor and servo
 * controllers.
 * @param servo_addr The address of the servo controller.
 * @param motor_addr The address of the motor controller.
 * @param parent The parent QObject for this instance.
 * @details Sets up the PeripheralController and initializes the servo and motor
 * controllers.
 */
EngineController::EngineController (int servo_addr, int motor_addr, QObject *parent)
    : QObject (parent), m_running (false), m_current_speed (0), m_current_angle (0) {
	pcontrol = new PeripheralController (servo_addr, motor_addr);
EngineController::EngineController (int servo_addr, int motor_addr, QObject *parent)
    : QObject (parent), m_running (false), m_current_speed (0), m_current_angle (0) {
	pcontrol = new PeripheralController (servo_addr, motor_addr);

	pcontrol->init_servo ();
	pcontrol->init_motors ();

	// Run servo initialization test to verify command transmission
	std::cout << "[EngineController] Initializing servo and motor controllers..." << std::endl;
	testServoInitialization ();
}

/*!
 * @brief Destructor for the EngineController class.
 *
 * @details Stops the engine and deletes the peripheral controller.
 */
EngineController::~EngineController () {
	stop ();
EngineController::~EngineController () {
	stop ();
	delete pcontrol;
}

/*!
 * @brief Starts the engine.
 *
 * @details Sets the m_running flag to true.
 */
void EngineController::start () {
void EngineController::start () {
	m_running = true;
}

/*!
 * @brief Stops the engine.
 *
 * @details Sets the m_running flag to false and sets both speed and steering to
 * 0.
 */
void EngineController::stop () {
void EngineController::stop () {
	m_running = false;
	set_speed (0);
	set_steering (0);
	set_speed (0);
	set_steering (0);
}

/*!
 * @brief Sets the direction of the car and emits the directionUpdated signal if the
 * direction has changed.
 *
 * @param newDirection The new direction to set.
 */
void EngineController::setDirection (CarDirection newDirection) {
	if (newDirection != this->m_currentDirection) {
		emit this->directionUpdated (newDirection);
void EngineController::setDirection (CarDirection newDirection) {
	if (newDirection != this->m_currentDirection) {
		emit this->directionUpdated (newDirection);
		this->m_currentDirection = newDirection;
	}
}

/*!
 * @brief Sets the speed of the car.
 *
 * @param speed The desired speed value, ranging from -100 to 100.
 *
 * @details This function adjusts the motor PWM signals based on the input speed
 * value. Positive values set the car to move in reverse due to joystick
 * reversal, while negative values move it forward. A speed of zero stops the
 * car. The function also updates the car's direction accordingly and clamps the
 * speed to ensure it is within the valid range.
 */

void EngineController::set_speed (int speed) {
void EngineController::set_speed (int speed) {
	// MOTOR: Funcionamento normal - motores são robustos e aceitam inputs extremos
	speed = clamp (speed, -100, 100);
	int pwm_value = static_cast<int> (std::abs (speed) / 100.0 * 4096);
	speed = clamp (speed, -100, 100);
	int pwm_value = static_cast<int> (std::abs (speed) / 100.0 * 4096);

	if (speed < 0) { // Reverse (negative speed)
		pcontrol->set_motor_pwm (0, pwm_value);
		pcontrol->set_motor_pwm (1, 0);
		pcontrol->set_motor_pwm (2, pwm_value);
		pcontrol->set_motor_pwm (5, pwm_value);
		pcontrol->set_motor_pwm (6, 0);
		pcontrol->set_motor_pwm (7, pwm_value);
		setDirection (CarDirection::Reverse);
	} else if (speed > 0) { // Forward (positive speed)
		pcontrol->set_motor_pwm (0, pwm_value);
		pcontrol->set_motor_pwm (1, pwm_value);
		pcontrol->set_motor_pwm (2, 0);
		pcontrol->set_motor_pwm (5, 0);
		pcontrol->set_motor_pwm (6, pwm_value);
		pcontrol->set_motor_pwm (7, pwm_value);
		setDirection (CarDirection::Drive);
	if (speed < 0) { // Reverse (negative speed)
		pcontrol->set_motor_pwm (0, pwm_value);
		pcontrol->set_motor_pwm (1, 0);
		pcontrol->set_motor_pwm (2, pwm_value);
		pcontrol->set_motor_pwm (5, pwm_value);
		pcontrol->set_motor_pwm (6, 0);
		pcontrol->set_motor_pwm (7, pwm_value);
		setDirection (CarDirection::Reverse);
	} else if (speed > 0) { // Forward (positive speed)
		pcontrol->set_motor_pwm (0, pwm_value);
		pcontrol->set_motor_pwm (1, pwm_value);
		pcontrol->set_motor_pwm (2, 0);
		pcontrol->set_motor_pwm (5, 0);
		pcontrol->set_motor_pwm (6, pwm_value);
		pcontrol->set_motor_pwm (7, pwm_value);
		setDirection (CarDirection::Drive);
	} else { // Stop - CRITICAL SAFETY: Force all motor channels to 0
		// Force ALL motor channels to 0 for safety
		for (int channel = 0; channel <= 15; ++channel) { // Expanded range for safety
			pcontrol->set_motor_pwm (channel, 0);
		for (int channel = 0; channel <= 15; ++channel) { // Expanded range for safety
			pcontrol->set_motor_pwm (channel, 0);
		}
		setDirection (CarDirection::Stop);
		setDirection (CarDirection::Stop);
	}
	m_current_speed = speed;
}

/*!
 * @brief Sets the steering angle of the car.
 *
 * @param angle The desired steering angle in degrees, ranging from -MAX_ANGLE to MAX_ANGLE.
 *
 * @details This function adjusts the servo PWM signal based on the input angle value.
 * The function clamps the angle to ensure it is within the valid range and calculates the
 * corresponding PWM value. The function also updates the internal steering angle and emits
 * the steeringUpdated signal.
 */
void EngineController::set_steering (int angle) {
	// === MPC FULL CONTROL MODE: Teste confirmou servo responsivo ===
	// Servo initialization test confirmed proper command transmission
	// MPC now has full control with safety switch as backup
	static int last_angle = 0;
	static auto last_servo_time = std::chrono::steady_clock::now ();
	static auto last_servo_time = std::chrono::steady_clock::now ();

	// Rate limiting temporal: mínimo 70ms entre comandos do servo (otimizado para MPC)
	auto now = std::chrono::steady_clock::now ();
	auto now = std::chrono::steady_clock::now ();
	auto elapsed =
	    std::chrono::duration_cast<std::chrono::milliseconds> (now - last_servo_time).count ();
	if (elapsed < 70) {
		// Comando muito rápido para o servo - manter sincronização com MPC
		INFO_STREAM ("SERVO_TIMING")
		    << "MPC timing: " << elapsed << "ms (min: 70ms) - comando postponed";
		return; // Manter sincronização temporal
	}

	// Clamp para limites físicos do hardware (±45° conforme especificações Jetracer)
	const int HARDWARE_MAX_ANGLE = 45; // Limite físico real do Jetracer
	angle = clamp (angle, -HARDWARE_MAX_ANGLE, HARDWARE_MAX_ANGLE);

	// Calculate PWM using full range for maximum steering authority
	int pwm = 0;
	if (angle < 0) {
		pwm =
		    SERVO_CENTER_PWM + static_cast<int> ((angle / static_cast<float> (HARDWARE_MAX_ANGLE)) *
		                                         (SERVO_CENTER_PWM - SERVO_LEFT_PWM));
	} else if (angle > 0) {
		pwm =
		    SERVO_CENTER_PWM + static_cast<int> ((angle / static_cast<float> (HARDWARE_MAX_ANGLE)) *
		                                         (SERVO_RIGHT_PWM - SERVO_CENTER_PWM));
	} else {
		pwm = SERVO_CENTER_PWM;
	}

	pcontrol->set_servo_pwm (STEERING_CHANNEL, 0, pwm);
	pcontrol->set_servo_pwm (STEERING_CHANNEL, 0, pwm);
	m_current_angle = angle;
	last_angle = angle;
	last_servo_time = now;

	// Enhanced logging for MPC steering commands
	INFO_STREAM ("MPC_SERVO") << "MPC Steering: " << angle << "° (PWM: " << pwm
	                          << ") - Full range: ±" << HARDWARE_MAX_ANGLE << "°";

	emit this->steeringUpdated (angle);
	emit this->steeringUpdated (angle);
}

/*!
 * @brief Forced motor stop - bypasses all normal logic for emergency situations
 * @details This method forces all motor channels to zero without any checks or logic.
 * It's designed to be called in emergency situations where normal motor control may fail.
 */
void EngineController::forcedMotorStop () {
void EngineController::forcedMotorStop () {

	try {
		// Force ALL possible motor channels to zero - no exceptions
		for (int channel = 0; channel <= 15; ++channel) {
			pcontrol->set_motor_pwm (channel, 0);
		for (int channel = 0; channel <= 15; ++channel) {
			pcontrol->set_motor_pwm (channel, 0);
		}

		// Double-verify critical channels are zero
		pcontrol->set_motor_pwm (0, 0);
		pcontrol->set_motor_pwm (1, 0);
		pcontrol->set_motor_pwm (2, 0);
		pcontrol->set_motor_pwm (5, 0);
		pcontrol->set_motor_pwm (6, 0);
		pcontrol->set_motor_pwm (7, 0);
		pcontrol->set_motor_pwm (0, 0);
		pcontrol->set_motor_pwm (1, 0);
		pcontrol->set_motor_pwm (2, 0);
		pcontrol->set_motor_pwm (5, 0);
		pcontrol->set_motor_pwm (6, 0);
		pcontrol->set_motor_pwm (7, 0);

		m_current_speed = 0;
		setDirection (CarDirection::Stop);
		setDirection (CarDirection::Stop);

	} catch (...) {
	} catch (...) {
	}
}

/*!
 * @brief Emergency hardware stop with multiple redundant calls
 * @details Makes multiple redundant calls to ensure motors are stopped.
 * This is the most robust stop method available.
 */
void EngineController::emergencyHardwareStop () {
void EngineController::emergencyHardwareStop () {

	// Call 1: Normal stop
	try {
		set_speed (0);
	} catch (...) {
		set_speed (0);
	} catch (...) {
	}

	// Small delay for hardware to process
	std::this_thread::sleep_for (std::chrono::milliseconds (5));
	std::this_thread::sleep_for (std::chrono::milliseconds (5));

	// Call 2: Forced stop
	try {
		forcedMotorStop ();
	} catch (...) {
		forcedMotorStop ();
	} catch (...) {
	}

	// Small delay for hardware to process
	std::this_thread::sleep_for (std::chrono::milliseconds (5));
	std::this_thread::sleep_for (std::chrono::milliseconds (5));

	// Call 3: Final redundant stop
	try {
		for (int i = 0; i < 3; ++i) {
			for (int channel = 0; channel <= 15; ++channel) {
				pcontrol->set_motor_pwm (channel, 0);
		for (int i = 0; i < 3; ++i) {
			for (int channel = 0; channel <= 15; ++channel) {
				pcontrol->set_motor_pwm (channel, 0);
			}
			std::this_thread::sleep_for (std::chrono::milliseconds (2));
			std::this_thread::sleep_for (std::chrono::milliseconds (2));
		}
	} catch (...) {
	} catch (...) {
	}

	m_current_speed = 0;
	setDirection (CarDirection::Stop);
}

/*!
 * @brief Tests servo movement during initialization to verify command transmission
 * @details Moves servo left, right, and back to center with visual feedback
 * This bypasses normal protection to ensure the test works during initialization
 */
void EngineController::testServoInitialization () {
	std::cout << "\n=== SERVO INITIALIZATION TEST ===" << std::endl;
	std::cout << "Testing servo movement to verify command transmission..." << std::endl;

	// Bypass normal protection for initialization test
	static auto init_test_time = std::chrono::steady_clock::now ();
	const int TEST_ANGLE = 45; // MAXIMUM hardware limit test angle (±45° as per Jetracer specs)

	try {
		// Step 1: Move to left
		std::cout << "[SERVO TEST] Moving to LEFT (" << -TEST_ANGLE << "°)..." << std::endl;
		int left_pwm =
		    SERVO_CENTER_PWM + static_cast<int> ((-TEST_ANGLE / static_cast<float> (MAX_ANGLE)) *
		                                         (SERVO_CENTER_PWM - SERVO_LEFT_PWM));
		std::cout << "[SERVO TEST] LEFT calculation: PWM=" << left_pwm << " (should be near "
		          << SERVO_LEFT_PWM << ")" << std::endl;
		pcontrol->set_servo_pwm (STEERING_CHANNEL, 0, left_pwm);
		m_current_angle = -TEST_ANGLE;
		std::this_thread::sleep_for (std::chrono::milliseconds (2000)); // Extra time for visibility

		// Step 2: Move to right
		std::cout << "[SERVO TEST] Moving to RIGHT (" << TEST_ANGLE << "°)..." << std::endl;
		int right_pwm =
		    SERVO_CENTER_PWM + static_cast<int> ((TEST_ANGLE / static_cast<float> (MAX_ANGLE)) *
		                                         (SERVO_RIGHT_PWM - SERVO_CENTER_PWM));
		std::cout << "[SERVO TEST] RIGHT calculation: PWM=" << right_pwm << " (should be near "
		          << SERVO_RIGHT_PWM << ")" << std::endl;
		pcontrol->set_servo_pwm (STEERING_CHANNEL, 0, right_pwm);
		m_current_angle = TEST_ANGLE;
		std::this_thread::sleep_for (std::chrono::milliseconds (2000)); // Extra time for visibility

		// Step 3: Return to center
		std::cout << "[SERVO TEST] Returning to CENTER (0°)..." << std::endl;
		pcontrol->set_servo_pwm (STEERING_CHANNEL, 0, SERVO_CENTER_PWM);
		m_current_angle = 0;
		std::this_thread::sleep_for (std::chrono::milliseconds (1000)); // Wait for center

		std::cout << "[SERVO TEST] ✅ Test completed successfully!" << std::endl;
		std::cout << "[SERVO TEST] PWM Values - Left: " << left_pwm << ", Right: " << right_pwm
		          << ", Center: " << SERVO_CENTER_PWM << std::endl;
		std::cout << "[SERVO TEST] PWM Limits - Left=" << SERVO_LEFT_PWM
		          << ", Center=" << SERVO_CENTER_PWM << ", Right=" << SERVO_RIGHT_PWM << std::endl;
		std::cout << "[SERVO TEST] Angle Range - Test used ±" << TEST_ANGLE << "° out of ±"
		          << MAX_ANGLE << "° max" << std::endl;

		// Show percentage of servo range used
		float left_usage = (abs (left_pwm - SERVO_CENTER_PWM) /
		                    static_cast<float> (SERVO_CENTER_PWM - SERVO_LEFT_PWM)) *
		                   100.0f;
		float right_usage = (abs (right_pwm - SERVO_CENTER_PWM) /
		                     static_cast<float> (SERVO_RIGHT_PWM - SERVO_CENTER_PWM)) *
		                    100.0f;
		std::cout << "[SERVO TEST] Phase 1 servo range usage - Left: " << left_usage
		          << "%, Right: " << right_usage << "%" << std::endl;

		// PHASE 2: Test using DIRECT extreme PWM values (maximum physical movement)
		std::cout << "\n[SERVO TEST] === PHASE 2: Testing with DIRECT extreme PWM values ==="
		          << std::endl;
		std::cout << "[SERVO TEST] This should show MAXIMUM possible servo movement!" << std::endl;

		// Step 1: Move to ABSOLUTE left limit
		std::cout << "[SERVO TEST] Moving to ABSOLUTE LEFT limit..." << std::endl;
		std::cout << "[SERVO TEST] Using SERVO_LEFT_PWM directly: " << SERVO_LEFT_PWM << std::endl;
		pcontrol->set_servo_pwm (STEERING_CHANNEL, 0, SERVO_LEFT_PWM);
		std::this_thread::sleep_for (
		    std::chrono::milliseconds (2500)); // Extra time for max visibility

		// Step 2: Move to ABSOLUTE right limit
		std::cout << "[SERVO TEST] Moving to ABSOLUTE RIGHT limit..." << std::endl;
		std::cout << "[SERVO TEST] Using SERVO_RIGHT_PWM directly: " << SERVO_RIGHT_PWM
		          << std::endl;
		pcontrol->set_servo_pwm (STEERING_CHANNEL, 0, SERVO_RIGHT_PWM);
		std::this_thread::sleep_for (
		    std::chrono::milliseconds (2500)); // Extra time for max visibility

		// Step 3: Return to center
		std::cout << "[SERVO TEST] Returning to CENTER..." << std::endl;
		pcontrol->set_servo_pwm (STEERING_CHANNEL, 0, SERVO_CENTER_PWM);
		m_current_angle = 0;
		std::this_thread::sleep_for (std::chrono::milliseconds (1000));

		std::cout << "[SERVO TEST] PWM Range - Total range: " << (SERVO_RIGHT_PWM - SERVO_LEFT_PWM)
		          << " PWM units" << std::endl;

	} catch (const std::exception &e) {
		std::cout << "[SERVO TEST] ❌ ERROR: " << e.what () << std::endl;
		// Ensure servo returns to center on error
		pcontrol->set_servo_pwm (STEERING_CHANNEL, 0, SERVO_CENTER_PWM);
		m_current_angle = 0;
	}

	std::cout << "=================================\n" << std::endl;
}
