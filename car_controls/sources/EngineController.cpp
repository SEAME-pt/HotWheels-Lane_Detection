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
#include "PeripheralController.hpp"
#include <QDebug>
#include <atomic>
#include <cmath>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>

/*!
 * @brief Clamps a value to a given range.
 *
 * @param value Value to be clamped.
 * @param min_val Minimum value of the range.
 * @param max_val Maximum value of the range.
 * @return The clamped value, or the original value if it is within the range.
 */
template <typename T> T clamp(T value, T min_val, T max_val) {
	return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

/*!
	* @brief Default constructor for the EngineController class.
	*/
EngineController::EngineController() {}

/*!
 * @brief Constructs an EngineController object, initializing motor and servo controllers.
 * @param servo_addr The address of the servo controller.
 * @param motor_addr The address of the motor controller.
 * @param parent The parent QObject for this instance.
 * @details Sets up the PeripheralController and initializes the servo and motor controllers.
 */
EngineController::EngineController(int servo_addr, int motor_addr,
																	 QObject *parent)
		: QObject(parent), m_running(false), m_current_speed(0),
			m_current_angle(0) {
	pcontrol = new PeripheralController(servo_addr, motor_addr);

	pcontrol->init_servo();
	pcontrol->init_motors();
}

/*!
 * @brief Destructor for the EngineController class.
 *
 * @details Stops the engine and deletes the peripheral controller.
 */
EngineController::~EngineController() {
	stop();
	delete pcontrol;
}

/*!
 * @brief Starts the engine.
 *
 * @details Sets the m_running flag to true.
 */
void EngineController::start() { m_running = true; }

/*!
 * @brief Stops the engine.
 *
 * @details Sets the m_running flag to false and sets both speed and steering to 0.
 */
void EngineController::stop() {
	m_running = false;
	set_speed(0);
	set_steering(0);
}

	/*!
	 * @brief Sets the direction of the car and emits the directionUpdated signal if the
	 * direction has changed.
	 *
	 * @param newDirection The new direction to set.
	 */
void EngineController::setDirection(CarDirection newDirection) {
	if (newDirection != this->m_currentDirection) {
		emit this->directionUpdated(newDirection);
		this->m_currentDirection = newDirection;
	}
}

/*!
 * @brief Sets the speed of the car.
 *
 * @param speed The desired speed value, ranging from -100 to 100.
 *
 * @details This function adjusts the motor PWM signals based on the input speed value.
 * Positive values set the car to move in reverse due to joystick reversal, while negative
 * values move it forward. A speed of zero stops the car. The function also updates the
 * car's direction accordingly and clamps the speed to ensure it is within the valid range.
 */

void EngineController::set_speed(int speed) {

	speed = clamp(speed, -100, 100);
	int pwm_value = static_cast<int>(std::abs(speed) / 100.0 * 4096);

	if (speed <
			0) { // Forward
		pcontrol->set_motor_pwm(0, pwm_value);
		pcontrol->set_motor_pwm(1, 0);
		pcontrol->set_motor_pwm(2, pwm_value);
		pcontrol->set_motor_pwm(5, pwm_value);
		pcontrol->set_motor_pwm(6, 0);
		pcontrol->set_motor_pwm(7, pwm_value);
		setDirection(CarDirection::Reverse);
	} else if (speed > 0) { // Backwards
		pcontrol->set_motor_pwm(0, pwm_value);
		pcontrol->set_motor_pwm(1, pwm_value);
		pcontrol->set_motor_pwm(2, 0);
		pcontrol->set_motor_pwm(5, 0);
		pcontrol->set_motor_pwm(6, pwm_value);
		pcontrol->set_motor_pwm(7, pwm_value);
		setDirection(CarDirection::Drive);
	} else { // Stop
		for (int channel = 0; channel < 9; ++channel)
			pcontrol->set_motor_pwm(channel, 0);
		setDirection(CarDirection::Stop);
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
void EngineController::set_steering(int angle) {
	angle = clamp(angle, -MAX_ANGLE, MAX_ANGLE);
	int pwm = 0;
	if (angle < 0) {
		pwm = SERVO_CENTER_PWM +
					static_cast<int>((angle / static_cast<float>(MAX_ANGLE)) *
													 (SERVO_CENTER_PWM - SERVO_LEFT_PWM));
	} else if (angle > 0) {
		pwm = SERVO_CENTER_PWM +
					static_cast<int>((angle / static_cast<float>(MAX_ANGLE)) *
													 (SERVO_RIGHT_PWM - SERVO_CENTER_PWM));
	} else {
		pwm = SERVO_CENTER_PWM;
	}

	pcontrol->set_servo_pwm(STEERING_CHANNEL, 0, pwm);
	m_current_angle = angle;
	emit this->steeringUpdated(angle);
}
