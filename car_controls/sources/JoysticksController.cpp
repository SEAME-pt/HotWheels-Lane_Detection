/*!
 * @file JoysticksController.cpp
 * @brief Implementation of the JoysticksController class.
 * @version 0.1
 * @date 2025-02-12
 * @details This file contains the implementation of the JoysticksController class,
 * which is responsible for handling joystick input and updating the steering and speed
 * of the car accordingly.
 * 
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "JoysticksController.hpp"
#include <QDebug>
#include <QThread>


/*!
 * @brief Construct a JoysticksController object.
 * @details This constructor takes a QObject parent and two std::function callbacks.
 * The first callback is called when the steering of the joystick is updated, and
 * the second one is called when the speed of the joystick is updated.
 * @param steeringCallback The callback to be called when the steering of the
 * joystick is updated.
 * @param speedCallback The callback to be called when the speed of the joystick
 * is updated.
 * @param parent The QObject parent of this JoysticksController object.
 */
JoysticksController::JoysticksController(
		std::function<void(int)> steeringCallback,
		std::function<void(int)> speedCallback, QObject *parent)
		: QObject(parent), m_joystick(nullptr),
			m_updateSteering(std::move(steeringCallback)),
			m_updateSpeed(std::move(speedCallback)), m_running(false) {}

/*!
 * @brief Destruct a JoysticksController object.
 * @details This destructor closes the SDL joystick and quits SDL if a joystick
 * was opened.
 */
JoysticksController::~JoysticksController() {
	if (m_joystick) {
		SDL_JoystickClose(m_joystick);
	}
	SDL_Quit();
}

/*!
 * @brief Initializes the joystick controller.
 * @details This function initializes the SDL joystick subsystem and opens the
 * first available joystick device. If SDL initialization fails, an error message
 * is logged and the function returns false.
 * @return True if the joystick is successfully initialized and opened, false otherwise.
 */
bool JoysticksController::init() {
	if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
		qDebug() << "Failed to initialize SDL:" << SDL_GetError();
		return false;
	}

	m_joystick = SDL_JoystickOpen(0);
	if (!m_joystick) {
		init();
	}

	return true;
}

/*!
 * @brief Requests the joystick controller to stop.
 * @details This function sets the running flag to false, which will stop the
 * joystick controller loop.
 */
void JoysticksController::requestStop() { m_running = false; }

/*!
 * @brief Runs the joystick controller loop.
 * @details This function is run in its own thread and waits for SDL events
 * from the joystick. If the joystick is not initialized, it logs an error
 * message and emits the finished() signal. The loop will stop when the
 * running flag is set to false (by calling requestStop()) or when the thread
 * is interrupted. The finished() signal is emitted when the loop finishes.
 * @returns if the joystick is not initialized.
 */
void JoysticksController::processInput() {
	m_running = true;

	if (!m_joystick) {
		qDebug() << "Joystick not initialized.";
		emit finished();
		return;
	}

	while (m_running && !QThread::currentThread()->isInterruptionRequested()) {
		SDL_Event e;
		while (SDL_PollEvent(&e)) {
			if (e.type == SDL_JOYAXISMOTION) {
				if (e.jaxis.axis == 0) {
					m_updateSteering(static_cast<int>(e.jaxis.value / 32767.0 * 180));
				} else if (e.jaxis.axis == 3) {
					m_updateSpeed(static_cast<int>(e.jaxis.value / 32767.0 * 100));
				}
			}
		}
		QThread::msleep(10);
	}

	// qDebug() << "Joystick controller loop finished.";
	emit finished();
}
