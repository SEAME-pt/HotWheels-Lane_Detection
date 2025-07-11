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
#include <QDebug>
#include <fcntl.h>
#include <sys/mman.h>
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
ControlsManager::ControlsManager (int argc, char **argv, QObject *parent)
    : QObject (parent), m_engineController (0x40, 0x60, this), m_manualController (nullptr),
      m_currentMode (DrivingMode::Manual), m_subscriberJoystickObject (nullptr),
      m_cameraStreamerObject (nullptr), m_autonomousModeObject (nullptr),
      m_manualControllerThread (nullptr), m_joystickControlThread (nullptr),
      m_subscriberJoystickThread (nullptr), m_cameraStreamerThread (nullptr), m_running (true) {

	// Initialize the autonomous mode object
	initializeAutonomousMode ();

	// Initialize the joystick controller with callbacks
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
		qDebug () << "Failed to initialize joystick controller.";
		return;
	}

	// Start the joystick controller in its own thread
	m_manualControllerThread = new QThread (this);
	m_manualController->moveToThread (m_manualControllerThread);

	connect (m_manualControllerThread, &QThread::started, m_manualController,
	         &JoysticksController::processInput);
	connect (m_manualController, &JoysticksController::finished, m_manualControllerThread,
	         &QThread::quit);

	m_manualControllerThread->start ();

	// **Running camera streamer**
	m_cameraStreamerThread = QThread::create ([this, argc, argv] () {
		try {
			m_cameraStreamerObject = new CameraStreamer (0.5);

			// Setup direct MPC callback to AutonomousMode
			setupDirectMPCCallback ();

			m_cameraStreamerObject->start ();
		} catch (const std::exception &e) {
			std::cerr << "Error: " << e.what () << std::endl;
		}
	});
	m_cameraStreamerThread->start ();

	// **Client Middleware Interface Thread**
	m_subscriberJoystickObject = new Subscriber ();
	m_subscriberJoystickThread = QThread::create ([this, argc, argv] () {
		m_subscriberJoystickObject->connect ("tcp://localhost:5555");
		m_subscriberJoystickObject->subscribe ("joystick_value");
		while (m_running) {
			try {
				zmq::pollitem_t items[] = {
				    {static_cast<void *> (m_subscriberJoystickObject->getSocket ()), 0, ZMQ_POLLIN,
				     0}};

				// Wait up to 100ms for a message
				zmq::poll (items, 1, 100);

				if (items[0].revents & ZMQ_POLLIN) {
					zmq::message_t message;
					if (!m_subscriberJoystickObject->getSocket ().recv (&message, 0)) {
						continue; // failed to receive
					}

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
			} catch (const zmq::error_t &e) {
				std::cerr << "[Subscriber] ZMQ error: " << e.what () << std::endl;
				break; // exit safely if socket is closed
			}
		}
	});
	m_subscriberJoystickThread->start ();
}

/*!
 * @brief Destructor for the ControlsManager class.
 * @details Safely stops and cleans up all threads and resources associated
 *          with the ControlsManager. This includes stopping the client,
 *          shared memory, process monitoring, joystick control, and manual
 *          controller threads. It also deletes associated objects such as
 *          m_carDataObject, m_subscriberJoystickThread, and m_manualController.
 */

ControlsManager::~ControlsManager () {
	m_running = false;

	// Stop autonomous mode first
	if (m_autonomousModeObject) {
		m_autonomousModeObject->stopAutonomousControl ();
	}

	// Stop the client thread safely
	if (m_subscriberJoystickThread) {
		if (m_subscriberJoystickObject) {
			m_subscriberJoystickObject->stop ();
		}
		m_subscriberJoystickThread->quit ();
		m_subscriberJoystickThread->wait ();

		m_subscriberJoystickObject->getSocket ().close ();

		delete m_subscriberJoystickThread;
		m_subscriberJoystickThread = nullptr;
	}

	// Stop manual controller thread
	if (m_manualControllerThread) {
		if (m_manualController) m_manualController->requestStop ();

		m_manualControllerThread->quit ();
		m_manualControllerThread->wait ();
		delete m_manualControllerThread;
		m_manualControllerThread = nullptr;
	}

	// Stop camera streamer thread
	if (m_cameraStreamerThread) {
		if (m_cameraStreamerObject) m_cameraStreamerObject->stop ();

		m_cameraStreamerThread->quit ();
		m_cameraStreamerThread->wait ();
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

	delete m_autonomousModeObject;
	m_autonomousModeObject = nullptr;
}

/*!
 * @brief Sets the driving mode.
 * @param mode The new driving mode.
 * @details Updates the current driving mode if it has changed.
 */
void ControlsManager::setMode (DrivingMode mode) {
	if (m_currentMode == mode) return;

	m_currentMode = mode;

	// Delegate autonomous operations to AutonomousMode
	if (m_autonomousModeObject) {
		if (m_currentMode == DrivingMode::Automatic) {
			m_autonomousModeObject->startAutonomousControl ();
		} else {
			m_autonomousModeObject->stopAutonomousControl ();
			// Stop motors when switching to manual
			m_engineController.set_speed (0);
			m_engineController.set_steering (0);
		}
	}
}

void ControlsManager::initializeAutonomousMode () {
	// Create AutonomousMode object with engine controller reference
	m_autonomousModeObject = new AutonomousMode (&m_engineController, this);

	qDebug () << "AutonomousMode initialized";
}

void ControlsManager::setupDirectMPCCallback () {
	// Setup direct MPC callback if autonomous mode is configured for direct flow
	if (m_autonomousModeObject && m_autonomousModeObject->isUsingDirectFlow ()) {
		m_cameraStreamerObject->setMPCCallback ([this] (const LaneInfo &lane_info) {
			m_autonomousModeObject->receiveLaneDataDirect (lane_info);
		});
		qDebug () << "Direct MPC flow enabled";
	}
}
