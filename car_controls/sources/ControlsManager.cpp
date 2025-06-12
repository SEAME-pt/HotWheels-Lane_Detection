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
#include <sstream>
#include <string>

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
	: QObject(parent), m_engineController(0x40, 0x60, this),
	  m_manualController(nullptr), m_currentMode(DrivingMode::Manual),
	  m_subscriberJoystickObject(nullptr), m_manualControllerThread(nullptr),
	  m_joystickControlThread(nullptr), m_subscriberJoystickThread(nullptr),
	  m_cameraStreamerThread(nullptr), m_running(true),
	  m_mpcPlanner(nullptr), m_autonomousMode(false), m_autonomousControlThread(nullptr)
{

	// Initialize the joystick controller with callbacks
	m_manualController = new JoysticksController(
		[this](int steering)
		{
			if (m_currentMode == DrivingMode::Manual)
			{
				m_engineController.set_steering(steering);
			}
		},
		[this](int speed)
		{
			if (m_currentMode == DrivingMode::Manual)
			{
				m_engineController.set_speed(speed);
			}
		});

	if (!m_manualController->init())
	{
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
	m_cameraStreamerThread = QThread::create([this, argc, argv]()
								{
		try {
			m_cameraStreamerObject = new CameraStreamer(0.5);
			m_cameraStreamerObject->start();
		} catch (const std::exception& e) {
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
	m_subscriberJoystickThread->start();
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
	// Stop the client thread safely
	if (m_subscriberJoystickThread) {
		if (m_subscriberJoystickObject) {
			m_subscriberJoystickObject->stop();
		}
		m_subscriberJoystickThread->quit();
		m_subscriberJoystickThread->wait();

		m_subscriberJoystickObject->getSocket().close();

		delete m_subscriberJoystickThread;
		m_subscriberJoystickThread = nullptr;
	}


	// Stop manual controller thread
	if (m_manualControllerThread) {
		if (m_manualController)
			m_manualController->requestStop();

		m_manualControllerThread->quit();
		m_manualControllerThread->wait();
		delete m_manualControllerThread;
		m_manualControllerThread = nullptr;
	}

	//Stop camera streamer thread
	if (m_cameraStreamerThread) {
		if (m_cameraStreamerObject)
			m_cameraStreamerObject->stop();

		m_cameraStreamerThread->quit();
		m_cameraStreamerThread->wait();
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
}

void ControlsManager::startAutonomousControl() {
    if (m_currentMode != DrivingMode::Automatic) return;
    
    m_autonomousMode = true;
    m_mpcPlanner = new MPCPlanner();
    
    m_autonomousControlThread = QThread::create([this]() {
        autonomousControlLoop();
    });
    m_autonomousControlThread->start();
}

void ControlsManager::autonomousControlLoop() {
    const double CONTROL_RATE = 20.0; // Hz
    const double CONTROL_PERIOD = 1.0 / CONTROL_RATE;
    
    auto last_control_time = std::chrono::steady_clock::now();
    
    while (m_autonomousMode && m_running) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - last_control_time).count();
        
        if (elapsed < CONTROL_PERIOD) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        last_control_time = now;
        
        try {
            // 1. Obter dados de percepção via ZeroMQ
            VehicleState current_state = getCurrentVehicleState();
            std::vector<Point2D> waypoints = getWaypointsFromVision();
            LaneInfo lane_info = getLaneInfoFromVision();
            
            // 2. Verificar obstáculos críticos
            if (checkEmergencyObstacles()) {
                m_engineController.set_speed(0);
                continue;
            }
            
            // 3. Calcular controle MPC
            ControlCommand control = m_mpcPlanner->plan(current_state, waypoints, &lane_info);
            
            // 4. Aplicar controles com limites de segurança
            int throttle_pct = static_cast<int>(std::clamp(control.throttle * 100, 0.0, 50.0));
            int steer_angle = static_cast<int>(std::clamp(control.steer * 45, -45.0, 45.0));
            
            m_engineController.set_speed(throttle_pct);
            m_engineController.set_steering(steer_angle);
            
        } catch (const std::exception& e) {
            qDebug() << "Autonomous control error:" << e.what();
            m_engineController.set_speed(0);
        }
    }
}

/*!
 * @brief Stops the autonomous control mode safely.
 * @details This function safely stops the autonomous control thread and 
 * cleans up associated resources. It sets the autonomous mode flag to false,
 * waits for the thread to finish, and applies emergency braking.
 */
void ControlsManager::stopAutonomousControl() {
    if (!m_autonomousMode) return; // Already stopped
    
    qDebug() << "Stopping autonomous control...";
    
    // Set flag to stop the autonomous loop
    m_autonomousMode = false;
    
    // Wait for autonomous thread to finish
    if (m_autonomousControlThread) {
        m_autonomousControlThread->quit();
        if (!m_autonomousControlThread->wait(2000)) { // Wait up to 2 seconds
            qDebug() << "Warning: Autonomous thread did not finish gracefully";
            m_autonomousControlThread->terminate();
            m_autonomousControlThread->wait(1000);
        }
        delete m_autonomousControlThread;
        m_autonomousControlThread = nullptr;
    }
    
    // Clean up MPC planner
    if (m_mpcPlanner) {
        delete m_mpcPlanner;
        m_mpcPlanner = nullptr;
    }
    
    // Apply emergency stop
    m_engineController.set_speed(0);
    m_engineController.set_steering(0);
    
    qDebug() << "Autonomous control stopped successfully";
}

// Adicionar ao ControlsManager
VehicleState ControlsManager::getCurrentVehicleState() {
    // Implementar odometria baseada nos encoders dos motores
    static VehicleState state{0.0, 0.0, 0.0, 0.0};
    
    // Atualizar baseado no histórico de comandos
    // Usar IMU se disponível
    return state;
}

std::vector<Point2D> ControlsManager::getWaypointsFromVision() {
    std::vector<Point2D> waypoints;
    
    // Subscriber para dados de segmentação
    Subscriber vision_sub;
    vision_sub.connect("tcp://localhost:5556");
    
    try {
        // Receber dados da linha central processada
        std::string lane_data = vision_sub.receive("centerline_waypoints");
        
        // Parse dos waypoints (formato: "x1,y1;x2,y2;...")
        std::istringstream ss(lane_data);
        std::string point;
        
        while (std::getline(ss, point, ';')) {
            size_t comma = point.find(',');
            if (comma != std::string::npos) {
                double x = std::stod(point.substr(0, comma));
                double y = std::stod(point.substr(comma + 1));
                waypoints.emplace_back(x, y);
            }
        }
        
    } catch (...) {
        // Fallback: waypoints retos à frente
        for (int i = 1; i <= 10; ++i) {
            waypoints.emplace_back(i * 2.0, 0.0);
        }
    }
    
    return waypoints;
}

bool ControlsManager::checkEmergencyObstacles() {
    Subscriber obstacle_sub;
    obstacle_sub.connect("tcp://localhost:5557");
    
    try {
        std::string obstacle_data = obstacle_sub.receive("emergency_stop");
        return (obstacle_data == "true");
    } catch (...) {
        return false;
    }
}
