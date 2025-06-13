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
	  m_mpcPlanner(nullptr), m_polyfitter(nullptr), 
	  m_autonomousMode(false), m_autonomousControlThread(nullptr)
{
	qDebug() << "[ControlsManager] Inicializado no modo MANUAL";

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
	m_polyfitter = new Polyfitter();
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

	delete m_polyfitter;
    m_polyfitter = nullptr;
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

    DrivingMode previous_mode = m_currentMode;
	m_currentMode = mode;

	if (mode == DrivingMode::Automatic) {
        qDebug() << "Switching to AUTONOMOUS mode";
        std::cout << "Starting autonomous control..." << std::endl;
        startAutonomousControl();
    } else {
        qDebug() << "Switching to MANUAL mode";
        stopAutonomousControl();
        
        // Parada suave se vinha do automático
        if (previous_mode == DrivingMode::Automatic) {
            m_engineController.set_speed(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
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

void ControlsManager::stopAutonomousControl() {
    if (!m_autonomousMode) return;
    
    qDebug() << "Stopping autonomous control...";
    
    m_autonomousMode = false;
    
    if (m_autonomousControlThread) {
        m_autonomousControlThread->quit();
        if (!m_autonomousControlThread->wait(2000)) {
            qDebug() << "Warning: Autonomous thread did not finish gracefully";
            m_autonomousControlThread->terminate();
            m_autonomousControlThread->wait(1000);
        }
        delete m_autonomousControlThread;
        m_autonomousControlThread = nullptr;
    }
    
    if (m_mpcPlanner) {
        delete m_mpcPlanner;
        m_mpcPlanner = nullptr;
    }
    
    m_engineController.set_speed(0);
    m_engineController.set_steering(0);
    
    qDebug() << "Autonomous control stopped successfully";
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
        std::cout << "Running autonomous control loop..." << std::endl;
        try {
            // Mostra feedback visual da visão
            showVisionDebug();
            // 1. Obter dados de percepção
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
            std::cout << "Throttle: " << throttle_pct << "%, Steering: " << steer_angle << " degrees" << std::endl;
            m_engineController.set_speed(throttle_pct);
            m_engineController.set_steering(steer_angle);
        } catch (const std::exception& e) {
            qDebug() << "Autonomous control error:" << e.what();
            m_engineController.set_speed(0);
        }
    }
}

// Adicionar ao ControlsManager
VehicleState ControlsManager::getCurrentVehicleState() {
    static VehicleState state{0.0, 0.0, 0.0, 0.0};
    
    // Implementação básica - pode ser melhorada com odometria real
    static auto last_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_time).count();
    last_time = now;
    
    // Simular movimento baseado nos controles aplicados
    state.x += state.velocity * std::cos(state.yaw) * dt;
    state.y += state.velocity * std::sin(state.yaw) * dt;
    state.velocity = 2.0; // Velocidade simulada
    
    return state;
}

std::vector<Point2D> ControlsManager::getWaypointsFromVision() {
    std::vector<Point2D> waypoints;

    Subscriber vision_sub;
    vision_sub.connect("tcp://localhost:5556");

    try {
        zmq::message_t topic_msg;
        zmq::message_t data_msg;
        vision_sub.getSocket().recv(&topic_msg);
        vision_sub.getSocket().recv(&data_msg);

        std::string topic(static_cast<char*>(topic_msg.data()), topic_msg.size());
        if (topic == "binary_mask") {
            std::string mask_data(static_cast<char*>(data_msg.data()), data_msg.size());
            cv::Mat binary_mask = deserializeMask(mask_data);

            // 1. Extraia as faixas
            auto lanes = m_polyfitter->fitLanesInImage(binary_mask);

            // 2. Calcule a centerline virtual
            CenterlineResult result = m_polyfitter->computeVirtualCenterline(lanes, binary_mask.cols, binary_mask.rows);

            // 3. Use result.blend como waypoints
            if (result.valid) {
                waypoints = result.blend;
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


LaneInfo ControlsManager::getLaneInfoFromVision() {
    Subscriber vision_sub;
    vision_sub.connect("tcp://localhost:5556");
    
    try {
        zmq::message_t topic_msg;
        zmq::message_t data_msg;
        vision_sub.getSocket().recv(&topic_msg);
        vision_sub.getSocket().recv(&data_msg);

        std::string topic(static_cast<char*>(topic_msg.data()), topic_msg.size());
        if (topic == "binary_mask") {
            std::string mask_data(static_cast<char*>(data_msg.data()), data_msg.size());
            cv::Mat binary_mask = deserializeMask(mask_data);
            // Use Polyfitter's fitLanesInImage and computeVirtualCenterline
            auto lanes = m_polyfitter->fitLanesInImage(binary_mask);
            auto centerline = m_polyfitter->computeVirtualCenterline(lanes, binary_mask.cols, binary_mask.rows);
            // You may want to extract LaneInfo from the centerline or lanes if needed
            // For now, just return a default LaneInfo if not implemented
            // TODO: Implement a method to extract LaneInfo from lanes/centerline if needed
            return LaneInfo(0.0, 0.0);
        }
    } catch (...) {
        return LaneInfo(0.0, 0.0);
    }
    return LaneInfo(0.0, 0.0); // fallback
}

bool ControlsManager::checkEmergencyObstacles() {
    Subscriber obstacle_sub;
    obstacle_sub.connect("tcp://localhost:5557");
    
    try {
        zmq::message_t topic_msg;
        zmq::message_t data_msg;
        obstacle_sub.getSocket().recv(&topic_msg); // CORRIGIDO: obstacle_sub
        obstacle_sub.getSocket().recv(&data_msg);
        
        std::string topic(static_cast<char*>(topic_msg.data()), topic_msg.size());
        if (topic == "emergency_stop") {
            std::string obstacle_data(static_cast<char*>(data_msg.data()), data_msg.size());
            return (obstacle_data == "true");
        }
    } catch (...) {
        return false;
    }
    return false; // Garante retorno em todos os paths
}

std::string ControlsManager::serializeMask(const cv::Mat& mask) {
    std::vector<uchar> buffer;
    cv::imencode(".png", mask, buffer);
    return std::string(buffer.begin(), buffer.end());
}

cv::Mat ControlsManager::deserializeMask(const std::string& data) {
    std::vector<uchar> buffer(data.begin(), data.end());
    return cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
}

void ControlsManager::showVisionDebug() {
    Subscriber vision_sub;
    vision_sub.connect("tcp://localhost:5556");
    try {
        zmq::message_t topic_msg, data_msg;
        vision_sub.getSocket().recv(&topic_msg);
        vision_sub.getSocket().recv(&data_msg);

        std::string topic(static_cast<char*>(topic_msg.data()), topic_msg.size());
        if (topic == "binary_mask") {
            std::string mask_data(static_cast<char*>(data_msg.data()), data_msg.size());
            cv::Mat binary_mask = deserializeMask(mask_data);

            // Visualização da máscara
            cv::Mat vis;
            cv::cvtColor(binary_mask, vis, cv::COLOR_GRAY2BGR);

            // Extraia lanes e centerline
            auto lanes = m_polyfitter->fitLanesInImage(binary_mask);
            for (const auto& lane : lanes) {
                for (size_t i = 1; i < lane.curve.size(); ++i) {
                    cv::line(vis, cv::Point(lane.curve[i-1].x, lane.curve[i-1].y),
                             cv::Point(lane.curve[i].x, lane.curve[i].y),
                             cv::Scalar(0, 255, 0), 2);
                }
            }
            auto centerline = m_polyfitter->computeVirtualCenterline(lanes, binary_mask.cols, binary_mask.rows);
            if (centerline.valid) {
                for (size_t i = 1; i < centerline.blend.size(); ++i) {
                    cv::line(vis, cv::Point(centerline.blend[i-1].x, centerline.blend[i-1].y),
                             cv::Point(centerline.blend[i].x, centerline.blend[i].y),
                             cv::Scalar(0, 128, 255), 2);
                }
            }

            // Opcional: desenhar waypoints do MPC (se disponíveis)
            // for (const auto& pt : mpc_waypoints) {
            //     cv::circle(vis, cv::Point(pt.x, pt.y), 3, cv::Scalar(255,0,0), -1);
            // }

            cv::imshow("Lane Detection Debug", vis);
            cv::waitKey(1);
        }
    } catch (...) {
        // Ignore errors
    }
}

