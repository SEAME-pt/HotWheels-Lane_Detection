/*!
 * @file ControlsManager.cpp
 * @brief Implementation of the ControlsManager class.
 * @version 0.2
 * @date 2025-07-07
 * @details This file contains the implementation of the ControlsManager class,
 * which is responsible for managing the different controllers and worker threads
 * for the car controls with hybrid MPC architecture.
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "ControlsManager.hpp"
#include "Debugger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>

/*!
 * @brief Constructs a ControlsManager object with hybrid MPC architecture.
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @param parent The parent QObject for this ControlsManager.
 * @details Initializes the engine controller, joystick controller, and various
 * worker threads for managing car controls with optimized direct MPC flow.
 */
ControlsManager::ControlsManager(int argc, char **argv, QObject *parent)
    : QObject(parent),
      // === Hardware Initialization ===
      m_engineController(0x40, 0x60, this),
      m_manualController(nullptr),
      
      // === State Initialization ===
      m_currentMode(DrivingMode::Manual),
      m_running(true),
      m_autonomousMode(false),
      
      // === Thread Pointers ===
      m_cameraStreamerThread(nullptr),
      m_manualControllerThread(nullptr),
      m_subscriberJoystickThread(nullptr),
      m_autonomousControlThread(nullptr),
      m_visionDataThread(nullptr),
      m_obstacleDataThread(nullptr),
      
      // === Object Pointers ===
      m_cameraStreamerObject(nullptr),
      m_subscriberJoystickObject(nullptr),
      m_mpcPlanner(nullptr),
      m_polyfitter(nullptr),
      
      // === Hybrid Flow Control ===
      m_useDirectFlow(true),
      m_maintainZeroMQ(true) {

    INFO_LOG("ControlsManager", "Initializing ControlsManager with hybrid MPC architecture");

    // === ETAPA 1: Hardware Controllers ===
    initializeHardwareControllers();
    
    // === ETAPA 2: MPC Components ===
    initializeMPCComponents();
    
    // === ETAPA 3: Vision Pipeline ===
    initializeVisionPipeline(argc, argv);
    
    // === ETAPA 4: Communication Layers ===
    initializeCommunication(argc, argv);
    
    // === ETAPA 5: Data Processing Threads ===
    initializeDataThreads();
    
    INFO_LOG("ControlsManager", "ControlsManager initialization complete");
}

/*!
 * @brief Initialize hardware controllers (joystick and engine)
 */
void ControlsManager::initializeHardwareControllers() {
    // === Joystick Controller Setup ===
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
        }
    );

    if(!m_manualController->init()) {
        ERROR_LOG("ControlsManager", "Failed to initialize joystick controller");
        throw std::runtime_error("Joystick initialization failed");
    }

    // === Joystick Thread Setup ===
    m_manualControllerThread = new QThread(this);
    m_manualController->moveToThread(m_manualControllerThread);
    
    connect(m_manualControllerThread, &QThread::started, 
            m_manualController, &JoysticksController::processInput);
    connect(m_manualController, &JoysticksController::finished, 
            m_manualControllerThread, &QThread::quit);
    
    m_manualControllerThread->start();
    
    INFO_LOG("ControlsManager", "Hardware controllers initialized");
}

/*!
 * @brief Initialize MPC components (Polyfitter and MPCPlanner)
 */
void ControlsManager::initializeMPCComponents() {
    // === Polyfitter for Lane Processing ===
    m_polyfitter = new Polyfitter();
    
    // Configure Polyfitter for ZeroMQ publishing if needed
    if(m_maintainZeroMQ) {
        m_polyfitter->enableZeroMQPublishing(true);
        INFO_LOG("ControlsManager", "Polyfitter ZeroMQ publishing enabled for external apps");
    }
    
    // === MPC Planner (lazy initialization) ===
    m_mpcPlanner = nullptr;  // Will be created when autonomous mode starts
    
    INFO_LOG("ControlsManager", "MPC components initialized");
}

/*!
 * @brief Initialize vision pipeline with direct MPC integration
 */
void ControlsManager::initializeVisionPipeline(int argc, char **argv) {
    // === Camera Streamer with Direct MPC Integration ===
    m_cameraStreamerThread = QThread::create([this, argc, argv]() {
        try {
            m_cameraStreamerObject = new CameraStreamer(0.5);
            
            // === FLUXO DIRETO: Direct MPC callback ===
            if(m_useDirectFlow) {
                m_cameraStreamerObject->setMPCCallback(
                    [this](const LaneInfo &lane_info) { 
                        receiveLaneDataDirect(lane_info); 
                    }
                );
                INFO_LOG("CameraStreamer", "Direct MPC flow enabled");
            }
            
            m_cameraStreamerObject->start();
            
        } catch(const std::exception &e) {
            ERROR_STREAM("ControlsManager") << "Vision pipeline error: " << e.what();
        }
    });
    
    m_cameraStreamerThread->start();
    INFO_LOG("ControlsManager", "Vision pipeline initialized");
}

/*!
 * @brief Initialize communication layers (ZeroMQ subscribers)
 */
void ControlsManager::initializeCommunication(int argc, char **argv) {
    // === ZeroMQ Joystick Subscriber ===
    m_subscriberJoystickObject = new Subscriber();
    m_subscriberJoystickThread = QThread::create([this, argc, argv]() {
        m_subscriberJoystickObject->connect("tcp://localhost:5555");
        m_subscriberJoystickObject->subscribe("joystick_value");
        
        while(m_running) {
            try {
                zmq::pollitem_t items[] = {
                    {static_cast<void *>(m_subscriberJoystickObject->getSocket()), 0, ZMQ_POLLIN, 0}
                };

                zmq::poll(items, 1, 100);

                if(items[0].revents & ZMQ_POLLIN) {
                    zmq::message_t message;
                    if(m_subscriberJoystickObject->getSocket().recv(&message, 0)) {
                        std::string received_msg(static_cast<char *>(message.data()), message.size());
                        
                        if(received_msg.find("joystick_value") == 0) {
                            std::string value = received_msg.substr(std::string("joystick_value ").length());
                            if(value == "true") {
                                setMode(DrivingMode::Manual);
                            } else if(value == "false") {
                                setMode(DrivingMode::Automatic);
                            }
                        }
                    }
                }
            } catch(const zmq::error_t &e) {
                ERROR_STREAM("ControlsManager") << "ZMQ communication error: " << e.what();
                break;
            }
        }
    });
    
    m_subscriberJoystickThread->start();
    INFO_LOG("ControlsManager", "Communication layer initialized");
}

/*!
 * @brief Initialize data processing threads for ZeroMQ fallback
 */
void ControlsManager::initializeDataThreads() {
    if(m_maintainZeroMQ) {
        // === Vision Data Thread ===
        m_visionSubscriber = std::make_unique<Subscriber>();
        m_visionDataThread = QThread::create([this]() { 
            visionDataUpdateLoop(); 
        });
        m_visionDataThread->start();

        // === Obstacle Data Thread ===
        m_obstacleSubscriber = std::make_unique<Subscriber>();
        m_obstacleDataThread = QThread::create([this]() { 
            obstacleDataUpdateLoop(); 
        });
        m_obstacleDataThread->start();
        
        INFO_LOG("ControlsManager", "ZeroMQ data threads initialized");
    }
}

/*!
 * @brief Destructor for the ControlsManager class.
 */
ControlsManager::~ControlsManager() {
    std::cout << "[~ControlsManager] CRITICAL SAFETY: Stopping all motors during cleanup" << std::endl;
    
    // CRITICAL SAFETY: Stop motors immediately during destruction
    try {
        m_engineController.emergencyHardwareStop();
        std::cout << "[~ControlsManager] Motors stopped successfully" << std::endl;
    } catch(const std::exception &e) {
        ERROR_STREAM("ControlsManager") << "[~ControlsManager] Error stopping motors: " << e.what();
        try {
            m_engineController.set_speed(0);
            m_engineController.set_steering(0);
        } catch(...) {
            std::cerr << "[~ControlsManager] CRITICAL: Failed to stop motors during cleanup!" << std::endl;
        }
    }

    m_running = false;
    stopAutonomousControl();

    // === Stop data update threads first ===
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

    // Stop threads safely
    if(m_subscriberJoystickThread) {
        if(m_subscriberJoystickObject) {
            m_subscriberJoystickObject->stop();
        }
        m_subscriberJoystickThread->quit();
        if(!m_subscriberJoystickThread->wait(3000)) {
            m_subscriberJoystickThread->terminate();
            m_subscriberJoystickThread->wait(1000);
        }
        delete m_subscriberJoystickThread;
        m_subscriberJoystickThread = nullptr;
    }

    if(m_manualControllerThread) {
        if(m_manualController)
            m_manualController->requestStop();
        m_manualControllerThread->quit();
        if(!m_manualControllerThread->wait(3000)) {
            m_manualControllerThread->terminate();
            m_manualControllerThread->wait(1000);
        }
        delete m_manualControllerThread;
        m_manualControllerThread = nullptr;
    }

    if(m_cameraStreamerThread) {
        if(m_cameraStreamerObject)
            m_cameraStreamerObject->stop();
        m_cameraStreamerThread->quit();
        if(!m_cameraStreamerThread->wait(3000)) {
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
    if(m_polyfitter) {
        delete m_polyfitter;
        m_polyfitter = nullptr;
    }
}

/*!
 * @brief Sets the driving mode.
 */
void ControlsManager::setMode(DrivingMode mode) {
    if(m_currentMode == mode)
        return;
    
    m_currentMode = mode;
    
    if(m_currentMode == DrivingMode::Automatic)
        startAutonomousControl();
    else
        stopAutonomousControlMotor();
}

/*!
 * @brief Stop autonomous control motors
 */
void ControlsManager::stopAutonomousControlMotor() {
    m_engineController.set_speed(0);
    m_engineController.set_steering(0);
    
    if(m_currentMode == DrivingMode::Manual) {
        std::cout << "[STOP] Autonomous control stopped, motors set to zero" << std::endl;
    } else {
        std::cout << "[STOP] Autonomous control stopped, but still in automatic mode" << std::endl;
    }
}

/*!
 * @brief Get lane data from direct flow (primary source)
 */
bool ControlsManager::getDirectLaneData(LaneInfo &lane_info) {
    std::lock_guard<std::mutex> lock(m_directMPCData.mutex);
    auto now = std::chrono::steady_clock::now();
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - m_directMPCData.timestamp).count();
    
    if (m_directMPCData.valid && age_ms < 100) { // 100ms timeout
        lane_info = m_directMPCData.current_lane_info;
        return true;
    }
    return false;
}

/*!
 * @brief Get lane data from ZeroMQ fallback
 */
bool ControlsManager::getZeroMQLaneData(LaneInfo &lane_info) {
    if (!m_maintainZeroMQ) return false;
    
    std::lock_guard<std::mutex> lock(m_cachedVisionData.mutex);
    auto now = std::chrono::steady_clock::now();
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - m_cachedVisionData.timestamp).count();
    
    if (m_cachedVisionData.valid && age_ms < 200) { // 200ms timeout
        lane_info = m_cachedVisionData.lane_info;
        return true;
    }
    return false;
}

/*!
 * @brief Generate straight trajectory fallback
 */
LaneInfo ControlsManager::generateStraightTrajectory() {
    LaneInfo fallback_lane;
    fallback_lane.left_boundary = -1.0;
    fallback_lane.right_boundary = 1.0;
    fallback_lane.center_line = 0.0;
    fallback_lane.lateral_offset = 0.0;
    fallback_lane.yaw_error = 0.0;
    fallback_lane.isValid = true;
    return fallback_lane;
}

/*!
 * @brief Apply smooth steering to prevent servo damage
 */
ControlCommand ControlsManager::applySmoothSteering(const ControlCommand &control) {
    static double last_steering = 0.0;
    double max_steering_change = 0.05; // rad per step
    double steering_diff = control.steer - last_steering;
    
    ControlCommand smooth_control = control;
    if(std::abs(steering_diff) > max_steering_change) {
        smooth_control.steer = last_steering + 
            (steering_diff > 0 ? max_steering_change : -max_steering_change);
    }
    last_steering = smooth_control.steer;
    
    return smooth_control;
}

/*!
 * @brief Apply controls with safety limits and servo protection
 */
void ControlsManager::applyControlsWithSafety(const ControlCommand &control, int control_counter) {
    // Convert to hardware values with safety limits
    int throttle_pct = static_cast<int>(
        std::clamp(control.throttle * 100, 0.0, 20.0)); // Max 20%

    // Servo protection: ±15° maximum
    int steer_angle = static_cast<int>(
        std::clamp(control.steer * 15, -15.0, 15.0));

    // Rate limiting for servo protection
    static int last_servo_angle = 0;
    int max_servo_change = 2; // Maximum 2° per iteration
    int servo_diff = steer_angle - last_servo_angle;

    if(std::abs(servo_diff) > max_servo_change) {
        steer_angle = last_servo_angle + 
            (servo_diff > 0 ? max_servo_change : -max_servo_change);
    }
    last_servo_angle = steer_angle;

    // Apply soft start to throttle
    double target_throttle = throttle_pct / 100.0;
    double final_throttle = applySoftStart(target_throttle);
    int final_throttle_pct = static_cast<int>(final_throttle * 100);

    // Store applied controls for state estimation
    m_lastThrottle = final_throttle;
    m_lastSteering = steer_angle * M_PI / 180.0;

    // Debug logging
    if(control_counter % 40 == 0) {
        std::cout << "Controls: Target=" << throttle_pct 
                  << "%, Final=" << final_throttle_pct 
                  << "%, Steering=" << steer_angle << "°" << std::endl;
    }

    // Apply to hardware (inverted speed for motor cross-connection fix)
    m_engineController.set_speed(-final_throttle_pct);
    m_engineController.set_steering(steer_angle);
}

/*!
 * @brief Extract waypoints from lane information
 */
std::vector<Point2D> ControlsManager::extractWaypointsFromLaneInfo(const LaneInfo &lane_info) {
    std::vector<Point2D> waypoints;
    
    if (lane_info.isValid) {
        // Generate waypoints based on lane boundaries
        double center_x = (lane_info.left_boundary + lane_info.right_boundary) / 2.0;
        
        // Generate forward waypoints
        for (int i = 1; i <= MPCConfig::horizon; ++i) {
            double distance_ahead = i * 2.0; // 2m spacing
            waypoints.emplace_back(distance_ahead, center_x + lane_info.lateral_offset);
        }
    }
    
    return waypoints;
}

/*!
 * @brief Start autonomous control with MPC
 */
void ControlsManager::startAutonomousControl() {
    if(m_currentMode != DrivingMode::Automatic)
        return;
    
    m_autonomousMode = true;
    
    // Create MPC planner if not already created
    if(!m_mpcPlanner) {
        m_mpcPlanner = new MPCPlanner();
    }
    
    // Initialize soft start system
    m_softStart.current_throttle_output = 0.0;
    m_softStart.start_time = std::chrono::steady_clock::now();
    
    std::cout << "[SOFT START] Autonomous mode activated with gradual acceleration" << std::endl;
    std::cout << "[SOFT START] Warmup period: " << m_softStart.warmup_duration_seconds << " seconds" << std::endl;
    std::cout << "[SOFT START] Max throttle change per step: " 
              << (m_softStart.max_throttle_change_per_step * 100) << "%" << std::endl;
    
    m_autonomousControlThread = QThread::create([this]() { autonomousControlLoop(); });
    m_autonomousControlThread->start();
}

/*!
 * @brief Receive lane data directly from CameraStreamer (direct flow)
 */
void ControlsManager::receiveLaneDataDirect(const LaneInfo &lane_info) {
    std::lock_guard<std::mutex> lock(m_directMPCData.mutex);
    m_directMPCData.current_lane_info = lane_info;
    m_directMPCData.timestamp = std::chrono::steady_clock::now();
    m_directMPCData.valid = true;
}

/*!
 * @brief Stop autonomous control
 */
void ControlsManager::stopAutonomousControl() {
    if(!m_autonomousMode)
        return;
    
    INFO_LOG("ControlsManager", "Stopping autonomous control...");
    m_autonomousMode = false;
    
    if(m_autonomousControlThread) {
        m_autonomousControlThread->quit();
        if(!m_autonomousControlThread->wait(2000)) {
            WARNING_LOG("ControlsManager", "Autonomous thread did not finish gracefully");
            m_autonomousControlThread->terminate();
            m_autonomousControlThread->wait(1000);
        }
        delete m_autonomousControlThread;
        m_autonomousControlThread = nullptr;
    }

    m_engineController.set_speed(0);
    m_engineController.set_steering(0);
    INFO_LOG("ControlsManager", "Autonomous control stopped successfully");
}

/*!
 * @brief Main autonomous control loop with optimized hybrid architecture
 */
void ControlsManager::autonomousControlLoop() {
    const double CONTROL_PERIOD = 1.0 / CONTROL_RATE;
    auto last_control_time = std::chrono::steady_clock::now();

    // Add external reference to global running flag
    extern std::atomic<bool> g_running;

    INFO_LOG("ControlsManager", "Autonomous control loop started with direct MPC flow");

    while(m_autonomousMode && m_running && g_running.load()) {
        // CRITICAL SAFETY: Check emergency stop flag first
        if(m_emergencyStop.load()) {
            m_engineController.set_speed(0);
            m_engineController.set_steering(0);
            INFO_LOG("ControlsManager", "Emergency stop is active - motors stopped");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - last_control_time).count();

        // Precise timing control
        if(elapsed < CONTROL_PERIOD) {
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long>(
                (CONTROL_PERIOD - elapsed) * 1000000 * 0.8)
            ));
            continue;
        }
        last_control_time = now;

        // Reduced logging frequency
        static int control_counter = 0;
        control_counter++;

        try {
            // === FLUXO DIRETO SIMPLIFICADO ===
            // 1. Get current vehicle state
            VehicleState current_state = getVehicleStateWithDiagnostics();

            // 2. Get lane data from direct flow (primary source)
            LaneInfo lane_info;
            bool has_valid_data = getDirectLaneData(lane_info);

            // 3. Fallback to ZeroMQ if direct flow fails (optional)
            if (!has_valid_data && m_maintainZeroMQ) {
                has_valid_data = getZeroMQLaneData(lane_info);
                if (control_counter % 40 == 0) {
                    DEBUG_LOG("ControlsManager", "Using ZeroMQ fallback");
                }
            }

            // 4. Final fallback: straight trajectory
            if (!has_valid_data) {
                lane_info = generateStraightTrajectory();
                if (control_counter % 40 == 0) {
                    DEBUG_LOG("ControlsManager", "Using straight fallback");
                }
            }

            // 5. Check for emergency obstacles
            if(getCachedEmergencyStop()) {
                m_engineController.set_speed(0);
                if(control_counter % 40 == 0) {
                    INFO_LOG("ControlsManager", "Emergency stop activated!");
                }
                continue;
            }

            // 6. Extract waypoints from lane info
            std::vector<Point2D> waypoints = extractWaypointsFromLaneInfo(lane_info);
            
            // 7. Calculate MPC control
            ControlCommand control = m_mpcPlanner->plan(current_state, waypoints, &lane_info);

            // 8. Apply constant speed override if enabled
            if(m_constantSpeedMode) {
                control.throttle = m_constantThrottle;
                control = applySmoothSteering(control);
            }

            // 9. Apply hardware controls with safety limits
            applyControlsWithSafety(control, control_counter);

        } catch(const std::exception &e) {
            ERROR_STREAM("ControlsManager") << "Autonomous control error: " << e.what();
            m_engineController.set_speed(0); // Safety stop
        }
    }

    INFO_LOG("ControlsManager", "Autonomous control loop ended");
}

// === Vehicle State Estimation Methods ===

VehicleState ControlsManager::getCurrentVehicleState() {
    return getEnhancedVehicleState();
}

VehicleState ControlsManager::getEnhancedVehicleState() {
    std::lock_guard<std::mutex> lock(m_stateEstimator.m_stateMutex);
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - m_stateEstimator.m_lastUpdate).count();
    m_stateEstimator.m_lastUpdate = now;

    if(!m_stateEstimator.m_initialized) {
        m_stateEstimator.m_estimatedState = {0.0, 0.0, 0.0, 0.5};
        m_stateEstimator.m_initialized = true;
        return m_stateEstimator.m_estimatedState;
    }

    dt = std::clamp(dt, 0.001, 0.1);
    double applied_throttle = m_lastThrottle.load();
    double applied_steering = m_lastSteering.load();

    updateVehicleStateEstimation(applied_throttle, applied_steering, dt);

    if(m_stateEstimator.m_useRealSensors.load()) {
        integrateRealSensorData();
    }

    return m_stateEstimator.m_estimatedState;
}

void ControlsManager::updateVehicleStateEstimation(double applied_throttle, double applied_steering, double dt) {
    VehicleState &state = m_stateEstimator.m_estimatedState;
    
    // Vehicle parameters (tuned for Jetracer)
    const double wheelbase = 0.15; // 15cm wheelbase
    const double max_acceleration = 3.0;
    const double max_deceleration = 4.0;
    const double rolling_resistance = 0.1;
    const double air_resistance = 0.05;
    const double max_velocity = 2.0;
    const double steering_response = 0.8;

    // === Velocity dynamics ===
    double target_acceleration = applied_throttle * max_acceleration;
    double resistance_force = rolling_resistance + air_resistance * state.velocity * state.velocity;
    double net_acceleration = target_acceleration - resistance_force;

    if(net_acceleration > 0) {
        net_acceleration = std::min(net_acceleration, max_acceleration);
    } else {
        net_acceleration = std::max(net_acceleration, -max_deceleration);
    }

    state.velocity += net_acceleration * dt;
    state.velocity = std::clamp(state.velocity, 0.0, max_velocity);

    if(state.velocity > 0.1) {
        state.velocity += (((double)rand() / RAND_MAX) - 0.5) * 0.02;
    }

    // === Position integration ===
    double distance = state.velocity * dt;
    state.x += distance * std::cos(state.yaw);
    state.y += distance * std::sin(state.yaw);

    // === Yaw dynamics ===
    if(std::abs(applied_steering) > 0.01 && state.velocity > 0.1) {
        double turning_radius = wheelbase / std::tan(applied_steering * steering_response);
        double angular_velocity = state.velocity / turning_radius;
        angular_velocity = std::clamp(angular_velocity, -2.0, 2.0);
        state.yaw += angular_velocity * dt;
        state.yaw += (((double)rand() / RAND_MAX) - 0.5) * 0.01;

        while(state.yaw > M_PI) state.yaw -= 2.0 * M_PI;
        while(state.yaw < -M_PI) state.yaw += 2.0 * M_PI;
    }
}

VehicleState ControlsManager::getVehicleStateWithDiagnostics() {
    VehicleState state = getEnhancedVehicleState();
    
    static int diagnostic_counter = 0;
    diagnostic_counter++;
    
    if(diagnostic_counter % 50 == 0) {
        INFO_LOG("ControlsManager", "Vehicle State Diagnostics:");
        INFO_STREAM("ControlsManager") << " Position: (" << state.x << ", " << state.y << ")";
        INFO_STREAM("ControlsManager") << " Velocity: " << state.velocity << " m/s";
        INFO_STREAM("ControlsManager") << " Yaw: " << state.yaw * 180.0 / M_PI << " degrees";
        INFO_STREAM("ControlsManager") << " Applied throttle: " << m_lastThrottle.load();
        INFO_STREAM("ControlsManager") << " Applied steering: " << m_lastSteering.load() * 180.0 / M_PI << " degrees";
    }

    return state;
}

// === ZeroMQ Methods (Fallback Support) ===

std::vector<Point2D> ControlsManager::getWaypointsFromVision() {
    std::vector<Point2D> waypoints;
    
    try {
        zmq::pollitem_t items[] = {
            {static_cast<void*>(m_visionSubscriber->getSocket()), 0, ZMQ_POLLIN, 0}
        };
        zmq::poll(items, 1, 50);
        
        if(items[0].revents & ZMQ_POLLIN) {
            zmq::message_t message;
            if(m_visionSubscriber->getSocket().recv(&message, 0)) {
                std::string received_msg(static_cast<char*>(message.data()), message.size());
                const std::string topic = "binary_mask ";
                
                if(received_msg.find(topic) == 0) {
                    std::string mask_data = received_msg.substr(topic.size());
                    cv::Mat binary_mask = deserializeMask(mask_data);
                    
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
        ERROR_STREAM("ControlsManager") << "ZMQ error: " << e.what();
    }

    if(waypoints.empty()) {
        for(int i = 1; i <= 10; ++i) {
            waypoints.emplace_back(i * 2.0, 0.0);
        }
    }

    return waypoints;
}

LaneInfo ControlsManager::getLaneInfoFromVision() {
    try {
        zmq::pollitem_t items[] = {
            {static_cast<void*>(m_visionSubscriber->getSocket()), 0, ZMQ_POLLIN, 0}
        };
        zmq::poll(items, 1, 50);
        
        if(items[0].revents & ZMQ_POLLIN) {
            zmq::message_t message;
            if(m_visionSubscriber->getSocket().recv(&message, 0)) {
                std::string received_msg(static_cast<char*>(message.data()), message.size());
                const std::string topic = "binary_mask ";
                
                if(received_msg.find(topic) == 0) {
                    std::string mask_data = received_msg.substr(topic.size());
                    cv::Mat binary_mask = deserializeMask(mask_data);
                    
                    auto lanes = m_polyfitter->fitLanesInImage(binary_mask);
                    auto centerline = m_polyfitter->computeVirtualCenterline(
                        lanes, binary_mask.cols, binary_mask.rows);
                    
                    // Convert to LaneInfo
                    LaneInfo lane_info;
                    if(centerline.valid && !lanes.empty()) {
                        // Extract lane boundaries from detected lanes
                        if(lanes.size() >= 2) {
                            lane_info.left_boundary = lanes[0].centroids.front().x;
                            lane_info.right_boundary = lanes[1].centroids.front().x;
                            lane_info.center_line = (lane_info.left_boundary + lane_info.right_boundary) / 2.0;
                            lane_info.lateral_offset = 0.0;
                            lane_info.yaw_error = 0.0;
                            lane_info.isValid = true;
                        }
                    }
                    return lane_info;
                }
            }
        }
    } catch(const zmq::error_t &e) {
        ERROR_STREAM("ControlsManager") << "ZMQ error: " << e.what();
    }

    return LaneInfo(0.0, 0.0);
}

bool ControlsManager::checkEmergencyObstacles() {
    try {
        zmq::pollitem_t items[] = {
            {static_cast<void*>(m_obstacleSubscriber->getSocket()), 0, ZMQ_POLLIN, 0}
        };
        zmq::poll(items, 1, 50);
        
        if(items[0].revents & ZMQ_POLLIN) {
            zmq::message_t message;
            if(m_obstacleSubscriber->getSocket().recv(&message, 0)) {
                std::string received_msg(static_cast<char*>(message.data()), message.size());
                const std::string topic = "emergency_stop ";
                
                if(received_msg.find(topic) == 0) {
                    std::string obstacle_data = received_msg.substr(topic.size());
                    return (obstacle_data == "true");
                }
            }
        }
    } catch(const zmq::error_t &e) {
        ERROR_STREAM("ControlsManager") << "ZMQ error: " << e.what();
    }

    return false;
}

// === Cached Data Access Methods ===

std::vector<Point2D> ControlsManager::getCachedWaypoints() {
    std::lock_guard<std::mutex> lock(m_cachedVisionData.mutex);
    auto now = std::chrono::steady_clock::now();
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - m_cachedVisionData.timestamp).count();

    if(m_cachedVisionData.valid && age_ms < DATA_TIMEOUT_MS) {
        return m_cachedVisionData.waypoints;
    }

    std::vector<Point2D> fallback_waypoints;
    for(int i = 1; i <= 10; ++i) {
        fallback_waypoints.emplace_back(i * 2.0, 0.0);
    }
    return fallback_waypoints;
}

LaneInfo ControlsManager::getCachedLaneInfo() {
    std::lock_guard<std::mutex> lock(m_cachedVisionData.mutex);
    auto now = std::chrono::steady_clock::now();
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - m_cachedVisionData.timestamp).count();

    if(m_cachedVisionData.valid && age_ms < DATA_TIMEOUT_MS) {
        return m_cachedVisionData.lane_info;
    }

    return LaneInfo(0.0, 0.0);
}

bool ControlsManager::getCachedEmergencyStop() {
    std::lock_guard<std::mutex> lock(m_cachedObstacleData.mutex);
    auto now = std::chrono::steady_clock::now();
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - m_cachedObstacleData.timestamp).count();

    if(m_cachedObstacleData.valid && age_ms < DATA_TIMEOUT_MS) {
        return m_cachedObstacleData.emergency_stop;
    }

    return false;
}

// === Background Data Update Threads ===

void ControlsManager::visionDataUpdateLoop() {
    const double UPDATE_PERIOD = 1.0 / VISION_UPDATE_RATE;
    auto last_update_time = std::chrono::steady_clock::now();
    extern std::atomic<bool> g_running;

    m_visionSubscriber->connect("tcp://localhost:5556");
    m_visionSubscriber->subscribe("binary_mask");
    INFO_LOG("ControlsManager", "Vision data update thread started");

    while(m_running && g_running.load()) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - last_update_time).count();

        if(elapsed < UPDATE_PERIOD) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        last_update_time = now;

        try {
            std::vector<Point2D> waypoints = getWaypointsFromVision();
            LaneInfo lane_info = getLaneInfoFromVision();

            {
                std::lock_guard<std::mutex> lock(m_cachedVisionData.mutex);
                m_cachedVisionData.waypoints = std::move(waypoints);
                m_cachedVisionData.lane_info = lane_info;
                m_cachedVisionData.timestamp = now;
                m_cachedVisionData.valid = true;
            }
        } catch(const std::exception &e) {
            ERROR_STREAM("ControlsManager") << "Vision data update error: " << e.what();
            {
                std::lock_guard<std::mutex> lock(m_cachedVisionData.mutex);
                m_cachedVisionData.valid = false;
            }
        }
    }

    INFO_LOG("ControlsManager", "Vision data update thread ended");
}

void ControlsManager::obstacleDataUpdateLoop() {
    const double UPDATE_PERIOD = 1.0 / OBSTACLE_UPDATE_RATE;
    auto last_update_time = std::chrono::steady_clock::now();
    extern std::atomic<bool> g_running;

    m_obstacleSubscriber->connect("tcp://localhost:5557");
    m_obstacleSubscriber->subscribe("emergency_stop");
    INFO_LOG("ControlsManager", "Obstacle data update thread started");

    while(m_running && g_running.load()) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - last_update_time).count();

        if(elapsed < UPDATE_PERIOD) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        last_update_time = now;

        try {
            bool emergency_stop = checkEmergencyObstacles();

            {
                std::lock_guard<std::mutex> lock(m_cachedObstacleData.mutex);
                m_cachedObstacleData.emergency_stop = emergency_stop;
                m_cachedObstacleData.timestamp = now;
                m_cachedObstacleData.valid = true;
            }
        } catch(const std::exception &e) {
            ERROR_STREAM("ControlsManager") << "Obstacle data update error: " << e.what();
            {
                std::lock_guard<std::mutex> lock(m_cachedObstacleData.mutex);
                m_cachedObstacleData.valid = false;
            }
        }
    }

    INFO_LOG("ControlsManager", "Obstacle data update thread ended");
}

// === Utility Methods ===

std::string ControlsManager::serializeMask(const cv::Mat &mask) {
    std::vector<uchar> buffer;
    cv::imencode(".png", mask, buffer);
    return std::string(buffer.begin(), buffer.end());
}

cv::Mat ControlsManager::deserializeMask(const std::string &data) {
    std::vector<uchar> buffer(data.begin(), data.end());
    return cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);
}

// === Safety and Control Methods ===

void ControlsManager::setConstantSpeedMode(bool enable, double target_speed, double throttle) {
    m_constantSpeedMode = enable;
    m_targetConstantSpeed = target_speed;
    m_constantThrottle = throttle;
    
    if(enable) {
        std::cout << "[ControlsManager] CONSTANT SPEED MODE ENABLED:" << std::endl;
        std::cout << " Target speed: " << target_speed << " m/s" << std::endl;
        std::cout << " Fixed throttle: " << throttle << std::endl;
    } else {
        std::cout << "[ControlsManager] Constant speed mode DISABLED" << std::endl;
    }
}

void ControlsManager::emergencyMotorStop() {
    std::cout << "\n*** EMERGENCY MOTOR STOP ACTIVATED ***" << std::endl;
    
    try {
        m_engineController.set_speed(0);
        m_engineController.emergencyHardwareStop();
    } catch(const std::exception &e) {
        ERROR_STREAM("ControlsManager") << "Emergency stop error: " << e.what();
        try {
            m_engineController.forcedMotorStop();
        } catch(...) {
            ERROR_LOG("ControlsManager", "CRITICAL: All motor stop methods failed!");
        }
    }

    try {
        m_engineController.set_steering(0);
    } catch(...) {
        ERROR_LOG("ControlsManager", "Warning: Could not center steering");
    }

    m_constantSpeedMode = false;
    m_emergencyStop = true;
    std::cout << "*** MOTORS STOPPED - SYSTEM SAFE ***" << std::endl;
}

void ControlsManager::emergencyStop() {
    std::cout << "\n*** CRITICAL EMERGENCY STOP ACTIVATED ***" << std::endl;
    
    m_emergencyStop = true;
    m_constantSpeedMode = false;
    m_currentMode = DrivingMode::Manual;

    try {
        m_engineController.emergencyHardwareStop();
        std::cout << "*** PRIMARY EMERGENCY STOP COMPLETED ***" << std::endl;
    } catch(const std::exception &e) {
        ERROR_STREAM("ControlsManager") << "Emergency stop error: " << e.what();
        try {
            m_engineController.forcedMotorStop();
            std::cout << "*** FALLBACK FORCED STOP COMPLETED ***" << std::endl;
        } catch(const std::exception &e2) {
            ERROR_STREAM("ControlsManager") << "Forced stop error: " << e2.what();
            try {
                for(int i = 0; i < 5; ++i) {
                    m_engineController.set_speed(0);
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                std::cout << "*** BASIC STOP FALLBACK COMPLETED ***" << std::endl;
            } catch(...) {
                ERROR_LOG("ControlsManager", "CRITICAL: ALL MOTOR STOP METHODS FAILED!");
            }
        }
    }

    try {
        m_engineController.set_steering(0);
    } catch(...) {
        ERROR_LOG("ControlsManager", "Warning: Could not center steering");
    }

    std::cout << "*** EMERGENCY STOP COMPLETE - ALL SYSTEMS HALTED ***" << std::endl;
}

void ControlsManager::resetEmergencyStop() {
    std::cout << "[SAFETY] Resetting emergency stop flag..." << std::endl;
    m_emergencyStop = false;
    std::cout << "[SAFETY] Emergency stop flag cleared - system ready for operation" << std::endl;
}

double ControlsManager::applySoftStart(double target_throttle) {
    if(!m_softStart.enabled) {
        return target_throttle;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed_seconds = std::chrono::duration<double>(now - m_softStart.start_time).count();

    double max_allowed_throttle;
    if(elapsed_seconds < m_softStart.warmup_duration_seconds) {
        double warmup_progress = elapsed_seconds / m_softStart.warmup_duration_seconds;
        max_allowed_throttle = m_softStart.initial_throttle_limit +
            (target_throttle - m_softStart.initial_throttle_limit) * warmup_progress;
        max_allowed_throttle = std::min(max_allowed_throttle, 
            m_softStart.initial_throttle_limit + (0.3 * warmup_progress));
    } else {
        max_allowed_throttle = target_throttle;
    }

    double throttle_change = target_throttle - m_softStart.current_throttle_output;
    double max_change = m_softStart.max_throttle_change_per_step;
    
    if(std::abs(throttle_change) > max_change) {
        if(throttle_change > 0) {
            m_softStart.current_throttle_output += max_change;
        } else {
            m_softStart.current_throttle_output -= max_change;
        }
    } else {
        m_softStart.current_throttle_output = target_throttle;
    }

    m_softStart.current_throttle_output = std::min(m_softStart.current_throttle_output, max_allowed_throttle);
    m_softStart.current_throttle_output = std::clamp(m_softStart.current_throttle_output, 0.0, 1.0);

    static int log_counter = 0;
    if(++log_counter % 40 == 0 && elapsed_seconds < m_softStart.warmup_duration_seconds) {
        std::cout << "[SOFT START] Elapsed: " << std::fixed << std::setprecision(1)
                  << elapsed_seconds << "s, Target: " << std::setprecision(2)
                  << (target_throttle * 100) << "%, Limited: " 
                  << (m_softStart.current_throttle_output * 100) << "%" << std::endl;
    }

    return m_softStart.current_throttle_output;
}

// === Sensor Integration Methods ===

void ControlsManager::enableRealSensors(bool enable) {
    m_stateEstimator.m_useRealSensors.store(enable);
    INFO_STREAM("ControlsManager") << "Real sensors " << (enable ? "enabled" : "disabled");
}

void ControlsManager::updateRealVelocity(double velocity) {
    m_stateEstimator.m_realVelocity.store(velocity);
    
    if(m_stateEstimator.m_useRealSensors.load()) {
        std::lock_guard<std::mutex> lock(m_stateEstimator.m_stateMutex);
        const double sensor_weight = 0.3;
        m_stateEstimator.m_estimatedState.velocity =
            sensor_weight * velocity +
            (1.0 - sensor_weight) * m_stateEstimator.m_estimatedState.velocity;
    }
}

void ControlsManager::updateRealYawRate(double yaw_rate) {
    m_stateEstimator.m_realYawRate.store(yaw_rate);
}

void ControlsManager::resetVehicleState(const VehicleState &initial_state) {
    std::lock_guard<std::mutex> lock(m_stateEstimator.m_stateMutex);
    m_stateEstimator.m_estimatedState = initial_state;
    m_stateEstimator.m_lastUpdate = std::chrono::steady_clock::now();
    
    qDebug() << "Vehicle state reset to: (" << initial_state.x << ", " << initial_state.y 
             << ", " << initial_state.yaw * 180.0 / M_PI << "°, " 
             << initial_state.velocity << " m/s)";
}

void ControlsManager::integrateRealSensorData() {
    VehicleState &state = m_stateEstimator.m_estimatedState;
    
    if(m_stateEstimator.m_useRealSensors.load()) {
        // Use vision data for position correction
        std::vector<Point2D> current_waypoints = getCachedWaypoints();
        if(!current_waypoints.empty() && current_waypoints.size() >= 2) {
            Point2D first_wp = current_waypoints[0];
            Point2D second_wp = current_waypoints[1];
            double expected_yaw = std::atan2(second_wp.y - first_wp.y, second_wp.x - first_wp.x);
            
            const double vision_weight = 0.1;
            double yaw_correction = expected_yaw - state.yaw;
            
            while(yaw_correction > M_PI) yaw_correction -= 2.0 * M_PI;
            while(yaw_correction < -M_PI) yaw_correction += 2.0 * M_PI;
            
            state.yaw += vision_weight * yaw_correction;
        }

        // Use applied controls for velocity estimation refinement
        double real_throttle_effect = m_stateEstimator.m_realVelocity.load();
        if(real_throttle_effect > 0.01) {
            const double feedback_weight = 0.2;
            state.velocity = feedback_weight * real_throttle_effect + 
                           (1.0 - feedback_weight) * state.velocity;
        }

        // Add realistic measurement noise
        double vision_noise_x = (((double)rand() / RAND_MAX) - 0.5) * 0.05;
        double vision_noise_y = (((double)rand() / RAND_MAX) - 0.5) * 0.05;
        state.x += vision_noise_x;
        state.y += vision_noise_y;

        double control_noise_vel = (((double)rand() / RAND_MAX) - 0.5) * 0.01;
        double control_noise_yaw = (((double)rand() / RAND_MAX) - 0.5) * 0.005;
        state.velocity += control_noise_vel;
        state.velocity = std::max(0.0, state.velocity);
        state.yaw += control_noise_yaw;

        while(state.yaw > M_PI) state.yaw -= 2.0 * M_PI;
        while(state.yaw < -M_PI) state.yaw += 2.0 * M_PI;
    }
}

// === Direct Control Methods ===

void ControlsManager::applyControlCommand(const ControlCommand &command) {
    applyThrottle(command.throttle);
    applySteering(command.steer);
}

void ControlsManager::applyThrottle(double throttle) {
    throttle = std::max(-1.0, std::min(1.0, throttle));
    int throttle_pwm = static_cast<int>(throttle * 100);
    m_engineController.set_speed(throttle_pwm);
    m_lastThrottle.store(throttle);
}

void ControlsManager::applySteering(double steering) {
    steering = std::max(-1.0, std::min(1.0, steering));
    int steering_pwm = static_cast<int>(steering * 100);
    m_engineController.set_steering(steering_pwm);
    m_lastSteering.store(steering);
}

// === Debug and Visualization Methods ===

void ControlsManager::showVisionDebug() {
    Subscriber vision_sub;
    vision_sub.connect("tcp://localhost:5556");
    vision_sub.subscribe("binary_mask");
    
    try {
        zmq::pollitem_t items[] = {{static_cast<void*>(vision_sub.getSocket()), 0, ZMQ_POLLIN, 0}};
        zmq::poll(items, 1, 100);
        
        if(items[0].revents & ZMQ_POLLIN) {
            zmq::message_t message;
            if(vision_sub.getSocket().recv(&message, 0)) {
                std::string received_msg(static_cast<char*>(message.data()), message.size());
                const std::string topic = "binary_mask ";
                
                if(received_msg.find(topic) == 0) {
                    std::string mask_data = received_msg.substr(topic.size());
                    cv::Mat binary_mask = deserializeMask(mask_data);
                    
                    cv::Mat vis;
                    cv::cvtColor(binary_mask, vis, cv::COLOR_GRAY2BGR);
                    
                    auto lanes = m_polyfitter->fitLanesInImage(binary_mask);
                    for(const auto &lane : lanes) {
                        for(size_t i = 1; i < lane.curve.size(); ++i) {
                            cv::line(vis, 
                                cv::Point(lane.curve[i-1].x, lane.curve[i-1].y),
                                cv::Point(lane.curve[i].x, lane.curve[i].y),
                                cv::Scalar(0, 255, 0), 2);
                        }
                    }
                    
                    auto centerline = m_polyfitter->computeVirtualCenterline(
                        lanes, binary_mask.cols, binary_mask.rows);
                    
                    if(centerline.valid) {
                        for(size_t i = 1; i < centerline.blend.size(); ++i) {
                            cv::line(vis,
                                cv::Point(centerline.blend[i-1].x, centerline.blend[i-1].y),
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
        ERROR_STREAM("ControlsManager") << "Vision debug error: " << e.what();
    }
}
