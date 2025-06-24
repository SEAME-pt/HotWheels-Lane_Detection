#include "car_controls/includes/CommonTypes.hpp"
#include "car_controls/includes/ControlsManager.hpp"
#include "car_controls/includes/MPCOptimizer.hpp"
#include "car_controls/includes/MPCPlanner.hpp"
#include "ZeroMQ/Subscriber.hpp"
#include <QApplication>
#include <QTimer>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <signal.h>

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

// Signal handler for Ctrl+C
void signalHandler(int signum) {
	std::cout << "\nReceived signal " << signum << ". Shutting down gracefully..." << std::endl;
	g_running = false;
	QApplication::quit();
}

class MPCIntegratedApp : public QObject {
		Q_OBJECT

	private:
		ControlsManager *controls_manager;
		MPCPlanner *mpc_planner;
		QTimer *mpc_timer;

		// Estado do MPC
		bool mpc_active = false;
		std::vector<Point2D> recorded_waypoints;
		VehicleState current_state{0.0, 0.0, 0.0, 0.5};
		int step_counter = 0;

		// Add visualization components
		cv::Mat m_visualizationFrame;
		QTimer *m_visualizationTimer;

		// Add camera integration
		cv::Mat m_currentCameraFrame;
		cv::Mat m_processedFrame;
		bool m_cameraFrameAvailable = false;
		
		// Add persistent subscriber for lane detection
		std::unique_ptr<Subscriber> m_laneDetectionSubscriber;

	public:
	MPCIntegratedApp(int argc, char **argv, QObject *parent = nullptr) : QObject(parent),
		controls_manager(nullptr), mpc_planner(nullptr), mpc_timer(nullptr) {
		// Setup signal handlers
		signal(SIGINT, signalHandler);
		signal(SIGTERM, signalHandler);

		try {
			// Inicializar o sistema de controles existente
			controls_manager = new ControlsManager(argc, argv, this);

			// Inicializar MPC
			mpc_planner = new MPCPlanner();

			// Timer para executar MPC periodicamente
			mpc_timer = new QTimer(this);
			connect(mpc_timer, &QTimer::timeout, this, &MPCIntegratedApp::runMPCStep);

			std::cout << "=== MPC Integrated System ===" << std::endl;
			std::cout << "Sistema iniciado com controle manual" << std::endl;
			std::cout << "Comandos disponíveis:" << std::endl;
			std::cout << "- Use joystick para mover e gravar trajetória" << std::endl;
			std::cout << "- Pressione ENTER para alternar Manual/MPC" << std::endl;
			std::cout << "- Pressione 'r' para iniciar/parar gravação" << std::endl;
			std::cout << "- Pressione 'q' para sair" << std::endl;

			// Conectar stdin para comandos
			setupKeyboardInput();

			// Setup visualization
			setupVisualization();
		} catch(const std::exception& e) {
			std::cerr << "[MPCIntegratedApp] Initialization error: " << e.what() << std::endl;
			throw;
		}
	}

	~MPCIntegratedApp() {
		// Stop timers first
		if(mpc_timer) {
			mpc_timer->stop();
		}
		
		// Cleanup publishers before destroying anything else
		Publisher::destroyAll();
		
		// Delete in reverse order of creation
		delete mpc_planner;
		mpc_planner = nullptr;
		
		// controls_manager is QObject child, will be deleted automatically
		std::cout << "[~MPCIntegratedApp] Cleanup complete" << std::endl;
	}

	private slots:
		void runMPCStep() {
			if(!mpc_active || recorded_waypoints.size() < 3 || !g_running) {
				return;
			}

			try {
				// Usar waypoints gravados como referência
				LaneInfo lane_info(0.0, 0.0);
				ControlCommand control =
				    mpc_planner->plan(current_state, recorded_waypoints, &lane_info);

				// Converter controles MPC para comandos do hardware
				int mpc_steering = static_cast<int>(control.steer * 45.0 / 0.35);
				int mpc_speed = static_cast<int>(control.throttle * 100.0 / 0.6);

				// Limitar comandos
				mpc_steering = std::max(-45, std::min(45, mpc_steering));
				mpc_speed = std::max(0, std::min(100, mpc_speed));

				// Aplicar através do EngineController
				// Nota: Você precisará expor métodos públicos no ControlsManager
				// para acessar o EngineController
				// controls_manager->getEngineController()->set_steering(mpc_steering);
				// controls_manager->getEngineController()->set_speed(mpc_speed);

				// Por enquanto, apenas log
				step_counter++;
				if(step_counter % 20 == 0) {
					std::cout << "MPC Step " << step_counter << " | Throttle: " << std::fixed
					          << std::setprecision(3) << control.throttle
					          << " | Steering: " << control.steer
					          << " | Target Speed: " << mpc_speed
					          << " | Target Steering: " << mpc_steering << std::endl;
				}

				// Simular atualização do estado (em um sistema real,
				// isso viria de sensores/odometria)
				updateVehicleState(control.throttle, control.steer);
			} catch(const std::exception &e) {
				std::cerr << "Erro no MPC: " << e.what() << std::endl;
			}
		}

		void handleKeyPress() {
			static bool recording = false;

			std::string input;
			std::getline(std::cin, input);

			if(input.empty() || input == "m") {
				// Alternar modo
				toggleMPCMode();
			} else if(input == "r") {
				// Alternar gravação
				recording = !recording;
				std::cout << (recording ? "Iniciando gravação de trajetória"
				                        : "Parando gravação de trajetória")
				          << std::endl;
				if(recording) {
					recorded_waypoints.clear();
					// Aqui você iniciaria a gravação baseada no movimento real do joystick
					startRecording();
				}
			} else if(input == "c") {
				// Limpar waypoints
				recorded_waypoints.clear();
				std::cout << "Trajetória limpa (" << recorded_waypoints.size() << " waypoints)"
				          << std::endl;
			} else if(input == "s") {
				// Mostrar status
				showStatus();
			} else if(input == "q") {
				// Sair
				QApplication::quit();
			} else {
				std::cout << "Comando não reconhecido. Use: m(modo), r(gravar), c(limpar), "
				             "s(status), q(sair)"
				          << std::endl;
			}
		}

		void getCameraFrame() {
            try {
                // Only use ZeroMQ connection - don't try to access camera directly
                static bool first_connection_attempt = true;
                static std::chrono::time_point<std::chrono::steady_clock> last_zmq_attempt;
                static bool zmq_connection_failed = false;

                auto now = std::chrono::steady_clock::now();

                // Try ZeroMQ connection periodically, but not too often
                if(first_connection_attempt || 
                   (!zmq_connection_failed && 
                    std::chrono::duration_cast<std::chrono::seconds>(now - last_zmq_attempt).count() > 2)) {
                    
                    first_connection_attempt = false;
                    last_zmq_attempt = now;

                    try {
                        // Try raw camera frames first (from port 5558)
                        Subscriber camera_sub;
                        camera_sub.connect("tcp://localhost:5558");
                        camera_sub.subscribe("camera_frame");

                        zmq::pollitem_t items[] = {
                            {static_cast<void*>(camera_sub.getSocket()), 0, ZMQ_POLLIN, 0}};
                        zmq::poll(items, 1, 200); // Longer timeout for connection attempt

                        if(items[0].revents & ZMQ_POLLIN) {
                            zmq::message_t message;
                            if(camera_sub.getSocket().recv(&message, ZMQ_DONTWAIT)) {
                                std::string received_msg(static_cast<char*>(message.data()), 
                                                       message.size());

                                if(received_msg.find("camera_frame ") == 0) {
                                    std::string frame_data = received_msg.substr(13); // "camera_frame ".length()
                                    
                                    std::vector<uchar> buffer(frame_data.begin(), frame_data.end());
                                    cv::Mat decoded_frame = cv::imdecode(buffer, cv::IMREAD_COLOR);

                                    if(!decoded_frame.empty()) {
                                        m_currentCameraFrame = decoded_frame.clone();
                                        m_cameraFrameAvailable = true;
                                        zmq_connection_failed = false;
                                        
                                        std::cout << "[DEBUG] Connected to raw camera feed: " 
                                                  << decoded_frame.cols << "x" << decoded_frame.rows << std::endl;
                                        
                                        // Get inference results from port 5556
                                        getLaneDetectionFrame();
                                        return;
                                    }
                                }
                            }
                        }
                        
                        // If raw camera fails, try inference frames from port 5556
                        Subscriber inference_sub;
                        inference_sub.connect("tcp://localhost:5556");
                        inference_sub.subscribe("inference_frame");

                        zmq::poll(items, 1, 100);
                        if(items[0].revents & ZMQ_POLLIN) {
                            zmq::message_t message;
                            if(inference_sub.getSocket().recv(&message, ZMQ_DONTWAIT)) {
                                std::string received_msg(static_cast<char*>(message.data()), 
                                                       message.size());
                                
                                if(received_msg.find("inference_frame ") == 0) {
                                    std::cout << "[DEBUG] Found inference data from CameraStreamer" << std::endl;
                                    zmq_connection_failed = false;
                                }
                            }
                        }

                        if(!zmq_connection_failed) {
                            std::cout << "[DEBUG] CameraStreamer is running but no frames yet" << std::endl;
                        }

                    } catch(const std::exception& e) {
                        if(!zmq_connection_failed) {
                            std::cout << "[DEBUG] ZeroMQ connection failed: " << e.what() << std::endl;
                            zmq_connection_failed = true;
                        }
                    }
                }

                // If ZeroMQ is not working, use synthetic feed
                if(zmq_connection_failed || !m_cameraFrameAvailable) {
                    // TODO: Implement createSyntheticCameraFeed();
                    // std::cerr << "[getCameraFrame] Warning: Camera feed not available" << std::endl;
                }

            } catch(const std::exception& e) {
                // std::cerr << "[getCameraFrame] Error: " << e.what() << std::endl;
                // TODO: Implement createSyntheticCameraFeed();
                // std::cerr << "[getCameraFrame] Warning: Falling back to synthetic feed not implemented" << std::endl;
            }
        }

        void getLaneDetectionFrame() {
            if(!m_laneDetectionSubscriber) {
                // std::cout << "[DEBUG] Lane detection subscriber not initialized" << std::endl;
                return;
            }
            
            try {
                zmq::pollitem_t items[] = {
                    {static_cast<void*>(m_laneDetectionSubscriber->getSocket()), 0, ZMQ_POLLIN, 0}};
                zmq::poll(items, 1, 5); // Short timeout to avoid blocking

                if(items[0].revents & ZMQ_POLLIN) {
                    zmq::message_t message;
                    if(m_laneDetectionSubscriber->getSocket().recv(&message, ZMQ_DONTWAIT)) {
                        std::cout << "[DEBUG] Received inference message, size: " << message.size() << std::endl;
                        
                        std::string received_msg(static_cast<char*>(message.data()), 
                                               message.size());
                    
                        if(received_msg.find("inference_frame ") == 0) {
                            std::cout << "[DEBUG] Found inference_frame header" << std::endl;
                            std::string mask_data = received_msg.substr(16); // "inference_frame ".length()
                            
                            std::vector<uchar> buffer(mask_data.begin(), mask_data.end());
                            cv::Mat binary_mask = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);

                            if(!binary_mask.empty()) {
                                std::cout << "[DEBUG] Successfully decoded lane detection mask: " 
                                          << binary_mask.cols << "x" << binary_mask.rows << std::endl;
                                createLaneVisualization(binary_mask);
                                return;
                            } else {
                                std::cout << "[DEBUG] Failed to decode lane detection mask" << std::endl;
                            }
                        } else {
                            std::cout << "[DEBUG] Received message but no inference_frame header found" << std::endl;
                        }
                    }
                } else {
                    // std::cout << "[DEBUG] No lane detection data available" << std::endl;
                }
                
            } catch(const std::exception& e) {
                std::cerr << "[getLaneDetectionFrame] Error: " << e.what() << std::endl;
            }
        }

		void updateVisualization() {
			if(!g_running) {
				return;
			}

			try {
				// Adjust for 1024x600 screen - create smaller visualization (900x550)
				m_visualizationFrame = cv::Mat::zeros(550, 900, CV_8UC3);

				// Adjust regions for smaller window
				cv::Rect cameraRegion(10, 10, 420, 240);        // Smaller camera feed
				cv::Rect processedRegion(10, 260, 420, 240);    // Smaller processed view
				cv::Rect trajectoryRegion(440, 10, 450, 450);   // Trajectory area

				// Get camera frame from CameraStreamer
				getCameraFrame();

				// Draw camera feed
				if(!m_currentCameraFrame.empty()) {
					cv::Mat resizedCamera;
					cv::resize(m_currentCameraFrame, resizedCamera, cv::Size(420, 240));

					if(resizedCamera.channels() == 1) {
						cv::cvtColor(resizedCamera, resizedCamera, cv::COLOR_GRAY2BGR);
					}

					resizedCamera.copyTo(m_visualizationFrame(cameraRegion));

					cv::putText(m_visualizationFrame, "Camera Feed", cv::Point(15, 30),
					            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
				} else {
					cv::rectangle(m_visualizationFrame, cameraRegion, cv::Scalar(50, 50, 50), -1);
					cv::putText(m_visualizationFrame, "No Camera Feed",
					            cv::Point(cameraRegion.x + 140, cameraRegion.y + 120),
					            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
				}

				// Draw processed frame (lane detection result)
				if(!m_processedFrame.empty()) {
					cv::Mat resizedProcessed;
					cv::resize(m_processedFrame, resizedProcessed, cv::Size(420, 240));

					if(resizedProcessed.channels() == 1) {
						cv::cvtColor(resizedProcessed, resizedProcessed, cv::COLOR_GRAY2BGR);
					}

					resizedProcessed.copyTo(m_visualizationFrame(processedRegion));

					cv::putText(m_visualizationFrame, "Lane Detection", cv::Point(15, 280),
					            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
				} else {
					cv::rectangle(m_visualizationFrame, processedRegion, cv::Scalar(30, 30, 30), -1);
					cv::putText(m_visualizationFrame, "No Lane Detection",
					            cv::Point(processedRegion.x + 120, processedRegion.y + 120),
					            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
				}			// Right side: Trajectory visualization
			// TODO: Implement drawTrajectoryVisualization(trajectoryRegion);
			cv::putText(m_visualizationFrame, "Trajectory View", 
			            cv::Point(trajectoryRegion.x + 10, trajectoryRegion.y + 30),
			            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

				// Add compact system status
				drawCompactSystemStatus();

				// Display the frame
				cv::imshow("MPC Integrated System", m_visualizationFrame);

				// Handle OpenCV events and check for exit
				int key = cv::waitKey(1) & 0xFF;
				if(key == 'q' || key == 27) { // 'q' or ESC
					g_running = false;
					QApplication::quit();
				} else if(key == 'm') {
					toggleMPCMode();
				} else if(key == 'r') {
					handleRecordCommand();
				} else if(key == 'c') {
					recorded_waypoints.clear();
					std::cout << "Trajetória limpa (" << recorded_waypoints.size() << " waypoints)"
					          << std::endl;
				} else if(key == 's') {
					showStatus();
				}

			} catch(const std::exception &e) {
				std::cerr << "Visualization error: " << e.what() << std::endl;
			}
		}

        void drawCompactSystemStatus() {
            // Compact system status for smaller screen
            int status_x = 450;
            int status_y = 470;
            int line_height = 15;

            cv::putText(m_visualizationFrame,
                        mpc_active ? "Mode: MPC" : "Mode: Manual",
                        cv::Point(status_x, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        mpc_active ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);

            cv::putText(m_visualizationFrame,
                        "WP: " + std::to_string(recorded_waypoints.size()),
                        cv::Point(status_x, status_y + line_height), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(255, 255, 255), 1);

            cv::putText(m_visualizationFrame,
                        "Pos: (" + std::to_string(current_state.x).substr(0, 4) + "," +
                            std::to_string(current_state.y).substr(0, 4) + ")",
                        cv::Point(status_x, status_y + 2*line_height), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(255, 255, 255), 1);

            cv::putText(m_visualizationFrame,
                        "Vel: " + std::to_string(current_state.velocity).substr(0, 4),
                        cv::Point(status_x, status_y + 3*line_height), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(255, 255, 255), 1);

            // Enhanced camera status
            std::string camera_status;
            if(m_cameraFrameAvailable && !m_currentCameraFrame.empty()) {
                // Detect if it's synthetic
                cv::Scalar mean_color = cv::mean(m_currentCameraFrame);
                if(mean_color[0] < 30 && mean_color[1] < 30 && mean_color[2] < 30) {
                    camera_status = "Cam: SYNTHETIC";
                } else {
                    camera_status = "Cam: REAL";
                }
            } else {
                camera_status = "Cam: WAITING";
            }

            cv::putText(m_visualizationFrame, camera_status,
                        cv::Point(status_x + 150, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        m_cameraFrameAvailable ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 0), 1);

            // Compact instructions
            cv::putText(m_visualizationFrame, 
                        "m:mode r:rec c:clear s:status q:quit",
                        cv::Point(10, 530), cv::FONT_HERSHEY_SIMPLEX, 0.4, 
                        cv::Scalar(200, 200, 200), 1);
        }

		void setupVisualization() {
			// Create appropriately sized OpenCV window for 1024x600 screen
			cv::namedWindow("MPC Integrated System", cv::WINDOW_AUTOSIZE);

			// Setup visualization timer
			m_visualizationTimer = new QTimer(this);
			connect(m_visualizationTimer, &QTimer::timeout, this,
			        &MPCIntegratedApp::updateVisualization);
			m_visualizationTimer->start(50); // 20 FPS
			
			// Initialize persistent lane detection subscriber
			try {
				m_laneDetectionSubscriber = std::make_unique<Subscriber>();
				m_laneDetectionSubscriber->connect("tcp://localhost:5556");
				m_laneDetectionSubscriber->subscribe("inference_frame");
				std::cout << "[DEBUG] Lane detection subscriber initialized" << std::endl;
			} catch(const std::exception& e) {
				std::cerr << "[setupVisualization] Failed to initialize lane detection subscriber: " 
				          << e.what() << std::endl;
			}
		}

		void handleRecordCommand() {
			static bool recording = false;
			recording = !recording;
			std::cout << (recording ? "Iniciando gravação de trajetória"
			                        : "Parando gravação de trajetória")
			          << std::endl;
			if(recording) {
				recorded_waypoints.clear();
				startRecording();
			}
		}

		void setupKeyboardInput() {
			std::cout << "Digite comandos (m: modo, r: gravar, c: limpar, s: status, q: sair):"
			          << std::endl;
			std::cout << "Ou use as teclas na janela OpenCV" << std::endl;

			// Timer para verificar input
			QTimer *input_timer = new QTimer(this);
			connect(input_timer, &QTimer::timeout, [this]() {
				if(!g_running) {
					return;
				}
				if(std::cin.rdbuf()->in_avail()) {
					handleKeyPress();
				}
			});
			input_timer->start(100); // Check every 100ms
		}

		void toggleMPCMode() {
			mpc_active = !mpc_active;

			if(mpc_active) {
				if(recorded_waypoints.size() < 3) {
					std::cout << "Erro: Precisa de pelo menos 3 waypoints gravados!" << std::endl;
					mpc_active = false;
					return;
				}

				// Mudar para modo autônomo
				controls_manager->setMode(DrivingMode::Automatic);
				mpc_timer->start(50); // 20 Hz
				std::cout << "Modo MPC ATIVADO - Seguindo trajetória com "
				          << recorded_waypoints.size() << " waypoints" << std::endl;
			} else {
				// Mudar para modo manual
				controls_manager->setMode(DrivingMode::Manual);
				mpc_timer->stop();
				std::cout << "Modo MANUAL ATIVADO - Use joystick" << std::endl;
			}
		}

		void startRecording() {
			// Em um sistema real, você conectaria aos sinais do joystick
			// para gravar as posições conforme o usuário move o carro

			// Simulação: gerar alguns waypoints de exemplo
			std::cout << "Simulando gravação... (em sistema real, use o joystick)" << std::endl;

			// Gerar trajetória curva simples
			for(int i = 0; i < 20; ++i) {
				double x = i * 0.5;
				double y = 1.5 * std::sin(x * 0.3);
				recorded_waypoints.emplace_back(x, y);
			}

			std::cout << "Trajetória simulada gravada com " << recorded_waypoints.size()
			          << " pontos" << std::endl;
		}

		void updateVehicleState(double throttle, double steer) {
			// Modelo cinemático simples para simular movimento
			double dt = 0.05;        // 20 Hz
			double wheelbase = 0.15; // 15cm

			current_state.velocity += throttle * dt;
			current_state.velocity = std::max(0.1, std::min(current_state.velocity, 1.5));

			current_state.x += current_state.velocity * std::cos(current_state.yaw) * dt;
			current_state.y += current_state.velocity * std::sin(current_state.yaw) * dt;
			current_state.yaw += (current_state.velocity / wheelbase) * std::tan(steer) * dt;

			// Normalizar ângulo
			while(current_state.yaw > M_PI)
				current_state.yaw -= 2.0 * M_PI;
			while(current_state.yaw < -M_PI)
				current_state.yaw += 2.0 * M_PI;
		}

		void showStatus() {
			std::cout << "\n=== STATUS ===" << std::endl;
			std::cout << "Modo: " << (mpc_active ? "MPC (Autônomo)" : "Manual") << std::endl;
			std::cout << "Waypoints gravados: " << recorded_waypoints.size() << std::endl;
			std::cout << "Posição atual: (" << std::fixed << std::setprecision(2) << current_state.x
			          << ", " << current_state.y << ")" << std::endl;
			std::cout << "Velocidade: " << current_state.velocity << " m/s" << std::endl;
			std::cout << "Orientação: " << current_state.yaw * 180 / M_PI << " graus" << std::endl;
			std::cout << "MPC Steps: " << step_counter << std::endl;
			std::cout << "===============\n" << std::endl;
		}

        void createLaneVisualization(const cv::Mat& binary_mask) {
            try {
                if(binary_mask.empty()) {
                    return;
                }

                // Create a colored visualization from the binary mask
                cv::Mat colored_lanes;
                cv::cvtColor(binary_mask, colored_lanes, cv::COLOR_GRAY2BGR);

                // Enhance lane lines with color overlay
                cv::Mat lane_overlay = cv::Mat::zeros(binary_mask.size(), CV_8UC3);
                
                // Find contours to identify lane lines
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                // Draw lane lines in different colors
                for(size_t i = 0; i < contours.size(); ++i) {
                    if(cv::contourArea(contours[i]) > 100) { // Filter small noise
                        cv::Scalar color;
                        if(i % 3 == 0) color = cv::Scalar(0, 255, 0);      // Green
                        else if(i % 3 == 1) color = cv::Scalar(255, 0, 0); // Blue
                        else color = cv::Scalar(0, 255, 255);              // Yellow
                        
                        cv::drawContours(lane_overlay, contours, static_cast<int>(i), color, 2);
                    }
                }

                // Blend the original mask with colored overlay
                cv::addWeighted(colored_lanes, 0.7, lane_overlay, 0.3, 0, m_processedFrame);

                // Add lane detection info text
                cv::putText(m_processedFrame, "Lanes: " + std::to_string(contours.size()), 
                           cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                           cv::Scalar(255, 255, 255), 2);

            } catch(const std::exception& e) {
                std::cerr << "[createLaneVisualization] Error: " << e.what() << std::endl;
                // Create a simple error visualization
                m_processedFrame = cv::Mat::zeros(binary_mask.size(), CV_8UC3);
                cv::putText(m_processedFrame, "Lane Processing Error", 
                           cv::Point(50, binary_mask.rows/2), cv::FONT_HERSHEY_SIMPLEX, 
                           0.8, cv::Scalar(0, 0, 255), 2);
            }
        }
};

int main(int argc, char *argv[]) {
	// Initialize Qt application first
	QApplication app(argc, argv);

	std::cout << "Iniciando sistema integrado MPC + Car Controls..." << std::endl;

	// Set up proper signal handling before creating objects
	signal(SIGINT, signalHandler);
	signal(SIGTERM, signalHandler);

	try {
		// Create the integrated app with proper exception handling
		std::unique_ptr<MPCIntegratedApp> integrated_app;
		
		try {
			integrated_app = std::make_unique<MPCIntegratedApp>(argc, argv);
		} catch(const std::exception& e) {
			std::cerr << "Failed to initialize application: " << e.what() << std::endl;
			return 1;
		}

		std::cout << "Sistema pronto! Use os comandos no terminal ou na janela OpenCV."
		          << std::endl;
		std::cout << "Pressione Ctrl+C ou 'q' para sair." << std::endl;

		// Setup periodic check for global running flag
		QTimer *shutdown_timer = new QTimer(&app);
		QObject::connect(shutdown_timer, &QTimer::timeout, [&]() {
			if(!g_running) {
				cv::destroyAllWindows();
				app.quit();
			}
		});
		shutdown_timer->start(100);

		int result = app.exec();

		// Cleanup
		integrated_app.reset(); // Explicit cleanup before destroying windows
		cv::destroyAllWindows();
		
		// Final cleanup of singletons
		Publisher::destroyAll();
		
		return result;

	} catch(const std::exception &e) {
		std::cerr << "Erro: " << e.what() << std::endl;
		cv::destroyAllWindows();
		Publisher::destroyAll();
		return 1;
	}
}

#include "main.moc"
