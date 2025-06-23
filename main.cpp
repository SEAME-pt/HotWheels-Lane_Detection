#include "car_controls/includes/CommonTypes.hpp"
#include "car_controls/includes/ControlsManager.hpp"
#include "car_controls/includes/MPCOptimizer.hpp"
#include "car_controls/includes/MPCPlanner.hpp"
#include <QApplication>
#include <QTimer>
#include <chrono>
#include <cmath>
#include <iostream>
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

	public:
		MPCIntegratedApp(int argc, char **argv, QObject *parent = nullptr) : QObject(parent) {
			// Setup signal handlers
			signal(SIGINT, signalHandler);
			signal(SIGTERM, signalHandler);

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
		}

		~MPCIntegratedApp() {
			delete mpc_planner;
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

		void updateVisualization() {
			if(!g_running) {
				return;
			}

			try {
				// Create main visualization frame (larger to accommodate camera feed)
				m_visualizationFrame = cv::Mat::zeros(800, 1200, CV_8UC3);

				// Left side: Camera feed and processing
				cv::Rect cameraRegion(10, 10, 580, 435);      // Camera feed area
				cv::Rect trajectoryRegion(600, 10, 580, 580); // Trajectory visualization area

				// Get camera frame from CameraStreamer
				getCameraFrame();

				// Draw camera feed
				if(!m_currentCameraFrame.empty()) {
					cv::Mat resizedCamera;
					cv::resize(m_currentCameraFrame, resizedCamera, cv::Size(580, 435));

					// Convert to BGR if needed
					if(resizedCamera.channels() == 1) {
						cv::cvtColor(resizedCamera, resizedCamera, cv::COLOR_GRAY2BGR);
					}

					resizedCamera.copyTo(m_visualizationFrame(cameraRegion));

					// Add camera label
					cv::putText(m_visualizationFrame, "Camera Feed", cv::Point(15, 35),
					            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
				} else {
					// No camera feed available
					cv::rectangle(m_visualizationFrame, cameraRegion, cv::Scalar(50, 50, 50), -1);
					cv::putText(m_visualizationFrame, "No Camera Feed",
					            cv::Point(cameraRegion.x + 200, cameraRegion.y + 200),
					            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
				}

				// Draw processed frame (lane detection result)
				cv::Rect processedRegion(10, 460, 580, 320);
				if(!m_processedFrame.empty()) {
					cv::Mat resizedProcessed;
					cv::resize(m_processedFrame, resizedProcessed, cv::Size(580, 320));

					if(resizedProcessed.channels() == 1) {
						cv::cvtColor(resizedProcessed, resizedProcessed, cv::COLOR_GRAY2BGR);
					}

					resizedProcessed.copyTo(m_visualizationFrame(processedRegion));

					cv::putText(m_visualizationFrame, "Lane Detection", cv::Point(15, 485),
					            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
				} else {
					cv::rectangle(m_visualizationFrame, processedRegion, cv::Scalar(30, 30, 30),
					              -1);
					cv::putText(m_visualizationFrame, "No Lane Detection",
					            cv::Point(processedRegion.x + 180, processedRegion.y + 150),
					            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
				}

				// Right side: Trajectory visualization (existing code)
				drawTrajectoryVisualization(trajectoryRegion);

				// Add system status
				drawSystemStatus();

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

	private:
		void getCameraFrame() {
			try {
				// Try multiple ZeroMQ topics and ports that CameraStreamer might use
				static bool debug_printed = false;
				static bool zmq_tried = false;

				// First try: ZeroMQ connection (only try once to avoid spam)
				if(!zmq_tried) {
					zmq_tried = true;
					try {
						Subscriber camera_sub;
						camera_sub.connect("tcp://localhost:5556");
						camera_sub.subscribe("inference_frame");

						zmq::pollitem_t items[] = {
						    {static_cast<void *>(camera_sub.getSocket()), 0, ZMQ_POLLIN, 0}};
						zmq::poll(items, 1, 50); // Longer timeout for first try

						if(items[0].revents & ZMQ_POLLIN) {
							zmq::message_t message;
							if(camera_sub.getSocket().recv(&message, 0)) {
								std::string received_msg(static_cast<char *>(message.data()),
								                         message.size());

								if(!debug_printed) {
									std::cout << "[DEBUG] ZeroMQ camera connected successfully"
									          << std::endl;
									debug_printed = true;
								}

								// Try multiple topic formats
								std::vector<std::string> topics = {
								    "camera_frame ", "inference_frame ", "raw_frame "};

								for(const auto &topic : topics) {
									if(received_msg.find(topic) == 0) {
										std::string frame_data = received_msg.substr(topic.size());

										// Deserialize frame
										std::vector<uchar> buffer(frame_data.begin(),
										                          frame_data.end());
										m_currentCameraFrame =
										    cv::imdecode(buffer, cv::IMREAD_COLOR);

										if(!m_currentCameraFrame.empty()) {
											m_cameraFrameAvailable = true;
											std::cout << "[DEBUG] ZeroMQ frame received: "
											          << m_currentCameraFrame.cols << "x"
											          << m_currentCameraFrame.rows << std::endl;
											getLaneDetectionFrame();
											return;
										}
									}
								}
							}
						}
					} catch(const std::exception &e) {
						std::cout << "[DEBUG] ZeroMQ connection failed: " << e.what() << std::endl;
					}
				}

				// Fallback: Use OpenCV VideoCapture with safer settings
				static cv::VideoCapture cap;
				static bool cap_initialized = false;
				static int failed_attempts = 0;
				static auto last_attempt = std::chrono::steady_clock::now();

				// Only try to reinitialize every 5 seconds to avoid spam
				auto now = std::chrono::steady_clock::now();
				if(!cap_initialized &&
				   std::chrono::duration_cast<std::chrono::seconds>(now - last_attempt).count() >
				       5) {

					last_attempt = now;
					failed_attempts++;

					if(failed_attempts > 3) {
						// After 3 failed attempts, create a synthetic camera feed
						createSyntheticCameraFeed();
						return;
					}

					std::cout << "[DEBUG] Attempting to initialize camera (attempt "
					          << failed_attempts << "/3)" << std::endl;

					// Try different camera backends in order of preference
					std::vector<int> backends = {cv::CAP_V4L2, cv::CAP_GSTREAMER, cv::CAP_ANY};

					for(int backend : backends) {
						try {
							cap.open(0, backend);
							if(cap.isOpened()) {
								// Set safe parameters
								cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
								cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
								cap.set(cv::CAP_PROP_FPS, 15);
								cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

								// Test if we can actually read a frame
								cv::Mat test_frame;
								if(cap.read(test_frame) && !test_frame.empty()) {
									std::cout
									    << "[DEBUG] Camera initialized successfully with backend "
									    << backend << std::endl;
									cap_initialized = true;
									failed_attempts = 0;
									break;
								} else {
									cap.release();
								}
							}
						} catch(const std::exception &e) {
							std::cout << "[DEBUG] Backend " << backend << " failed: " << e.what()
							          << std::endl;
							cap.release();
						}
					}
				}

				if(cap_initialized && cap.isOpened()) {
					try {
						cv::Mat frame;
						if(cap.read(frame) && !frame.empty()) {
							m_currentCameraFrame = frame.clone();
							m_cameraFrameAvailable = true;

							// Create a simple lane detection simulation for testing
							createTestLaneDetection(frame);
							return;
						} else {
							// Camera disconnected
							cap_initialized = false;
							cap.release();
							std::cout << "[DEBUG] Camera disconnected, will retry..." << std::endl;
						}
					} catch(const std::exception &e) {
						std::cout << "[DEBUG] Camera read error: " << e.what() << std::endl;
						cap_initialized = false;
						cap.release();
					}
				}

				// If we get here, create synthetic feed
				createSyntheticCameraFeed();

			} catch(const std::exception &e) {
				std::cerr << "[getCameraFrame] Error: " << e.what() << std::endl;
				m_cameraFrameAvailable = false;
				createSyntheticCameraFeed();
			}
		}

		void createSyntheticCameraFeed() {
			try {
				// Create a synthetic camera feed for testing when no real camera is available
				static int frame_counter = 0;
				frame_counter++;

				// Create a 640x480 synthetic image
				m_currentCameraFrame = cv::Mat::zeros(480, 640, CV_8UC3);

				// Add some dynamic content
				cv::putText(m_currentCameraFrame, "SYNTHETIC CAMERA FEED", cv::Point(150, 50),
				            cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

				cv::putText(m_currentCameraFrame, "Frame: " + std::to_string(frame_counter),
				            cv::Point(250, 100), cv::FONT_HERSHEY_SIMPLEX, 0.8,
				            cv::Scalar(255, 255, 255), 2);

				// Add animated elements
				int x_offset = (frame_counter * 2) % 640;
				cv::circle(m_currentCameraFrame, cv::Point(x_offset, 200), 10,
				           cv::Scalar(0, 255, 0), -1);

				// Add fake road lines
				cv::line(m_currentCameraFrame, cv::Point(200, 480), cv::Point(250, 200),
				         cv::Scalar(255, 255, 255), 3);
				cv::line(m_currentCameraFrame, cv::Point(390, 200), cv::Point(440, 480),
				         cv::Scalar(255, 255, 255), 3);

				m_cameraFrameAvailable = true;

				// Create test lane detection
				createTestLaneDetection(m_currentCameraFrame);

			} catch(const std::exception &e) {
				std::cerr << "[createSyntheticCameraFeed] Error: " << e.what() << std::endl;
				m_cameraFrameAvailable = false;
			}
		}

		void createTestLaneDetection(const cv::Mat &frame) {
			try {
				// Create a simple test lane detection for debugging
				cv::Mat gray, binary_mask;
				cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

				// Simple threshold to create fake lane detection
				cv::threshold(gray, binary_mask, 100, 255, cv::THRESH_BINARY);

				// Add some fake lane lines that are more visible
				cv::line(binary_mask, cv::Point(frame.cols * 0.25, frame.rows),
				         cv::Point(frame.cols * 0.35, frame.rows * 0.3), cv::Scalar(255), 8);
				cv::line(binary_mask, cv::Point(frame.cols * 0.65, frame.rows * 0.3),
				         cv::Point(frame.cols * 0.75, frame.rows), cv::Scalar(255), 8);

				// Add center dashed line
				for(int y = frame.rows * 0.3; y < frame.rows; y += 40) {
					cv::line(binary_mask, cv::Point(frame.cols * 0.5, y),
					         cv::Point(frame.cols * 0.5, y + 20), cv::Scalar(255), 4);
				}

				if(!binary_mask.empty()) {
					createLaneVisualization(binary_mask);
				}
			} catch(const std::exception &e) {
				std::cerr << "[createTestLaneDetection] Error: " << e.what() << std::endl;
			}
		}

		void drawTrajectoryVisualization(const cv::Rect &region) {
			// Draw trajectory visualization in the specified region
			cv::Mat trajectoryArea = m_visualizationFrame(region);
			trajectoryArea.setTo(cv::Scalar(20, 20, 20));

			// Draw coordinate system
			cv::Point2f center(region.width / 2, region.height / 2);
			cv::line(m_visualizationFrame, cv::Point(region.x + center.x, region.y + 50),
			         cv::Point(region.x + center.x, region.y + region.height - 50),
			         cv::Scalar(100, 100, 100), 1);
			cv::line(m_visualizationFrame, cv::Point(region.x + 50, region.y + center.y),
			         cv::Point(region.x + region.width - 50, region.y + center.y),
			         cv::Scalar(100, 100, 100), 1);

			// Draw current vehicle position (center)
			cv::Point2f vehicle_pos(region.x + center.x, region.y + center.y);
			cv::circle(m_visualizationFrame, vehicle_pos, 8, cv::Scalar(0, 255, 0), -1);
			cv::putText(m_visualizationFrame, "Vehicle",
			            cv::Point(vehicle_pos.x + 10, vehicle_pos.y + 5), cv::FONT_HERSHEY_SIMPLEX,
			            0.5, cv::Scalar(0, 255, 0), 1);

			// Draw recorded waypoints
			if(!recorded_waypoints.empty()) {
				for(size_t i = 0; i < recorded_waypoints.size(); ++i) {
					cv::Point2f wp_pos(
					    region.x + center.x + recorded_waypoints[i].x * 20, // Scale factor
					    region.y + center.y - recorded_waypoints[i].y * 20  // Invert Y and scale
					);

					// Ensure points are within region
					if(wp_pos.x >= region.x && wp_pos.x < region.x + region.width &&
					   wp_pos.y >= region.y && wp_pos.y < region.y + region.height) {
						cv::circle(m_visualizationFrame, wp_pos, 3, cv::Scalar(255, 0, 0), -1);

						// Draw line between consecutive waypoints
						if(i > 0) {
							cv::Point2f prev_wp(
							    region.x + center.x + recorded_waypoints[i - 1].x * 20,
							    region.y + center.y - recorded_waypoints[i - 1].y * 20);
							if(prev_wp.x >= region.x && prev_wp.x < region.x + region.width &&
							   prev_wp.y >= region.y && prev_wp.y < region.y + region.height) {
								cv::line(m_visualizationFrame, prev_wp, wp_pos,
								         cv::Scalar(255, 0, 0), 2);
							}
						}
					}
				}
			}

			// Draw vehicle orientation
			double yaw_vis = current_state.yaw;
			cv::Point2f direction_end(vehicle_pos.x + 30 * cos(yaw_vis),
			                          vehicle_pos.y - 30 * sin(yaw_vis) // Invert Y
			);
			cv::arrowedLine(m_visualizationFrame, vehicle_pos, direction_end,
			                cv::Scalar(0, 255, 255), 2);

			// Add trajectory label
			cv::putText(m_visualizationFrame, "Trajectory Visualization",
			            cv::Point(region.x + 10, region.y + 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
			            cv::Scalar(255, 255, 255), 1);
		}

		void drawSystemStatus() {
			// Draw system status on the right side
			int status_x = 610;
			int status_y = 620;

			cv::putText(m_visualizationFrame,
			            mpc_active ? "Mode: MPC (Autonomous)" : "Mode: Manual",
			            cv::Point(status_x, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.7,
			            mpc_active ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

			cv::putText(m_visualizationFrame,
			            "Waypoints: " + std::to_string(recorded_waypoints.size()),
			            cv::Point(status_x, status_y + 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
			            cv::Scalar(255, 255, 255), 1);

			cv::putText(m_visualizationFrame,
			            "Position: (" + std::to_string(current_state.x).substr(0, 5) + ", " +
			                std::to_string(current_state.y).substr(0, 5) + ")",
			            cv::Point(status_x, status_y + 60), cv::FONT_HERSHEY_SIMPLEX, 0.6,
			            cv::Scalar(255, 255, 255), 1);

			cv::putText(m_visualizationFrame,
			            "Velocity: " + std::to_string(current_state.velocity).substr(0, 5) + " m/s",
			            cv::Point(status_x, status_y + 90), cv::FONT_HERSHEY_SIMPLEX, 0.6,
			            cv::Scalar(255, 255, 255), 1);

			cv::putText(m_visualizationFrame, "MPC Steps: " + std::to_string(step_counter),
			            cv::Point(status_x, status_y + 120), cv::FONT_HERSHEY_SIMPLEX, 0.6,
			            cv::Scalar(255, 255, 255), 1);

			// Enhanced camera status with source indication
			std::string camera_status;
			if(m_cameraFrameAvailable && !m_currentCameraFrame.empty()) {
				camera_status = "Connected (" + std::to_string(m_currentCameraFrame.cols) + "x" +
				                std::to_string(m_currentCameraFrame.rows) + ")";

				// Detect if it's synthetic by checking for specific text
				cv::Mat gray;
				cv::cvtColor(m_currentCameraFrame, gray, cv::COLOR_BGR2GRAY);
				if(cv::mean(gray)[0] < 50) { // Mostly black = synthetic
					camera_status += " [SYNTHETIC]";
				}
			} else {
				camera_status = "Disconnected";
			}

			cv::putText(m_visualizationFrame, std::string("Camera: ") + camera_status,
			            cv::Point(status_x, status_y + 150), cv::FONT_HERSHEY_SIMPLEX, 0.6,
			            m_cameraFrameAvailable ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);

			// Add debug info
			cv::putText(m_visualizationFrame, "ZMQ Port: 5556", cv::Point(status_x, status_y + 180),
			            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1);

			// Instructions
			cv::putText(
			    m_visualizationFrame, "Commands: m(mode) r(record) c(clear) s(status) q(quit)",
			    cv::Point(10, 790), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
		}

		void setupVisualization() {
			// Create larger OpenCV window for camera feed
			cv::namedWindow("MPC Integrated System", cv::WINDOW_AUTOSIZE);

			// Setup visualization timer
			m_visualizationTimer = new QTimer(this);
			connect(m_visualizationTimer, &QTimer::timeout, this,
			        &MPCIntegratedApp::updateVisualization);
			m_visualizationTimer->start(50); // 20 FPS
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
};

int main(int argc, char *argv[]) {
	QApplication app(argc, argv);

	std::cout << "Iniciando sistema integrado MPC + Car Controls..." << std::endl;

	try {
		MPCIntegratedApp integrated_app(argc, argv);

		std::cout << "Sistema pronto! Use os comandos no terminal ou na janela OpenCV."
		          << std::endl;
		std::cout << "Pressione Ctrl+C ou 'q' para sair." << std::endl;

		// Setup periodic check for global running flag
		QTimer *shutdown_timer = new QTimer();
		QObject::connect(shutdown_timer, &QTimer::timeout, [&]() {
			if(!g_running) {
				cv::destroyAllWindows();
				app.quit();
			}
		});
		shutdown_timer->start(100);

		int result = app.exec();

		// Cleanup
		cv::destroyAllWindows();
		return result;

	} catch(const std::exception &e) {
		std::cerr << "Erro: " << e.what() << std::endl;
		cv::destroyAllWindows();
		return 1;
	}
}

#include "main.moc"
