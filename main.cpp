#include "ZeroMQ/Subscriber.hpp"
#include "car_controls/includes/CommonTypes.hpp"
#include "car_controls/includes/ControlsManager.hpp"
#include "car_controls/includes/MPCOptimizer.hpp"
#include "car_controls/includes/MPCPlanner.hpp"
#include "car_controls/includes/inference/TensorRTInferencer.hpp"
#include <QApplication>
#include <QTimer>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <signal.h>
#include <thread>

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

// Signal handler for Ctrl+C
void signalHandler(int signum) {
	std::cout << "\nReceived signal " << signum << ". Shutting down gracefully..." << std::endl;
	g_running = false;

	// Give some time for cleanup
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	// Force quit if Qt is running
	if(QApplication::instance()) {
		QApplication::quit();
	}
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

		// Add TensorRT inferencer for local lane detection
		std::unique_ptr<TensorRTInferencer> m_inferencer;

		// MPC trajectory prediction
		std::vector<Point2D> m_predictedTrajectory;
		cv::Mat m_currentLaneMask;

	public:
		MPCIntegratedApp(int argc, char **argv, QObject *parent = nullptr)
		    : QObject(parent), controls_manager(nullptr), mpc_planner(nullptr), mpc_timer(nullptr) {
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
				std::cout << "- Pressione ENTER ou 'm' para alternar Manual/MPC" << std::endl;
				std::cout << "- Pressione 'r' para iniciar/parar gravação" << std::endl;
				std::cout << "- Pressione 'd' para ativar/desativar logs detalhados do MPC"
				          << std::endl;
				std::cout << "- Pressione 's' para mostrar status do sistema" << std::endl;
				std::cout << "- Pressione 'q' para sair" << std::endl;

				// Conectar stdin para comandos
				setupKeyboardInput();

				// Setup visualization
				setupVisualization();

				// Initialize TensorRT inferencer for lane detection
				try {
					m_inferencer = std::make_unique<TensorRTInferencer>(
					    "/home/jetson/models/lane-detection/model.engine");
					std::cout << "[MPCIntegratedApp] TensorRT inferencer initialized" << std::endl;
				} catch(const std::exception &e) {
					std::cerr << "[MPCIntegratedApp] Failed to initialize TensorRT inferencer: "
					          << e.what() << std::endl;
					// Continue without local inference - will use remote inference
				};
			} catch(const std::exception &e) {
				std::cerr << "[MPCIntegratedApp] Initialization error: " << e.what() << std::endl;
				throw;
			}
		}

		~MPCIntegratedApp() {
			std::cout << "[~MPCIntegratedApp] Starting cleanup..." << std::endl;

			try {
				// Stop all timers first
				if(mpc_timer) {
					mpc_timer->stop();
					mpc_timer = nullptr;
				}

				if(m_visualizationTimer) {
					m_visualizationTimer->stop();
					m_visualizationTimer = nullptr;
				}

				// Close OpenCV windows
				cv::destroyAllWindows();

				// Reset smart pointers (this will call destructors)
				m_laneDetectionSubscriber.reset();
				m_inferencer.reset();

				// Delete MPC planner
				if(mpc_planner) {
					delete mpc_planner;
					mpc_planner = nullptr;
				}

				// controls_manager is QObject child, will be deleted automatically
				// but we can set it to nullptr for safety
				controls_manager = nullptr;

				std::cout << "[~MPCIntegratedApp] Cleanup complete" << std::endl;

			} catch(const std::exception &e) {
				std::cerr << "[~MPCIntegratedApp] Error during cleanup: " << e.what() << std::endl;
			} catch(...) {
				std::cerr << "[~MPCIntegratedApp] Unknown error during cleanup" << std::endl;
			}
		}

	private slots:
		void runMPCStep() {
			if(!mpc_active || !g_running) {
				return;
			}

			try {
				// Use MPC predicted trajectory if available, otherwise fall back to recorded
				// waypoints
				std::vector<Point2D> reference_trajectory;

				if(!m_predictedTrajectory.empty()) {
					reference_trajectory = m_predictedTrajectory;
				} else if(recorded_waypoints.size() >= 3) {
					reference_trajectory = recorded_waypoints;
				} else {
					std::cout << "[MPC] No trajectory available for control" << std::endl;
					return;
				}

				// Usar trajetória como referência
				LaneInfo lane_info(0.0, 0.0);
				ControlCommand control =
				    mpc_planner->plan(current_state, reference_trajectory, &lane_info);

				// Printar a trajetória prevista do MPC
				const auto &predicted_traj = mpc_planner->getPredictedTrajectory();
				if(!predicted_traj.empty()) {
					std::cout << "[MPC Predicted Trajectory] ";
					for(const auto &pt : predicted_traj) {
						std::cout << "(" << pt.x << "," << pt.y << ") ";
					}
					std::cout << std::endl;
				}

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
				if(step_counter % 10 == 0) { // More frequent logging for MPC data
					std::cout << "\n=== MPC CONTROL STEP " << step_counter << " ===" << std::endl;
					std::cout << "[MPC] Vehicle State:" << std::endl;
					std::cout << "  Position: (" << std::fixed << std::setprecision(3)
					          << current_state.x << ", " << current_state.y << ")" << std::endl;
					std::cout << "  Velocity: " << current_state.velocity << " m/s" << std::endl;
					std::cout << "  Yaw: " << current_state.yaw * 180 / M_PI << " degrees"
					          << std::endl;

					std::cout << "[MPC] Control Commands:" << std::endl;
					std::cout << "  Throttle: " << std::fixed << std::setprecision(3)
					          << control.throttle << " (Target Speed: " << mpc_speed << "%)"
					          << std::endl;
					std::cout << "  Steering: " << control.steer << " rad (Target: " << mpc_steering
					          << " degrees)" << std::endl;

					std::cout << "[MPC] Reference Trajectory:" << std::endl;
					if(!m_predictedTrajectory.empty()) {
						std::cout << "  Using LANE DETECTION trajectory ("
						          << m_predictedTrajectory.size() << " points)" << std::endl;
						// Show next 3 target points
						std::cout << "  Next targets:" << std::endl;
						for(size_t i = 0; i < std::min(size_t(3), m_predictedTrajectory.size());
						    i++) {
							std::cout << "    [" << i << "] x=" << std::fixed
							          << std::setprecision(3) << m_predictedTrajectory[i].x
							          << "m, y=" << m_predictedTrajectory[i].y << "m" << std::endl;
						}
					} else if(!recorded_waypoints.empty()) {
						std::cout << "  Using RECORDED trajectory (" << recorded_waypoints.size()
						          << " points)" << std::endl;
					}

					// Show MPC internal prediction
					const auto &predicted_traj = mpc_planner->getPredictedTrajectory();
					if(!predicted_traj.empty()) {
						std::cout << "[MPC] Internal Prediction (" << predicted_traj.size()
						          << " points):" << std::endl;
						for(size_t i = 0; i < std::min(size_t(3), predicted_traj.size()); i++) {
							std::cout << "    [" << i << "] x=" << std::fixed
							          << std::setprecision(3) << predicted_traj[i].x
							          << "m, y=" << predicted_traj[i].y << "m" << std::endl;
						}
					}
					std::cout << "================================\n" << std::endl;
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
			} else if(input == "d") {
				// Toggle detailed MPC logs
				static bool detailed_logs = false;
				detailed_logs = !detailed_logs;
				std::cout << "Logs detalhados do MPC: "
				          << (detailed_logs ? "ATIVADOS" : "DESATIVADOS") << std::endl;
			} else if(input == "q") {
				// Sair
				QApplication::quit();
			} else {
				std::cout << "Comando não reconhecido. Use: m(modo), r(gravar), c(limpar), "
				             "d(debug), s(status), q(sair)"
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
				    std::chrono::duration_cast<std::chrono::seconds>(now - last_zmq_attempt)
				            .count() > 2)) {

					first_connection_attempt = false;
					last_zmq_attempt = now;

					try {
						// Try raw camera frames first (from port 5558)
						Subscriber camera_sub;
						camera_sub.connect("tcp://localhost:5558");
						camera_sub.subscribe("camera_frame");

						zmq::pollitem_t items[] = {
						    {static_cast<void *>(camera_sub.getSocket()), 0, ZMQ_POLLIN, 0}};
						zmq::poll(items, 1, 200); // Longer timeout for connection attempt

						if(items[0].revents & ZMQ_POLLIN) {
							zmq::message_t message;
							if(camera_sub.getSocket().recv(&message, ZMQ_DONTWAIT)) {
								std::string received_msg(static_cast<char *>(message.data()),
								                         message.size());

								if(received_msg.find("camera_frame ") == 0) {
									std::string frame_data =
									    received_msg.substr(13); // "camera_frame ".length()

									std::vector<uchar> buffer(frame_data.begin(), frame_data.end());
									cv::Mat decoded_frame = cv::imdecode(buffer, cv::IMREAD_COLOR);

									if(!decoded_frame.empty()) {
										m_currentCameraFrame = decoded_frame.clone();
										m_cameraFrameAvailable = true;
										zmq_connection_failed = false;

										// Execute local lane detection inference
										executeLocalInference(decoded_frame);
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
								std::string received_msg(static_cast<char *>(message.data()),
								                         message.size());

								if(received_msg.find("inference_frame ") == 0) {
									std::cout << "[DEBUG] Found inference data from CameraStreamer"
									          << std::endl;
									zmq_connection_failed = false;
								}
							}
						}

						if(!zmq_connection_failed) {
							std::cout << "[DEBUG] CameraStreamer is running but no frames yet"
							          << std::endl;
						}

					} catch(const std::exception &e) {
						if(!zmq_connection_failed) {
							std::cout << "[DEBUG] ZeroMQ connection failed: " << e.what()
							          << std::endl;
							zmq_connection_failed = true;
						}
					}
				}

				// If ZeroMQ is not working, use synthetic feed
				if(zmq_connection_failed || !m_cameraFrameAvailable) {
					// TODO: Implement createSyntheticCameraFeed();
					// std::cerr << "[getCameraFrame] Warning: Camera feed not available" <<
					// std::endl;
				}

			} catch(const std::exception &e) {
				// std::cerr << "[getCameraFrame] Error: " << e.what() << std::endl;
				// TODO: Implement createSyntheticCameraFeed();
				// std::cerr << "[getCameraFrame] Warning: Falling back to synthetic feed not
				// implemented" << std::endl;
			}
		}

		void executeLocalInference(const cv::Mat &frame) {
			if(!m_inferencer || frame.empty()) {
				// Fallback to remote inference if local inferencer not available
				getLaneDetectionFrame();
				return;
			}

			try {
				// Execute local inference (same logic as lane_detection_video_test.cpp)
				m_inferencer->doInference(frame);
				cv::Mat mask = m_inferencer->getLastMask();

				if(!mask.empty()) {
					// Process the mask for MPC trajectory generation
					createLaneVisualization(mask);

					// Log lane detection data
					static int inference_counter = 0;
					if(++inference_counter % 30 == 0) { // Log every 30th inference
						std::cout << "\n=== LANE DETECTION DATA ===" << std::endl;
						std::cout << "[LaneDetection] Mask size: " << mask.cols << "x" << mask.rows
						          << std::endl;

						// Count white pixels (detected lanes)
						int white_pixels = cv::countNonZero(mask);
						double lane_coverage = (white_pixels * 100.0) / (mask.cols * mask.rows);
						std::cout << "[LaneDetection] Lane pixels: " << white_pixels << " ("
						          << std::fixed << std::setprecision(1) << lane_coverage
						          << "% coverage)" << std::endl;

						if(!m_predictedTrajectory.empty()) {
							std::cout << "[LaneDetection] Generated "
							          << m_predictedTrajectory.size()
							          << " trajectory points for MPC" << std::endl;
						}
						std::cout << "===========================\n" << std::endl;
					}
				} else {
					std::cerr << "[executeLocalInference] Warning: Empty mask from inference"
					          << std::endl;
				}
			} catch(const std::exception &e) {
				std::cerr << "[executeLocalInference] Error: " << e.what() << std::endl;
				// Fallback to remote inference
				getLaneDetectionFrame();
			}
		}

		void getLaneDetectionFrame() {
			if(!m_laneDetectionSubscriber) {
				// std::cout << "[DEBUG] Lane detection subscriber not initialized" << std::endl;
				return;
			}

			try {
				zmq::pollitem_t items[] = {
				    {static_cast<void *>(m_laneDetectionSubscriber->getSocket()), 0, ZMQ_POLLIN,
				     0}};
				zmq::poll(items, 1, 5); // Short timeout to avoid blocking

				if(items[0].revents & ZMQ_POLLIN) {
					zmq::message_t message;
					if(m_laneDetectionSubscriber->getSocket().recv(&message, ZMQ_DONTWAIT)) {
						std::string received_msg(static_cast<char *>(message.data()),
						                         message.size());

						if(received_msg.find("inference_frame ") == 0) {
							std::string mask_data =
							    received_msg.substr(16); // "inference_frame ".length()

							std::vector<uchar> buffer(mask_data.begin(), mask_data.end());
							cv::Mat binary_mask = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);

							if(!binary_mask.empty()) {
								createLaneVisualization(binary_mask);
								return;
							}
						}
					}
				} else {
					// std::cout << "[DEBUG] No lane detection data available" << std::endl;
				}

			} catch(const std::exception &e) {
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
				cv::Rect cameraRegion(10, 10, 420, 240);      // Smaller camera feed
				cv::Rect processedRegion(10, 260, 420, 240);  // Smaller processed view
				cv::Rect trajectoryRegion(440, 10, 450, 450); // Trajectory area

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
					cv::rectangle(m_visualizationFrame, processedRegion, cv::Scalar(30, 30, 30),
					              -1);
					cv::putText(m_visualizationFrame, "No Lane Detection",
					            cv::Point(processedRegion.x + 120, processedRegion.y + 120),
					            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
				} // Right side: Trajectory visualization
				drawTrajectoryVisualization(trajectoryRegion);

				// Add compact system status
				drawCompactSystemStatus();

				// Display the frame
				cv::imshow("MPC Integrated System", m_visualizationFrame);

				// Handle OpenCV events and check for exit
				int key = cv::waitKey(1) & 0xFF;
				if(key == 'q' || key == 27) { // 'q' or ESC
					g_running = false;
					cv::destroyAllWindows();
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

			cv::putText(m_visualizationFrame, mpc_active ? "Mode: MPC" : "Mode: Manual",
			            cv::Point(status_x, status_y), cv::FONT_HERSHEY_SIMPLEX, 0.5,
			            mpc_active ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 1);

			cv::putText(m_visualizationFrame, "WP: " + std::to_string(recorded_waypoints.size()),
			            cv::Point(status_x, status_y + line_height), cv::FONT_HERSHEY_SIMPLEX, 0.4,
			            cv::Scalar(255, 255, 255), 1);

			cv::putText(m_visualizationFrame,
			            "Pos: (" + std::to_string(current_state.x).substr(0, 4) + "," +
			                std::to_string(current_state.y).substr(0, 4) + ")",
			            cv::Point(status_x, status_y + 2 * line_height), cv::FONT_HERSHEY_SIMPLEX,
			            0.4, cv::Scalar(255, 255, 255), 1);

			cv::putText(m_visualizationFrame,
			            "Vel: " + std::to_string(current_state.velocity).substr(0, 4),
			            cv::Point(status_x, status_y + 3 * line_height), cv::FONT_HERSHEY_SIMPLEX,
			            0.4, cv::Scalar(255, 255, 255), 1);

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

			cv::putText(m_visualizationFrame, camera_status, cv::Point(status_x + 150, status_y),
			            cv::FONT_HERSHEY_SIMPLEX, 0.4,
			            m_cameraFrameAvailable ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 0),
			            1);

			// Compact instructions
			cv::putText(m_visualizationFrame, "m:mode r:rec c:clear s:status q:quit",
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
				std::cout << "[DEBUG] Lane detection subscriber initialized (fallback mode)"
				          << std::endl;
			} catch(const std::exception &e) {
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
				// Check if we have either recorded waypoints or live lane detection
				if(recorded_waypoints.size() < 3 && m_predictedTrajectory.empty()) {
					std::cout << "Erro: Precisa de waypoints gravados OU detecção de pistas ativa!"
					          << std::endl;
					mpc_active = false;
					return;
				}

				// Mudar para modo autônomo
				controls_manager->setMode(DrivingMode::Automatic);
				mpc_timer->start(50); // 20 Hz

				if(!m_predictedTrajectory.empty()) {
					std::cout << "Modo MPC ATIVADO - Seguindo detecção de pistas com "
					          << m_predictedTrajectory.size() << " pontos" << std::endl;
				} else {
					std::cout << "Modo MPC ATIVADO - Seguindo trajetória gravada com "
					          << recorded_waypoints.size() << " waypoints" << std::endl;
				}
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
			std::cout << "\n========== SYSTEM STATUS ==========" << std::endl;
			std::cout << "Modo: " << (mpc_active ? "MPC (Autônomo)" : "Manual") << std::endl;
			std::cout << "MPC Steps executados: " << step_counter << std::endl;

			std::cout << "\n--- Trajetórias ---" << std::endl;
			std::cout << "Waypoints gravados: " << recorded_waypoints.size() << std::endl;
			std::cout << "Trajetória MPC (lane detection): " << m_predictedTrajectory.size()
			          << " pontos" << std::endl;

			std::cout << "\n--- Estado do Veículo ---" << std::endl;
			std::cout << "Posição atual: (" << std::fixed << std::setprecision(2) << current_state.x
			          << ", " << current_state.y << ")" << std::endl;
			std::cout << "Velocidade: " << current_state.velocity << " m/s" << std::endl;
			std::cout << "Orientação: " << current_state.yaw * 180 / M_PI << " graus" << std::endl;

			std::cout << "\n--- Sistema de Visão ---" << std::endl;
			std::cout << "Camera frame disponível: " << (m_cameraFrameAvailable ? "SIM" : "NÃO")
			          << std::endl;
			std::cout << "TensorRT inferencer: " << (m_inferencer ? "ATIVO" : "INATIVO")
			          << std::endl;
			std::cout << "Processed frame: " << (!m_processedFrame.empty() ? "SIM" : "NÃO")
			          << std::endl;

			if(!m_currentLaneMask.empty()) {
				int white_pixels = cv::countNonZero(m_currentLaneMask);
				double coverage =
				    (white_pixels * 100.0) / (m_currentLaneMask.cols * m_currentLaneMask.rows);
				std::cout << "Lane mask: " << m_currentLaneMask.cols << "x"
				          << m_currentLaneMask.rows << " (" << std::fixed << std::setprecision(1)
				          << coverage << "% cobertura)" << std::endl;
			}

			std::cout << "===================================\n" << std::endl;
		}

		void createLaneVisualization(const cv::Mat &binary_mask) {
			try {
				if(binary_mask.empty()) {
					return;
				}

				// Store current lane mask for MPC trajectory prediction
				m_currentLaneMask = binary_mask.clone();

				// Generate MPC trajectory based on lane detection
				generateMPCTrajectory(binary_mask);

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
						if(i % 3 == 0)
							color = cv::Scalar(0, 255, 0); // Green
						else if(i % 3 == 1)
							color = cv::Scalar(255, 0, 0); // Blue
						else
							color = cv::Scalar(0, 255, 255); // Yellow

						cv::drawContours(lane_overlay, contours, static_cast<int>(i), color, 2);
					}
				}

				// Blend the original mask with colored overlay
				cv::addWeighted(colored_lanes, 0.7, lane_overlay, 0.3, 0, m_processedFrame);

				// Add lane detection info text
				cv::putText(m_processedFrame, "Lanes: " + std::to_string(contours.size()),
				            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
				            cv::Scalar(255, 255, 255), 2);

			} catch(const std::exception &e) {
				std::cerr << "[createLaneVisualization] Error: " << e.what() << std::endl;
				// Create a simple error visualization
				m_processedFrame = cv::Mat::zeros(binary_mask.size(), CV_8UC3);
				cv::putText(m_processedFrame, "Lane Processing Error",
				            cv::Point(50, binary_mask.rows / 2), cv::FONT_HERSHEY_SIMPLEX, 0.8,
				            cv::Scalar(0, 0, 255), 2);
			}
		}

		void generateMPCTrajectory(const cv::Mat &binary_mask) {
			try {
				m_predictedTrajectory.clear();

				if(binary_mask.empty()) {
					return;
				}

				int height = binary_mask.rows;
				int width = binary_mask.cols;

				// Extract centerline points from lane mask
				std::vector<cv::Point2f> centerline_points;

				// Scan from bottom to top of the image
				for(int y = height - 1; y >= height / 4;
				    y -= 10) { // Skip top quarter, scan every 10 pixels
					std::vector<int> lane_pixels;

					// Find all white pixels in this row
					for(int x = 0; x < width; x++) {
						if(binary_mask.at<uchar>(y, x) > 128) {
							lane_pixels.push_back(x);
						}
					}

					if(lane_pixels.size() >= 2) {
						// Find left and right boundaries
						int left_boundary =
						    *std::min_element(lane_pixels.begin(), lane_pixels.end());
						int right_boundary =
						    *std::max_element(lane_pixels.begin(), lane_pixels.end());

						// Calculate centerline
						int center_x = (left_boundary + right_boundary) / 2;
						centerline_points.push_back(cv::Point2f(center_x, y));

					} else if(!lane_pixels.empty()) {
						// Single lane detected, use it as reference
						int avg_x = std::accumulate(lane_pixels.begin(), lane_pixels.end(), 0) /
						            lane_pixels.size();
						centerline_points.push_back(cv::Point2f(avg_x, y));
					}
				}

				// Convert image coordinates to world coordinates and create trajectory
				if(centerline_points.size() >= 3) {
					// Simple coordinate transformation (adjust scale as needed)
					double pixel_to_meter = 0.01; // 1 pixel = 1cm
					double image_center_x = width / 2.0;

					for(const auto &point : centerline_points) {
						// Convert to vehicle-centric coordinates
						double world_x = (height - point.y) * pixel_to_meter; // Forward distance
						double world_y =
						    (point.x - image_center_x) * pixel_to_meter; // Lateral offset

						m_predictedTrajectory.emplace_back(world_x, world_y);
					}

					// Smooth the trajectory using simple moving average
					smoothTrajectory(m_predictedTrajectory);

					// Enhanced logging for MPC trajectory data
					static int log_counter = 0;
					if(++log_counter % 20 == 0) { // Log every 20th trajectory
						std::cout << "\n=== MPC TRAJECTORY DATA ===" << std::endl;
						std::cout << "[MPC] Generated trajectory with "
						          << m_predictedTrajectory.size() << " points" << std::endl;
						std::cout << "[MPC] Lane pixels detected: " << centerline_points.size()
						          << " centerline points" << std::endl;

						// Show first few trajectory points
						std::cout << "[MPC] Trajectory preview (first 5 points):" << std::endl;
						for(size_t i = 0; i < std::min(size_t(5), m_predictedTrajectory.size());
						    i++) {
							std::cout << "  Point " << i << ": x=" << std::fixed
							          << std::setprecision(3) << m_predictedTrajectory[i].x
							          << "m, y=" << m_predictedTrajectory[i].y << "m" << std::endl;
						}

						// Show trajectory curvature analysis
						if(m_predictedTrajectory.size() >= 3) {
							double total_curvature = 0.0;
							for(size_t i = 1; i < m_predictedTrajectory.size() - 1; i++) {
								double dx1 =
								    m_predictedTrajectory[i].x - m_predictedTrajectory[i - 1].x;
								double dy1 =
								    m_predictedTrajectory[i].y - m_predictedTrajectory[i - 1].y;
								double dx2 =
								    m_predictedTrajectory[i + 1].x - m_predictedTrajectory[i].x;
								double dy2 =
								    m_predictedTrajectory[i + 1].y - m_predictedTrajectory[i].y;

								double angle1 = atan2(dy1, dx1);
								double angle2 = atan2(dy2, dx2);
								double curvature = angle2 - angle1;

								// Normalize angle difference
								while(curvature > M_PI)
									curvature -= 2.0 * M_PI;
								while(curvature < -M_PI)
									curvature += 2.0 * M_PI;

								total_curvature += abs(curvature);
							}

							double avg_curvature =
							    total_curvature / (m_predictedTrajectory.size() - 2);
							std::cout << "[MPC] Average curvature: " << std::fixed
							          << std::setprecision(4) << avg_curvature << " rad/point"
							          << std::endl;

							// Classify trajectory type
							if(avg_curvature < 0.1) {
								std::cout << "[MPC] Trajectory type: STRAIGHT" << std::endl;
							} else if(avg_curvature < 0.3) {
								std::cout << "[MPC] Trajectory type: GENTLE_CURVE" << std::endl;
							} else {
								std::cout << "[MPC] Trajectory type: SHARP_CURVE" << std::endl;
							}
						}
						std::cout << "========================\n" << std::endl;
					}
				}

			} catch(const std::exception &e) {
				std::cerr << "[generateMPCTrajectory] Error: " << e.what() << std::endl;
			}
		}

		void smoothTrajectory(std::vector<Point2D> &trajectory) {
			if(trajectory.size() < 3)
				return;

			std::vector<Point2D> smoothed = trajectory;
			int window = 3;

			for(size_t i = window; i < trajectory.size() - window; i++) {
				double sum_x = 0, sum_y = 0;
				for(int j = -window; j <= window; j++) {
					sum_x += trajectory[i + j].x;
					sum_y += trajectory[i + j].y;
				}
				smoothed[i].x = sum_x / (2 * window + 1);
				smoothed[i].y = sum_y / (2 * window + 1);
			}

			trajectory = smoothed;
		}

		void drawTrajectoryVisualization(const cv::Rect &region) {
			try {
				// Fill background
				cv::rectangle(m_visualizationFrame, region, cv::Scalar(20, 20, 20), -1);

				// Add title
				cv::putText(m_visualizationFrame, "MPC Trajectory",
				            cv::Point(region.x + 10, region.y + 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
				            cv::Scalar(255, 255, 255), 2);

				// Draw coordinate system
				int center_x = region.x + region.width / 2;
				int center_y = region.y + region.height - 50; // Bottom of region

				// Draw axes
				cv::line(m_visualizationFrame, cv::Point(center_x, region.y + 40),
				         cv::Point(center_x, region.y + region.height - 20),
				         cv::Scalar(100, 100, 100), 1); // Y-axis (forward)

				cv::line(m_visualizationFrame, cv::Point(region.x + 20, center_y),
				         cv::Point(region.x + region.width - 20, center_y),
				         cv::Scalar(100, 100, 100), 1); // X-axis (lateral)

				// Draw vehicle position (red dot at bottom center)
				cv::circle(m_visualizationFrame, cv::Point(center_x, center_y), 5,
				           cv::Scalar(0, 0, 255), -1);
				cv::putText(m_visualizationFrame, "Car", cv::Point(center_x - 15, center_y + 20),
				            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 1);

				// Draw recorded waypoints (if any) in blue
				if(!recorded_waypoints.empty()) {
					cv::Scalar waypoint_color(255, 100, 0); // Orange
					for(size_t i = 1; i < recorded_waypoints.size(); i++) {
						cv::Point p1 =
						    worldToScreen(recorded_waypoints[i - 1], region, center_x, center_y);
						cv::Point p2 =
						    worldToScreen(recorded_waypoints[i], region, center_x, center_y);

						if(isPointInRegion(p1, region) && isPointInRegion(p2, region)) {
							cv::line(m_visualizationFrame, p1, p2, waypoint_color, 2);
						}
					}

					// Add legend
					cv::putText(m_visualizationFrame, "Recorded Path",
					            cv::Point(region.x + 10, region.y + 45), cv::FONT_HERSHEY_SIMPLEX,
					            0.4, waypoint_color, 1);
				}

				// Draw MPC predicted trajectory (green)
				if(!m_predictedTrajectory.empty()) {
					cv::Scalar trajectory_color(0, 255, 0); // Green

					// Draw trajectory points and lines
					for(size_t i = 1; i < m_predictedTrajectory.size(); i++) {
						cv::Point p1 =
						    worldToScreen(m_predictedTrajectory[i - 1], region, center_x, center_y);
						cv::Point p2 =
						    worldToScreen(m_predictedTrajectory[i], region, center_x, center_y);

						if(isPointInRegion(p1, region) && isPointInRegion(p2, region)) {
							cv::line(m_visualizationFrame, p1, p2, trajectory_color, 3);
							cv::circle(m_visualizationFrame, p2, 2, trajectory_color, -1);
						}
					}

					// Add legend
					cv::putText(m_visualizationFrame, "MPC Prediction",
					            cv::Point(region.x + 10, region.y + 65), cv::FONT_HERSHEY_SIMPLEX,
					            0.4, trajectory_color, 1);

					// Show trajectory info
					cv::putText(m_visualizationFrame,
					            "Points: " + std::to_string(m_predictedTrajectory.size()),
					            cv::Point(region.x + 10, region.y + 85), cv::FONT_HERSHEY_SIMPLEX,
					            0.4, cv::Scalar(200, 200, 200), 1);
				}

				// Draw current vehicle state
				cv::putText(m_visualizationFrame,
				            "Vel: " + std::to_string(current_state.velocity).substr(0, 4) + " m/s",
				            cv::Point(region.x + 10, region.y + region.height - 40),
				            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

				cv::putText(m_visualizationFrame,
				            "Yaw: " + std::to_string(current_state.yaw * 180 / M_PI).substr(0, 5) +
				                "°",
				            cv::Point(region.x + 10, region.y + region.height - 20),
				            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

			} catch(const std::exception &e) {
				std::cerr << "[drawTrajectoryVisualization] Error: " << e.what() << std::endl;
				// Draw error message
				cv::putText(m_visualizationFrame, "Trajectory Error",
				            cv::Point(region.x + 50, region.y + region.height / 2),
				            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
			}
		}

		cv::Point worldToScreen(const Point2D &world_point, const cv::Rect &region, int center_x,
		                        int center_y) {
			// Scale factor: 1 meter = 100 pixels in visualization
			double scale = 100.0;

			// Convert world coordinates to screen coordinates
			int screen_x = center_x + static_cast<int>(world_point.y * scale); // Lateral
			int screen_y =
			    center_y - static_cast<int>(world_point.x * scale); // Forward (inverted Y)

			return cv::Point(screen_x, screen_y);
		}

		bool isPointInRegion(const cv::Point &point, const cv::Rect &region) {
			return point.x >= region.x && point.x <= region.x + region.width &&
			       point.y >= region.y && point.y <= region.y + region.height;
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
		} catch(const std::exception &e) {
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

		// Graceful cleanup sequence
		std::cout << "[main] Starting application cleanup..." << std::endl;

		try {
			// 1. Reset the main application object first
			integrated_app.reset();

			// 2. Close all OpenCV windows
			cv::destroyAllWindows();

			// 3. Give time for threads to finish
			std::this_thread::sleep_for(std::chrono::milliseconds(200));

			// 4. Final cleanup of singletons (with error handling)
			try {
				Publisher::destroyAll();
			} catch(const std::exception &e) {
				std::cerr << "[main] Warning during Publisher cleanup: " << e.what() << std::endl;
			}

			std::cout << "[main] Application cleanup complete" << std::endl;
		} catch(const std::exception &e) {
			std::cerr << "[main] Error during cleanup: " << e.what() << std::endl;
		}

		return result;

	} catch(const std::exception &e) {
		std::cerr << "Erro: " << e.what() << std::endl;
		try {
			cv::destroyAllWindows();
			Publisher::destroyAll();
		} catch(...) {
			std::cerr << "[main] Additional errors during exception cleanup" << std::endl;
		}
		return 1;
	}
}

#include "main.moc"
