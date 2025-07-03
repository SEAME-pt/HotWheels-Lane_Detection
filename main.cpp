#include "Debugger.hpp"
#include "Subscriber.hpp"
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

// === CONFIGURATION MACROS FOR EASY ADJUSTMENT ===
//
// Para alterar rapidamente os valores de velocidade constante, modifique as macros abaixo:
//
// DEFAULT_CONSTANT_SPEED_KMH: Velocidade alvo em km/h (será convertida automaticamente para m/s)
//   - Valores seguros: 1 a 5 km/h
//   - Valor padrão: 2 km/h (velocidade normal para testes)
//   - Para testes mais rápidos: 4 km/h
//
// DEFAULT_CONSTANT_THROTTLE: Valor do throttle (0.0 a 1.0) para velocidade constante
//   - Valores seguros: 0.1 a 0.5 (motores são robustos)
//   - Valor padrão: 0.15 (15% de potência)
//   - Para mais velocidade: 0.3 (30% de potência)
//   - NOTA: Throttle NÃO afeta o servo - apenas os motores das rodas
//
#define DEFAULT_CONSTANT_SPEED_KMH 2.0 // km/h - Velocidade normal dos motores
#define DEFAULT_CONSTANT_SPEED (DEFAULT_CONSTANT_SPEED_KMH / 3.6) // Auto conversion to m/s
#define DEFAULT_CONSTANT_THROTTLE 0.15            // Throttle normal (15%) - motores são robustos
#define MIN_SAFE_SPEED_KMH 0.2                    // km/h - Minimum safe speed
#define MAX_SAFE_SPEED_KMH 2.0                    // km/h - REDUZIDO para proteger servo frágil
#define MIN_SAFE_SPEED (MIN_SAFE_SPEED_KMH / 3.6) // Auto conversion to m/s
#define MAX_SAFE_SPEED (MAX_SAFE_SPEED_KMH / 3.6) // Auto conversion to m/s

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

// Global pointer to controls manager for emergency motor stop
static ControlsManager *g_emergency_controls = nullptr;

// Emergency motor stop function
void emergencyMotorStop() {
	if(g_emergency_controls) {
		try {
			std::cout << "[EMERGENCY] Stopping all motors..." << std::endl;

			// Use BOTH emergency stop methods for maximum safety
			g_emergency_controls->emergencyStop();      // Critical emergency stop
			g_emergency_controls->emergencyMotorStop(); // Motor-specific stop

			std::cout << "[EMERGENCY] Motors stopped successfully" << std::endl;

		} catch(const std::exception &e) {
			ERROR_STREAM("Main") << "[EMERGENCY] Error stopping motors: " << e.what();
			// Try direct hardware stop as last resort
			try {
				std::cout << "[EMERGENCY] Attempting direct motor stop..." << std::endl;
				g_emergency_controls->emergencyMotorStop();
			} catch(...) {
				ERROR_LOG("Main", "[EMERGENCY] CRITICAL: All motor stop attempts failed!");
			}
		} catch(...) {
			ERROR_LOG("Main", "[EMERGENCY] Unknown error stopping motors");
		}
	} else {
		std::cout << "[EMERGENCY] No controls manager available for motor stop" << std::endl;
	}
}

// Signal handler for Ctrl+C
void signalHandler(int signum) {
	std::cout << "\nReceived signal " << signum << ". Shutting down gracefully..." << std::endl;
	g_running = false;

	// CRITICAL: Stop motors immediately on signal
	emergencyMotorStop();

	// Avoid CUDA operations during signal handling - they can cause core dumps
	// Just set the flag and let the main cleanup handle CUDA resources

	// Give minimal time for Qt to process the shutdown
	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	// Signal Qt to quit if available
	if(QApplication::instance()) {
		QApplication::quit();
	}

	// Give Qt time to shutdown properly
	std::this_thread::sleep_for(std::chrono::milliseconds(200));

	// If we're still here after reasonable time, force exit to avoid hangs
	std::cout << "[SignalHandler] Force exit to avoid CUDA/TensorRT issues" << std::endl;
	std::_Exit(0); // Use _Exit to avoid destructors that might call CUDA
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

		// Debug control
		bool verbose_logging = false;
		bool recording_active = false;

		// Constant speed mode for real-world testing
		bool constant_speed_mode = false;
		double target_constant_speed = DEFAULT_CONSTANT_SPEED; // Use macro for easy adjustment
		double constant_throttle = DEFAULT_CONSTANT_THROTTLE;  // Use macro for easy adjustment

	public:
		MPCIntegratedApp(int argc, char **argv, QObject *parent = nullptr)
		    : QObject(parent), controls_manager(nullptr), mpc_planner(nullptr), mpc_timer(nullptr) {
			// Setup signal handlers
			signal(SIGINT, signalHandler);
			signal(SIGTERM, signalHandler);

			try {
				// Inicializar o sistema de controles existente
				controls_manager = new ControlsManager(argc, argv, this);

				// Register for emergency motor stop
				g_emergency_controls = controls_manager;
				std::cout << "[MPCIntegratedApp] Emergency motor stop system registered"
				          << std::endl;

				// Inicializar MPC
				mpc_planner = new MPCPlanner();

				// Timer para executar MPC periodicamente
				mpc_timer = new QTimer(this);
				connect(mpc_timer, &QTimer::timeout, this, &MPCIntegratedApp::runMPCStep);

				std::cout << "=== MPC Integrated System ===" << std::endl;
				std::cout << "Sistema iniciado com controle manual" << std::endl;
				std::cout << "Comandos disponíveis:" << std::endl;
				std::cout << "- Use joystick para mover e gravar trajetória" << std::endl;
				std::cout << "- Pressione '1' para ATIVAR MPC / '2' para MANUAL" << std::endl;
				std::cout << "- Pressione '3' para INICIAR gravação / '4' para PARAR gravação"
				          << std::endl;
				std::cout << "- Pressione '5' para ATIVAR logs / '6' para DESATIVAR logs"
				          << std::endl;
				std::cout << "- Pressione 's' para mostrar status do sistema" << std::endl;
				std::cout << "- Pressione 'c' para limpar trajetória gravada" << std::endl;
				std::cout << "- Pressione 'e' para EMERGENCY STOP (parada imediata)" << std::endl;
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
				}
			} catch(const std::exception &e) {
				ERROR_STREAM("Main") << "[MPCIntegratedApp] Initialization error: " << e.what();
				throw;
			}
		}

		~MPCIntegratedApp() {
			std::cout << "[~MPCIntegratedApp] Starting cleanup..." << std::endl;

			try {
				// CRITICAL: Stop motors first thing in cleanup
				std::cout << "[~MPCIntegratedApp] FAILSAFE: Stopping all motors..." << std::endl;
				emergencyMotorStop();

				// Set global flag to stop all operations
				g_running = false;

				// Clear the global emergency controls pointer
				g_emergency_controls = nullptr;

				// Stop all timers first
				if(mpc_timer) {
					mpc_timer->stop();
					mpc_timer->deleteLater();
					mpc_timer = nullptr;
				}

				if(m_visualizationTimer) {
					m_visualizationTimer->stop();
					m_visualizationTimer->deleteLater();
					m_visualizationTimer = nullptr;
				}

				// Close OpenCV windows safely
				try {
					cv::destroyAllWindows();
					cv::waitKey(1); // Process any pending events

					// Clean up member matrices safely (avoid release() on fixed-size matrices)
					if(!m_currentLaneMask.empty()) {
						m_currentLaneMask = cv::Mat(); // Safe cleanup without release()
					}
					if(!m_processedFrame.empty()) {
						m_processedFrame = cv::Mat(); // Safe cleanup without release()
					}

					// Force cleanup of OpenCV internal memory (safer method)
					cv::Mat temp;
					temp.create(1, 1, CV_8UC1);
					temp = cv::Mat(); // Safe cleanup
					std::this_thread::sleep_for(std::chrono::milliseconds(50));

				} catch(const cv::Exception &e) {
					std::cerr << "[~MPCIntegratedApp] OpenCV cleanup warning: " << e.what()
					          << std::endl;
				} catch(...) {
					// Ignore other OpenCV cleanup errors
				}

				// For SIGINT shutdowns, avoid CUDA operations entirely
				if(m_inferencer) {
					try {
						// Don't try to synchronize or reset CUDA during signal shutdown
						// Just reset the pointer to avoid double-free
						m_inferencer.reset();
					} catch(...) {
						std::cerr << "[~MPCIntegratedApp] Warning: TensorRT cleanup skipped "
						             "(shutdown in progress)"
						          << std::endl;
						// Ignore all TensorRT/CUDA errors during shutdown
					}
				}

				// Reset other smart pointers safely
				try {
					m_laneDetectionSubscriber.reset();
				} catch(...) {
					// Ignore subscriber cleanup errors
				}

				// Delete MPC planner safely
				if(mpc_planner) {
					try {
						delete mpc_planner;
					} catch(...) {
						// Ignore MPC cleanup errors
					}
					mpc_planner = nullptr;
				}

				// controls_manager is QObject child, will be deleted automatically
				// but we can set it to nullptr for safety
				controls_manager = nullptr;

				std::cout << "[~MPCIntegratedApp] Cleanup complete" << std::endl;

			} catch(const std::exception &e) {
				ERROR_STREAM("Main") << "[~MPCIntegratedApp] Error during cleanup: " << e.what();
			} catch(...) {
				ERROR_LOG("Main", "[~MPCIntegratedApp] Unknown error during cleanup");
			}
		}

	private slots:
		void runMPCStep() {
			if(!mpc_active || !g_running) {
				return;
			}

			try {
				// CRITICAL: Always try to get fresh lane detection data for MPC
				// This ensures MPC has the latest trajectory data independent of visualization
				getLaneDetectionFrame();

				// Use MPC predicted trajectory if available, otherwise fall back to recorded
				// waypoints
				std::vector<Point2D> reference_trajectory;

				if(!m_predictedTrajectory.empty()) {
					reference_trajectory = m_predictedTrajectory;

					// Log when using lane detection trajectory
					static int lane_traj_counter = 0;
					if(++lane_traj_counter % 100 == 0) {
						std::cout << "[MPC] Using LANE DETECTION trajectory ("
						          << m_predictedTrajectory.size() << " points)" << std::endl;
					}
				} else if(recorded_waypoints.size() >= 3) {
					reference_trajectory = recorded_waypoints;

					// Log when falling back to recorded waypoints
					static int recorded_traj_counter = 0;
					if(++recorded_traj_counter % 100 == 0) {
						std::cout << "[MPC] Using RECORDED waypoints (" << recorded_waypoints.size()
						          << " points)" << std::endl;
					}
				} else {
					// Enhanced logging for no trajectory case
					static int no_traj_counter = 0;
					if(++no_traj_counter % 50 == 0) {
						std::cout << "[MPC] WARNING: No trajectory available for control!"
						          << std::endl;
						std::cout << "[MPC] - Lane detection trajectory: "
						          << m_predictedTrajectory.size() << " points" << std::endl;
						std::cout << "[MPC] - Recorded waypoints: " << recorded_waypoints.size()
						          << " points" << std::endl;
						std::cout << "[MPC] - Check ZeroMQ connection on port 5556 for lane "
						             "detection data"
						          << std::endl;
					}
					return;
				}

				// Usar trajetória como referência
				LaneInfo lane_info(0.0, 0.0);

				// Get current state from ControlsManager instead of using static local variable
				VehicleState current_state_from_controls =
				    controls_manager->getCurrentVehicleState();

				ControlCommand control = mpc_planner->plan(current_state_from_controls,
				                                           reference_trajectory, &lane_info);

				// Apply constant speed mode if enabled
				if(constant_speed_mode) {
					// Override throttle for constant speed
					control.throttle = constant_throttle;

					// Optional: limit steering rate for smoother operation
					static double last_steering = 0.0;
					double max_steering_change = 0.05; // rad per step
					double steering_diff = control.steer - last_steering;

					if(std::abs(steering_diff) > max_steering_change) {
						control.steer = last_steering + (steering_diff > 0 ? max_steering_change
						                                                   : -max_steering_change);
					}
					last_steering = control.steer;

					// Log constant speed mode
					if(step_counter % 100 == 0) {
						std::cout << "[CONSTANT SPEED] Target: " << std::fixed
						          << std::setprecision(1) << (target_constant_speed * 3.6)
						          << " km/h (" << std::setprecision(2) << target_constant_speed
						          << " m/s)"
						          << ", Throttle: " << std::setprecision(3) << control.throttle
						          << ", Steering: " << control.steer << " rad" << std::endl;
					}
				}

				// Add MPC diagnostic logging
				if(verbose_logging && step_counter % 50 == 0) {
					std::cout << "[MPC Debug] Input state: pos(" << current_state_from_controls.x
					          << "," << current_state_from_controls.y
					          << ") yaw=" << current_state_from_controls.yaw
					          << " vel=" << current_state_from_controls.velocity << std::endl;
					std::cout << "[MPC Debug] Trajectory points: " << reference_trajectory.size()
					          << std::endl;
					if(!reference_trajectory.empty()) {
						std::cout << "[MPC Debug] First 3 ref points: ";
						for(size_t i = 0; i < std::min(size_t(3), reference_trajectory.size());
						    i++) {
							std::cout << "(" << reference_trajectory[i].x << ","
							          << reference_trajectory[i].y << ") ";
						}
						std::cout << std::endl;
					}
					std::cout << "[MPC Debug] Output: throttle=" << control.throttle
					          << " steering=" << control.steer << std::endl;
				}

				// Printar a trajetória prevista do MPC (only in verbose mode)
				const auto &predicted_traj = mpc_planner->getPredictedTrajectory();
				if(verbose_logging && !predicted_traj.empty() &&
				   step_counter % 100 == 0) { // Reduced frequency from 20 to 100
					std::cout << "[MPC Predicted] ";
					for(size_t i = 0; i < std::min(size_t(5), predicted_traj.size()); i++) {
						std::cout << "(" << std::fixed << std::setprecision(1)
						          << predicted_traj[i].x << "," << predicted_traj[i].y << ") ";
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

				// Compact logging by default, verbose when enabled
				if(verbose_logging && step_counter % 10 == 0) {
					// Verbose mode - more frequent and detailed logging
					std::cout << "\n=== MPC CONTROL STEP " << step_counter << " ===" << std::endl;
					std::cout << "[MPC] Pos: (" << std::fixed << std::setprecision(3)
					          << current_state_from_controls.x << ", "
					          << current_state_from_controls.y
					          << ") Vel: " << current_state_from_controls.velocity
					          << " m/s Yaw: " << current_state_from_controls.yaw * 180 / M_PI << "°"
					          << std::endl;
					std::cout << "[MPC] Throttle: " << std::setprecision(3) << control.throttle
					          << " Steering: " << control.steer << " rad" << std::endl;

					if(!m_predictedTrajectory.empty()) {
						std::cout << "[MPC] Lane trajectory: " << m_predictedTrajectory.size()
						          << " points";
						// Show first 3 points
						for(size_t i = 0; i < std::min(size_t(3), m_predictedTrajectory.size());
						    i++) {
							std::cout << " (" << std::setprecision(2) << m_predictedTrajectory[i].x
							          << "," << m_predictedTrajectory[i].y << ")";
						}
						std::cout << std::endl;
					}
					std::cout << "================================\n" << std::endl;
				} else if(!verbose_logging && step_counter % 100 == 0) {
					// Compact mode - minimal logging
					std::cout << "[MPC #" << step_counter << "] Pos:(" << std::fixed
					          << std::setprecision(1) << current_state_from_controls.x << ","
					          << current_state_from_controls.y << ") T:" << std::setprecision(2)
					          << control.throttle << " S:" << control.steer << std::endl;
				}

				// Simular atualização do estado (em um sistema real,
				// isso viria de sensores/odometria)
				// updateVehicleState(control.throttle, control.steer); // Now using ControlsManager
				// state instead
			} catch(const std::exception &e) {
				ERROR_STREAM("Main") << "Erro no MPC: " << e.what();
			}
		}

		void handleKeyPress() {
			std::string input;
			std::getline(std::cin, input);

			if(input == "1") {
				// ATIVAR MPC
				if(!mpc_active) {
					activateMPC();
				} else {
					std::cout << "MPC já está ATIVO" << std::endl;
				}
			} else if(input == "2") {
				// ATIVAR MANUAL
				if(mpc_active) {
					deactivateMPC();
				} else {
					std::cout << "Modo MANUAL já está ativo" << std::endl;
				}
			} else if(input == "3") {
				// INICIAR gravação
				if(!recording_active) {
					recording_active = true;
					recorded_waypoints.clear();
					startRecording();
					std::cout << "INICIANDO gravação de trajetória" << std::endl;
				} else {
					std::cout << "Gravação já está ATIVA" << std::endl;
				}
			} else if(input == "4") {
				// PARAR gravação
				if(recording_active) {
					recording_active = false;
					std::cout << "PARANDO gravação de trajetória (" << recorded_waypoints.size()
					          << " waypoints gravados)" << std::endl;
				} else {
					std::cout << "Gravação já está PARADA" << std::endl;
				}
			} else if(input == "5") {
				// ATIVAR logs
				if(!verbose_logging) {
					verbose_logging = true;
					std::cout << "Logs detalhados ATIVADOS" << std::endl;
				} else {
					static int repeat_counter = 0;
					if(++repeat_counter % 10 == 0) { // Only show every 10th time
						std::cout << "Logs detalhados já estão ATIVOS (reminder #"
						          << repeat_counter / 10 << ")" << std::endl;
					}
				}
			} else if(input == "6") {
				// DESATIVAR logs
				if(verbose_logging) {
					verbose_logging = false;
					std::cout << "Logs detalhados DESATIVADOS" << std::endl;
				} else {
					std::cout << "Logs detalhados já estão DESATIVADOS" << std::endl;
				}
			} else if(input == "7") {
				// LIMPEZA DE MEMÓRIA
				std::cout << "Executando limpeza de memória..." << std::endl;
				// Force OpenCV cleanup (safer method)
				cv::Mat temp;
				temp.create(1, 1, CV_8UC1);
				temp = cv::Mat(); // Safe cleanup
// Force CUDA memory cleanup if available
#ifdef CUDA_AVAILABLE
				try {
					cudaDeviceSynchronize();
					cudaError_t error = cudaGetLastError();
					if(error == cudaSuccess) {
						std::cout << "CUDA memory synchronized" << std::endl;
					}
				} catch(...) {
					// Ignore CUDA errors during cleanup
				}
#endif
				std::cout << "Limpeza de memória concluída" << std::endl;
			} else if(input == "8") {
				// ATIVAR modo velocidade constante
				if(!controls_manager->isConstantSpeedMode()) {
					controls_manager->setConstantSpeedMode(true, target_constant_speed,
					                                       constant_throttle);
					constant_speed_mode = true; // Update local flag for UI consistency
				} else {
					std::cout << "Modo velocidade constante já está ATIVO" << std::endl;
				}
			} else if(input == "9") {
				// DESATIVAR modo velocidade constante
				if(controls_manager->isConstantSpeedMode()) {
					controls_manager->setConstantSpeedMode(false);
					constant_speed_mode = false; // Update local flag for UI consistency
					std::cout << "MODO VELOCIDADE CONSTANTE DESATIVADO" << std::endl;
					std::cout << "MPC voltará a controlar velocidade normalmente" << std::endl;
				} else {
					std::cout << "Modo velocidade constante já está DESATIVO" << std::endl;
				}
			} else if(input == "0") {
				// Ajustar velocidade constante
				double current_kmh = target_constant_speed * 3.6; // Convert m/s to km/h
				std::cout << "Digite nova velocidade:" << std::endl;
				std::cout << "  Em km/h [atual: " << std::fixed << std::setprecision(1)
				          << current_kmh << " km/h]: ";
				std::string speed_input;
				std::getline(std::cin, speed_input);
				try {
					double new_speed_kmh = std::stod(speed_input);
					double new_speed_ms = new_speed_kmh / 3.6; // Convert km/h to m/s

					if(new_speed_ms >= MIN_SAFE_SPEED && new_speed_ms <= MAX_SAFE_SPEED) {
						target_constant_speed = new_speed_ms;
						// Adjust throttle proportionally using intelligent scaling
						// Base throttle + proportional adjustment based on speed ratio
						double speed_ratio = new_speed_ms / DEFAULT_CONSTANT_SPEED;
						constant_throttle = DEFAULT_CONSTANT_THROTTLE * speed_ratio;
						// Clamp to safe throttle range
						constant_throttle = std::clamp(constant_throttle, 0.05, 0.35);

						std::cout << "✓ Nova velocidade: " << std::fixed << std::setprecision(1)
						          << (target_constant_speed * 3.6) << " km/h ("
						          << std::setprecision(2) << target_constant_speed << " m/s)"
						          << std::endl;
						std::cout << "✓ Throttle ajustado: " << std::setprecision(3)
						          << constant_throttle << " (" << (constant_throttle * 100) << "%)"
						          << std::endl;
					} else {
						double min_kmh = MIN_SAFE_SPEED * 3.6;
						double max_kmh = MAX_SAFE_SPEED * 3.6;
						std::cout << "❌ Velocidade deve estar entre " << std::fixed
						          << std::setprecision(1) << min_kmh << " e " << max_kmh << " km/h"
						          << std::endl;
					}
				} catch(...) {
					std::cout << "Valor inválido. Mantendo velocidade atual." << std::endl;
				}
			} else if(input == "c") {
				// Limpar waypoints
				recorded_waypoints.clear();
				std::cout << "Trajetória limpa (" << recorded_waypoints.size() << " waypoints)"
				          << std::endl;
			} else if(input == "s") {
				// Mostrar status
				showStatus();
			} else if(input == "z") {
				// Test ZeroMQ connection manually
				testZeroMQConnection();
			} else if(input == "h" || input == "help") {
				// Mostrar ajuda
				std::cout << "\n=== COMANDOS DISPONÍVEIS ===" << std::endl;
				std::cout << "=== CONTROLE BÁSICO ===" << std::endl;
				std::cout << "1: Ativar MPC" << std::endl;
				std::cout << "2: Ativar Manual" << std::endl;
				std::cout << "=== GRAVAÇÃO ===" << std::endl;
				std::cout << "3: Iniciar Gravação" << std::endl;
				std::cout << "4: Parar Gravação" << std::endl;
				std::cout << "=== DEBUG ===" << std::endl;
				std::cout << "5: Ativar Logs Detalhados" << std::endl;
				std::cout << "6: Desativar Logs Detalhados" << std::endl;
				std::cout << "7: Limpeza de Memória" << std::endl;
				std::cout << "=== VELOCIDADE CONSTANTE (TESTE REAL) ===" << std::endl;
				std::cout << "8: Ativar Modo Velocidade Constante" << std::endl;
				std::cout << "9: Desativar Modo Velocidade Constante" << std::endl;
				std::cout << "0: Ajustar Velocidade Constante" << std::endl;
				std::cout << "=== UTILIDADES ===" << std::endl;
				std::cout << "s: Mostrar Status" << std::endl;
				std::cout << "z: Testar Conexão ZeroMQ" << std::endl;
				std::cout << "c: Limpar Waypoints" << std::endl;
				std::cout << "h: Mostrar esta Ajuda" << std::endl;
				std::cout << "e: EMERGENCY STOP (parar motores imediatamente)" << std::endl;
				std::cout << "test: Testar Sistema de Emergência" << std::endl;
				std::cout << "q: Sair" << std::endl;
				std::cout << "===========================" << std::endl;
			} else if(input == "e" || input == "E" || input == "emergency") {
				// EMERGENCY STOP - parada imediata dos motores
				std::cout << "\n*** TERMINAL EMERGENCY STOP ACTIVATED ***" << std::endl;

				// Use ControlsManager's emergency stop method
				controls_manager->emergencyMotorStop();

				// Also deactivate MPC for safety
				if(mpc_active) {
					mpc_active = false;
					mpc_timer->stop();
					std::cout << "MPC DESATIVADO por parada de emergência" << std::endl;
				}

				std::cout << "*** Veículo em modo PARADO ***" << std::endl;
				std::cout << "*** Digite '2' para retomar controle manual ***" << std::endl;
			} else if(input == "test" || input == "emergency-test") {
				// Test emergency stop system
				testEmergencyStop();
			} else if(input == "soft") {
				// Toggle soft start
				bool current_enabled = controls_manager->isSoftStartEnabled();
				controls_manager->setSoftStartEnabled(!current_enabled);
				std::cout << "Soft Start agora está: "
				          << (!current_enabled ? "ATIVADO" : "DESATIVADO") << std::endl;
			} else if(input == "softp") {
				// Configure soft start parameters
				std::cout << "=== CONFIGURAÇÃO DE SOFT START ===" << std::endl;
				std::cout
				    << "Digite nova taxa máxima de mudança por passo (% por iteração) [atual: "
				    << (controls_manager->isSoftStartEnabled() ? "1" : "N/A") << "]: ";
				std::string change_input;
				std::getline(std::cin, change_input);

				std::cout << "Digite novo limite inicial (% durante aquecimento) [atual: 5]: ";
				std::string limit_input;
				std::getline(std::cin, limit_input);

				std::cout << "Digite duração do aquecimento (segundos) [atual: 3.0]: ";
				std::string duration_input;
				std::getline(std::cin, duration_input);

				try {
					double max_change =
					    change_input.empty() ? 0.01 : std::stod(change_input) / 100.0;
					double initial_limit =
					    limit_input.empty() ? 0.05 : std::stod(limit_input) / 100.0;
					double warmup_duration =
					    duration_input.empty() ? 3.0 : std::stod(duration_input);

					// Validate ranges
					max_change = std::clamp(max_change, 0.001, 0.1);      // 0.1% to 10% per step
					initial_limit = std::clamp(initial_limit, 0.01, 0.3); // 1% to 30% initial limit
					warmup_duration = std::clamp(warmup_duration, 0.5, 10.0); // 0.5s to 10s warmup

					controls_manager->setSoftStartParameters(max_change, initial_limit,
					                                         warmup_duration);
				} catch(...) {
					std::cout << "Valores inválidos. Mantendo configuração atual." << std::endl;
				}
			} else if(input == "q") {
				// Sair
				g_running = false;
				QApplication::quit();
			} else {
				std::cout << "Comando não reconhecido. Digite 'h' para ajuda ou use:" << std::endl;
				std::cout << "1:MPC  2:Manual  3:InicGrav  4:PararGrav  5:LogsON  6:LogsOFF  "
				             "s:Status  z:TestZMQ  q:Sair"
				          << std::endl;
			}
		}

		void getCameraFrame() {
			// This method is now only for visualization - lane detection is handled separately
			// Create a simple status indicator frame for visualization
			static cv::Mat status_frame;
			if(status_frame.empty()) {
				status_frame = cv::Mat::zeros(240, 320, CV_8UC3);
				cv::putText(status_frame, "External Camera", cv::Point(80, 120),
				            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(100, 100, 100), 2);
				cv::putText(status_frame, "Mode Active", cv::Point(100, 150),
				            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(100, 100, 100), 2);
			}
			m_currentCameraFrame = status_frame.clone();
			m_cameraFrameAvailable = true;
		}

		void executeLocalInference(const cv::Mat &frame) {
			// Since we removed CameraStreamer, always fallback to remote inference
			// The frame parameter is kept for compatibility but not used
			getLaneDetectionFrame();
		}

		void getLaneDetectionFrame() {
			if(!m_laneDetectionSubscriber) {
				static int no_sub_counter = 0;
				if(++no_sub_counter % 300 == 0) { // Log every ~5 seconds
					std::cout << "[getLaneDetectionFrame] ERROR: Lane detection subscriber not "
					             "initialized!"
					          << std::endl;
					std::cout
					    << "[getLaneDetectionFrame] MPC will not work without lane detection data!"
					    << std::endl;
				}
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

						static int message_counter = 0;
						message_counter++;

						if(message_counter % 60 == 0) { // Log every 60 messages (~2 seconds)
							std::cout << "[getLaneDetectionFrame] ✓ Received message "
							          << message_counter << ", size: " << received_msg.size()
							          << " bytes" << std::endl;
						}

						// Check for inference_frame topic
						if(received_msg.find("inference_frame ") == 0) {
							std::string mask_data =
							    received_msg.substr(16); // "inference_frame ".length()

							std::vector<uchar> buffer(mask_data.begin(), mask_data.end());
							cv::Mat binary_mask = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);

							if(!binary_mask.empty()) {
								createLaneVisualization(binary_mask);

								if(message_counter % 60 == 0) {
									std::cout
									    << "[getLaneDetectionFrame] ✓ Successfully decoded mask: "
									    << binary_mask.cols << "x" << binary_mask.rows
									    << ", generated " << m_predictedTrajectory.size()
									    << " trajectory points" << std::endl;
								}
								return;
							} else {
								if(message_counter % 60 == 0) {
									std::cout
									    << "[getLaneDetectionFrame] ✗ Failed to decode mask data"
									    << std::endl;
								}
							}
						} else {
							// Check for binary_mask topic (alternative)
							if(received_msg.find("binary_mask ") == 0) {
								std::string mask_data =
								    received_msg.substr(12); // "binary_mask ".length()

								std::vector<uchar> buffer(mask_data.begin(), mask_data.end());
								cv::Mat binary_mask = cv::imdecode(buffer, cv::IMREAD_GRAYSCALE);

								if(!binary_mask.empty()) {
									createLaneVisualization(binary_mask);

									if(message_counter % 60 == 0) {
										std::cout
										    << "[getLaneDetectionFrame] ✓ Successfully decoded "
										       "binary_mask: "
										    << binary_mask.cols << "x" << binary_mask.rows
										    << ", generated " << m_predictedTrajectory.size()
										    << " trajectory points" << std::endl;
									}
									return;
								}
							} else {
								if(message_counter % 60 == 0) {
									std::cout << "[getLaneDetectionFrame] ✗ Unknown topic: "
									          << received_msg.substr(0, 30) << "..." << std::endl;
								}
							}
						}
					}
				} else {
					static int no_data_counter = 0;
					if(++no_data_counter % 600 == 0) { // Log every ~10 seconds
						std::cout << "[getLaneDetectionFrame] ⚠ No inference data available on "
						             "port 5556 (waiting "
						          << no_data_counter << " attempts)" << std::endl;
						std::cout << "[getLaneDetectionFrame] ⚠ MPC needs lane detection data to "
						             "work properly!"
						          << std::endl;
					}
				}

			} catch(const std::exception &e) {
				ERROR_STREAM("Main") << "[getLaneDetectionFrame] Error: " << e.what();
			}
		}

		void updateVisualization() {
			if(!g_running) {
				return;
			}

			try {
				// Always try to get fresh lane detection data for MPC
				getLaneDetectionFrame();

				// Adjust for 1024x600 screen - create smaller visualization (900x550)
				m_visualizationFrame = cv::Mat::zeros(550, 900, CV_8UC3);

				// Adjust regions for smaller window
				cv::Rect cameraRegion(10, 10, 420, 240);      // Smaller camera feed
				cv::Rect processedRegion(10, 260, 420, 240);  // Smaller processed view
				cv::Rect trajectoryRegion(440, 10, 450, 450); // Trajectory area

				// Get camera frame from ZeroMQ stream (for visualization only)
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
				} else if(key == '1') {
					// Ativar MPC
					if(!mpc_active) {
						activateMPC();
					} else {
						std::cout << "MPC já está ATIVO" << std::endl;
					}
				} else if(key == '2') {
					// Ativar Manual
					if(mpc_active) {
						deactivateMPC();
					} else {
						std::cout << "Modo MANUAL já está ativo" << std::endl;
					}
				} else if(key == '3') {
					// Iniciar gravação
					if(!recording_active) {
						recording_active = true;
						recorded_waypoints.clear();
						startRecording();
						std::cout << "INICIANDO gravação de trajetória" << std::endl;
					} else {
						std::cout << "Gravação já está ATIVA" << std::endl;
					}
				} else if(key == '4') {
					// Parar gravação
					if(recording_active) {
						recording_active = false;
						std::cout << "PARANDO gravação de trajetória (" << recorded_waypoints.size()
						          << " waypoints gravados)" << std::endl;
					} else {
						std::cout << "Gravação já está PARADA" << std::endl;
					}
				} else if(key == '5') {
					// Ativar logs
					if(!verbose_logging) {
						verbose_logging = true;
						std::cout << "Logs detalhados ATIVADOS" << std::endl;
					} else {
						static int opencv_repeat_counter = 0;
						if(++opencv_repeat_counter % 10 == 0) { // Only show every 10th time
							std::cout << "Logs detalhados já estão ATIVOS (opencv reminder #"
							          << opencv_repeat_counter / 10 << ")" << std::endl;
						}
					}
				} else if(key == '6') {
					// Desativar logs
					if(verbose_logging) {
						verbose_logging = false;
						std::cout << "Logs detalhados DESATIVADOS" << std::endl;
					} else {
						std::cout << "Logs detalhados já estão DESATIVADOS" << std::endl;
					}
				} else if(key == '7') {
					// Limpeza de memória
					std::cout << "Executando limpeza de memória..." << std::endl;
					// Safe OpenCV memory cleanup
					cv::Mat temp;
					temp.create(1, 1, CV_8UC1);
					temp = cv::Mat(); // Safe cleanup
#ifdef CUDA_AVAILABLE
					try {
						cudaDeviceSynchronize();
						std::cout << "CUDA memory synchronized" << std::endl;
					} catch(...) {
					}
#endif
					std::cout << "Limpeza de memória concluída" << std::endl;
				} else if(key == 'c') {
					recorded_waypoints.clear();
					std::cout << "Trajetória limpa (" << recorded_waypoints.size() << " waypoints)"
					          << std::endl;
				} else if(key == 's') {
					showStatus();
				} else if(key == 'h') {
					std::cout << "=== AJUDA - TECLAS OPENCV ===" << std::endl;
					std::cout << "1:MPC  2:Manual  3:InicGrav  4:PararGrav" << std::endl;
					std::cout << "5:LogsON  6:LogsOFF  7:LimpMem  s:Status  c:Limpar  q:Sair"
					          << std::endl;
					std::cout << "e:EMERGENCY STOP (parar motores imediatamente)" << std::endl;
				} else if(key == 'e' || key == 'E') {
					// EMERGENCY STOP - parada imediata dos motores
					std::cout << "\n*** OPENCV EMERGENCY STOP ACTIVATED ***" << std::endl;

					// Use ControlsManager's emergency stop method
					controls_manager->emergencyMotorStop();

					// Also deactivate MPC for safety
					if(mpc_active) {
						mpc_active = false;
						mpc_timer->stop();
						std::cout << "MPC DESATIVADO por parada de emergência" << std::endl;
					}

					std::cout << "*** Veículo em modo PARADO ***" << std::endl;
					std::cout << "*** Pressione '2' para retomar controle manual ***" << std::endl;
				}

			} catch(const std::exception &e) {
				ERROR_STREAM("Main") << "Visualization error: " << e.what();
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

			// Add emergency status indicator
			if(g_emergency_controls) {
				cv::putText(m_visualizationFrame, "EMERGENCY: Ready",
				            cv::Point(status_x, status_y + 2 * line_height),
				            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 0),
				            1); // Green when ready
			} else {
				cv::putText(m_visualizationFrame, "EMERGENCY: N/A",
				            cv::Point(status_x, status_y + 2 * line_height),
				            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255),
				            1); // Red when not available
			}

			// Add constant speed mode indicator
			if(controls_manager && controls_manager->isConstantSpeedMode()) {
				cv::putText(
				    m_visualizationFrame,
				    "CONST SPEED: " +
				        std::to_string(controls_manager->getTargetConstantSpeed()).substr(0, 3) +
				        "m/s",
				    cv::Point(status_x, status_y + 3 * line_height), cv::FONT_HERSHEY_SIMPLEX, 0.4,
				    cv::Scalar(0, 255, 255), 1); // Yellow for constant speed mode
			}

			cv::putText(m_visualizationFrame,
			            "Pos: (" + std::to_string(current_state.x).substr(0, 4) + "," +
			                std::to_string(current_state.y).substr(0, 4) + ")",
			            cv::Point(status_x, status_y + 4 * line_height), cv::FONT_HERSHEY_SIMPLEX,
			            0.4, cv::Scalar(255, 255, 255), 1);

			cv::putText(m_visualizationFrame,
			            "Vel: " + std::to_string(current_state.velocity).substr(0, 4),
			            cv::Point(status_x, status_y + 5 * line_height), cv::FONT_HERSHEY_SIMPLEX,
			            0.4, cv::Scalar(255, 255, 255), 1);

			// Enhanced camera status
			std::string camera_status;
			static int camera_frame_counter = 0;
			if(m_cameraFrameAvailable && !m_currentCameraFrame.empty()) {
				camera_frame_counter++;
				// Detect if it's synthetic
				cv::Scalar mean_color = cv::mean(m_currentCameraFrame);
				if(mean_color[0] < 30 && mean_color[1] < 30 && mean_color[2] < 30) {
					camera_status = "Cam: SYNTHETIC #" + std::to_string(camera_frame_counter);
				} else {
					camera_status = "Cam: REAL #" + std::to_string(camera_frame_counter);
				}
			} else {
				camera_status = "Cam: WAITING";
			}

			cv::putText(m_visualizationFrame, camera_status, cv::Point(status_x + 150, status_y),
			            cv::FONT_HERSHEY_SIMPLEX, 0.4,
			            m_cameraFrameAvailable ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 255, 0),
			            1);

			// Compact instructions
			cv::putText(m_visualizationFrame, "m:mode r:rec c:clear s:status e:EMERGENCY q:quit",
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
				std::cout << "[setupVisualization] Lane detection subscriber initialized for MPC"
				          << std::endl;
			} catch(const std::exception &e) {
				std::cerr << "[setupVisualization] Failed to initialize lane detection subscriber: "
				          << e.what() << std::endl;
				std::cerr << "[setupVisualization] MPC will not receive lane detection data!"
				          << std::endl;
			}
		}

		void setupKeyboardInput() {
			std::cout << "\n=== COMANDOS DISPONÍVEIS ===" << std::endl;
			std::cout << "=== BÁSICO ===" << std::endl;
			std::cout << "1: Ativar MPC       2: Ativar Manual" << std::endl;
			std::cout << "3: Iniciar Gravação 4: Parar Gravação" << std::endl;
			std::cout << "5: Logs ON          6: Logs OFF        7: Limpeza Memória" << std::endl;
			std::cout << "=== VELOCIDADE CONSTANTE (TESTE REAL) ===" << std::endl;
			std::cout << "8: Ativar Vel. Const.   9: Desativar Vel. Const." << std::endl;
			std::cout << "0: Ajustar Velocidade (entrada em km/h)" << std::endl;
			std::cout << "=== ACELERAÇÃO GRADUAL (SOFT START) ===" << std::endl;
			std::cout << "soft: Toggle Soft Start    softp: Configurar Parâmetros" << std::endl;
			std::cout << "=== SEGURANÇA ===" << std::endl;
			std::cout << "e: EMERGENCY STOP   test: Testar Sistema Emergência" << std::endl;
			std::cout << "=== UTILIDADES ===" << std::endl;
			std::cout << "s: Status    z: Test ZMQ    c: Limpar    h: Ajuda    q: Sair"
			          << std::endl;
			std::cout << "===============================================" << std::endl;
			std::cout << "⚠️  EMERGENCY STOP: Pressione 'e' para parar motores imediatamente"
			          << std::endl;
			std::cout << "Digite o comando ou use as teclas na janela OpenCV:" << std::endl;

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

		void activateMPC() {
			if(mpc_active) {
				return; // Already active
			}

			// Check if we have either recorded waypoints or live lane detection
			if(recorded_waypoints.size() < 3 && m_predictedTrajectory.empty()) {
				std::cout << "ERRO: Precisa de waypoints gravados OU detecção de pistas ativa!"
				          << std::endl;
				return;
			}

			mpc_active = true;
			// Mudar para modo autônomo
			controls_manager->setMode(DrivingMode::Automatic);
			mpc_timer->start(100); // 10 Hz - Reduced frequency to avoid interfering with camera

			if(!m_predictedTrajectory.empty()) {
				std::cout << "MPC ATIVADO - Seguindo detecção de pistas com "
				          << m_predictedTrajectory.size() << " pontos" << std::endl;
			} else {
				std::cout << "MPC ATIVADO - Seguindo trajetória gravada com "
				          << recorded_waypoints.size() << " waypoints" << std::endl;
			}
		}

		void deactivateMPC() {
			if(!mpc_active) {
				return; // Already inactive
			}

			mpc_active = false;
			// Mudar para modo manual
			controls_manager->setMode(DrivingMode::Manual);

			// Reset emergency stop flag when switching to manual mode
			if(controls_manager->isEmergencyStopActive()) {
				std::cout << "Resetando flag de emergência..." << std::endl;
				controls_manager->resetEmergencyStop();
			}

			mpc_timer->stop();
			std::cout << "MANUAL ATIVADO - Use joystick" << std::endl;
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

			// DIAGNÓSTICO CRÍTICO: Lane Detection Status
			std::cout << "\n--- LANE DETECTION STATUS ---" << std::endl;
			std::cout << "ZeroMQ Subscriber (port 5556): "
			          << (m_laneDetectionSubscriber ? "CONNECTED" : "NOT_CONNECTED") << std::endl;

			if(!m_predictedTrajectory.empty()) {
				std::cout << "✓ Lane trajectory: AVAILABLE (" << m_predictedTrajectory.size()
				          << " points)" << std::endl;
				// Show first few points
				std::cout << "  First 3 points: ";
				for(size_t i = 0; i < std::min(size_t(3), m_predictedTrajectory.size()); i++) {
					std::cout << "(" << std::fixed << std::setprecision(2)
					          << m_predictedTrajectory[i].x << "," << m_predictedTrajectory[i].y
					          << ") ";
				}
				std::cout << std::endl;
			} else {
				std::cout << "✗ Lane trajectory: NOT_AVAILABLE" << std::endl;
				std::cout << "  ⚠ MPC cannot work without lane detection data!" << std::endl;
				std::cout << "  ⚠ Check if inference system is publishing on port 5556"
				          << std::endl;
			}

			if(!m_currentLaneMask.empty()) {
				int white_pixels = cv::countNonZero(m_currentLaneMask);
				double coverage =
				    (white_pixels * 100.0) / (m_currentLaneMask.cols * m_currentLaneMask.rows);
				std::cout << "✓ Lane mask: " << m_currentLaneMask.cols << "x"
				          << m_currentLaneMask.rows << " (" << std::fixed << std::setprecision(1)
				          << coverage << "% coverage)" << std::endl;
			} else {
				std::cout << "✗ Lane mask: NOT_AVAILABLE" << std::endl;
			}

			std::cout << "\n--- Estado do Veículo ---" << std::endl;
			std::cout << "Posição atual: (" << std::fixed << std::setprecision(2) << current_state.x
			          << ", " << current_state.y << ")" << std::endl;
			std::cout << "Velocidade: " << std::setprecision(1) << (current_state.velocity * 3.6)
			          << " km/h (" << std::setprecision(2) << current_state.velocity << " m/s)"
			          << std::endl;
			std::cout << "Orientação: " << std::setprecision(1) << (current_state.yaw * 180 / M_PI)
			          << " graus" << std::endl;

			std::cout << "\n--- Controle de Velocidade ---" << std::endl;
			std::cout << "Modo velocidade constante: "
			          << (constant_speed_mode ? "ATIVO" : "INATIVO") << std::endl;
			if(constant_speed_mode) {
				std::cout << "Velocidade alvo: " << std::setprecision(1)
				          << (target_constant_speed * 3.6) << " km/h (" << std::setprecision(2)
				          << target_constant_speed << " m/s)" << std::endl;
				std::cout << "Throttle fixo: " << constant_throttle << std::endl;
			} else {
				std::cout << "Controle de velocidade: MPC automático" << std::endl;
			}

			std::cout << "\n--- Sistema de Visão ---" << std::endl;
			std::cout << "Camera frame disponível: " << (m_cameraFrameAvailable ? "SIM" : "NÃO")
			          << std::endl;
			std::cout << "TensorRT inferencer: " << (m_inferencer ? "ATIVO" : "INATIVO")
			          << std::endl;
			std::cout << "Processed frame: " << (!m_processedFrame.empty() ? "SIM" : "NÃO")
			          << std::endl;

			// MPC READINESS CHECK
			std::cout << "\n--- MPC READINESS ---" << std::endl;
			bool mpc_ready = !m_predictedTrajectory.empty() || recorded_waypoints.size() >= 3;
			std::cout << "MPC Operational: " << (mpc_ready ? "✓ READY" : "✗ NOT_READY")
			          << std::endl;

			if(!mpc_ready) {
				std::cout << "  Issues:" << std::endl;
				if(m_predictedTrajectory.empty()) {
					std::cout << "  - No lane detection trajectory (check port 5556)" << std::endl;
				}
				if(recorded_waypoints.size() < 3) {
					std::cout << "  - No recorded waypoints (record at least 3 points)"
					          << std::endl;
				}
				std::cout << "  Solutions:" << std::endl;
				std::cout << "  - Start external inference system on port 5556" << std::endl;
				std::cout << "  - OR record waypoints using manual mode (press 3)" << std::endl;
			}

			std::cout << "===================================\n" << std::endl;
		}

		void testEmergencyStop() {
			std::cout << "\n=== TESTE DO SISTEMA DE EMERGÊNCIA ===" << std::endl;
			std::cout << "⚠️  ATENÇÃO: Este teste irá parar todos os motores!" << std::endl;
			std::cout << "Confirma teste? (s/N): ";

			std::string confirmation;
			std::getline(std::cin, confirmation);

			if(confirmation == "s" || confirmation == "S" || confirmation == "sim") {
				std::cout << "Executando teste de emergência..." << std::endl;

				// Use ControlsManager's emergency stop method
				controls_manager->emergencyMotorStop();

				// Also stop MPC
				if(mpc_active) {
					mpc_active = false;
					mpc_timer->stop();
					std::cout << "MPC parado durante teste de emergência" << std::endl;
				}

				std::cout << "✅ Teste de emergência concluído" << std::endl;
				std::cout << "💡 Para retomar operação, pressione '2' (modo manual)" << std::endl;
			} else {
				std::cout << "Teste cancelado" << std::endl;
			}
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
				ERROR_STREAM("Main") << "[createLaneVisualization] Error: " << e.what();
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
				ERROR_STREAM("Main") << "[generateMPCTrajectory] Error: " << e.what();
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
				ERROR_STREAM("Main") << "[drawTrajectoryVisualization] Error: " << e.what();
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

		void testZeroMQConnection() {
			std::cout << "\n=== TESTE DE CONEXÃO ZEROMQ ===" << std::endl;

			try {
				// Test subscriber connection to port 5556
				std::cout << "Testando conexão com porta 5556..." << std::endl;

				if(m_laneDetectionSubscriber) {
					std::cout << "✅ Subscriber já inicializado" << std::endl;

					// Try to receive data with short timeout
					std::cout << "Testando recepção de dados..." << std::endl;

					zmq::pollitem_t items[] = {
					    {static_cast<void *>(m_laneDetectionSubscriber->getSocket()), 0, ZMQ_POLLIN,
					     0}};
					int poll_result = zmq::poll(items, 1, 1000); // 1 second timeout

					if(poll_result > 0 && (items[0].revents & ZMQ_POLLIN)) {
						std::cout << "✅ Dados disponíveis na porta 5556!" << std::endl;
					} else if(poll_result == 0) {
						std::cout << "⚠️  Timeout: Nenhum dado recebido em 1 segundo" << std::endl;
						std::cout
						    << "   Verifique se o sistema de inferência está rodando na porta 5556"
						    << std::endl;
					} else {
						std::cout << "❌ Erro no polling" << std::endl;
					}
				} else {
					std::cout << "❌ Subscriber não inicializado" << std::endl;
					std::cout << "   Tente reinicializar o sistema" << std::endl;
				}

			} catch(const std::exception &e) {
				std::cout << "❌ Erro na conexão ZeroMQ: " << e.what() << std::endl;
			}

			std::cout << "=== FIM DO TESTE ===" << std::endl;
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
			ERROR_STREAM("Main") << "Failed to initialize application: " << e.what();
			return 1;
		}

		std::cout << "Sistema pronto! Use os comandos no terminal ou na janela OpenCV."
		          << std::endl;
		std::cout << "Pressione Ctrl+C ou 'q' para sair." << std::endl;

		// Setup periodic check for global running flag
		QTimer *shutdown_timer = new QTimer(&app);
		QObject::connect(shutdown_timer, &QTimer::timeout, [&]() {
			if(!g_running) {
				std::cout << "[main] Shutting down due to signal..." << std::endl;

				// Stop the app immediately
				if(integrated_app) {
					integrated_app.reset();
				}

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
				ERROR_STREAM("Main") << "[main] Warning during Publisher cleanup: " << e.what();
			}

			std::cout << "[main] Application cleanup complete" << std::endl;
		} catch(const std::exception &e) {
			ERROR_STREAM("Main") << "[main] Error during cleanup: " << e.what();
		}

		return result;

	} catch(const std::exception &e) {
		ERROR_STREAM("Main") << "Erro: " << e.what();
		try {
			cv::destroyAllWindows();
			Publisher::destroyAll();
		} catch(...) {
			ERROR_LOG("Main", "[main] Additional errors during exception cleanup");
		}
		return 1;
	}
}

#include "main.moc"
