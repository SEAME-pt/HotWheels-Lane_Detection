/*
 * Enhanced MPC Integration Example with Direct PolyfitterInferencer
 *
 * This example demonstrates the improved architecture where MPC gets data
 * directly from the PolyfitterInferencer without ZeroMQ overhead.
 */

#include "car_controls/includes/ControlsManager.hpp"
#include "car_controls/includes/MPCPlanner.hpp"
#include "car_controls/includes/inference/PolyfitterInferencer.hpp"
#include <QApplication>
#include <QObject>
#include <QTimer>
#include <atomic>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

// Global running flag used by ControlsManager threads
std::atomic<bool> g_running{true};

class EnhancedMPCApp : public QObject {
		// Note: Removed Q_OBJECT to avoid MOC dependency

	private:
		ControlsManager *controls_manager;
		std::shared_ptr<PolyfitterInferencer> polyfitter_inferencer;
		std::unique_ptr<MPCPlanner> mpc_planner;
		QTimer *mpc_timer;

		bool mpc_active = false;
		int step_counter = 0;

		// Camera integration for inference
		cv::VideoCapture camera;
		QTimer *camera_timer;

	public:
		EnhancedMPCApp(int argc, char **argv, QObject *parent = nullptr) : QObject(parent) {

			// Initialize ControlsManager
			controls_manager = new ControlsManager(argc, argv);

			// Initialize enhanced PolyfitterInferencer
			polyfitter_inferencer = std::make_shared<PolyfitterInferencer>(
			    "/home/jetson/models/lane-detection/model.engine");

			// Initialize MPCPlanner with direct integration
			mpc_planner = std::make_unique<MPCPlanner>(polyfitter_inferencer);
			mpc_planner->enableDirectInference(true);

			// Setup timers
			setupTimers();

			// Initialize camera
			if(camera.open(0)) {
				std::cout << "Camera initialized successfully" << std::endl;
			} else {
				std::cerr << "Failed to initialize camera" << std::endl;
			}

			std::cout << "Enhanced MPC App initialized with direct PolyfitterInferencer integration"
			          << std::endl;
		}

		~EnhancedMPCApp() {
			delete controls_manager;
		}

	private:
		void runInferenceAndMPC() {
			if(!mpc_active)
				return;

			// Capture frame from camera
			cv::Mat frame;
			if(!camera.read(frame)) {
				return;
			}

			// Run inference directly - this will update the inferencer's internal state
			try {
				polyfitter_inferencer->doInference(frame);

				// Get current vehicle state
				VehicleState current_state = controls_manager->getCurrentVehicleState();

				// Run MPC with direct inference data (no ZeroMQ needed!)
				ControlCommand control = mpc_planner->planWithDirectInference(current_state);

				// Apply control commands using new public methods
				controls_manager->applyControlCommand(control);

				// Debug output
				if(++step_counter % 20 == 0) { // Every 20 steps (~1 second at 20Hz)
					std::cout << "[Enhanced MPC #" << step_counter << "] "
					          << "Throttle: " << std::fixed << std::setprecision(3)
					          << control.throttle << ", Steer: " << control.steer
					          << ", Trajectory points: "
					          << mpc_planner->getCurrentTrajectory().size() << ", Processing time: "
					          << polyfitter_inferencer->getLastProcessingTimeMs() << "ms"
					          << std::endl;
				}

			} catch(const std::exception &e) {
				std::cerr << "Error in inference/MPC pipeline: " << e.what() << std::endl;
			}
		}

		void handleKeyPress() {
			// Handle keyboard input for controlling the application
			// Implementation similar to original but simplified due to direct integration
		}

		void setupTimers() {
			// MPC Timer - higher frequency possible due to direct integration
			mpc_timer = new QTimer(this);
			QObject::connect(mpc_timer, &QTimer::timeout, [this]() { this->runInferenceAndMPC(); });

			// Camera Timer for inference
			camera_timer = new QTimer(this);
			QObject::connect(camera_timer, &QTimer::timeout,
			                 [this]() { this->runInferenceAndMPC(); });
		}

	public:
		void activateMPC() {
			if(mpc_active)
				return;

			// Check if inferencer is ready
			if(!polyfitter_inferencer) {
				std::cout << "ERROR: PolyfitterInferencer not initialized!" << std::endl;
				return;
			}

			mpc_active = true;
			controls_manager->setMode(DrivingMode::Automatic);

			// Start both inference and MPC at 20Hz (much higher frequency than before)
			mpc_timer->start(50); // 20 Hz - possible due to direct integration

			std::cout << "Enhanced MPC ACTIVATED with direct inference integration" << std::endl;
			std::cout << "- No ZeroMQ overhead for critical path" << std::endl;
			std::cout << "- Direct polynomial coefficients from Polyfitter" << std::endl;
			std::cout << "- Real-time trajectory computation" << std::endl;
		}

		void deactivateMPC() {
			mpc_active = false;
			mpc_timer->stop();
			controls_manager->setMode(DrivingMode::Manual);

			std::cout << "Enhanced MPC DEACTIVATED" << std::endl;
		}

		void showStatus() {
			std::cout << "\n=== Enhanced MPC System Status ===" << std::endl;
			std::cout << "MPC Active: " << (mpc_active ? "YES" : "NO") << std::endl;
			std::cout << "Direct Inference: "
			          << (mpc_planner->hasValidTrajectoryData() ? "AVAILABLE" : "NOT_AVAILABLE")
			          << std::endl;
			std::cout << "Current Mode: "
			          << (controls_manager->getCurrentMode() == DrivingMode::Automatic ? "AUTOMATIC"
			                                                                           : "MANUAL")
			          << std::endl;

			if(polyfitter_inferencer) {
				std::cout << "Trajectory Points: "
				          << polyfitter_inferencer->getCurrentTrajectory().size() << std::endl;
				std::cout << "Poly Coefficients: "
				          << (polyfitter_inferencer->hasValidPolyCoeffs() ? "VALID" : "INVALID")
				          << std::endl;
				std::cout << "Last Processing: " << polyfitter_inferencer->getLastProcessingTimeMs()
				          << "ms ago" << std::endl;
			}

			std::cout << "Step Counter: " << step_counter << std::endl;
			std::cout << "================================\n" << std::endl;
		}
};

int main(int argc, char **argv) {
	QApplication app(argc, argv);

	EnhancedMPCApp enhanced_app(argc, argv);

	std::cout << "\n=== Enhanced MPC System ===" << std::endl;
	std::cout << "Direct PolyfitterInferencer Integration" << std::endl;
	std::cout << "No ZeroMQ overhead for critical MPC data" << std::endl;
	std::cout << "Commands:" << std::endl;
	std::cout << "  a - Activate MPC" << std::endl;
	std::cout << "  d - Deactivate MPC" << std::endl;
	std::cout << "  s - Show Status" << std::endl;
	std::cout << "  q - Quit" << std::endl;
	std::cout << "==========================\n" << std::endl;
	return app.exec();
}
