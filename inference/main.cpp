#include "TensorRTInferencer.hpp"
#include "CameraStreamer.hpp"
#include <iostream>
#include <signal.h>

// Global flag for graceful shutdown
volatile sig_atomic_t stop_flag = 0;

// Signal handler for Ctrl+C
void signal_handler(int signal) {
    std::cout << "Received shutdown signal" << std::endl;
    stop_flag = 1;
}

int main(int argc, char* argv[]) {
    // Set up signal handling for graceful shutdown
    signal(SIGINT, signal_handler);

    try {
        std::cout << "Starting TensorRT Inference on Jetson..." << std::endl;

        // Path to your TensorRT engine file - adjust path as needed for Jetson
        std::string enginePath = "models/model.engine";

        // Create the TensorRT inferencer
        std::cout << "Loading TensorRT engine from: " << enginePath << std::endl;
        TensorRTInferencer inferencer(enginePath);

        // Create the camera streamer with the inferencer
        std::cout << "Initializing CSI camera..." << std::endl;
        CameraStreamer streamer(inferencer, 0.5, "Jetson Camera Inference", true);
        streamer.initUndistortMaps();

        // Start the stream (this will run until ESC is pressed or program is terminated)
        std::cout << "Starting camera stream with inference. Press ESC to exit." << std::endl;
        streamer.start();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Shutting down..." << std::endl;
    return 0;
}
