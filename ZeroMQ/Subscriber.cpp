#include "Subscriber.hpp"
#include <iostream>
#include <chrono>
#include <thread>

Subscriber::Subscriber() : context(1), subscriber(context, ZMQ_SUB), running(false) {}

Subscriber::~Subscriber() {
	stop();  // Ensure that the subscriber stops when destroyed
}

void Subscriber::connect(const std::string& address) {
	bool connected = false;

	// Attempt to connect until successful
	while (!connected) {
		try {
			subscriber.connect(address);  // Attempt to connect to the publisher
			std::cout << "Subscriber connected to " << address << std::endl;
			connected = true; // Exit the loop once the connection is successful
		}
		catch (const zmq::error_t& e) {
			std::cout << "Connection failed, retrying in 1 second..." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(1));  // Wait before retrying
		}
	}
}

zmq::socket_t& Subscriber::getSocket() {
	return subscriber;
}

void Subscriber::subscribe(const std::string& topic) {
	// Subscribe to a topic only after successfully connecting
	subscriber.setsockopt(ZMQ_SUBSCRIBE, topic.c_str(), topic.size());
}

/* void Subscriber::listen() {
    running = true;  // Mark the subscriber as running

    while (running) {
        try {
            zmq::message_t message;
            subscriber.recv(&message, 0);

        }
        catch (const zmq::error_t& e) {
            if (running) {  // If running is still true, handle reconnection
                std::cout << "Connection lost. Attempting to reconnect..." << std::endl;
                reconnect("tcp://localhost:5555");
            }
        }
    }

    std::cout << "Listener stopped." << std::endl;
} */

/* void Subscriber::listenFrames() {
    running = true;

    while (running) {
        try {
            zmq::message_t topic_msg;
            zmq::message_t image_msg;

            subscriber.recv(&topic_msg, 0);
            subscriber.recv(&image_msg, 0);

            std::string topic(static_cast<char*>(topic_msg.data()), topic_msg.size());

            if (topic != "inference_frame") {
                std::cerr << "[Subscriber] Unexpected topic: " << topic << std::endl;
                continue;
            }

            std::vector<uchar> jpeg_data(
                static_cast<uchar*>(image_msg.data()),
                static_cast<uchar*>(image_msg.data()) + image_msg.size());

            cv::Mat decoded = cv::imdecode(jpeg_data, cv::IMREAD_COLOR);
            if (!decoded.empty()) {
                std::cout << "[Subscriber] Received and decoded image.\n";
            } else {
                std::cerr << "[Subscriber] Failed to decode JPEG image." << std::endl;
            }

        } catch (const zmq::error_t& e) {
            std::cerr << "[Subscriber] Error while receiving image: " << e.what() << std::endl;
            if (running) reconnect("tcp://localhost:5555");
        }
    }

    std::cout << "[Subscriber] Image listener stopped." << std::endl;
} */

/* void Subscriber::reconnect(const std::string& address) {
    bool connected = false;

    // Retry the connection until successful
    while (!connected && running) {
        try {
            std::cout << "Reconnecting to " << address << "..." << std::endl;
            subscriber.connect(address);  // Attempt to reconnect to the publisher
            std::cout << "Reconnected successfully." << std::endl;
            connected = true; // Exit the loop once the connection is successful
        }
        catch (const zmq::error_t& e) {
            if (!running) return;  // Exit if running is set to false
            std::cout << "Reconnection failed, retrying in 1 second..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));  // Wait before retrying
        }
    }
    if (connected) {
        listen();  // Recurse into listen after reconnecting
    }
} */

void Subscriber::stop() {
	running = false;
	//subscriber.close();  // Close the socket gracefully
}
