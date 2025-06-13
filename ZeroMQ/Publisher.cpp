#include "Publisher.hpp"

std::unordered_map<int, Publisher*> Publisher::instances;

Publisher::Publisher(int port) : context(1), publisher(context, ZMQ_PUB), joytstick_value(true), running(false) {
	boundAddress = "tcp://*:" + std::to_string(port);
	publisher.bind(boundAddress);  // Dynamic port binding
}

Publisher::~Publisher() {
	try {
		publisher.unbind(boundAddress);  // Use stored address
		publisher.close();
		context.close();
		std::cout << "[Publisher] Unbound from " << boundAddress << std::endl;
	} catch (const zmq::error_t& e) {
		std::cerr << "[Publisher] Failed to unbind: " << e.what() << std::endl;
	}
}

Publisher* Publisher::m_instance = nullptr;

Publisher* Publisher::instance(int port) {
	if (instances.find(port) == instances.end()) {
		instances[port] = new Publisher(port);
	}
	return instances[port];
}

void Publisher::destroyAll() {
	for (auto& pair : instances) {
		delete pair.second;
	}
	instances.clear();
}

void Publisher::publish(const std::string& topic, const std::string& message) {
	std::cout << "[Publisher] Full message: " << topic << " " << message << std::endl;

	std::string full_message = topic + " " + message;
	zmq::message_t zmq_message(full_message.begin(), full_message.end());

	publisher.send(zmq_message);  // Send the message
}

void Publisher::setJoystickStatus(bool new_joytstick_value) {
	std::cout << "[Publisher] Publishing joystick_value: " << (joytstick_value ? "true" : "false") << std::endl;

	std::lock_guard<std::mutex> lock(joystick_mtx);  // Ensure thread safety
	if (new_joytstick_value != joytstick_value) {
		joytstick_value = new_joytstick_value;
		std::string bool_str = joytstick_value ? "true" : "false";
		publish("joystick_value", bool_str);  // Publish a new status message
	}
}

void Publisher::publishInferenceFrame(const std::string& topic, const cv::cuda::GpuMat& gpu_image) {
	std::lock_guard<std::mutex> lock(frame_mtx);  // Ensure thread safety
	try {
		// Download GPU image to CPU
		cv::Mat cpu_image;
		gpu_image.download(cpu_image);

		if (cpu_image.empty()) {
			std::cerr << "[Publisher] Skipped: empty CPU image." << std::endl;
			return;
		}

		// Encode to JPEG
		std::vector<uchar> encoded;
		if (!cv::imencode(".jpg", cpu_image, encoded)) {
			std::cerr << "[Publisher] Encoding failed." << std::endl;
			return;
		}

		// Build single message: "topic " + raw image bytes
		std::string header = topic + " ";
		std::vector<uchar> messageData;
		messageData.reserve(header.size() + encoded.size());
		messageData.insert(messageData.end(), header.begin(), header.end());
		messageData.insert(messageData.end(), encoded.begin(), encoded.end());

		zmq::message_t zmq_message(messageData.data(), messageData.size());
		publisher.send(zmq_message);

		//std::cout << "[Publisher] Sent image as single-part message. Size: " << messageData.size() << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "[Publisher] Failed to publish image: " << e.what() << std::endl;
	}
}

/* void Publisher::publishCameraFrame(const std::string& topic, const cv::Mat& frame) {
	std::lock_guard<std::mutex> lock(frame_mtx);  // Ensure thread safety
	try {
		if (frame.empty()) {
			std::cerr << "[Publisher] Skipped: empty CPU image." << std::endl;
			return;
		}

		// Encode to JPEG
		std::vector<uchar> encoded;
		if (!cv::imencode(".jpg", frame, encoded)) {
			std::cerr << "[Publisher] Encoding failed." << std::endl;
			return;
		}

		// Build single message: "topic " + raw image bytes
		std::string header = topic + " ";
		std::vector<uchar> messageData;
		messageData.reserve(header.size() + encoded.size());
		messageData.insert(messageData.end(), header.begin(), header.end());
		messageData.insert(messageData.end(), encoded.begin(), encoded.end());

		zmq::message_t zmq_message(messageData.data(), messageData.size());
		publisher.send(zmq_message);

		//std::cout << "[Publisher] Sent image as single-part message. Size: " << messageData.size() << std::endl;

	} catch (const std::exception& e) {
		std::cerr << "[Publisher] Failed to publish image: " << e.what() << std::endl;
	}
} */
