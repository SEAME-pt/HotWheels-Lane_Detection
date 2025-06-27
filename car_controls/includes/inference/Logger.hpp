#ifndef LOGGER_HPP
#define LOGGER_HPP

#pragma once
#include <NvInfer.h>
#include <iostream>

class Logger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) noexcept override {
		if (severity <= Severity::kWARNING) {
			std::cout << "[TensorRT] " << msg << std::endl;
		}
	}

	static Logger& instance() {
		static Logger logger;
		return logger;
	}

private:
	Logger() = default;
};

#endif // LOGGER_HPP
