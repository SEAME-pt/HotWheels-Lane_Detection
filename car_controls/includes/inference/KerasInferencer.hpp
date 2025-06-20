#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "IInferencer.hpp"

// Para usar modelos Keras, geralmente precisamos do TensorFlow C++ API
// ou converter o modelo para um formato suportado (como ONNX)

class KerasInferencer : public IInferencer {
	private:
		std::string model_path_;
		cv::Size input_size_;
		cv::Scalar mean_;
		cv::Scalar std_;
		bool model_loaded_;

		// Para TensorFlow C++ (se disponível)
		// std::unique_ptr<tensorflow::Session> session_;
		// std::unique_ptr<tensorflow::GraphDef> graph_def_;

		// Métodos internos
		cv::Mat preprocessImage(const cv::Mat &image);
		std::vector<float> matToVector(const cv::Mat &mat);
		cv::Mat postprocessOutput(const std::vector<float> &output);

	public:
		KerasInferencer(const std::string &model_path);
		~KerasInferencer();

		// Inherited from IInferencer
		cv::cuda::GpuMat makePrediction(const cv::cuda::GpuMat &gpuImage) override;
		void doInference(const cv::Mat &frame) override;

		// Keras specific methods
		bool loadModel(const std::string &model_path);
		cv::Mat predict(const cv::Mat &image);

		// Alternative: usar Python embedding para chamar Keras
		bool initializePython();
		cv::Mat predictWithPython(const cv::Mat &image);
		void cleanupPython();

		// Configuration methods
		void setInputSize(const cv::Size &size) {
			input_size_ = size;
		}
		void setNormalization(const cv::Scalar &mean, const cv::Scalar &std) {
			mean_ = mean;
			std_ = std;
		}

		// Utility methods
		bool isModelLoaded() const {
			return model_loaded_;
		}
		cv::Size getInputSize() const {
			return input_size_;
		}
};
