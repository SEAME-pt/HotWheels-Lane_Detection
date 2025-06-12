#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <chrono>
#include "LabelManager.hpp"
#include "../../../ZeroMQ/Subscriber.hpp"
#include "../../../ZeroMQ/Publisher.hpp"

using namespace nvinfer1;

/**
 * @struct Detection
 * @brief Estrutura para armazenar uma detecção (bounding box, confiança e classe).
 */
struct Detection {
	float x, y, w, h, conf; ///< Coordenadas e confiança
	int class_id; ///< Índice da classe
};

/**
 * @class YOLOv5TRT
 * @brief Gerencia o engine TensorRT e executa inferência do modelo YOLOv5.
 */
class YOLOv5TRT {
public:
	/**
	 * @brief Construtor. Carrega o engine e aloca buffers.
	 * @param enginePath Caminho para o arquivo do engine TensorRT.
	 */
	YOLOv5TRT(const std::string& enginePath, const std::string& labelPath);

	/**
	 * @brief Destrutor. Libera recursos.
	 */
	~YOLOv5TRT();

	void process_image(const cv::Mat& frame);

private:
	class Logger : public ILogger {
	public:
		void log(Severity severity, const char* msg) noexcept override {
			if (severity <= Severity::kWARNING) {
				std::cout << "[TensorRT] " << msg << std::endl;
			}
		}
	} logger;

	// Buffers reutilizáveis
	cv::cuda::GpuMat gpu_image, gpu_resized, gpu_float;
	cv::Mat blob;
	std::vector<cv::Mat> channels;
	float* hostDataBuffer;

	IRuntime* runtime{nullptr};
	ICudaEngine* engine{nullptr};
	IExecutionContext* context{nullptr};
	cudaStream_t stream;
	void* inputDevice{nullptr};
	void* outputDevice{nullptr};
	float* outputHost{nullptr};
	size_t inputSize{0};
	size_t outputSize{0};
	std::vector<void*> bindings;

	LabelManager labelManager;
	float conf_thresh = 0.25f;
	float nms_thresh = 0.45f;
	int num_classes;

	void loadEngine(const std::string& enginePath);
	void allocateBuffers();
	std::vector<float> infer(const cv::Mat& image);

	static std::string lastClassName;
	static std::chrono::steady_clock::time_point lastNotificationTime;

protected:
	size_t calculateVolume(const nvinfer1::Dims& dims);
	std::vector<Detection> postprocess(const std::vector<float>& output, int num_classes, float conf_thresh, float nms_thresh);

public:
};
