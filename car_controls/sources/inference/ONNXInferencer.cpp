#include "ONNXInferencer.hpp"
#include <exception>
#include <iostream>

ONNXInferencer::ONNXInferencer(const std::string &modelPath, const cv::Size &inputSize,
                               const cv::Scalar &meanValues, double scaleFactor, bool swapRB)
    : modelPath(modelPath), inputSize(inputSize), isModelLoaded(false), meanValues(meanValues),
      scaleFactor(scaleFactor), swapRB(swapRB) {

	if(!loadModel(modelPath)) {
		std::cerr << "[ONNXInferencer] Erro ao carregar modelo: " << modelPath << std::endl;
	}
}

ONNXInferencer::~ONNXInferencer() {
	// OpenCV DNN cuida da limpeza automaticamente
}

bool ONNXInferencer::loadModel(const std::string &modelPath) {
	try {
		// Carregar modelo ONNX usando OpenCV DNN
		net = cv::dnn::readNetFromONNX(modelPath);

		if(net.empty()) {
			std::cerr << "[ONNXInferencer] Erro: Modelo vazio ou inválido" << std::endl;
			return false;
		}

		// Configurar backend e target
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

		// Tentar usar GPU se disponível
		if(cv::cuda::getCudaEnabledDeviceCount() > 0) {
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
			std::cout << "[ONNXInferencer] Usando GPU para inferência" << std::endl;
		} else {
			std::cout << "[ONNXInferencer] Usando CPU para inferência" << std::endl;
		}

		isModelLoaded = true;
		std::cout << "[ONNXInferencer] Modelo carregado com sucesso: " << modelPath << std::endl;
		return true;

	} catch(const cv::Exception &e) {
		std::cerr << "[ONNXInferencer] Erro OpenCV ao carregar modelo: " << e.what() << std::endl;
		return false;
	} catch(const std::exception &e) {
		std::cerr << "[ONNXInferencer] Erro ao carregar modelo: " << e.what() << std::endl;
		return false;
	}
}

cv::Mat ONNXInferencer::preprocess(const cv::Mat &input) {
	if(input.empty()) {
		std::cerr << "[ONNXInferencer] Imagem de entrada vazia" << std::endl;
		return cv::Mat();
	}

	cv::Mat blob;
	try {
		// Criar blob da imagem com redimensionamento e normalização
		cv::dnn::blobFromImage(input, blob, scaleFactor, inputSize, meanValues, swapRB, false,
		                       CV_32F);

	} catch(const cv::Exception &e) {
		std::cerr << "[ONNXInferencer] Erro no pré-processamento: " << e.what() << std::endl;
		return cv::Mat();
	}

	return blob;
}

std::vector<cv::Mat> ONNXInferencer::infer(const cv::Mat &preprocessedInput) {
	std::vector<cv::Mat> outputs;

	if(!isModelLoaded) {
		std::cerr << "[ONNXInferencer] Modelo não carregado" << std::endl;
		return outputs;
	}

	if(preprocessedInput.empty()) {
		std::cerr << "[ONNXInferencer] Entrada pré-processada vazia" << std::endl;
		return outputs;
	}

	try {
		// Definir entrada da rede
		net.setInput(preprocessedInput);

		// Executar inferência
		cv::Mat output;
		net.forward(output);

		outputs.push_back(output);

	} catch(const cv::Exception &e) {
		std::cerr << "[ONNXInferencer] Erro durante inferência: " << e.what() << std::endl;
	} catch(const std::exception &e) {
		std::cerr << "[ONNXInferencer] Erro durante inferência: " << e.what() << std::endl;
	}

	return outputs;
}

cv::Mat ONNXInferencer::postprocess(const std::vector<cv::Mat> &modelOutput,
                                    const cv::Mat &originalInput) {
	if(modelOutput.empty() || originalInput.empty()) {
		std::cerr << "[ONNXInferencer] Saída do modelo ou imagem original vazia" << std::endl;
		return cv::Mat();
	}

	// Por padrão, retorna a primeira saída redimensionada para o tamanho original
	cv::Mat result = modelOutput[0];

	// Se a saída for uma máscara de segmentação, redimensionar para o tamanho
	// original
	if(result.dims == 4) { // Formato batch
		// Assumindo formato [1, channels, height, width]
		std::vector<int> sizes = {result.size[2], result.size[3]};
		result = result.reshape(1, sizes);
	}

	if(result.size() != originalInput.size()) {
		cv::resize(result, result, originalInput.size());
	}

	return result;
}

cv::Mat ONNXInferencer::predict(const cv::Mat &input) {
	// Pipeline completo de inferência
	cv::Mat preprocessed = preprocess(input);
	if(preprocessed.empty()) {
		return cv::Mat();
	}

	std::vector<cv::Mat> outputs = infer(preprocessed);
	if(outputs.empty()) {
		return cv::Mat();
	}

	cv::Mat result = postprocess(outputs, input);
	return result;
}

cv::cuda::GpuMat ONNXInferencer::makePrediction(const cv::cuda::GpuMat &gpuImage) {
	// Converter de GPU para CPU
	cv::Mat cpuImage;
	gpuImage.download(cpuImage);

	// Fazer predição
	cv::Mat result = predict(cpuImage);

	// Converter de volta para GPU
	cv::cuda::GpuMat gpuResult;
	gpuResult.upload(result);

	return gpuResult;
}

void ONNXInferencer::doInference(const cv::Mat &frame) {
	if(!isModelLoaded) {
		std::cerr << "[ONNXInferencer] Erro: Modelo não carregado!" << std::endl;
		return;
	}

	cv::Mat result = predict(frame);

	// Aqui você pode processar o resultado conforme necessário
	std::cout << "[ONNXInferencer] Inferência concluída. Resultado: " << result.size() << std::endl;
}
