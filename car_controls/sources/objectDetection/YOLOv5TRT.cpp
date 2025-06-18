#include "../../includes/objectDetection/YOLOv5TRT.hpp"

std::string YOLOv5TRT::lastClassName = "";
std::chrono::steady_clock::time_point YOLOv5TRT::lastNotificationTime = std::chrono::steady_clock::now();

YOLOv5TRT::YOLOv5TRT(const std::string& enginePath, const std::string& labelPath)
	: labelManager(labelPath) {
	// Correção: verificar valores de retorno do system()
	int result1 = system("sudo nvpmodel -m 0");
	if (result1 != 0) {
		std::cerr << "[AVISO] Falha ao configurar nvpmodel" << std::endl;
	}

	int result2 = system("sudo jetson_clocks");
	if (result2 != 0) {
		std::cerr << "[AVISO] Falha ao configurar jetson_clocks" << std::endl;
	}

	// Configurar OpenCV para usar CUDA
	cv::cuda::setDevice(0);
	loadEngine(enginePath);
	allocateBuffers();

	// Pré-alocar buffers reutilizáveis
	channels.resize(3);
	hostDataBuffer = new float[3*640*640];

	num_classes = static_cast<int>(labelManager.getNumClasses());

	Publisher::instance(5557); //Initialize publisher
}

YOLOv5TRT::~YOLOv5TRT() {
	cudaStreamDestroy(stream);
	delete[] hostDataBuffer;
	delete[] outputHost;
	cudaFree(inputDevice);
	cudaFree(outputDevice);
}

/**
 * @brief Calcula o volume (número total de elementos) de um tensor dado suas dimensões.
 * @param dims Dimensões do tensor.
 * @return Volume total.
 */
size_t YOLOv5TRT::calculateVolume(const nvinfer1::Dims& dims) {
	size_t volume = 1;
	for (int i = 0; i < dims.nbDims; ++i) {
		volume *= dims.d[i];
	}
	return volume;
}

void YOLOv5TRT::loadEngine(const std::string& enginePath) {
	std::ifstream file(enginePath, std::ios::binary);
	if (!file) {
		std::cerr << "[ERRO] Falha ao carregar o engine TensorRT!" << std::endl;
		exit(EXIT_FAILURE);
	}

	file.seekg(0, file.end);
	size_t size = file.tellg();
	file.seekg(0, file.beg);
	std::vector<char> engineData(size);
	file.read(engineData.data(), size);

	runtime = createInferRuntime(logger);
	engine = runtime->deserializeCudaEngine(engineData.data(), size);
	if (!engine) {
		std::cerr << "[ERRO] Falha ao desserializar o engine TensorRT!" << std::endl;
		exit(EXIT_FAILURE);
	}

	context = engine->createExecutionContext();
}

void YOLOv5TRT::allocateBuffers() {
	inputSize = calculateVolume(engine->getBindingDimensions(0)) * sizeof(float);
	outputSize = calculateVolume(engine->getBindingDimensions(1)) * sizeof(float);

	cudaMalloc(&inputDevice, inputSize);
	cudaMalloc(&outputDevice, outputSize);

	bindings.push_back(inputDevice);
	bindings.push_back(outputDevice);

	cudaStreamCreate(&stream);
	outputHost = new float[outputSize / sizeof(float)];
}

/**
 * @brief Executa inferência em uma imagem.
 * @param image Imagem de entrada (cv::Mat BGR).
 * @return Vetor de floats com a saída do modelo.
 */
std::vector<float> YOLOv5TRT::infer(const cv::Mat& image) {
	// Usar GPU para processamento
	gpu_image.upload(image);
	cv::cuda::resize(gpu_image, gpu_resized, cv::Size(640, 640));
	gpu_resized.convertTo(gpu_float, CV_32FC3, 1.0/255.0);

	// Download otimizado
	gpu_float.download(blob);
	cv::split(blob, channels);

	// Cópia otimizada dos canais (HWC -> CHW)
	for (int c = 0; c < 3; c++) {
		memcpy(hostDataBuffer + c*640*640,
				channels[c].ptr<float>(),
				640*640*sizeof(float));
	}

	// Copiar dados para GPU
	cudaMemcpyAsync(inputDevice, hostDataBuffer, 3*640*640*sizeof(float),
					cudaMemcpyHostToDevice, stream);

	// Executar inferência
	context->enqueueV2(bindings.data(), stream, nullptr);

	// Copiar resultados para o host
	cudaMemcpyAsync(outputHost, outputDevice, outputSize,
					cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	return std::vector<float>(outputHost, outputHost + outputSize / sizeof(float));
}

/**
 * @brief Pós-processa a saída do modelo, aplicando threshold de confiança e NMS.
 * @param output Saída bruta do modelo.
 * @param num_classes Número de classes.
 * @param conf_thresh Threshold de confiança.
 * @param nms_thresh Threshold de NMS.
 * @return Vetor de detecções finais.
 */
std::vector<Detection> YOLOv5TRT::postprocess(const std::vector<float>& output, int num_classes, float conf_thresh, float nms_thresh) {
	std::vector<Detection> dets;
	int num_preds = output.size() / (5 + num_classes);

	for (int i = 0; i < num_preds; ++i) {
		const float* pred = &output[i * (5 + num_classes)];
		float obj = pred[4];
		if (obj < conf_thresh) continue;

		// Encontrar a classe com maior probabilidade
		float max_cls = pred[5];
		int class_id = 0;
		for (int c = 1; c < num_classes; ++c) {
			if (pred[5 + c] > max_cls) {
				max_cls = pred[5 + c];
				class_id = c;
			}
		}

		float score = obj * max_cls;
		if (score < conf_thresh) continue;

		dets.push_back({pred[0], pred[1], pred[2], pred[3], score, class_id});
	}

	// NMS
	std::vector<Detection> result;
	std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) {
		return a.conf > b.conf;
	});

	std::vector<bool> removed(dets.size(), false);
	for (size_t i = 0; i < dets.size(); ++i) {
		if (removed[i]) continue;
		result.push_back(dets[i]);

		for (size_t j = i + 1; j < dets.size(); ++j) {
			if (removed[j]) continue;

			// Calcular IoU
			float xx1 = std::max(dets[i].x - dets[i].w/2, dets[j].x - dets[j].w/2);
			float yy1 = std::max(dets[i].y - dets[i].h/2, dets[j].y - dets[j].h/2);
			float xx2 = std::min(dets[i].x + dets[i].w/2, dets[j].x + dets[j].w/2);
			float yy2 = std::min(dets[i].y + dets[i].h/2, dets[j].y + dets[j].h/2);

			float w = std::max(0.0f, xx2 - xx1);
			float h = std::max(0.0f, yy2 - yy1);
			float inter = w * h;
			float area1 = dets[i].w * dets[i].h;
			float area2 = dets[j].w * dets[j].h;
			float ovr = inter / (area1 + area2 - inter);

			if (ovr > nms_thresh) removed[j] = true;
		}
	}
	return result;
}

/**
 * @brief Função principal. Inicializa recursos, executa loop de inferência e exibe resultados.
 * @return 0 em caso de sucesso.
 */
void YOLOv5TRT::process_image(const cv::Mat& frame) {
	auto output = infer(frame);
	std::vector<Detection> dets = postprocess(output, num_classes, conf_thresh, nms_thresh);

	for (const auto& det : dets) {
		// Converter coordenadas normalizadas para absolutas
		// Corrigir: tratar det.x, det.y, det.w, det.h como coordenadas absolutas (input 640x640)
		float x_center = (det.x / 640.0f) * frame.cols;
		float y_center = (det.y / 640.0f) * frame.rows;
		float width = (det.w / 640.0f) * frame.cols;
		float height = (det.h / 640.0f) * frame.rows;

		// Calcular coordenadas do retângulo
		int x1 = static_cast<int>(x_center - width / 2);
		int y1 = static_cast<int>(y_center - height / 2);
		int x2 = static_cast<int>(x_center + width / 2);
		int y2 = static_cast<int>(y_center + height / 2);

		// Garantir que as coordenadas estão dentro da imagem
		x1 = std::max(0, std::min(x1, frame.cols - 1));
		y1 = std::max(0, std::min(y1, frame.rows - 1));
		x2 = std::max(0, std::min(x2, frame.cols - 1));
		y2 = std::max(0, std::min(y2, frame.rows - 1));

		// Verificar se o retângulo é válido
		if (x2 > x1 && y2 > y1) {
			std::string className = labelManager.getLabel(det.class_id);
			std::cout << "Object found: " << className << " at (" << x1 << "," << y1 << ")-(" << x2 << "," << y2 << ")" << std::endl;

			auto now = std::chrono::steady_clock::now();
			auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastNotificationTime).count();

			if (className != lastClassName || elapsedMs > 2000) {  // Only notify again if different or 2s passed
				lastClassName = className;
				lastNotificationTime = now;
				Publisher::instance(5557)->publish("notification", className);
			}

			/* if (className != lastClassName)
			{
				lastClassName = className;
				std::lock_guard<std::mutex> lock(pubMutex);
				Publisher::instance(5557)->publish("notification", className);
			} */
			// Desenhar retângulo usando coordenadas Point
			/* cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 3);

			// Desenhar label
			std::string label = className + " (" + std::to_string(int(det.conf * 100)) + "%)";

			int baseline = 0;
			cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

			// Fundo do texto
			cv::rectangle(frame,
						cv::Point(x1, y1 - textSize.height - 10),
						cv::Point(x1 + textSize.width, y1),
						cv::Scalar(0, 255, 0), -1);

			// Texto
			cv::putText(frame, label,
						cv::Point(x1, y1 - 5),
						cv::FONT_HERSHEY_SIMPLEX, 0.6,
						cv::Scalar(0, 0, 0), 2); */
		} /* else {
			std::cout << "Invalid rectangle coordinates for detection: "
					<< "x1=" << x1 << ", y1=" << y1
					<< ", x2=" << x2 << ", y2=" << y2
					<< ", width=" << width << ", height=" << height
					<< ", det.x=" << det.x << ", det.y=" << det.y
					<< ", det.w=" << det.w << ", det.h=" << det.h
					<< ", class_id=" << det.class_id << ", conf=" << det.conf << std::endl;
		} */
	}
}
