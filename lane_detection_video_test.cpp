#include "car_controls/includes/inference/TensorRTInferencer.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
	if(argc < 2) {
		std::cerr << "Uso: " << argv[0] << " <caminho_do_video>" << std::endl;
		return 1;
	}
	std::string video_path = argv[1];
	cv::VideoCapture cap(video_path);
	if(!cap.isOpened()) {
		std::cerr << "Erro ao abrir o vídeo: " << video_path << std::endl;
		return 1;
	}
	TensorRTInferencer inferencer("/home/jetson/models/lane-detection/model.engine");
	cv::Mat frame;
	int frame_idx = 0;
	auto t_start = std::chrono::high_resolution_clock::now();
	while(cap.read(frame)) {
		std::cout << "[Frame] " << frame_idx << " shape: " << frame.cols << "x" << frame.rows
		          << std::endl;
		try {
			// Executa inferência e obtém a máscara binária
			inferencer.doInference(frame);           // Executa inferência (void)
			cv::Mat mask = inferencer.getLastMask(); // Obtém a máscara após inferência
			if(mask.empty()) {
				std::cerr << "[Aviso] Máscara vazia no frame " << frame_idx << std::endl;
				continue;
			}
			cv::imshow("Lane Mask", mask);
			// Salvar máscara (opcional)
			// cv::imwrite("mask_" + std::to_string(frame_idx) + ".png", mask);
		} catch(const std::exception &e) {
			std::cerr << "Erro na inferência: " << e.what() << std::endl;
		}
		frame_idx++;
		if(cv::waitKey(30) == 27)
			break; // ESC para sair
	}
	auto t_end = std::chrono::high_resolution_clock::now();
	double elapsed = std::chrono::duration<double>(t_end - t_start).count();
	std::cout << "Processados " << frame_idx << " frames em " << elapsed << " s ("
	          << (frame_idx / elapsed) << " FPS)" << std::endl;
	cap.release();
	cv::destroyAllWindows();
	return 0;
}
