#include <NvInfer.h>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @class LabelManager
 * @brief Gerencia o carregamento e acesso às labels das classes.
 */
class LabelManager
{
private:
  std::vector<std::string> labels; ///< Lista de labels

public:
	/**
	 * @brief Construtor que carrega as labels de um arquivo.
	 * @param labelPath Caminho para o arquivo de labels.
	 */
	LabelManager(const std::string &labelPath);

	void loadLabels(const std::string &labelPath);
	std::string getLabel(int classId) const;
	size_t getNumClasses() const;
};
