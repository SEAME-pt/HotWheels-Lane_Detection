#ifndef ONNXINFERENCER_HPP
#define ONNXINFERENCER_HPP

#include "IInferencer.hpp"
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief Classe para inferência usando modelos ONNX com OpenCV DNN
 *
 * Esta classe implementa a interface IInferencer para carregar e executar
 * modelos ONNX usando o módulo DNN do OpenCV. Suporta modelos .onnx e .pt
 * convertidos para ONNX.
 */
class ONNXInferencer : public IInferencer {
private:
  cv::dnn::Net net;      // Rede neural do OpenCV DNN
  std::string modelPath; // Caminho para o modelo
  cv::Size inputSize;    // Tamanho de entrada esperado
  bool isModelLoaded;    // Flag indicando se o modelo foi carregado

  // Parâmetros de normalização
  cv::Scalar meanValues; // Valores médios para normalização
  double scaleFactor;    // Fator de escala para normalização
  bool swapRB;           // Se deve trocar canais R e B

public:
  /**
   * @brief Construtor
   * @param modelPath Caminho para o arquivo do modelo ONNX (.onnx)
   * @param inputSize Tamanho da imagem de entrada (default: 640x640)
   * @param meanValues Valores médios para normalização (default: 0,0,0)
   * @param scaleFactor Fator de escala (default: 1/255.0)
   * @param swapRB Se deve trocar canais R e B (default: true para RGB)
   */
  ONNXInferencer(const std::string &modelPath,
                 const cv::Size &inputSize = cv::Size(640, 640),
                 const cv::Scalar &meanValues = cv::Scalar(0, 0, 0),
                 double scaleFactor = 1.0 / 255.0, bool swapRB = true);

  ~ONNXInferencer() override;

  // Implementação da interface IInferencer
  cv::cuda::GpuMat makePrediction(const cv::cuda::GpuMat &gpuImage) override;
  void doInference(const cv::Mat &frame) override;

  // Métodos específicos para ONNX
  bool loadModel(const std::string &modelPath);
  cv::Mat preprocess(const cv::Mat &input);
  std::vector<cv::Mat> infer(const cv::Mat &preprocessedInput);
  cv::Mat postprocess(const std::vector<cv::Mat> &modelOutput,
                      const cv::Mat &originalInput);

  // Métodos específicos para ONNX
  bool isLoaded() const { return isModelLoaded; }
  void setInputSize(const cv::Size &size) { inputSize = size; }
  void setNormalizationParams(const cv::Scalar &mean, double scale,
                              bool swapChannels) {
    meanValues = mean;
    scaleFactor = scale;
    swapRB = swapChannels;
  }

  // Método conveniente para inferência completa
  cv::Mat predict(const cv::Mat &input);
};

#endif // ONNXINFERENCER_HPP
