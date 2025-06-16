#include "inference/InferenceManager.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * @brief Exemplo de como usar o InferenceManager com modelos PyTorch/ONNX e
 * Keras
 */
void demonstrateInferenceManager() {
  InferenceManager manager;

  // Carregar modelo PyTorch/ONNX (por exemplo, um modelo de lane detection)
  std::string onnx_model_path = "/path/to/your/lane_detection_model.onnx";
  if (manager.loadModel("lane_detection", onnx_model_path,
                        ModelType::PYTORCH_ONNX)) {
    std::cout << "Modelo ONNX carregado com sucesso!" << std::endl;
  }

  // Carregar modelo Keras (por exemplo, um modelo de classification)
  std::string keras_model_path = "/path/to/your/classification_model.h5";
  if (manager.loadModel("object_classifier", keras_model_path,
                        ModelType::KERAS)) {
    std::cout << "Modelo Keras carregado com sucesso!" << std::endl;
  }

  // Carregar uma imagem de teste
  cv::Mat test_image = cv::imread("/path/to/test_image.jpg");
  if (test_image.empty()) {
    std::cerr << "Erro: Não foi possível carregar imagem de teste" << std::endl;
    return;
  }

  // Listar modelos carregados
  auto loaded_models = manager.getLoadedModels();
  std::cout << "Modelos carregados: ";
  for (const auto &model : loaded_models) {
    std::cout << model << " ";
  }
  std::cout << std::endl;

  // Usar modelo ONNX para lane detection
  if (manager.isModelLoaded("lane_detection")) {
    std::cout << "Executando lane detection..." << std::endl;
    manager.selectModel("lane_detection");
    cv::Mat lane_result = manager.predict(test_image);
    // Processar resultado do lane detection
  }

  // Usar modelo Keras para classification
  if (manager.isModelLoaded("object_classifier")) {
    std::cout << "Executando object classification..." << std::endl;
    manager.selectModel("object_classifier");
    cv::Mat class_result = manager.predict(test_image);
    // Processar resultado da classificação
  }

  // Alternar entre modelos conforme necessário
  manager.selectModel("lane_detection");
  cv::Mat result1 = manager.predict(test_image);

  manager.selectModel("object_classifier");
  cv::Mat result2 = manager.predict(test_image);
}

/**
 * @brief Exemplo de configuração específica para cada tipo de modelo
 */
void demonstrateModelSpecificConfiguration() {
  // Configurar modelo ONNX específico
  ONNXInferencer onnx_inferencer("/path/to/model.onnx");
  // Configurações específicas para ONNX podem ser adicionadas aqui

  // Configurar modelo Keras específico
  KerasInferencer keras_inferencer("/path/to/model.h5");
  keras_inferencer.setInputSize(cv::Size(224, 224)); // ImageNet size
  keras_inferencer.setNormalization(
      cv::Scalar(0.485, 0.456, 0.406), // ImageNet mean
      cv::Scalar(0.229, 0.224, 0.225)  // ImageNet std
  );

  // Usar modelos diretamente
  cv::Mat test_image = cv::imread("/path/to/test_image.jpg");
  if (!test_image.empty()) {
    cv::Mat onnx_result = onnx_inferencer.predict(test_image);
    cv::Mat keras_result = keras_inferencer.predict(test_image);
  }
}

int main() {
  std::cout << "=== Demonstração do Sistema de Inferência Multi-Modelo ==="
            << std::endl;

  try {
    demonstrateInferenceManager();
    demonstrateModelSpecificConfiguration();
  } catch (const std::exception &e) {
    std::cerr << "Erro: " << e.what() << std::endl;
    return -1;
  }

  std::cout << "Demonstração concluída!" << std::endl;
  return 0;
}
