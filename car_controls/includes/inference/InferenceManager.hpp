#pragma once

#include <string>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>

#include "IInferencer.hpp"
#include "ONNXInferencer.hpp"
#include "KerasInferencer.hpp"

enum class ModelType {
    PYTORCH_ONNX,    // Modelos PyTorch (.pt) ou ONNX (.onnx)
    KERAS,           // Modelos Keras (.h5, SavedModel)
    AUTO_DETECT      // Detectar automaticamente baseado na extensão
};

/**
 * @brief Gerenciador de inferenciadores que suporta múltiplos tipos de modelos
 * 
 * Esta classe permite carregar e usar diferentes tipos de modelos de ML:
 * - Modelos PyTorch/ONNX usando ONNXInferencer
 * - Modelos Keras usando KerasInferencer
 */
class InferenceManager {
private:
    std::map<std::string, std::unique_ptr<IInferencer>> inferencers_;
    std::string current_model_name_;
    ModelType current_model_type_;
    
    // Detectar tipo do modelo baseado na extensão do arquivo
    ModelType detectModelType(const std::string& model_path);
    
public:
    InferenceManager();
    ~InferenceManager();
    
    /**
     * @brief Carregar um modelo
     * @param model_name Nome identificador para o modelo
     * @param model_path Caminho para o arquivo do modelo
     * @param model_type Tipo do modelo (AUTO_DETECT por padrão)
     * @return true se carregado com sucesso
     */
    bool loadModel(const std::string& model_name, 
                   const std::string& model_path, 
                   ModelType model_type = ModelType::AUTO_DETECT);
    
    /**
     * @brief Selecionar modelo ativo para inferência
     * @param model_name Nome do modelo carregado
     * @return true se modelo existe e foi selecionado
     */
    bool selectModel(const std::string& model_name);
    
    /**
     * @brief Fazer predição com o modelo ativo
     * @param image Imagem de entrada
     * @return Resultado da predição
     */
    cv::Mat predict(const cv::Mat& image);
    
    /**
     * @brief Fazer predição com modelo específico
     * @param model_name Nome do modelo
     * @param image Imagem de entrada
     * @return Resultado da predição
     */
    cv::Mat predict(const std::string& model_name, const cv::Mat& image);
    
    /**
     * @brief Verificar se um modelo está carregado
     * @param model_name Nome do modelo
     * @return true se modelo está carregado
     */
    bool isModelLoaded(const std::string& model_name) const;
    
    /**
     * @brief Obter lista de modelos carregados
     * @return Vetor com nomes dos modelos carregados
     */
    std::vector<std::string> getLoadedModels() const;
    
    /**
     * @brief Remover um modelo da memória
     * @param model_name Nome do modelo
     */
    void unloadModel(const std::string& model_name);
    
    /**
     * @brief Limpar todos os modelos
     */
    void clear();
    
    /**
     * @brief Obter tipo do modelo ativo
     * @return Tipo do modelo atual
     */
    ModelType getCurrentModelType() const { return current_model_type_; }
    
    /**
     * @brief Obter nome do modelo ativo
     * @return Nome do modelo atual
     */
    std::string getCurrentModelName() const { return current_model_name_; }
};
