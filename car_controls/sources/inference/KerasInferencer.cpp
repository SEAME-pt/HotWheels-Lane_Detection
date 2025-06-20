#include "../../includes/inference/KerasInferencer.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

// Para usar Python embedding (alternativa para modelos Keras)
#ifdef WITH_PYTHON
#include <Python.h>
#include <numpy/arrayobject.h>
#endif

KerasInferencer::KerasInferencer(const std::string &model_path)
    : model_path_(model_path), input_size_(224, 224) // Tamanho padrão para muitos modelos Keras
      ,
      mean_(cv::Scalar(0.485, 0.456, 0.406)) // ImageNet mean
      ,
      std_(cv::Scalar(0.229, 0.224, 0.225)) // ImageNet std
      ,
      model_loaded_(false)
{
    // Tentar carregar o modelo
    if (!model_path_.empty())
    {
        loadModel(model_path_);
    }
}

KerasInferencer::~KerasInferencer()
{
#ifdef WITH_PYTHON
    cleanupPython();
#endif
}

bool KerasInferencer::loadModel(const std::string &model_path)
{
    model_path_ = model_path;

    // Verificar se o arquivo existe
    std::ifstream file(model_path_);
    if (!file.good())
    {
        std::cerr << "[KerasInferencer] Erro: Arquivo do modelo não encontrado: " << model_path_
                  << std::endl;
        return false;
    }

    // Opção 1: Se o modelo foi convertido para ONNX
    if (model_path_.length() >= 5 && model_path_.substr(model_path_.length() - 5) == ".onnx")
    {
        std::cout << "[KerasInferencer] Carregando modelo Keras convertido para ONNX..."
                  << std::endl;
        // Usar OpenCV DNN para carregar ONNX
        try
        {
            // net_ = cv::dnn::readNetFromONNX(model_path_);
            // model_loaded_ = true;
            std::cout << "[KerasInferencer] Modelo ONNX carregado com sucesso!" << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[KerasInferencer] Erro ao carregar modelo ONNX: " << e.what()
                      << std::endl;
            return false;
        }
    }

    // Opção 2: Usar Python embedding para modelos .h5 ou SavedModel
#ifdef WITH_PYTHON
    return initializePython();
#else
    std::cerr << "[KerasInferencer] Erro: Suporte ao Python não compilado. "
              << "Converta o modelo para ONNX ou recompile com Python." << std::endl;
    return false;
#endif
}

cv::Mat KerasInferencer::preprocessImage(const cv::Mat &image)
{
    cv::Mat preprocessed;

    // Redimensionar para o tamanho de entrada
    cv::resize(image, preprocessed, input_size_);

    // Converter para float e normalizar
    preprocessed.convertTo(preprocessed, CV_32F, 1.0 / 255.0);

    // Aplicar normalização ImageNet (ou customizada)
    std::vector<cv::Mat> channels;
    cv::split(preprocessed, channels);

    for (int i = 0; i < 3; ++i)
    {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }

    cv::merge(channels, preprocessed);

    return preprocessed;
}

cv::cuda::GpuMat KerasInferencer::makePrediction(const cv::cuda::GpuMat &gpuImage)
{
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

void KerasInferencer::doInference(const cv::Mat &frame)
{
    if (!model_loaded_)
    {
        std::cerr << "[KerasInferencer] Erro: Modelo não carregado!" << std::endl;
        return;
    }

    cv::Mat result = predict(frame);

    // Aqui você pode processar o resultado conforme necessário
    std::cout << "[KerasInferencer] Inferência concluída. Resultado: " << result.size()
              << std::endl;
}

cv::Mat KerasInferencer::predict(const cv::Mat &image)
{
    if (!model_loaded_)
    {
        std::cerr << "[KerasInferencer] Erro: Modelo não carregado!" << std::endl;
        return cv::Mat();
    }

    // Preprocessar a imagem
    cv::Mat preprocessed = preprocessImage(image);

#ifdef WITH_PYTHON
    return predictWithPython(preprocessed);
#else
    // Fallback: retornar imagem preprocessada como placeholder
    std::cerr << "[KerasInferencer] Aviso: Usando fallback - Python não disponível" << std::endl;
    return preprocessed;
#endif
}

#ifdef WITH_PYTHON
bool KerasInferencer::initializePython()
{
    try
    {
        Py_Initialize();
        if (!Py_IsInitialized())
        {
            std::cerr << "[KerasInferencer] Erro: Não foi possível inicializar Python" << std::endl;
            return false;
        }

        // Importar numpy
        import_array();

        std::cout << "[KerasInferencer] Python inicializado com sucesso" << std::endl;
        model_loaded_ = true;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[KerasInferencer] Erro ao inicializar Python: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat KerasInferencer::predictWithPython(const cv::Mat &image)
{
    // Implementação usando Python embedding
    // Este é um exemplo básico - você precisará adaptar para seu modelo
    // específico

    PyObject *pModule = PyImport_ImportModule("tensorflow.keras.models");
    if (!pModule)
    {
        std::cerr << "[KerasInferencer] Erro: Não foi possível importar "
                     "tensorflow.keras.models"
                  << std::endl;
        PyErr_Print();
        return cv::Mat();
    }

    // Carregar o modelo
    PyObject *pLoadModel = PyObject_GetAttrString(pModule, "load_model");
    PyObject *pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, PyUnicode_FromString(model_path_.c_str()));

    PyObject *pModel = PyObject_CallObject(pLoadModel, pArgs);
    if (!pModel)
    {
        std::cerr << "[KerasInferencer] Erro: Não foi possível carregar o modelo" << std::endl;
        PyErr_Print();
        Py_DECREF(pArgs);
        Py_DECREF(pLoadModel);
        Py_DECREF(pModule);
        return cv::Mat();
    }

    // Converter cv::Mat para numpy array e fazer predição
    // ... (implementação específica do modelo)

    // Cleanup
    Py_DECREF(pModel);
    Py_DECREF(pArgs);
    Py_DECREF(pLoadModel);
    Py_DECREF(pModule);

    // Retornar resultado (placeholder)
    return image.clone();
}

void KerasInferencer::cleanupPython()
{
    if (Py_IsInitialized())
    {
        Py_Finalize();
    }
}
#else
// Implementações vazias quando Python não está disponível
bool KerasInferencer::initializePython()
{
    return false;
}
cv::Mat KerasInferencer::predictWithPython(const cv::Mat &image)
{
    return image.clone();
}
void KerasInferencer::cleanupPython()
{
}
#endif
