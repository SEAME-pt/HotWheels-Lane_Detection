#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class TensorRTInferencer {
private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    } logger;

    std::vector<char> engineData;
    nvinfer1::IRuntime* runtime;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* context;

    int inputBindingIndex;
    int outputBindingIndex;
    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;
    cv::Size inputSize;

    // Pre-calculated counts and sizes
    size_t inputElementCount;
    size_t outputElementCount;
    size_t inputByteSize;
    size_t outputByteSize;

    // Persistent CUDA resources
    void* deviceInput;
    void* deviceOutput;
    cudaStream_t stream;
    std::vector<void*> bindings;

    // Pinned host memory
    float* hostInput;
    float* hostOutput;

    std::vector<char> readEngineFile(const std::string& enginePath);
    void cleanupResources();

public:
    TensorRTInferencer(const std::string& enginePath);
    ~TensorRTInferencer();

    //cv::Mat preprocessImage(const cv::Mat& image);
    cv::cuda::GpuMat preprocessImage(const cv::cuda::GpuMat& gpuImage);
    //std::vector<float> runInference(const cv::Mat& inputImage);
    void runInference(const cv::cuda::GpuMat& gpuInput);
    //cv::Mat makePrediction(const cv::Mat& image);
    cv::cuda::GpuMat makePrediction(const cv::cuda::GpuMat& gpuImage);
};
