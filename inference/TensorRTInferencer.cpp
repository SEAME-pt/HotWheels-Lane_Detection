#include "TensorRTInferencer.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <numeric>

void TensorRTInferencer::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
}

TensorRTInferencer::TensorRTInferencer(const std::string& enginePath) :
    runtime(nullptr),
    engine(nullptr),
    context(nullptr),
    inputBindingIndex(-1),
    outputBindingIndex(-1),
    inputSize(256, 256),
    deviceInput(nullptr),
    deviceOutput(nullptr),
    stream(nullptr),
    hostInput(nullptr),
    hostOutput(nullptr) {

    cudaSetDevice(0);

    engineData = readEngineFile(enginePath);

    runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT Runtime");
    }

    engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        throw std::runtime_error("Failed to deserialize engine");
    }

    context = engine->createExecutionContext();
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }

    for (int i = 0; i < engine->getNbBindings(); i++) {
        if (engine->bindingIsInput(i)) {
            inputBindingIndex = i;
        } else {
            outputBindingIndex = i;
        }
    }

    if (inputBindingIndex == -1 || outputBindingIndex == -1) {
        throw std::runtime_error("Could not find input and output bindings");
    }

    inputDims = engine->getBindingDimensions(inputBindingIndex);
    outputDims = engine->getBindingDimensions(outputBindingIndex);

    if (inputDims.d[0] == -1) {
        nvinfer1::Dims4 explicitDims(1, inputSize.height, inputSize.width, 1);
        context->setBindingDimensions(inputBindingIndex, explicitDims);
        inputDims = context->getBindingDimensions(inputBindingIndex);
    }

    outputDims = context->getBindingDimensions(outputBindingIndex);

    for (int i = 0; i < outputDims.nbDims; i++) {
        if (outputDims.d[i] < 0) {
            throw std::runtime_error("Output shape is undefined or dynamic");
        }
    }

    // Pre-calculate sizes
    inputElementCount = 1;
    for (int i = 0; i < inputDims.nbDims; i++) {
        inputElementCount *= static_cast<size_t>(inputDims.d[i]);
    }
    inputByteSize = inputElementCount * sizeof(float);

    outputElementCount = 1;
    for (int i = 0; i < outputDims.nbDims; i++) {
        outputElementCount *= static_cast<size_t>(outputDims.d[i]);
    }
    outputByteSize = outputElementCount * sizeof(float);

    // Pre-allocate CUDA resources
    cudaError_t status;

    // Create stream
    status = cudaStreamCreate(&stream);
    if (status != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(status)));
    }

    // Allocate device memory
    status = cudaMalloc(&deviceInput, inputByteSize);
    if (status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate input memory on GPU: " + std::string(cudaGetErrorString(status)));
    }

    status = cudaMalloc(&deviceOutput, outputByteSize);
    if (status != cudaSuccess) {
        cudaFree(deviceInput);
        throw std::runtime_error("Failed to allocate output memory on GPU: " + std::string(cudaGetErrorString(status)));
    }

    // Pre-allocate and configure bindings
    bindings.resize(engine->getNbBindings());
    bindings[inputBindingIndex] = deviceInput;
    bindings[outputBindingIndex] = deviceOutput;

    // Use pinned memory for faster transfers
    status = cudaHostAlloc((void**)&hostInput, inputByteSize, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        cleanupResources();
        throw std::runtime_error("Failed to allocate pinned host memory for input: " + std::string(cudaGetErrorString(status)));
    }

    status = cudaHostAlloc((void**)&hostOutput, outputByteSize, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        cudaFreeHost(hostInput);
        cleanupResources();
        throw std::runtime_error("Failed to allocate pinned host memory for output: " + std::string(cudaGetErrorString(status)));
    }
}

void TensorRTInferencer::cleanupResources() {
    if (deviceInput) cudaFree(deviceInput);
    if (deviceOutput) cudaFree(deviceOutput);
    if (stream) cudaStreamDestroy(stream);
    deviceInput = nullptr;
    deviceOutput = nullptr;
    stream = nullptr;
}

TensorRTInferencer::~TensorRTInferencer() {
    if (hostInput) cudaFreeHost(hostInput);
    if (hostOutput) cudaFreeHost(hostOutput);
    cleanupResources();

    if (context) {
        context->destroy();
    }
    if (engine) {
        engine->destroy();
    }
    if (runtime) {
        runtime->destroy();
    }
}

std::vector<char> TensorRTInferencer::readEngineFile(const std::string& enginePath) {
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        throw std::runtime_error("Engine file not found: " + enginePath);
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Failed to read engine file");
    }

    return buffer;
}

cv::cuda::GpuMat TensorRTInferencer::preprocessImage(const cv::cuda::GpuMat& gpuImage) {
    if (gpuImage.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    // Convert to grayscale if needed (remains on GPU)
    cv::cuda::GpuMat gpuGray;
    if (gpuImage.channels() > 1) {
        cv::cuda::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY);
    } else {
        gpuGray = gpuImage;
    }

    // Resize on GPU
    cv::cuda::GpuMat gpuResized;
    cv::cuda::resize(gpuGray, gpuResized, inputSize, 0, 0, cv::INTER_LINEAR);

    // Convert to float and normalize to [0, 1]
    cv::cuda::GpuMat gpuFloat;
    gpuResized.convertTo(gpuFloat, CV_32F, 1.0 / 255.0);

    return gpuFloat;  // Keeps everything on GPU!
}

/* std::vector<float> TensorRTInferencer::runInference(const cv::Mat& inputImage) {
    // Check for correct input dimensions
    if (inputImage.rows != inputSize.height || inputImage.cols != inputSize.width) {
        throw std::runtime_error("Input image dimensions do not match expected dimensions");
    }

    // Profile the bottleneck
    auto start = std::chrono::high_resolution_clock::now();

    // Copy input data to pinned memory buffer - this can be a CPU bottleneck
    if (inputImage.isContinuous()) {
        // Fast path for continuous data
        std::memcpy(hostInput, inputImage.ptr<float>(0), inputByteSize);
    } else {
        // Handle non-continuous data
        for (int h = 0; h < inputSize.height; h++) {
            const float* rowPtr = inputImage.ptr<float>(h);
            std::memcpy(hostInput + h * inputSize.width, rowPtr, inputSize.width * sizeof(float));
        }
    }

    auto memcpyEnd = std::chrono::high_resolution_clock::now();
    float memcpyMs = std::chrono::duration<float, std::milli>(memcpyEnd - start).count();
    // Uncomment for debugging: std::cout << "CPU memcpy took: " << memcpyMs << "ms\n";

    try {
        // Copy data from host to device asynchronously
        cudaMemcpyAsync(deviceInput, hostInput, inputByteSize, cudaMemcpyHostToDevice, stream);

        // Execute inference asynchronously
        bool status = context->enqueueV2(bindings.data(), stream, nullptr);
        if (!status) {
            throw std::runtime_error("TensorRT inference execution failed");
        }

        // Copy results back from device to host asynchronously
        cudaMemcpyAsync(hostOutput, deviceOutput, outputByteSize, cudaMemcpyDeviceToHost, stream);

        // Synchronize to ensure operations are complete
        cudaStreamSynchronize(stream);

        auto inferEnd = std::chrono::high_resolution_clock::now();
        float inferMs = std::chrono::duration<float, std::milli>(inferEnd - memcpyEnd).count();
        // Uncomment for debugging: std::cout << "GPU inference took: " << inferMs << "ms\n";

        // Create return vector - avoid extra copy when possible
        std::vector<float> outputBuffer(outputElementCount);
        std::memcpy(outputBuffer.data(), hostOutput, outputByteSize);

        return outputBuffer;
    } catch (const std::exception& e) {
        cudaStreamSynchronize(stream);
        std::cerr << "Error during inference: " << e.what() << std::endl;
        throw;
    }
} */

void TensorRTInferencer::runInference(const cv::cuda::GpuMat& gpuInput) {
    // Check for correct input dimensions
    if (gpuInput.rows != inputSize.height || gpuInput.cols != inputSize.width) {
        throw std::runtime_error("Input image dimensions do not match expected dimensions");
    }

    // Device-to-device copy from GpuMat to TensorRT input buffer
    cudaError_t err = cudaMemcpy2DAsync(
        deviceInput,                          // TensorRT input buffer (GPU memory)
        inputSize.width * sizeof(float),      // destination pitch (row stride)
        gpuInput.ptr<float>(),                // source pointer (GpuMat data)
        gpuInput.step,                        // source pitch (GpuMat stride)
        inputSize.width * sizeof(float),      // width in bytes
        inputSize.height,                     // height in rows
        cudaMemcpyDeviceToDevice,             // GPU to GPU transfer
        stream                                // CUDA stream
    );

    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy2DAsync failed: " + std::string(cudaGetErrorString(err)));
    }

    // Execute inference on the GPU asynchronously
    if (!context->enqueueV2(bindings.data(), stream, nullptr)) {
        throw std::runtime_error("TensorRT inference execution failed");
    }

    // No CPU transfer! Output remains on deviceOutput buffer (GPU memory).
}

cv::cuda::GpuMat TensorRTInferencer::makePrediction(const cv::cuda::GpuMat& gpuImage) {
    cv::cuda::GpuMat gpuInputFloat = preprocessImage(gpuImage);  // Preprocess entirely on GPU

    runInference(gpuInputFloat);  // Run inference directly on GPU input

    // Create a GpuMat output mask (assume float output)
    int height = outputDims.d[1];
    int width  = outputDims.d[2];
    cv::cuda::GpuMat outputMaskGpu(height, width, CV_32F);

    // Copy inference output directly from deviceOutput buffer into GpuMat
    cudaMemcpy2DAsync(
        outputMaskGpu.ptr<float>(),                 // destination pointer (GPU memory)
        outputMaskGpu.step,                         // destination pitch
        deviceOutput,                               // source pointer (raw TensorRT output buffer)
        width * sizeof(float),                      // source pitch
        width * sizeof(float),                      // width in bytes
        height,                                     // height in rows
        cudaMemcpyDeviceToDevice,                   // device-to-device copy
        stream                                      // CUDA stream
    );

    cudaStreamSynchronize(stream);  // Ensure copy is complete before returning

    return outputMaskGpu;
}
