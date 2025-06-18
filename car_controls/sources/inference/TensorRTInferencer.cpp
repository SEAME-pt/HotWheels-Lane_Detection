#include "../../includes/inference/TensorRTInferencer.hpp"  // Include TensorRTInferencer header
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <numeric>

// Logger callback for TensorRT to print warnings and errors
void TensorRTInferencer::Logger::log(Severity severity, const char* msg) noexcept {
	if (severity <= Severity::kWARNING)  // Only log warnings or more severe messages
		std::cout << msg << std::endl;   // Print the message to the console
}

// Constructor: loads TensorRT engine, allocates memory and sets up execution context
TensorRTInferencer::TensorRTInferencer(const std::string& enginePath) :
	runtime(nullptr),        // Initialize runtime pointer to nullptr
	engine(nullptr),         // Initialize engine pointer to nullptr
	context(nullptr),        // Initialize execution context pointer to nullptr
	inputBindingIndex(-1),   // Initialize input binding index
	outputBindingIndex(-1),  // Initialize output binding index
	inputSize(208, 208),     // Set default input image size
	deviceInput(nullptr),    // Initialize device input pointer to nullptr
	deviceOutput(nullptr),   // Initialize device output pointer to nullptr
	stream(nullptr),         // Initialize CUDA stream pointer to nullptr
	hostInput(nullptr),      // Initialize host input pointer to nullptr
	hostOutput(nullptr) {    // Initialize host output pointer to nullptr

	cudaSetDevice(0);  // Set CUDA device to GPU 0

	engineData = readEngineFile(enginePath);  // Load serialized engine file into memory

	runtime = nvinfer1::createInferRuntime(logger);  // Create TensorRT runtime with logger
	if (!runtime) {  // Check if runtime creation failed
		throw std::runtime_error("Failed to create TensorRT Runtime");
	}

	engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());  // Deserialize the engine from the loaded data
	if (!engine) {  // Check if deserialization failed
		throw std::runtime_error("Failed to deserialize engine");
	}

	context = engine->createExecutionContext();  // Create execution context from the engine
	if (!context) {  // Check if context creation failed
		throw std::runtime_error("Failed to create execution context");
	}

	lanePostProcessor = new LanePostProcessor(350, 260, 10.0f, 10.0f);  // Initialize lane post-processor with parameters
	laneCurveFitter = new LaneCurveFitter(5.0f, 20, 20, 300); // Initialize lane curve fitter with parameters

	for (int i = 0; i < engine->getNbBindings(); i++) {  // Loop through all bindings
		if (engine->bindingIsInput(i)) {  // If binding is input
			inputBindingIndex = i;  // Save input binding index
		} else {
			outputBindingIndex = i;  // Otherwise, save output binding index
		}
	}

	if (inputBindingIndex == -1 || outputBindingIndex == -1) {  // Verify both input and output were found
		throw std::runtime_error("Could not find input and output bindings");
	}

	inputDims = engine->getBindingDimensions(inputBindingIndex);  // Get input tensor dimensions
	outputDims = engine->getBindingDimensions(outputBindingIndex);  // Get output tensor dimensions

	if (inputDims.d[0] == -1) {  // If input has dynamic batch dimension
		nvinfer1::Dims4 explicitDims(1, inputSize.height, inputSize.width, 1);  // Define explicit batch size and dimensions
		context->setBindingDimensions(inputBindingIndex, explicitDims);  // Set explicit input dimensions
		inputDims = context->getBindingDimensions(inputBindingIndex);  // Update inputDims after setting
	}

	outputDims = context->getBindingDimensions(outputBindingIndex);  // Confirm and update outputDims

	for (int i = 0; i < outputDims.nbDims; i++) {  // Check if any output dimension is dynamic
		if (outputDims.d[i] < 0) {
			throw std::runtime_error("Output shape is undefined or dynamic");  // Throw error if output is not fully defined
		}
	}

	inputElementCount = 1;  // Initialize input element count
	for (int i = 0; i < inputDims.nbDims; i++) {  // Multiply all input dimensions
		inputElementCount *= static_cast<size_t>(inputDims.d[i]);
	}
	inputByteSize = inputElementCount * sizeof(float);  // Calculate input buffer size in bytes

	outputElementCount = 1;  // Initialize output element count
	for (int i = 0; i < outputDims.nbDims; i++) {  // Multiply all output dimensions
		outputElementCount *= static_cast<size_t>(outputDims.d[i]);
	}
	outputByteSize = outputElementCount * sizeof(float);  // Calculate output buffer size in bytes

	cudaError_t status;  // Define variable for checking CUDA errors

	status = cudaStreamCreate(&stream);  // Create a CUDA stream for async operations
	if (status != cudaSuccess) {  // Check stream creation
		throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(status)));
	}

	status = cudaMalloc(&deviceInput, inputByteSize);  // Allocate device memory for input tensor
	if (status != cudaSuccess) {  // Check input memory allocation
		throw std::runtime_error("Failed to allocate input memory on GPU: " + std::string(cudaGetErrorString(status)));
	}

	status = cudaMalloc(&deviceOutput, outputByteSize);  // Allocate device memory for output tensor
	if (status != cudaSuccess) {  // Check output memory allocation
		cudaFree(deviceInput);  // Free previously allocated input memory if failed
		throw std::runtime_error("Failed to allocate output memory on GPU: " + std::string(cudaGetErrorString(status)));
	}

	bindings.resize(engine->getNbBindings());  // Resize bindings array to number of bindings
	bindings[inputBindingIndex] = deviceInput;  // Assign device input buffer
	bindings[outputBindingIndex] = deviceOutput;  // Assign device output buffer

	Publisher::instance(5556); // Initialize publisher for inference results

	initUndistortMaps();  // Initialize undistortion maps for camera calibration
	cudaStream = cv::cuda::Stream();  // CUDA stream for asynchronous operations
}

// Clean up allocated GPU resources (device memory, streams)
void TensorRTInferencer::cleanupResources() {
	if (deviceInput) cudaFree(deviceInput);   // Free input buffer if allocated
	if (deviceOutput) cudaFree(deviceOutput); // Free output buffer if allocated
	if (stream) cudaStreamDestroy(stream);    // Destroy CUDA stream if created
	if (lanePostProcessor) delete lanePostProcessor; // Delete post-processor object
	if (laneCurveFitter) delete laneCurveFitter; // Delete post-processor and curve fitter objects
	deviceInput = nullptr;    // Set pointers to nullptr after freeing
	deviceOutput = nullptr;
	stream = nullptr;
}

// Destructor: free all allocated resources
TensorRTInferencer::~TensorRTInferencer() {
	delete m_publisherObject;
	m_publisherObject = nullptr;

	if (hostInput) cudaFreeHost(hostInput);   // Free pinned host memory for input
	if (hostOutput) cudaFreeHost(hostOutput); // Free pinned host memory for output
	cleanupResources();  // Free GPU resources

	if (context) {
		context->destroy();   // Destroy TensorRT execution context
	}
	if (engine) {
		engine->destroy();    // Destroy TensorRT engine
	}
	if (runtime) {
		runtime->destroy();   // Destroy TensorRT runtime
	}
}

// Read the serialized TensorRT engine file into memory
std::vector<char> TensorRTInferencer::readEngineFile(const std::string& enginePath) {
	std::ifstream file(enginePath, std::ios::binary | std::ios::ate);  // Open file in binary mode, go to end
	if (!file.good()) {   // Check if file opened successfully
		throw std::runtime_error("Engine file not found: " + enginePath);
	}

	size_t size = file.tellg();  // Get file size
	file.seekg(0, std::ios::beg);  // Go back to beginning of file

	std::vector<char> buffer(size);  // Create buffer of the correct size
	if (!file.read(buffer.data(), size)) {  // Read file into buffer
		throw std::runtime_error("Failed to read engine file");
	}

	return buffer;  // Return loaded engine buffer
}

// Preprocess input image on GPU: convert to grayscale, resize, normalize
cv::cuda::GpuMat TensorRTInferencer::preprocessImage(const cv::cuda::GpuMat& gpuImage) {
	if (gpuImage.empty()) {  // Validate input image
		throw std::runtime_error("Input image is empty");
	}

	if (gpuImage.type() != CV_8UC3 && gpuImage.type() != CV_8UC1) {  // Check if input image is in expected format
		throw std::runtime_error("Input image must be CV_8UC3 (color) or CV_8UC1 (grayscale)");
	}

	cv::cuda::GpuMat gpuGray;
	if (gpuImage.channels() > 1) {   // If input has multiple channels (color)
		cv::cuda::cvtColor(gpuImage, gpuGray, cv::COLOR_BGR2GRAY); // Convert to grayscale
	} else {
		gpuGray = gpuImage;  // Already grayscale, no conversion needed
	}

	cv::cuda::GpuMat gpuResized;
	cv::cuda::resize(gpuGray, gpuResized, inputSize, 0, 0, cv::INTER_LINEAR); // Resize to network input size

	cv::cuda::GpuMat gpuFloat;
	gpuResized.convertTo(gpuFloat, CV_32F, 1.0 / 255.0); // Normalize to [0,1] and convert to float32

	return gpuFloat;  // Return preprocessed image (still on GPU)
}

// Run inference given a GpuMat input (already preprocessed)
void TensorRTInferencer::runInference(const cv::cuda::GpuMat& gpuInput) {
	if (gpuInput.rows != inputSize.height || gpuInput.cols != inputSize.width) {  // Verify input dimensions
		throw std::runtime_error("Input image dimensions do not match expected dimensions");
	}

	cudaError_t err = cudaMemcpy2DAsync(
		deviceInput,                          // Destination: TensorRT input buffer
		inputSize.width * sizeof(float),      // Destination row stride
		gpuInput.ptr<float>(),                // Source pointer: GpuMat data
		gpuInput.step,                        // Source stride
		inputSize.width * sizeof(float),      // Width to copy in bytes
		inputSize.height,                     // Height to copy (rows)
		cudaMemcpyDeviceToDevice,             // Type of copy: GPU to GPU
		stream                                // Use CUDA stream
	);

	if (err != cudaSuccess) {  // Check if memory copy failed
		throw std::runtime_error("cudaMemcpy2DAsync failed: " + std::string(cudaGetErrorString(err)));
	}

	if (!context->enqueueV2(bindings.data(), stream, nullptr)) {  // Enqueue inference on the GPU
		throw std::runtime_error("TensorRT inference execution failed");
	}
}

// Perform full prediction pipeline: preprocess, inference, and extract output
cv::cuda::GpuMat TensorRTInferencer::makePrediction(const cv::cuda::GpuMat& gpuImage) {
	cv::cuda::GpuMat gpuInputFloat = preprocessImage(gpuImage);  // Preprocess input image on GPU

	runInference(gpuInputFloat);  // Run inference

	int height = outputDims.d[1];
	int width  = outputDims.d[2];

	// Allocate or resize the output mask on GPU if it's not allocated or has wrong size
	if (outputMaskGpu.empty() || outputMaskGpu.rows != height || outputMaskGpu.cols != width) {
		outputMaskGpu = cv::cuda::GpuMat(height, width, CV_32F);
	}

	// Copy the raw prediction output from TensorRT device memory to `outputMaskGpu`
	// - Assumes output is already in device memory (deviceOutput)
	// - No CPU-GPU transfer, all device-to-device
	cudaMemcpy2DAsync(
		outputMaskGpu.ptr<float>(), outputMaskGpu.step,
		deviceOutput, width * sizeof(float),
		width * sizeof(float), height,
		cudaMemcpyDeviceToDevice, stream
	);

	//post-process starts here
/* 	cv::cuda::GpuMat postProcessedMaskGpu = lanePostProcessor->process(outputMaskGpu);

	// Download post-processed binary mask for polyfitting
	cv::Mat maskCpu;
	postProcessedMaskGpu.download(maskCpu);

	// Fit lanes and compute centerline
	std::vector<LaneCurveFitter::LaneCurve> lanes = laneCurveFitter->fitLanes(maskCpu);
	std::cout << "[DEBUG] Number of fitted lanes: " << lanes.size() << std::endl;
	auto centerlineOpt = laneCurveFitter->computeVirtualCenterline(lanes, maskCpu.cols, maskCpu.rows);
	if (!centerlineOpt.has_value()) {
		std::cout << "[DEBUG] No centerline could be computed." << std::endl;
	}

	// Draw centerline on CPU
	if (centerlineOpt.has_value()) {
		const auto& centerline = centerlineOpt.value().blended;
		for (size_t i = 1; i < centerline.size(); ++i) {
			cv::line(maskCpu,
					centerline[i - 1],
					centerline[i],
					cv::Scalar(255), // White
					2,               // Thickness
					cv::LINE_AA);
		}
	}

	// Upload mask with centerline back to GPU
	postProcessedMaskGpu.upload(maskCpu); */

	return outputMaskGpu;
}

void TensorRTInferencer::initUndistortMaps() {
	cv::Mat cameraMatrix, distCoeffs;
	cv::FileStorage fs("/home/jetson/models/lane-detection/camera_calibration.yml", cv::FileStorage::READ);  // Open calibration file

	if (!fs.isOpened()) {
		std::cerr << "[Error] Failed to open camera_calibration.yml" << std::endl;
		return;  // Handle file opening error
	}

	fs["camera_matrix"] >> cameraMatrix;  // Read camera matrix
	fs["distortion_coefficients"] >> distCoeffs;  // Read distortion coefficients
	fs.release();  // Close file

	cv::Mat mapx, mapy;
	cv::initUndistortRectifyMap(
		cameraMatrix, distCoeffs, cv::Mat(), cameraMatrix,
		cv::Size(1280, 720),
		CV_32FC1, mapx, mapy
	);  // Compute undistortion mapping

	d_mapx.upload(mapx);  // Upload X map to GPU
	d_mapy.upload(mapy);  // Upload Y map to GPU
}

void TensorRTInferencer::doInference(const cv::Mat& frame) {
	if (frame.empty()) {
		throw std::runtime_error("Input frame is empty");
	}

	cv::cuda::GpuMat d_frame(frame);  // Upload frame to GPU
	cv::cuda::GpuMat d_undistorted;
	cv::cuda::remap(d_frame, d_undistorted, d_mapx, d_mapy, cv::INTER_LINEAR, 0, cv::Scalar(), cudaStream);  // Undistort frame

	cv::cuda::GpuMat d_prediction_mask = makePrediction(d_undistorted);  // Run model inference

	// Convert to 8-bit (0 or 255) in a new GpuMat
	cv::cuda::GpuMat d_mask_u8;
	d_prediction_mask.convertTo(d_mask_u8, CV_8U, 255.0);  // Multiply 0/1 float to 0/255

	cv::Mat binary_mask_cpu;
	d_mask_u8.download(binary_mask_cpu, cudaStream);
	cv::threshold(binary_mask_cpu, binary_mask_cpu, 128, 255, cv::THRESH_BINARY);
	cudaStream.waitForCompletion();  // Ensure async operations are complete

	// Convert model output to 8-bit binary mask on GPU
	cv::cuda::GpuMat d_visualization;
	d_prediction_mask.convertTo(d_visualization, CV_8U, 255.0, 0, cudaStream);

	cv::cuda::GpuMat d_resized_mask;

	cv::cuda::resize(d_visualization, d_resized_mask,
						cv::Size(frame.cols * 0.5, frame.rows * 0.5),
						0, 0, cv::INTER_LINEAR, cudaStream);  // Resize for display
	cudaStream.waitForCompletion();  // Synchronize

	Publisher::instance(5556)->publishInferenceFrame("inference_frame", d_resized_mask); //Publish frame to ZeroMQ publisher
}
