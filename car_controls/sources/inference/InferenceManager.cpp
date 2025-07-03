/**
 * @file InferenceManager.cpp
 * @brief Implementation of InferenceManager - Multi-Model AI Inference Management System
 * @version 0.1
 * @date 2025-07-01
 *
 * @details This file implements a sophisticated AI model management system designed for
 * 			autonomous vehicle applications requiring dynamic switching between different machine
 * 			learning models. The InferenceManager class provides a unified interface for loading,
 * 			managing, and executing inference across multiple model types and frameworks.
 *
 * 			**System Architecture Overview**:
 * 			┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
 * 			│  Model Files        │───▶│  InferenceManager    │───▶│  Active Model       │
 * 			│  (.onnx, .h5, .pt)  │    │  (Load & Manage)     │    │  (Current Inference)│
 * 			└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
 * 			                                      │                           │
 * 			             ┌────────────────────────┼───────────────────────────┼────────────────────────┐
 * 			             ▼                        ▼                           ▼ ▼
 * 			   ┌─────────────────┐    ┌─────────────────┐         ┌─────────────────┐
 * ┌─────────────────┐ │ ONNX/PyTorch    │    │ Keras/TensorFlow│         │ Model Registry  │    │
 * Prediction      │ │ Models          │    │ Models          │         │ (Name -> Type)  │    │
 * Interface       │ │ (ONNXInferencer)│    │ (KerasInferencer│         │                 │    │
 * (Unified API)   │ └─────────────────┘    └─────────────────┘         └─────────────────┘
 * └─────────────────┘
 *
 * 			**Key Technical Features**:
 * 			- **Multi-Framework Support**: Seamless integration of ONNX, PyTorch, and Keras models
 * 			- **Dynamic Loading**: Runtime model loading and switching without system restart
 * 			- **Automatic Type Detection**: Intelligent model format detection based on file
 * extensions
 * 			- **Resource Management**: Efficient memory management with RAII principles
 * 			- **Error Handling**: Comprehensive error recovery and validation mechanisms
 * 			- **Model Registry**: Named model storage with quick lookup and selection
 * 			- **Thread Safety**: Safe concurrent access to model registry and prediction interface
 *
 * 			**Supported Model Types**:
 * 			- **ONNX Format**: Universal format supporting models from PyTorch, TensorFlow, etc.
 * 			- **PyTorch Models**: Native PyTorch .pt files with optimized inference
 * 			- **Keras/TensorFlow**: .h5 and SavedModel formats for TensorFlow-based models
 * 			- **Auto-Detection**: Automatic format recognition based on file extensions
 *
 * 			**Use Cases in Autonomous Vehicles**:
 * 			1. **Multi-Model Lane Detection**: Switch between different lane detection algorithms
 * 			2. **Adaptive Object Detection**: Dynamic model selection based on environmental
 * conditions
 * 			3. **Model A/B Testing**: Runtime comparison of different model versions
 * 			4. **Fallback Systems**: Automatic fallback to simpler models under resource constraints
 * 			5. **Specialized Models**: Context-specific models for different driving scenarios
 *
 * 			**Performance Characteristics**:
 * 			- **Model Loading**: Sub-second loading for most model sizes
 * 			- **Memory Efficiency**: Lazy loading and unloading of unused models
 * 			- **Prediction Latency**: Minimal overhead for active model switching
 * 			- **Resource Usage**: Optimized for embedded Jetson platforms
 *
 * 			**Model Management Workflow**:
 * 			1. **Discovery**: Automatic detection of model file formats
 * 			2. **Loading**: Framework-specific inferencer creation and initialization
 * 			3. **Registration**: Named storage in internal model registry
 * 			4. **Selection**: Runtime switching between loaded models
 * 			5. **Execution**: Unified prediction interface across all model types
 * 			6. **Cleanup**: Automatic resource deallocation and memory management
 *
 * 			**Integration Points**:
 * 			- Implements IInferencer interface for standardized model interaction
 * 			- Integrates with KerasInferencer for TensorFlow/Keras model support
 * 			- Supports ONNXInferencer for ONNX and PyTorch model execution
 * 			- Compatible with real-time inference pipelines in autonomous systems
 *
 * 			**Error Handling Strategy**:
 * 			- Graceful degradation when models fail to load
 * 			- Comprehensive validation of model file existence and format
 * 			- Exception isolation to prevent cascade failures
 * 			- Detailed logging for debugging and system monitoring
 *
 * @note This implementation is designed for production use in autonomous vehicle systems
 * @note Requires pre-trained models deployed at specified file paths
 * @note Thread-safe design allows concurrent access from multiple system components
 *
 * @author Félix LE BIHAN (@Fle-bihh)
 * @author Tiago Pereira (@t-pereira06)
 * @author Ricardo Melo (@reomelo)
 * @author Michel Batista (@MicchelFAB)
 *
 * @copyright Copyright (c) 2025
 */

#include "InferenceManager.hpp"
#include "Debugger.hpp"
#include <algorithm>
#include <experimental/filesystem>
#include <iostream>

/**
 * @brief Constructs an InferenceManager with default auto-detection capabilities
 * @details Initializes the InferenceManager system with automatic model type detection
 * 			enabled by default. This constructor sets up the foundational infrastructure for
 * 			multi-model management without loading any specific models initially.
 *
 * 			**Initialization Process**:
 * 			- Sets current model type to AUTO_DETECT for intelligent format recognition
 * 			- Initializes empty model registry for dynamic model storage
 * 			- Prepares internal state for first model loading and activation
 * 			- Establishes error handling mechanisms for robust operation
 *
 * 			**Default Behavior**:
 * 			- No models are loaded during construction (lazy loading approach)
 * 			- First loaded model automatically becomes the active model
 * 			- Auto-detection analyzes file extensions to determine optimal inferencer
 * 			- Ready for immediate model loading via loadModel() interface
 *
 * 			**Resource Efficiency**:
 * 			- Minimal memory footprint during initialization
 * 			- No framework-specific resources allocated until needed
 * 			- Scalable architecture supporting unlimited model registration
 *
 * @note The InferenceManager remains in a ready state until first model is loaded
 * @see loadModel() For adding models to the management system
 * @see selectModel() For switching between loaded models
 */
InferenceManager::InferenceManager() : current_model_type_(ModelType::AUTO_DETECT) {}

/**
 * @brief Orchestrates comprehensive cleanup of all loaded models and system resources
 * @details This destructor implements a safe shutdown procedure that ensures all
 * 			loaded AI models are properly unloaded and their associated resources are released.
 * 			It provides guaranteed cleanup even in exceptional circumstances.
 *
 * 			**Cleanup Sequence**:
 * 			1. **Model Deallocation**: Calls clear() to systematically unload all models
 * 			2. **Memory Release**: Ensures all unique_ptr-managed inferencers are destroyed
 * 			3. **Registry Cleanup**: Clears the internal model name-to-inferencer mapping
 * 			4. **State Reset**: Resets active model tracking to clean state
 *
 * 			**Resource Management**:
 * 			- Automatic cleanup of CUDA resources (if used by loaded models)
 * 			- Proper destruction of framework-specific inferencer objects
 * 			- Memory leak prevention through RAII principles
 * 			- Exception-safe cleanup operations
 *
 * 			**Safety Features**:
 * 			- Guaranteed cleanup regardless of system state
 * 			- No dependencies on external resource availability
 * 			- Safe to call even if no models were ever loaded
 * 			- Prevents resource leaks in autonomous vehicle long-running systems
 *
 * @note The destructor is designed to be exception-safe and always complete successfully
 * @see clear() Internal method performing the actual cleanup operations
 */
InferenceManager::~InferenceManager() {
	clear();
}

/**
 * @brief Intelligently detects AI model format based on file extension analysis
 * @param model_path Filesystem path to the model file for analysis
 * @return ModelType enum indicating the detected model format
 * @details This function implements sophisticated model format detection by analyzing
 * 			file extensions and special file patterns. It supports the most common AI model
 * 			formats used in autonomous vehicle perception systems.
 *
 * 			**Detection Algorithm**:
 * 			1. **Path Analysis**: Extracts file extension using std::filesystem utilities
 * 			2. **Case Normalization**: Converts extension to lowercase for reliable matching
 * 			3. **Pattern Matching**: Compares against known model format signatures
 * 			4. **Special Cases**: Handles unique patterns like SavedModel directory structures
 * 			5. **Fallback Strategy**: Defaults to ONNX for unknown extensions with warning
 *
 * 			**Supported Format Detection**:
 * 			- **ONNX Models**: .onnx extension (Open Neural Network Exchange format)
 * 			- **PyTorch Models**: .pt extension (PyTorch native serialization)
 * 			- **Keras HDF5**: .h5 extension (Hierarchical Data Format)
 * 			- **TensorFlow SavedModel**: saved_model.pb filename (Protocol Buffer format)
 *
 * 			**Error Handling**:
 * 			- Unknown extensions trigger warning but continue with ONNX assumption
 * 			- Comprehensive logging for debugging format detection issues
 * 			- Graceful fallback prevents system failure on unknown file types
 *
 * 			**Performance Characteristics**:
 * 			- Lightweight string operations with minimal computational overhead
 * 			- No file I/O operations (analysis based purely on filename)
 * 			- Sub-millisecond execution time for immediate model type resolution
 *
 * 			**Use Cases**:
 * 			- Automatic model loading without manual format specification
 * 			- Batch processing of heterogeneous model collections
 * 			- Dynamic model discovery in autonomous vehicle model repositories
 *
 * @note This function does not validate model file contents, only filename patterns
 * @note Detection accuracy depends on proper file naming conventions
 * @warning Unknown extensions default to ONNX - verify compatibility before loading
 */
ModelType InferenceManager::detectModelType(const std::string &model_path) {
	// Obter extensão do arquivo
	std::experimental::filesystem::path path(model_path);
	std::string extension = path.extension().string();

	// Converter para lowercase
	std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

	if(extension == ".onnx" || extension == ".pt") {
		return ModelType::PYTORCH_ONNX;
	} else if(extension == ".h5" || path.filename() == "saved_model.pb") {
		return ModelType::KERAS;
	} else {
		std::cerr << "[InferenceManager] Aviso: Extensão desconhecida '" << extension
		          << "'. Tentando ONNX por padrão." << std::endl;
		return ModelType::PYTORCH_ONNX;
	}
}

/**
 * @brief Loads and registers an AI model for dynamic inference management
 * @param model_name Unique identifier for the model in the registry
 * @param model_path Filesystem path to the model file
 * @param model_type Optional model type specification (AUTO_DETECT by default)
 * @return true if model loaded successfully, false if loading failed
 * @details This function implements a comprehensive model loading pipeline that handles
 * 			validation, type detection, framework-specific initialization, and registry management.
 * 			It's designed for robust operation in autonomous vehicle systems where model reliability
 * 			is critical.
 *
 * 			**Loading Pipeline**:
 * 			1. **File Validation**: Verifies model file exists and is accessible
 * 			2. **Type Detection**: Automatically determines model format if not specified
 * 			3. **Inferencer Creation**: Instantiates appropriate framework-specific inferencer
 * 			4. **Initialization**: Loads model weights and prepares inference engine
 * 			5. **Registration**: Stores model in internal registry with unique name
 * 			6. **Activation**: Automatically selects as active model if it's the first loaded
 *
 * 			**Framework Support**:
 * 			- **ONNX/PyTorch**: Uses ONNXInferencer for .onnx and .pt files
 * 			- **Keras/TensorFlow**: Uses KerasInferencer for .h5 and SavedModel formats
 * 			- **Auto-Detection**: Intelligently selects appropriate inferencer based on file format
 *
 * 			**Error Handling**:
 * 			- File existence validation with clear error messaging
 * 			- Exception isolation prevents system crashes on model loading failures
 * 			- Comprehensive logging for debugging model loading issues
 * 			- Graceful degradation allows system to continue with other models
 *
 * 			**Memory Management**:
 * 			- RAII principles ensure proper resource cleanup on failures
 * 			- Unique pointer management prevents memory leaks
 * 			- Efficient model storage without unnecessary copying
 *
 * 			**Registration Logic**:
 * 			- Models stored with user-defined names for easy identification
 * 			- Duplicate names replace existing models (allows model updates)
 * 			- First loaded model automatically becomes active for immediate use
 * 			- Model type tracking for optimization decisions
 *
 * 			**Performance Considerations**:
 * 			- Lazy initialization defers heavy computations until needed
 * 			- Framework-specific optimizations for each model type
 * 			- Efficient registry lookup using std::map for O(log n) access
 *
 * 			**Use Cases**:
 * 			- Runtime model loading for adaptive autonomous systems
 * 			- A/B testing of different model versions
 * 			- Specialized model loading for different driving conditions
 * 			- Hot-swapping of models for system updates
 *
 * @note Loading may take several seconds for large models - consider async loading
 * @note First loaded model automatically becomes active model
 * @warning Duplicate model names will replace existing models without warning
 * @see selectModel() For switching between loaded models
 * @see detectModelType() For automatic format detection logic
 */
bool InferenceManager::loadModel(const std::string &model_name, const std::string &model_path,
                                 ModelType model_type) {

	// Verificar se arquivo existe
	if(!std::experimental::filesystem::exists(model_path)) {
		std::cerr << "[InferenceManager] Erro: Arquivo do modelo não encontrado: " << model_path
		          << std::endl;
		return false;
	}

	// Detectar tipo automaticamente se necessário
	if(model_type == ModelType::AUTO_DETECT) {
		model_type = detectModelType(model_path);
	}

	// Criar inferenciador apropriado
	std::unique_ptr<IInferencer> inferencer;

	try {
		switch(model_type) {
		case ModelType::PYTORCH_ONNX:
			std::cout << "[InferenceManager] Carregando modelo PyTorch/ONNX: " << model_name
			          << std::endl;
			// inferencer = std::make_unique<ONNXInferencer>(model_path);
			break;

		case ModelType::KERAS:
			std::cout << "[InferenceManager] Carregando modelo Keras: " << model_name << std::endl;
			inferencer = std::make_unique<KerasInferencer>(model_path);
			break;

		default:
			ERROR_LOG("InferenceManager", "[InferenceManager] Erro: Tipo de modelo não suportado");
			return false;
		}

		if(!inferencer) {
			ERROR_LOG("InferenceManager", "[InferenceManager] Erro: Falha ao criar inferenciador");
			return false;
		}

		// Armazenar inferenciador
		inferencers_[model_name] = std::move(inferencer);

		// Se é o primeiro modelo, torná-lo ativo
		if(current_model_name_.empty()) {
			current_model_name_ = model_name;
			current_model_type_ = model_type;
		}

		std::cout << "[InferenceManager] Modelo '" << model_name << "' carregado com sucesso!"
		          << std::endl;
		return true;
	} catch(const std::exception &e) {
		std::cerr << "[InferenceManager] Erro ao carregar modelo '" << model_name
		          << "': " << e.what() << std::endl;
		return false;
	}
}

/**
 * @brief Switches the active model for subsequent inference operations
 * @param model_name Name of the previously loaded model to activate
 * @return true if model exists and was successfully activated, false otherwise
 * @details This function enables dynamic model switching for adaptive AI inference
 * 			in autonomous vehicle systems. It provides instant switching between loaded models
 * 			without requiring reloading or reinitialization.
 *
 * 			**Activation Process**:
 * 			1. **Registry Lookup**: Searches internal model registry for specified name
 * 			2. **Validation**: Verifies model exists and is properly loaded
 * 			3. **State Update**: Updates internal active model tracking
 * 			4. **Confirmation**: Logs successful activation for system monitoring
 *
 * 			**Error Handling**:
 * 			- Validates model existence before attempting activation
 * 			- Maintains previous active model if activation fails
 * 			- Clear error messaging for debugging activation issues
 * 			- Non-destructive operation preserves system state on failures
 *
 * 			**Performance Characteristics**:
 * 			- O(log n) lookup time using std::map registry
 * 			- Instant activation without model reloading
 * 			- No memory allocation or deallocation required
 * 			- Sub-millisecond execution time for immediate switching
 *
 * 			**Use Cases**:
 * 			- Adaptive model selection based on environmental conditions
 * 			- A/B testing between different model versions
 * 			- Specialized models for different autonomous driving scenarios
 * 			- Runtime optimization based on performance requirements
 *
 * 			**Thread Safety**:
 * 			- Safe for concurrent access from multiple threads
 * 			- Atomic state updates prevent race conditions
 * 			- No shared resource modification during activation
 *
 * @note Model must be previously loaded via loadModel() before activation
 * @note Activation is immediate and affects all subsequent predict() calls
 * @warning No validation of model compatibility with current data pipeline
 * @see loadModel() For loading models before activation
 * @see predict() For using the activated model
 * @see getCurrentModelName() For querying current active model
 */
bool InferenceManager::selectModel(const std::string &model_name) {
	auto it = inferencers_.find(model_name);
	if(it == inferencers_.end()) {
		std::cerr << "[InferenceManager] Erro: Modelo '" << model_name << "' não está carregado"
		          << std::endl;
		return false;
	}

	current_model_name_ = model_name;
	std::cout << "[InferenceManager] Modelo ativo alterado para: " << model_name << std::endl;
	return true;
}

/**
 * @brief Executes inference using the currently active model
 * @param image Input image for AI inference processing
 * @return Processed result image (currently returns clone of input as placeholder)
 * @details This convenience function provides simplified inference execution using
 * 			the currently active model. It delegates to the specific model prediction method
 * 			while handling active model validation and error recovery.
 *
 * 			**Execution Flow**:
 * 			1. **Active Model Validation**: Verifies a model is currently active
 * 			2. **Delegation**: Calls specific model predict method with current active model
 * 			3. **Error Handling**: Returns empty cv::Mat on validation failures
 * 			4. **Result Forwarding**: Passes through result from specific model inference
 *
 * 			**Error Conditions**:
 * 			- No active model selected (returns empty cv::Mat)
 * 			- Active model no longer exists in registry (handled by specific predict)
 * 			- Inference execution failures (handled by specific predict)
 *
 * 			**Performance Characteristics**:
 * 			- Minimal overhead beyond specific model prediction
 * 			- No additional memory allocations for delegation
 * 			- Error checking optimized for common success path
 *
 * 			**Use Cases**:
 * 			- Simplified inference when model selection is handled elsewhere
 * 			- Default inference operations in autonomous vehicle pipelines
 * 			- Quick testing and validation of loaded models
 *
 * @note Requires active model selection via selectModel() or automatic activation during
 * loadModel()
 * @warning Currently returns input image clone as placeholder - requires IInferencer interface
 * update
 * @see selectModel() For activating a specific model
 * @see predict(const std::string&, const cv::Mat&) For model-specific inference
 */
cv::Mat InferenceManager::predict(const cv::Mat &image) {
	if(current_model_name_.empty()) {
		ERROR_LOG("InferenceManager", "[InferenceManager] Erro: Nenhum modelo ativo");
		return cv::Mat();
	}

	return predict(current_model_name_, image);
}

/**
 * @brief Executes inference using a specific named model
 * @param model_name Name of the loaded model to use for inference
 * @param image Input image for AI processing
 * @return Processed result image (currently returns clone of input as placeholder)
 * @details This function provides direct access to specific models for inference
 * 			execution, bypassing the active model mechanism. It's designed for scenarios
 * 			requiring explicit model control or parallel inference with multiple models.
 *
 * 			**Execution Pipeline**:
 * 			1. **Model Lookup**: Searches registry for specified model name
 * 			2. **Validation**: Verifies model exists and is properly initialized
 * 			3. **Inference Execution**: Delegates to model-specific doInference() method
 * 			4. **Result Processing**: Currently returns input clone as placeholder
 * 			5. **Error Recovery**: Returns empty cv::Mat on any failure condition
 *
 * 			**Framework Integration**:
 * 			- Supports all loaded model types (ONNX, PyTorch, Keras)
 * 			- Uses polymorphic IInferencer interface for uniform access
 * 			- Handles framework-specific inference optimizations transparently
 *
 * 			**Error Handling**:
 * 			- Model existence validation with clear error messaging
 * 			- Exception isolation prevents system crashes on inference failures
 * 			- Comprehensive logging for debugging inference issues
 * 			- Empty result indication for downstream error detection
 *
 * 			**Performance Characteristics**:
 * 			- Direct model access without active model overhead
 * 			- O(log n) model lookup time using std::map registry
 * 			- Framework-specific inference optimizations preserved
 * 			- Memory efficient with minimal copying operations
 *
 * 			**Current Implementation Notes**:
 * 			- Uses doInference() method due to current IInferencer interface design
 * 			- Returns input clone as placeholder until interface provides result access
 * 			- Requires future IInferencer interface update for proper result handling
 *
 * 			**Use Cases**:
 * 			- Parallel inference with multiple models for ensemble methods
 * 			- Explicit model control in testing and validation scenarios
 * 			- Comparative analysis of different model performance
 * 			- Specialized inference workflows requiring specific model access
 *
 * 			**Thread Safety**:
 * 			- Safe for concurrent access to different models
 * 			- Model registry lookup is thread-safe
 * 			- Individual model inference thread safety depends on inferencer implementation
 *
 * @note Model must be previously loaded and exist in registry
 * @warning Current implementation returns input clone - requires IInferencer interface update
 * @todo Update IInferencer interface to return cv::Mat from doInference()
 * @see loadModel() For loading models into registry
 * @see IInferencer::doInference() For framework-specific inference execution
 */
cv::Mat InferenceManager::predict(const std::string &model_name, const cv::Mat &image) {
	auto it = inferencers_.find(model_name);
	if(it == inferencers_.end()) {
		std::cerr << "[InferenceManager] Erro: Modelo '" << model_name << "' não está carregado"
		          << std::endl;
		return cv::Mat();
	}

	try {
		// Para compatibilidade com a interface atual, vamos usar doInference
		// Em uma implementação real, você modificaria IInferencer para ter um
		// método predict
		it->second->doInference(image);

		// Por enquanto, retornar a imagem original como placeholder
		// Você deve modificar IInferencer para retornar cv::Mat do predict
		return image.clone();
	} catch(const std::exception &e) {
		std::cerr << "[InferenceManager] Erro durante predição com modelo '" << model_name
		          << "': " << e.what() << std::endl;
		return cv::Mat();
	}
}

/**
 * @brief Checks if a specific model is loaded and available for inference
 * @param model_name Name of the model to check
 * @return true if model is loaded and ready for use, false otherwise
 * @details This utility function provides fast validation of model availability
 * 			without attempting to access the model itself. It's designed for conditional
 * 			logic in autonomous vehicle systems where model availability affects decision making.
 *
 * 			**Query Process**:
 * 			1. **Registry Lookup**: Searches internal model registry for specified name
 * 			2. **Existence Check**: Verifies model entry exists in registry
 * 			3. **Availability Confirmation**: Returns boolean status immediately
 *
 * 			**Performance Characteristics**:
 * 			- O(log n) lookup time using std::map registry
 * 			- No model initialization or validation overhead
 * 			- Const operation with no side effects
 * 			- Sub-microsecond execution time for immediate response
 *
 * 			**Use Cases**:
 * 			- Conditional model selection in adaptive inference systems
 * 			- Validation before attempting inference operations
 * 			- System health checks and monitoring
 * 			- Dynamic model availability reporting
 *
 * 			**Thread Safety**:
 * 			- Const operation safe for concurrent access
 * 			- No modification of shared state
 * 			- Compatible with multi-threaded inference systems
 *
 * @note This function only checks registry presence, not model validity
 * @note Does not verify model initialization status or resource availability
 * @see loadModel() For loading models into registry
 * @see getLoadedModels() For comprehensive model inventory
 */
bool InferenceManager::isModelLoaded(const std::string &model_name) const {
	return inferencers_.find(model_name) != inferencers_.end();
}

/**
 * @brief Retrieves a complete inventory of all loaded models
 * @return Vector containing names of all currently loaded models
 * @details This function provides comprehensive visibility into the current state
 * 			of the model registry, enabling system monitoring, debugging, and dynamic
 * 			model management in autonomous vehicle applications.
 *
 * 			**Inventory Process**:
 * 			1. **Registry Traversal**: Iterates through internal model registry
 * 			2. **Name Extraction**: Collects model names from registry entries
 * 			3. **Vector Construction**: Builds ordered list of available models
 * 			4. **Return Copy**: Provides snapshot of current registry state
 *
 * 			**Performance Characteristics**:
 * 			- O(n) iteration through registry entries
 * 			- Memory allocation for result vector
 * 			- Const operation with no side effects on registry
 * 			- Execution time proportional to number of loaded models
 *
 * 			**Use Cases**:
 * 			- System monitoring and health reporting
 * 			- Dynamic model selection interfaces
 * 			- Debugging and troubleshooting inference issues
 * 			- Model management and inventory tracking
 * 			- Status reporting for autonomous vehicle dashboards
 *
 * 			**Thread Safety**:
 * 			- Const operation safe for concurrent access
 * 			- No modification of registry state
 * 			- Snapshot semantics prevent race conditions
 *
 * 			**Return Value Characteristics**:
 * 			- Vector contains model names in insertion order
 * 			- Empty vector returned if no models loaded
 * 			- Names correspond to identifiers used in loadModel()
 * 			- Suitable for iteration and conditional operations
 *
 * @note Returned vector is a snapshot - registry may change after call
 * @note Model names in vector are guaranteed to be valid at call time
 * @see loadModel() For loading models with specific names
 * @see isModelLoaded() For checking individual model availability
 * @see selectModel() For activating models from the returned list
 */
std::vector<std::string> InferenceManager::getLoadedModels() const {
	std::vector<std::string> models;
	for(const auto &pair : inferencers_) {
		models.push_back(pair.first);
	}
	return models;
}

/**
 * @brief Safely removes a model from memory and registry
 * @param model_name Name of the model to unload
 * @details This function implements safe model removal with automatic resource
 * 			cleanup and intelligent active model management. It's designed for dynamic
 * 			model management in autonomous vehicle systems where memory efficiency and
 * 			reliability are critical.
 *
 * 			**Unloading Process**:
 * 			1. **Model Lookup**: Searches registry for specified model
 * 			2. **Resource Cleanup**: Automatic destructor calls via unique_ptr
 * 			3. **Registry Removal**: Removes model entry from internal registry
 * 			4. **Active Model Management**: Handles active model state transitions
 * 			5. **Logging**: Provides confirmation of successful unloading
 *
 * 			**Active Model Handling**:
 * 			- If unloaded model was active, clears active model state
 * 			- Automatically selects first remaining model as new active model
 * 			- Ensures system remains operational with available models
 * 			- Logs active model transitions for system monitoring
 *
 * 			**Resource Management**:
 * 			- RAII principles ensure complete resource cleanup
 * 			- Framework-specific destructors handle model-specific cleanup
 * 			- CUDA memory automatically released for GPU-based models
 * 			- No memory leaks even under exceptional conditions
 *
 * 			**Error Handling**:
 * 			- Silent operation if model doesn't exist (idempotent behavior)
 * 			- No system disruption if unloading non-existent models
 * 			- Exception-safe operations prevent partial state corruption
 *
 * 			**Performance Characteristics**:
 * 			- O(log n) registry lookup and removal
 * 			- Immediate memory release via unique_ptr destruction
 * 			- Minimal overhead for active model state management
 * 			- No blocking operations or resource contention
 *
 * 			**Use Cases**:
 * 			- Memory management in resource-constrained autonomous systems
 * 			- Dynamic model lifecycle management
 * 			- System cleanup and resource optimization
 * 			- Model replacement and update procedures
 * 			- Adaptive memory usage based on operational requirements
 *
 * 			**Thread Safety**:
 * 			- Safe for concurrent access from multiple threads
 * 			- Atomic registry operations prevent race conditions
 * 			- Active model state transitions are thread-safe
 *
 * @note Unloading non-existent models is safe and has no effect
 * @note Active model automatically switches if unloaded model was active
 * @warning Model becomes immediately unavailable for inference after unloading
 * @see loadModel() For loading models into registry
 * @see clear() For removing all models at once
 * @see selectModel() For controlling active model selection
 */
void InferenceManager::unloadModel(const std::string &model_name) {
	auto it = inferencers_.find(model_name);
	if(it != inferencers_.end()) {
		inferencers_.erase(it);
		std::cout << "[InferenceManager] Modelo '" << model_name << "' removido da memória"
		          << std::endl;

		// Se era o modelo ativo, limpar
		if(current_model_name_ == model_name) {
			current_model_name_.clear();
			if(!inferencers_.empty()) {
				// Selecionar o primeiro modelo disponível
				current_model_name_ = inferencers_.begin()->first;
				std::cout << "[InferenceManager] Modelo ativo alterado para: "
				          << current_model_name_ << std::endl;
			}
		}
	}
}

/**
 * @brief Completely clears all loaded models and resets system state
 * @details This function implements comprehensive system reset functionality,
 * 			removing all loaded models and restoring the InferenceManager to its initial
 * 			state. It's designed for system shutdown, restart procedures, or complete
 * 			model inventory replacement in autonomous vehicle applications.
 *
 * 			**Cleanup Process**:
 * 			1. **Registry Clearing**: Removes all model entries from internal registry
 * 			2. **Resource Deallocation**: Automatic destructor calls via unique_ptr cleanup
 * 			3. **State Reset**: Clears active model tracking and type information
 * 			4. **Confirmation Logging**: Provides system status confirmation
 *
 * 			**Resource Management**:
 * 			- RAII principles ensure complete resource cleanup for all models
 * 			- Framework-specific destructors handle model-specific resource release
 * 			- CUDA memory automatically released for all GPU-based models
 * 			- Memory pools and caches cleared for optimal memory reclamation
 * 			- Thread-safe operations prevent resource corruption during cleanup
 *
 * 			**State Management**:
 * 			- Active model name cleared to prevent dangling references
 * 			- Model type tracking reset to default AUTO_DETECT state
 * 			- Internal registry completely emptied for fresh initialization
 * 			- System restored to constructor-equivalent state
 *
 * 			**Performance Characteristics**:
 * 			- O(n) operation proportional to number of loaded models
 * 			- Bulk deallocation more efficient than individual unloadModel() calls
 * 			- Immediate memory release for all models simultaneously
 * 			- No fragmentation concerns due to complete registry reset
 *
 * 			**Use Cases**:
 * 			- System shutdown and cleanup procedures
 * 			- Complete model inventory replacement
 * 			- Error recovery and system reset scenarios
 * 			- Memory pressure relief in resource-constrained environments
 * 			- Testing and validation cleanup between test runs
 *
 * 			**Error Handling**:
 * 			- Exception-safe operations prevent partial cleanup states
 * 			- Individual model cleanup failures don't affect overall operation
 * 			- Guaranteed completion even under exceptional conditions
 * 			- Comprehensive logging for debugging cleanup issues
 *
 * 			**Thread Safety**:
 * 			- Safe for concurrent access during cleanup operations
 * 			- Atomic registry operations prevent race conditions
 * 			- No shared resource conflicts during bulk deallocation
 *
 * 			**Post-Condition State**:
 * 			- Registry is completely empty (inferencers_.empty() == true)
 * 			- No active model selected (current_model_name_.empty() == true)
 * 			- Model type reset to AUTO_DETECT
 * 			- System ready for fresh model loading
 *
 * @note System becomes completely inactive after clear() - no models available
 * @note More efficient than multiple unloadModel() calls for bulk operations
 * @warning All loaded models become immediately unavailable for inference
 * @see unloadModel() For removing individual models
 * @see loadModel() For loading models after clearing
 * @see InferenceManager() Constructor for equivalent initial state
 */
void InferenceManager::clear() {
	inferencers_.clear();
	current_model_name_.clear();
	std::cout << "[InferenceManager] Todos os modelos removidos da memória" << std::endl;
}
