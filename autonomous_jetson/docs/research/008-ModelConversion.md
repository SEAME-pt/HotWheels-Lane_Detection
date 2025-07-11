# Model Conversion
## Introduction
The target machine that will run the deep learning models has a NVIDIA TensorRT inference, developed to accelerate deep learning models on GPUs and achieve maximum performance. However, the host machine used in the training of said models uses a TensorFlow approach. Instead of remaking the model all over again with a different framework, it is possible to apply a conversion process and migrate the models from one machine to another.  

## Open Neural Network Exchange (ONNX)
ONNX is an open format for representing machine learning models. It enables models to be shared between different frameworks, such as PyTorch, TensorFlow, and others, by providing a common standard. It supports both training and inference and is widely used to export models from one environment and run them in another (especially with optimized runtimes like ONNX Runtime or NVIDIA TensorRT, making it a key tool for ensuring flexibility, portability, and performance in machine learning workflows).

## Application
The correct process for integration would be to convert the keras file that the TensorFlow model outputs into a ONNX file and then convert it to TensorRT. After that we grab the resulting .engine file and use it in the inference. 

> **Why not convert it to TensorRT directly?**
> - Less flexible: TensorRT for TensorFlow does not support all operations and may require changes to the model  
> - Worst performance: Direct conversion may not be as optimized as an ONNX → TensorRT pipeline  

### Convert Tensorflow to ONNX (host machine)
	# Load the Keras model
	keras_model = tf.keras.models.load_model(keras_model_path)
	
	# Convert to ONNX using tf2onnx
	input_signature = [tf.TensorSpec(shape=keras_model.inputs[0].shape, dtype=keras_model.inputs[0].dtype)]

	# Convert the model
	onnx_model, _ = tf2onnx.convert.from_keras(keras_model, input_signature, opset=13)
	
	# Save the ONNX model
	onnx.save(onnx_model, onnx_model_path)

### SCP to Jetson Nano
	scp models/lane_detector_lab_finetuned.onnx $JETSON:/home/hotweels/dev/model_loader/models

### Convert ONNX to TensorRT (target machine)
	def build_engine(onnx_file_path, engine_file_path, precision='fp16', max_workspace_size=1<<28):
		"""
		Build TensorRT engine from ONNX file specifically for Jetson Nano
		
		Args:
			onnx_file_path: Path to the ONNX model
			engine_file_path: Path to save the TensorRT engine
			precision: 'fp16' or 'fp32'
			max_workspace_size: Maximum workspace size in bytes (default: 256 MiB)
		"""
		# Initialize TensorRT stuff
		TRT_LOGGER = trt.Logger(trt.Logger.INFO)
		
		print(f"TensorRT version: {trt.__version__}")
		print(f"Building TensorRT engine for Jetson Nano from: {onnx_file_path}")
		print(f"Target engine path: {engine_file_path}")
		
		# Create builder and config
		builder = trt.Builder(TRT_LOGGER)
		config = builder.create_builder_config()
		config.max_workspace_size = max_workspace_size
		
		# Set precision flag
		if precision == 'fp16' and builder.platform_has_fast_fp16:
			print("Using FP16 precision")
			config.set_flag(trt.BuilderFlag.FP16)
		else:
			print("Using FP32 precision")
		
		# Create network definition
		explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
		network = builder.create_network(explicit_batch)
		
		# Parse ONNX file
		parser = trt.OnnxParser(network, TRT_LOGGER)
		
		# Read the ONNX file
		with open(onnx_file_path, 'rb') as model:
			print("Parsing ONNX file...")
			if not parser.parse(model.read()):
				print("ERROR: Failed to parse the ONNX file.")
				for error in range(parser.num_errors):
					print(parser.get_error(error))
				return False
		
		print("Successfully parsed ONNX model.")
		
		# Print input dimensions
		input_tensor = network.get_input(0)
		input_shape = input_tensor.shape
		print(f"Input tensor shape: {input_shape}")
		
		# Create optimization profile for dynamic shape support
		profile = builder.create_optimization_profile()
		
		# Set shapes for optimization profile
		# Assuming NHWC format (batch, height, width, channels)
		min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
		opt_shape = (1, input_shape[1], input_shape[2], input_shape[3])
		max_shape = (1, input_shape[1], input_shape[2], input_shape[3])
		
		profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
		config.add_optimization_profile(profile)
		
		# Build the engine
		print("Building TensorRT engine... This might take a while.")
		engine = builder.build_engine(network, config)
		if engine is None:
			print("Failed to build TensorRT engine!")
			return False
		
		# Serialize the engine to a file
		print("Serializing engine to file...")
		with open(engine_file_path, 'wb') as f:
			f.write(engine.serialize())
		
		print(f"TensorRT engine successfully built and saved to: {engine_file_path}")
		return True

___