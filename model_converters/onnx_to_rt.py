import tensorrt as trt

onnx_model_path = "model.onnx"
trt_engine_path = "model.trt"

# Criar logger
logger = trt.Logger(trt.Logger.WARNING)

# Criar builder, rede e parser
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# Ler e parsear o modelo ONNX
with open(onnx_model_path, "rb") as model_file:
    if not parser.parse(model_file.read()):
        print("Erro ao converter ONNX para TensorRT!")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        exit(1)

# Criar configuração do builder
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * (1024 * 1024))  # 2GB workspace

# Criar um perfil de otimização para entradas dinâmicas
profile = builder.create_optimization_profile()

for i in range(network.num_inputs):
    input_tensor = network.get_input(i)
    input_name = input_tensor.name
    input_shape = input_tensor.shape

    # Definir tamanhos mínimo, ótimo e máximo
    min_shape = [1 if dim == -1 else dim for dim in input_shape]  # Batch mínimo = 1
    opt_shape = [max(1, dim) if dim == -1 else dim for dim in input_shape]  # Ótimo = valor realista
    max_shape = [max(4, dim) if dim == -1 else dim for dim in input_shape]  # Batch máximo = 4

    profile.set_shape(input_name, min_shape, opt_shape, max_shape)

# Adicionar o perfil à configuração
config.add_optimization_profile(profile)

# Ativar FP16 se suportado
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# Criar o runtime e o serializador do engine
runtime = trt.Runtime(logger)
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    print("Falha ao serializar o modelo TensorRT!")
    exit(1)

# Guardar o modelo TensorRT
with open(trt_engine_path, "wb") as engine_file:
    engine_file.write(serialized_engine)

print(f"Modelo convertido para TensorRT e salvo em {trt_engine_path}")