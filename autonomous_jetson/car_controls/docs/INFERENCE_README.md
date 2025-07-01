# Sistema de Inferência Multi-Modelo

Este sistema suporta dois tipos principais de modelos de machine learning:

## 1. Modelos PyTorch/ONNX (ONNXInferencer)

### Formatos Suportados:
- `.onnx` - Modelos ONNX nativos
- `.pt` - Modelos PyTorch convertidos para ONNX

### Como Converter PyTorch para ONNX:
```python
import torch
import torchvision

# Carregar seu modelo PyTorch
model = torch.load('model.pt')
model.eval()

# Criar input dummy para tracing
dummy_input = torch.randn(1, 3, 224, 224)

# Exportar para ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)
```

### Uso no C++:
```cpp
ONNXInferencer onnx_model("/path/to/model.onnx");
cv::Mat result = onnx_model.predict(input_image);
```

## 2. Modelos Keras (KerasInferencer)

### Formatos Suportados:
- `.h5` - Modelos Keras salvos
- `SavedModel` - Formato TensorFlow SavedModel

### Opções de Implementação:

#### Opção A: Converter Keras para ONNX (Recomendado)
```python
import tensorflow as tf
import tf2onnx

# Carregar modelo Keras
model = tf.keras.models.load_model('model.h5')

# Converter para ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "keras_model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
```

#### Opção B: Usar Python Embedding (Mais Complexo)
- Requer compilação com suporte ao Python
- Usa a API Python/C para executar modelos Keras

### Uso no C++:
```cpp
// Para modelos convertidos para ONNX
KerasInferencer keras_model("/path/to/keras_model.onnx");

// Para modelos .h5 nativos (requer Python)
KerasInferencer keras_model("/path/to/model.h5");
keras_model.setInputSize(cv::Size(224, 224));
keras_model.setNormalization(
    cv::Scalar(0.485, 0.456, 0.406),  // mean
    cv::Scalar(0.229, 0.224, 0.225)   // std
);

cv::Mat result = keras_model.predict(input_image);
```

## 3. Gerenciador de Inferência (InferenceManager)

Permite usar múltiplos modelos simultaneamente:

```cpp
InferenceManager manager;

// Carregar diferentes tipos de modelos
manager.loadModel("lane_detection", "/path/to/lane_model.onnx");
manager.loadModel("object_classification", "/path/to/class_model.h5");

// Alternar entre modelos
manager.selectModel("lane_detection");
cv::Mat lanes = manager.predict(input_image);

manager.selectModel("object_classification");
cv::Mat objects = manager.predict(input_image);
```

## Configuração de Compilação

### Dependências Básicas:
- OpenCV (com DNN module)
- Qt5 Core

### Para Suporte ONNX:
```bash
# OpenCV já inclui suporte básico ao ONNX
# Nenhuma dependência adicional necessária
```

### Para Suporte Keras Nativo (Opcional):
```bash
# Instalar Python development headers
sudo apt install python3-dev python3-numpy

# Adicionar ao CMake/qmake:
# DEFINES += WITH_PYTHON
# LIBS += -lpython3.x
```

## Estrutura de Arquivos

```
includes/inference/
├── IInferencer.hpp          # Interface base
├── ONNXInferencer.hpp       # Inferenciador ONNX/PyTorch
├── KerasInferencer.hpp      # Inferenciador Keras
└── InferenceManager.hpp     # Gerenciador de modelos

sources/inference/
├── ONNXInferencer.cpp
├── KerasInferencer.cpp
└── InferenceManager.cpp

examples/
└── inference_example.cpp    # Exemplo de uso
```

## Recomendações

1. **Para Novos Projetos**: Use modelos ONNX quando possível
2. **Para Modelos Existentes Keras**: Converta para ONNX para melhor compatibilidade
3. **Para Máxima Flexibilidade**: Use o InferenceManager para alternar entre modelos
4. **Para Performance**: Considere otimizações específicas para sua plataforma (TensorRT na Jetson)

## Troubleshooting

### Erro: "NvInfer.h not found"
- Remova dependências do TensorRT se não estiver na Jetson
- Use modelos ONNX em vez de TensorRT

### Erro: "Python.h not found"
- Instale `python3-dev`
- Ou use conversão para ONNX em vez de Python embedding

### Erro: OpenCV DNN
- Verifique se OpenCV foi compilado com suporte DNN
- `pkg-config --cflags --libs opencv4` deve incluir DNN
