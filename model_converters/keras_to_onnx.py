import tensorflow as tf
import os
import subprocess
import sys
from tensorflow.keras.metrics import MeanIoU

# Registrar a métrica personalizada corretamente
@tf.keras.utils.register_keras_serializable()
class BinaryMeanIoU(MeanIoU):
    def __init__(self, num_classes=2, name="binary_mean_iou", dtype=None):
        super().__init__(num_classes=num_classes, name=name, dtype=dtype)

# Adicionar ao dicionário de objetos personalizados
tf.keras.utils.get_custom_objects()["BinaryMeanIoU"] = BinaryMeanIoU

def convert_keras_to_onnx(keras_model_path, onnx_model_path):
    # Criar diretório temporário para SavedModel
    saved_model_dir = "saved_model_temp"

    # Carregar o modelo .keras
    model = tf.keras.models.load_model(keras_model_path)

    # Exportar como SavedModel (usando export() no Keras 3)
    model.export(saved_model_dir)

    # Converter para ONNX usando tf2onnx CLI
    command = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", onnx_model_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Conversão bem-sucedida! Modelo ONNX salvo como {onnx_model_path}")
    except subprocess.CalledProcessError as e:
        print("Erro na conversão para ONNX:", e)

    # Remover o diretório temporário
    os.system(f"rm -rf {saved_model_dir}")

# Caminho do modelo .keras e do output ONNX
keras_model_path = "lane_detector_final.keras"
onnx_model_path = "model.onnx"

convert_keras_to_onnx(keras_model_path, onnx_model_path)
