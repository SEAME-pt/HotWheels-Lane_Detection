import cv2
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def fit_lane_with_ransac(x, y):
    if len(x) < 10:
        return None
    model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=5, random_state=42))
    try:
        model.fit(y.reshape(-1, 1), x)
        return model.predict(plot_y.reshape(-1, 1))
    except Exception as e:
        print(f"Erro ao ajustar RANSAC: {e}")
        return None

def extract_center_curve_with_lanes(binary_mask):
    """
    Extrai curvas de faixas e a curva central usando RANSAC.
    
    Args:
        binary_mask: Máscara binária da segmentação de faixas
        
    Returns:
        tuple: (center_curve, spline_curves) - center_curve é (center_x, center_y) e 
               spline_curves é uma lista de tuplas (avg_x, curve_x, curve_y)
    """
    # Obter coordenadas dos pixels brancos
    y_indices, x_indices = np.nonzero(binary_mask)
    
    # Verificar se há pontos suficientes - reduzido para 10 para ser mais sensível
    if len(x_indices) < 10:
        return None
    
    mid_x = binary_mask.shape[1] // 2

    left_mask = x_indices < mid_x
    right_mask = x_indices >= mid_x

    left_x = x_indices[left_mask]
    left_y = y_indices[left_mask]
    right_x = x_indices[right_mask]
    right_y = y_indices[right_mask]

    # Definir os pontos y para o plot
    plot_y = np.linspace(0, binary_mask.shape[0] - 1, 200)

    # Ajustar faixas usando RANSAC
    left_x_pred = None
    right_x_pred = None
    
    if len(left_x) >= 10:
        model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=10, random_state=42))
        try:
            model.fit(left_y.reshape(-1, 1), left_x)
            left_x_pred = model.predict(plot_y.reshape(-1, 1))
        except Exception as e:
            print(f"Erro ao ajustar RANSAC para faixa esquerda: {e}")
    
    if len(right_x) >= 10:
        model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(residual_threshold=10, random_state=42))
        try:
            model.fit(right_y.reshape(-1, 1), right_x)
            right_x_pred = model.predict(plot_y.reshape(-1, 1))
        except Exception as e:
            print(f"Erro ao ajustar RANSAC para faixa direita: {e}")
    
    # Preparar retorno no mesmo formato do ransacV2
    spline_curves = []
    
    if left_x_pred is not None:
        # Garantir que os valores estão dentro dos limites da imagem
        in_bounds = (left_x_pred >= 0) & (left_x_pred < binary_mask.shape[1])
        if np.sum(in_bounds) >= 10:
            # (avg_x, curve_x, curve_y)
            left_x_valid = left_x_pred[in_bounds]
            plot_y_valid = plot_y[in_bounds]
            spline_curves.append((np.mean(left_x_valid), left_x_valid, plot_y_valid))
    
    if right_x_pred is not None:
        # Garantir que os valores estão dentro dos limites da imagem
        in_bounds = (right_x_pred >= 0) & (right_x_pred < binary_mask.shape[1])
        if np.sum(in_bounds) >= 10:
            # (avg_x, curve_x, curve_y)
            right_x_valid = right_x_pred[in_bounds]
            plot_y_valid = plot_y[in_bounds]
            spline_curves.append((np.mean(right_x_valid), right_x_valid, plot_y_valid))
    
    # Ordenar faixas da esquerda para a direita
    spline_curves.sort(key=lambda item: item[0])
    
    # Calcular trajetória central
    center_curve = None
    if len(spline_curves) >= 2:
        left_x = spline_curves[0][1]
        right_x = spline_curves[1][1]
        shared_y = spline_curves[0][2]  # Assumindo mesmo intervalo Y
        min_len = min(len(left_x), len(right_x))
        
        # Ordenar por Y do mais próximo (maior Y) para o mais distante (menor Y)
        center_x = (left_x[:min_len] + right_x[:min_len]) / 2
        center_y = shared_y[:min_len]
        
        # Verificar se os pontos estão ordenados e inverter se necessário (y aumenta para baixo na imagem)
        if len(center_y) > 1 and center_y[0] < center_y[-1]:
            center_x = center_x[::-1]
            center_y = center_y[::-1]
            
        center_curve = (center_x, center_y)
    
    return center_curve, spline_curves

# Código para execução direta (quando o arquivo é executado como script)
if __name__ == "__main__":
    # Verificar argumento
    if len(sys.argv) != 2:
        print("Uso: python script.py <imagem_binaria>")
        sys.exit(1)

    img_path = sys.argv[1]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem.")
        sys.exit(1)

    # Obter coordenadas dos pixels brancos
    y_indices, x_indices = np.nonzero(img)
    mid_x = img.shape[1] // 2

    left_mask = x_indices < mid_x
    right_mask = x_indices >= mid_x

    left_x = x_indices[left_mask]
    left_y = y_indices[left_mask]
    right_x = x_indices[right_mask]
    right_y = y_indices[right_mask]

    plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])

    left_x_pred = fit_lane_with_ransac(left_x, left_y)
    right_x_pred = fit_lane_with_ransac(right_x, right_y)

    if left_x_pred is not None and right_x_pred is not None:
        center_x = (left_x_pred + right_x_pred) / 2
    else:
        center_x = None

    # Visualização
    plt.imshow(img, cmap='gray')
    if left_x_pred is not None:
        plt.plot(left_x_pred, plot_y, color='red', label='Left (RANSAC)')
    if right_x_pred is not None:
        plt.plot(right_x_pred, plot_y, color='blue', label='Right (RANSAC)')
    if center_x is not None:
        plt.plot(center_x, plot_y, color='green', label='Center')
    plt.title("Lane Detection with RANSAC")
    plt.legend(loc='upper right')
    plt.show()
