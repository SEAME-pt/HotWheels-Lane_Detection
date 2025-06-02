"""
optimizer.py

Este arquivo implementa o otimizador para o Controle Preditivo de Modelo (MPC). Ele utiliza a biblioteca scipy.optimize para resolver o problema de otimização e calcular os controles ótimos para o veículo.

Classes:
- MPCOptimizer: Classe principal que implementa o otimizador MPC.

Métodos:
- __init__: Inicializa o otimizador com as configurações fornecidas.
- solve: Resolve o problema de otimização para calcular os controles ótimos.
- _cost_function: Define a função de custo para o problema de otimização.
- _update_state: Atualiza o estado do veículo usando o modelo cinemático de bicicleta.
- _normalize_angle: Normaliza um ângulo para o intervalo [-pi, pi].

Dependências:
- numpy: Biblioteca para cálculos numéricos.
- scipy.optimize: Biblioteca para resolver problemas de otimização.
"""

import numpy as np
from scipy.optimize import minimize

class MPCOptimizer:
    def __init__(self, config):
        """
        Inicializa o otimizador MPC com as configurações fornecidas.

        Args:
            config (dict): Dicionário contendo parâmetros como dt, horizon, wheelbase, pesos para a função de custo, etc.
        """
        self.config = config
        self.dt = config.get('dt', 0.1)
        self.horizon = config.get('horizon', 15)  # Número de passos de previsão
        self.wheelbase = config.get('wheelbase', 2.7)  # Distância entre eixos do veículo

        # Pesos para a função de custo
        self.w_cte = config.get('w_cte', 1.0)  # Peso do erro de trajetória
        self.w_etheta = config.get('w_etheta', 1.0)  # Peso do erro de orientação
        self.w_vel = config.get('w_vel', 1.0)  # Peso do erro de velocidade
        self.w_steer = config.get('w_steer', 1.0)  # Peso do controle de direção
        self.w_accel = config.get('w_accel', 1.0)  # Peso do controle de aceleração
        self.w_steer_rate = config.get('w_steer_rate', 1.0)  # Peso da taxa de variação da direção
        self.w_lane = config.get('w_lane', 5.0)  # Peso para manter-se dentro das faixas

        # Restrições de controle
        self.max_steer = config.get('max_steer', 0.5)  # Ângulo máximo de direção
        self.max_throttle = config.get('max_throttle', 1.0)  # Aceleração máxima
        self.max_brake = config.get('max_brake', 1.0)  # Frenagem máxima
        self.w_lane_yaw = config.get('w_lane_yaw', 3.0) # Peso para o erro de orientação em relação à faixa

    def solve(self, x0, y0, yaw0, v0, reference, lane_info=None):
        """
        Resolve o problema de otimização do MPC.

        Args:
            x0, y0 (float): Posição inicial.
            yaw0 (float): Orientação inicial.
            v0 (float): Velocidade inicial.
            reference (list): Lista de waypoints (x, y) no referencial local.

        Returns:
            tuple: Controles ótimos (throttle, steer).
        """
        # Estado inicial
        state = np.array([0.0, 0.0, 0.0, v0])  # x, y, yaw, velocidade no referencial local

        # Chute inicial para os controles (aceleração, direção) - começar com aceleração positiva
        u0 = np.zeros(2 * self.horizon)
        # Inicializar com throttle positivo para garantir movimento
        for i in range(0, 2 * self.horizon, 2):
            u0[i] = 0.5  # Throttle inicial de 0.5

        # Define os limites para os controles
        bounds = []
        for _ in range(self.horizon):
            bounds.append((0.0, self.max_throttle))  # Limites de aceleração (sempre positiva)
            bounds.append((-self.max_steer, self.max_steer))  # Limites de direção

        # Resolve o problema de otimização
        result = minimize(
            fun=lambda u: self._cost_function(u, state, reference, lane_info),
            x0=u0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'disp': False}
        )

        # Extrai a primeira ação de controle
        throttle = result.x[0]
        steer = result.x[1]

        return throttle, steer

    def _cost_function(self, u, state, reference, lane_info=None):
        cost = 0.0
        x, y, yaw, v = state

        controls = []
        for i in range(self.horizon):
            throttle = u[2 * i]
            steer = u[2 * i + 1]
            controls.append((throttle, steer))

        current_x, current_y, current_yaw, current_v = x, y, yaw, v

        # Definir valores padrão para os pesos
        w_cte = 5.0
        w_etheta = 3.0
        w_velocity = 2.0
        w_throttle = 0.01
        w_steer = 0.1
        target_speed = 4.0

        # Detectar se estamos em uma curva
        curvature = self._calculate_path_curvature(reference)
        is_curve = abs(curvature) > 0.1

        # Ajustar pesos baseado na curvatura
        if is_curve:
            w_cte = 15.0
            w_etheta = 10.0
            w_velocity = 0.5
            w_throttle = 0.1
            w_steer = 0.05
            target_speed = 2.0

        for i in range(self.horizon):
            throttle, steer = controls[i]
            dt = 0.1
            current_v += throttle * dt
            current_v = max(0, min(current_v, 8))
            current_x += current_v * np.cos(current_yaw) * dt
            current_y += current_v * np.sin(current_yaw) * dt
            current_yaw += (current_v / 2.5) * np.tan(steer) * dt

            if i < len(reference):
                ref_x, ref_y = reference[i]
                cost += w_cte * ((current_x - ref_x)**2 + (current_y - ref_y)**2)
            cost += w_velocity * (current_v - target_speed)**2
            cost += w_throttle * (throttle**2)
            cost += w_steer * (steer**2)

        return cost


    def _update_state(self, x, y, yaw, v, throttle, steer):
        """
        Atualiza o estado do veículo usando o modelo cinemático de bicicleta.

        Args:
            x, y (float): Posição atual.
            yaw (float): Orientação atual.
            v (float): Velocidade atual.
            throttle (float): Controle de aceleração.
            steer (float): Controle de direção.

        Returns:
            tuple: Novo estado (x, y, yaw, v).
        """
        # Modelo de aceleração simples
        v_next = v + throttle * self.dt

        # Garante que a velocidade seja positiva e tenha um mínimo
        v_next = max(5.0, v_next)  # Velocidade mínima de 5.0 m/s (aproximadamente 18 km/h)

        # Modelo de bicicleta
        beta = np.arctan(0.5 * np.tan(steer))
        x_next = x + v * np.cos(yaw + beta) * self.dt
        y_next = y + v * np.sin(yaw + beta) * self.dt
        yaw_next = yaw + v * np.sin(beta) / (self.wheelbase/2) * self.dt

        return x_next, y_next, yaw_next, v_next

    def _normalize_angle(self, angle):
        """
        Normaliza um ângulo para o intervalo [-pi, pi].

        Args:
            angle (float): Ângulo em radianos.

        Returns:
            float: Ângulo normalizado.
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _calculate_path_curvature(self, reference):
        """Calcular curvatura do caminho de referência"""
        if len(reference) < 3:
            return 0.0
        
        # Usar os primeiros 3 pontos para estimar curvatura
        p1, p2, p3 = reference[0], reference[1], reference[2]
        
        # Calcular curvatura usando fórmula de Menger
        a = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        b = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)
        c = np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2)
        
        if a * b * c == 0:
            return 0.0
        
        # Área do triângulo
        s = (a + b + c) / 2
        area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
        
        # Curvatura = 4 * área / (a * b * c)
        curvature = 4 * area / (a * b * c) if (a * b * c) > 0 else 0.0
        
        return curvature