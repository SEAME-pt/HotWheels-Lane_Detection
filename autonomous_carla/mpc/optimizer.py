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
        self.horizon = config.get('horizon', 10)
        self.wheelbase = config.get('wheelbase', 2.7)  # Distância entre eixos do veículo

        # Pesos para a função de custo
        self.w_cte = config.get('w_cte', 1.0)  # Peso do erro de trajetória
        self.w_etheta = config.get('w_etheta', 1.0)  # Peso do erro de orientação
        self.w_vel = config.get('w_vel', 1.0)  # Peso do erro de velocidade
        self.w_steer = config.get('w_steer', 1.0)  # Peso do controle de direção
        self.w_accel = config.get('w_accel', 1.0)  # Peso do controle de aceleração
        self.w_steer_rate = config.get('w_steer_rate', 1.0)  # Peso da taxa de variação da direção

        # Restrições de controle
        self.max_steer = config.get('max_steer', 0.5)  # Ângulo máximo de direção
        self.max_throttle = config.get('max_throttle', 1.0)  # Aceleração máxima
        self.max_brake = config.get('max_brake', 1.0)  # Frenagem máxima

    def solve(self, x0, y0, yaw0, v0, reference):
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

        # Chute inicial para os controles (aceleração, direção)
        u0 = np.zeros(2 * self.horizon)

        # Define os limites para os controles
        bounds = []
        for _ in range(self.horizon):
            bounds.append((-self.max_throttle, self.max_throttle))  # Limites de aceleração
            bounds.append((-self.max_steer, self.max_steer))  # Limites de direção

        # Resolve o problema de otimização
        result = minimize(
            fun=lambda u: self._cost_function(u, state, reference),
            x0=u0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'disp': False}
        )

        # Extrai a primeira ação de controle
        throttle = result.x[0]
        steer = result.x[1]

        return throttle, steer

    def _cost_function(self, u, state, reference):
        """
        Função de custo para a otimização do MPC.

        Args:
            u (array): Vetor de controles (aceleração, direção).
            state (array): Estado inicial do veículo.
            reference (list): Trajetória de referência.

        Returns:
            float: Custo total.
        """
        cost = 0.0
        x, y, yaw, v = state

        # Reorganiza os controles
        controls = u.reshape(self.horizon, 2)

        for i in range(self.horizon):
            # Referência para este passo
            ref_x, ref_y = reference[min(i, len(reference)-1)]

            # Erro de trajetória
            cte = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)

            # Erro de orientação
            ref_yaw = np.arctan2(ref_y - y, ref_x - x)
            etheta = self._normalize_angle(yaw - ref_yaw)

            # Adiciona ao custo
            cost += self.w_cte * cte**2
            cost += self.w_etheta * etheta**2

            # Custos de controle
            throttle, steer = controls[i]
            cost += self.w_steer * steer**2
            cost += self.w_accel * throttle**2

            # Custos de taxa de controle
            if i > 0:
                prev_throttle, prev_steer = controls[i-1]
                cost += self.w_steer_rate * (steer - prev_steer)**2

            # Atualiza o estado para o próximo passo
            x, y, yaw, v = self._update_state(x, y, yaw, v, throttle, steer)

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

        # Garante que a velocidade seja positiva
        v_next = max(0.0, v_next)

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
