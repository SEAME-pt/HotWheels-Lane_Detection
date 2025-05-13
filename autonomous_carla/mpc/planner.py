"""
planner.py

Este arquivo implementa o planejador de trajetória baseado em Controle Preditivo de Modelo (MPC). Ele utiliza o otimizador definido no arquivo optimizer.py para calcular os controles ótimos (aceleração e direção) para o veículo seguir uma trajetória desejada.

Classes:
- MPCPlanner: Classe principal que implementa o planejamento de trajetória usando MPC.

Métodos:
- __init__: Inicializa o planejador com as configurações fornecidas.
- plan: Gera a trajetória ótima com base no estado atual e nos waypoints fornecidos.
- _prepare_reference: Converte os waypoints globais em uma trajetória de referência no referencial local do veículo.

Dependências:
- numpy: Biblioteca para cálculos numéricos.
- optimizer.MPCOptimizer: Classe que resolve o problema de otimização do MPC.
"""

import numpy as np
from .optimizer import MPCOptimizer

class MPCPlanner:
    def __init__(self, config):
        """
        Inicializa o planejador MPC com as configurações fornecidas.

        Args:
            config (dict): Dicionário contendo parâmetros como dt (passo de tempo), horizon (horizonte de previsão), etc.
        """
        self.config = config
        self.optimizer = MPCOptimizer(config)
        self.dt = config.get('dt', 0.1)  # Passo de tempo
        self.horizon = config.get('horizon', 10)  # Horizonte de previsão

    def plan(self, current_state, waypoints):
        """
        Gera a trajetória ótima usando MPC.

        Args:
            current_state (dict): Estado atual do veículo com x, y, yaw e velocity.
            waypoints (list): Lista de coordenadas (x, y) que o veículo deve seguir.

        Returns:
            dict: Controles ótimos contendo throttle (aceleração) e steer (direção).
        """
        # Extrai o estado atual
        x0 = current_state['x']
        y0 = current_state['y']
        yaw0 = current_state['yaw']
        v0 = current_state['velocity']

        # Prepara a trajetória de referência a partir dos waypoints
        reference = self._prepare_reference(x0, y0, yaw0, waypoints)

        # Resolve o problema de otimização do MPC
        throttle, steer = self.optimizer.solve(x0, y0, yaw0, v0, reference)

        return {
            'throttle': throttle,
            'steer': steer
        }

    def _prepare_reference(self, x0, y0, yaw0, waypoints):
        """
        Converte os waypoints globais em uma trajetória de referência no referencial local do veículo.

        Args:
            x0, y0 (float): Posição atual do veículo.
            yaw0 (float): Orientação atual do veículo.
            waypoints (list): Lista de coordenadas globais (x, y).

        Returns:
            list: Lista de pontos (x, y) no referencial local do veículo.
        """
        local_points = []

        for wp_x, wp_y in waypoints:
            # Translação
            dx = wp_x - x0
            dy = wp_y - y0

            # Rotação
            local_x = dx * np.cos(-yaw0) - dy * np.sin(-yaw0)
            local_y = dx * np.sin(-yaw0) + dy * np.cos(-yaw0)

            local_points.append((local_x, local_y))

        return local_points[:self.horizon]
