"""
main.py

Este script é responsável por controlar a simulação no ambiente CARLA. Ele carrega as configurações, inicializa a interface com o CARLA, configura o planejador de controle preditivo (MPC) e executa o loop principal de controle do veículo.

Funções principais:
- load_config: Carrega as configurações de um arquivo YAML.
- main: Função principal que gerencia a simulação.

Fluxo principal:
1. Carrega as configurações do arquivo 'config/carla_config.yaml'.
2. Inicializa a interface com o CARLA e conecta ao servidor.
3. Cria um veículo no ambiente de simulação.
4. Inicializa o planejador MPC com as configurações apropriadas.
5. Obtém waypoints dinâmicos para o planejamento de trajetória.
6. Executa o loop principal que:
   - Obtém o estado atual do veículo.
   - Planeja a trajetória usando o MPC.
   - Aplica os controles calculados ao veículo.
   - Atualiza a câmera do espectador para seguir o veículo.
7. Lida com interrupções do usuário e realiza a limpeza final.

Dependências:
- yaml: Para carregar configurações do arquivo YAML.
- time: Para gerenciar atrasos no loop principal.
- numpy: Para cálculos numéricos.
- carla_interface: Interface personalizada para interagir com o CARLA.
- mpc.planner: Planejador MPC para controle do veículo.

"""

import pygame
import cv2
import yaml
import time
import numpy as np
from carla_interface import CarlaInterface
from process_image import KerasLaneDetector
from mpc.planner import MPCPlanner

detector = KerasLaneDetector('models/lane_detector_combined.keras')
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Lane Detection")

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config('config/carla_config.yaml')
    
    # Initialize Carla interface
    carla_interface = CarlaInterface(config)
    
    try:
        # Connect to Carla
        carla_interface.connect()
        
        # Spawn vehicle
        carla_interface.spawn_vehicle()
        
        # Initialize MPC planner
        mpc_planner = MPCPlanner(config.get('mpc', {}))
        
        #! Substitua a definição estática:
        # Simple waypoints for testing (replace with lane detection output later)
        # waypoints = [(10, 0), (20, 0), (30, 0), (40, 10), (50, 20)]

        # Por uma chamada dinâmica:
        waypoints = carla_interface.get_waypoints_ahead(distance=5.0, count=20)
        
        # Main control loop
        while True:
            # Get current vehicle state
            current_state = carla_interface.get_vehicle_state()
            if not current_state:
                print("No vehicle state available")
                time.sleep(0.1)
                continue
        
            # Pegue a imagem já em numpy array RGB
            image = carla_interface.get_camera_image()
            if image is not None:
                # Passe diretamente para o detector
                mask = detector.process_image(image)
                vis = detector.process_and_visualize(mask)
        
                if vis is not None:
                    surf = pygame.surfarray.make_surface(vis.swapaxes(0, 1))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
        
                # Se quiser usar OpenCV para visualizar também:
                lanes = detector.detect_lanes(image)
                cv2.imshow("Lane Detection", lanes)
                cv2.waitKey(1)
        
            # Plan trajectory using MPC
            control = mpc_planner.plan(current_state, waypoints)
        
            # Apply control to vehicle
            carla_interface.apply_control(
                throttle=control['throttle'],
                steer=control['steer']
            )
        
            # Update spectator camera to follow the vehicle
            carla_interface.update_spectator()
        
            # Sleep to maintain control frequency
            time.sleep(config.get('control_frequency', 0.1))
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        # Clean up
        carla_interface.cleanup()

if __name__ == "__main__":
    main()
