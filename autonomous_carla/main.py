import os
import pygame
import cv2
import yaml
import time
import numpy as np

# CONFIGURAÇÃO CRÍTICA PARA GPU - DEVE VIR ANTES DE QUALQUER IMPORT DO TENSORFLOW
os.environ['XLA_FLAGS'] = '--xla_gpu_strict_conv_algorithm_picker=false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduzir logs verbosos

import tensorflow as tf

DEBUG = True  # Ativar modo de debug

def print_debug_info(text):
	"""Função de debug para imprimir mensagens com timestamp"""
	if DEBUG:
		print(f"[DEBUG] {time.strftime('%Y-%m-%d %H:%M:%S')} - {text}")

# Configurar GPU para crescimento dinâmico de memória
def configure_gpu():
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			# Permitir crescimento dinâmico de memória
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			
			# Limitar memória se necessário
			tf.config.experimental.set_virtual_device_configuration(
				gpus[0],
				[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
			)
			print(f"GPU configurada: {gpus[0]}")
		except RuntimeError as e:
			print(f"Erro ao configurar GPU: {e}")
	else:
		print("Nenhuma GPU detectada, usando CPU")

# Configurar GPU ANTES de importar outros módulos
configure_gpu()

import carla
from carla_interface import CarlaInterface
from process_image import KerasLaneDetector
from mpc.planner import MPCPlanner
from mpc.projection_utils import convert_image_points_to_world
from polinomial.polyfit import fit_lanes_in_image, compute_virtual_centerline

# Inicializar detector APÓS configuração da GPU
detector = KerasLaneDetector('models/lane_detector_combined_v2.keras')

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Lane Detection with Polyfit")

def extract_center_curve_with_lanes(binary_mask):
    """Versão com suavização para curvas"""
    if binary_mask is None:
        return None
    
    try:
        # Aplicar filtro Gaussiano para suavizar a máscara
        smoothed_mask = cv2.GaussianBlur(binary_mask, (5, 5), 1.0)
        
        # Usar o algoritmo do polyfit.py
        lanes = fit_lanes_in_image(smoothed_mask)
        
        if not lanes:
            return None
        
        # Calcular a linha central virtual
        result = compute_virtual_centerline(lanes, smoothed_mask.shape[1], smoothed_mask.shape[0])
        
        if result is not None:
            x_blend, y_blend, x_c1, x_c2 = result
            
            # Suavizar a trajetória central usando filtro de média móvel
            window_size = 5
            if len(x_blend) >= window_size:
                x_smooth = np.convolve(x_blend, np.ones(window_size)/window_size, mode='same')
                y_smooth = np.convolve(y_blend, np.ones(window_size)/window_size, mode='same')
                
                print_debug_info(f"Trajetória suavizada: {len(x_smooth)} pontos")
                return (x_smooth, y_smooth), lanes
            else:
                return (x_blend, y_blend), lanes
        
        return None
        
    except Exception as e:
        print(f"[ERROR] Erro em extract_center_curve_with_lanes: {e}")
        return None


def extract_lane_info(mask, center_curve=None):
    """Extrai informações das faixas usando a trajetória central calculada."""
    if mask is None or center_curve is None:
        return 0.0, 0.0
    center_x, center_y = center_curve
    height, width = mask.shape[:2]
    # Calcular desvio lateral baseado na trajetória central
    # Ponto mais próximo do carro (maior Y)
    idx = np.argmax(center_y)
    car_center_x = width // 2
    lane_center_x = center_x[idx]
    # Escala em metros por pixel (deve ser igual à usada na projeção)
    real_height_m = 8.0
    escala_m_por_pixel = real_height_m / height
    # Inverter o sinal do erro lateral
    lateral_offset = (car_center_x - lane_center_x) * escala_m_por_pixel
    # Calcular erro de orientação usando a inclinação da trajetória
    yaw_error = 0.0
    if len(center_x) > 5:
        # Usar os últimos pontos para calcular a inclinação
        x1, y1 = center_x[-5], center_y[-5]
        x2, y2 = center_x[-1], center_y[-1]
        if x2 != x1:
            yaw_error = np.arctan2(y2 - y1, x2 - x1) - np.pi / 2  # Corrigir para eixo Y para frente
    return lateral_offset, yaw_error


def load_config(config_path):
	"""Load configuration from YAML file"""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config

def create_polyfit_visualization(binary_mask, center_curve, lanes):
	"""
	Cria visualização das faixas detectadas e trajetória central.
	"""
	if binary_mask is None:
		return None
	
	# Criar superfície de visualização
	height, width = binary_mask.shape[:2]
	vis_surface = pygame.Surface((width, height))
	
	# Converter máscara para RGB
	binary_rgb = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
	binary_surface = pygame.surfarray.make_surface(binary_rgb.swapaxes(0, 1))
	vis_surface.blit(binary_surface, (0, 0))
	
	# Desenhar faixas detectadas
	colors = [(255, 0, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255)]
	
	if lanes:
		for i, lane in enumerate(lanes):
			if 'curve' in lane:
				x_coords, y_coords = lane['curve']
				color = colors[i % len(colors)]
				
				# Converter para pontos válidos
				points = []
				for x, y in zip(x_coords, y_coords):
					if 0 <= x < width and 0 <= y < height:
						points.append((int(x), int(y)))
				
				if len(points) > 1:
					pygame.draw.lines(vis_surface, color, False, points, 2)
	
	# Desenhar trajetória central em verde
	if center_curve is not None:
		center_x, center_y = center_curve
		center_points = []
		
		for x, y in zip(center_x, center_y):
			if 0 <= x < width and 0 <= y < height:
				center_points.append((int(x), int(y)))
		
		if len(center_points) > 1:
			pygame.draw.lines(vis_surface, (0, 255, 0), False, center_points, 4)
	
	return vis_surface

def safe_model_predict(detector, rgb_array):
	"""
	Função segura para predição do modelo com tratamento de erros GPU.
	"""
	try:
		# Primeira tentativa com configuração padrão
		binary_mask = detector.predict(detector.preprocess_image(rgb_array))
		return binary_mask
	except Exception as e:
		print(f"[WARNING] Erro na predição GPU: {e}")
		try:
			# Segunda tentativa forçando CPU
			with tf.device('/CPU:0'):
				binary_mask = detector.predict(detector.preprocess_image(rgb_array))
				print("[INFO] Usando CPU para predição")
				return binary_mask
		except Exception as e2:
			print(f"[ERROR] Erro também na CPU: {e2}")
			return None

def initialize_system(config_path):
	"""
	Inicializa o sistema carregando a configuração e preparando o ambiente.
	"""
	config = load_config(config_path)
	carla_interface = CarlaInterface(config)
	carla_interface.connect()
	carla_interface.spawn_vehicle()
	mpc_planner = MPCPlanner(config.get('mpc', {}))
	return config,carla_interface, mpc_planner

def calculate_curve_curvature(x_coords, y_coords):
    """Calcular curvatura da trajetória"""
    if len(x_coords) < 5:
        return 0.0
    dx = np.gradient(x_coords)
    dy = np.gradient(y_coords)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5)
    denominator = np.where(denominator == 0, 1e-6, denominator)
    curvature = numerator / denominator
    # Use a média dos 2/3 centrais para evitar bordas ruidosas

    n = len(curvature)
    return np.mean(curvature[n//6:-n//6])


def main():
	
	# Add tracking for previous positions to detect when car is stuck
	last_positions = []
	max_position_history = 5
	is_in_recovery = False
	recovery_end_time = 0
	
	# Initialize Carla interface
	try:
		# Load configuration
		config, carla_interface, mpc_planner = initialize_system('config/carla_config.yaml')		
		# Impulso inicial
		carla_interface.apply_control(throttle=0.3, steer=0.0, brake=0.0)
		time.sleep(1.5)
		
		# Main control loop
		while True:
			# Get current vehicle state
			current_state = carla_interface.get_vehicle_state()
			if not current_state:
				print("No vehicle state available")
				time.sleep(0.1)
				continue
			# ADICIONE O TESTE DE CONTROLE MANUAL AQUI:
			# Teste de controle manual - aplicar ANTES do processamento de imagem
			if current_state['velocity'] < 0.1:  # Se o carro está parado
				print_debug_info("Carro parado, aplicando controle manual de teste")
				carla_interface.apply_control(throttle=0.4, steer=0.0, brake=0.0)
				time.sleep(0.5)  # Aguardar um pouco
				continue  # Pular o resto do loop e ir para próxima iteração
			
			# Monitor if vehicle is moving (seu código existente continua aqui...)
			print_debug_info(f"Vehicle state: Position ({current_state['x']:.1f}, {current_state['y']:.1f}), "
					  f"Velocity: {current_state['velocity']:.2f} m/s, Yaw: {current_state['yaw']:.2f}")
			
			# Obter imagem da câmera
			image = carla_interface.get_camera_image()
			lane_info = (0.0, 0.0)
			waypoints_from_lanes = False
			
			if image is not None:
				try:
					# Converter imagem CARLA para numpy array
					array = np.frombuffer(image.raw_data, dtype=np.uint8)
					array = array.reshape((image.height, image.width, 4))
					rgb_array = array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB (as in validatePolly)

					# === PREPROCESSAMENTO EXATAMENTE IGUAL AO validatePolly.py ===
					img_resized = cv2.resize(rgb_array, (detector.input_shape[1], detector.input_shape[0]))
					img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
					img_input = img_gray.astype(np.float32) / 255.0
					img_input = np.expand_dims(img_input, axis=-1)
					img_input = np.expand_dims(img_input, axis=0)
					pred = detector.model.predict(img_input, verbose=0)[0]
					mask = (pred.squeeze() > 0.5).astype(np.uint8) * 255
					binary_mask_resized = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)
					
					if binary_mask_resized is not None and np.sum(binary_mask_resized) > 0:
						# Usar polyfit para extrair trajetória
						result = extract_center_curve_with_lanes(binary_mask_resized)
						
						if result is not None:
							center_curve, lanes = result
							
							# TESTE ISOLADO COMPLETO DA DETECÇÃO DE FAIXAS
							print(f"[TEST] === ANÁLISE DA DETECÇÃO DE FAIXAS ===")
							print(f"[TEST] Número de faixas detectadas: {len(lanes)}")
							
							# Analisar cada faixa individual
							for i, lane in enumerate(lanes):
								if 'curve' in lane:
									x_coords, y_coords = lane['curve']
									print(f"[TEST] Faixa {i+1}: {len(x_coords)} pontos")
									print(f"[TEST] Faixa {i+1} X range: [{min(x_coords):.1f}, {max(x_coords):.1f}]")
									print(f"[TEST] Faixa {i+1} Y range: [{min(y_coords):.1f}, {max(y_coords):.1f}]")
							
							# Analisar trajetória central
							if center_curve is not None:
								center_x, center_y = center_curve
								
								# Calcular curvatura da trajetória
								curvature = calculate_curve_curvature(center_x, center_y)
								is_curve = abs(curvature) > 0.01
								
								if is_curve:
									print_debug_info(f"CURVA detectada - curvatura: {curvature:.3f}")
									# Parâmetros para curvas
									max_throttle = 0.25
									max_steer = 0.4
									target_speed = 2.0
								else:
									print_debug_info(f"RETA detectada - curvatura: {curvature:.3f}")
									# Parâmetros para retas
									max_throttle = 0.4
									max_steer = 0.3
									target_speed = 4.0
							else:
								# Valores padrão se não há trajetória
								max_throttle = 0.3
								max_steer = 0.3
								target_speed = 3.0
								is_curve = False
							
							# Extrair informações da faixa (seu código existente)
							lateral_offset, yaw_error = extract_lane_info(binary_mask_resized, center_curve)
							lane_info = (lateral_offset, yaw_error)
							
							# Converter para waypoints do mundo
							if center_curve is not None:
								# Converter coordenadas de imagem para coordenadas do mundo
								lane_waypoints = convert_image_points_to_world(center_curve, carla_interface)
								
								if len(lane_waypoints) > 5:
									waypoints = lane_waypoints
									waypoints_from_lanes = True
									print(f"[POLYFIT] Usando {len(waypoints)} waypoints da trajetória central")
									
									# Debug: mostrar os primeiros waypoints
									print_debug_info(f"Primeiros 3 waypoints: {waypoints[:3]}")
									print_debug_info(f"Posição atual: ({current_state['x']:.1f}, {current_state['y']:.1f})")
								else:
									# Se poucos waypoints, use waypoints simples à frente
									waypoints = [
										(current_state['x'] + 5, current_state['y']),
										(current_state['x'] + 10, current_state['y']),
										(current_state['x'] + 15, current_state['y'])
									]
								print_debug_info(f"Usando waypoints simples à frente: {waypoints}")
							else:
								waypoints = carla_interface.get_waypoints_ahead(distance=2.0, count=20)
								print_debug_info("Fallback: waypoints CARLA (sem trajetória)")
							
							# Criar visualização
							vis_surface = create_polyfit_visualization(binary_mask_resized, center_curve, lanes)
							if vis_surface:
								# Redimensionar para a tela
								scaled_surface = pygame.transform.scale(vis_surface, (800, 600))
								screen.blit(scaled_surface, (0, 0))
								pygame.display.flip()
						
						else:
							print_debug_info("Polyfit não retornou resultado válido")
							waypoints = carla_interface.get_waypoints_ahead(distance=2.0, count=20)
					else:
						print_debug_info("Máscara binária inválida")
						waypoints = carla_interface.get_waypoints_ahead(distance=2.0, count=20)
						
				except Exception as e:
					print(f"[ERROR] Erro no processamento da imagem: {e}")
					waypoints = carla_interface.get_waypoints_ahead(distance=2.0, count=20)
			else:
				waypoints = carla_interface.get_waypoints_ahead(distance=2.0, count=20)
			
			# Verificar se temos waypoints válidos
			if len(waypoints) == 0:
				print("WARNING: Nenhum waypoint disponível!")
				waypoints = [(current_state['x'] + 5, current_state['y'])]
			
			# Validar waypoints antes de usar no MPC
			if len(waypoints) > 0:
				# Calcular distâncias dos waypoints
				distances = []
				for wp in waypoints[:3]:  # Verificar os 3 primeiros
					dist = np.sqrt((wp[0] - current_state['x'])**2 + (wp[1] - current_state['y'])**2)
					distances.append(dist)
				
				avg_distance = np.mean(distances)
				print_debug_info(f"Distância média dos waypoints: {avg_distance:.2f}m")
				
				# Se waypoints muito distantes, usar waypoints simples à frente
				if avg_distance > 20:  # Se mais de 20m de distância
					print_debug_info("Waypoints muito distantes, usando waypoints simples")
					waypoints = [
						(current_state['x'] + 5, current_state['y']),
						(current_state['x'] + 10, current_state['y']),
						(current_state['x'] + 15, current_state['y'])
					]
			# Gerar comandos de controle
			try:
				# Após calcular o controle MPC
				control = mpc_planner.plan(current_state, waypoints, lane_info)
				
				# Debug do MPC
				print_debug_info(f"MPC output - Throttle: {control.get('throttle', 0):.3f}, Steer: {control.get('steer', 0):.3f}")
				
				# Aplicar controle adaptativo baseado na curvatura
				throttle_value = control.get('throttle', 0)
				
				# Forçar throttle mínimo adaptativo
				if abs(throttle_value) < 0.1:
					if 'is_curve' in locals() and is_curve:
						throttle_value = 0.2  # Throttle menor em curvas
						print_debug_info(f"Forçando throttle mínimo para CURVA: {throttle_value:.2f}")
					else:
						throttle_value = 0.3  # Throttle normal em retas
						print_debug_info(f"Forçando throttle mínimo para RETA: {throttle_value:.2f}")
				
				# Aplicar limites adaptativos
				if 'max_throttle' in locals():
					throttle_value = np.clip(throttle_value, 0.0, max_throttle)
					steer_value = np.clip(control.get('steer', 0), -max_steer, max_steer)
				else:
					# Fallback para valores padrão
					throttle_value = np.clip(throttle_value, 0.0, 0.3)
					steer_value = np.clip(control.get('steer', 0), -0.3, 0.3)
				
				print_debug_info(f"Controle adaptativo - Throttle: {throttle_value:.2f}, Steer: {steer_value:.2f}, Tipo: {'CURVA' if locals().get('is_curve', False) else 'RETA'}")

				carla_interface.apply_control(
					throttle=throttle_value,
					steer=steer_value,
					brake=0.0
				)
				
			except Exception as e:
				print(f"[ERROR] Erro no MPC: {e}")
				# Controle de emergência - parar o carro
				carla_interface.apply_control(throttle=0.0, steer=0.0, brake=0.5)
			
			# Update spectator camera
			carla_interface.update_spectator()
			
			# Improved position tracking and stuck detection
			if current_state:
				current_pos = (current_state['x'], current_state['y'])
				last_positions.append(current_pos)
				
				# Keep only recent positions
				if len(last_positions) > max_position_history:
					last_positions.pop(0)
				
				# Check if we need to enter recovery mode
				if not is_in_recovery and len(last_positions) >= 3:
					# Calculate total distance moved
					total_distance = 0
					for i in range(1, len(last_positions)):
						prev = last_positions[i-1]
						curr = last_positions[i]
						dist = ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)**0.5
						total_distance += dist
					
					# Calculate average movement per control cycle
					avg_movement = total_distance / (len(last_positions)-1)
					
					# If hardly moving despite applying throttle, consider stuck
					is_stuck = (avg_movement < 0.05 and 'throttle_value' in locals() and
							   throttle_value > 0.2 and current_state['velocity'] < 0.5)
					
					if is_stuck:
						print_debug_info(f"Car is stuck! Avg movement: {avg_movement:.4f}m, Speed: {current_state['velocity']:.2f}m/s")
						is_in_recovery = True
						recovery_end_time = time.time() + 2.0
						print_debug_info("Applying gentle reverse for 2 seconds...")
				
				# Handle recovery mode
				if is_in_recovery:
					if time.time() < recovery_end_time:
						# Progressive recovery
						progress = (time.time() - (recovery_end_time - 2.0)) / 2.0
						reverse_throttle = -0.4 * (1.0 - progress * 0.5)
						reverse_steer = -control['steer'] if 'control' in locals() else 0.0
						
						if progress > 0.7:
							reverse_steer *= (1.0 - (progress - 0.7) / 0.3)
						
						print_debug_info(f"Recovery - progress: {progress:.2f}, reverse: {reverse_throttle:.2f}, steer: {reverse_steer:.2f}")
						
						carla_interface.apply_control(
							throttle=reverse_throttle,
							steer=reverse_steer * 0.7,
							brake=0.0
						)
					else:
						is_in_recovery = False
						print_debug_info("Recovery completed, resuming normal control")
						last_positions = []
			
			# Sleep to maintain control frequency
			time.sleep(config.get('control_frequency', 0.1))
			
	except KeyboardInterrupt:
		print("Simulation interrupted by user")
	finally:
		# Clean up
		carla_interface.cleanup()
		pygame.quit()

if __name__ == "__main__":
	main()
