"""
main.py

Este script é responsável por controlar a simulação no ambiente CARLA. Ele carrega as configurações, inicializa a interface com o CARLA, configura o planejador de controle preditivo (MPC) e executa o loop principal d					# Apply control to vehicle
					throttle_value = control['throttle'] * 0.7  # Apply a speed reduction factor
					
					# Aplicar um valor mínimo de throttle para garantir que o carro sempre se mova
					throttle_value = max(0.2, throttle_value)
					
					carla_interface.apply_control(
						throttle=throttle_value,
						steer=control['steer']
					)trole do veículo.

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
import carla  # Add carla import
from carla_interface import CarlaInterface
from process_image import KerasLaneDetector
from mpc.planner import MPCPlanner
from polinomial.ransacV1 import extract_center_curve_with_lanes
from mpc.projection_utils import convert_image_points_to_world

detector = KerasLaneDetector('models/lane_detector_combined_v2.keras')
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Lane Detection")

def extract_lane_info(mask):
	"""Extrai informações das faixas detectadas com detecção mais robusta.
	
	Args:
		mask: Máscara binária da segmentação de faixas
		
	Returns:
		tuple: (lateral_offset, yaw_error)
	"""
	if mask is None:
		return 0.0, 0.0
		
	# Assumindo que mask é uma imagem binária onde pixels brancos são faixas
	height, width = mask.shape[:2]
	
	# Use more of the image to improve lane detection
	roi_height = height // 2  # Use half of the image height
	
	# Divide the ROI into multiple horizontal segments for more robust detection
	segment_count = 4
	segment_height = roi_height // segment_count
	
	all_offsets = []
	segment_weights = [0.1, 0.2, 0.3, 0.4]  # Give more weight to lower segments (closer to car)
	
	# Analyze each horizontal segment
	for i in range(segment_count):
		y_start = height - roi_height + (i * segment_height)
		y_end = y_start + segment_height
		segment = mask[y_start:y_end, :]
		
		# Find lane points in this segment
		lane_points = np.where(segment > 0)
		
		if len(lane_points[1]) > 10:  # Ensure we have enough points
			# Calculate lane center in this segment
			lane_center_x = np.mean(lane_points[1])
			# Calculate normalized offset (-1 to 1)
			segment_offset = (lane_center_x - width/2) / (width/2)
			
			# Apply stronger weight to segments with more lane points
			confidence = min(1.0, len(lane_points[1]) / 200.0)  # Normalize by expected max points
			adjusted_weight = segment_weights[i] * (0.5 + 0.5 * confidence)
			
			all_offsets.append((segment_offset, adjusted_weight))
	
	# Calculate weighted average of offsets
	if all_offsets:
		lateral_offset = sum(offset * weight for offset, weight in all_offsets) / sum(weight for _, weight in all_offsets)
	else:
		lateral_offset = 0.0
	
	# Calculate orientation error using linear regression on lane points
	points_y = []
	points_x = []
	
	# Collect points from the entire ROI
	roi = mask[height-roi_height:height, :]
	lane_y, lane_x = np.where(roi > 0)
	
	if len(lane_x) > 20:  # Ensure we have enough points for regression (increased threshold)
		# Normalize coordinates
		lane_y = roi_height - lane_y  # Invert Y so higher values are farther from car
		
		# Simple linear regression
		if np.std(lane_x) > 0:  # Ensure we have horizontal variance
			try:
				# Use numpy's polynomial fit to get a quadratic fit for better curve handling
				# This helps better predict curved roads
				coeffs = np.polyfit(lane_y, lane_x, 2)
				
				# We'll use the first derivative at the bottom of the image as our yaw error
				# derivative of ax^2 + bx + c = 2ax + b
				a, b, _ = coeffs
				bottom_y = roi_height * 0.1  # 10% up from the bottom
				slope_at_bottom = 2 * a * bottom_y + b
				
				# Convert slope to angle in radians
				yaw_error = np.arctan(slope_at_bottom)
				
				# Apply additional sensitivity adjustment based on curve intensity
				# Stronger correction for sharper curves (when 'a' coefficient is larger)
				curve_intensity = min(1.0, abs(a) * 10.0)  # Normalized curve intensity
				yaw_sensitivity = 1.2 + (curve_intensity * 0.3)  # 1.2 to 1.5 range
				yaw_error = yaw_error * yaw_sensitivity
				
				print(f"Curve info: a={a:.4f}, intensity={curve_intensity:.2f}, sensitivity={yaw_sensitivity:.2f}")
			except Exception as e:
				print(f"Error in curve fitting: {e}")
				yaw_error = 0.0
		else:
			yaw_error = 0.0
	else:
		yaw_error = 0.0
	
	print(f"Enhanced lane info: lateral_offset={lateral_offset:.3f}, yaw_error={yaw_error:.3f}")
	return lateral_offset, yaw_error

def load_config(config_path):
	"""Load configuration from YAML file"""
	with open(config_path, 'r') as f:
		config = yaml.safe_load(f)
	return config

def main():
	# Load configuration
	config = load_config('config/carla_config.yaml')
	
	# Add debug mode
	debug_mode = True  # Set to True to print additional debug information
	
	# Add tracking for previous positions to detect when car is stuck
	last_positions = []
	max_position_history = 5  # Number of previous positions to keep
	is_in_recovery = False  # Flag to track if we're in recovery mode
	recovery_end_time = 0  # Time when recovery should end
	
	# Initialize Carla interface
	carla_interface = CarlaInterface(config)
	
	try:
		# Connect to Carla
		carla_interface.connect()
		
		# Spawn vehicle
		carla_interface.spawn_vehicle()
		
		# Inicializar o MPC planner
		mpc_planner = MPCPlanner(config.get('mpc', {}))
		
		# Aplicar um impulso inicial mais suave para garantir que o carro começa a se mover lentamente
		carla_interface.apply_control(throttle=0.3, steer=0.0, brake=0.0)
		time.sleep(1.5)  # Esperar um pouco mais para o carro começar a se mover suavemente
		
		# Add variables to track car state
		last_positions = []
		stuck_counter = 0
		max_stuck_count = 5
		
		#! Substitua a definição estática:
		# Simple waypoints for testing (replace with lane detection output later)
		# waypoints = [(10, 0), (20, 0), (30, 0), (40, 10), (50, 20)]
		# Por uma chamada dinâmica:
		# waypoints = carla_interface.get_waypoints_ahead(distance=2.0, count=20)
		# Main control loop
		while True:
			# Get current vehicle state
			current_state = carla_interface.get_vehicle_state()
			if not current_state:
				print("No vehicle state available")
				time.sleep(0.1)
				continue
			
			# Monitor if vehicle is moving
			if debug_mode:
				print(f"Vehicle state: Position ({current_state['x']:.1f}, {current_state['y']:.1f}), "
				      f"Velocity: {current_state['velocity']:.2f} m/s, Yaw: {current_state['yaw']:.2f}")
				
				# We already track positions in the recovery section below, just show info here
				if len(last_positions) >= 3:
					distances = []
					for i in range(1, len(last_positions)):
						prev = last_positions[i-1]
						curr = last_positions[i]
						dist = ((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)**0.5
						distances.append(dist)
					
					if distances:
						avg_movement = sum(distances) / len(distances)
						if avg_movement < 0.1:  # Less than 10cm movement on average
							print(f"Low movement detected: {avg_movement:.3f}m, Counter: {stuck_counter}/{max_stuck_count}")
							stuck_counter += 1
						else:
							stuck_counter = 0  # Reset counter if moving well
				
				# If stuck for several cycles, we'll rely on our recovery mode
				if stuck_counter >= max_stuck_count:
					print("Car may be stuck! Recovery will be handled in the dedicated recovery code.")
					# Reset counter
					stuck_counter = 0
		
			# Pegue a imagem já em numpy array RGB
			image = carla_interface.get_camera_image()

			lane_info = None

			if image is not None:
				# Converta a imagem CARLA para numpy array
				array = np.frombuffer(image.raw_data, dtype=np.uint8)
				array = array.reshape((image.height, image.width, 4))  # BGRA
				rgb_array = array[:, :, :3][:, :, ::-1]  # BGR para RGB

				# Passe diretamente para o detector
				mask = detector.process_image(rgb_array)

				if mask is not None:
					# Converter para formato adequado para processamento
					binary_mask = detector.predict(detector.preprocess_image(rgb_array))
					if binary_mask is not None:
						# Criar uma superfície com canal alpha
						vis_surface = pygame.Surface((binary_mask.shape[1], binary_mask.shape[0]), pygame.SRCALPHA)

						# Desenhar a máscara binária como fundo
						binary_rgb = np.stack([binary_mask, binary_mask, binary_mask], axis=2)
						# Vamos garantir que a orientação esteja correta para o Pygame
						binary_surface = pygame.surfarray.make_surface(binary_rgb.transpose(1, 0, 2))
						vis_surface.blit(binary_surface, (0, 0))
	
						# Extrair as curvas de faixa usando a função extract_center_curve modificada
						# que retorna tanto center_curve quanto spline_curves
						result = extract_center_curve_with_lanes(binary_mask)
						if result is None:
							# Se não tivermos curvas, continuamos sem desenhar
							pass
						else:
							center_curve, spline_curves = result
	
							# Desenhar as faixas detectadas
							colors = [(255, 0, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255)]  # vermelho, amarelo, azul, roxo

							for i, (_, curve_x, curve_y, *_) in enumerate(spline_curves):
								color = colors[i % len(colors)]
								# Converter para inteiros para desenho
								points = [(int(x), int(y)) for x, y in zip(curve_x, curve_y)]

								# Desenhar linha com pygame
								if len(points) > 1:
									pygame.draw.lines(vis_surface, color, False, points, 2)

							# Desenhar a trajetória central em verde
							if center_curve is not None:
								center_x, center_y = center_curve
								center_points = [(int(x), int(y)) for x, y in zip(center_x, center_y)]
								
								if len(center_points) > 1:
									pygame.draw.lines(vis_surface, (0, 255, 0), False, center_points, 3)

						# Exibir a superfície
						screen.blit(vis_surface, (0, 0))
						pygame.display.flip()

					# Extrair desvio lateral e erro de orientação
					lateral_offset, yaw_error = extract_lane_info(binary_mask)
					lane_info = (lateral_offset, yaw_error)
					
					# Enhanced waypoint generation - Track if we successfully detected lanes
					waypoints_from_lanes = False
					
					# Try to use lane-based waypoints first
					if 'result' in locals() and result is not None:
						if 'center_curve' in locals() and center_curve is not None:
							# Converter coordenadas de imagem para coordenadas do mundo
							lane_waypoints = convert_image_points_to_world(center_curve, carla_interface)
							if len(lane_waypoints) > 5:  # Verificar se há waypoints suficientes
								waypoints = lane_waypoints
								waypoints_from_lanes = True
								print(f"Using lane-based waypoints: {len(waypoints)} points")
					
					# Fallback to CARLA waypoints if lane detection failed
					if not waypoints_from_lanes:
						waypoints = carla_interface.get_waypoints_ahead(distance=2.0, count=20)
						print("Using CARLA waypoints (lane detection failed)")
					
					# Debug: print lane information
					print(f"Lane info - offset: {lateral_offset:.3f}, yaw_error: {yaw_error:.3f}, " 
						f"using lane waypoints: {waypoints_from_lanes}")
					
					# Generate control commands
					control = mpc_planner.plan(current_state, waypoints, lane_info)
					
					# Aplicamos a visualização apenas uma vez na parte superior do código
					# Não precisamos chamar novamente aqui
					# Isso evita o efeito de piscar entre visualizações diferentes
					
					# Apply control to vehicle
					# Progressive throttle based on lane detection confidence
					if waypoints_from_lanes:
						# If we have strong lane detection, we can drive with more confidence
						base_throttle = 0.4  # Reduced throttle for slower driving
					else:
						# More cautious when using CARLA waypoints
						base_throttle = 0.35  # Even slower if not confident
					
					# Adjust throttle based on steering - reduce speed in sharper turns
					steer_factor = 1.0 - (abs(control['steer']) * 0.4)  # Reduce throttle more in turns
					
					# Apply a curve-based speed reduction based on lane detection
					curve_speed_factor = 1.0
					if lane_info is not None and abs(lane_info[1]) > 0.2:  # Using yaw_error as indicator of curve
						# Reduce speed more in sharper curves 
						curve_speed_factor = max(0.6, 1.0 - (abs(lane_info[1]) * 0.5))
						print(f"Curve speed factor based on yaw: {curve_speed_factor:.2f}")
					
					# Combine all throttle factors
					throttle_value = base_throttle * steer_factor * curve_speed_factor
					
					# Ensure minimum throttle but keep it low for gentle movement
					throttle_value = max(0.25, throttle_value) 
					
					# Apply speed limiting at higher velocities for safety
					if current_state['velocity'] > 8.0:  # Lower threshold for speed limiting
						# Gradually reduce throttle as speed increases beyond 8 m/s
						speed_limit_factor = max(0.3, 1.0 - ((current_state['velocity'] - 8.0) / 10.0))
						throttle_value *= speed_limit_factor
						print(f"Speed limiting factor: {speed_limit_factor:.2f}")
					
					print(f"Applying throttle: {throttle_value:.2f}, steer: {control['steer']:.2f}")
					
					# Only apply control if not in recovery mode
					if not is_in_recovery:
						carla_interface.apply_control(
							throttle=throttle_value,
							steer=control['steer']
						)
			
			# Update spectator camera to follow the vehicle
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
						print(f"Car is stuck or crashed! Avg movement: {avg_movement:.4f}m, "
							  f"Speed: {current_state['velocity']:.2f}m/s")
						is_in_recovery = True
						recovery_end_time = time.time() + 2.0  # 2 seconds of reverse
						print("Applying gentle reverse for 2 seconds...")
				
				# Handle recovery mode
				if is_in_recovery:
					if time.time() < recovery_end_time:
						# Progressive recovery - starting with gentle reverse then gradually straightening
						progress = (time.time() - (recovery_end_time - 2.0)) / 2.0  # 0 to 1 over 2 seconds
						
						# Start with higher reverse throttle, then gradually reduce
						reverse_throttle = -0.4 * (1.0 - progress * 0.5)
						
						# Start with opposite steering, then gradually straighten
						reverse_steer = -control['steer'] if 'control' in locals() else 0.0
						if progress > 0.7:  # In the last 30% of recovery, straighten the wheels
							reverse_steer *= (1.0 - (progress - 0.7) / 0.3)
						
						print(f"Recovery - progress: {progress:.2f}, reverse: {reverse_throttle:.2f}, steer: {reverse_steer:.2f}")
						
						carla_interface.apply_control(
							throttle=reverse_throttle,
							steer=reverse_steer * 0.7,  # Reduced intensity for smoother reverse
							brake=0.0
						)
					else:
						# Exit recovery mode
						is_in_recovery = False
						print("Recovery completed, resuming normal control")
						last_positions = []  # Reset position history
			
			# Sleep to maintain control frequency
			time.sleep(config.get('control_frequency', 0.1))
			
	except KeyboardInterrupt:
		print("Simulation interrupted by user")
	finally:
		# Clean up
		carla_interface.cleanup()

if __name__ == "__main__":
	main()
