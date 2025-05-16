import pygame
import cv2
import yaml
import time
import numpy as np
import carla
from carla_interface import CarlaInterface
from process_image import KerasLaneDetector
from mpc.planner import MPCPlanner
from mpc.projection_utils import convert_image_points_to_world

# Remova a importação do polinomial
# from polinomial.ransacV1 import extract_center_curve_with_lanes

detector = KerasLaneDetector('models/lane_detector_combined_v2.keras')

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Lane Detection")

# Implementação alternativa para extract_center_curve_with_lanes
def extract_center_curve_with_lanes(binary_mask):
    """
    Versão aprimorada: extrai o centro da faixa considerando as faixas esquerda e direita.
    Se só uma faixa for detectada, mantém o centro no meio da imagem para evitar enviesamento.
    """
    if binary_mask is None:
        return None
    
    height, width = binary_mask.shape[:2]
    num_segments = 10
    segment_height = height // num_segments
    center_x = []
    center_y = []
    for i in range(num_segments):
        y_pos = height - (i + 0.5) * segment_height
        y_start = max(0, int(y_pos - segment_height // 2))
        y_end = min(height, int(y_pos + segment_height // 2))
        segment = binary_mask[y_start:y_end, :]
        lane_points = np.where(segment > 0)
        if len(lane_points[1]) > 10:
            min_x = np.min(lane_points[1])
            max_x = np.max(lane_points[1])
            # Se as faixas estão bem separadas, use o centro entre elas
            if (max_x - min_x) > width * 0.3:
                x_center = (min_x + max_x) / 2.0
            else:
                # Só uma faixa visível, mantenha o centro no meio da imagem
                x_center = width / 2.0
            center_x.append(x_center)
            center_y.append(y_pos)
    if len(center_x) > 3:
        spline_curves = []
        return (np.array(center_x), np.array(center_y)), spline_curves
    return None

def extract_lane_info(mask):
    """Extrai informações das faixas detectadas com detecção mais robusta.
    Args:
        mask: Máscara binária da segmentação de faixas
    Returns:
        tuple: (lateral_offset, yaw_error)
    """
    if mask is None:
        return 0.0, 0.0
    
    height, width = mask.shape[:2]
    roi_height = int(height * 0.6)  # Usar 60% da imagem
    
    # Dividir em mais segmentos para análise mais detalhada
    segment_count = 6
    segment_height = roi_height // segment_count
    
    # Dar mais peso aos segmentos mais próximos do veículo
    segment_weights = [0.05, 0.1, 0.15, 0.2, 0.25, 0.25]
    
    all_offsets = []
    
    # Analyze each horizontal segment
    for i in range(segment_count):
        y_start = height - roi_height + (i * segment_height)
        y_end = y_start + segment_height
        segment = mask[y_start:y_end, :]
        lane_points = np.where(segment > 0)
        if len(lane_points[1]) > 10:
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
    roi = mask[height-roi_height:height, :]
    lane_y, lane_x = np.where(roi > 0)
    
    if len(lane_x) > 20:
        lane_y = roi_height - lane_y  # Invert Y so higher values are farther from car
        if np.std(lane_x) > 0:
            try:
                coeffs = np.polyfit(lane_y, lane_x, 2)
                a, b, _ = coeffs
                bottom_y = roi_height * 0.1
                slope_at_bottom = 2 * a * bottom_y + b
                yaw_error = np.arctan(slope_at_bottom)
                curve_intensity = min(1.0, abs(a) * 10.0)
                yaw_sensitivity = 1.2 + (curve_intensity * 0.3)
                # NOVO: Se a curvatura for muito pequena, forçar yaw_error a zero
                if abs(a) < 0.0005:
                    yaw_error = 0.0
                else:
                    yaw_error = yaw_error * yaw_sensitivity
                # Filtro extra para ruído em retas
                if abs(yaw_error) < 0.04:
                    yaw_error = 0.0
                # print(f"Curve info: a={a:.4f}, intensity={curve_intensity:.2f}, sensitivity={yaw_sensitivity:.2f}")
            except Exception as e:
                # print(f"Error in curve fitting: {e}")
                yaw_error = 0.0
        else:
            yaw_error = 0.0
    else:
        yaw_error = 0.0
    # print(f"Enhanced lane info: lateral_offset={lateral_offset:.3f}, yaw_error={yaw_error:.3f}")
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
    debug_mode = False  # DEBUG desativado
    
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
                        if debug_mode:
                            print(f"Low movement detected: {avg_movement:.3f}m, Counter: {stuck_counter}/{max_stuck_count}")
                        stuck_counter += 1
                    else:
                        stuck_counter = 0  # Reset counter if moving well
                
                # If stuck for several cycles, we'll rely on our recovery mode
                if stuck_counter >= max_stuck_count:
                    if debug_mode:
                        print("Car may be stuck! Recovery will be handled in the dedicated recovery code.")
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
            
            # Fallback to CARLA waypoints if lane detection failed
            if not waypoints_from_lanes:
                waypoints = carla_interface.get_waypoints_ahead(distance=2.0, count=20)
                if debug_mode:
                    print("Using CARLA waypoints (lane detection failed)")
            
            # Debug: print lane information
            if lane_info and debug_mode:
                print(f"Lane info - offset: {lane_info[0]:.3f}, yaw_error: {lane_info[1]:.3f}, "
                      f"using lane waypoints: {waypoints_from_lanes}")
            
            # Generate control commands
            control = mpc_planner.plan(current_state, waypoints, lane_info)
            
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
            
            if lane_info is not None and abs(lane_info[1]) > 0.1:  # Using yaw_error as indicator of curve
                # Reduzir velocidade em curvas mais acentuadas
                curve_factor = abs(lane_info[1])
                curve_speed_factor = max(0.4, 1.0 - (curve_factor * 1.2))
            
            # Combine all throttle factors
            throttle_value = base_throttle * steer_factor * curve_speed_factor
            
            # Ensure minimum throttle but keep it low for gentle movement
            throttle_value = max(0.25, throttle_value)
            
            # Apply speed limiting at higher velocities for safety
            if current_state['velocity'] > 8.0:  # Lower threshold for speed limiting
                # Gradually reduce throttle as speed increases beyond 8 m/s
                speed_limit_factor = max(0.3, 1.0 - ((current_state['velocity'] - 8.0) / 10.0))
                throttle_value *= speed_limit_factor
                if debug_mode:
                    print(f"Speed limiting factor: {speed_limit_factor:.2f}")
            
            if debug_mode:
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
                    print(f"Car is stuck! Avg movement: {avg_movement:.4f}m, Speed: {current_state['velocity']:.2f}m/s")
                    is_in_recovery = True
                    recovery_end_time = time.time() + 2.0
                    if debug_mode:
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
                    
                    if debug_mode:
                        print(f"Recovery - progress: {progress:.2f}, reverse: {reverse_throttle:.2f}, steer: {reverse_steer:.2f}")
                    
                    carla_interface.apply_control(
                        throttle=reverse_throttle,
                        steer=reverse_steer * 0.7,
                        brake=0.0
                    )
                else:
                    is_in_recovery = False
                    print("Recovery completed, resuming normal control")
                    last_positions = []
    
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    
    finally:
        # Clean up
        carla_interface.cleanup()

if __name__ == "__main__":
    main()
