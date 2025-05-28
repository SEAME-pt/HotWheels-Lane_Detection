import numpy as np

def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K

def convert_image_points_to_world(center_curve, carla_interface):
    """Conversão corrigida de coordenadas para qualquer resolução de câmera"""
    if center_curve is None:
        return []
        
    vehicle = carla_interface.vehicle
    if vehicle is None:
        return []
    
    vehicle_transform = vehicle.get_transform()
    center_x, center_y = center_curve
    waypoints_world = []

    # Obter resolução da câmera da interface
    img_width = getattr(carla_interface, 'camera_width', 640)
    img_height = getattr(carla_interface, 'camera_height', 640)
    center_x_img = img_width // 2

    # Defina a escala correta (ajuste conforme necessário para seu setup)
    # Exemplo: 8 metros de visão vertical (de cima a baixo da imagem)
    real_height_m = 8.0
    escala_m_por_pixel = real_height_m / img_height

    for i in range(0, len(center_y), 30):  # Pegar apenas alguns pontos
        if len(waypoints_world) >= 10:
            break
            
        x_img, y_img = center_x[i], center_y[i]
        
        # Converter para coordenadas relativas ao veículo
        distance_ahead = (img_height - y_img) * escala_m_por_pixel
        lateral_offset = (x_img - center_x_img) * escala_m_por_pixel
        
        # Coordenadas no mundo
        yaw = vehicle_transform.rotation.yaw * np.pi / 180
        world_x = vehicle_transform.location.x + distance_ahead * np.cos(yaw) - lateral_offset * np.sin(yaw)
        world_y = vehicle_transform.location.y + distance_ahead * np.sin(yaw) + lateral_offset * np.cos(yaw)
        
        waypoints_world.append((world_x, world_y))
    
    return waypoints_world


