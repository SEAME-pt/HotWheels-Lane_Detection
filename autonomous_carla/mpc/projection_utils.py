import numpy as np

def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K

def convert_image_points_to_world(center_curve, carla_interface):
    if center_curve is None:
        return []
    vehicle = carla_interface.vehicle
    if vehicle is None:
        return []
    vehicle_transform = vehicle.get_transform()
    center_x, center_y = center_curve
    waypoints_world = []

    # Detectar se é uma curva
    curvature = calculate_curve_curvature(center_x, center_y)
    is_curve = abs(curvature) > 0.05

    if is_curve:
        # Em curvas: usar mais pontos próximos e menos distantes
        step = max(1, len(center_y) // 15)  # 15 pontos para curvas
        max_distance = 8.0  # Máximo 8m à frente em curvas
    else:
        # Em retas: usar pontos mais espaçados
        step = max(1, len(center_y) // 10)  # 10 pontos para retas
        max_distance = 15.0  # Máximo 15m à frente em retas

    for i in range(0, len(center_y), step):
        if len(waypoints_world) >= (15 if is_curve else 10):
            break
        x_img, y_img = center_x[i], center_y[i]
        distance_ahead = (640 - y_img) * 0.05  # tente aumentar para 0.07 ou 0.1
        if distance_ahead > max_distance:
            continue
        lateral_offset = (x_img - 320) * 0.02  # 2cm por pixel
        yaw = vehicle_transform.rotation.yaw * np.pi / 180
        world_x = vehicle_transform.location.x + distance_ahead * np.cos(yaw) - lateral_offset * np.sin(yaw)
        world_y = vehicle_transform.location.y + distance_ahead * np.sin(yaw) + lateral_offset * np.cos(yaw)
        waypoints_world.append((world_x, world_y))
    return waypoints_world

def calculate_curve_curvature(x_coords, y_coords):
    """Calcular curvatura da trajetória"""
    if len(x_coords) < 3:
        return 0.0
    
    # Usar derivadas para calcular curvatura
    dx = np.gradient(x_coords)
    dy = np.gradient(y_coords)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvatura = |dx*ddy - dy*ddx| / (dx² + dy²)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.power(dx**2 + dy**2, 1.5)
    
    # Evitar divisão por zero
    denominator = np.where(denominator == 0, 1e-6, denominator)
    curvature = numerator / denominator
    
    return np.mean(curvature)


