import numpy as np
import carla

def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K

def convert_image_points_to_world(center_curve, carla_interface):
    # --- POLYNOMIAL PROJECTION, SEM API CARLA --- (IMPROVED: always start from vehicle center)
    if center_curve is None:
        return []
    vehicle = carla_interface.vehicle
    if vehicle is None:
        return []
    vehicle_transform = vehicle.get_transform()
    center_x, center_y = center_curve
    waypoints_world = []
    img_width = getattr(carla_interface, 'camera_width', 640)
    img_height = getattr(carla_interface, 'camera_height', 640)
    center_x_img = img_width // 2
    real_height_m = 8.0
    escala_m_por_pixel = real_height_m / img_height

    # Always start from the bottom of the image (vehicle center)
    # Find the point in center_y closest to the bottom (max y)
    if len(center_y) == 0:
        return []
    start_idx = np.argmax(center_y)  # y increases downward
    # Sample N points from start_idx upwards (towards the horizon)
    N = 10
    indices = np.linspace(start_idx, 0, N, dtype=int)
    for i in indices:
        x_img, y_img = center_x[i], center_y[i]
        distance_ahead = (img_height - y_img) * escala_m_por_pixel
        lateral_offset = (center_x_img - x_img) * escala_m_por_pixel
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


