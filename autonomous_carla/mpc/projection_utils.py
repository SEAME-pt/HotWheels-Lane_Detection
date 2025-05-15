import numpy as np

def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K

def convert_image_points_to_world(center_curve, carla_interface):
    """
    Converte pontos da curva central (em pixels da imagem) para coordenadas do mundo CARLA.
    center_curve: (center_x, center_y) arrays da imagem
    carla_interface: instância para acessar a câmera e o veículo
    """
    if center_curve is None:
        return []
        
    camera = carla_interface.camera
    vehicle = carla_interface.vehicle
    
    if camera is None or vehicle is None:
        print("Camera or vehicle not available for coordinate conversion")
        return []
    
    try:
        # Parâmetros da câmera
        image_w = int(camera.attributes["image_size_x"])
        image_h = int(camera.attributes["image_size_y"])
        fov = float(camera.attributes["fov"])
        
        # Matriz de projeção intrínseca
        K = build_projection_matrix(image_w, image_h, fov)
        
        # Extrínseca: matriz de transformação da câmera para o mundo
        camera_transform = camera.get_transform()
        world_2_camera = np.array(camera_transform.get_inverse_matrix())
        
        # Para cada ponto da curva central
        waypoints_world = []
        center_x, center_y = center_curve
        
        # Garantir que temos pontos suficientes e estamos processando apenas uma amostra dos pontos
        # para evitar waypoints muito próximos
        step = max(1, len(center_x) // 10)  # Pegar aproximadamente 10 pontos
        
        for i in range(0, len(center_x), step):
            x_img = center_x[i]
            y_img = center_y[i]
            
            # Coordenada homogênea do pixel
            pixel = np.array([x_img, y_img, 1.0])
            
            # Escolha uma profundidade Z (em metros) variável 
            # Pontos mais distantes na imagem (menor y_img) terão maior profundidade
            # Normalizar y_img para estar entre 0 e 1 (invertido)
            y_norm = 1.0 - (y_img / image_h)
            # Distribuição não linear para profundidade (mais pontos próximos)
            # Escalar para profundidade entre 3m e 25m
            Z = 3.0 + (y_norm * y_norm) * 22.0
            
            # Projeção reversa para coordenadas de câmera
            K_inv = np.linalg.inv(K)
            point_cam = K_inv @ (pixel * Z)
            point_cam = np.array([point_cam[0], point_cam[1], Z, 1.0])
            
            # Transforma para o mundo
            cam_2_world = np.linalg.inv(world_2_camera)
            point_world = cam_2_world @ point_cam
            
            # Adiciona como waypoint (x, y)
            waypoints_world.append((point_world[0], point_world[1]))
        
        # Adicionar debug para verificar os waypoints
        if len(waypoints_world) > 0:
            print(f"DEBUG - Primeiro waypoint: {waypoints_world[0]}, Último waypoint: {waypoints_world[-1]}")
        
        return waypoints_world
    except Exception as e:
        print(f"Error converting image points to world coordinates: {e}")
        return []
