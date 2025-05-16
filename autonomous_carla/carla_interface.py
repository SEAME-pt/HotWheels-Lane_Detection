import carla
import numpy as np
import time  # Ensure time module is imported
import random
import time

"""
    CarlaInterface class provides an interface to interact with the CARLA simulator.

    Methods:
        __init__(config): Initializes the CarlaInterface with the given configuration.
        connect(): Connects to the CARLA server.
        spawn_vehicle(): Spawns a vehicle and attaches sensors (camera and dummy sensor) to it.
        apply_control(throttle, steer, brake): Applies control inputs to the vehicle.
        get_vehicle_state(): Retrieves the current state of the vehicle.
        update_spectator(): Updates the spectator's view to follow the vehicle or dummy sensor.
        get_waypoints_ahead(distance, count): Gets waypoints ahead of the vehicle.
        cleanup(): Cleans up and destroys all actors in the simulation.
    """
class CarlaInterface:
    def __init__(self, config):
        """
        Initialize the CarlaInterface with the given configuration.

        Args:
            config: Configuration dictionary containing simulation parameters.
        """
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.last_image = None
        self.config = config
    
    # def _camera_callback(self, image):
    #         # Converta a imagem CARLA para numpy array (BGRA para RGB)
    #         array = np.frombuffer(image.raw_data, dtype=np.uint8)
    #         array = array.reshape((image.height, image.width, 4))  # BGRA
    #         rgb_array = array[:, :, :3][:, :, ::-1]  # BGR para RGB
    #         self.last_image = rgb_array

    def _camera_callback(self, image):
        self.last_image = image  # Armazene o objeto CARLA, não o numpy array

    def connect(self):
        """
        Connect to the CARLA server.

        Raises:
            RuntimeError: If the connection to the server fails.
        """
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print(f"Connected to Carla version: {self.client.get_server_version()}")
        
    def spawn_vehicle(self):
        """
        Spawn a vehicle in the CARLA world and attach sensors to it.

        Raises:
            RuntimeError: If the vehicle or sensors cannot be spawned.
        """
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Obtenha todos os spawn points válidos (apenas em faixas de direção)
        spawn_points = self.world.get_map().get_spawn_points()
        valid_spawn_points = []
        for sp in spawn_points:
            waypoint = self.world.get_map().get_waypoint(sp.location)
            if waypoint and waypoint.lane_type == carla.LaneType.Driving:
                valid_spawn_points.append(sp)
        
        # Escolha um spawn point aleatório entre os válidos
        if valid_spawn_points:
            spawn_point = random.choice(valid_spawn_points)
        else:
            spawn_point = carla.Transform()  # fallback
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")
        
        # Dar tempo para o veículo cair no chão e se estabilizar
        self.world.tick()
        time.sleep(1.0)
        
        # Aplicar um impulso físico para o carro começar a se mover - reduzido para um início mais suave
        impulse = carla.Vector3D(x=0.0, y=0.0, z=0.0)  # Impulso reduzido para movimento mais suave
        self.vehicle.add_impulse(impulse)
        print("Applied gentler initial impulse to vehicle to start movement")
        
        # Set up camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '100')
        
        # Adjust camera position for better visibility
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        if self.camera:
            self.camera.listen(self._camera_callback)
        else:
            print("Camera was not spawned correctly!")
        # Adicionar este trecho para criar o sensor dummy
        dummy_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        dummy_transform = carla.Transform(carla.Location(x=-4, z=2.5))
        self.dummy = self.world.spawn_actor(dummy_bp, dummy_transform, attach_to=self.vehicle)

    def get_camera_image(self):
        """
        Retorna a última imagem capturada pela câmera (numpy array RGB).
        """
        return self.last_image


    def apply_control(self, throttle, steer, brake=0.0):
        """
        Apply control inputs to the vehicle.

        Args:
            throttle: Throttle value (0.0 to 1.0) ou (-1.0 a 0.0) para ré.
            steer: Steering angle (-1.0 to 1.0).
            brake: Brake value (0.0 to 1.0, default is 0.0).
        """
        if self.vehicle:
            # Verifica se é pedido para ir em marcha ré
            reverse = throttle < 0
            
            # Certifique-se de que os valores estão nos limites corretos
            throttle_abs = min(1.0, abs(float(throttle)))
            steer = max(-1.0, min(1.0, float(steer)))
            brake = max(0.0, min(1.0, float(brake)))
            
            control = carla.VehicleControl(
                throttle=float(throttle_abs),
                steer=float(steer),
                brake=float(brake),
                hand_brake=False,  # Garantir que o freio de mão está desativado
                reverse=False,     # Garantir que o carro não está em marcha ré
                manual_gear_shift=False  # Deixe o CARLA gerenciar as marchas
            )
            
            if self.config.get('DEBUG', False):
                print(f"CARLA Control - throttle: {throttle_abs:.2f}, steer: {steer:.2f}, brake: {brake:.2f}, reverse: {reverse}")
            self.vehicle.apply_control(control)
            
    def get_vehicle_state(self):
        """
        Retrieve the current state of the vehicle.

        Returns:
            dict: A dictionary containing the vehicle's x, y, yaw, and velocity.
        """
        if not self.vehicle:
            return None
            
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        # Convert to more usable format
        x = transform.location.x
        y = transform.location.y
        yaw = transform.rotation.yaw * np.pi / 180.0  # Convert to radians
        
        # Calculate velocity magnitude
        v = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        if self.config.get('DEBUG', False):
            print(f"Velocity components: vx={velocity.x:.2f}, vy={velocity.y:.2f}, vz={velocity.z:.2f}, |v|={v:.2f}")
        return {
            'x': x,
            'y': y,
            'yaw': yaw,
            'velocity': v
        }
    
    def update_spectator(self):
        """
        Update the spectator's view to follow the vehicle with smooth interpolation.
        Na primeira chamada, posiciona exatamente no dummy para evitar teleporte.
        Aproxima a câmera do dummy ajustando a posição relativa.
        """
        if self.dummy and self.world:
            spectator = self.world.get_spectator()
            target_transform = self.dummy.get_transform()
            # Aproxima a câmera do dummy (ex: desloca para trás e para cima)
            offset = carla.Location(x=-2.0, z=1.5)  # Mais próximo e levemente acima
            target_location = target_transform.location + offset
            target_transform = carla.Transform(target_location, target_transform.rotation)
            if not hasattr(self, '_spectator_initialized'):
                spectator.set_transform(target_transform)
                self._spectator_initialized = True
                return
            current_transform = spectator.get_transform()
            lerp_factor = 0.05
            new_location = carla.Location(
                x=current_transform.location.x + (target_transform.location.x - current_transform.location.x) * lerp_factor,
                y=current_transform.location.y + (target_transform.location.y - current_transform.location.y) * lerp_factor,
                z=current_transform.location.z + (target_transform.location.z - current_transform.location.z) * lerp_factor
            )
            def lerp_angle(a, b, t):
                d = (b - a + 180) % 360 - 180
                return a + d * t
            new_rotation = carla.Rotation(
                pitch=current_transform.rotation.pitch + (target_transform.rotation.pitch - current_transform.rotation.pitch) * lerp_factor,
                yaw=lerp_angle(current_transform.rotation.yaw, target_transform.rotation.yaw, lerp_factor),
                roll=current_transform.rotation.roll + (target_transform.rotation.roll - current_transform.rotation.roll) * lerp_factor
            )
            spectator.set_transform(carla.Transform(new_location, new_rotation))

    def get_waypoints_ahead(self, distance=2.0, count=20):
        """
        Get waypoints ahead of the vehicle with improved spacing and curve handling.

        Args:
            distance: Base distance between waypoints in meters.
            count: Number of waypoints to generate.

        Returns:
            list: A list of (x, y) coordinates for the waypoints.
        """
        if not self.vehicle or not self.world:
            return []

        # Get current vehicle transform
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        
        # Get current velocity
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Adjust waypoint spacing based on speed
        # More distant for faster speeds, closer for slower speeds
        min_spacing = 1.0  # Minimum spacing
        speed_factor = 1.0 + (speed * 0.1)  # Increase spacing with speed
        
        # Obter waypoint atual
        current_waypoint = self.world.get_map().get_waypoint(vehicle_location)
        
        # Generate progressive distances (closer at start, further at end)
        # This helps with better path planning around curves
        base_distances = [min_spacing * speed_factor * (1.0 + 0.15 * i) for i in range(count)]
        
        # Apply a non-linear progression for better curve handling
        distances = []
        total_dist = 0
        for d in base_distances:
            total_dist += d
            distances.append(total_dist)
            
        waypoints_ahead = []
        waypoint = current_waypoint
        
        for dist in distances:
            # Get next waypoint at the current distance
            next_waypoints = waypoint.next(dist)
            if not next_waypoints:
                break
                
            waypoint = next_waypoints[0]
            # Add to our list of waypoints
            waypoints_ahead.append((waypoint.transform.location.x, waypoint.transform.location.y))
        
        print_debug = self.config.get('DEBUG', False)
        if print_debug:
            print(f"Generated {len(waypoints_ahead)} CARLA waypoints with speed-adaptive spacing")
            if len(waypoints_ahead) > 0:
                first_wp = waypoints_ahead[0]
                last_wp = waypoints_ahead[-1]
                first_dist = np.sqrt((first_wp[0] - vehicle_location.x)**2 + (first_wp[1] - vehicle_location.y)**2)
                total_path = np.sqrt((last_wp[0] - vehicle_location.x)**2 + (last_wp[1] - vehicle_location.y)**2)
                print(f"First waypoint: {first_dist:.1f}m, Total path: {total_path:.1f}m")
        return waypoints_ahead

    def cleanup(self):
        """
        Clean up and destroy all actors (vehicle and sensors) in the simulation.
        """
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
