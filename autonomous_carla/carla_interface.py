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
        
        # Find a suitable spawn point - escolher um ponto de spawn na estrada
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        
        # Tentar alguns pontos de spawn diferentes se o primeiro não funcionar
        for sp in spawn_points[:5]:
            # Verificar se o spawn point está em uma estrada
            waypoint = self.world.get_map().get_waypoint(sp.location)
            if waypoint and waypoint.lane_type == carla.LaneType.Driving:
                spawn_point = sp
                break
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")
        
        # Dar tempo para o veículo cair no chão e se estabilizar
        self.world.tick()
        time.sleep(1.0)
        
        # Aplicar um impulso físico para o carro começar a se mover - reduzido para um início mais suave
        impulse = carla.Vector3D(x=2000.0, y=0.0, z=0.0)  # Impulso reduzido para movimento mais suave
        self.vehicle.add_impulse(impulse)
        print("Applied gentler initial impulse to vehicle to start movement")
        
        # Set up camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '100')
        
        # Adjust camera position for better visibility
        camera_transform = carla.Transform(carla.Location(x=1.8, z=1.7), carla.Rotation(pitch=-15))
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
            
            # Create control with correct parameter types, ensuring all are native Python types
            control = carla.VehicleControl()
            control.throttle = float(throttle_abs)
            control.steer = float(steer)
            control.brake = float(brake)
            control.hand_brake = False
            control.reverse = bool(reverse)
            control.manual_gear_shift = False
            
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
        
        # Print velocity components for debugging
        print(f"Velocity components: vx={velocity.x:.2f}, vy={velocity.y:.2f}, vz={velocity.z:.2f}, |v|={v:.2f}")
        
        return {
            'x': x,
            'y': y,
            'yaw': yaw,
            'velocity': v
        }
    
    def update_spectator(self):
        """
        Update the spectator's view to follow the vehicle with improved camera settings.
        Uses smooth camera following with interpolation based on vehicle speed.
        """
        if self.vehicle and self.world:
            # Get vehicle transform and velocity
            vehicle_transform = self.vehicle.get_transform()
            vehicle_velocity = self.vehicle.get_velocity()
            velocity_magnitude = np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
            
            # Adjust camera distance based on vehicle speed (closer when slow, further when fast)
            base_distance_x = -6.0  # Base distance behind the vehicle
            base_height_z = 4.0     # Base height above the vehicle
            
            # Scale distance slightly with speed, but maintain minimums for visibility
            distance_x = min(base_distance_x - (velocity_magnitude * 0.5), -3.0)  
            height_z = max(base_height_z + (velocity_magnitude * 0.2), 3.0)
            
            # Create a transform for the spectator that's behind and above the vehicle
            spectator_transform = carla.Transform(
                location=vehicle_transform.location + carla.Location(x=distance_x, z=height_z),
                rotation=carla.Rotation(pitch=-15, yaw=vehicle_transform.rotation.yaw)
            )
            
            # Apply the transform to the spectator
            spectator = self.world.get_spectator()
            spectator.set_transform(spectator_transform)

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
