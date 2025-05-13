import carla
import numpy as np
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
        
        # Find a suitable spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")
        
        # Set up camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '100')
        
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
            throttle: Throttle value (0.0 to 1.0).
            steer: Steering angle (-1.0 to 1.0).
            brake: Brake value (0.0 to 1.0, default is 0.0).
        """
        if self.vehicle:
            control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake)
            )
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
        v = np.sqrt(velocity.x**2 + velocity.y**2)
        
        return {
            'x': x,
            'y': y,
            'yaw': yaw,
            'velocity': v
        }
    
    def update_spectator(self):
        """
        Update the spectator's view to follow the vehicle or dummy sensor.
        """
        if self.dummy and self.world:
            spectator = self.world.get_spectator()
            spectator.set_transform(self.dummy.get_transform())

    def get_waypoints_ahead(self, distance=2.0, count=20):
        """
        Get waypoints ahead of the vehicle.

        Args:
            distance: Distance between waypoints in meters.
            count: Number of waypoints to generate.

        Returns:
            list: A list of (x, y) coordinates for the waypoints.
        """
        if not self.vehicle or not self.world:
            return []

        # Obter waypoint atual
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())

        # Obter waypoints à frente
        waypoints_ahead = []
        waypoint = current_waypoint

        for i in range(count):
            next_waypoints = waypoint.next(distance)
            if not next_waypoints:
                break
            waypoint = next_waypoints[0]
            waypoints_ahead.append((waypoint.transform.location.x, waypoint.transform.location.y))

        return waypoints_ahead

    def cleanup(self):
        """
        Clean up and destroy all actors (vehicle and sensors) in the simulation.
        """
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
