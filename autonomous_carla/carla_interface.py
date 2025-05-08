import carla
import numpy as np
import time

class CarlaInterface:
    def __init__(self, config):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.config = config
        
    def connect(self):
        """Connect to Carla server"""
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        print(f"Connected to Carla version: {self.client.get_server_version()}")
        
    def spawn_vehicle(self):
        """Spawn vehicle in the world"""
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
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        # Adicionar este trecho para criar o sensor dummy
        dummy_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        dummy_transform = carla.Transform(carla.Location(x=-4, z=2.5))
        self.dummy = self.world.spawn_actor(dummy_bp, dummy_transform, attach_to=self.vehicle)
        
    def apply_control(self, throttle, steer, brake=0.0):
        """Apply control to the vehicle"""
        if self.vehicle:
            control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake)
            )
            self.vehicle.apply_control(control)
            
    def get_vehicle_state(self):
        """Get current vehicle state"""
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
        if self.dummy and self.world:
            spectator = self.world.get_spectator()
            spectator.set_transform(self.dummy.get_transform())
    # def update_spectator(self):
    #     if self.vehicle and self.world:
    #         # Esperar pelo tick do simulador para sincronizar
    #         timestamp = self.world.wait_for_tick()

    #         # Obter a transformação do veículo
    #         vehicle_transform = self.vehicle.get_transform()

    #         # Calcular posição do espectador
    #         spectator = self.world.get_spectator()
    #         spectator_transform = carla.Transform(
    #             vehicle_transform.location + carla.Location(x=-4, z=2.5),
    #             vehicle_transform.rotation
    #         )
    #         spectator.set_transform(spectator_transform)

    def get_waypoints_ahead(self, distance=2.0, count=20):
        """Get waypoints ahead of the vehicle

        Args:
            distance: Distance between waypoints in meters
            count: Number of waypoints to generate

        Returns:
            List of (x, y) coordinates
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
        """Clean up actors"""
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
