import carla
import numpy as np
import time
import random

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
        self.dummy = None
        self.last_image = None
        self.config = config
    
    # def _camera_callback(self, image):
    #         # Converta a imagem CARLA para numpy array (BGRA para RGB)
    #         array = np.frombuffer(image.raw_data, dtype=np.uint8)
    #         array = array.reshape((image.height, image.width, 4))  # BGRA
    #         rgb_array = array[:, :, :3][:, :, ::-1]  # BGR para RGB
    #         self.last_image = rgb_array

    def _camera_callback(self, image):
        """Store the CARLA image object."""
        self.last_image = image

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
        
        # Configure synchronous mode and map layers
        self._configure_world_settings()
        self._configure_map_layers()

    def _configure_world_settings(self):
        """Configure synchronous mode for stable operation."""
        if self.config.get('synchronous_mode', True):
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.config.get('fixed_delta_seconds', 0.05)
            self.world.apply_settings(settings)
            
            # Configure traffic manager
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            print("Synchronous mode configured")

    def _configure_map_layers(self):
        """Configure map layers for light version."""
        try:
            self.world.unload_map_layer(carla.MapLayer.All)
            self.world.load_map_layer(carla.MapLayer.Ground)
            
            layers_to_unload = [
                carla.MapLayer.Decals,
                carla.MapLayer.Props,
                carla.MapLayer.StreetLights,
                carla.MapLayer.Foliage,
                carla.MapLayer.ParkedVehicles,
                carla.MapLayer.Particles,
                carla.MapLayer.Walls,
            ]
            
            for layer in layers_to_unload:
                try:
                    self.world.unload_map_layer(layer)
                except Exception:
                    pass
            print("Map layers configured for light version")
        except Exception as e:
            print(f"Erro ao configurar camadas do mapa: {e}")

    def reset_world(self):
        """Reset world with better error handling."""
        print("Resetando mundo CARLA...")
        
        try:
            # Cleanup current actors first
            self.cleanup()
            
            # Wait a bit before reloading
            time.sleep(2.0)
            
            # Try to reload world
            self.world = self.client.reload_world()
            
            # Wait for world to stabilize
            for _ in range(5):
                try:
                    self.world.tick()
                    break
                except:
                    time.sleep(0.5)
            
            time.sleep(1.0)
            
            # Reconfigure world settings
            self._configure_world_settings()
            
            # Skip map layer configuration on reset to avoid segfault
            # Only configure on initial connect
            print("Mundo resetado com sucesso")
            
        except Exception as e:
            print(f"Erro durante reset: {e}")
            # Try to reconnect if reset fails
            try:
                self.connect()
            except Exception as e2:
                print(f"Erro na reconexão: {e2}")
                raise


    def spawn_vehicle(self, spawn_index=None):
        """Spawn a vehicle and attach sensors to it. Optionally choose a specific spawn index."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        
        # Get valid spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        valid_spawn_points = []
        
        for sp in spawn_points:
            waypoint = self.world.get_map().get_waypoint(sp.location)
            if waypoint and waypoint.lane_type == carla.LaneType.Driving:
                valid_spawn_points.append(sp)
        
        if valid_spawn_points:
            if spawn_index is not None and 0 <= spawn_index < len(valid_spawn_points):
                spawn_point = valid_spawn_points[spawn_index]
            else:
                spawn_point = random.choice(valid_spawn_points)
        else:
            spawn_point = carla.Transform()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        print(f"Vehicle spawned at {spawn_point.location}")
        
        # Stabilize vehicle
        self.world.tick()
        time.sleep(1.0)
        
        # Setup camera
        self._setup_camera()
        
        # Setup dummy sensor for spectator
        self._setup_dummy_sensor()

    def _setup_camera(self):
        """Setup RGB camera sensor."""
        self.camera_width = 640
        self.camera_height = 640
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.camera_width))
        camera_bp.set_attribute('image_size_y', str(self.camera_height))
        camera_bp.set_attribute('fov', '100')
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        
        if self.camera:
            self.camera.listen(self._camera_callback)
        else:
            print("Camera was not spawned correctly!")

    def _setup_dummy_sensor(self):
        """Setup dummy sensor for spectator following."""
        dummy_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        dummy_transform = carla.Transform(carla.Location(x=-4, z=2.5))
        self.dummy = self.world.spawn_actor(dummy_bp, dummy_transform, attach_to=self.vehicle)

    def get_camera_image_as_array(self):
        """
        Get camera image converted to numpy RGB array.
        Returns None if no image available.
        """
        if self.last_image is None:
            return None
            
        if hasattr(self.last_image, 'raw_data'):
            array = np.frombuffer(self.last_image.raw_data, dtype=np.uint8)
            array = array.reshape((self.last_image.height, self.last_image.width, 4))
            rgb_array = array[:, :, :3]  # BGRA to RGB
            return rgb_array
        return None

    def get_camera_image(self):
        """Returns the last image captured by the camera (CARLA image object)."""
        return self.last_image

    def tick_world(self):
        """Tick the world in synchronous mode."""
        if self.config.get('synchronous_mode', True):
            self.world.tick()

    def enable_autopilot(self, speed=10.0):
        """Enable autopilot for the vehicle."""
        if self.vehicle:
            self.vehicle.set_autopilot(True)
            tm = self.client.get_trafficmanager()
            tm.vehicle_percentage_speed_difference(self.vehicle, 100 - (speed / 30.0 * 100))
            print(f"Autopilot enabled. Speed limited to {speed} km/h.")

    def disable_autopilot(self):
        """Disable autopilot for the vehicle."""
        if self.vehicle:
            self.vehicle.set_autopilot(False)
            print("Autopilot disabled.")

    def update_spectator(self):
        """Update spectator view to follow the vehicle."""
        if self.dummy and self.world:
            spectator = self.world.get_spectator()
            target_transform = self.dummy.get_transform()
            
            offset = carla.Location(x=-2.0, z=1.5)
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

    def apply_control(self, throttle=0.0, steer=0.0, brake=0.0):
        """
        Apply control commands to the vehicle.
        
        Args:
            throttle (float): Throttle value between 0.0 and 1.0
            steer (float): Steering value between -1.0 and 1.0
            brake (float): Brake value between 0.0 and 1.0
        """
        if self.vehicle:
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            )
            self.vehicle.apply_control(control)
            
            # Tick world if in synchronous mode
            self.tick_world()
    def get_vehicle_state(self):
        """
        Get current vehicle state.
        
        Returns:
            dict: Vehicle state containing x, y, yaw, velocity
        """
        if not self.vehicle:
            return None
            
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        return {
            'x': transform.location.x,
            'y': transform.location.y,
            'yaw': np.radians(transform.rotation.yaw),  # Convert to radians
            'velocity': velocity_magnitude
        }
    def get_waypoints_ahead(self, distance=2.0, count=20):
        """
        Get waypoints ahead of the vehicle.
        
        Args:
            distance: Distance between waypoints in meters
            count: Number of waypoints to generate
            
        Returns:
            List of (x, y) coordinates
        """
        if not self.vehicle or not self.world:
            return []
            
        # Get current waypoint
        current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        
        # Get waypoints ahead
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
        """Enhanced cleanup to prevent sensor warnings."""
        actors_to_destroy = []

        # Stop camera listening before destroying
        if self.camera:
            try:
                self.camera.stop()  # Stop listening first
            except:
                pass
            actors_to_destroy.append(self.camera)
            self.camera = None

        if hasattr(self, 'dummy') and self.dummy:
            actors_to_destroy.append(self.dummy)
            self.dummy = None

        if self.vehicle:
            # Disable autopilot before destroying
            try:
                self.vehicle.set_autopilot(False)
            except:
                pass
            actors_to_destroy.append(self.vehicle)
            self.vehicle = None

        # Destroy all actors in batch
        if actors_to_destroy:
            try:
                self.client.apply_batch_sync([carla.command.DestroyActor(actor) for actor in actors_to_destroy])
            except:
                # Fallback to individual destruction
                for actor in actors_to_destroy:
                    try:
                        actor.destroy()
                    except:
                        pass
                    
        # Clear image buffer and reset flags
        self.last_image = None
        if hasattr(self, '_spectator_initialized'):
            delattr(self, '_spectator_initialized')

        # Force garbage collection
        import gc
        gc.collect()

        print("Cleanup complete.")
