import carla
import numpy as np

def spawn_traffic(client, num_vehicles=30):
    """Spawns traffic vehicles and enables autopilot."""
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    traffic_manager = client.get_trafficmanager(8000)  # Use Traffic Manager port
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)  # Maintain safe distance
    traffic_manager.set_random_device_seed(42)  # Set seed for randomness

    vehicles_list = []  # Store spawned vehicles

    for i in range(num_vehicles):
        vehicle_bp = np.random.choice(blueprint_library.filter("vehicle.*"))  # Random car
        spawn_point = np.random.choice(spawn_points)

        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            vehicle.set_autopilot(True, traffic_manager.get_port())  # Enable TM autopilot
            vehicles_list.append(vehicle)

    print(f"Spawned {len(vehicles_list)} traffic vehicles.")

    return vehicles_list  # Return list to manage cleanup

def spawn_pedestrians(client, num_pedestrians=20):
    """Spawns pedestrians and sets them walking."""
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    pedestrians_list = []
    walker_controllers = []

    for i in range(num_pedestrians):
        walker_bp = np.random.choice(blueprint_library.filter("walker.pedestrian.*"))
        spawn_point = np.random.choice(spawn_points)

        walker = world.try_spawn_actor(walker_bp, spawn_point)
        if walker:
            pedestrians_list.append(walker)

            # Spawn a walker controller
            walker_controller_bp = blueprint_library.find("controller.ai.walker")
            walker_controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
            walker_controllers.append(walker_controller)

            # Set random walking speed
            walker_controller.start()
            walker_controller.go_to_location(world.get_random_location_from_navigation())
            walker_controller.set_max_speed(1.5)  # Speed in m/s

    print(f"Spawned {len(pedestrians_list)} pedestrians.")

    return pedestrians_list, walker_controllers
