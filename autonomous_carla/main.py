import yaml
import time
import numpy as np
from carla_interface import CarlaInterface
from mpc.planner import MPCPlanner

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config('config/carla_config.yaml')
    
    # Initialize Carla interface
    carla_interface = CarlaInterface(config)
    
    try:
        # Connect to Carla
        carla_interface.connect()
        
        # Spawn vehicle
        carla_interface.spawn_vehicle()
        
        # Initialize MPC planner
        mpc_planner = MPCPlanner(config.get('mpc', {}))
        
        #! Substitua a definição estática:
        # Simple waypoints for testing (replace with lane detection output later)
        # waypoints = [(10, 0), (20, 0), (30, 0), (40, 10), (50, 20)]

        # Por uma chamada dinâmica:
        waypoints = carla_interface.get_waypoints_ahead(distance=5.0, count=20)
        
        # Main control loop
        while True:
            # Get current vehicle state
            current_state = carla_interface.get_vehicle_state()
            if not current_state:
                print("No vehicle state available")
                time.sleep(0.1)
                continue
            
            # Plan trajectory using MPC
            control = mpc_planner.plan(current_state, waypoints)
            
            # Apply control to vehicle
            carla_interface.apply_control(
                throttle=control['throttle'],
                steer=control['steer']
            )
            
            # Update spectator camera to follow the vehicle
            carla_interface.update_spectator()
            # Sleep to maintain control frequency
            time.sleep(config.get('control_frequency', 0.1))
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        # Clean up
        carla_interface.cleanup()

if __name__ == "__main__":
    main()
