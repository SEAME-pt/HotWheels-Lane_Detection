import numpy as np
from .optimizer import MPCOptimizer

class MPCPlanner:
    def __init__(self, config):
        self.config = config
        self.optimizer = MPCOptimizer(config)
        self.dt = config.get('dt', 0.1)  # Time step
        self.horizon = config.get('horizon', 10)  # Prediction horizon
        
    def plan(self, current_state, waypoints):
        """Generate optimal trajectory using MPC
        
        Args:
            current_state: Dictionary with current x, y, yaw, velocity
            waypoints: List of (x, y) coordinates to follow
            
        Returns:
            control: Dictionary with throttle, steer values
        """
        # Extract current state
        x0 = current_state['x']
        y0 = current_state['y']
        yaw0 = current_state['yaw']
        v0 = current_state['velocity']
        
        # Prepare reference trajectory from waypoints
        reference = self._prepare_reference(x0, y0, yaw0, waypoints)
        
        # Solve MPC optimization problem
        throttle, steer = self.optimizer.solve(x0, y0, yaw0, v0, reference)
        
        return {
            'throttle': throttle,
            'steer': steer
        }
    
    def _prepare_reference(self, x0, y0, yaw0, waypoints):
        """Convert global waypoints to local reference trajectory"""
        # Transform waypoints to vehicle's local coordinate frame
        local_points = []
        
        for wp_x, wp_y in waypoints:
            # Translate
            dx = wp_x - x0
            dy = wp_y - y0
            
            # Rotate
            local_x = dx * np.cos(-yaw0) - dy * np.sin(-yaw0)
            local_y = dx * np.sin(-yaw0) + dy * np.cos(-yaw0)
            
            local_points.append((local_x, local_y))
            
        return local_points[:self.horizon]
