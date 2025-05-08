import numpy as np
from scipy.optimize import minimize

class MPCOptimizer:
    def __init__(self, config):
        self.config = config
        self.dt = config.get('dt', 0.1)
        self.horizon = config.get('horizon', 10)
        self.wheelbase = config.get('wheelbase', 2.7)  # Vehicle wheelbase
        
        # Weights for the cost function
        self.w_cte = config.get('w_cte', 1.0)  # Cross-track error weight
        self.w_etheta = config.get('w_etheta', 1.0)  # Heading error weight
        self.w_vel = config.get('w_vel', 1.0)  # Velocity error weight
        self.w_steer = config.get('w_steer', 1.0)  # Steering input weight
        self.w_accel = config.get('w_accel', 1.0)  # Acceleration input weight
        self.w_steer_rate = config.get('w_steer_rate', 1.0)  # Steering rate weight
        
        # Control constraints
        self.max_steer = config.get('max_steer', 0.5)  # Maximum steering angle
        self.max_throttle = config.get('max_throttle', 1.0)  # Maximum throttle
        self.max_brake = config.get('max_brake', 1.0)  # Maximum brake
        
    def solve(self, x0, y0, yaw0, v0, reference):
        """Solve the MPC optimization problem
        
        Args:
            x0, y0: Initial position
            yaw0: Initial heading
            v0: Initial velocity
            reference: List of (x, y) waypoints in local coordinates
            
        Returns:
            throttle, steer: Optimal control inputs
        """
        # Initial state
        state = np.array([0.0, 0.0, 0.0, v0])  # Local x, y, yaw, velocity
        
        # Initial guess for controls (throttle, steer)
        u0 = np.zeros(2 * self.horizon)
        
        # Define bounds for controls
        bounds = []
        for _ in range(self.horizon):
            bounds.append((-self.max_throttle, self.max_throttle))  # Throttle bounds
            bounds.append((-self.max_steer, self.max_steer))  # Steering bounds
        
        # Solve optimization problem
        result = minimize(
            fun=lambda u: self._cost_function(u, state, reference),
            x0=u0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'disp': False}
        )
        
        # Extract first control action
        throttle = result.x[0]
        steer = result.x[1]
        
        return throttle, steer
    
    def _cost_function(self, u, state, reference):
        """Cost function for MPC optimization"""
        cost = 0.0
        x, y, yaw, v = state
        
        # Reshape control inputs
        controls = u.reshape(self.horizon, 2)
        
        for i in range(self.horizon):
            # Get reference for this step
            ref_x, ref_y = reference[min(i, len(reference)-1)]
            
            # Cross-track error
            cte = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)
            
            # Heading error (angle to reference point)
            ref_yaw = np.arctan2(ref_y - y, ref_x - x)
            etheta = self._normalize_angle(yaw - ref_yaw)
            
            # Add to cost
            cost += self.w_cte * cte**2
            cost += self.w_etheta * etheta**2
            
            # Control input costs
            throttle, steer = controls[i]
            cost += self.w_steer * steer**2
            cost += self.w_accel * throttle**2
            
            # Control rate costs (if not first step)
            if i > 0:
                prev_throttle, prev_steer = controls[i-1]
                cost += self.w_steer_rate * (steer - prev_steer)**2
            
            # Update state for next step using kinematic bicycle model
            x, y, yaw, v = self._update_state(x, y, yaw, v, throttle, steer)
        
        return cost
    
    def _update_state(self, x, y, yaw, v, throttle, steer):
        """Update state using kinematic bicycle model"""
        # Simple acceleration model
        v_next = v + throttle * self.dt
        
        # Ensure velocity is positive
        v_next = max(0.0, v_next)
        
        # Bicycle model
        beta = np.arctan(0.5 * np.tan(steer))
        x_next = x + v * np.cos(yaw + beta) * self.dt
        y_next = y + v * np.sin(yaw + beta) * self.dt
        yaw_next = yaw + v * np.sin(beta) / (self.wheelbase/2) * self.dt
        
        return x_next, y_next, yaw_next, v_next
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
