import numpy as np

class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 64])
        self._car_vec = np.array([0, -1])
        
    def pure_pursuit_control(self, trajectory: np.ndarray, speed: float, lookahead_gain: float=0.3) -> float:
        """Computes the steering command using Pure Pursuit control.

        Args:
            trajectory (np.ndarray): A 2D array of (x, y) points representing the trajectory.
            speed (float): Current speed of the vehicle.
            lookahead_gain (float, optional): Gain factor to compute the lookahead distance. Defaults to 0.3.

        Returns:
            float: The computed steering value, constrained to the range [-1.0, 1.0].
        """

        # L_d = np.clip(lookahead_gain * (speed ** 1.2), 5, 35)
        # L_d = np.clip(lookahead_gain * np.log1p(speed), 5, 35)
        # L_d = np.clip(lookahead_gain * np.sqrt(speed), 5, 35)
        L_d = np.clip(lookahead_gain * speed, 5, 35) # calculate lookahead distance L_d
        
        # find the first trajectory point farther than L_d
        dists = np.linalg.norm(trajectory - self._car_position, axis=1)
        target_idx = np.where(dists > L_d)[0]
        
        if len(target_idx) == 0:
            target_point = trajectory[-1] # no target_idx, take last point of trajectory
        else:
            target_point = trajectory[target_idx[0]]

        # Transform target point into vehicle coordinates
        delta_x = target_point[0] - self._car_position[0] # lateral offset
        delta_y = target_point[1] - self._car_position[1] # longitudinal offset (optimal: always positive)
        if delta_y >= -0.01:
            return 0.0 # If target is behind or very close drive straight
        
        # Calculate kappa k = 2dx / L_d^2 and map to steering
        kappa = 2.0 * delta_x / (L_d ** 2)
        x = 10

        # Map steering between -1, 1 using tanh
        steering = np.tanh(kappa * x)
        
        return steering
    
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle between [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def stanley_controller(self, trajectory: np.ndarray, speed: float, k: float=4.0) -> float:
        """
        Stanley controller for lateral control.

        Args:
            vehicle_pos (np.ndarray): Vehicle position as np.array([x, y])
            vehicle_dir (np.ndarray): Vehicle direction as np.array([dx, dy])
            trajectory (np.ndarray): List of s (x, y)
            speed (float): Vehicle speed (v > 0)
            k (float): Gain factor for lateral error

        Returns:
            float: The computed steering value, constrained to the range [-1.0, 1.0].
        """
        
        # Find target point (closest point on the trajectory)
        traj_points = np.array(trajectory)
        distances = np.linalg.norm(traj_points - self._car_position, axis=1)
        closest_idx = np.argmin(distances)
        target_point = traj_points[closest_idx]

        # Calculate tangent angle at the target point
        if closest_idx < len(traj_points) - 1:
            next_point = traj_points[closest_idx + 1]
        else:
            next_point = traj_points[closest_idx]

        tangent_vec = next_point - target_point
        psi_path = np.arctan2(tangent_vec[1], tangent_vec[0])

        # Calculate vehicle angle
        psi_vehicle = -np.pi / 2

        # Calculate heading-error
        heading_error = self.normalize_angle(psi_path - psi_vehicle)

        # Calculate crosstrack error
        vec_to_traj = target_point - self._car_position
        crosstrack_error = np.cross(self._car_vec, vec_to_traj)  # # Dot product in 2D

        # Stanley formula
        steering_angle = heading_error + np.arctan2(k * crosstrack_error, speed + 1e-5)

        # Map steering between -1, 1 using tanh
        steering_output = np.tanh(steering_angle)
        return steering_output

    def control(self, trajectory, speed):
        """Controls the steering of the vehicle based on the given trajectory and speed.

        The function selects the appropriate steering control strategy based on the 
        current speed. If the speed is below 80, it uses the Stanley controller for 
        tight curves; otherwise, it uses the Pure Pursuit controller for higher speeds 
        or straighter paths.

        Args:
            trajectory (list): A list of waypoints representing the path to follow.
            speed (float): The current speed of the vehicle in some units (e.g., km/h or m/s).

        Returns:
            float: The calculated steering value:
                -1 for turning left
                1 for turning right
                0 for no steering required (straight ahead).
        """
        
        if len(trajectory) == 0:
            return 0
        
        trajectory = np.asarray(trajectory)
        # print(f"speed: {speed:.3f}")

        if speed < 15:
            return 0 # drive straight when accelerating from start position
        if speed < 75:
            steering = self.stanley_controller(trajectory, speed) # low speed -> for curves
        else:
            steering = self.pure_pursuit_control(trajectory, speed) # higher speed -> for straighter sections
            
        # print(f"steering: {steering}")
        return steering