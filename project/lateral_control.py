import numpy as np

class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 64])
        self._car_vec = np.array([0, -1])
        
    def pure_pursuit_control(self, trajectory, speed, lookahead_gain=0.2, max_steering=1.0):
        """Computes the steering command using Pure Pursuit control.

        Args:
            trajectory (np.ndarray): A 2D array of (x, y) points representing the trajectory.
            speed (float): Current speed of the vehicle.
            lookahead_gain (float, optional): Gain factor to compute the lookahead distance. Defaults to 0.2.
            max_steering (float, optional): Maximum allowed steering value. Defaults to 1.0.

        Returns:
            float: The computed steering value, constrained to the range [-1.0, 1.0].
        """

        lookahead_gain = 0.3
        # L_d = np.clip(lookahead_gain * (speed ** 1.2), 5, 35)
        # L_d = np.clip(lookahead_gain * np.log1p(speed), 5, 35)
        # L_d = np.clip(lookahead_gain * np.sqrt(speed), 5, 35)
        L_d = np.clip(lookahead_gain * speed, 5, 35)
        
        # find the first trajectory point farther than L_d
        dists = np.linalg.norm(trajectory - self._car_position, axis=1)
        target_idx = np.where(dists > L_d)[0]
        
        if len(target_idx) == 0:
            target_point = trajectory[-1] # no target_idx, take last point of trajectory
        else:
            target_point = trajectory[target_idx[0]]

        # Transform target point into vehicle coordinates
        delta_x = target_point[0] - self._car_position[0] # lateral offset
        delta_y = target_point[1] - self._car_position[1] # longitudal offset (optimal: always positive)
        if delta_y >= -0.01:
            return 0.0 # , target_point # If target is behind or very close drive straight
        
        # Calculate kappa k = 2dx / L_d^2 and map to steering
        kappa = 2.0 * delta_x / (L_d ** 2)
        x = 10
        # steering = np.clip(kappa * x, -max_steering, max_steering)
        steering = np.tanh(kappa * x)
        
        return steering
    
    def normalize_angle(self, angle):
        """Normalize angle between [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def stanley_controller(self, trajectory, velocity, k=4.0):
        """
        Stanley controller for lateral control.

        Args:
            vehicle_pos: Vehicle position as np.array([x, y])
            vehicle_dir: Vehicle direction as np.array([dx, dy])
            trajectory: List of points (x, y)
            velocity: Vehicle speed (v > 0)
            k: Gain factor for lateral error

        Returns:
            steering: Steering value between -1 and 1
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
        steering_angle = heading_error + np.arctan2(k * crosstrack_error, velocity + 1e-5)

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
        
        # print(f"speed: {speed:.3f}")
        
        # Es konnte zu debugzwecken der angepeilte Trajektorie-Punkt ausgegeben werden
        if speed < 75:
            steering = self.stanley_controller(trajectory, speed) # für kurven -> niedrige Geschw.
        else:
            steering = self.pure_pursuit_control(trajectory, speed) # für "geraden" -> höhere Geschw.
            
        #print(f"steering: {steering}")
        return steering
        # return steering, trajectory, target_point

        return steering