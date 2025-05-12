import numpy as np


class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 64])
        self._car_vec = np.array([0, -1])
        
    def pure_pursuit_control(self, trajectory, speed, lookahead_gain=0.2, max_steering=1.0):

        # Clip trajectory so it is inside bounds
        mask_trajectory = (0 < trajectory[:,0]) & (trajectory[:, 0] < 96) & (0 < trajectory[:,1]) & (trajectory[:,1] < 67)
        trajectory = np.array(trajectory[mask_trajectory])
        
        # Optional - compute lookahead distance L_d+
        lookahead_gain = 0.3
        # lookahead_gain = 0.2
        # L_d = np.clip(lookahead_gain * (speed ** 1.2), 5, 35)
        # L_d = np.clip(lookahead_gain * np.log1p(speed), 5, 35)
        # L_d = np.clip(lookahead_gain * np.sqrt(speed), 5, 35)
        L_d = np.clip(lookahead_gain * speed, 5, 35)
        
        # Optional - find the first trajectory point farther than L_d
        dists = np.linalg.norm(trajectory - self._car_position, axis=1)
        target_idx = np.where(dists > L_d)[0]
        
        if len(target_idx) == 0:
            target_point = trajectory[-1] # no target_idx, take last point of trajectory
        else:
            target_point = trajectory[target_idx[0]]

        # Transform target point into vehicle coordinates
        dx = target_point[0] - self._car_position[0] # lateral offset
        dy = target_point[1] - self._car_position[1] # longitudal offset (optimal: always positive)
        if dy >= -0.01:
            return 0.0, target_point # If target is behind or very close drive straight
        
        # Calculate kappa k = 2dx / L_d^2 and map to steering
        kappa = 2.0 * dx / (L_d ** 2)
        x = 10
        # steering = np.clip(kappa * x, -max_steering, max_steering)
        steering = np.tanh(kappa * x)
        
        return steering, target_point
    
    def normalize_angle(self, angle):
        """Normiert Winkel auf [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def stanley_controller(self, trajectory, velocity, k=4.0):
        """
        Stanley Controller zur Querregelung.
        
        Args:
            vehicle_pos: Fahrzeugposition als np.array([x, y])
            vehicle_dir: Fahrzeugrichtung als np.array([dx, dy])
            trajectory: Liste von Punkten (x, y)
            velocity: Fahrzeuggeschwindigkeit (v > 0)
            k: Verstärkungsfaktor für lateralen Fehler

        Returns:
            steering: Steuerwert zwischen -1 und 1
        """
        
        # 1. Finde Zielpunkt (nächster Punkt auf der Trajektorie)
        traj_points = np.array(trajectory)
        distances = np.linalg.norm(traj_points - self._car_position, axis=1)
        closest_idx = np.argmin(distances)
        target_point = traj_points[closest_idx]

        # 2. Tangentenwinkel am Zielpunkt berechnen
        if closest_idx < len(traj_points) - 1:
            next_point = traj_points[closest_idx + 1]
        else:
            next_point = traj_points[closest_idx]

        tangent_vec = next_point - target_point
        psi_path = np.arctan2(tangent_vec[1], tangent_vec[0])

        # 3. Fahrzeugwinkel berechnen
        psi_vehicle = -np.pi / 2

        # 4. Heading-Fehler
        heading_error = self.normalize_angle(psi_path - psi_vehicle)

        # 5. Lateraler Fehler (e): senkrechter Abstand mit Vorzeichen
        vec_to_traj = target_point - self._car_position
        lateral_error = np.cross(self._car_vec, vec_to_traj)  # Skalarprodukt im 2D

        # 6. Stanley-Steuerung
        steering_angle = heading_error + np.arctan2(k * lateral_error, velocity + 1e-5)

        # 7. Steuerung auf [-1, 1] normalisieren
        steering_output = np.tanh(steering_angle)
        return steering_output, target_point

    def control(self, trajectory, speed):
        # steering = -1: driving left
        # steering = 1: driving right
        
        if len(trajectory) == 0:
            return 0
        
        # print(f"speed: {speed:.3f}")
        
        if speed < 65:
            steering, target_point = self.stanley_controller(trajectory, speed)
        else:
            steering, target_point  = self.pure_pursuit_control(trajectory, speed)
        
        return steering, target_point
        # return steering, trajectory, target_point

