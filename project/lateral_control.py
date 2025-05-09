import numpy as np


class LateralControl:

    def __init__(self):
        self._car_position = np.array([48, 64])
        
<<<<<<< Updated upstream
    def pure_pursuit_control(self, trajectory, speed, current_pos=np.array([48.0, 64.0]), lookahead_gain=0.2, max_steering=1.0):
=======
    def pure_pursuit_control(self, trajectory, speed, current_pos=np.array([48.0, 64.0]), lookahead_gain=0.1, max_steering=1.0):
>>>>>>> Stashed changes
        # Clip trajectory so it is inside bounds
        mask_trajectory = (0 < trajectory[:,0]) & (trajectory[:, 0] < 96) & (0 < trajectory[:,1]) & (trajectory[:,1] < 67)
        trajectory = np.array(trajectory[mask_trajectory])
        
        # Optional - compute lookahead distance L_d
<<<<<<< Updated upstream
        L_d = np.clip(lookahead_gain * (speed ** 1.2), 5, 35)
=======
        lookahead_gain = 0.2
        L_d = np.clip(lookahead_gain * (speed ** 1.2), 5, 35)
        print("L_d: ", L_d)
>>>>>>> Stashed changes
        # L_d = np.clip(lookahead_gain * np.log1p(speed), 5, 35)
        # L_d = np.clip(lookahead_gain * np.sqrt(speed), 5, 35)
        # L_d = np.clip(lookahead_gain * speed, 5, 35)
        
        # Optional - find the first trajectory point farther than L_d
        dists = np.linalg.norm(trajectory - current_pos, axis=1)
        target_idx = np.where(dists > L_d)[0]
        
        if len(target_idx) == 0:
            target_point = trajectory[-1] # no target_idx, take last point of trajectory
        else:
            target_point = trajectory[target_idx[0]]

        # Transform target point into vehicle coordinates
        dx = target_point[0] - current_pos[0] # lateral offset
        dy = target_point[1] - current_pos[1] # longitudal offset (optimal: always positive)
        if dy >= -0.01:
            return 0.0, target_point # If target is behind or very close drive straight
        
        # Calculate kappa k = 2dx / L_d^2 and map to steering
        kappa = 2.0 * dx / (L_d ** 2)
        print("kappa: ", kappa)
        x = 10
        steering = np.clip(kappa * x, -max_steering, max_steering)
        
        return steering, target_point

    def control(self, trajectory, speed):
        # steering = -1: driving left
        # steering = 1: driving right
        
<<<<<<< Updated upstream
        steering, trajectory, target_point  = self.pure_pursuit_control(trajectory, speed, self._car_position)
=======
        if len(trajectory) == 0:
            return 0
        
        steering, target_point  = self.pure_pursuit_control(trajectory, speed, self._car_position)
>>>>>>> Stashed changes
        
        return steering, target_point
        # return steering, trajectory, target_point

