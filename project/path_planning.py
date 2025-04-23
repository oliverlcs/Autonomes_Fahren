import numpy as np
from scipy.interpolate import make_splprep

class PathPlanning:

    def __init__(self):
        pass

    def plan(self, left_lane, right_lane):
        left_lane = np.array(left_lane)
        right_lane = np.array(right_lane)
        
        spl_left, _ = make_splprep([left_lane[:,0], left_lane[:,1]], s=1.0)
        spl_right, _ = make_splprep([right_lane[:,0], right_lane[:,1]], s=1.0)
        
        # Sample 500 points along each border
        x_vals = np.linspace(0.15, 1, 500)
        
        points_left = np.array(spl_left(x_vals)).T
        points_right = np.array(spl_right(x_vals)).T
    
        centerline = np.array(np.mean([points_left, points_right], axis=0))
        print(centerline)
        curvature = 0
    
        return centerline, curvature
        