import numpy as np
from scipy.interpolate import make_splprep, splev, splprep
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

class PathPlanning:

    def __init__(self):
        pass
    
    def adjust_lanes(self, lane, smoothing_factor=0, sample_points=10):
        # spl_left is an instance of BSpline object used for fitting spline function
        spl_lane, _ = make_splprep(lane.T, s=smoothing_factor)
        
        # Sample x points along each border
        x_vals = np.linspace(0, 1, sample_points)
        
        # points_lane of shape (x, 2): arrays containing x-y-value pairs
        points_lane = np.array(spl_lane(x_vals)).T
        
        return points_lane
    
    def calculate_centerline(self, left_lane, right_lane):
        centerline = np.array(np.mean([left_lane, right_lane], axis=0))
        return centerline
    
    def calculate_curvature(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        dxx = np.gradient(dx)
        dyy = np.gradient(dy)
        curvature = (dx * dyy - dy * dxx) / np.power(dx**2 + dy**2, 1.5)
        return np.array(curvature)
    
    def scale_shift(curvature_value):
        abs_curvature = np.abs(curvature_value)
        
        # Sharp turn: > 0.015 (larger curvature)
        if abs_curvature > 0.015:
            shift = abs_curvature * 1000  # significant shift for sharp turns
        # Moderate turn: 0.01 to 0.015
        elif abs_curvature > 0.01:
            shift = abs_curvature * 500  # moderate shift for 90-degree turn
        # Gentle turn: 0.001 to 0.01
        elif abs_curvature > 0.001:
            shift = abs_curvature * 100  # small shift for gentle turns
        # Almost flat road: < 0.001
        else:
            shift = 0  # no shift for flat roads
        
        # Clip the shift so it doesn’t go beyond reasonable limits
        shift = np.clip(shift, -10, 10)
        return shift
    
    
    def optimize_trajectory(self, centerline, centerline_curvature):
        optimized_trajectory = []
        centerline = np.asarray(centerline)
        centerline_curvature = np.asarray(centerline_curvature)
        
        # Compute tangent vectors
        dx = np.gradient(centerline[:,0])
        dy = np.gradient(centerline[:,1])
        
        for i in range(len(centerline)):
            # Tangent vectors
            tangent = np.array([dx[i], dy[i]])
            tangent_norm = np.linalg.norm(tangent)
            if tangent_norm == 0:
                tangent_norm = 1e-5 # avoid division by zero
            tangent = tangent / tangent_norm
            
            # Normal vectors
            normal = np.array([-tangent[1], tangent[0]])
            
            # Amount of shift based on curvature
            print(f"{centerline_curvature[i]*200:.5f}")
            shift = np.clip(centerline_curvature[i] * 200, -7, 7)
            
            
            # Shift point along the normal
            new_point = centerline[i] + normal * shift
            optimized_trajectory.append(new_point)
            
        optimized_trajectory = np.array(optimized_trajectory)
        
        return optimized_trajectory
    
    def optimize_trajectory_optimized(self, centerline, centerline_curvature):
        centerline = np.asarray(centerline)
        centerline_curvature = np.asarray(centerline_curvature)
        
        # print(f"{np.sum(np.abs(centerline_curvature)):.5f}")
        if np.sum(np.abs(centerline_curvature)) < 0.0001:
            return centerline
        
        # Improves performance because no dynamic memory allocation
        optimized_trajectory = np.zeros_like(centerline)
        
        # Compute tangent vectors
        dx, dy = np.gradient(centerline[:, 0]), np.gradient(centerline[:, 1])
        
        # Precompute tangent vectors
        tangent = np.stack((dx, dy), axis=1)
        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.clip(tangent_norm, 1e-5, None)

        # Precompute normal vectors
        normal = np.stack((-tangent[:,1], tangent[:,0]), axis=1)
        
        # Now do the loop
        for i in range(len(centerline)):
            shift = np.clip(centerline_curvature[i] * 200, -7, 7)
            optimized_trajectory[i] = centerline[i] + normal[i] * shift
        
        return optimized_trajectory
        
    def plan(self, left_lane_points, right_lane_points):
        
        # Adjust and sample lanes
        left_lane = self.adjust_lanes(np.array(left_lane_points), sample_points=15)
        right_lane = self.adjust_lanes(np.array(right_lane_points), sample_points=15)
        
        # Calculate centerline
        centerline = self.calculate_centerline(left_lane, right_lane)
    
        # Clip centerline to stay above car
        mask = (0 < centerline[:,0]) & (centerline[:, 0] < 96) & (0 < centerline[:,1]) & (centerline[:,1] < 67)
        centerline = np.array(centerline[mask])
        
        # Calculate curvature of centerline
        centerline_curvature = self.calculate_curvature(centerline[:,0], centerline[:,1])
        
        # Optimize trajectory
        optimized_trajectory = self.optimize_trajectory_optimized(centerline, centerline_curvature)

        # Smooth trajectory
        optimized_trajectory = self.adjust_lanes(optimized_trajectory, smoothing_factor=0.2, sample_points=20)
           
        return centerline, optimized_trajectory