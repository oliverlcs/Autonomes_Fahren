import numpy as np
from scipy.interpolate import make_splprep, splev, splprep
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

class PathPlanning:

    def __init__(self):
        pass
    
    def adjust_lanes(self, lane, smoothing_factor=0, sample_points=10):
        try:
            # spl_left is an instance of BSpline object used for fitting spline function
            spl_lane, _ = make_splprep(lane.T, s=smoothing_factor)
            
            # Sample x points along each border
            x_vals = np.linspace(0, 1, sample_points)
            
            # points_lane of shape (x, 2): arrays containing x-y-value pairs
            points_lane = np.array(spl_lane(x_vals)).T
            
            return points_lane
        except ValueError:
            print("ValueError: nc = 4 > m = 3")
            return np.empty((0, 2))
        except IndexError:
            print("IndexError: index -1 is out of bounds for axis 0 with size 0")
            return np.empty((0, 2))
    
    def calculate_centerline(self, left_lane: np.array, right_lane: np.array):
        
        if left_lane.shape == right_lane.shape:
            centerline = np.array(np.mean([left_lane, right_lane], axis=0))
            return centerline
        else:
            print("Incompatible shapes: left_lane and right_lane must have the same length")
            centerline = np.empty(0) 
        
    
    def calculate_curvature(self, x, y):
        try:
            dx = np.gradient(x)
            dy = np.gradient(y)
            dxx = np.gradient(dx)
            dyy = np.gradient(dy)
            curvature = (dx * dyy - dy * dxx) / np.power(dx**2 + dy**2, 1.5)
            return np.array(curvature)
        except Exception as e:
            print("Curvature calculation error: x or y too small to calculate gradient")
            return np.empty(0)
    
    
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
    
    def filter_unitl_jumps(self, trajectory: np.array, threshold):
        trajectory = np.array(trajectory).reshape(-1, 2)  # Ensure 2D shape

        if trajectory.shape[0] == 0:
            return np.empty((0, 2))  # Return empty result if no points

        result_trajectory = [trajectory[0]]
        for i in range(1, len(trajectory)):
            dist = np.linalg.norm(trajectory[i] - trajectory[i - 1])
            if dist > threshold:
                break
            result_trajectory.append(trajectory[i])
        return np.array(result_trajectory)
        
    def plan(self, left_lane_points, right_lane_points):
        
        # Adjust and sample lanes
        left_lane = self.adjust_lanes(np.array(left_lane_points), sample_points=15)
        right_lane = self.adjust_lanes(np.array(right_lane_points), sample_points=15)
        
        y = 30
        test_points = [[10, y], [20, y], [30, y], [40, y], [50, y], [60, y], [70, y], [80, y]]
        
        # check for empty lanes when outside of a a track
        # if len(left_lane) == 0 or len(right_lane) == 0:
        #     return [], [], test_points
        
        # Calculate centerline
        centerline = self.calculate_centerline(left_lane, right_lane)
        
        if centerline is None or centerline.shape[0] == 0:
            return np.empty((0, 2)), np.empty(0)
        
        # Clip centerline to stay above car        
        mask = (0 < centerline[:,0]) & (centerline[:, 0] < 96) & (0 < centerline[:,1]) & (centerline[:,1] < 67)
        centerline = np.array(centerline[mask])
        
        # Calculate curvature of centerline
        centerline_curvature = self.calculate_curvature(x=centerline[:,0], y=centerline[:,1])
        
        if len(centerline_curvature) == 0:
            return centerline, np.empty(0)
        
        # Optimize trajectory
        optimized_trajectory = self.optimize_trajectory_optimized(centerline, centerline_curvature)

        # Smooth trajectory
        optimized_trajectory = self.adjust_lanes(optimized_trajectory, smoothing_factor=0.2, sample_points=20)
        
        # Filter points in trajectory so curves aren't too early detected
        mask_trajectory = (0 < optimized_trajectory[:,0]) & (optimized_trajectory[:, 0] < 96) & (30 < optimized_trajectory[:,1]) & (optimized_trajectory[:,1] < 67)
        optimized_trajectory = np.array(optimized_trajectory[mask_trajectory])
        
        # Filter sudden jumps in trajectory
        optimized_trajectory = self.filter_unitl_jumps(optimized_trajectory, 10)
        
        if len(optimized_trajectory) == 0:
            return centerline, centerline_curvature
        
        # Calculate curvature for each point of optimized_trajectory
        optimized_trajectory_curvature = self.calculate_curvature(x=optimized_trajectory[:,0], y=optimized_trajectory[:,1])
        
        return optimized_trajectory, optimized_trajectory_curvature
        # return centerline, optimized_trajectory, test_points