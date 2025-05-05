import numpy as np
from scipy.interpolate import make_splprep, splev, splprep
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull
from matplotlib.path import Path

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
            # print("ValueError: nc = 4 > m = 3")
            return np.empty((0, 2))
        except IndexError:
            # print("IndexError: index -1 is out of bounds for axis 0 with size 0")
            return np.empty((0, 2))
    
    def calculate_centerline(self, left_lane: np.array, right_lane: np.array):
        
        if left_lane.shape == right_lane.shape:
            centerline = np.array(np.mean([left_lane, right_lane], axis=0))
            return centerline
        else:
            # print("Incompatible shapes: left_lane and right_lane must have the same length")
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
            # print("Curvature calculation error: x or y too small to calculate gradient")
            return np.empty(0)

    def apply_mask(self, lane: np.array, min_x, max_x, min_y, max_y):
        # x goes from 0 - 96, y goes from 0 - 84
        mask = (min_x < lane[:,0]) & (lane[:, 0] < max_x) & (min_y < lane[:,1]) & (lane[:,1] < max_y)
        return np.array(lane[mask])
    
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
            shift = np.clip(centerline_curvature[i] * 200, -7, 7)
            
            # Shift point along the normal
            new_point = centerline[i] + normal * shift
            optimized_trajectory.append(new_point)
            
        optimized_trajectory = np.array(optimized_trajectory)
        
        return optimized_trajectory
    
    def optimize_trajectory_optimized(self, centerline, centerline_curvature):
        centerline = np.asarray(centerline)
        centerline_curvature = np.asarray(centerline_curvature)
        
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
    
    def is_point_in_boundary(self, edges, x, y):
        # Ray tracing algorithm to calculate if certain point is inside boundaries
        cnt = 0
        for edge in edges:
            (x1, y1), (x2, y2) = edge
            if (y < y1) != (y < y2) and x < x1 + ((y - y1) / (y2 - y1) * (x2 - x1)):
                cnt += 1
        return cnt % 2 == 1
    
    def is_point_in_boundary_opt(self, edges, x, y):
        # Ray tracing algorithm to calculate if certain point is inside boundaries - optimized
        inside = False
        for (x1, y1), (x2, y2) in edges:
            if (y < y1) != (y < y2):
                if x <= x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                    inside = not inside
        return inside
        
    def get_hull_points(self, left_lane):
        try:
            hull = ConvexHull(left_lane)
            hull_points = left_lane[hull.vertices]
            
            return hull_points
        except:
            return []
        
    def filter_outside_track_points(self, left_lane, trajectory, centerline_curvature):
        if np.sum(centerline_curvature) < 0: # only on left turns
            try:
                hull = ConvexHull(left_lane)
                hull_points = left_lane[hull.vertices]

                # mask = Path.contains_points(trajectory)
                # return trajectory[mask]

                edges = list(zip(hull_points, np.roll(hull_points, -1, axis=0)))

                filtered_trajectory = [
                    point for point in trajectory
                    if not self.is_point_in_boundary_opt(edges, point[0], point[1])
                ]
                filtered_trajectory = np.array(filtered_trajectory)
                filtered_trajectory = self.apply_mask(filtered_trajectory, 48-10, 48+10, 0, 83)

                return np.array(filtered_trajectory)
            except Exception as e:
                return trajectory
        else:
            return trajectory
    
    def filter_jumps(self, trajectory: np.array, threshold):
        trajectory = np.asarray(trajectory).reshape(-1, 2)  # Ensure 2D shape

        if trajectory.shape[0] == 0:
            return np.empty((0, 2))  # Return empty result if no points
        
        car_pos = np.array([48, 64])
        used_indices = set()
        start_idx = np.argmin(np.linalg.norm(trajectory - car_pos, axis=1))
        
        result_trajectory = [trajectory[start_idx]]
        used_indices.add(start_idx)
        current_point = trajectory[start_idx]
        
        while len(used_indices) < len(trajectory):
            min_dist = 10
            next_idx = None
            for i, point in enumerate(trajectory):
                if i in used_indices:
                    continue
                dist = np.linalg.norm(point - current_point)
                if dist < min_dist:
                    min_dist = dist
                    next_idx = i
                    
            if next_idx is None or min_dist > threshold:
                break

            current_point = trajectory[next_idx]
            result_trajectory.append(current_point)
            used_indices.add(next_idx)

        if len(result_trajectory) < 5:
            return trajectory

        return np.array(result_trajectory)
        
    def plan(self, left_lane_points, right_lane_points):
        
        # Adjust and sample lanes
        left_lane = self.adjust_lanes(np.array(left_lane_points), sample_points=15)
        
        right_lane = self.adjust_lanes(np.array(right_lane_points), sample_points=15)
        
        # Calculate centerline
        centerline = self.calculate_centerline(left_lane, right_lane)
        
        if centerline is None or centerline.shape[0] == 0:
            return np.empty((0, 2)), np.empty(0)
        
        # Clip centerline to stay above car   
        centerline = self.apply_mask(centerline, 0, 96, 0, 67)
        
        # Calculate curvature of centerline
        centerline_curvature = self.calculate_curvature(x=centerline[:,0], y=centerline[:,1])
        
        if len(centerline_curvature) == 0:
            return centerline, np.empty(0)
        
        # Optimize trajectory
        optimized_trajectory = self.optimize_trajectory_optimized(centerline, centerline_curvature)

        # Smooth trajectory
        optimized_trajectory = self.adjust_lanes(optimized_trajectory, smoothing_factor=0.2, sample_points=20)
        
        # Filter points in trajectory so curves aren't too early detected
        optimized_trajectory = self.apply_mask(optimized_trajectory, 0, 96, 30, 67)
        centerline_fb = self.apply_mask(centerline, 0, 96, 30, 67)
        
        # Keep only valid points (i.e. inside track boundaries and not in the past)
        optimized_trajectory = self.filter_outside_track_points(left_lane, optimized_trajectory, centerline_curvature)
        centerline_fb = self.filter_outside_track_points(left_lane, centerline_fb, centerline_curvature)
        
        # if np.sum(np.linalg.norm(np.diff(optimized_trajectory, axis=0), axis=1)) / (len(optimized_trajectory) - 1) > 5:
        #     return centerline_fb, centerline_curvature
        
        if len(optimized_trajectory) == 0:
            return centerline_fb, centerline_curvature
        
        # Calculate curvature for each point of optimized_trajectory
        optimized_trajectory_curvature = self.calculate_curvature(x=optimized_trajectory[:,0], y=optimized_trajectory[:,1])
        
        return optimized_trajectory, optimized_trajectory_curvature
        # return centerline, optimized_trajectory, test_points