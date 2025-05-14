from typing import Tuple
import numpy as np
from scipy.interpolate import make_splprep, splev, splprep
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.spatial import ConvexHull, KDTree
from matplotlib.path import Path
from sklearn.neighbors import NearestNeighbors

class PathPlanning:

    def __init__(self):
        self.car_position = np.array([48, 67])
        pass
    
    def adjust_lanes(self, lane: np.ndarray, smoothing_factor: int=0, sample_points: int=10) -> np.ndarray:
        """Fits and samples a B-spline curve along a given lane.

        Args:
            lane (np.ndarray): A NumPy array of shape (n, 2), where each row represents a (x, y) coordinate along the lane.
            smoothing_factor (int, optional): Smoothing condition for the B-spline. A value of 0 means no smoothing. Defaults to 0.
            sample_points (int, optional): Number of evenly spaced points to sample along the fitted spline. Defaults to 10.

        Returns:
            np.ndarray: An array of shape (sample_points, 2) containing the sampled points
            along the spline curve, or an empty array of shape (0, 2) if an error occurs.
        """
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
        except Exception:
            return np.empty((0, 2))
    
    def calculate_centerline(self, left_lane: np.ndarray, right_lane: np.ndarray) -> np.ndarray:
        """Calculates the geometric centerline between two lanes.

        Args:
            left_lane (np.ndarray): A NumPy array of shape (n, 2) representing the coordinates of the left lane boundary.
            right_lane (np.ndarray): A NumPy array of shape (n, 2) representing the coordinates of the right lane boundary.

        Returns:
            np.ndarray: A NumPy array of shape (n, 2) representing the centerline between the 
            left and right lanes, or an empty array if the input shapes are not equal.
        """
        
        if left_lane.shape == right_lane.shape:
            centerline = np.array(np.mean([left_lane, right_lane], axis=0))
            return centerline
        else:
            # print("Incompatible shapes: left_lane and right_lane must have the same length")
            centerline = np.empty((0, 2))
    
    def calculate_curvature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculates the curvature k of a 2D curve defined by x and y coordinates using:

            κ = (dx * d²y - dy * d²x) / (dx² + dy²)^(3/2)

        where:
            - dx and dy are the first derivatives of x and y,
            - d²x and d²y are the second derivatives of x and y.

        Args:
            x (np.ndarray): 1D array of x-coordinates along the curve.
            y (np.ndarray): 1D array of y-coordinates along the curve.

        Returns:
            np.ndarray: 1D array of curvature values at each point.
            If an error occurs (e.g., due to insufficient data), returns an empty array.
        """
        try:
            dx = np.gradient(x)
            dy = np.gradient(y)
            dxx = np.gradient(dx)
            dyy = np.gradient(dy)
            curvature = (dx * dyy - dy * dxx) / np.power(dx**2 + dy**2, 1.5)
            return np.array(curvature)
        except Exception as e:
            # print("Curvature calculation error: x or y too small to calculate gradient")
            return np.empty((0, 2))

    def apply_mask(self, lane: np.ndarray, min_x: int, max_x: int, min_y: int, max_y: int) -> np.ndarray:
        """Applies a rectangular mask to filter lane points within specified bounds.

        Args:
            lane (np.ndarray): A NumPy array of shape (n, 2) representing 2D lane points,where each row is a (x, y) coordinate.
            min_x (int): Minimum x-coordinate for the mask (exclusive).
            max_x (int): Maximum x-coordinate for the mask (exclusive).
            min_y (int): Minimum y-coordinate for the mask (exclusive).
            max_y (int): Maximum y-coordinate for the mask (exclusive).

        Returns:
            np.ndarray: A filtered NumPy array containing only the points within the
            specified bounding box. If `lane` is not a 2D array, the original array is returned.
        """
        if lane.ndim != 2:
            return lane
        
        # x goes from 0 - 96, y goes from 0 - 84
        mask = (min_x < lane[:,0]) & (lane[:, 0] < max_x) & (min_y < lane[:,1]) & (lane[:,1] < max_y)
        return np.array(lane[mask])
    
    def find_nearest_neighbour(self, lane: np.ndarray, radius: int=12) -> np.ndarray:
        """Filter points of a lane based on the nearest neighbors algorithm.

        This function performs a search to find the closest points along a lane to the car's 
        position and then iteratively selects neighbors that are within the specified radius. 
        It chooses the next point with the best forward alignment (dot product) relative to 
        the current direction.

        The search stops when no further valid neighbors can be found within the radius.

        Args:
            lane (np.ndarray): A 2D NumPy array of shape (n, 2), where each row represents an (x, y) coordinate along the lane.
            radius (int, optional): The radius within which to search for neighbors. Defaults to 12.

        Returns:
            np.ndarray: A 2D array of shape (m, 2) containing the sequence of points visited 
            along the lane, starting from the closest point to the car. If no valid lane points 
            are found, the original `lane` is returned.
        """
        if lane.ndim != 2:
            return lane
        
        # initialization
        visited = []
        visited_set = set()
        
        # find closest lane point to car
        distances = np.linalg.norm(lane - self.car_position, axis=1)
        if len(distances) == 0:
            return lane
        start_idx = np.argmin(distances)
        current_point = lane[start_idx]
        
        visited.append(current_point)
        visited_set.add(tuple(current_point))
        
        current_vec = np.array([0.0, -1.0])
        
        # radius based neighbour index
        nn = NearestNeighbors(radius=radius)
        nn.fit(lane)
        
        while True:
            # Find all neighbour indices within radius
            neighbors_idx = nn.radius_neighbors([current_point], return_distance=False)[0]
            candidates = []
            
            for idx in neighbors_idx:
                point = lane[idx]
                point_tuple = tuple(point)
                if point_tuple in visited_set:
                    continue # skip if point already visited
                
                # Vector from current point to candidate
                direction = point - current_point
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue # Skip zero-length vectors
                
                # Normalize direction and compute alignment (dot product)
                direction_unit = direction / norm
                angle = np.dot(direction_unit, current_vec)
                
                if angle > 0:
                    candidates.append((point, angle))
                    
            if not candidates:
                break
            
            # Select candidate with best forward alignment (max angle)
            next_point, _ = max(candidates, key=lambda x: x[1])
            visited.append(next_point)
            visited_set.add(tuple(next_point))
            
            # Update direction vector and move forward
            current_vec = (next_point - current_point)
            current_vec = current_vec / np.linalg.norm(current_vec)
            current_point = next_point
        
        return np.array(visited)
    
    def trim_by_pairing(self, left_lane: np.ndarray, right_lane: np.ndarray, max_pairing_distance: float=30.0) -> Tuple[np.ndarray, np.ndarray]:
        """Performs pairing of points between two lanes based on proximity.

        This function pairs points from the `left_lane` and `right_lane` arrays using a 
        greedy algorithm. The function iterates through the points in `left_lane` and 
        finds the closest unpaired point in `right_lane` within a specified maximum 
        distance (`max_pairing_distance`). Once a point is paired, it is marked as used, 
        and the algorithm moves to the next point in the `left_lane`. The pairing stops 
        when no more valid pairs can be found within the given maximum distance.

        Args:
            left_lane (np.ndarray): A 2D array of shape (n, 2) representing points along the left lane.
            right_lane (np.ndarray): A 2D array of shape (n, 2) representing points along the right lane.
            max_pairing_distance (float, optional): The maximum allowed distance between points from the left and right lanes to consider them a pair. Defaults to 30.0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing the paired points from the 
            left and right lanes, respectively. If no points are paired, empty arrays are returned.
        """
        
        if len(left_lane) == 0 or len(right_lane) == 0:
            return left_lane, right_lane
        
        left_lane = np.asarray(left_lane)
        right_lane = np.asarray(right_lane)

        visited_left = []
        visited_right = []
        used_right_indices = set()

        for i, left_point in enumerate(left_lane):
            # Compute distances to all right border points
            dists = np.linalg.norm(right_lane - left_point, axis=1)

            # Mask out already used right indices
            for idx in used_right_indices:
                dists[idx] = np.inf

            min_dist_idx = np.argmin(dists)
            min_dist = dists[min_dist_idx]

            if min_dist <= max_pairing_distance:
                visited_left.append(left_point)
                visited_right.append(right_lane[min_dist_idx])
                used_right_indices.add(min_dist_idx)
            else:
                # Stop when we can't find a suitable match
                break

        return np.array(visited_left), np.array(visited_right)
    
    def optimize_trajectory(self, centerline: np.ndarray, centerline_curvature: np.ndarray) -> np.ndarray:
        """Optimizes the trajectory of a vehicle along a centerline using curvature information.

        This function adjusts the centerline by shifting each point along its normal vector
        based on the curvature at that point. The magnitude of the shift is determined by the
        curvature, and it is capped to a range of [-7, 7] for stability. The trajectory is optimized
        by modifying the centerline's points in a direction perpendicular to the tangent vector.

        Args:
            centerline (np.ndarray): A 2D array of shape (n, 2) representing the points of the centerline.
            centerline_curvature (np.ndarray): A 1D array of shape (n,) representing the curvature at each point along the centerline.

        Returns:
            np.ndarray: A 2D array of shape (n, 2) representing the optimized trajectory based on the centerline
            and the curvature information. If curvature is negligible, the original centerline is returned.
        """
        
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
    
    def is_point_in_boundary(self, edges: list, x: float, y: float) -> bool:
        """Determines if a point is inside a polygon using the ray tracing algorithm.

        Args:
            edges (list): A list of tuples, where each tuple contains two points representing an edge of the polygon, e.g., [(x1, y1), (x2, y2)].
            x (float): The x-coordinate of the point to check.
            y (float): The y-coordinate of the point to check.

        Returns:
            bool: True if the point (x, y) is inside the polygon; False otherwise.
        """
        inside = False
        for (x1, y1), (x2, y2) in edges:
            if (y < y1) != (y < y2):
                if x <= x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                    inside = not inside
        return inside
        
    def get_hull_points(self, left_lane):
        """Computes the convex hull of a set of points representing a lane.

        Args:
            left_lane (np.ndarray): A 2D array of shape (n, 2), where each row represents a (x, y) coordinate of a point along the left lane.

        Returns:
            np.ndarray: An array of points forming the convex hull, or an empty list if an error occurs during the computation (e.g., if the input data is insufficient).
        """
        try:
            hull = ConvexHull(left_lane)
            hull_points = left_lane[hull.vertices]
            
            return hull_points
        except:
            return []
        
    def filter_outside_track_points(self, left_lane: np.ndarray, trajectory: np.ndarray, centerline_curvature: np.ndarray) -> np.ndarray:
        """Filters out points from the trajectory that lie outside the left lane boundary.

        Args:
            left_lane (np.ndarray): A 2D array of (x, y) coordinates representing the left lane boundary.
            trajectory (np.ndarray): A 2D array of (x, y) coordinates representing the trajectory points to be filtered.
            centerline_curvature (np.ndarray): A 1D array representing the curvature values
                of the centerline at each point. A negative sum of curvature values is used
                to determine if the trajectory is for a left turn.

        Returns:
            np.ndarray: The filtered trajectory points that lie within the lane boundary
            and the specified mask. If an error occurs, the original trajectory is returned.
        """
        if np.sum(centerline_curvature) < 0: # only for left turns
            try:
                hull = ConvexHull(left_lane)
                hull_points = left_lane[hull.vertices]

                # mask = Path.contains_points(trajectory)
                # return trajectory[mask]

                edges = list(zip(hull_points, np.roll(hull_points, -1, axis=0)))

                filtered_trajectory = [
                    point for point in trajectory
                    if not self.is_point_in_boundary(edges, point[0], point[1])
                ]
                filtered_trajectory = np.array(filtered_trajectory)
                filtered_trajectory = self.apply_mask(filtered_trajectory, 48-10, 48+10, 0, 83)

                return np.array(filtered_trajectory)
            except Exception as e:
                return trajectory
        else:
            return trajectory
        
    def calculate_curvature_output(self, points: np.ndarray) -> float:
        """Calculates the total curvature of a 2D curve based on the change in direction between points for longitudinal control.

        This function computes the total change in direction between consecutive points along
        a 2D curve, normalizes it, and returns the curvature value between 0.0 and 1.0.
        A straight line will return a curvature of 0.0, and a maximal curvature (180° change in direction)
        will return a value of 1.0.

        Args:
            points (np.ndarray): An Nx2 array of 2D coordinates representing points along the curve.

        Returns:
            float: A curvature value between 0.0 and 1.0, representing the total curvature of the curve.
        """
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
            return 0.0

        if points.shape[0] < 3:
            return 0.0

        # Calculate direction vectors between consecutive points
        d = np.diff(points, axis=0)

        # Normalize the direction vectors
        norm = np.linalg.norm(d, axis=1)
        d_unit = d / norm[:, None]

        # Compute the dot product of adjacent unit vectors to get cos(θ)
        dot = np.einsum('ij,ij->i', d_unit[:-1], d_unit[1:])
        angles = np.arccos(np.clip(dot, -1.0, 1.0))  # Numerically stable calculation

        # Compute the total change in angle
        total_angle = angles.sum()

        # Normalize by the maximum possible change (π)
        return min(1.0, total_angle / np.pi)

    def calculate_arc_length(self, points: np.ndarray) -> float:
        """Calcuates the total length of a lane using the sum of all the euclidean distances between points.
        
        Args:
        points (np.ndarray): A 2D NumPy array of shape (n, 2), where each row represents a point (x, y) along the lane.

        Returns:
            float: The total length of the lane as the sum of Euclidean distances between consecutive points.
        """
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    
    def plan(self, left_lane_points: list, right_lane_points: list) -> Tuple[np.ndarray, float]:
        """Plans a trajectory based on left and right lane points.

        This function adjusts and samples the provided lane points, calculates the centerline,
        and optimizes the trajectory. It also calculates the curvature along the centerline and 
        smooths the resulting trajectory to ensure it is feasible for the lateral control.

        If the arc length difference between the left and right lanes exceeds a threshold, the lanes 
        are trimmed to align with each other. After generating the trajectory, it is smoothed to ensure a stable lateral control.

        Args:
            left_lane_points (list): List of points representing the left lane boundary.
            right_lane_points (list): List of points representing the right lane boundary.

        Returns:
            Tuple[np.ndarray, float]: The optimized trajectory as a 2D NumPy array of shape (n, 2),
            and the curvature of the trajectory.
        """

        left_lane_points = self.apply_mask(np.asarray(left_lane_points), 0, 96, 0, 77)
        right_lane_points = self.apply_mask(np.asarray(right_lane_points), 0, 96, 0, 77)
            

        left_lane_points = self.find_nearest_neighbour(np.array(left_lane_points))
        right_lane_points = self.find_nearest_neighbour(np.array(right_lane_points))

        # Adjust and sample lanes
        left_lane = self.adjust_lanes(left_lane_points,  sample_points=15)
        right_lane = self.adjust_lanes(right_lane_points, sample_points=15)

        l_l_l = self.calculate_arc_length(left_lane) # l_l_l: left_lane_length
        r_l_l = self.calculate_arc_length(right_lane) # r_l_l: left_lane_length

        if l_l_l - r_l_l > 40:
            left_lane, right_lane = self.trim_by_pairing(left_lane, right_lane, max_pairing_distance=30.0)
            left_lane = self.adjust_lanes(left_lane,  sample_points=15)
            right_lane = self.adjust_lanes(right_lane, sample_points=15)

        # Calculate centerline
        centerline = self.calculate_centerline(left_lane, right_lane)
        if centerline is None or centerline.shape[0] == 0:
            return np.empty((0, 2)), 0.0

        # Clip centerline to stay above car
        centerline = self.apply_mask(centerline, 0, 96, 0, 67)
        centerline = self.find_nearest_neighbour(centerline, radius=8)
        centerline = self.adjust_lanes(centerline, smoothing_factor=5.0, sample_points=15)

        # Calculate curvature of centerline
        centerline_curvature = self.calculate_curvature(x=centerline[:,0], y=centerline[:,1])

        if len(centerline_curvature) == 0:
            return centerline, 0.0

        # Optimize trajectory
        optimized_trajectory = self.optimize_trajectory(centerline, centerline_curvature)

        # Smooth trajectory
        optimized_trajectory = self.adjust_lanes(optimized_trajectory, smoothing_factor=10.0, sample_points=20)

        # Filter points in trajectory so curves aren't too early detected
        optimized_trajectory = self.apply_mask(optimized_trajectory, 0, 96, 10, 64)

        if len(optimized_trajectory) == 0:
            return centerline, self.calculate_curvature_output(centerline)

        return optimized_trajectory, self.calculate_curvature_output(optimized_trajectory)