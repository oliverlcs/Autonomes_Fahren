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
        self.own_lane_detection = True
        pass
    
    def adjust_lanes(self, lane, smoothing_factor=0, sample_points=10):
        try:
            lane = np.asarray(lane)
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
        # except Exception as e:
        #     print(e)
    
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

    def apply_mask(self, lane: np.ndarray, min_x, max_x, min_y, max_y):
        if lane.ndim != 2:
            return lane
        
        # x goes from 0 - 96, y goes from 0 - 84
        mask = (min_x < lane[:,0]) & (lane[:, 0] < max_x) & (min_y < lane[:,1]) & (lane[:,1] < max_y)
        return np.array(lane[mask])
    
    def find_nearest_neighbour(self, lane: np.ndarray, radius=12):
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
    
    def trim_by_greedy_pairing(self, left_lane, right_lane, max_pairing_distance=30.0):
        
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
    
    def sort_points_by_path(self, points):
        """
        Sort a set of 2D points in order by following the nearest neighbor path.
        
        Parameters:
            points (np.ndarray): Array of shape (N, 2)
        
        Returns:
            np.ndarray: Ordered points (N, 2)
        """
        points = np.array(points)
        n_points = len(points)
        visited = np.zeros(n_points, dtype=bool)
        ordered_points = []

        # Start at the first point
        current_index = 0
        ordered_points.append(points[current_index])
        visited[current_index] = True

        tree = KDTree(points)

        for _ in range(1, n_points):
            # Find nearest unvisited neighbor
            dist, idx = tree.query(points[current_index], k=n_points)
            for neighbor_index in idx:
                if not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    ordered_points.append(points[neighbor_index])
                    current_index = neighbor_index
                    break

        return np.array(ordered_points)

    def calculate_curvature_output(self, points: np.ndarray) -> float:
        """
        #     Funktion übertragen aus longitudinal_control.py, um gesamt Krümmung zu berechnen
        #     Hochperformante Krümmungsabschätzung einer 2D-Linie.
        #     0 = Gerade, 1 = maximale Krümmung (180° Richtungsänderung).

        #     Args:
        #         points (np.ndarray): Nx2-Array mit 2D-Koordinaten.

        #     Returns:
        #         float: Krümmung zwischen 0.0 und 1.0
        #     """

        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
            return 0.0

        if points.shape[0] < 3:
            return 0.0

        # Richtungsvektoren berechnen
        d = np.diff(points, axis=0)

        # Normalisieren
        norm = np.linalg.norm(d, axis=1)
        d_unit = d / norm[:, None]

        # Skalarprodukt benachbarter Einheitsvektoren → cos(θ)
        dot = np.einsum('ij,ij->i', d_unit[:-1], d_unit[1:])
        angles = np.arccos(np.clip(dot, -1.0, 1.0))  # numerisch stabil

        # Gesamtwinkeländerung
        total_angle = angles.sum()

        # Normierung auf maximalen möglichen Wert (pi)
        return min(1.0, total_angle / np.pi)
    
    def compute_arc_length(self, points):
        return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    

    def plan(self, left_lane_points, right_lane_points):
        
        if self.own_lane_detection:
            
            left_lane_points = self.apply_mask(np.asarray(left_lane_points), 0, 96, 0, 77)
            right_lane_points = self.apply_mask(np.asarray(right_lane_points), 0, 96, 0, 77)
            
            left_lane_points = self.find_nearest_neighbour(np.array(left_lane_points))
            right_lane_points = self.find_nearest_neighbour(np.array(right_lane_points))
            
        # Adjust and sample lanes
        left_lane = self.adjust_lanes(left_lane_points,  sample_points=15)
        right_lane = self.adjust_lanes(right_lane_points, sample_points=15)
        
        l_l_l = self.compute_arc_length(left_lane)
        r_l_l = self.compute_arc_length(right_lane)
        
        # print(f"{(l_l_l - r_l_l):.3f}")
        
        if l_l_l - r_l_l > 40:
            left_lane, right_lane = self.trim_by_greedy_pairing(left_lane, right_lane, max_pairing_distance=30.0)
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
        optimized_trajectory = self.optimize_trajectory_optimized(centerline, centerline_curvature)

        # Smooth trajectory
        optimized_trajectory = self.adjust_lanes(optimized_trajectory, smoothing_factor=10.0, sample_points=20)

        # Filter points in trajectory so curves aren't too early detected
        optimized_trajectory = self.apply_mask(optimized_trajectory, 0, 96, 10, 64)

        # Keep only valid points (i.e. inside track boundaries and not in the past)
        # optimized_trajectory = self.filter_outside_track_points(left_lane, optimized_trajectory, centerline_curvature)

        # if np.sum(np.linalg.norm(np.diff(optimized_trajectory, axis=0), axis=1)) / (len(optimized_trajectory) - 1) > 5:
        #     return centerline_fb, centerline_curvature

        if len(optimized_trajectory) == 0:
            return centerline, self.calculate_curvature_output(centerline)

        return optimized_trajectory, self.calculate_curvature_output(optimized_trajectory)
        # return centerline, optimized_trajectory, test_points