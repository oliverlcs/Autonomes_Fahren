
        # Calculate curvature of left and right lane
        # left_lane_curvature = self.calculate_curvature(left_lane[:,0], left_lane[:,1])
        # right_lane_curvature = self.calculate_curvature(right_lane[:,0], right_lane[:,1])
    
        # Find peaks in left and right lane
        # left_lane_peaks, _ = find_peaks(np.abs(left_lane_curvature), height=0.1, distance=20)
        # right_lane_peaks, _ = find_peaks(np.abs(right_lane_curvature), height=0.1, distance=20)
        
        # Select top 3 apexes from left and right lane
        # left_lane_peaks_top_3 = self.get_highest_peaks(left_lane_curvature, left_lane_peaks, 3)
        # right_lane_peaks_top_3 = self.get_highest_peaks(right_lane_curvature, right_lane_peaks, 3)
        
        # Find apexes (peaks) of centerline
        # centerline_peaks, _ = find_peaks(np.abs(centerline_curvature), height=0.1, distance=10)
        # peaks_top_1 = self.get_highest_peaks(centerline_curvature, centerline_peaks, 1)
        
        # optimized_trajectory = self.optimize_trajectory_combined(left_lane, right_lane, centerline_clipped, curvature_clipped, peaks_clipped, peaks_top_1, left_lane_peaks_top_3, right_lane_peaks_top_3)
        
        
        # optimized_trajectory = self.optimize_trajectory_scipy(centerline, 5)
        
        # optimized_trajectory = self.optimize_trajectory_with_bias(
        #     left_lane, right_lane,
        #     centerline, 
        #     centerline_curvature, 
        #     peaks_top_1,
        #     left_lane[left_lane_peaks_top_3],
        #     right_lane[right_lane_peaks_top_3])
        
        # optimized_trajectory = self.optimize_trajectory_bspline(left_lane_points, right_lane_points, scale=30, max_shift=20)
        
        # Add Curvature & Steering Smoothness Constraints

    
# --- Plotting ---
# plt.figure(figsize=(10,6))
        
# # Original lanes
# plt.plot(left_lane[:,0], left_lane[:,1], 'g--', label='Left Lane')
# plt.plot(right_lane[:,0], right_lane[:,1], 'g--', label='Right Lane')
        
# # Centerline
# plt.plot(centerline[:,0], centerline[:,1], 'k-', label='Centerline')
        
# # Apex points (curvature peaks)
# plt.plot(centerline[peaks, 0], centerline[peaks, 1], 'ro', label='Apex Points')
        
# plt.xlim(0, 96)
# plt.ylim(0, 84)
# plt.legend()
# plt.title('Trajectory Planning with Apex Points')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.grid(True)
# plt.axis('equal')
# plt.show()


# def optimize_trajectory_scipy(self, centerline, offset_distance):
#         centerline = np.asarray(centerline)
#         tck, u = splprep(centerline.T, s=0.5)
        
#         u_dense = np.linspace(0, 1, len(centerline))
#         x, y = splev(u_dense, tck)
        
#         # Derivatives (tangent vectors)
#         dx, dy = splev(u_dense, tck, der=1)
        
#         # Normalize tangents
#         norms = np.hypot(dx, dy)
#         tangents = np.vstack((dx / norms, dy / norms)).T  # (N, 2)

#         # Calculate normals (perpendicular vectors)
#         normals = np.vstack((-tangents[:,1], tangents[:,0])).T  # rotate +90°

#         # Offset the points
#         offset_points = np.vstack((x, y)).T + normals * offset_distance

#         return offset_points        
    
#     def calculate_attraction_strength(self, curvature_value):
#         curvature_abs = np.abs(curvature_value)
#         scaling_factor = 10.0  # because track units are big (~100px)
#         min_pull = 0.5          # minimum pull strength (~half a pixel)
#         max_pull = 5.0          # maximum pull (~5 pixels)
#         strength = np.clip(curvature_abs * scaling_factor, min_pull, max_pull)
#         return strength

# def get_highest_peaks(self, curvature, peaks, top_x_peaks):
#         peak_curvatures = np.abs(curvature[peaks])
#         sorted_indices = np.argsort(peak_curvatures)[::-1]
#         peaks_top_x = peaks[sorted_indices[:top_x_peaks]]
        
#         return peaks_top_x[0]


# def find_nearest_apex_point(self, centerline_point, apex_points):
#         distances = np.linalg.norm(apex_points - centerline_point, axis=1)
#         nearest_idx = np.argmin(distances)
#         return apex_points[nearest_idx]
    
    
#     def smooth_centerline(self, points, smoothing_factor=0.5):
#         n_points = len(points)
#         t = np.linspace(0, 1, n_points)
#         spl, _ = make_splprep(points.T, u=t, s=smoothing_factor)  # Small smoothing factor
#         u_dense = np.linspace(0, 1, n_points)
#         smoothed_points = np.array(spl(u_dense)).T
#         return smoothed_points
    
#     def correct_out_of_bounds(self, centerline, left_lane, right_lane):
#         corrected_centerline = centerline.copy()

#         for idx, point in enumerate(centerline):
#             left_x, left_y = left_lane[idx]
#             right_x, right_y = right_lane[idx]

#             if not (min(left_x, right_x) <= point[0] <= max(left_x, right_x)):
#                 corrected_centerline[idx][0] = np.clip(point[0], min(left_x, right_x), max(left_x, right_x))

#             if not (min(left_y, right_y) <= point[1] <= max(left_y, right_y)):
#                 corrected_centerline[idx][1] = np.clip(point[1], min(left_y, right_y), max(left_y, right_y))

#         return corrected_centerline
        
#     def optimize_trajectory_with_bias(self, left_lane, right_lane, centerline, centerline_curvature, peaks_top_1, left_apex_points, right_apex_points):
#         if len(peaks_top_1) == 0:
#             return centerline  # No apex detected, fallback to original
        
#         main_apex_idx = peaks_top_1[0]  # Index of highest curvature apex
        
#         # Determine curve direction
#         curve_direction = 'right' if centerline_curvature[main_apex_idx] > 0 else 'left'
        
#         # Select apex points based on curve direction
#         selected_apex_points = right_apex_points if curve_direction == 'right' else left_apex_points
        
#         if selected_apex_points is None or len(selected_apex_points) == 0:
#             return centerline  # No apex points found, fallback to original

#         # Define adjustment window (~60 points)
#         window_size = 30
#         window_start = max(main_apex_idx - window_size, 0)
#         window_end = min(main_apex_idx + window_size, len(centerline))
#         adjustment_window = np.arange(window_start, window_end)
        
#          # Bias the centerline
#         new_centerline = centerline.copy()
        
#         for idx in adjustment_window:
#             curvature_at_point = centerline_curvature[idx]
#             attraction_strength = self.calculate_attraction_strength(curvature_at_point)
            
#             nearest_apex = self.find_nearest_apex_point(new_centerline[idx], selected_apex_points)
#             direction_vector = nearest_apex - new_centerline[idx]
#             new_centerline[idx] += attraction_strength * direction_vector
            
#         # Smooth the biased centerline
#         new_centerline = self.smooth_centerline(new_centerline, smoothing_factor=10)
        
#         # Validate and push points back inside track boundaries
#         new_centerline = self.correct_out_of_bounds(new_centerline, left_lane, right_lane)

#         return new_centerline
    
#     def optimize_trajectory_bspline(self, left_lane: np.ndarray, right_lane: np.ndarray, scale: float = 30.0, max_shift: float = 5.0, smoothing_factor: float = 0.5, n_samples: int = 1000) -> np.ndarray:
#         """
#         Optimize the trajectory by fitting splines to left/right lanes,
#         calculating a centerline, adjusting it based on curvature, and refitting a smooth spline.

#         Args:
#             left_lane (np.ndarray): (N, 2) array of left lane points.
#             right_lane (np.ndarray): (N, 2) array of right lane points.
#             scale (float): Multiplier for curvature-based shifting.
#             max_shift (float): Maximum allowed lateral shift (in lane units).
#             smoothing_factor (float): Spline smoothing parameter.
#             n_samples (int): Number of samples along the splines.

#         Returns:
#             np.ndarray: (n_samples, 2) optimized centerline points.
#         """
#         # Fit splines to left and right lanes
#         spl_left, _ = make_splprep(left_lane.T, s=smoothing_factor)
#         spl_right, _ = make_splprep(right_lane.T, s=smoothing_factor)
        
#         u_vals = np.linspace(0, 1, n_samples)
        
#         left_points = np.array(spl_left(u_vals)).T
#         right_points = np.array(spl_right(u_vals)).T
        
#         centerline = np.array((left_points + right_points) / 2.0)
        
#         spl_center, _ = make_splprep(centerline.T, s=smoothing_factor)
#         spl_center_der = spl_center.derivative(1)
#         spl_center_der_der = spl_center.derivative(2)
        
#         x, y = spl_center(u_vals)
#         dx, dy = spl_center_der(u_vals)
#         ddx, ddy = spl_center_der_der(u_vals)
        
#         curvature = np.array((dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5))
        
#         norms = np.hypot(dx, dy)
#         tangents = np.vstack((dx / norms, dy / norms)).T
#         normals = np.vstack((-tangents[:,1], tangents[:,0])).T

#         shift_amount = np.clip(curvature * scale, -max_shift, max_shift)
        
#         shifted_x = x + normals[:,0] * shift_amount
#         shifted_y = y + normals[:,1] * shift_amount

#         shifted_points = np.vstack((shifted_x, shifted_y)).T

#         # --- Step 7: Fit final spline to shifted points ---
#         spl_final, _ = make_splprep(shifted_points.T, s=2)
#         optimized_centerline = np.array(spl_final(u_vals)).T

#         return optimized_centerline