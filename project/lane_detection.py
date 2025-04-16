import numpy as np
from scipy.ndimage import convolve

class LaneDetection:
    def __init__(self):
        self.debug_image = None

    def detect(self, image: np.ndarray):
        # Sobel filter kernels for edge detection
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Mask the car area (assume the car is in the center-bottom of the image)
        mask = np.ones(image.shape[:2], dtype=np.uint8)  # Create a mask with all ones
        car_position = np.array([48, 64])
        car_mask_height = 20  # Height of the car mask
        car_mask_width = 40   # Width of the car mask
        center_x = image.shape[1] // 2
        center_y = image.shape[0] - car_mask_height
        mask[car_position[1]:, car_position[0] - car_mask_width // 2:car_position[0]  + car_mask_width // 2] = 0

        # Apply the mask to the image
        masked_image = image * mask[..., None]

        # Apply Sobel filter to each RGB channel
        edges_r_x = convolve(masked_image[..., 0], sobel_x)
        edges_r_y = convolve(masked_image[..., 0], sobel_y)
        edges_g_x = convolve(masked_image[..., 1], sobel_x)
        edges_g_y = convolve(masked_image[..., 1], sobel_y)
        edges_b_x = convolve(masked_image[..., 2], sobel_x)
        edges_b_y = convolve(masked_image[..., 2], sobel_y)

        # Combine edges from all channels
        edges_r = np.sqrt(edges_r_x**2 + edges_r_y**2)
        edges_g = np.sqrt(edges_g_x**2 + edges_g_y**2)
        edges_b = np.sqrt(edges_b_x**2 + edges_b_y**2)

        # Combine RGB edges into a single edge map
        edges = np.sqrt(edges_r**2 + edges_g**2 + edges_b**2)
        edges = (edges / edges.max() * 255).astype(np.uint8)

        # Threshold edges to create a binary edge map
        edge_threshold = 50
        edge_binary = (edges > edge_threshold).astype(np.uint8) * 255

        # Detect lines (simple approach: return non-zero coordinates)
        lines = np.column_stack(np.where(edge_binary > 0))

        # Separate left and right lines based on the image center
        image_center = image.shape[1] // 2
        left_lines = lines[lines[:, 1] < image_center]
        right_lines = lines[lines[:, 1] >= image_center]

        # Store the result for debugging/visualization
        self.debug_image = edge_binary

        return left_lines, right_lines
