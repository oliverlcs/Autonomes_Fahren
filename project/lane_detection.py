import numpy as np
from scipy.ndimage import convolve, gaussian_filter

class LaneDetection:
    def __init__(self):
        self.debug_image = None

    def detect(self, image: np.ndarray):
        # Convert to grayscale (assuming input is RGB)
        gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

        # Apply Gaussian blur to reduce noise
        blurred = gaussian_filter(gray, sigma=1)

        # Create a binary image using thresholding
        threshold = 120  # Adjust this value based on your input images
        binary = (blurred > threshold).astype(np.uint8) * 255

        # Sobel filter kernels for edge detection
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply Sobel filter to detect edges
        edges_x = convolve(binary, sobel_x)
        edges_y = convolve(binary, sobel_y)

        # Combine edges from both directions
        edges = np.sqrt(edges_x**2 + edges_y**2)
        edges = (edges / edges.max() * 255).astype(np.uint8)

        # Threshold edges to create a binary edge map
        edge_threshold = 50
        edge_binary = (edges > edge_threshold).astype(np.uint8) * 255

        # Detect lines (simple approach: return non-zero coordinates)
        lines = np.column_stack(np.where(edge_binary > 0))

        # Store the result for debugging/visualization
        self.debug_image = edge_binary

        return lines
