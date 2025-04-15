import numpy as np

class LaneDetection:
    def __init__(self):
        self.debug_image = None

    def detect(self, image: np.ndarray):
        grayscale = np.dot(image[..., :3], [0.299, 0.587, 0.114])

        # X- und Y-Gradienten berechnen
        grad_x = np.abs(np.diff(grayscale, axis=1))  # horizontal
        grad_x = np.pad(grad_x, ((0, 0), (1, 0)), mode='constant')

        grad_y = np.abs(np.diff(grayscale, axis=0))  # vertikal
        grad_y = np.pad(grad_y, ((1, 0), (0, 0)), mode='constant')

        # Kantenstärke kombinieren
        edge_strength = grad_x + grad_y

        h, w = edge_strength.shape
        threshold = 80  # kann angepasst werden

        left_points = []
        right_points = []

        for y in range(h // 3, h - 5):  # nur unteren Bildbereich analysieren
            row = edge_strength[y]

            # Linker Rand: Suche von links
            for x in range(w // 2):
                if row[x] > threshold:
                    left_points.append((x, y))
                    break

            # Rechter Rand: Suche von rechts
            for x in reversed(range(w // 2, w)):
                if row[x] > threshold:
                    right_points.append((x, y))
                    break

        # Debug-Bild vorbereiten
        debug_image = np.copy(image)

        # Linienpunkte einfärben
        for (x, y) in left_points:
            debug_image[y, x] = [255, 0, 0]  # Blau

        for (x, y) in right_points:
            debug_image[y, x] = [0, 255, 0]  # Grün

        self.debug_image = debug_image
        return left_points, right_points
