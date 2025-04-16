import numpy as np
from scipy.ndimage import uniform_filter

class LaneDetection:
    def __init__(self):
        self.debug_image = None

    def detect(self, image: np.ndarray):
        # Konvertiere das Bild in Graustufen
        grayscale = np.dot(image[..., :3], [0.299, 0.587, 0.114])

        # Binärer Filter: Schwellenwert setzen
        threshold = 120  # Schwellenwert für die Trennung
        binary_mask = (grayscale > threshold).astype(np.uint8) * 255

        # Glättung der binären Maske
        smoothed_mask = uniform_filter(binary_mask, size=5)

        # X- und Y-Gradienten berechnen
        grad_x = np.abs(np.diff(smoothed_mask, axis=1))  # horizontal
        grad_x = np.pad(grad_x, ((0, 0), (1, 0)), mode='constant')

        grad_y = np.abs(np.diff(smoothed_mask, axis=0))  # vertikal
        grad_y = np.pad(grad_y, ((1, 0), (0, 0)), mode='constant')

        # Kantenstärke kombinieren
        edge_strength = grad_x + grad_y

        h, w = edge_strength.shape
        edge_threshold = 50  # Reduzierter Schwellenwert für Kantenstärke

        left_points = []
        right_points = []

        for y in range(h // 3, h - 5):  # nur unteren Bildbereich analysieren
            row = edge_strength[y]

            # Linker Rand: Suche von links
            found_left = False
            for x in range(w // 2):
                if row[x] > edge_threshold:
                    left_points.append((x, y))
                    found_left = True
                    break

            # Falls kein Punkt gefunden wurde, suche weiter rechts
            if not found_left:
                for x in range(w // 2, w // 2 + 20):  # Erweiterter Bereich
                    if row[x] > edge_threshold:
                        left_points.append((x, y))
                        break

            # Rechter Rand: Suche von rechts
            for x in reversed(range(w // 2, w)):
                if row[x] > edge_threshold:
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
