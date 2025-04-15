import numpy as np

class LaneDetection:
    def __init__(self):
        self.debug_image = None

    def detect(self, image: np.ndarray):
        # 1. Graustufenbild erzeugen
        grayscale = np.dot(image[..., :3], [0.299, 0.587, 0.114])

        # 2. X-Gradient berechnen (horizontaler Gradient)
        grad_x = np.abs(np.diff(grayscale, axis=1))  # Ableitung entlang der x-Achse
        grad_x = np.pad(grad_x, ((0, 0), (1, 0)), mode='constant')  # Padding, um Bildgröße zu halten

        # 3. Spaltenweise nach starkem Gradient suchen (nur ein Bereich der Bildhöhe ist relevant)
        h, w = grad_x.shape
        middle = h // 2

        left_edge = None
        right_edge = None

        # Mittelbereich analysieren
        threshold = 50  # Experimentell – muss ggf. angepasst werden
        scan_line = grad_x[middle]

        # Suche von links nach starkem Anstieg
        for x in range(w // 2):
            if scan_line[x] > threshold:
                left_edge = x
                break

        # Suche von rechts nach starkem Anstieg
        for x in reversed(range(w // 2, w)):
            if scan_line[x] > threshold:
                right_edge = x
                break

        # Debug-Bild: Zeichne detektierte Kanten ein (für Visualisierung)
        debug_image = np.copy(image)
        if left_edge is not None:
            debug_image[:, left_edge:left_edge+1] = [255, 0, 0]  # Blau für linken Rand
        if right_edge is not None:
            debug_image[:, right_edge:right_edge+1] = [0, 255, 0]  # Grün für rechten Rand

        self.debug_image = debug_image
        return left_edge, right_edge
