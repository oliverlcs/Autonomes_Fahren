import numpy as np
from scipy.ndimage import convolve

class LaneDetection:
    def __init__(self):
        self.debug_image = None

    def detect(self, image: np.ndarray):
        # Konvertiere das Bild von RGB nach HSV (manuell mit NumPy)
        image = image.astype(np.float32) / 255.0  # Normiere auf [0, 1]
        r, g, b = image[..., 0], image[..., 1], image[..., 2]
        max_val = np.max(image, axis=-1)
        min_val = np.min(image, axis=-1)
        delta = max_val - min_val

        # Berechne den Farbton (Hue)
        hue = np.zeros_like(max_val)
        mask = delta > 0
        hue[mask & (max_val == r)] = (60 * ((g - b) / delta) % 360)[mask & (max_val == r)]
        hue[mask & (max_val == g)] = (60 * ((b - r) / delta) + 120)[mask & (max_val == g)]
        hue[mask & (max_val == b)] = (60 * ((r - g) / delta) + 240)[mask & (max_val == b)]

        # Berechne die Sättigung (Saturation)
        saturation = np.zeros_like(max_val)
        saturation[max_val > 0] = (delta / max_val)[max_val > 0]

        # Der Wert (Value) ist einfach der Maximalwert
        value = max_val

        # Erstelle das HSV-Bild
        hsv = np.stack([hue, saturation, value], axis=-1)

        # Definiere eine Schwelle für "graue Straße"
        lower_gray = np.array([0, 0, 0.2])  # Angepasst für HSV-Werte
        upper_gray = np.array([360, 0.2, 0.8])  # Angepasst für HSV-Werte

        # Erstelle die Maske für graue Straßen
        road_mask = (
            (hsv[..., 0] >= lower_gray[0]) & (hsv[..., 0] <= upper_gray[0]) &
            (hsv[..., 1] >= lower_gray[1]) & (hsv[..., 1] <= upper_gray[1]) &
            (hsv[..., 2] >= lower_gray[2]) & (hsv[..., 2] <= upper_gray[2])
        ).astype(np.uint8)

        # Maskiere den Bereich des Autos
        mask = np.ones(image.shape[:2], dtype=np.uint8)  # Erstelle eine Maske mit Einsen
        car_position = np.array([48, 64])  # Position des Autos (x, y)
        car_mask_width = 40   # Breite der Auto-Maske
        mask[car_position[1]:, car_position[0] - car_mask_width // 2:car_position[0] + car_mask_width // 2] = 0

        # Kombiniere die Straßenmaske mit der Auto-Maske
        road_mask = road_mask * mask

        # Wende die Maske auf das Bild an
        masked_image = image[..., 0] * road_mask  # Verwende nur einen Kanal für Kanten

        # Wende Sobel-Filter an, um Kanten zu erkennen
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        edges_x = convolve(masked_image, sobel_x)
        edges_y = convolve(masked_image, sobel_y)
        edges = np.sqrt(edges_x**2 + edges_y**2)

        # Normalisiere die Kanten und erstelle ein binäres Bild
        edges = (edges / edges.max() * 255).astype(np.uint8)
        edge_threshold = 50
        edge_binary = (edges > edge_threshold).astype(np.uint8) * 255

        # Optional: Debug-Image aktualisieren
        self.debug_image = edge_binary

        # Linien extrahieren
        lines = np.column_stack(np.where(edge_binary > 0))

        # Gruppiere die Linien basierend auf ihrer Nähe
        left_lines = []
        right_lines = []
        image_center = image.shape[1] // 2

        for y, x in lines:
            if x < image_center:
                left_lines.append((y, x))
            else:
                right_lines.append((y, x))

        # Sortiere die Linienpunkte nach ihrer vertikalen Position (y-Wert)
        left_lines = np.array(sorted(left_lines, key=lambda point: point[0]))
        right_lines = np.array(sorted(right_lines, key=lambda point: point[0]))

        return left_lines, right_lines
