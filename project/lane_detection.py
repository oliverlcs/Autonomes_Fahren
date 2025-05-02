import numpy as np
from scipy import ndimage

class LaneDetection:
    def __init__(self):
        self.debug_image = None
        self.car_position = np.array([48, 64])

    def detect(self, image: np.ndarray):
        # --- Schritt 1: In Graustufen umwandeln ---
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB -> Grau: einfache Durchschnittsbildung
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        # --- Schritt 2: Sobel-Kantenfilter anwenden ---
        sobel_x = ndimage.sobel(gray, axis=1, mode='reflect')  # Kanten in x-Richtung
        sobel_y = ndimage.sobel(gray, axis=0, mode='reflect')  # Kanten in y-Richtung
        sobel_magnitude = np.hypot(sobel_x, sobel_y)

        # --- Schritt 2.5: Fahrzeugbereich maskieren ---
        vehicle_mask = np.zeros_like(sobel_magnitude, dtype=bool)

        # Das Auto ist auf self.car_position = [48, 64]
        # Auto Breite: 20 Pixel -> x-Range: [64-10, 64+10] = [54, 74]
        car_y, car_x = self.car_position

        vehicle_mask[
            car_y+16:car_y+32,         # ca. 16 Pixel hoch
            car_x-20:car_x-12 # 20 Pixel breit
        ] = True

        # Setze in der sobel_magnitude die Fahrzeugregion auf 0, damit keine Kanten dort erkannt werden
        sobel_magnitude[vehicle_mask] = 0


        # --- Schritt 3: Starke Kantenpunkte extrahieren ---
        threshold = np.percentile(sobel_magnitude, 93.5)  # Nur die stärksten 6.5% der Kanten nehmen
        edges = sobel_magnitude > threshold

        # --- Schritt 4: Kantenpunkte sammeln als Linienpunkte ---
        y_coords, x_coords = np.nonzero(edges)
        lines = np.column_stack((y_coords, x_coords))  # Form: [y, x] (wie erwartet)

        np.set_printoptions(threshold=np.inf)

        # Gruppiere die Linien basierend auf ihrer Nähe
        left_lines = []
        right_lines = []

        left_lines, right_lines = self.group_lines(lines)

        # Sortiere die Linienpunkte nach ihrer vertikalen Position (y-Wert)
        left_lines = np.array(sorted(left_lines, key=lambda point: point[0]))
        right_lines = np.array(sorted(right_lines, key=lambda point: point[0]))

        return left_lines, right_lines

    def group_lines(self, points):
        # Konvertiere zu NumPy-Array für Performance
        points = np.array(points)
        # Entferne alle Punkte mit y > 80
        points = points[points[:, 0] <= 80]

        if points.size == 0:
            return [], []

        right_points = np.array([])
        left_points = np.array([])

        # Finde alle Punkte auf Höhe y=80
        target_y = 80
        match = (points[:, 0] >= target_y) & (points[:, 0] <= target_y)
        # Filtere die Punkte, die auf der Zielhöhe 80 liegen
        matched_points = points[match]

        # Berechne den Durchschnitt der x-Werte
        while True:
            average_x = np.mean(matched_points[:, 1])

            # Prüfe, ob alle Punkte innerhalb von ±2 vom Durchschnitt liegen
            if not(np.all((matched_points[:, 1] >= average_x - 2) & (matched_points[:, 1] <= average_x + 2))):
                break  # Wenn die Bedingung erfüllt ist, beende die Schleife

            # Setze target_y um 2 niedriger und suche erneut
            target_y -= 2
            match = (points[:, 0] >= target_y) & (points[:, 0] <= target_y)
            matched_points = points[match]

            # Beende die Schleife, wenn keine Punkte mehr übrig sind
            if matched_points.size == 0:
                break

        # Sortiere die Punkte basierend auf ihrem x-Wert
        left_points = matched_points[matched_points[:, 1] < average_x]
        right_points = matched_points[matched_points[:, 1] >= average_x]

        border_points = self.find_border_points(points)

        # Sortieren der Grenzpunkte in left und right
        if np.all(border_points[:, 1] < border_points[:, 0]):  # Prüfe, ob x < y für alle Punkte in border_points
            avg_y = np.mean(border_points[:, 0])  # Durchschnitt aller y-Werte
            # Abfrage, ab alle punkte auf einem punkt  liegen
            if np.all((border_points[:, 0] >= avg_y - 2) & (border_points[:, 0] <= avg_y + 2)):  # Prüfe, ob alle y-Werte in avg_y ± 2 liegen
                right_points = np.vstack([right_points, border_points])  # Alle Punkte in right_points einsortieren

            else:

                for point in border_points:
                    if point[0] < avg_y:  # y kleiner als Durchschnitt
                        right_points = np.vstack([right_points, point])
                    else:  # y größer oder gleich Durchschnitt
                        left_points = np.vstack([left_points, point])
        else:
            if np.any(border_points[:, 1] <= 5):  # Prüfe, ob es Punkte mit x = 0 gibt
                points_with_x_5 = border_points[border_points[:, 1] <= 5]
                avg_y_for_x_5 = np.mean(points_with_x_5[:, 0])
                self.skip_max_x_point = False
                for point in border_points:
                    if point[1] <= 5:  # x = 0

                        if(point[0] < avg_y_for_x_5 - 5):
                            right_points = np.vstack([right_points, point])
                        else:
                            left_points = np.vstack([left_points, point])
                    else:  # x > 0
                        if not self.skip_max_x_point:
                            # Finde den Punkt mit dem größten x-Wert
                            max_x_point = border_points[np.argmax(border_points[:, 1])]

                            # Füge den Punkt zu right_points hinzu
                            right_points = np.vstack([right_points, max_x_point])
                        #right_points = np.vstack([right_points, point])
            else:
                avg_sum = np.mean(border_points[:, 0] + border_points[:, 1])  # Durchschnitt von x + y
                if np.all((border_points[:, 0] + border_points[:, 1] >= avg_sum - 2) &
                          (border_points[:, 0] + border_points[:, 1] <= avg_sum + 2)):  # Prüfe, ob alle Werte in avg_sum ± 2 liegen
                    if np.all(border_points[:, 0] <= 6):
                        pass
                    else:
                        left_points = np.vstack([left_points, border_points])  # Alle Punkte in left_points einsortieren
                elif np.any(border_points[:, 0] == 0) and np.any(border_points[:, 0] != 0):
                    max_sum_point = border_points[np.argmax(border_points[:, 0] + border_points[:, 1])]
                    right_points = np.vstack([right_points, max_sum_point])

                    # Finde den Punkt mit dem niedrigsten x + y und füge ihn zu left_points hinzu
                    min_sum_point = border_points[np.argmin(border_points[:, 0] + border_points[:, 1])]
                    left_points = np.vstack([left_points, min_sum_point])
                else:
                    for point in border_points:
                        if point[0] + point[1] > avg_sum:  # Summe größer als Durchschnitt
                            right_points = np.vstack([right_points, point])
                        else:  # Summe kleiner oder gleich Durchschnitt
                            left_points = np.vstack([left_points, point])

        #NeueStartpunkte erstellen durch durchqueren des randes (Neue Maske benötigt)
        for point in points[::-1]:
            dist_left = self.short_dist(point, left_points)

            dist_right = self.short_dist(point, right_points)

            if dist_left < dist_right:
                left_points = np.vstack([left_points, point])
            else:
                right_points = np.vstack([right_points, point])

        return left_points, right_points

    def find_border_points(self, points):
        # points: numpy array der Form (N, 2), wobei jede Zeile [y, x] ist
        y = points[:, 0]
        x = points[:, 1]

        # Masken für die drei Randlinien
        mask_left   = (x <= 5)  & (y >= 0) & (y < 80)
        mask_top    = (y == 0)  & (x >= 0) & (x <= 95)
        mask_right  = (x >= 90) & (y >= 0) & (y < 80)

        # Kombinierte Maske
        mask = mask_left | mask_top | mask_right

        return points[mask]

    def short_dist(self, punkt, punkte_array):
        if punkte_array.size == 0:  # Überprüfen, ob das Array leer ist
            return float('inf')  # Unendliche Distanz zurückgeben
        abstaende = np.linalg.norm(punkte_array - punkt, axis=1)
        return np.min(abstaende)
