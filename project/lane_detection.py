import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_dilation, binary_erosion

class LaneDetection:
    def __init__(self):
        """
        Initialisiert die LaneDetection-Klasse.

        Attributes:
            debug_image (None): Debug-Bild zur Visualisierung.
            car_position (np.ndarray): Position des Fahrzeugs im Bild.
        """
        self.debug_image = None
        self.car_position = np.array([48, 64])

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Erkennt Fahrspuren im gegebenen Bild.

        Args:
            image (np.ndarray): Eingabebild als 2D- oder 3D-Array.

        Returns:
            tuple: Zwei Arrays mit Punkten der linken und rechten Fahrspur.
        """
        # --- Schritt 1: In Graustufen umwandeln ---
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB -> Grau: einfache Durchschnittsbildung
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        # --- Schritt 1.5: Pixel mit y > 80 entfernen ---
        height, width = gray.shape
        mask = np.arange(height)[:, None] <= 82  # Maske für y <= 82
        mask = np.tile(mask, (1, width))  # Erweitere die Maske auf die gleiche Breite wie das Bild
        gray[~mask] = 0  # Setze Pixel mit y > 80 auf 0

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
            car_y+16:car_y+32,         # 16 Pixel hoch
            car_x-20:car_x-12          # 8 Pixel breit
        ] = True

        # Setze in der sobel_magnitude die Fahrzeugregion auf 0, damit keine Kanten dort erkannt werden
        sobel_magnitude[vehicle_mask] = 0


        # --- Schritt 3: Starke Kantenpunkte extrahieren ---
        threshold = np.percentile(sobel_magnitude, 94)  # Nur die stärksten 6% der Kanten nehmen
        edges = sobel_magnitude > threshold

        # --- Schritt 4: Kantenpunkte sammeln als Linienpunkte ---
        y_coords, x_coords = np.nonzero(edges)
        lines = np.column_stack((y_coords, x_coords))  # Form: [y, x]

        # Ausgabe der Linienpunkte für Debugging
        np.set_printoptions(threshold=np.inf)

        # --- Schritt 5: Gruppiere die Linien basierend auf ihrer Nähe
        left_lines = []
        right_lines = []

        left_lines, right_lines = self.group_lines(lines)

        # Sortiere die Linienpunkte nach ihrer vertikalen Position (y-Wert)
        left_lines = np.array(sorted(left_lines, key=lambda point: point[0]))
        right_lines = np.array(sorted(right_lines, key=lambda point: point[0]))

        if len(left_lines) == 0 or len(right_lines) == 0:
            return [], []
        left_lines = left_lines[::, ::-1]
        right_lines = right_lines[::, ::-1]

        return left_lines, right_lines

    def group_lines(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Gruppiert Linienpunkte basierend auf ihrer Nähe.

        Args:
            points (np.ndarray): Array mit Punkten der Form [y, x].

        Returns:
            tuple: Arrays mit Punkten der linken und rechten Fahrspur.
        """
        # Konvertiere zu NumPy-Array für Performance
        points = np.array(points)
        # Entferne alle Punkte mit y > 80
        points = points[points[:, 0] <= 80]

        if points.size == 0:
            return [], []

        left_points = np.empty((0, 2), dtype=int)  # Leeres Array mit 2 Spalten
        right_points = np.empty((0, 2), dtype=int)  # Leeres Array mit 2 Spalten

        # Finde alle Punkte auf Höhe y=80
        target_y = 80
        match = (points[:, 0] >= target_y) & (points[:, 0] <= target_y)
        # Filtere die Punkte, die auf der Zielhöhe 80 liegen
        matched_points = points[match]
        # Sortiere die matched_points basierend auf den x-Werten (zweite Spalte)
        matched_points = matched_points[np.argsort(matched_points[:, 1])]

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

        #Punkte auf der unteren Grenze zuordnen
        tolerance = 4  # Maximale Toleranz für den x-Abstand
        flag =False
        if average_x < 48:
            # Rückwärts durchlaufen
            clusters = self.cluster_points(matched_points[::-1], tolerance)
            toggle = True
            for cluster in clusters:
                if toggle:
                    right_points = np.vstack((right_points, cluster))
                    toggle = not toggle
                else:
                    left_points = np.vstack((left_points, cluster))
                    if flag == False:
                        flag = True
                    #abfrage, für den größten abstand zwischen punkten in clusters
                    else:
                        toggle = not toggle

        else:
            # Vorwärts durchlaufen
            clusters = self.cluster_points(matched_points, tolerance)
            toggle = True
            for cluster in clusters:
                if toggle:
                    left_points = np.vstack((left_points, cluster))
                    toggle = not toggle
                else:
                    right_points = np.vstack((right_points, cluster))
                    if flag == False:
                        flag = True
                    else:
                        toggle = not toggle

        border_points = self.find_border_points(points)

        # Sortieren der Grenzpunkte in left und right
        if np.all(border_points[:, 1] < border_points[:, 0]):  # Prüfe, ob x < y für alle Punkte in border_points
            avg_y = np.mean(border_points[:, 0])  # Durchschnitt aller y-Werte
            # Abfrage, ab alle punkte auf einem punkt  liegen
            if np.all((border_points[:, 0] >= avg_y - 7) & (border_points[:, 0] <= avg_y + 7)):  # Prüfe, ob alle y-Werte in avg_y ± 2 liegen
                right_points = np.vstack([right_points, border_points])  # Alle Punkte in right_points einsortieren

            else:
                if len(clusters) > 2:
                    #entferne alle werte mit y >78 aus borderpoints
                    border_points = border_points[border_points[:, 0] <= 78]
                for point in border_points:
                    if point[0] < avg_y:  # y kleiner als Durchschnitt
                        right_points = np.vstack([right_points, point])
                    else:  # y größer oder gleich Durchschnitt
                        left_points = np.vstack([left_points, point])
        else:
            if np.any(border_points[:, 1] <= 1):  # Prüfe, ob es Punkte mit x = 0 gibt
                points_with_x_1 = border_points[border_points[:, 1] <= 1]
                avg_y_for_x_1 = np.mean(points_with_x_1[:, 0])
                for point in border_points:
                    if point[1] <= 1:  # x = 0
                        if len(clusters) > 2:
                            if point[0] < 68:
                                right_points = np.vstack([right_points, point])
                        else:
                            if(point[0] < avg_y_for_x_1 - 8):
                                right_points = np.vstack([right_points, point])
                            else:
                                left_points = np.vstack([left_points, point])
                    else:  # x > 0
                        # Finde den Punkt mit dem größten x-Wert
                        max_x_point = border_points[np.argmax(border_points[:, 1])]

                        # Füge den Punkt zu right_points hinzu
                        right_points = np.vstack([right_points, max_x_point])

            else:
                avg_sum = np.mean(border_points[:, 0] + border_points[:, 1])  # Durchschnitt von x + y
                if np.all((border_points[:, 0] + border_points[:, 1] >= avg_sum - 7) &
                          (border_points[:, 0] + border_points[:, 1] <= avg_sum + 7)):  # Prüfe, ob alle Werte in avg_sum ± 2 liegen
                    if np.all(border_points[:, 0] <= 6):
                        pass
                    else:
                        left_points = np.vstack([left_points, border_points])  # Alle Punkte in left_points einsortieren
                elif np.any(border_points[:, 0] == 0) and np.any(border_points[:, 0] != 0):
                    if len(clusters) > 2:
                        # Entferne alle Punkte mit y > 68 und y < 10 aus border_points
                        border_points = border_points[(border_points[:, 0] <= 68) & (border_points[:, 0] >= 10)]
                        left_points = np.vstack([left_points, border_points])
                    else:
                        max_sum_point = border_points[np.argmax(border_points[:, 0] + border_points[:, 1])]
                        right_points = np.vstack([right_points, max_sum_point])

                        # Finde den Punkt mit dem niedrigsten x + y und füge ihn zu left_points hinzu
                        min_sum_point = border_points[np.argmin(border_points[:, 0] + border_points[:, 1])]
                        left_points = np.vstack([left_points, min_sum_point])
                else:
                    for point in border_points:
                        if len(clusters) > 2:
                            left_points = np.vstack([left_points, point])
                        elif point[0] + point[1] > avg_sum:  # Summe größer als Durchschnitt
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

        #Maske um falsch zugeordnete Borderpoints zu entfernen
        left_points = left_points[(left_points[:, 1] > 2) & (left_points[:, 1] < 93) & (left_points[:, 0] >= 2)]
        right_points = right_points[(right_points[:, 1] > 2) & (right_points[:, 1] < 93) & (right_points[:, 0] >= 2)]

        return left_points, right_points

    def find_border_points(self, points: np.ndarray) -> np.ndarray:
        """
        Findet Punkte entlang der Bildränder.

        Args:
            points (np.ndarray): Array mit Punkten der Form [y, x].

        Returns:
            np.ndarray: Punkte, die sich entlang der Bildränder befinden.
        """
        # points: numpy array der Form (N, 2), wobei jede Zeile [y, x] ist
        y = points[:, 0]
        x = points[:, 1]

        # Masken für die drei Randlinien
        mask_left   = (x <= 2)  & (y >= 0) & (y < 80)
        mask_top    = (y == 0)  & (x >= 0) & (x <= 95)
        mask_right  = (x >= 93) & (y >= 0) & (y < 80)

        # Kombinierte Maske
        mask = mask_left | mask_top | mask_right

        return points[mask]

    def short_dist(self, punkt: np.ndarray, punkte_array: np.ndarray) -> float:
        """
        Berechnet die kürzeste Distanz zwischen einem Punkt und einem Array von Punkten.

        Args:
            punkt (np.ndarray): Einzelner Punkt [y, x].
            punkte_array (np.ndarray): Array von Punkten.

        Returns:
            float: Kürzeste Distanz.
        """
        if punkte_array.size == 0:  # Überprüfen, ob das Array leer ist
            return float('inf')  # Unendliche Distanz zurückgeben
        abstaende = np.linalg.norm(punkte_array - punkt, axis=1)
        return np.min(abstaende)

    def cluster_points(self, points: np.ndarray, tolerance: int = 2) -> list[np.ndarray]:
        """
        Gruppiert Punkte in Cluster basierend auf einer Toleranz.

        Args:
            points (np.ndarray): Array von Punkten der Form [y, x].
            tolerance (int): Maximale Toleranz für den Abstand zwischen Punkten.

        Returns:
            list: Liste von Clustern, wobei jedes Cluster ein Array von Punkten ist.
        """
        if len(points) == 0:
            return []

        clusters = []
        current_cluster = [points[0]]

        for i in range(1, len(points)):
            if abs(points[i][1] - current_cluster[-1][1]) <= tolerance:
                current_cluster.append(points[i])
            else:
                clusters.append(np.array(current_cluster))
                current_cluster = [points[i]]
        clusters.append(np.array(current_cluster))
        return clusters