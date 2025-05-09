import numpy as np


class LongitudinalControl:
    def __init__(self, kp=0.1, ki=0.0, kd=0.02):
        """
        Initialisiert die LongitudinalControl-Klasse mit den PID-Regler-Parametern.

        Args:
            kp (float): Proportionaler Verstärkungsfaktor (Standard: 0.1).
            ki (float): Integraler Verstärkungsfaktor (Standard: 0.0).
            kd (float): Differentieller Verstärkungsfaktor (Standard: 0.02).

        Attributes:
            kp (float): Proportionaler Verstärkungsfaktor.
            ki (float): Integraler Verstärkungsfaktor.
            kd (float): Differentieller Verstärkungsfaktor.
            previous_error (float): Fehlerwert der vorherigen Iteration.
            integral (float): Akkumulierter Fehler für den integralen Anteil.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.previous_error = 0
        self.integral = 0

    def control(self, current_speed, target_speed, steering_angle):
        """
        Berechnet Gas und Bremse basierend auf der aktuellen und Zielgeschwindigkeit.

        Args:
            current_speed (float): Aktuelle Geschwindigkeit des Fahrzeugs.
            target_speed (float): Zielgeschwindigkeit basierend auf der Krümmung.
            steering_angle (float): Lenkwinkel des Fahrzeugs.

        Returns:
            tuple: (gas, brake), Werte zwischen 0 und 1.
        """
        # Berechne den Fehler
        error = target_speed - current_speed

        # PID-Regler
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error

        # Gewichtung für Gas und Bremse
        gas_weight = 4.0  # Verstärkt die Gasreaktion
        brake_weight = 0.08  # Bremse bleibt unverändert


        # Gas und Bremse berechnen
        gas = max(0, output * gas_weight)  # Nur positive Werte für Gas
        brake = max(0, -output * brake_weight)  # Nur negative Werte für Bremse

        # Verhindere starkes Gasgeben und Lenken gleichzeitig
        if abs(steering_angle) > 0.1:  # Beispielschwelle für starkes Lenken
            gas *= 0.01

        return gas, brake

    def predict_target_speed(self, curvature):
        """
        Berechnet die Zielgeschwindigkeit basierend auf der Straßenkrümmung.

        Args:
            curvature (float): Krümmungswert der Kurve.

        Returns:
            float: Zielgeschwindigkeit.
        """
        # Zielgeschwindigkeit basierend auf der Krümmung
        max_speed = 100  # Maximale Geschwindigkeit (z. B. 100 km/h)
        min_speed = 45   # Minimale Geschwindigkeit (z. B. 45 km/h)

        # Berechne die Zielgeschwindigkeit basierend auf der Krümmung
        target_speed = max(min_speed, max_speed * (1 - curvature))

        return target_speed

    #Funktion übertragen in pathplanning, um gegebene Objektübergaben beizubehalten
    # def curvature_score(self, points: np.ndarray) -> float:
    #     """
    #     Hochperformante Krümmungsabschätzung einer 2D-Linie.
    #     0 = Gerade, 1 = maximale Krümmung (180° Richtungsänderung).

    #     Args:
    #         points (np.ndarray): Nx2-Array mit 2D-Koordinaten.

    #     Returns:
    #         float: Krümmung zwischen 0.0 und 1.0
    #     """
    #     if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
    #         return 0.0

    #     if points.shape[0] < 3:
    #         return 0.0

    #     # Richtungsvektoren berechnen
    #     d = np.diff(points, axis=0)

    #     # Normalisieren
    #     norm = np.linalg.norm(d, axis=1)
    #     d_unit = d / norm[:, None]

    #     # Skalarprodukt benachbarter Einheitsvektoren → cos(θ)
    #     dot = np.einsum('ij,ij->i', d_unit[:-1], d_unit[1:])
    #     angles = np.arccos(np.clip(dot, -1.0, 1.0))  # numerisch stabil

    #     # Gesamtwinkeländerung
    #     total_angle = angles.sum()

    #     # Normierung auf maximalen möglichen Wert (pi)
    #     return min(1.0, total_angle / np.pi)
