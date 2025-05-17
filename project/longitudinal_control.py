import numpy as np


# Unterdrücken von RuntimeWarnings, die beim Streckenwechsel auftreten können, haben keinen EInfluss auf die Funktionalität
import warnings
# Unterdrückt RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

        # Zielgeschwindigkeit basierend auf der Krümmung
        self.max_speed = 100  # Maximale Geschwindigkeit (z. B. 100 km/h)
        self.min_speed = 40   # Minimale Geschwindigkeit (z. B. 45 km/h)

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
        brake_weight = 0.16  # Bremse bleibt unverändert

        # Gas und Bremse berechnen
        gas = max(0, output * gas_weight)  # Nur positive Werte für Gas
        brake = max(0, -output * brake_weight)  # Nur negative Werte für Bremse

        # Verhindere starkes Gasgeben und Lenken gleichzeitig
        if abs(steering_angle) > 0.08 and current_speed > 25:  # Beispielschwelle für starkes Lenken
            gas = 0.2

        return gas, brake

    def predict_target_speed(self, curvature):
        """
        Berechnet die Zielgeschwindigkeit basierend auf der Straßenkrümmung.

        Args:
            curvature (float): Krümmungswert der Kurve.

        Returns:
            float: Zielgeschwindigkeit.
        """

        # Berechne die Zielgeschwindigkeit basierend auf der Krümmung
        target_speed = max(self.min_speed, self.max_speed * (1 - curvature))

        return target_speed

    #Funktion übertragen in pathplanning, um gegebene Objektübergaben beizubehalten
    # def curvature_score(self, points: np.ndarray) -> float:
    #     """Calculates the total curvature of a 2D curve based on the change in direction between points for longitudinal control.
        # """"
        # This function computes the total change in direction between consecutive points along
        # a 2D curve, normalizes it, and returns the curvature value between 0.0 and 1.0.
        # A straight line will return a curvature of 0.0, and a maximal curvature (180° change in direction)
        # will return a value of 1.0. S-curves (with direction changes) are weighted higher.

        # Args:
        #     points (np.ndarray): An Nx2 array of 2D coordinates representing points along the curve.

        # Returns:
        #     float: A curvature value between 0.0 and 1.0, representing the total curvature of the curve.
        # """
        # if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
        #     return 0.0

        # if points.shape[0] < 3:
        #     return 0.0

        # # Calculate direction vectors between consecutive points
        # d = np.diff(points, axis=0)

        # # Normalize the direction vectors
        # norm = np.linalg.norm(d, axis=1)
        # d_unit = d / norm[:, None]

        # # Compute the dot product of adjacent unit vectors to get cos(θ)
        # dot = np.einsum('ij,ij->i', d_unit[:-1], d_unit[1:])
        # angles = np.arccos(np.clip(dot, -1.0, 1.0))  # Numerically stable calculation

        # # Bestimme das Vorzeichen der Kreuzprodukte (gibt die Drehrichtung an)
        # cross = d_unit[:-1, 0] * d_unit[1:, 1] - d_unit[:-1, 1] * d_unit[1:, 0]
        # sign_changes = np.sum(np.diff(np.sign(cross)) != 0)

        # # Compute the total change in angle
        # total_angle = angles.sum()

        # # Normalize by the maximum possible change (π)
        # curvature = min(1.0, total_angle / np.pi)

        # # Erhöhe die Krümmung, falls eine S-Kurve erkannt wurde (mind. 1 Vorzeichenwechsel)
        # if sign_changes > 0:
        #     curvature = min(1.0, curvature * (1.2 + 0.2 * sign_changes))  # z.B. 20% mehr pro Wechsel

        # return curvature
