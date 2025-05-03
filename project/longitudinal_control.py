import numpy as np


class LongitudinalControl:
    def __init__(self, kp=0.1, ki=0.01, kd=0.05):
        # PID-Regler-Parameter
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # PID-Regler-Zustände
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

        # Gas und Bremse berechnen
        gas = max(0, output)  # Nur positive Werte für Gas
        brake = max(0, -output)  # Nur negative Werte für Bremse

        # Verhindere starkes Gasgeben und Lenken gleichzeitig
        if abs(steering_angle) > 0.5:  # Beispielschwelle für starkes Lenken
            gas *= 0.5

        return gas, brake

    def predict_target_speed(self, curvature):
        """
        Berechnet die Zielgeschwindigkeit basierend auf der Straßenkrümmung.

        Args:
            curvature (np.ndarray): Array von Krümmungswerten entlang der optimalen Linie.

        Returns:
            float: Zielgeschwindigkeit.
        """
        # Zielgeschwindigkeit basierend auf der Krümmung
        max_speed = 100  # Maximale Geschwindigkeit (z. B. 100 km/h)
        min_speed = 30   # Minimale Geschwindigkeit (z. B. 30 km/h)

        # Falls curvature ein Array ist, berechne eine Metrik (z. B. den maximalen Krümmungswert)
        if isinstance(curvature, (np.ndarray, list)):
            if len(curvature) == 0:
                curvature_value = 0  # Standardwert, wenn keine Krümmung vorhanden ist
            else:
                curvature_value = np.max(np.abs(curvature))  # Maximaler Krümmungswert
        else:
            curvature_value = abs(curvature)  # Einzelner Krümmungswert

        # Berechne die Zielgeschwindigkeit basierend auf der Krümmung
        target_speed = max(min_speed, max_speed * (1 - curvature_value))
        return target_speed
