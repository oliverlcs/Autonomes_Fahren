# Zusätzliche Libraries

- sklearn (pip install scikit-learn) für das Path Planning

# Abschlussprojekt: Autonomer Rennwagen

## Spurerkennung

## Pfadplanung

**Gesamtkrümmungsberechnung**
Die Funktion **`calculate_curvature_output`** dient zur hochperformanten Abschätzung der Krümmung einer 2D-Linie. Sie berechnet die Gesamtwinkeländerung entlang einer gegebenen Punktfolge und normiert diese auf einen Wert zwischen 0 (gerade Linie) und 1 (maximale Krümmung, 180° Richtungsänderung). Die Funktion wird verwendet, um die Krümmung des Pfades zu bestimmen, was essenziell für die Geschwindigkeitsvorhersage und -regelung ist. Sie wurde aus der ursprünglichen Klasse der Längsregelung in die Pfadplanung übertragen, um den Aufbau von main nicht verändern zu müssen.

## Querregelung

## Längsregelung

**PID-Regler**
Werte P = 0,1; I = 0,0; D = 0,02
Die Werte wurden durch Testing ermittelt und auf eine optimale Längskontrolle angepasst. Durch den Wert I = 0,0 wurde der Regler zu einem PD-Regler "umgewandelt", da der integrale Fehler die Steuerung verfälscht hat. Dies kommt durch die langsame Beschleunigung des Fahrzeugs, wodurch der Integrale Fehler stark steigt und auf kurze Bremsanforderungen nicht mehr reagiert wird.

**Geschwindigkeitsvorhersage**
Die Geschwindigkeit wird aus Gesamtkrümmung der aktuell vorgegebenen Kurve bestimmt. Dazu wird die Krümmung als Wert zwischen 0 (0°) und 1 (>= 180°) als "curvature" von der Pfadplanung zurückgegeben. Sie wird durch die Funktion "calculate_curvature_output" bestimmt. Diese Funktion wurde aus der ursprünglichen Klasse der Längsregelung in die Pfadplanung verlegt, um den Aufbau der main nicht ändern zu müssen.
Mit dem Wert der Krümmung kann der Wert für die aktuell gewünschte Geschwindigkeit ausgegeben werden, der aus dem Maximum zwischen der Minimalgeschwindigkeit und der Multiplikation von Krümmung und Maximalgeschwindigkeit ermittelt wird.

**Geschwindigkeitskontrolle**
Mit dem PID-Regler wird ein Ausgabesignal erzeugt, dass falls positiv eine Beschleunigung und falls Negativ eine Entschleunigung bewirkt. Die Werte für Gas und Bremse sind gewichtet und durch Tests ermittelt worden. Dabei werden Werte für die Beschleunigung deutlich verstärkt um das Auto schnell wieder auf mögliche Geschwindigkeiten zu beschleunigen. Die Werte für die Entschleunigung werden deutlich reduziert, dass das Auto nicht immer bei Erkennung einer Kurve abrupt bremst, sondern gleichmäßig in die Kurve hinein bremst, um möglichst lang die Geschwindigkeit auszunutzen.
Die Werte der Beschleunigung werden bei Lenkeinschlägen, größer eines Grenzwertes stark reduziert um Ausbrechen zu vermeiden.
