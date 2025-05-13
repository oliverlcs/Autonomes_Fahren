# Autonomes Fahren - Abschlussprojekt

Ziel des Projekts ist die Entwicklung einer Pipeline für einen selbstfahrenden Rennwagen mit Fokus auf einer effizienten und stabilen Fahrweise. Dabei setzen sich die Module der Pipeline aus Fahrbahnerkennung, Trajektorienplanung, Quer- und Längstregelung zusammen.

## Installation & verwendete Bibliotheken

- Python 3.12.x
- numpy (pip install numpy)
- scipy (pip install scipy)
- scikit-learn (pip install scikit-learn)

- Alle benötigten Bibliotheken installieren: pip install -r requirements.txt

- Projekt ausführen: python main.py

## Fahrbahnerkennung

## Trajektorienplanung

Die Trajektorienplanung basiert auf der Berechnung einer Mittellinie, die abhängig von der lokalen Kurvenkrümmung zu einer Ideallinie verschoben wird.

Ablauf der Planung:

1. Vorverarbeitung der Fahrbahnbegrenzungen:
   • Die linke und rechte Fahrbahnbegrenzung filtert die Methode `find_nearest_neighbors` und bringt die Punkte in eine geordnete Reihenfolge.
   • Bei komplexen Kurven, wie Haarnadelkurven, entsteht eine Ungleichheit in der Punktanzahl (z. B. linke Seite deutlich länger als rechte), was die Mittellinienberechnung verzerrt.
   • Das kompensiert die Methode `trim_by_pairing`, die paarweise linke und rechte Punkte innerhalb eines Schwellwertes zusammenführt.
2. Berechnung der Mittellinie:
   • Der Durchschnitt der gefilterten Punktepaare bestimmt Mittellinie.
   • Diese Linie wird erneut mit `find_nearest_neighbors` sortiert und anschließend geglättet.
3. Anpassung zur Ideallinie:
   • Die Mittelline wird abhängig von ihrer berechneten Krümmung seitlich verschoben, um eine optimale Fahrlinie (Ideallinie) zu ermöglichen.
   • Abschließend berechnet die Funktion `calculate_curvature_output` die Gesamtkrümmung einer Kurve als normierter Wert (zwischen 0 = gerade und 1 = scharfe Kurve).
   • Dieses Krümmungsmaß wird direkt als Eingabe für die Längsregelung verwendet, um die Geschwindigkeit der Streckengeometrie anzupassen.

**Hinweise**: Die Pfadplanung ist auf die Spurenerkennung angepasst, daher ist der Boolean `use_given_borders` standardmäßig auf `False` im File `test_path_planning` gestellt. Dieser kann auf `True` umgestellt werden, um die vorgegebenen Systemkanten zu verwenden.

---

Die Anpassung der Trajektorie basierend auf der Kurvenkrümmung ermöglicht eine fahrdynamisch günstigere Linie, wodurch höhere Kurvengeschwindigkeiten und ein insgesamt flüssigerer Fahrverlauf erreicht werden. Gleichzeitig dient die berechnete Krümmung als Grundlage für die Längsregelung, um die Geschwindigkeit vorausschauend an die Streckenführung anzupassen.

## Querregelung

Für die Querregelung wird ein hybrider Ansatz verwendet, der je nach Geschwindigkeit zwischen zwei Reglern wechselt:  
Bei niedrigen Geschwindigkeiten kommt der Stanley-Controller zum Einsatz, der präzise Spurverfolgung auch bei engen Kurven erlaubt.  
Bei höheren Geschwindigkeiten wird auf den Pure-Pursuit-Controller umgeschaltet, der durch vorausschauendes Steuern eine stabile Fahrdynamik bei schneller Fahrt ermöglicht.  
Diese Kombination erlaubt sowohl schnelles Beschleunigen auf Geraden als auch sicheres Folgen der Ideallinie in kurvenreichen Streckenabschnitten.

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
