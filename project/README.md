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

1. **Vorverarbeitung:**
   Das Bild wird zunächst in Graustufen umgewandelt. Anschließend werden alle Pixel mit y > 82 (unterer Bildrand) maskiert, um störende Einflüsse durch die Balken beim Lenken und Beschleunigen auf die Linienerkennung zu vermeiden.

2. **Kantenerkennung:**
   Ein Sobel-Kantenfilter wird angewendet, um die markanten Kanten im Bild hervorzuheben. Die Bereiche, die durch die Fahrzeugstruktur beeinflusst werden, werden dabei gezielt berücksichtigt. Kanten, die durch die Balken am unteren Bildrand entstehen, werden bereits vor der Filterung entfernt, da sie die Erkennung der Fahrbahnlinien negativ beeinflussen würden.

3. **Selektion starker Kanten:**
   Es werden ausschließlich die stärksten 6 % der erkannten Kantenpunkte ausgewählt. Dadurch wird sichergestellt, dass nur die relevantesten Linien für die weitere Verarbeitung genutzt werden und Störungen durch Kanten im Umfeld oder auf der Fahrbahn minimiert werden.

4. **Extraktion und Sortierung der Linienpunkte:**
   Die ausgewählten Kantenpunkte werden als Linienpunkte extrahiert und in einem zweidimensionalen Array (Form: [y ; x]) gespeichert. Dieses Array wird anschließend nach den y-Koordinaten (vertikale Bildachse) sortiert.

5. **Gruppierung der Linienpunkte:**
   Die erkannten Linienpunkte werden in rechte und linke Fahrbahnränder gruppiert, um sie für die Trajektorienplanung weiterzuverarbeiten.

   **Vorgehen bei der Gruppierung:**
   - Zunächst werden die Punkte am unteren Bildrand (y = 80) identifiziert und mithilfe eines Clustering-Verfahrens in Gruppen eingeteilt. Gibt es zwei Cluster, werden diese direkt als rechter und linker Fahrbahnrand zugeordnet. Bei mehr als zwei Clustern erfolgt die Zuordnung anhand des Durchschnitts der x-Koordinaten. Ist der Durchschnitt kleiner des Bildmittelpunkts (x = 48) handelt es sich um eine Linkskurve und die Clusters werden von links nach rechts einer Linkskurve zugeordnet (rechts, links, links, rechts). Ist der Durchschnitt größer handelt es sich um eine recht Kurve und die Zuordnung verläuft in anderer Richtung.

   - Anschließend werden die Punkte an den übrigen Bildrändern gesucht und mithilfe einer Verzweigungslogik den rechten und linken Linien zugeordnet. Die Punkte die als Randpunkte identifiziert werden werden später nicht in den Linien berücksichtigt, da sie in seltenen Randfällen nicht der wahren Zuordnung entsprechen, was wiederum keinen Einfluss auf die Zuordnung hat.

   - Zuletzt werden die verbleibenden, noch nicht zugeordneten Punkte von unten nach oben (hohe zu niedrigen y-Werten) durchlaufen. Für jeden Punkt wird berechnet, zu welchem der beiden Randlinien-Arrays (rechts oder links) der Abstand geringer ist. Der Punkt wird dann entsprechend zugeordnet, sodass die Arrays für die rechten und linken Fahrbahnlinien kontinuierlich wachsen.

6. **Rückgabe der Linien**
   Vor der Rückgabe werden die Koordinaten zweidimensionalen Linienarrays gedreht (Form [x ; y]), dass die Ausgabe mit der Eingabe der Trajektorienplanung übereinstimmt.

   **----------Umsetztung Optional----------**
   1. **Zufallsfarben**
   Es werden zusätzliche Farben zu mehr als 75% erkannt. Manche Farbkombinationen werden nicht erkannt, da sie zwar in Farbe gut zu unterscheiden sind, aber nach der Umwandlung in Graustufen sehr nah beieinander liegen. Dadurch werden die Linien in der Umgebung als stärkere Differenz erkannt und in rechts und links zugeordnet.
   2. **Stabile Zuordnung**
   Die Gruppierung der Linienpunkte erfolgt nicht nur durch einfache Schwellenwerte wie den Bildmittelpunkt, sondern nutzt Clustering-Algorithmen und eine spezielle Logik für Randpunkte. Dadurch werden auch in schwierigen Linienführungen die Fahrbahnbegrenzungen zuverlässig erkannt. Auch bei leichten Abweichungen von der Fahrbahn können die Linien zuverlässig zugeordnet werden. (siehe: Vorgehen bei der Gruppierung)


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
Die Zielgeschwindigkeit des Fahrzeugs wird dynamisch anhand der aktuellen Streckengeometrie bestimmt. Dazu berechnet die Funktion `calculate_curvature_output` aus der geplanten Trajektorie einen normierten Krümmungswert zwischen 0 (gerade Strecke) und 1 (maximale Kurve). Diese Funktion analysiert die Richtungsänderungen entlang der Trajektorie und gewichtet dabei auch S-Kurven entsprechend stärker, da S-Kurven im Grenzbereich zu Problemen führen können. Wenn das Auto in der ersten Kurve aufgrund Geschwindigkeiten nahe dem Grenzbereich zu Rutschen beginnt kann die Gegenkurve nur schwer korrekt geregelt werden.

Der berechnete Krümmungswert dient direkt als Eingabe für die Längsregelung:
- Bei geringer Krümmung (nahe 0) wird eine höhere Geschwindigkeit angestrebt, da die Strecke überwiegend gerade ist.
- Bei hoher Krümmung (nahe 1) wird die Geschwindigkeit reduziert, um eine sichere Kurvendurchfahrt zu gewährleisten.

Die Zielgeschwindigkeit ergibt sich dabei aus dem Maximum von Minimalgeschwindigkeit und dem Produkt aus Krümmungswert und Maximalgeschwindigkeit. So wird sichergestellt, dass das Fahrzeug stets an die aktuelle Streckensituation angepasst fährt und sowohl auf Geraden als auch in Kurven optimal beschleunigt oder abbremst.

**Geschwindigkeitskontrolle**
Mit dem PID-Regler wird ein Ausgabesignal erzeugt, dass falls positiv eine Beschleunigung und falls Negativ eine Entschleunigung bewirkt. Die Werte für Gas und Bremse sind gewichtet und durch Tests ermittelt worden. Dabei werden Werte für die Beschleunigung deutlich verstärkt um das Auto schnell wieder auf mögliche Geschwindigkeiten zu beschleunigen. Die Werte für die Entschleunigung werden deutlich reduziert, dass das Auto nicht immer bei Erkennung einer Kurve abrupt bremst, sondern gleichmäßig in die Kurve hinein bremst, um möglichst lang die Geschwindigkeit auszunutzen.
Die Werte der Beschleunigung werden bei Lenkeinschlägen, größer eines Grenzwertes Null gesetzt um Ausbrechen zu vermeiden.

   **----------Umsetztung Optional----------**
   Die Verbesserung der Längskontrolle wird durch unterschiedliche Heuristiken ermöglicht. In erster Linie wird durch die Betrachtung der Krümmung der Kurve durch `calculate_curvature_output` die Zielgeschwindigkeit aus einem Intervall zwischen der minimalen und der maximalen Geschwindigkeit berechnet. Zusätzlich zur Krümmung werden auch spezielle Streckenverläufe (S-Kurven) betrachtet, da sie eine besondere Regelung benötigen. Danach werden der Beschleunigungsparameter berechnet, wobei Gewichtungen von Gas und Bremse die Performance deutlich verbessert. Dadurch, dass teilweise schwarze Linien auf der Farbahn gezeigt werden, ist anzunehmen, dass die Längsregelung den Wagen nahe dem Grenzbereich fährt und somit die Performance star verbessert wurde.
