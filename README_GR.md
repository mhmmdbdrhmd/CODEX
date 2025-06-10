<p align="right">
  <a href="README.md"><img src="https://img.shields.io/badge/EN-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README_GR.md"><img src="https://img.shields.io/badge/GR-Deutsch-red?style=for-the-badge" alt="Deutsch"></a>
</p>


# Winkelsensorfilterung und Extremumverfolgung

## Übersicht
Rohe Licht-Sensor-Arrays bieten einen lauten Einblick in die Ausrichtung des Traktors und des Traktors. Ziel ist es, diese kostengünstigen Messwerte zusammen mit der Fahrzeuggeschwindigkeit in ein stabiles Hitch-Winkel-Signal zu verwandeln. Ein hochwertiger Sensor liefert die Grundwahrheit, die zur Kalibrierung und Bewertung verwendet wird.
Dieser Workflow ist geplant, anstatt feste spezifische Dateinamen zu kodieren, auf ** jede neue Aufzeichnung ** (CSV -Datei), die Sie bereitstellen. Solange jeder CSV denselben Spaltenkonventionen (linke/rechte Sensorsummen, eine Referenzwinkelsäule, eine Geschwindigkeitsspalte und ein Zeitstempel) folgt, gelten dieselbe Vorverarbeitungs-, Filter-, Ausrichtung- und Bewertungsschritte.

Für aktuelle Metriken und Handlungen, die vom CI-Workflow erzeugt werden, siehe die
[Neueste Ergebnisse] (#10-latest-Reseults) Abschnitt unten.

## Hauptaufgabe

Unser Ziel ist es, eine klare Beispielpipeline zu liefern, die:
1. liest einen CSV aus "Aufzeichnungen/".
2. leitet eine Hitch-Winkel-Schätzung aus den Licht-Sensor-Arrays ab.
3. Reinigen Sie die Sensor- und Fahrzeuggeschwindigkeitsspalten.
4. führt einen Geschwindigkeitsbewusstsein von Kalman-Filter und optionalen Alternativen aus.
5. Ausrichtet das gefilterte Ergebnis mit den Metriken zur Referenz- und Protokollfehler.
6. Speichert reproduzierbare Diagramme in "Ergebnissen/" zur Inspektion.

Stellen Sie sich dieses Repository als Ausgangspunkt für Entwickler vor, um mit Filtermethoden zu experimentieren und Verbesserungen beizutragen.


---

## Verzeichnisstruktur

`` `
.
├── Readme.md ← Diese Datei (generische Anweisungen)
├── filter_analysis.py ← Python -Skript (Laden Sie CSV, Prozess, Diagramm, Metriken)
├── Aufzeichnungen/ ← Platzieren Sie hier ein oder mehrere CSV -Aufnahmen
│ ├── log_1618_76251.csv ← Beispielaufzeichnungsdatei
│ ├── log_1622_64296.csv
│ └── log_1626_93118.csv
└── Ergebnisse/ ← automatisch generierte Zahlen und `Performance.csv`
`` `

- ** filter_analysis.py **: Enthält alle Code zum Laden, Vorverarbeitung, Filterung, Ausrichtung und Leistungsbewertung. Es verarbeitet automatisch jede CSV -Aufzeichnung ** im Ordner "Aufnahmen/".
- ** Aufnahmen/**: Verzeichnis, in dem Sie jedes neue CSV -Protokoll platzieren. Jede Datei muss die folgenden Spalten mit den genauen deutschen Namen des Loggers enthalten:
- `Durchschnitt_L_SE` and `Durchschnitt_L_Be_SE` – left light-sensor sums,
- `Durchschnitt_R_SE` and `Durchschnitt_R_Be_SE` – right light-sensor sums,
- `Deichsel_angle` - Drawbar (Hitch) Winkelreferenz,
- `Geschwindigkeit` – vehicle speed,
- `esp time` - Zeitstempel in Millisekunden.
- Optional redundant sensors `Durchschnitt_L_SE2`, `Durchschnitt_L_Be_SE2`,
`Durchschnitt_R_SE2`, and `Durchschnitt_R_Be_SE2` (all zeros when not used).
- ** Ergebnisse/**: automatisch erstellt, wenn Sie das Skript ausführen. Hält PNG -Diagramme und eine Summary -Tabelle von `Performance.csv`.

---

## 1. Was wir erreichen wollen

1. ** Berechnen Sie einen rohen Hitch -Winkel -Proxy ** aus linken/rechten Sensorsummen.
2. ** Reinigen Sie die Zielreferenz ** Durch Interpolieren von Nulldropouts.
3. ** Vorverarbeitungsgeschwindigkeit ** Durch das Ausschneiden negativer Werte, die Interpolation von Nullen (nur zwischen der ersten und letzten Bewegung) und ein kleines Rolling -Durchschnitt, um Jitter zu entfernen.
4. ** Spikes ** aus dem rohen Proxy mit einem kausalen Hampelfilter (Fenster = 5) entfernen.
5. ** Wenden Sie in Echtzeit einen Geschwindigkeitsbewusstsein von Kalman -Filter ** (KF_INV) an, bei dem der Prozessverhältnis mit 1/Geschwindigkeit skaliert wird.
6. ** Erkennen Sie die lokalen Extrema (Peaks & Täler) ** im sauberen Referenzsignal, um zu quantifizieren, wie gut jeder Filter hohe/Tiefpunkte aufbewahrt.
7. ** Ausrichtung ** Jede gefilterte Ausgabe auf die Referenz durch Übereinstimmung mit extremen Indizes, so dass Peak/Valley -Fehler zu den richtigen Zeitpunkten gemessen werden können.
8. ** Berechnen Sie Metriken ** (RMSE, MAE, plus spezialisierte Extrema -Mae), um die Leistung zu bewerten.
9. ** Testen Sie optional alternative Filter ** (z. B. Savitzky -Golay und Hybrid kf_on_sg) auf dieselben Aufzeichnungen und vergleichen die Ergebnisse.

Da jede Aufzeichnungsdatei identische Spaltennamen/-muster verwendet, können Sie einfach neue CSVs in das Verzeichnis "Aufzeichnungen/" "fallen lassen und das Skript erneut ausführen.

---

## 2. Vorverarbeitungsschritte (einzelne Aufzeichnung)

Wenn Sie das Analyseskript auf einer neuen Aufzeichnung ausführen:

1. ** Laden Sie den CSV ** in einen Datenrahmen.
2. ** Identifizieren Sie Spalten ** über ihre Spaltennamenmuster:
- `Deichsel_angle` (Drawbar -Winkelreferenz).
- `Geschwindigkeit` (speed).
- `Durchschnitt_L_SE` and `Durchschnitt_L_Be_SE` (left light-sensor sums).
- `Durchschnitt_R_SE` and `Durchschnitt_R_Be_SE` (right light-sensor sums).
- Optional redundant sensors `Durchschnitt_L_SE2`, `Durchschnitt_L_Be_SE2`,
`Durchschnitt_R_SE2`, and `Durchschnitt_R_Be_SE2`.
3. ** Berechnen Sie den rohen Proxy **:
- Das Skript prüft automatisch, ob die redundanten Sensorspalten eine enthalten
Werte ungleich Null. Wenn dies der Fall ist, umfassen die linken und rechten Summen alle vier
Sensoren pro Seite und eine Konstante von "32" werden hinzugefügt. Wenn die redundanten Spalten
sind alle Null, die Datei wird angenommen, dass sie nur zwei Sensoren pro Seite und a verwenden
Konstante von `` 16``:
`` `
# Ohne redundante Sensoren
L = (l_se + l_be_se) + 16
R = (r_se + r_be_se) + 16

# Mit redundanten Sensoren
L = (l_se + l_se2 + l_be_se + l_be_se2) + 32
R = (r_se + r_se2 + r_be_se + r_be_se2) + 32
`` `
- Der Rohwinkel -Proxy ist dann
`` `
RAW = 90 + (((l + r) / 2) * (l - r)) / (l * r) * 100
`` `
*(Diese Übersetzung ist unvollständig. Weitere Abschnitte folgen.)*
