# Datengenerator für Job-Shop-Scheduling

## Überblick
Die `data_generator.py` Datei erzeugt realistische Testdaten für Job-Shop-Scheduling-Probleme (JSP) in einer Fahrradproduktion. Sie generiert eine strukturierte JSON-Datei mit Jobs, Maschinen und Umrüstzeiten.

## Hauptfunktionalität

### `generate_jsp_data(num_jobs)`
- Erstellt eine definierte Anzahl von Fahrradproduktionsaufträgen (Jobs)
- Jeder Job besteht aus 5-8 Operationen mit realistischer Sequenz
- Berücksichtigt verschiedene Fahrradmodelle (Mountainbike, Racebike, Citybike, E-Bike)
- Generiert realistische Bearbeitungszeiten je nach Maschinentyp
- Erstellt Vorgängerbeziehungen zwischen Operationen
- Weist Prioritäten und Deadlines basierend auf Fahrradtyp zu

### Datenstruktur
- **Maschinen**: Rahmenfertigung (M1), CNC-Bearbeitung (M2), Endmontage (M3)
- **Umrüstzeiten**: Standard- und Materialwechselzeiten für jede Maschine
- **Jobs**: Enthält Operationen mit Bearbeitungszeiten, Materialien und Vorgängern
- **Prioritäten**: Höhere Werte für E-Bikes und Rennräder (7-10), niedrigere für andere (1-8)
- **Deadlines**: Berechnet aus Bearbeitungszeit, Priorität und zufälliger Variation

## Besonderheiten
- Realistische Produktionsreihenfolge (Rahmen → CNC → Endmontage)
- Materialwechsel mit entsprechenden Umrüstzeiten
- Prioritätsbasierte Deadlines (höhere Priorität = engere Deadline)
- Detaillierte Produktionsschritte für jede Maschinengruppe

## Anwendung
Die generierten Daten dienen als Eingabe für Scheduling-Algorithmen und Reinforcement Learning Modelle zur Optimierung der Produktionsplanung in der Fahrradherstellung.