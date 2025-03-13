# JSP Graph Module

## Überblick
Die `jsp_graph.py` Datei erstellt und visualisiert einen gerichteten Graphen für Job-Shop-Scheduling-Probleme (JSP). Sie wandelt die JSP-Daten in eine Graphenstruktur um, die für die Analyse und Visualisierung des Problems verwendet werden kann.

## Hauptfunktionalität

### `create_jsp_graph(data)`
- Erstellt einen gerichteten Graphen (DiGraph) aus JSP-Daten
- Jede Operation wird als Knoten dargestellt
- Zwei Arten von Kanten werden erstellt:
  - **Konjunktive Kanten** (blau): Stellen die Reihenfolge der Operationen innerhalb eines Jobs dar
  - **Disjunktive Kanten** (rot): Stellen Konflikte zwischen Operationen dar, die dieselbe Maschine benötigen

### `visualize_jsp_graph(G, pos)`
- Visualisiert den JSP-Graphen mit NetworkX und Matplotlib
- Verwendet unterschiedliche Farben und Stile für die verschiedenen Kantentypen
- Speichert die Visualisierung als PNG-Datei mit Zeitstempel

## Besonderheiten
- Unterstützt Vorgängerbeziehungen zwischen Operationen
- Berücksichtigt Prioritäten und Deadlines von Jobs
- Enthält Informationen über Materialien und Bearbeitungszeiten
- Verwendet START- und END-Knoten zur Darstellung des Gesamtablaufs

## Anwendung
Die Graphendarstellung ermöglicht:
- Visualisierung der Abhängigkeiten zwischen Operationen
- Identifikation von Ressourcenkonflikten
- Grundlage für graphbasierte Optimierungsalgorithmen
- Analyse des kritischen Pfads im Scheduling-Problem