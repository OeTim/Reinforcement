# Visualizer Module

## Überblick
Die `visualizer.py` Datei implementiert eine Funktion zur Visualisierung von Job-Shop-Scheduling-Problemen (JSP) als Gantt-Chart. Sie stellt den Ablaufplan grafisch dar und hebt wichtige Aspekte wie Deadlines, Prioritäten und Umrüstzeiten hervor.

## Hauptfunktionalität
Die Funktion `visualize_schedule()` nimmt zwei Parameter entgegen:
- `jsp_data_path`: Pfad zur JSON-Datei mit JSP-Daten
- `actions`: Liste von Job-Indizes, die die Reihenfolge der Ausführung bestimmen

## Ablauf
1. **Daten laden**: Lädt JSP-Daten aus einer JSON-Datei
2. **Schedule erstellen**: 
   - Verarbeitet die Aktionen in der angegebenen Reihenfolge
   - Berücksichtigt Vorgängerbeziehungen zwischen Operationen
   - Berechnet Umrüstzeiten basierend auf Materialwechseln
   - Erstellt einen vollständigen Zeitplan mit Start- und Endzeiten

3. **Visualisierung**:
   - Erstellt ein Gantt-Chart mit Matplotlib
   - Zeigt Operationen als farbige Balken auf Maschinen-Zeitlinien
   - Stellt Umrüstzeiten als schraffierte Bereiche dar
   - Markiert Deadlines und zeigt den Makespan (Gesamtdauer) an
   - Verwendet unterschiedliche Farben für verschiedene Jobs

4. **Ergebnisse**:
   - Speichert das Gantt-Chart als PNG-Datei
   - Gibt Warnungen aus, wenn Deadlines nicht eingehalten wurden
   - Gibt den erstellten Schedule und den Makespan zurück

## Besonderheiten
- Berücksichtigt Prioritäten der Jobs
- Berechnet Umrüstzeiten basierend auf Materialwechseln
- Visualisiert Deadlines und deren Einhaltung
- Unterstützt Vorgängerbeziehungen zwischen Operationen