# FIFO Scheduler Module

## Überblick
Die `fifo_scheduler.py` Datei implementiert einen First-In-First-Out (FIFO) Scheduler für Job-Shop-Scheduling-Probleme (JSP). Der Scheduler berücksichtigt Prioritäten, Vorgängerbeziehungen, Deadlines und Umrüstzeiten.

## Hauptfunktionalität

### `fifo_schedule(jsp_data_path)`
- Lädt JSP-Daten aus einer JSON-Datei
- Erstellt einen Zeitplan nach dem FIFO-Prinzip mit Prioritätsberücksichtigung
- Berechnet den Makespan (Gesamtfertigstellungszeit)
- Überprüft die Einhaltung von Deadlines

## Ablauf
1. **Daten laden**: JSP-Daten werden aus einer JSON-Datei geladen
2. **Initialisierung**: Erstellen von Datenstrukturen für Jobs, Maschinen und Zeitplan
3. **Scheduling-Schleife**:
   - Jobs werden nach Priorität sortiert
   - Für jeden Job wird die nächste ausführbare Operation gesucht
   - Operationen werden eingeplant, wenn alle Vorgänger abgeschlossen sind
4. **Umrüstzeiten**: Berechnung basierend auf Materialwechseln
5. **Ergebnis**: Rückgabe des Zeitplans und des Makespans

## Besonderheiten
- Berücksichtigt komplexe Vorgängerbeziehungen zwischen Operationen
- Priorisiert Jobs nach ihrer Wichtigkeit
- Optimiert für Materialwechsel und Umrüstzeiten
- Erkennt zyklische Abhängigkeiten und warnt davor
- Überprüft die Einhaltung von Deadlines

## Anwendung
Der FIFO-Scheduler dient als Baseline-Vergleich für komplexere Scheduling-Algorithmen wie Reinforcement Learning und kann für einfache Scheduling-Probleme eingesetzt werden.