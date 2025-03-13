# Gym Environment Module

## Überblick
Die `gym_environment.py` Datei implementiert eine OpenAI Gym-kompatible Umgebung für Job-Shop-Scheduling-Probleme (JSP). Sie ermöglicht das Training von Reinforcement Learning Agenten mit standardisierten Schnittstellen.

## Hauptkomponenten

### `JSPGymEnvironment` Klasse
- Erbt von `gym.Env` und implementiert die Gym-Schnittstelle
- Verwaltet den Zustand des JSP-Problems und die Interaktion mit dem Agenten

### Zustandsraum (Observation Space)
- Umfasst mehrere Aspekte des JSP-Problems:
  - Fortschritt jedes Jobs
  - Verfügbarkeitszeiten der Maschinen
  - Aktuelle Zeit
  - Prioritäten und Deadlines der Jobs
  - Materialien auf den Maschinen
  - Maske für gültige Aktionen

### Aktionsraum (Action Space)
- Diskrete Auswahl eines Jobs für die nächste Bearbeitung

### Belohnungsfunktion (Reward)
- Komplexe Belohnungsberechnung basierend auf:
  - Makespan-Minimierung
  - Prioritäten der Jobs
  - Einhaltung von Deadlines
  - Umrüstzeiten
  - Maschinenauslastung und -balance
  - Kritische Pfade

## Hauptfunktionalität
- **Initialisierung**: Lädt JSP-Daten und erstellt Zustandsräume
- **Reset**: Setzt die Umgebung auf den Anfangszustand zurück
- **Step**: Führt eine Aktion aus und berechnet den neuen Zustand und die Belohnung
- **Vorgängerprüfung**: Stellt sicher, dass Operationen erst nach ihren Vorgängern ausgeführt werden
- **Umrüstzeiten**: Berechnet Umrüstzeiten basierend auf Materialwechseln

## Besonderheiten
- Unterstützt komplexe JSP-Probleme mit Vorgängerbeziehungen
- Berücksichtigt Prioritäten, Deadlines und Materialwechsel
- Bietet eine detaillierte Belohnungsfunktion für mehrere Optimierungsziele
- Liefert umfangreiche Informationen für Debugging und Analyse