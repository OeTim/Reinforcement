# Job-Shop Scheduling Befehle

Diese Dokumentation erklärt die verschiedenen Befehle und Modi, die in diesem Job-Shop Scheduling Projekt verfügbar sind.

## Grundlegende Ausführung

Das Hauptprogramm `main.py` kann mit verschiedenen Modi ausgeführt werden:

```bash
python main.py --mode [MODE] --data [DATEI] --episodes [ANZAHL]
```

## Verfügbare Modi

### 1. Graph-Modus

```bash
python main.py --mode graph
```

**Was dieser Befehl macht:**
- Lädt die JSP-Daten aus der angegebenen Datei (Standard: `data.json`)
- Erstellt einen Graphen, der das Job-Shop Scheduling Problem darstellt
  - Knoten repräsentieren Operationen
  - Konjunktive Kanten (durchgezogen) zeigen die Reihenfolge der Operationen innerhalb eines Jobs
  - Disjunktive Kanten (gestrichelt) zeigen Ressourcenkonflikte zwischen Operationen auf derselben Maschine
- Visualisiert den Graphen mit einer übersichtlichen Darstellung
- Hilft dabei, die Struktur und Abhängigkeiten des Problems zu verstehen

### 2. FIFO-Modus

```bash
python main.py --mode fifo
```

**Was dieser Befehl macht:**
- Führt den FIFO-Scheduler (First-In-First-Out) auf den JSP-Daten aus
- Berücksichtigt dabei:
  - Prioritäten der Jobs
  - Vorgängerbeziehungen zwischen Operationen
  - Umrüstzeiten der Maschinen
  - Materialwechsel
- Berechnet einen vollständigen Schedule mit Start- und Endzeiten für jede Operation
- Gibt den Makespan (Gesamtbearbeitungszeit) aus
- Warnt, wenn Deadlines nicht eingehalten werden können
- Visualisiert den erzeugten Schedule als Gantt-Chart

### 3. PPO-Modus

```bash
python main.py --mode ppo --episodes 200
```

**Was dieser Befehl macht:**
- Trainiert einen PPO-Agenten (Proximal Policy Optimization) mit Reinforcement Learning
- Verwendet die angegebene Anzahl von Trainingsepisoden (Standard: 200)
- Der Agent lernt, Operationen so zu planen, dass:
  - Der Makespan minimiert wird
  - Prioritäten berücksichtigt werden
  - Deadlines eingehalten werden
  - Umrüstzeiten minimiert werden
- Testet den trainierten Agenten auf dem Problem
- Visualisiert den vom Agenten erzeugten Schedule
- Gibt den erreichten Makespan aus

### 4. Vergleichs-Modus

```bash
python main.py --mode compare --episodes 200
```

**Was dieser Befehl macht:**
- Führt sowohl den FIFO-Scheduler als auch den PPO-Agenten aus
- Trainiert den PPO-Agenten mit der angegebenen Anzahl von Episoden
- Vergleicht die Ergebnisse beider Ansätze:
  - Makespan (Gesamtbearbeitungszeit)
  - Einhaltung von Deadlines
  - Berücksichtigung von Prioritäten
  - Maschinenauslastung
- Visualisiert beide Schedules nebeneinander für einen direkten Vergleich
- Speichert die Vergleichsvisualisierung als Bild

## Parameter

- `--data`: Pfad zur JSP-Datendatei (Standard: `data.json`)
- `--mode`: Ausführungsmodus (`graph`, `fifo`, `ppo`, `compare`)
- `--episodes`: Anzahl der Trainingsepisoden für den PPO-Agenten (Standard: 200)

## Beispiele

```bash
# Visualisiere den JSP-Graphen mit benutzerdefinierten Daten
python main.py --mode graph --data custom_data.json

# Führe nur den FIFO-Scheduler aus
python main.py --mode fifo

# Trainiere den PPO-Agenten mit 500 Episoden
python main.py --mode ppo --episodes 500

# Vergleiche FIFO und PPO mit benutzerdefinierten Daten
python main.py --mode compare --data custom_data.json --episodes 300
```
