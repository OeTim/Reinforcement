# Job-Shop Scheduling mit Reinforcement Learning

Dieses Projekt implementiert verschiedene Lösungsansätze für das Job-Shop Scheduling Problem (JSP) mit besonderem Fokus auf Reinforcement Learning.

## Was ist das Job-Shop Scheduling Problem?

Das JSP ist ein klassisches Optimierungsproblem:
- Mehrere Jobs müssen auf verschiedenen Maschinen bearbeitet werden
- Jeder Job besteht aus einer Reihe von Operationen in einer festgelegten Reihenfolge
- Jede Operation benötigt eine bestimmte Maschine für eine bestimmte Zeit
- Ziel: Minimierung der Gesamtbearbeitungszeit (Makespan)

## Besonderheiten dieser Implementierung

Diese Implementierung erweitert das klassische JSP um:
- **Vorgängerbeziehungen**: Operationen können beliebige Vorgänger haben (nicht nur sequentiell)
- **Materialtypen**: Operationen verarbeiten bestimmte Materialien
- **Umrüstzeiten**: Maschinen benötigen Zeit zum Umrüsten zwischen verschiedenen Materialien
- **Prioritäten**: Jobs haben unterschiedliche Prioritäten (1-10)
- **Deadlines**: Jobs müssen bis zu einem bestimmten Zeitpunkt fertig sein

## Implementierte Lösungsansätze

### 1. FIFO-Scheduler
Ein einfacher First-In-First-Out Scheduler, der Jobs nach Priorität sortiert und Vorgängerbeziehungen berücksichtigt.

```python
# Beispiel: FIFO-Scheduler ausführen
schedule, makespan = fifo_schedule("data.json")
print(f"FIFO Makespan: {makespan}")
```

### 2. PPO-Agent (Reinforcement Learning)
Ein Proximal Policy Optimization (PPO) Agent, der mit PyTorch implementiert ist.

```python
# Beispiel: PPO-Agent trainieren und testen
trained_agent, env = train_torch_agent("data.json", num_episodes=200)
final_state, actions = test_torch_agent(trained_agent, env)
```

## Visualisierungen

Das Projekt bietet verschiedene Visualisierungsmöglichkeiten:

### 1. JSP-Graph
Visualisiert das Problem als Graph mit konjunktiven (Operationsreihenfolge) und disjunktiven (Maschinenkonflikte) Kanten.

```python
# Beispiel: JSP-Graph visualisieren
run_jsp_graph("data.json")
```

### 2. Schedule-Visualisierung
Zeigt den erzeugten Schedule als Gantt-Chart mit Umrüstzeiten, Prioritäten und Deadlines.

```python
# Beispiel: Schedule visualisieren
schedule, makespan = visualize_schedule("data.json", actions)
```

### 3. Vergleichsvisualisierung
Vergleicht die Ergebnisse des FIFO-Schedulers mit dem PPO-Agenten.

```python
# Beispiel: FIFO und PPO vergleichen
compare_torch_fifo("data.json", num_episodes=200)
```

## Ausführung

```bash
# Alle Modi vergleichen
python main.py --mode compare --episodes 200

# Nur FIFO ausführen
python main.py --mode fifo

# Nur PPO ausführen
python main.py --mode ppo --episodes 300

# Nur JSP-Graph visualisieren
python main.py --mode graph
```

## Datenstruktur

Die Daten werden in einem JSON-Format gespeichert, das Jobs, Operationen, Maschinen und ihre Beziehungen definiert. Die Struktur verwendet sprechende IDs (z.B. "J1", "OP1", "M1") für bessere Lesbarkeit.
