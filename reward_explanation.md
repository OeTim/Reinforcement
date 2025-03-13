# Belohnungssystem in der JSP Gym-Umgebung

Dieses Dokument bietet eine detaillierte Erklärung, wie Belohnungen in der Job Shop Scheduling (JSP) Gym-Umgebung berechnet werden. Das Belohnungssystem wurde entwickelt, um den Reinforcement-Learning-Agenten zu optimalen Planungsentscheidungen zu führen, wobei mehrere Ziele wie Makespan, Fristen, Prioritäten und Maschinenauslastung berücksichtigt werden.

## Übersicht der Belohnungskomponenten

Die Belohnungsfunktion in `_calculate_reward()` kombiniert 11 verschiedene Komponenten, die jeweils entsprechend ihrer Bedeutung im Planungsproblem gewichtet sind:

1. **Makespan-Optimierung** (Gewicht: 3,5)
2. **Rüstzeit-Optimierung** (Gewicht: 1,2)
3. **Maschinenleerlauf-Strafe** (Gewicht: 1,2)
4. **Maschinenausgleich** (Gewicht: 0,7)
5. **Einhaltung von Fristen** (Gewicht: 6,0)
6. **Prioritätsbasierte Belohnung** (Gewicht: 3,0)
7. **Fortschrittsbelohnung** (Gewicht: 0,5)
8. **Auftragsabschluss-Bonus** (Gewicht: 2,5)
9. **Priorisierung kritischer Aufträge** (Gewicht: 2,5)
10. **Globaler Fortschritt** (Gewicht: 0,5)
11. **Verbesserung der Zielfunktion** (Gewicht: 4,0)

Die endgültige Belohnung ist zwischen -15,0 und 20,0 begrenzt, um extreme Werte zu vermeiden.

## Detaillierte Erklärung jeder Komponente

### 1. Makespan-Optimierung (Gewicht: 3,5)

Diese Komponente belohnt oder bestraft Operationen basierend darauf, ob sie auf dem kritischen Pfad liegen (die Sequenz von Operationen, die den gesamten Makespan bestimmt).

```python
# Berechne Zeiteffizienz
time_efficiency = (avg_proc_time - proc_time) / avg_proc_time if avg_proc_time > 0 else 0

# Wenn die Operation nicht auf dem kritischen Pfad liegt
if self.machine_times[machine_idx] < current_makespan:
    makespan_reward = 3.0 + time_efficiency
else:
    # Operation liegt auf dem kritischen Pfad
    makespan_reward = -2.0 + time_efficiency + priority_factor * 2.0
```

- **Belohnung**: +3,0 plus Effizienzbonus für Operationen, die nicht auf dem kritischen Pfad liegen
- **Strafe**: -2,0 plus Effizienzbonus für Operationen auf dem kritischen Pfad, gemildert durch Auftragspriorität
- **Zweck**: Förderung effizienter Verarbeitung und Priorisierung von Aufträgen mit hoher Priorität auf dem kritischen Pfad

### 2. Rüstzeit-Optimierung (Gewicht: 1,2)

Diese Komponente belohnt die Minimierung von Maschinenrüstzeiten, insbesondere für Aufträge mit hoher Priorität.

```python
if setup_time == 0:
    setup_reward = 1.5 * (1 + priority_factor)  # Bonus für keine Rüstzeit
elif setup_time <= self.setupTimes[machine_id]["standard"]:
    setup_reward = 0.7 * (1 + priority_factor)  # Moderater Bonus für Standardrüstzeit
else:
    # Strafe für Materialwechsel, gemildert durch hohe Priorität
    setup_reward = -1.5 + priority_factor
```

- **Belohnung**: Bis zu 3,0 für keine Rüstzeit, skaliert nach Priorität
- **Strafe**: Bis zu -1,5 für Materialwechsel, gemildert durch Priorität
- **Zweck**: Minimierung unnötiger Maschinenumrüstungen und Materialwechsel

### 3. Maschinenleerlauf-Strafe (Gewicht: 1,2)

Diese Komponente bestraft Maschinenleerlaufzeiten und belohnt kontinuierliche Maschinenauslastung.

```python
if machine_idle_time > 0:
    idle_penalty = -1.5 * min(1.0, machine_idle_time / (avg_proc_time * 2.0)) * (1 - priority_factor * 0.5)
else:
    idle_penalty = 0.8 * (1 + priority_factor * 0.5)  # Bonus für keine Leerlaufzeit
```

- **Belohnung**: Bis zu 1,2 für keine Leerlaufzeit, skaliert nach Priorität
- **Strafe**: Bis zu -1,5 für Leerlaufzeit, skaliert nach dem Verhältnis von Leerlaufzeit zur durchschnittlichen Verarbeitungszeit
- **Zweck**: Maximierung der Maschinenauslastung

### 4. Maschinenausgleich (Gewicht: 0,7)

Diese Komponente belohnt eine ausgewogene Maschinenauslastung über alle Maschinen hinweg.

```python
if current_time > 0:
    relative_imbalance = machine_time_std / current_time
    balance_reward = 0.6 * (1.0 - min(1.0, relative_imbalance * (1 - priority_factor * 0.5)))
else:
    balance_reward = 0.0
```

- **Belohnung**: Bis zu 0,6 für perfekt ausgewogene Maschinenauslastung
- **Strafe**: Nimmt ab, wenn die Maschinenauslastung unausgeglichener wird
- **Zweck**: Verhinderung von Engpässen durch gleichmäßige Auslastung aller Maschinen

### 5. Einhaltung von Fristen (Gewicht: 6,0)

Dies ist die am höchsten gewichtete Komponente, die die Bedeutung der Einhaltung von Auftragsfristen unterstreicht.

```python
if job_completed:
    completion_time = self.machine_times[machine_idx]
    
    if completion_time <= job_deadline:
        # Massive Belohnung für die Einhaltung von Fristen
        deadline_reward = 12.0 * (1 + priority_factor * 0.5)
    else:
        # Starke Strafe für verpasste Fristen
        overdue_ratio = (completion_time - job_deadline) / job_deadline
        deadline_reward = -7.0 * min(1.0, overdue_ratio) * (1 - priority_factor * 0.3)
else:
    # Belohnung für Fortschritte bei Aufträgen, die wahrscheinlich die Frist einhalten werden
    time_margin = job_deadline - estimated_finish_time
    if time_margin > 0:
        deadline_reward = 3.0 * (progress_ratio) * (1 + priority_factor * 0.5)
    else:
        deadline_reward = -1.5 * (1.0 - progress_ratio) * (1 - priority_factor * 0.3)
```

- **Belohnung**: Bis zu 18,0 für den Abschluss von Aufträgen mit hoher Priorität vor der Frist
- **Strafe**: Bis zu -7,0 für das Verpassen von Fristen, skaliert danach, wie stark die Frist überschritten wurde
- **Zweck**: Starker Anreiz zur Einhaltung von Fristen, besonders für Aufträge mit hoher Priorität

### 6. Prioritätsbasierte Belohnung (Gewicht: 3,0)

Diese Komponente belohnt Aktionen direkt basierend auf der Auftragspriorität.

```python
priority_reward = 2.5 * priority_factor
```

- **Belohnung**: Bis zu 2,5 für Aufträge mit höchster Priorität (Priorität 10)
- **Zweck**: Sicherstellen, dass Aufträge mit hoher Priorität Aufmerksamkeit erhalten

### 7. Fortschrittsbelohnung (Gewicht: 0,5)

Diese Komponente belohnt Fortschritte bei einzelnen Aufträgen.

```python
progress_ratio = self.job_progress[job_idx] / len(self.jobs[job_idx]["operations"])
progress_reward = 0.5 * progress_ratio * (1 + priority_factor * 0.5)
```

- **Belohnung**: Steigt mit dem Fortschritt der Aufträge bis zum Abschluss, skaliert nach Priorität
- **Zweck**: Förderung stetiger Fortschritte bei allen Aufträgen

### 8. Auftragsabschluss-Bonus (Gewicht: 2,5)

Diese Komponente bietet einen erheblichen Bonus für den Abschluss von Aufträgen.

```python
if job_completed:
    completion_reward = 4.0 * (1 + priority_factor)
```

- **Belohnung**: Bis zu 8,0 für den Abschluss von Aufträgen mit höchster Priorität
- **Zweck**: Anreiz zum Abschluss von Aufträgen

### 9. Priorisierung kritischer Aufträge (Gewicht: 2,5)

Diese Komponente identifiziert und belohnt die Verarbeitung von Aufträgen, bei denen die Gefahr besteht, dass sie ihre Fristen verpassen.

```python
if not job_completed:
    if remaining_ops > 0:
        urgency = (job_deadline - current_time) / (remaining_ops * avg_op_time)
        if urgency < 1.0:  # Sehr kritischer Auftrag
            critical_job_reward = 4.0 * (1 + priority_factor * 0.5)
        elif urgency < 1.5:  # Kritischer Auftrag
            critical_job_reward = 2.0 * (1 + priority_factor * 0.3)
```

- **Belohnung**: Bis zu 6,0 für die Verarbeitung sehr kritischer Aufträge
- **Zweck**: Identifizierung und Priorisierung von Aufträgen, bei denen die Gefahr besteht, dass sie ihre Fristen verpassen

### 10. Globaler Fortschritt (Gewicht: 0,5)

Diese Komponente belohnt den Gesamtfortschritt über alle Aufträge hinweg.

```python
global_progress = sum(self.job_progress) / sum(len(job["operations"]) for job in self.jobs)
global_progress_reward = 0.5 * global_progress * (1 + priority_factor * 0.2)
```

- **Belohnung**: Steigt mit zunehmendem Gesamtabschlussprozentsatz
- **Zweck**: Förderung stetiger Fortschritte bei allen Aufträgen

### 11. Verbesserung der Zielfunktion (Gewicht: 4,0)

Diese Komponente belohnt Aktionen, die die Gesamtzielfunktion verbessern (Makespan + Makespan * (1 - Pünktlichkeit)).

```python
if hasattr(self, 'previous_objective'):
    objective_improvement = self.previous_objective - current_objective
    if objective_improvement > 0:
        objective_reward = 3.0 * min(1.0, objective_improvement / current_objective) * (1 + priority_factor * 0.5)
    else:
        objective_reward = -1.0 * min(1.0, -objective_improvement / current_objective) * (1 - priority_factor * 0.3)
```

- **Belohnung**: Bis zu 4,5 für signifikante Verbesserungen der Zielfunktion
- **Strafe**: Bis zu -1,0 für Verschlechterung der Zielfunktion
- **Zweck**: Direkte Belohnung von Aktionen, die das Gesamtplanungsziel verbessern

## Endgültige Belohnungsberechnung

Alle Komponenten werden mit ihren jeweiligen Gewichtungen kombiniert:

```python
total_reward = (
    makespan_reward * 3.5 +
    setup_reward * 1.2 +
    idle_penalty * 1.2 +
    balance_reward * 0.7 +
    deadline_reward * 6.0 +
    priority_reward * 3.0 +
    progress_reward * 0.5 +
    completion_reward * 2.5 +
    critical_job_reward * 2.5 +
    global_progress_reward * 0.5 +
    objective_reward * 4.0
)

# Begrenzung der Belohnung zur Vermeidung extremer Werte
total_reward = max(min(total_reward, 20.0), -15.0)
```

## Wann Belohnungen vergeben werden

Belohnungen werden berechnet und vergeben:
- Nach jedem Schritt (Aktion) in der Umgebung
- Basierend auf dem aktuellen Zustand, der ausgewählten Aktion und dem resultierenden nächsten Zustand
- Unter Berücksichtigung der Priorität, Frist und des Fortschritts des Auftrags
- Unter Berücksichtigung der Maschinenauslastung und Rüstzeiten

## Zusammenfassung

Das Belohnungssystem wurde entwickelt, um mehrere Ziele auszubalancieren:
1. Minimierung des Makespan (Gesamtfertigstellungszeit)
2. Einhaltung von Auftragsfristen, insbesondere für Aufträge mit hoher Priorität
3. Maximierung der Maschinenauslastung
4. Minimierung der Rüstzeiten
5. Ausgleich der Arbeitsbelastung über alle Maschinen hinweg
6. Priorisierung kritischer Aufträge, bei denen die Gefahr besteht, dass sie ihre Fristen verpassen

Durch die Kombination dieser Komponenten mit angemessenen Gewichtungen führt die Belohnungsfunktion den Reinforcement-Learning-Agenten zu optimalen Planungsentscheidungen, die Effizienz, Pünktlichkeit und Priorität ausbalancieren.