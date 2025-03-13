# Reward Function im JSP-Environment

## Überblick
Die Reward Function ist ein zentraler Bestandteil des Reinforcement Learning für das Job-Shop-Scheduling-Problem. Sie bewertet die Qualität der vom Agenten getroffenen Entscheidungen und steuert so den Lernprozess.

## Funktionsweise

### Aufruf der Reward Function
Die Reward Function wird in der `step`-Methode des Gym-Environments aufgerufen, nachdem eine Aktion (Auswahl eines Jobs) ausgeführt wurde:

```python
reward = self._calculate_reward(job_idx, job_completed, setup_time, 
                               prev_time, current_time)
```
### Parameter der Reward Function
- job_idx : Index des ausgewählten Jobs
- job_completed : Boolean, ob der Job vollständig abgeschlossen wurde
- setup_time : Umrüstzeit für die Operation
- prev_time : Vorherige aktuelle Zeit
- current_time : Neue aktuelle Zeit nach Ausführung der Operation
## Belohnungskomponenten
### 1. Makespan-Minimierung
- Ziel : Gesamtfertigstellungszeit minimieren
- Umsetzung : Negative Belohnung für Zeitfortschritt
- Beispiel : Wenn eine Operation 30 Minuten dauert, gibt es eine negative Belohnung von -0.3
### 2. Job-Fertigstellung
- Ziel : Abschluss von Jobs belohnen
- Umsetzung : Positive Belohnung, wenn ein Job vollständig abgeschlossen wird
- Beispiel : Bei Abschluss eines Jobs mit Priorität 8 gibt es eine Belohnung von +8.0
### 3. Prioritätsbasierte Belohnung
- Ziel : Hochprioritäre Jobs bevorzugen
- Umsetzung : Höhere Belohnung für Operationen an Jobs mit höherer Priorität
- Beispiel : Operation an einem Job mit Priorität 9 gibt +0.9, an einem Job mit Priorität 3 nur +0.3
### 4. Deadline-Einhaltung
- Ziel : Jobs vor ihrer Deadline abschließen
- Umsetzung : Bonus für Einhaltung der Deadline, Strafe für Überschreitung
- Beispiel : Abschluss vor Deadline gibt +5.0, Überschreitung gibt -10.0
### 5. Umrüstzeitminimierung
- Ziel : Umrüstzeiten minimieren
- Umsetzung : Negative Belohnung proportional zur Umrüstzeit
- Beispiel : Eine Umrüstzeit von 25 Minuten gibt eine Strafe von -2.5
### 6. Maschinenauslastung
- Ziel : Gleichmäßige Auslastung aller Maschinen
- Umsetzung : Belohnung für ausgewogene Maschinennutzung
- Beispiel : Bei gleichmäßiger Auslastung gibt es einen Bonus von +1.0
## Beispielberechnung
Angenommen, der Agent wählt Job 2 (Priorität 8) für die nächste Operation:

1. Die Operation dauert 30 Minuten → -0.3 (Zeitstrafe)
2. Die Umrüstzeit beträgt 15 Minuten → -1.5 (Umrüstzeitstrafe)
3. Die Priorität des Jobs ist 8 → +0.8 (Prioritätsbonus)
4. Der Job wird abgeschlossen → +8.0 (Fertigstellungsbonus)
5. Der Job wird vor der Deadline abgeschlossen → +5.0 (Deadline-Bonus)
6. Die Maschinenauslastung ist ausgewogen → +1.0 (Auslastungsbonus)
Gesamtbelohnung : -0.3 - 1.5 + 0.8 + 8.0 + 5.0 + 1.0 = +13.0

## Implementierungsdetails
Die Reward Function ist in der _calculate_reward -Methode der JSPGymEnvironment -Klasse implementiert. Sie kombiniert mehrere Belohnungskomponenten:

```python
def _calculate_reward(self, job_idx, job_completed, setup_time, prev_time, current_time):
    # Basisbelohnung für Zeitfortschritt (negativ, um Makespan zu minimieren)
    time_reward = -0.01 * (current_time - prev_time)
    
    # Belohnung basierend auf Jobpriorität
    priority_reward = 0.1 * self.jobs[job_idx]["priority"]
    
    # Belohnung für Jobfertigstellung
    completion_reward = 0.0
    if job_completed:
        completion_reward = self.jobs[job_idx]["priority"]
        
        # Zusätzliche Belohnung für Einhaltung der Deadline
        if current_time <= self.jobs[job_idx]["deadline"]:
            completion_reward += 5.0
        else:
            # Bestrafung für Überschreitung der Deadline
            completion_reward -= 10.0
    
    # Bestrafung für Umrüstzeiten
    setup_penalty = -0.1 * setup_time
    
    # Gesamtbelohnung
    total_reward = time_reward + priority_reward + completion_reward + setup_penalty
    
    return total_reward
````

## Anpassungsmöglichkeiten
Die Reward Function kann durch Gewichtungsfaktoren angepasst werden, um verschiedene Aspekte zu betonen:

- Höhere Gewichtung der Deadline-Einhaltung für zeitkritische Anwendungen
- Stärkere Bestrafung von Umrüstzeiten in materialintensiven Produktionen
- Erhöhte Belohnung für hochprioritäre Jobs bei kundenspezifischen Aufträgen
Diese flexible Gestaltung ermöglicht es, den Agenten auf unterschiedliche Optimierungsziele zu trainieren.

## Einfluss auf das Lernverhalten
Die Reward Function beeinflusst direkt, welche Strategien der Agent entwickelt:

1. Kurzfristige vs. langfristige Optimierung :
   
   - Hohe Belohnungen für Jobfertigstellung fördern langfristiges Denken
   - Zeitstrafen fördern kurzfristige Effizienz
2. Prioritätsmanagement :
   
   - Prioritätsbasierte Belohnungen lehren den Agenten, wichtigere Jobs vorzuziehen
   - Der Agent lernt, zwischen konkurrierenden Prioritäten abzuwägen
3. Deadline-Management :
   
   - Starke Belohnungen für Deadline-Einhaltung trainieren den Agenten, zeitkritische Jobs rechtzeitig abzuschließen
   - Der Agent entwickelt Strategien, um Verspätungen zu minimieren
4. Umrüstzeitoptimierung :
   
   - Strafen für Umrüstzeiten fördern die Gruppierung ähnlicher Operationen
   - Der Agent lernt, Materialwechsel zu minimieren
## Fortgeschrittene Belohnungskomponenten
Neben den Grundkomponenten können auch fortgeschrittene Belohnungsmechanismen implementiert werden:

### Kritischer Pfad
- Belohnung für Operationen auf dem kritischen Pfad des Schedules
- Fördert die Priorisierung von Engpässen im Produktionsablauf
### Ressourcenbalance
- Belohnung für gleichmäßige Auslastung aller Maschinen
- Verhindert Überlastung einzelner Ressourcen
### Lernende Gewichtung
- Dynamische Anpassung der Gewichtungsfaktoren während des Trainings
- Ermöglicht Fokussierung auf unterschiedliche Aspekte in verschiedenen Trainingsphasen
## Praktische Tipps zur Optimierung
1. Skalierung der Belohnungen :
   
   - Alle Belohnungskomponenten sollten in ähnlichen Größenordnungen liegen
   - Zu große Unterschiede können zu instabilem Lernverhalten führen
2. Ausgewogenheit :
   
   - Balance zwischen positiven und negativen Belohnungen finden
   - Zu viele negative Belohnungen können zu pessimistischem Verhalten führen
3. Exploration fördern :
   
   - Kleine positive Belohnungen für neue Aktionen können die Exploration verbessern
   - Hilft dem Agenten, aus lokalen Optima auszubrechen
4. Validierung :
   
   - Regelmäßige Überprüfung, ob die Belohnungsfunktion das gewünschte Verhalten fördert
   - Anpassung der Gewichtungen basierend auf Leistungsmetriken