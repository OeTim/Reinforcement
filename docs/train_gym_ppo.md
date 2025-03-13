# PPO Training Module

## Überblick
Die `train_gym_ppo.py` Datei implementiert das Training eines Reinforcement Learning Agenten mit dem Proximal Policy Optimization (PPO) Algorithmus für Job-Shop-Scheduling-Probleme (JSP).

## Hauptfunktionalität
Die Datei enthält zwei Hauptfunktionen:

### `train_gym_ppo()`
- Lädt JSP-Daten aus einer JSON-Datei
- Erstellt eine Gym-Umgebung und einen PPO-Agenten
- Trainiert den Agenten über mehrere Episoden
- Verfolgt und visualisiert verschiedene Leistungsmetriken:
  - Belohnungen (Rewards)
  - Makespan (Gesamtfertigstellungszeit)
  - Verlustfunktion (Loss)
  - Prioritäten abgeschlossener Jobs
  - Eingehaltene Deadlines
  - Maschinenauslastung
- Speichert Checkpoints und das finale Modell

### `test_gym_ppo()`
- Testet einen trainierten Agenten
- Führt eine vollständige Episode aus
- Berechnet und zeigt Leistungsmetriken an

## Ablauf
1. **Daten laden**: JSP-Daten werden aus einer JSON-Datei geladen
2. **Training**: Der Agent lernt durch wiederholte Interaktion mit der Umgebung
3. **Visualisierung**: Trainingsfortschritt wird als Diagramm dargestellt
4. **Evaluation**: Der trainierte Agent wird getestet und bewertet

## Besonderheiten
- Verwendet PyTorch für die Implementierung des PPO-Algorithmus
- Berücksichtigt Prioritäten und Deadlines von Jobs
- Optimiert für Makespan-Minimierung und Maschinenauslastung
- Speichert Modelle und Visualisierungen mit Zeitstempeln