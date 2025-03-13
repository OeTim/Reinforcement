# PPO Agent Implementation

## Überblick
Die `torch_ppo_agent.py` Datei implementiert einen Reinforcement Learning Agenten basierend auf dem Proximal Policy Optimization (PPO) Algorithmus mit PyTorch für Job-Shop-Scheduling-Probleme (JSP).

## Hauptkomponenten

### Graph-basiertes Neuronales Netzwerk
- Verwendet einen gerichteten Graphen zur Repräsentation des JSP-Problems
- Implementiert ein einfaches Graph Neural Network (GNN) mit:
  - Embedding-Layer für Knotenmerkmale
  - Zwei Graph-Verarbeitungsschichten
  - Output-Layer für Aktionswahrscheinlichkeiten

### Exploration-Strategien
- Epsilon-Greedy mit anpassbarer Abklingrate
- Boltzmann-Exploration mit Temperaturparameter
- UCB-ähnliche Exploration (Upper Confidence Bound)
- Automatische Anpassung der Explorationsrate während des Trainings

### PPO-Algorithmus
- Implementiert den PPO-Clipping-Mechanismus für stabiles Training
- Verwendet Generalized Advantage Estimation (GAE)
- Mehrere Trainings-Epochen mit Minibatches
- Entropy-Bonus zur Förderung der Exploration

### Belohnungsfunktion
- Optimiert mehrere Ziele gleichzeitig:
  - Makespan-Minimierung
  - Maschinenauslastung
  - Prioritätsgewichtung von Jobs
  - Deadline-Einhaltung
  - Umrüstzeiten-Optimierung

## Besonderheiten
- Berücksichtigt Vorgängerbeziehungen zwischen Operationen
- Unterstützt Prioritäten und Deadlines für Jobs
- Optimiert für Materialwechsel und Umrüstzeiten
- Anpassungsfähig an verschiedene Umgebungen (SimpleEnvironment und GymEnvironment)