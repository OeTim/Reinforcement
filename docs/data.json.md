# JSP Datendatei

## Überblick
Die `data.json` Datei enthält die Eingabedaten für das Job-Shop-Scheduling-Problem (JSP) in einer Fahrradproduktion. Sie definiert Jobs, Maschinen und Umrüstzeiten in einem strukturierten JSON-Format.

## Struktur

### Jobs
- Enthält zwei Fahrradproduktionsaufträge (J1: Citybike, J2: Racebike)
- Jeder Job hat:
  - Eine eindeutige ID
  - Prioritätswert (1-10)
  - Deadline (in Zeiteinheiten)
  - Sequenz von Operationen

### Operationen
- Jede Operation definiert einen Produktionsschritt
- Attribute:
  - ID (z.B. "OP1")
  - Zugewiesene Maschine
  - Bearbeitungszeit (in Minuten)
  - Vorgängerbeziehungen (technologische Reihenfolge)
  - Material/Komponente

### Maschinen
- Drei Maschinentypen:
  - M1: Rahmenfertigung (Rohrzuschnitt, Rahmenlötung, Schweißen)
  - M2: CNC-Bearbeitung (Fräsen, Bohren, Gewindeschneiden)
  - M3: Endmontage (Radmontage, Komponentenmontage, Qualitätskontrolle)

### Umrüstzeiten
- Definiert für jede Maschine:
  - Standardumrüstzeit (zwischen gleichen Materialien)
  - Materialwechselzeit (zwischen verschiedenen Materialien)

## Verwendung
Diese Datei dient als Eingabe für:
- Scheduling-Algorithmen (FIFO, PPO)
- Visualisierungstools (JSP-Graph)
- Simulationsumgebungen (GymEnvironment)
- Leistungsbewertung verschiedener Scheduling-Strategien