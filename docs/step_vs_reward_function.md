# Unterschied zwischen Step und Calculate Reward Funktionen

## Die Step Funktion

Die `step` Funktion ist die Hauptfunktion in der JSP-Umgebung, die einen einzelnen Schritt in der Simulation ausführt. Sie ist verantwortlich für:

1. **Validierung der Aktion**: Überprüft, ob die gewählte Aktion (Job-Index) gültig ist.
2. **Ausführung der Operation**: Führt die nächste Operation des ausgewählten Jobs aus.
3. **Aktualisierung des Umgebungszustands**: 
   - Aktualisiert die Maschinenzeiten
   - Aktualisiert das aktuelle Material auf der Maschine
   - Aktualisiert den Fortschritt des Jobs
   - Aktualisiert die aktuelle Zeit
4. **Überprüfung von Jobabschluss und Deadlines**: Prüft, ob Jobs abgeschlossen wurden und ob Deadlines eingehalten wurden.
5. **Berechnung der Belohnung**: Ruft die `_calculate_reward` Funktion auf, um die Belohnung für diesen Schritt zu berechnen.
6. **Aktualisierung der Episodenstatistiken**: Aktualisiert Werte wie Gesamtbelohnung und Makespan.
7. **Rückgabe von Informationen**: Gibt den neuen Zustand, die Belohnung, ein "Done"-Flag und zusätzliche Informationen zurück.

Die Step-Funktion ist also für den gesamten Ablauf eines Zeitschritts in der Umgebung verantwortlich und koordiniert alle notwendigen Aktualisierungen.

## Die Calculate Reward Funktion

Die `_calculate_reward` Funktion ist spezialisiert auf die Berechnung der Belohnung für einen bestimmten Schritt. Sie berücksichtigt verschiedene Faktoren:

1. **Makespan-Optimierung**: Belohnt Operationen, die den Makespan (Gesamtfertigstellungszeit) nicht erhöhen.
2. **Umrüstzeit-Optimierung**: Belohnt geringe Umrüstzeiten und bestraft Materialwechsel.
3. **Maschinenauslastung**: Bestraft Leerlaufzeiten und belohnt kontinuierliche Nutzung.
4. **Maschinenbalance**: Belohnt eine gleichmäßige Auslastung aller Maschinen.
5. **Deadline-Einhaltung**: Stark erhöhte Belohnung für die Einhaltung von Deadlines.
6. **Prioritätsbasierte Belohnung**: Höhere Belohnung für Jobs mit höherer Priorität.
7. **Fortschrittsbelohnung**: Belohnt den Fortschritt bei der Ausführung von Jobs.
8. **Jobabschluss-Bonus**: Zusätzliche Belohnung für abgeschlossene Jobs.
9. **Kritische Jobs bevorzugen**: Belohnt die Bearbeitung von Jobs, die zeitkritisch sind.
10. **Globaler Fortschritt**: Belohnt den Gesamtfortschritt aller Jobs.

Die Funktion kombiniert all diese Faktoren mit unterschiedlichen Gewichtungen, um eine Gesamtbelohnung zu berechnen, die dann auf einen bestimmten Bereich begrenzt wird.

## Zusammenfassung

- **Step Funktion**: Steuert den gesamten Ablauf eines Zeitschritts in der Umgebung und führt alle notwendigen Aktualisierungen durch.
- **Calculate Reward Funktion**: Berechnet die Belohnung für einen bestimmten Schritt basierend auf verschiedenen Faktoren und Gewichtungen.

Die Step-Funktion ruft die Calculate Reward Funktion auf, um die Belohnung für den aktuellen Schritt zu ermitteln, und verwendet diesen Wert dann für die Rückgabe und die Aktualisierung der Episodenstatistiken.

## Warum werden die Step und Calculate Reward Funktionen benötigt?

### Notwendigkeit der Step Funktion

Die `step` Funktion ist ein fundamentaler Bestandteil jeder OpenAI Gym-Umgebung und erfüllt mehrere kritische Aufgaben:

1. **Schnittstelle zum Reinforcement Learning Agenten**: Sie bildet die Hauptschnittstelle zwischen dem RL-Agenten und der Umgebung. Der Agent übergibt eine Aktion, und die Umgebung gibt einen neuen Zustand, eine Belohnung und weitere Informationen zurück.

2. **Simulation der Umgebungsdynamik**: Sie simuliert, wie sich die Umgebung als Reaktion auf die Aktion des Agenten verändert. Im JSP-Kontext bedeutet das, dass sie die Ausführung von Operationen auf Maschinen, die Aktualisierung von Zeiten und den Fortschritt von Jobs verwaltet.

3. **Zustandsübergänge**: Sie berechnet den neuen Zustand der Umgebung nach der Ausführung einer Aktion, was für das Lernen des Agenten entscheidend ist.

4. **Terminierungsbedingungen**: Sie überprüft, ob die Episode beendet ist (z.B. wenn alle Jobs abgeschlossen sind), was für das episodische Training wichtig ist.

Ohne die `step` Funktion könnte der RL-Agent nicht mit der Umgebung interagieren und somit nicht lernen, wie er optimale Entscheidungen treffen kann.

### Notwendigkeit der Calculate Reward Funktion

Die `_calculate_reward` Funktion ist aus folgenden Gründen entscheidend:

1. **Lernziel definieren**: Die Belohnungsfunktion definiert das Ziel, das der Agent erreichen soll. Sie bestimmt, welches Verhalten als "gut" oder "schlecht" angesehen wird.

2. **Komplexe Bewertungskriterien**: In komplexen Umgebungen wie dem JSP gibt es viele Faktoren, die zur Bewertung einer Aktion beitragen. Die separate Funktion ermöglicht eine strukturierte und modulare Berechnung dieser komplexen Bewertung.

3. **Steuerung des Lernverhaltens**: Durch die Gewichtung verschiedener Faktoren kann das Lernverhalten des Agenten gesteuert werden. Zum Beispiel kann eine höhere Gewichtung der Deadline-Einhaltung den Agenten dazu bringen, Deadlines stärker zu priorisieren.

4. **Trennung von Zustandsübergang und Bewertung**: Die Trennung der Belohnungsberechnung von der Zustandsübergangslogik verbessert die Modularität und Wartbarkeit des Codes.

Ohne eine gut definierte Belohnungsfunktion würde der Agent nicht wissen, welche Aktionen vorteilhaft sind, und könnte keine optimale Strategie erlernen.

### Was genau passiert in diesen Funktionen?

#### In der Step Funktion:

1. Der Agent wählt eine Aktion (einen Job) aus.
2. Die Funktion überprüft, ob die Aktion gültig ist (z.B. ob der Job noch nicht abgeschlossen ist).
3. Sie identifiziert die nächste Operation des gewählten Jobs und die benötigte Maschine.
4. Sie berechnet die Umrüstzeit basierend auf dem aktuellen und dem neuen Material.
5. Sie berechnet die Start- und Endzeit der Operation unter Berücksichtigung der Maschinenverfügbarkeit.
6. Sie aktualisiert den Zustand der Umgebung (Maschinenzeiten, Jobfortschritt, aktuelle Zeit).
7. Sie überprüft, ob Jobs abgeschlossen wurden und ob Deadlines eingehalten wurden.
8. Sie ruft die `_calculate_reward` Funktion auf, um die Belohnung zu berechnen.
9. Sie aktualisiert die Episodenstatistiken und gibt den neuen Zustand, die Belohnung und weitere Informationen zurück.

#### In der Calculate Reward Funktion:

1. Sie extrahiert Details über die gerade ausgeführte Operation.
2. Sie berechnet verschiedene Belohnungskomponenten basierend auf verschiedenen Faktoren:
   - Wie die Operation den Makespan beeinflusst
   - Wie hoch die Umrüstzeit war
   - Ob die Maschine Leerlaufzeiten hatte
   - Wie ausgeglichen die Maschinenauslastung ist
   - Ob Deadlines eingehalten werden
   - Die Priorität des Jobs
   - Den Fortschritt des Jobs und den globalen Fortschritt
   - Ob der Job abgeschlossen wurde
   - Wie kritisch der Job in Bezug auf seine Deadline ist
3. Sie kombiniert diese Komponenten mit unterschiedlichen Gewichtungen zu einer Gesamtbelohnung.
4. Sie begrenzt die Belohnung auf einen bestimmten Bereich, um numerische Stabilität zu gewährleisten.

Diese detaillierte Belohnungsberechnung ermöglicht es dem Agenten, komplexe Zusammenhänge zu erlernen und eine optimale Scheduling-Strategie zu entwickeln, die verschiedene Faktoren wie Makespan, Deadlines und Umrüstzeiten berücksichtigt.