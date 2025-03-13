"""
Job-Shop Scheduling Datengenerator
Generiert eine data.json-Datei mit einer angegebenen Anzahl von Jobs
"""

import json
import random
import sys
import os
from datetime import datetime

def generate_jsp_data(num_jobs):
    """
    Generiert Job-Shop Scheduling Daten mit der angegebenen Anzahl von Jobs
    
    Args:
        num_jobs: Anzahl der zu generierenden Jobs
        
    Returns:
        Ein Dictionary mit den generierten Daten
    """
    # Maschinen in einer realistischen Fahrradproduktion
    MACHINES = [
        {"id": "M1", "name": "Rahmenfertigung", "capabilities": ["Rohrzuschnitt", "Rahmenlötung", "Rahmenschweissung"]},
        {"id": "M2", "name": "CNC-Bearbeitung", "capabilities": ["Fräsen", "Bohren", "Gewindeschneiden"]},
        {"id": "M3", "name": "Endmontage", "capabilities": ["Radmontage", "Komponentenmontage", "Qualitätskontrolle"]}
    ]
    
    # Umrüstzeiten zwischen verschiedenen Produktionsschritten (in Minuten)
    SETUP_TIMES = {
        "M1": {"standard": 25, "materialChange": 40},  # Umrüstung der Schweissvorrichtungen dauert länger
        "M2": {"standard": 15, "materialChange": 20},  # CNC-Maschinen können schneller umgerüstet werden
        "M3": {"standard": 10, "materialChange": 15}   # Montagestation ist flexibler
    }
    
    # Realistische Fahrradmodelle
    BIKE_MODELS = [
        "Mountainbike",      # Mountainbike-Modell
        "Racebike",    # Rennrad-Modell
        "Citybike",   # Stadtrad-Modell
        "E-Bike",       # E-Bike-Modell
    ]
    
    # Produktionsschritte und Komponenten nach Maschinengruppe
    PRODUCTION_STEPS = {
        # Rahmenfertigung
        "M1": [
            "Hauptrahmen_zugeschnitten",
            "Hinterbau_vorbereitet", 
            "Tretlageraufnahme_geschweisst", 
            "Steuerrohr_montiert",
            "Rahmen_komplettiert"
        ],
        # CNC-Bearbeitung
        "M2": [
            "Tretlager_gefraest", 
            "Schaltauge_bearbeitet", 
            "Steuerrohr_ausgerieben",
            "Bremsaufnahme_gebohrt",
            "Rahmen_nachbearbeitet"
        ],
        # Endmontage
        "M3": [
            "Rahmen_lackiert", 
            "Laufraeder_montiert", 
            "Antrieb_installiert",
            "Bremsen_justiert",
            "Fahrrad_endmontiert"
        ]
    }
    
    # Generiere Jobs
    jobs = []
    for j in range(1, num_jobs + 1):
        # Wähle ein Fahrradmodell für diesen Produktionsauftrag
        bike_model = random.choice(BIKE_MODELS)
        
        # Priorität basierend auf Fahrradtyp und Kundenstatus
        # E-Bikes und Rennräder haben oft höhere Priorität
        if "E-Bike" in bike_model or "Race" in bike_model:
            priority = random.randint(7, 10)  # Höhere Priorität
        else:
            priority = random.randint(1, 8)   # Normale Priorität
        
        # Deadlines werden später berechnet, nachdem wir die Operationen generiert haben
        deadline = 0
        
        # Ein realistischer Produktionsablauf hat 5-8 Schritte
        # Jedes Fahrrad durchläuft alle drei Maschinenstationen, manche mehrfach
        num_operations = random.randint(5, 8)
        
        # Generiere Operationen
        operations = []
        
        # Realistische Produktionsreihenfolge für Fahrräder:
        # 1. Immer mit Rahmenfertigung beginnen (M1)
        # 2. Dann CNC-Bearbeitung (M2)
        # 3. Dann Endmontage (M3)
        # 4. Bei Bedarf weitere Schritte in verschiedenen Stationen
        
        # Basis-Sequenz für die ersten drei Operationen
        machine_sequence = ["M1", "M2", "M3"]
        
        # Ergänze mit zufälligen weiteren Schritten wenn nötig
        while len(machine_sequence) < num_operations:
            # Zusätzliche Schritte folgen typischerweise einem realistischen Muster
            # Häufig wird zwischen CNC und Endmontage gewechselt
            machine_sequence.append(random.choice(["M2", "M3"]))
        
        for op in range(1, num_operations + 1):
            # Verwende die vorbereitete Maschinensequenz
            machine_id = machine_sequence[op-1]
            
            # Realistische Verarbeitungszeiten je nach Maschinentyp und Produktionsschritt
            if machine_id == "M1":
                # Rahmenfertigung dauert typischerweise länger (15-45 Minuten)
                processing_time = random.randint(15, 45)
            elif machine_id == "M2":
                # CNC-Bearbeitung (10-30 Minuten)
                processing_time = random.randint(10, 30)
            else:  # M3
                # Endmontage (20-60 Minuten)
                processing_time = random.randint(20, 60)
            
            # Vorgänger (leer für die erste Operation, sonst die vorherige Operation)
            predecessors = []
            if op > 1:
                predecessors.append(f"J{j}:OP{op-1}")
            
            # Wähle einen realistischen Produktionsschritt basierend auf der Maschine
            # und der Position im Produktionsablauf
            if op == 1:
                # Erster Schritt ist immer Rahmenvorbereitung
                production_step = PRODUCTION_STEPS["M1"][0]
            elif op == num_operations:
                # Letzter Schritt ist immer Endmontage
                production_step = "Fahrrad_endmontiert"
            else:
                # Zwischenschritte entsprechend der Maschinenstation
                # Vermeide Wiederholung des gleichen Schritts
                available_steps = PRODUCTION_STEPS[machine_id]
                if op > 1 and op < len(machine_sequence):
                    # Wähle Schritte basierend auf der Position im Produktionsablauf
                    step_index = min(op-1, len(available_steps)-1)
                    production_step = available_steps[step_index]
                else:
                    production_step = random.choice(available_steps)
            
            # Kombiniere Fahrradmodell mit Produktionsschritt
            material = f"{bike_model}_{production_step}"
            
            operations.append({
                "id": f"OP{op}",
                "machineId": machine_id,
                "processingTime": processing_time,
                "predecessors": predecessors,
                "material": material
            })
        
        # Berechne die Gesamtbearbeitungszeit des Jobs
        total_processing_time = sum(op["processingTime"] for op in operations)
        
        # Setup-Puffer: Durchschnittliche Umrüstzeit pro Operation
        avg_setup_time = sum(SETUP_TIMES[op["machineId"]]["standard"] for op in operations) / len(operations)
        setup_buffer = avg_setup_time * (len(operations) - 1)  # Keine Umrüstung für erste Operation
        
        # Minimale Bearbeitungszeit für diesen Job (Basis + Setup)
        min_completion_time = total_processing_time + setup_buffer
        
        # Maximale Zeit für den 2-Tage-Zeitraum (in Minuten)
        MAX_TIME = 2 * 24 * 60  # 2 Tage in Minuten = 2880
        
        # Verbesserte Berechnung des Workshop-Load-Faktors
        # Berücksichtigt die Anzahl der Jobs und die durchschnittliche Anzahl von Operationen
        avg_operations_per_job = 6.5  # Durchschnitt zwischen 5-8 Operationen pro Job
        total_estimated_operations = num_jobs * avg_operations_per_job
        
        # Schätze die Gesamtauslastung basierend auf der Anzahl der Operationen und Maschinen
        num_machines = len(MACHINES)
        workshop_load_factor = max(2.0, min(5.0, total_estimated_operations / (num_machines * 5)))
        
        # Berechne eine realistische Deadline basierend auf der Priorität und Werkstattauslastung
        if priority >= 7:
            # Hohe Priorität: Engere Deadline, aber realistisch
            deadline_factor = random.uniform(2.5, 3.5) * workshop_load_factor
        else:
            # Normale Priorität: Großzügigere Deadline
            deadline_factor = random.uniform(3.0, 4.5) * workshop_load_factor
        
        # Berechne die Deadline, aber stelle sicher, dass sie innerhalb des 2-Tage-Fensters liegt
        deadline = min(int(min_completion_time * deadline_factor), MAX_TIME)
        
        # Stelle sicher, dass die Deadline realistisch und einhaltbar ist
        # Sie sollte mindestens die Mindestbearbeitungszeit plus einen angemessenen Puffer sein
        deadline = max(deadline, int(min_completion_time * 2.0))
        
        # Füge eine kleine zufällige Variation hinzu (±5%), um natürlichere Verteilung zu erhalten
        deadline = int(deadline * random.uniform(0.95, 1.05))
        
        # Stelle sicher, dass die Deadline nicht den maximalen Zeitraum überschreitet
        deadline = min(deadline, MAX_TIME)
        
        jobs.append({
            "id": f"J{j}",
            "priority": priority,
            "deadline": deadline,
            "operations": operations
        })
    
    # Erstelle das Gesamtdatenstruktur
    data = {
        "jobs": jobs,
        "machines": MACHINES,
        "setupTimes": SETUP_TIMES
    }
    
    return data

def save_data(data, filename="data.json"):
    """Speichert die generierten Daten in einer JSON-Datei"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Daten wurden in {filename} gespeichert.")

def main():
    """Hauptfunktion"""
    # Überprüfe die Kommandozeilenargumente
    if len(sys.argv) < 2:
        print("Verwendung: python data_generator.py <anzahl_jobs> [ausgabedatei]")
        sys.exit(1)
    
    try:
        num_jobs = int(sys.argv[1])
        if num_jobs <= 0:
            raise ValueError("Anzahl der Jobs muss positiv sein")
    except ValueError as e:
        print(f"Fehler: {e}")
        print("Verwendung: python data_generator.py <anzahl_jobs> [ausgabedatei]")
        sys.exit(1)
    
    # Bestimme den Ausgabedateinamen
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Standardmäßig data_<anzahl_jobs>_<timestamp>.json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_file = f"data_{num_jobs}_{timestamp}.json"
        output_file = "data.json"
    
    # Generiere und speichere die Daten
    data = generate_jsp_data(num_jobs)
    save_data(data, output_file)
    
    # Ausgabe einiger Statistiken
    print(f"Generierte Daten:")
    print(f"- Anzahl Jobs: {len(data['jobs'])}")
    print(f"- Anzahl Maschinen: {len(data['machines'])}")
    total_operations = sum(len(job['operations']) for job in data['jobs'])
    print(f"- Gesamtanzahl Operationen: {total_operations}")
    print(f"- Durchschnittliche Operationen pro Job: {total_operations/len(data['jobs']):.1f}")

if __name__ == "__main__":
    main()
