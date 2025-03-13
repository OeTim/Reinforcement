import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def visualize_schedule(jsp_data_path, actions):
    """
    Visualisiert den erzeugten Schedule als Gantt-Chart
    
    Args:
        jsp_data_path: Pfad zur JSP-Datendatei
        actions: Liste der ausgeführten Aktionen (Job-Indizes)
    """
    # JSP-Daten laden
    with open(jsp_data_path, 'r') as f:
        jsp_data = json.load(f)
    
    # Mapping von Job-IDs zu Indizes erstellen
    job_id_to_idx = {job["id"]: idx for idx, job in enumerate(jsp_data["jobs"])}
    
    # Mapping von Maschinen-IDs zu Indizes erstellen
    machine_id_to_idx = {machine["id"]: idx for idx, machine in enumerate(jsp_data["machines"])}
    
    # Schedule erstellen
    schedule = []  # Liste von (job_id, operation_id, machine_id, start_time, end_time)
    job_progress = [0 for _ in range(len(jsp_data["jobs"]))]
    machine_times = {m["id"]: 0 for m in jsp_data["machines"]}
    
    # Aktuelles Material auf jeder Maschine (für Umrüstzeiten)
    current_machine_material = {m["id"]: "" for m in jsp_data["machines"]}
    
    # Hilfsfunktion zum Prüfen, ob alle Vorgänger einer Operation abgeschlossen sind
    def are_predecessors_completed(job_id, op_id, completed_ops):
        job_idx = job_id_to_idx[job_id]
        operation = None
        
        # Finde die Operation
        for op in jsp_data["jobs"][job_idx]["operations"]:
            if op["id"] == op_id:
                operation = op
                break
        
        if operation is None:
            return False
        
        # Wenn keine Vorgänger definiert sind, kann die Operation ausgeführt werden
        if not operation["predecessors"]:
            return True
        
        # Prüfe alle Vorgänger
        for pred in operation["predecessors"]:
            if pred not in completed_ops:
                return False
        
        return True
    
    # Hilfsfunktion zum Berechnen der Umrüstzeit
    def calculate_setup_time(machine_id, new_material):
        current_material = current_machine_material[machine_id]
        
        # Wenn die Maschine noch kein Material bearbeitet hat, keine Umrüstzeit
        if current_material == "":
            return 0
        
        # Wenn das Material gleich bleibt, Standard-Umrüstzeit
        if current_material == new_material:
            return jsp_data["setupTimes"][machine_id]["standard"]
        else:
            # Bei Materialwechsel höhere Umrüstzeit
            return jsp_data["setupTimes"][machine_id]["materialChange"]
    
    # Verfolge abgeschlossene Operationen
    completed_operations = set()
    
    for action in actions:
        job_idx = action
        job = jsp_data["jobs"][job_idx]
        job_id = job["id"]
        op_idx = job_progress[job_idx]
        
        if op_idx >= len(job["operations"]):
            continue
            
        op = job["operations"][op_idx]
        op_id = op["id"]
        
        # Prüfe, ob alle Vorgänger abgeschlossen sind
        if not are_predecessors_completed(job_id, op_id, completed_operations):
            continue
        
        machine_id = op["machineId"]
        machine_idx = machine_id_to_idx[machine_id]
        proc_time = op["processingTime"]
        material = op["material"]
        
        # Berechne Umrüstzeit
        setup_time = calculate_setup_time(machine_id, material)
        
        # Frühestmögliche Startzeit für diese Operation
        earliest_start = max(machine_times[machine_id], job_progress[job_idx] > 0 and max(machine_times.values()) or 0)
        
        # Umrüstzeit hinzufügen
        earliest_start += setup_time
        
        # Operation einplanen
        end_time = earliest_start + proc_time
        
        schedule.append({
            "job_id": job_id,
            "operation_id": op_id,
            "machine_id": machine_id,
            "machine_idx": machine_idx,
            "start_time": earliest_start,
            "end_time": end_time,
            "setup_time": setup_time,
            "material": material,
            "priority": job["priority"],
            "deadline": job["deadline"]
        })
        
        # Markiere Operation als abgeschlossen
        completed_operations.add(f"{job_id}:{op_id}")
        
        # Verfügbarkeitszeiten aktualisieren
        machine_times[machine_id] = end_time
        
        # Aktuelles Material der Maschine aktualisieren
        current_machine_material[machine_id] = material
        
        # Fortschritt für diesen Job aktualisieren
        job_progress[job_idx] += 1
    
    # Gantt-Chart erstellen
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Farben für Jobs basierend auf Priorität (höhere Priorität = intensivere Farbe)
    priority_norm = plt.Normalize(1, 10)  # Priorität von 1-10
    cmap = plt.cm.get_cmap('viridis', len(jsp_data["jobs"]))
    
    # Y-Positionen für Maschinen
    y_positions = {m["id"]: i for i, m in enumerate(jsp_data["machines"])}
    
    for item in schedule:
        job_id = item["job_id"]
        job_idx = job_id_to_idx[job_id]
        machine_id = item["machine_id"]
        y_pos = y_positions[machine_id]
        start = item["start_time"]
        duration = item["end_time"] - item["start_time"]
        setup_time = item["setup_time"]
        priority = item["priority"]
        deadline = item["deadline"]
        
        # Farbe basierend auf Job und Priorität
        color_intensity = priority_norm(priority)
        color = cmap(job_idx)
        
        # Umrüstzeit als helleren Bereich darstellen
        if setup_time > 0:
            ax.barh(y_pos, setup_time, left=start-setup_time, color='lightgray', 
                    edgecolor='gray', alpha=0.5, hatch='/')
        
        # Hauptbalken für die Operation
        bar = ax.barh(y_pos, duration, left=start, color=color, 
                edgecolor='black', alpha=0.7)
        
        # Operation-ID in Bar anzeigen
        ax.text(start + duration/2, y_pos, f'{job_id}:{item["operation_id"]}', 
                ha='center', va='center', color='white', fontweight='bold')
        
        # Deadline-Markierung, falls vorhanden
        if deadline > 0 and item["end_time"] > deadline and job_progress[job_idx] == len(jsp_data["jobs"][job_idx]["operations"]):
            ax.plot([deadline, deadline], [y_pos-0.4, y_pos+0.4], color='red', linewidth=2)
            ax.text(deadline, y_pos+0.5, f'Deadline', color='red', ha='center', va='bottom', fontsize=8)
    
    # Achsenbeschriftungen
    ax.set_yticks(range(len(jsp_data["machines"])))
    ax.set_yticklabels([m["id"] for m in jsp_data["machines"]])
    ax.set_xlabel('Zeit')
    ax.set_ylabel('Maschine')
    ax.set_title('Job-Shop-Schedule mit Prioritäten, Deadlines und Umrüstzeiten')
    
    # Legende für Jobs mit Prioritäten
    handles = [plt.Rectangle((0,0), 1, 1, color=cmap(i), alpha=0.7) 
               for i in range(len(jsp_data["jobs"]))]
    labels = [f'{job["id"]} (Prio: {job["priority"]})' for job in jsp_data["jobs"]]
    ax.legend(handles, labels, loc='upper right', title='Jobs')
    
    # Grid anzeigen
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Makespan anzeigen
    makespan = max(item["end_time"] for item in schedule) if schedule else 0
    ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2)
    ax.text(makespan, -0.5, f'Makespan: {makespan}', color='red', ha='right', va='bottom', fontweight='bold')
    
    # Sicherstellen, dass der Ordner existiert
    os.makedirs('results/images', exist_ok=True)
    
    # Aktuelles Datum und Uhrzeit für den Dateinamen
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/images/jsp_schedule_{timestamp}.png'
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    
    print(f"Schedule-Visualisierung gespeichert unter: {filename}")
    
    # Prüfen, ob Deadlines eingehalten wurden
    for job_idx, job in enumerate(jsp_data["jobs"]):
        job_id = job["id"]
        deadline = job["deadline"]
        job_end_time = max([item["end_time"] for item in schedule if item["job_id"] == job_id], default=0)
        
        if deadline > 0 and job_end_time > deadline:
            print(f"Warnung: Deadline für Job {job_id} nicht eingehalten. "
                  f"Fertigstellung: {job_end_time}, Deadline: {deadline}")
    
    return schedule, makespan