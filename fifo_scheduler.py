import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from datetime import datetime
from gym_environment import JSPGymEnvironment
from torch_ppo_agent import TorchPPOAgent  # Geändert von ppo_agent import PPOAgent

def fifo_schedule(jsp_data_path):
    """
    Implementiert einen einfachen FIFO-Scheduler für das JSP-Problem.
    Berücksichtigt Vorgängerbeziehungen, Prioritäten, Deadlines und Umrüstzeiten.
    
    Args:
        jsp_data_path: Pfad zur JSP-Datendatei
        
    Returns:
        schedule: Dictionary mit Zeitplan (Maschine -> Liste von Operationen)
        makespan: Gesamtdauer des Zeitplans
        met_deadlines: Anzahl der eingehaltenen Deadlines
        machine_utilization: Durchschnittliche Maschinenauslastung
    """
    # Daten laden
    with open(jsp_data_path, 'r') as file:
        data = json.load(file)
    
    # Initialisierung
    num_jobs = len(data["jobs"])
    
    # Mapping von Job-IDs zu Indizes erstellen
    job_id_to_idx = {job["id"]: idx for idx, job in enumerate(data["jobs"])}
    
    # Mapping von Maschinen-IDs zu Indizes erstellen
    machine_id_to_idx = {machine["id"]: idx for idx, machine in enumerate(data["machines"])}
    
    # Fortschritt für jeden Job (welche Operation als nächstes)
    job_progress = [0] * num_jobs
    
    # Verfügbarkeitszeit für jede Maschine
    machine_available_time = {m["id"]: 0 for m in data["machines"]}
    
    # Aktuelles Material auf jeder Maschine (für Umrüstzeiten)
    current_machine_material = {m["id"]: "" for m in data["machines"]}
    
    # Fertigstellungszeit für jeden Job
    job_completion_time = [0] * num_jobs
    
    # Schedule (für jede Maschine eine Liste von Operationen mit Start- und Endzeit)
    schedule = {m["id"]: [] for m in data["machines"]}
    
    # Hilfsfunktion zum Prüfen, ob alle Vorgänger einer Operation abgeschlossen sind
    def are_predecessors_completed(job_id, op_id):
        job_idx = job_id_to_idx[job_id]
        op_idx = None
        
        # Finde den Index der Operation
        for i, op in enumerate(data["jobs"][job_idx]["operations"]):
            if op["id"] == op_id:
                op_idx = i
                break
        
        if op_idx is None:
            return False
        
        operation = data["jobs"][job_idx]["operations"][op_idx]
        
        # Wenn keine Vorgänger definiert sind, kann die Operation ausgeführt werden
        if not operation["predecessors"]:
            return True
        
        # Prüfe alle Vorgänger
        for pred in operation["predecessors"]:
            # Format der Vorgänger ist "J1:OP1"
            pred_job_id, pred_op_id = pred.split(":")
            pred_job_idx = job_id_to_idx[pred_job_id]
            
            # Finde den Index der Vorgängeroperation
            pred_op_idx = None
            for i, op in enumerate(data["jobs"][pred_job_idx]["operations"]):
                if op["id"] == pred_op_id:
                    pred_op_idx = i
                    break
            
            if pred_op_idx is None:
                return False  # Vorgängeroperation nicht gefunden
            
            # Prüfe, ob die Vorgängeroperation abgeschlossen ist
            if job_progress[pred_job_idx] <= pred_op_idx:
                return False  # Vorgängeroperation noch nicht abgeschlossen
        
        return True  # Alle Vorgänger sind abgeschlossen
    
    # Hilfsfunktion zum Berechnen der Umrüstzeit
    def calculate_setup_time(machine_id, new_material):
        current_material = current_machine_material[machine_id]
        
        # Wenn die Maschine noch kein Material bearbeitet hat, keine Umrüstzeit
        if current_material == "":
            return 0
        
        # Wenn das Material gleich bleibt, Standard-Umrüstzeit
        if current_material == new_material:
            return data["setupTimes"][machine_id]["standard"]
        else:
            # Bei Materialwechsel höhere Umrüstzeit
            return data["setupTimes"][machine_id]["materialChange"]
    
    # FIFO-Reihenfolge: Wir gehen die Jobs der Reihe nach durch
    # und planen jeweils die nächste Operation ein, wenn alle Vorgänger abgeschlossen sind
    operations_completed = 0
    total_operations = sum(len(job["operations"]) for job in data["jobs"])
    
    while operations_completed < total_operations:
        progress_made = False
        
        # Sortiere Jobs nach Priorität (höhere Priorität zuerst)
        sorted_jobs = sorted(range(num_jobs), key=lambda j: data["jobs"][j]["priority"], reverse=True)
        
        for job_idx in sorted_jobs:
            job = data["jobs"][job_idx]
            job_id = job["id"]
            
            # Wenn dieser Job noch nicht fertig ist
            if job_progress[job_idx] < len(job["operations"]):
                # Nächste Operation dieses Jobs
                op_idx = job_progress[job_idx]
                operation = job["operations"][op_idx]
                op_id = operation["id"]
                
                # Prüfe, ob alle Vorgänger abgeschlossen sind
                if are_predecessors_completed(job_id, op_id):
                    machine_id = operation["machineId"]
                    proc_time = operation["processingTime"]
                    material = operation["material"]
                    
                    # Berechne Umrüstzeit
                    setup_time = calculate_setup_time(machine_id, material)
                    
                    # Frühestmögliche Startzeit für diese Operation
                    earliest_start = max(machine_available_time[machine_id], job_completion_time[job_idx])
                    
                    # Umrüstzeit hinzufügen
                    earliest_start += setup_time
                    
                    # Operation einplanen
                    end_time = earliest_start + proc_time
                    
                    # Schedule aktualisieren
                    schedule[machine_id].append({
                        "job_id": job_id,
                        "operation_id": op_id,
                        "start_time": earliest_start,
                        "end_time": end_time,
                        "setup_time": setup_time,
                        "material": material,
                        "priority": job["priority"],
                        "deadline": job["deadline"]
                    })
                    
                    # Verfügbarkeitszeiten aktualisieren
                    machine_available_time[machine_id] = end_time
                    job_completion_time[job_idx] = end_time
                    
                    # Aktuelles Material der Maschine aktualisieren
                    current_machine_material[machine_id] = material
                    
                    # Fortschritt für diesen Job aktualisieren
                    job_progress[job_idx] += 1
                    operations_completed += 1
                    progress_made = True
        
        # Wenn in diesem Durchlauf kein Fortschritt gemacht wurde, gibt es eine Blockierung
        if not progress_made:
            print("Warnung: Keine Operation konnte eingeplant werden. Möglicherweise gibt es zyklische Abhängigkeiten.")
            break
    
    # Makespan berechnen (maximale Endzeit aller Operationen)
    makespan = max(job_completion_time)
    
    # Prüfen, ob Deadlines eingehalten wurden
    met_deadlines = 0
    for job_idx, job in enumerate(data["jobs"]):
        if job_completion_time[job_idx] <= job["deadline"]:
            met_deadlines += 1
        else:
            print(f"Warnung: Deadline für Job {job['id']} nicht eingehalten. "
                  f"Fertigstellung: {job_completion_time[job_idx]}, Deadline: {job['deadline']}")
    
    # Berechne Maschinenauslastung
    total_processing_time = sum(op["end_time"] - op["start_time"] - op["setup_time"] 
                               for ops in schedule.values() for op in ops)
    total_machine_time = sum(max([op["end_time"] for op in ops] or [0]) for ops in schedule.values())
    machine_utilization = total_processing_time / total_machine_time if total_machine_time > 0 else 0
    
    return schedule, makespan, met_deadlines, machine_utilization

def run_rl_model(model_path, jsp_data_path):
    """
    Führt ein trainiertes RL-Modell auf dem JSP-Problem aus.
    
    Args:
        model_path: Pfad zum gespeicherten Modell
        jsp_data_path: Pfad zur JSP-Datendatei
        
    Returns:
        action_sequence: Sequenz der ausgewählten Jobs
        makespan: Gesamtdauer des Zeitplans
        met_deadlines: Anzahl der eingehaltenen Deadlines
        machine_utilization: Durchschnittliche Maschinenauslastung
    """
    # Daten laden
    with open(jsp_data_path, 'r') as file:
        data = json.load(file)
    
    # Umgebung erstellen - Versuche beide Möglichkeiten
    try:
        # Versuche zuerst, das geladene JSON-Objekt zu übergeben
        env = JSPGymEnvironment(data)
    except TypeError:
        try:
            # Wenn das nicht funktioniert, versuche den Dateipfad zu übergeben
            env = JSPGymEnvironment(jsp_data_path)
        except Exception as e:
            print(f"Fehler beim Erstellen der Umgebung: {e}")
            raise
    
    # Modell laden
    agent = TorchPPOAgent(num_jobs=len(data["jobs"]), jsp_data=data)
    
    # Lade das gespeicherte Modell
    checkpoint = torch.load(model_path)
    agent.node_embedding.load_state_dict(checkpoint['node_embedding'])
    agent.graph_layer1.load_state_dict(checkpoint['graph_layer1'])
    agent.graph_layer2.load_state_dict(checkpoint['graph_layer2'])
    agent.output_layer.load_state_dict(checkpoint['output_layer'])
    
    # Umgebung zurücksetzen
    state = env.reset()
    done = False
    action_sequence = []
    total_reward = 0
    
    # Führe Aktionen aus, bis alle Jobs abgeschlossen sind
    while not done:
        # Aktion auswählen (ohne Exploration)
        action, _ = agent.select_action(state)
        action_sequence.append(action)
        
        # Aktion ausführen
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Zustand aktualisieren
        state = next_state
    
    # Ergebnisse extrahieren
    makespan = env.episode_makespan
    met_deadlines = env.episode_met_deadlines
    
    # Maschinenauslastung berechnen
    total_proc_time = sum(sum(op["processingTime"] for op in job["operations"]) 
                          for job in env.jobs)
    machine_utilization = total_proc_time / (makespan * env.num_machines)
    
    return action_sequence, makespan, met_deadlines, machine_utilization, total_reward

def find_best_model(models_dir="results/models"):
    """
    Findet das beste Modell basierend auf dem Checkpoint mit dem niedrigsten Makespan.
    
    Args:
        models_dir: Verzeichnis mit den gespeicherten Modellen
        
    Returns:
        str: Pfad zum besten Modell
    """
    best_model = None
    best_makespan = float('inf')
    
    # Suche nach Checkpoint-Dateien
    for filename in os.listdir(models_dir):
        if (filename.startswith("gym_ppo_checkpoint_") or filename.startswith("torch_ppo_")) and filename.endswith(".pt"):
            # Teste das Modell
            model_path = os.path.join(models_dir, filename)
            try:
                _, makespan, _, _, _ = run_rl_model(model_path, "data.json")
                
                # Aktualisiere das beste Modell, wenn ein besserer Makespan gefunden wurde
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_model = model_path
                    print(f"Neues bestes Modell gefunden: {filename} mit Makespan {makespan}")
            except Exception as e:
                print(f"Fehler beim Testen von {filename}: {e}")
    
    return best_model

def visualize_comparison(fifo_results, rl_results):
    """
    Visualisiert den Vergleich zwischen FIFO und RL-Scheduler.
    
    Args:
        fifo_results: Ergebnisse des FIFO-Schedulers (makespan, met_deadlines, utilization)
        rl_results: Ergebnisse des RL-Schedulers (makespan, met_deadlines, utilization)
    """
    labels = ['FIFO', 'RL']
    
    # Makespan (niedriger ist besser)
    makespan_data = [fifo_results[0], rl_results[0]]
    
    # Eingehaltene Deadlines (höher ist besser)
    deadlines_data = [fifo_results[1], rl_results[1]]
    
    # Maschinenauslastung (höher ist besser)
    utilization_data = [fifo_results[2] * 100, rl_results[2] * 100]  # In Prozent
    
    # Erstelle Subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Makespan-Plot
    ax1.bar(labels, makespan_data, color=['blue', 'green'])
    ax1.set_title('Makespan (niedriger ist besser)')
    ax1.set_ylabel('Zeit')
    for i, v in enumerate(makespan_data):
        ax1.text(i, v + 50, f"{v:.1f}", ha='center')
    
    # Deadlines-Plot
    ax2.bar(labels, deadlines_data, color=['blue', 'green'])
    ax2.set_title('Eingehaltene Deadlines (höher ist besser)')
    ax2.set_ylabel('Anzahl')
    for i, v in enumerate(deadlines_data):
        ax2.text(i, v + 0.1, f"{v}", ha='center')
    
    # Auslastungs-Plot
    ax3.bar(labels, utilization_data, color=['blue', 'green'])
    ax3.set_title('Maschinenauslastung (höher ist besser)')
    ax3.set_ylabel('Prozent')
    for i, v in enumerate(utilization_data):
        ax3.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    
    # Speichere das Diagramm
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/images/fifo_vs_rl_comparison_{timestamp}.png")
    print(f"Vergleichsdiagramm gespeichert unter: results/images/fifo_vs_rl_comparison_{timestamp}.png")
    
    # Zeige das Diagramm
    plt.show()

def compare_schedulers(model_path=None, jsp_data_path="data.json"):
    """
    Vergleicht den FIFO-Scheduler mit dem RL-Scheduler.
    
    Args:
        model_path: Pfad zum RL-Modell (wenn None, wird das beste Modell gesucht)
        jsp_data_path: Pfad zur JSP-Datendatei
    """
    print("\n=== Vergleich: FIFO vs. RL-Scheduler ===\n")
    
    # FIFO-Scheduler ausführen
    print("Führe FIFO-Scheduler aus...")
    fifo_schedule_result, fifo_makespan, fifo_met_deadlines, fifo_utilization = fifo_schedule(jsp_data_path)
    
    # Bestes Modell finden, falls nicht angegeben
    if model_path is None:
        print("\nSuche nach dem besten RL-Modell...")
        model_path = find_best_model()
        if model_path is None:
            print("Kein gültiges RL-Modell gefunden!")
            return
    
    # RL-Scheduler ausführen
    print(f"\nFühre RL-Scheduler mit Modell {os.path.basename(model_path)} aus...")
    rl_action_sequence, rl_makespan, rl_met_deadlines, rl_utilization, rl_reward = run_rl_model(model_path, jsp_data_path)
    
    # Ergebnisse anzeigen
    print("\n=== Ergebnisse ===")
    print(f"FIFO Makespan: {fifo_makespan}")
    print(f"RL Makespan: {rl_makespan}")
    print(f"Verbesserung: {((fifo_makespan - rl_makespan) / fifo_makespan) * 100:.2f}%")
    
    print(f"\nFIFO Eingehaltene Deadlines: {fifo_met_deadlines}/10")
    print(f"RL Eingehaltene Deadlines: {rl_met_deadlines}/10")
    
    print(f"\nFIFO Maschinenauslastung: {fifo_utilization:.2f} ({fifo_utilization*100:.1f}%)")
    print(f"RL Maschinenauslastung: {rl_utilization:.2f} ({rl_utilization*100:.1f}%)")
    
    print(f"\nRL Gesamtbelohnung: {rl_reward:.2f}")
    print(f"RL Aktionssequenz: {rl_action_sequence}")
    
    # Visualisiere den Vergleich
    visualize_comparison(
        (fifo_makespan, fifo_met_deadlines, fifo_utilization),
        (rl_makespan, rl_met_deadlines, rl_utilization)
    )

if __name__ == "__main__":
    # Vergleiche FIFO mit dem besten RL-Modell
    compare_schedulers("results/models/gym_ppo_model_20250313_114630.pt")
    
    # Alternativ: Vergleiche FIFO mit einem bestimmten Modell
    # compare_schedulers("results/models/gym_ppo_model_20250313_114630.pt")
