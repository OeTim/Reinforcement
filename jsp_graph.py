import json
import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_jsp_graph(data):
    """
    Erstellt einen Graphen für ein JSP-Problem mit disjunktiven und konjunktiven Kanten.
    Unterstützt die neue Datenstruktur mit IDs wie "J1" und "OP1" sowie Vorgängerbeziehungen.
    """
    # Graph erstellen
    G = nx.DiGraph()
    
    # Dummy-Knoten für Start und Ende
    G.add_node("START", pos=(0, 0))
    G.add_node("END", pos=(10, 0))
    
    # Positions-Dictionary für das Layout
    pos = {"START": (0, 0), "END": (10, 0)}
    
    # Mapping für Maschinen-IDs zu Indizes erstellen
    machine_id_to_idx = {machine["id"]: idx for idx, machine in enumerate(data["machines"])}
    
    # Knoten hinzufügen (jede Operation ist ein Knoten)
    for job_idx, job in enumerate(data["jobs"]):
        job_id = job["id"]
        job_priority = job["priority"]
        job_deadline = job["deadline"]
        
        for op_idx, op in enumerate(job["operations"]):
            op_id = op["id"]
            machine_id = op["machineId"]
            machine_idx = machine_id_to_idx[machine_id]
            proc_time = op["processingTime"]
            material = op["material"]
            
            # Knoten-ID: J1:OP2 = Job J1, Operation OP2
            node_id = f"{job_id}:{op_id}"
            
            # Position im Layout (x = Operation #, y = Job #)
            # Verwende op_idx und job_idx für die Position, da diese numerisch sind
            pos[node_id] = (2 * (op_idx + 1), -(job_idx + 1) * 2)
            
            # Knoten mit Attributen hinzufügen
            G.add_node(node_id, 
                      job=job_idx + 1,  # Numerischer Wert für die Visualisierung
                      operation=op_idx + 1,  # Numerischer Wert für die Visualisierung
                      machine=machine_idx + 1,  # Numerischer Wert für die Visualisierung
                      time=proc_time,
                      job_id=job_id,  # Original-ID
                      op_id=op_id,    # Original-ID
                      machine_id=machine_id,  # Original-ID
                      priority=job_priority,
                      deadline=job_deadline,
                      material=material)
    
    # Kanten basierend auf Vorgängerbeziehungen hinzufügen
    for job in data["jobs"]:
        job_id = job["id"]
        
        # Operationen ohne Vorgänger mit START verbinden
        for op in job["operations"]:
            if not op["predecessors"]:
                G.add_edge("START", f"{job_id}:{op['id']}", color='blue', style='solid', weight=2)
        
        # Vorgängerbeziehungen hinzufügen
        for op in job["operations"]:
            op_id = op["id"]
            
            # Für jeden Vorgänger eine Kante hinzufügen
            for pred in op["predecessors"]:
                G.add_edge(pred, f"{job_id}:{op_id}", color='blue', style='solid', weight=2)
            
            # Wenn keine Nachfolger vorhanden sind, mit END verbinden
            has_successor = False
            for other_op in job["operations"]:
                if f"{job_id}:{op_id}" in other_op["predecessors"]:
                    has_successor = True
                    break
            
            if not has_successor:
                G.add_edge(f"{job_id}:{op_id}", "END", color='blue', style='solid', weight=2)
    
    # Disjunktive Kanten hinzufügen (Operationen auf der gleichen Maschine)
    # Diese stellen potenzielle Konflikte zwischen Operationen dar
    for machine_id in machine_id_to_idx.keys():
        # Operationen auf dieser Maschine finden
        ops_on_machine = []
        
        for job in data["jobs"]:
            job_id = job["id"]
            for op in job["operations"]:
                if op["machineId"] == machine_id:
                    ops_on_machine.append((job_id, op["id"]))
        
        # Disjunktive Kanten zwischen allen Operationen auf dieser Maschine
        for i in range(len(ops_on_machine)):
            for j in range(i+1, len(ops_on_machine)):
                job1, op1 = ops_on_machine[i]
                job2, op2 = ops_on_machine[j]
                
                # Bidirektionale Kante (beide Operationen können nicht gleichzeitig laufen)
                G.add_edge(f"{job1}:{op1}", f"{job2}:{op2}", 
                          color='red', style='dashed', weight=1)
                G.add_edge(f"{job2}:{op2}", f"{job1}:{op1}", 
                          color='red', style='dashed', weight=1)
    
    return G, pos

def visualize_jsp_graph(G, pos):
    """
    Visualisiert den JSP-Graphen mit unterschiedlichen Farben für disjunktive und konjunktive Kanten.
    """
    plt.figure(figsize=(12, 8))
    
    # Knoten zeichnen
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    
    # Kanten nach Typ gruppieren
    blue_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('color') == 'blue']
    red_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('color') == 'red']
    
    # Konjunktive Kanten zeichnen (blau, durchgezogen)
    nx.draw_networkx_edges(G, pos, edgelist=blue_edges, 
                         width=2, edge_color='blue', arrows=True)
    
    # Disjunktive Kanten zeichnen (rot, gestrichelt)
    nx.draw_networkx_edges(G, pos, edgelist=red_edges, 
                         width=1, edge_color='red', style='dashed', arrows=True,
                         connectionstyle='arc3,rad=0.1')  # Gebogene Linien für bessere Sichtbarkeit
    
    # Labels
    nx.draw_networkx_labels(G, pos)
    
    # Legende
    plt.plot([0], [0], color='blue', linewidth=2, label='Konjunktiv (Job-Reihenfolge)')
    plt.plot([0], [0], color='red', linewidth=1, linestyle='--', label='Disjunktiv (Maschinen-Konflikte)')
    plt.legend()
    
    plt.title("Job-Shop-Scheduling als Graph mit disjunktiven und konjunktiven Kanten")
    plt.axis('off')
    
    # Sicherstellen, dass der Ordner existiert
    os.makedirs('results/images', exist_ok=True)
    
    # Aktuelles Datum und Uhrzeit für den Dateinamen
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/images/jsp_graph_{timestamp}.png'
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    
    print(f"JSP-Graph gespeichert unter: {filename}")

# Hauptcode
if __name__ == "__main__":
    # JSP-Daten aus JSON-Datei laden
    try:
        with open('data.json', 'r') as file:
            data = json.load(file)
        print("Daten erfolgreich aus data.json geladen.")
    except FileNotFoundError:
        print("Fehler: data.json nicht gefunden. Stelle sicher, dass die Datei im gleichen Verzeichnis liegt.")
        exit(1)
    except json.JSONDecodeError:
        print("Fehler: Die Datei data.json enthält kein gültiges JSON-Format.")
        exit(1)
    
    # Graph erstellen
    G, pos = create_jsp_graph(data)
    
    # Graph visualisieren
    visualize_jsp_graph(G, pos)