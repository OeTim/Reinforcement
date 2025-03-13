import torch
import numpy as np
from jsp_graph import create_jsp_graph


class TorchPPOAgent:
    def __init__(self, num_jobs, jsp_data):
        self.num_jobs = num_jobs
        self.jsp_data = jsp_data
        
        # Erstelle Mapping von Job-IDs zu Indizes und umgekehrt
        self.job_id_to_idx = {job["id"]: idx for idx, job in enumerate(jsp_data["jobs"])}
        self.idx_to_job_id = {idx: job["id"] for idx, job in enumerate(jsp_data["jobs"])}
        
        # Erstelle Mapping von Maschinen-IDs zu Indizes und umgekehrt
        self.machine_id_to_idx = {machine["id"]: idx for idx, machine in enumerate(jsp_data["machines"])}
        self.idx_to_machine_id = {idx: machine["id"] for idx, machine in enumerate(jsp_data["machines"])}
        
        # JSP-Graph aus jsp_graph.py verwenden
        self.G, _ = create_jsp_graph(jsp_data)
        # Graph für das Lernen vorbereiten
        self.G = self.prepare_graph_for_learning(self.G)
        
        # Anzahl der Features für jeden Knoten
        # job_id, op_id, machine_id, processing_time, priority, deadline, material_type
        node_features = 7
        
        # Einfaches Graph-basiertes Modell
        self.embedding_dim = 64   # Erhöht für mehr Kapazität
        self.hidden_dim = 128     # Erhöht für mehr Kapazität
        
        # Embedding-Layer für Knoten-Features
        self.node_embedding = torch.nn.Linear(node_features, self.embedding_dim)
        
        # Graph Attention Layer (einfache Version)
        self.graph_layer1 = torch.nn.Linear(self.embedding_dim, self.hidden_dim)
        self.graph_layer2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Output-Layer
        self.output_layer = torch.nn.Linear(self.hidden_dim, num_jobs)
        
        # Optimizer für alle Parameter des Graph-Modells mit angepasster Lernrate
        self.optimizer = torch.optim.Adam(list(self.node_embedding.parameters()) + 
                                      list(self.graph_layer1.parameters()) + 
                                      list(self.graph_layer2.parameters()) + 
                                      list(self.output_layer.parameters()), 
                                      lr=0.001,  # Noch niedrigere Lernrate für stabileres Training
                                      weight_decay=1e-5)  # L2-Regularisierung zur Vermeidung von Overfitting
        
        # PPO-Parameter
        self.epsilon = 0.3  # Increased clipping parameter for more exploration
        self.gamma = 0.99  # Discount-Faktor
        
        # Verbesserte Exploration-Parameter
        self.exploration_rate = 0.8  # Higher initial exploration rate
        self.exploration_decay = 0.9998  # Slower decay for longer exploration
        self.min_exploration_rate = 0.15  # Higher minimum exploration rate
        
        # Temperature parameter for Boltzmann exploration
        self.temperature = 2.5  # Higher temperature means more exploration
        
        # Speicher für Erfahrungen
        self.experiences = []
        
        # Extrahiere alle eindeutigen Materialtypen für One-Hot-Encoding
        self.materials = set()
        for job in jsp_data["jobs"]:
            for op in job["operations"]:
                self.materials.add(self._extract_material_type(op["material"]))
        self.materials = list(self.materials)
    
    def _extract_material_type(self, material_string):
        """Extrahiert den Materialtyp aus dem Material-String (z.B. 'Citybike' aus 'Citybike_Hauptrahmen_zugeschnitten')"""
        if "_" in material_string:
            return material_string.split("_")[0]
        return material_string
    
    def _get_material_index(self, material_string):
        """Gibt den Index des Materialtyps zurück"""
        material_type = self._extract_material_type(material_string)
        if material_type in self.materials:
            return self.materials.index(material_type)
        return 0  # Fallback
    
    def prepare_graph_for_learning(self, G):
        """Bereitet den Graphen für das Lernen vor, indem edge_type Attribute hinzugefügt werden"""
        # Konjunktive Kanten (blau) bekommen edge_type=1
        # Disjunktive Kanten (rot) bekommen edge_type=2
        for u, v, data in G.edges(data=True):
            if data.get('color') == 'blue':
                G.edges[u, v]['edge_type'] = 1
            elif data.get('color') == 'red':
                G.edges[u, v]['edge_type'] = 2
        
        return G
    
    def state_to_tensor(self, state):
        """Konvertiert den Zustand in einen Tensor für das Netzwerk unter Berücksichtigung des Graphen"""
        # Extrahiere Zustandsinformationen
        job_progress = state['job_progress']
        machine_times = state['machine_times']
        current_machine_material = state.get('current_machine_material', [""] * len(self.jsp_data["machines"]))
        
        # Aktualisiere den Graphen mit dem aktuellen Zustand
        node_features = {}
        
        # Für jeden Job
        for job_idx, job_id in self.idx_to_job_id.items():
            job = self.jsp_data["jobs"][job_idx]
            progress = job_progress[job_idx]
            
            # Für alle Operationen dieses Jobs
            for op_idx, operation in enumerate(job["operations"]):
                node_id = f"{job_id}:{operation['id']}"
                
                if node_id in self.G.nodes:
                    # Extrahiere Maschinendaten
                    machine_id = operation["machineId"]
                    machine_idx = self.machine_id_to_idx[machine_id]
                    
                    # Extrahiere Material
                    material = operation["material"]
                    material_idx = self._get_material_index(material)
                    
                    # Ist diese Operation bereits abgeschlossen?
                    completed = 1.0 if progress > op_idx else 0.0
                    
                    # Normalisiere Features
                    normalized_feature = [
                        job_idx / self.num_jobs,                                # Job-Index
                        op_idx / max(1, len(job["operations"])),                # Operation-Index
                        machine_idx / len(self.jsp_data["machines"]),          # Maschinen-Index
                        operation["processingTime"] / 100.0,                    # Bearbeitungszeit
                        job["priority"] / 10.0,                                 # Priorität
                        job["deadline"] / 200.0,                                # Deadline
                        material_idx / max(1, len(self.materials))             # Material-Typ
                    ]
                    
                    node_features[node_id] = torch.tensor(normalized_feature, dtype=torch.float32)
        
        # Einfaches Message-Passing auf dem Graphen
        node_embeddings = {}
        for node_id, features in node_features.items():
            # Initialer Embedding für jeden Knoten
            node_embeddings[node_id] = self.node_embedding(features)
        
        # Aggregiere Nachbarn (einfache Mittelwertbildung)
        aggregated_features = {}
        for node_id in node_features.keys():
            neighbors = list(self.G.neighbors(node_id))
            if neighbors:
                neighbor_embeds = [node_embeddings.get(n, torch.zeros(self.embedding_dim)) for n in neighbors]
                if neighbor_embeds:
                    aggregated_features[node_id] = torch.mean(torch.stack(neighbor_embeds), dim=0)
                else:
                    aggregated_features[node_id] = torch.zeros(self.embedding_dim)
            else:
                aggregated_features[node_id] = torch.zeros(self.embedding_dim)
        
        # Kombiniere zu einem Gesamtembedding für den Zustand
        job_embeddings = []
        for job_idx, job_id in self.idx_to_job_id.items():
            job = self.jsp_data["jobs"][job_idx]
            
            # Finde alle Knoten für diesen Job
            job_nodes = [f"{job_id}:{op['id']}" for op in job["operations"]]
            job_node_embeds = [node_embeddings.get(n, torch.zeros(self.embedding_dim)) for n in job_nodes if n in node_embeddings]
            
            if job_node_embeds:
                job_embed = torch.mean(torch.stack(job_node_embeds), dim=0)
            else:
                job_embed = torch.zeros(self.embedding_dim)
            
            job_embeddings.append(job_embed)
        
        # Kombiniere alle Job-Embeddings zu einem Zustandsvektor
        if job_embeddings:
            return torch.mean(torch.stack(job_embeddings), dim=0)
        else:
            return torch.zeros(self.embedding_dim)
    
    def select_action(self, state):
        """Wählt eine Aktion basierend auf dem aktuellen Zustand unter Verwendung des Graph-Modells"""
        # Welche Jobs können noch ausgeführt werden?
        valid_jobs = []
        
        # Check if we have a valid_actions_mask in the state (from Gym environment)
        if 'valid_actions_mask' in state:
            for job_idx, is_valid in enumerate(state['valid_actions_mask']):
                if is_valid == 1:
                    valid_jobs.append(job_idx)
        else:
            # Original logic for SimpleEnvironment
            for job_idx in range(self.num_jobs):
                # Prüfe, ob der Job noch nicht abgeschlossen ist
                if state['job_progress'][job_idx] < len(self.jsp_data["jobs"][job_idx]["operations"]):
                    # Prüfe, ob die nächste Operation ausführbar ist (Vorgänger abgeschlossen)
                    op_idx = state['job_progress'][job_idx]
                    job_id = self.idx_to_job_id[job_idx]
                    operation = self.jsp_data["jobs"][job_idx]["operations"][op_idx]
                    
                    # Prüfe Vorgänger
                    predecessors_completed = True
                    for pred in operation["predecessors"]:
                        # Format der Vorgänger ist "J1:OP1"
                        pred_job_id, pred_op_id = pred.split(":")
                        pred_job_idx = self.job_id_to_idx[pred_job_id]
                        
                        # Finde den Index der Vorgängeroperation
                        pred_op_idx = None
                        for i, op in enumerate(self.jsp_data["jobs"][pred_job_idx]["operations"]):
                            if op["id"] == pred_op_id:
                                pred_op_idx = i
                                break
                        
                        if pred_op_idx is None or state['job_progress'][pred_job_idx] <= pred_op_idx:
                            predecessors_completed = False
                            break
                    
                    if predecessors_completed:
                        valid_jobs.append(job_idx)
        
        if not valid_jobs:
            # Wenn keine Jobs ausführbar sind, wähle einen zufälligen nicht abgeschlossenen Job
            for job_idx in range(self.num_jobs):
                if state['job_progress'][job_idx] < len(self.jsp_data["jobs"][job_idx]["operations"]):
                    valid_jobs.append(job_idx)
            
            if not valid_jobs:
                return 0, 1.0  # Sollte nicht vorkommen, alle Jobs sind abgeschlossen
        
        # Verbesserte Exploration mit Epsilon-Greedy-Strategie und Annealing
        import random
        import math
        
        # Berechne aktuelle Exploration-Rate mit Annealing
        # Exploration-Rate nimmt mit der Zeit ab (simuliert durch die Länge der Erfahrungen)
        if hasattr(self, 'experiences') and len(self.experiences) > 0:
            # Reduziere Exploration-Rate exponentiell mit der Anzahl der gesammelten Erfahrungen
            # Minimum self.min_exploration_rate für eine Basis-Exploration
            self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)
            
            # Periodically boost exploration to escape local optima
            if len(self.experiences) % 500 == 0:
                self.exploration_rate = min(0.8, self.exploration_rate * 1.5)  # Boost exploration
        
        # Enhanced exploration strategy
        if random.random() < self.exploration_rate:
            # Use a mix of strategies for better exploration
            strategy = random.random()
            
            if strategy < 0.3:  # 30% completely random
                chosen_job = random.choice(valid_jobs)
            elif strategy < 0.7:  # 40% Boltzmann exploration
                # Calculate Boltzmann distribution over actions with higher temperature
                temperature = self.temperature  # Higher values = more exploration
                
                # Berechne Logits und Wahrscheinlichkeiten
                state_embedding = self.state_to_tensor(state)
                with torch.no_grad():
                    hidden = torch.nn.functional.relu(self.graph_layer1(state_embedding))
                    hidden = torch.nn.functional.relu(self.graph_layer2(hidden))
                    logits = self.output_layer(hidden)
                    
                    # Boltzmann-Verteilung mit Temperatur
                    scaled_logits = logits / temperature
                    probs = torch.nn.functional.softmax(scaled_logits, dim=0)
                    
                    # Maskiere ungültige Jobs
                    valid_mask = torch.zeros(self.num_jobs)
                    for job_idx in valid_jobs:
                        valid_mask[job_idx] = 1.0
                    
                    masked_probs = probs * valid_mask
                    if torch.sum(masked_probs) > 0:
                        masked_probs = masked_probs / torch.sum(masked_probs)
                    else:
                        # Fallback: Gleichverteilung über gültige Jobs
                        masked_probs = torch.tensor([1.0/len(valid_jobs) if i in valid_jobs else 0.0 
                                                   for i in range(self.num_jobs)])
                    
                    # Wähle Job basierend auf Boltzmann-Verteilung
                    chosen_job = torch.multinomial(masked_probs, 1).item()
            else:  # remaining 30% - use a different strategy
                # Use UCB-like exploration (Upper Confidence Bound)
                state_embedding = self.state_to_tensor(state)
                with torch.no_grad():
                    hidden = torch.nn.functional.relu(self.graph_layer1(state_embedding))
                    hidden = torch.nn.functional.relu(self.graph_layer2(hidden))
                    logits = self.output_layer(hidden)
                    
                    # Add exploration bonus based on visitation frequency
                    if hasattr(self, 'action_counts'):
                        # Calculate UCB scores
                        total_actions = sum(self.action_counts.values())
                        exploration_bonus = [2.0 * math.sqrt(math.log(total_actions + 1) / (self.action_counts.get(i, 1)))
                                            for i in range(self.num_jobs)]
                        ucb_scores = logits + torch.tensor(exploration_bonus, dtype=torch.float32)
                        
                        # Mask invalid actions
                        valid_mask = torch.zeros(self.num_jobs)
                        for job_idx in valid_jobs:
                            valid_mask[job_idx] = 1.0
                        
                        masked_scores = ucb_scores * valid_mask
                        if valid_jobs:  # Ensure there are valid jobs
                            chosen_job = torch.argmax(masked_scores).item()
                        else:
                            chosen_job = 0  # Fallback
                    else:
                        # Initialize action counts if not present
                        self.action_counts = {i: 1 for i in range(self.num_jobs)}
                        chosen_job = random.choice(valid_jobs)
            
            # Update action counts for UCB strategy
            if not hasattr(self, 'action_counts'):
                self.action_counts = {i: 1 for i in range(self.num_jobs)}
            self.action_counts[chosen_job] = self.action_counts.get(chosen_job, 0) + 1
            
            # Get probabilities for the chosen job
            state_embedding = self.state_to_tensor(state)
            with torch.no_grad():
                hidden = torch.nn.functional.relu(self.graph_layer1(state_embedding))
                hidden = torch.nn.functional.relu(self.graph_layer2(hidden))
                logits = self.output_layer(hidden)
                probs = torch.nn.functional.softmax(logits, dim=0)
            
            return chosen_job, probs[chosen_job].item()
        
        # Umwandeln des Zustands in einen Tensor mit Graph-Informationen
        state_embedding = self.state_to_tensor(state)
        
        # Forward-Pass durch das Graph-Modell
        with torch.no_grad():
            # Graph Neural Network Verarbeitung
            hidden = torch.nn.functional.relu(self.graph_layer1(state_embedding))
            hidden = torch.nn.functional.relu(self.graph_layer2(hidden))
            
            # Ausgabe-Layer für Aktionswahrscheinlichkeiten
            logits = self.output_layer(hidden)
            
            # Softmax, um Wahrscheinlichkeiten zu erhalten
            probs = torch.nn.functional.softmax(logits, dim=0)
        
        # Nur valide Jobs berücksichtigen
        valid_probs = torch.tensor([probs[i].item() if i in valid_jobs else 0.0 for i in range(self.num_jobs)])
        
        # Normalisieren
        if torch.sum(valid_probs) > 0:
            valid_probs = valid_probs / torch.sum(valid_probs)
        else:
            valid_probs = torch.tensor([1.0/len(valid_jobs) if i in valid_jobs else 0.0 for i in range(self.num_jobs)])
        
        # Aktion wählen
        action = torch.multinomial(valid_probs, 1).item()
        
        # Speichere Wahrscheinlichkeit für das Update
        action_prob = valid_probs[action].item()
        
        return action, action_prob
    
    def store_experience(self, state, action, action_prob, reward, next_state, done):
        """Speichert eine Erfahrung für späteres Training"""
        self.experiences.append({
            'state': state,
            'action': action,
            'action_prob': action_prob,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def update(self, batch_size=32):
        """Führt ein PPO-Update basierend auf gesammelten Erfahrungen durch"""
        if len(self.experiences) < batch_size:  # Flexible Batch-Größe für stabileres Training
            return 0.0
        
        # Berechne Vorteile und Rewards-to-go mit Generalized Advantage Estimation (GAE)
        rewards = [exp['reward'] for exp in self.experiences]
        dones = [exp['done'] for exp in self.experiences]
        returns = []
        advantages = []
        
        # Parameter für GAE
        lambda_gae = 0.95
        
        # Berechne diskontierte Returns und Advantages mit GAE
        gae = 0
        next_value = 0  # Terminal state hat Wert 0
        
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                delta = r - 0  # Terminal state hat Wert 0
                gae = delta
            else:
                delta = r + self.gamma * next_value - 0  # Vereinfachte Version ohne Value-Funktion
                gae = delta + self.gamma * lambda_gae * gae
            
            next_value = 0 if done else r  # Vereinfachte Approximation
            returns.insert(0, gae)
            advantages.insert(0, gae)
            
        # Add exploration bonus to advantages based on action rarity
        # This encourages exploring less-frequently chosen actions
        if len(self.experiences) > 100:
            # Count action frequencies
            action_counts = {i: 0 for i in range(self.num_jobs)}
            recent_experiences = self.experiences[-100:]
            for exp in recent_experiences:
                action_counts[exp['action']] = action_counts.get(exp['action'], 0) + 1
                
            # Calculate rarity-based bonus
            total_actions = sum(action_counts.values())
            for i in range(len(advantages)):
                action = self.experiences[i]['action']
                action_freq = action_counts.get(action, 0) / max(1, total_actions)
                rarity_bonus = 0.2 * (1.0 - action_freq)  # Higher bonus for rare actions
                advantages[i] += rarity_bonus
        
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalisiere Returns und Advantages für stabileres Training
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mehrere Epochen für besseres Lernen mit Minibatches
        epochs = 10  # Mehr Epochen für besseres Lernen
        # batch_size wird als Parameter übergeben
        total_loss = 0.0
        
        for _ in range(epochs):
            # Shuffling der Erfahrungen für bessere Generalisierung
            indices = torch.randperm(len(self.experiences))
            
            # Minibatch-Training
            epoch_loss = 0.0
            for start_idx in range(0, len(self.experiences), batch_size):
                # Minibatch erstellen
                batch_indices = indices[start_idx:min(start_idx + batch_size, len(indices))]
                
                # Batch-Daten sammeln
                batch_states = []
                batch_actions = []
                batch_old_probs = []
                batch_advantages = []
                
                for idx in batch_indices:
                    exp = self.experiences[idx.item()]
                    batch_states.append(self.state_to_tensor(exp['state']))
                    batch_actions.append(exp['action'])
                    batch_old_probs.append(exp['action_prob'])
                    batch_advantages.append(advantages[idx.item()])
                
                # Konvertiere Listen zu Tensoren
                batch_actions = torch.tensor(batch_actions, dtype=torch.long)
                batch_old_probs = torch.tensor(batch_old_probs, dtype=torch.float32)
                batch_advantages = torch.tensor(batch_advantages, dtype=torch.float32)
                
                # Forward-Pass durch das Graph-Modell für den gesamten Batch
                batch_logits = []
                for state_embedding in batch_states:
                    hidden = torch.nn.functional.relu(self.graph_layer1(state_embedding))
                    hidden = torch.nn.functional.relu(self.graph_layer2(hidden))
                    logits = self.output_layer(hidden)
                    batch_logits.append(logits)
                
                # Berechne neue Aktionswahrscheinlichkeiten
                batch_new_probs = []
                batch_entropies = []
                for i, logits in enumerate(batch_logits):
                    probs = torch.nn.functional.softmax(logits, dim=0)
                    batch_new_probs.append(probs[batch_actions[i]])
                    # Entropy für jede Aktion berechnen
                    batch_entropies.append(-torch.sum(probs * torch.log(probs + 1e-10)))
                
                batch_new_probs = torch.stack(batch_new_probs)
                batch_entropies = torch.stack(batch_entropies)
                
                # PPO-Ratio berechnen
                ratio = batch_new_probs / (batch_old_probs + 1e-10)
                
                # Clipped loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy-Bonus für mehr Exploration (stärker gewichtet)
                entropy_loss = batch_entropies.mean()
                
                # Kombinierte Loss mit viel stärkerem Entropy-Bonus
                loss = actor_loss - 0.1 * entropy_loss  # Significantly increased entropy bonus
                
                # Gradient-Schritt mit Clipping
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_indices)
            
            total_loss += epoch_loss / len(self.experiences)
        
        # Durchschnittlicher Loss über alle Epochen
        total_loss /= epochs
        
        # Erfahrungen zurücksetzen
        self.experiences = []
        
        return total_loss / len(rewards)
    
    def get_makespan_reward(self, state, action, next_state):
        """
        Berechnet eine Belohnung basierend auf der Verbesserung des Makespans, Prioritäten, Deadlines und anderen Faktoren.
        Unterstützt sowohl SimpleEnvironment als auch JSPGymEnvironment.
        """
        # Aktuelle Schätzung des Makespans
        current_makespan = max(state['machine_times'])
        
        # Schätzung des Makespans nach der Aktion
        next_makespan = max(next_state['machine_times'])
        
        # Belohnung basierend auf der Differenz
        makespan_diff = next_makespan - current_makespan
        
        # Extrahiere current_time aus dem Zustand (unterschiedliches Format je nach Umgebung)
        if 'current_time' in next_state:
            if isinstance(next_state['current_time'], (list, np.ndarray)):
                current_time = next_state['current_time'][0]  # Gym-Umgebung
            else:
                current_time = next_state['current_time']  # Simple-Umgebung
        else:
            current_time = next_makespan  # Fallback
        
        # Maschinen-Auslastung berechnen mit verbesserter Metrik
        machine_times = next_state['machine_times']
        if next_makespan > 0:
            # Berechne Standardabweichung der Maschinenzeiten für gleichmäßigere Auslastung
            mean_time = sum(machine_times) / len(machine_times)
            variance = sum((t - mean_time) ** 2 for t in machine_times) / len(machine_times)
            std_dev = variance ** 0.5
            
            # Zwei Metriken für Maschinenauslastung:
            # 1. Gesamtauslastung (höher ist besser)
            total_util = sum(machine_times) / (current_time * len(machine_times)) if current_time > 0 else 0
            
            # 2. Gleichmäßigkeit der Auslastung (niedrigere Standardabweichung ist besser)
            balance_util = 1.0 / (1.0 + std_dev / mean_time) if mean_time > 0 else 0
            
            # Kombinierte Metrik (gewichtet)
            machine_util = (total_util * 0.7) + (balance_util * 0.3)
        else:
            machine_util = 0
        
        # Job-Fortschritt berechnen mit Gewichtung nach Priorität
        job_progress = next_state['job_progress']
        total_operations = 0
        weighted_progress = 0
        completed_jobs = 0
        
        for job_idx, progress in enumerate(job_progress):
            job = self.jsp_data["jobs"][job_idx]
            job_priority = job["priority"]
            job_ops_count = len(job["operations"])
            total_operations += job_ops_count
            
            # Gewichte den Fortschritt mit der Priorität
            weighted_progress += progress * job_priority
            
            # Zähle abgeschlossene Jobs
            if progress >= job_ops_count:
                completed_jobs += 1
        
        # Normalisiere den gewichteten Fortschritt
        total_priority = sum(job["priority"] for job in self.jsp_data["jobs"])
        progress_ratio = weighted_progress / (total_operations * total_priority / self.num_jobs) if total_operations > 0 else 0
        
        # Überprüfe, ob ein Job durch die Aktion abgeschlossen wurde
        job_completed = False
        job_priority = 0
        job_deadline = 0
        deadline_exceeded = False
        remaining_time = 0
        
        if action < self.num_jobs:
            job_idx = action
            job = self.jsp_data["jobs"][job_idx]
            
            # Prüfe, ob der Job durch diese Aktion abgeschlossen wurde
            if state['job_progress'][job_idx] < len(job["operations"]) and \
               next_state['job_progress'][job_idx] >= len(job["operations"]):
                job_completed = True
                job_priority = job["priority"]
                job_deadline = job["deadline"]
                deadline_exceeded = current_time > job_deadline
                
                if deadline_exceeded:
                    # Berechne, wie stark die Deadline überschritten wurde
                    remaining_time = current_time - job_deadline
        
        # Überprüfe Deadlines für alle Jobs
        met_deadlines = 0
        for job_idx, progress in enumerate(job_progress):
            job = self.jsp_data["jobs"][job_idx]
            if progress >= len(job["operations"]) and current_time <= job["deadline"]:
                met_deadlines += 1
        
        # Berechne Deadline-Einhaltungsrate
        deadline_ratio = met_deadlines / max(1, completed_jobs) if completed_jobs > 0 else 0
        
        # Belohnungskomponenten mit angepasster Gewichtung
        makespan_reward = -makespan_diff * 3.0 if makespan_diff > 0 else 10.0  # Stärkerer Fokus auf Makespan-Optimierung
        utilization_reward = machine_util * 8.0  # Höhere Belohnung für gute Maschinenauslastung
        progress_reward = progress_ratio * 5.0   # Belohnung für Fortschritt
        deadline_overall_reward = deadline_ratio * 7.0  # Belohnung für Deadline-Einhaltung insgesamt
        
        # Prioritätsbasierte Belohnung für den aktuell ausgeführten Job
        priority_reward = 0.0
        deadline_job_reward = 0.0
        
        if job_completed:
            # Höhere Belohnung für Jobs mit höherer Priorität
            priority_reward = job_priority * 3.0
            
            # Belohnung/Strafe für Deadline-Einhaltung mit progressiver Bestrafung
            if deadline_exceeded:
                # Progressive Bestrafung: Je weiter über der Deadline, desto höher die Strafe
                deadline_job_reward = -15.0 - (remaining_time / 5.0)
            else:
                # Bonus für frühe Fertigstellung
                time_before_deadline = job_deadline - current_time
                deadline_job_reward = 20.0 + (time_before_deadline / 3.0)
        
        # Umrüstzeiten-Optimierung
        setup_reward = 0.0
        if 'setup_time' in next_state:
            setup_time = next_state['setup_time']
            # Bestrafung für hohe Umrüstzeiten
            setup_reward = -setup_time / 3.0
        
        # Belohnung für kritischen Pfad und Engpässe
        critical_path_reward = 0.0
        if 'critical_path' in next_state:
            critical_path = next_state['critical_path']
            # Belohnung für Reduzierung des kritischen Pfades
            if 'critical_path' in state:
                critical_path_diff = state['critical_path'] - critical_path
                critical_path_reward = critical_path_diff * 2.0
        
        # Kombinierte Belohnung
        reward = (
            makespan_reward +           # Makespan-Optimierung
            utilization_reward +        # Maschinenauslastung
            progress_reward +           # Jobfortschritt
            priority_reward +           # Priorität des abgeschlossenen Jobs
            deadline_job_reward +       # Deadline-Einhaltung des aktuellen Jobs
            deadline_overall_reward +   # Gesamte Deadline-Einhaltung
            setup_reward +              # Umrüstzeiten-Optimierung
            critical_path_reward        # Kritischer Pfad
        )
        
        # Zusätzliche Belohnung für abgeschlossene Jobs
        if 'job_completed' in next_state and next_state['job_completed']:
            reward += 15.0
        
        # Begrenze die Belohnung auf einen sinnvollen Bereich
        reward = max(min(reward, 75.0), -75.0)
        
        return reward
    
    def save_model(self, path):
        """Speichert das Modell"""
        # Speichere alle Netzwerk-Parameter
        model_state = {
            'node_embedding': self.node_embedding.state_dict(),
            'graph_layer1': self.graph_layer1.state_dict(),
            'graph_layer2': self.graph_layer2.state_dict(),
            'output_layer': self.output_layer.state_dict()
        }
        torch.save(model_state, path)
    
    def load_model(self, path):
        """Lädt ein gespeichertes Modell"""
        model_state = torch.load(path)
        self.node_embedding.load_state_dict(model_state['node_embedding'])
        self.graph_layer1.load_state_dict(model_state['graph_layer1'])
        self.graph_layer2.load_state_dict(model_state['graph_layer2'])
        self.output_layer.load_state_dict(model_state['output_layer'])
        
    def parameters(self):
        """Gibt alle Parameter des Modells zurück für Gradient Clipping"""
        return list(self.node_embedding.parameters()) + \
               list(self.graph_layer1.parameters()) + \
               list(self.graph_layer2.parameters()) + \
               list(self.output_layer.parameters())
