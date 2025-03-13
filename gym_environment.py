import gym
from gym import spaces
import numpy as np
import logging
from datetime import datetime


class JSPGymEnvironment(gym.Env):
    """
    A Gym-compatible environment for Job-Shop Scheduling Problems.
    
    This environment follows the OpenAI Gym interface, making it compatible with
    standard RL algorithms and frameworks.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, jsp_data, enable_logging=False, log_level=logging.INFO):
        """
        Initialize the JSP Gym environment.
        
        Args:
            jsp_data: Dictionary containing JSP problem data
            enable_logging: Whether to enable detailed logging
            log_level: Logging level (default: INFO)
        """
        super(JSPGymEnvironment, self).__init__()
        
        self.jobs = jsp_data["jobs"]
        self.machines = jsp_data["machines"]
        self.setupTimes = jsp_data["setupTimes"]
        self.num_jobs = len(self.jobs)
        self.num_machines = len(self.machines)
        
        # Setup logging
        self.enable_logging = enable_logging
        if enable_logging:
            self._setup_logging(log_level)
        
        # Create mappings between IDs and indices
        self.job_id_to_idx = {job["id"]: idx for idx, job in enumerate(self.jobs)}
        self.idx_to_job_id = {idx: job["id"] for idx, job in enumerate(self.jobs)}
        self.machine_id_to_idx = {machine["id"]: idx for idx, machine in enumerate(self.machines)}
        self.idx_to_machine_id = {idx: machine["id"] for idx, machine in enumerate(self.machines)}
        
        # Define action and observation spaces
        # Action space: Choose one of the jobs
        self.action_space = spaces.Discrete(self.num_jobs)
        
        # Observation space: Dictionary with job progress, machine times, etc.
        self.observation_space = spaces.Dict({
            'job_progress': spaces.Box(
                low=0, 
                high=max([len(job["operations"]) for job in self.jobs]),
                shape=(self.num_jobs,), 
                dtype=np.int32
            ),
            'machine_times': spaces.Box(
                low=0, 
                high=float('inf'),  # No upper bound on machine times
                shape=(self.num_machines,), 
                dtype=np.float32
            ),
            'current_time': spaces.Box(
                low=0, 
                high=float('inf'),
                shape=(1,), 
                dtype=np.float32
            ),
            'job_priorities': spaces.Box(
                low=0, 
                high=10,  # Assuming priorities are between 0-10
                shape=(self.num_jobs,), 
                dtype=np.float32
            ),
            'job_deadlines': spaces.Box(
                low=0, 
                high=float('inf'),
                shape=(self.num_jobs,), 
                dtype=np.float32
            ),
            'machine_materials': spaces.Box(
                low=0, 
                high=len(set([op["material"] for job in self.jobs for op in job["operations"]])),
                shape=(self.num_machines,), 
                dtype=np.int32
            ),
            'valid_actions_mask': spaces.Box(
                low=0, 
                high=1,
                shape=(self.num_jobs,), 
                dtype=np.int32
            )
        })
        
        # Initialize state
        self.reset()
        
        # Create a mapping for materials to indices for observation space
        self.materials = list(set([op["material"] for job in self.jobs for op in job["operations"]]))
        self.material_to_idx = {material: idx for idx, material in enumerate(self.materials)}
        self.idx_to_material = {idx: material for idx, material in enumerate(self.materials)}
        
        # Initialize operation tracking for detailed logging
        self.operation_history = []
        self.machine_utilization = [[] for _ in range(self.num_machines)]
        self.material_changes = []
        self.job_completion_times = {}
    
    def _setup_logging(self, log_level):
        """
        Setup logging for the environment.
        
        Args:
            log_level: Logging level
        """
        # Create logger
        self.logger = logging.getLogger('JSPGymEnvironment')
        self.logger.setLevel(log_level)
        
        # Create file handler
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fh = logging.FileHandler(f'jsp_env_{timestamp}.log')
        fh.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(fh)
    
    def reset(self):
        """
        Reset the environment to its initial state.
        
        Returns:
            observation: The initial observation
        """
        # Reset job progress
        self.job_progress = np.zeros(self.num_jobs, dtype=np.int32)
        
        # Reset machine times
        self.machine_times = np.zeros(self.num_machines, dtype=np.float32)
        
        # Reset current time
        self.current_time = 0.0
        
        # Reset completed jobs
        self.completed_jobs = 0
        
        # Reset current material on each machine
        self.current_machine_material = [""] * self.num_machines
        self.machine_material_idx = np.zeros(self.num_machines, dtype=np.int32)
        
        # Track episode statistics
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_makespan = 0.0
        self.episode_completed_jobs = 0
        self.episode_met_deadlines = 0
        
        # Reset operation tracking for detailed logging
        self.operation_history = []
        self.machine_utilization = [[] for _ in range(self.num_machines)]
        self.material_changes = []
        self.job_completion_times = {}
        
        if self.enable_logging:
            self.logger.info("Environment reset")
        
        # Get initial observation
        return self._get_observation()
    
    def _get_observation(self):
        """
        Get the current observation of the environment.
        
        Returns:
            observation: Dictionary containing the current state
        """
        # Calculate job priorities and deadlines for observation
        job_priorities = np.array([job["priority"] for job in self.jobs], dtype=np.float32)
        job_deadlines = np.array([job["deadline"] for job in self.jobs], dtype=np.float32)
        
        # Calculate valid actions mask
        valid_actions_mask = np.zeros(self.num_jobs, dtype=np.int32)
        for job_idx in range(self.num_jobs):
            # Check if job is not completed and predecessors are completed
            if (self.job_progress[job_idx] < len(self.jobs[job_idx]["operations"]) and
                self._check_predecessors(job_idx, self.job_progress[job_idx])):
                valid_actions_mask[job_idx] = 1
        
        return {
            'job_progress': self.job_progress.copy(),
            'machine_times': self.machine_times.copy(),
            'current_time': np.array([self.current_time], dtype=np.float32),
            'job_priorities': job_priorities,
            'job_deadlines': job_deadlines,
            'machine_materials': self.machine_material_idx.copy(),
            'valid_actions_mask': valid_actions_mask
        }
    
    def _check_predecessors(self, job_idx, op_idx):
        """
        Check if all predecessor operations are completed.
        
        Args:
            job_idx: Index of the job
            op_idx: Index of the operation
            
        Returns:
            bool: True if all predecessors are completed, False otherwise
        """
        # If operation index is out of bounds, return False
        if op_idx >= len(self.jobs[job_idx]["operations"]):
            return False
            
        operation = self.jobs[job_idx]["operations"][op_idx]
        
        # If no predecessors defined, operation can be executed
        if not operation.get("predecessors", []):
            return True
        
        # Check all predecessors
        for pred in operation["predecessors"]:
            # Format of predecessors is "J1:OP1"
            pred_job_id, pred_op_id = pred.split(":")
            pred_job_idx = self.job_id_to_idx[pred_job_id]
            
            # Find the index of the predecessor operation
            pred_op_idx = None
            for i, op in enumerate(self.jobs[pred_job_idx]["operations"]):
                if op["id"] == pred_op_id:
                    pred_op_idx = i
                    break
            
            if pred_op_idx is None:
                return False  # Predecessor operation not found
            
            # Check if the predecessor operation is completed
            if self.job_progress[pred_job_idx] <= pred_op_idx:
                return False  # Predecessor operation not completed yet
        
        return True  # All predecessors are completed
    
    def _calculate_setup_time(self, machine_id, new_material):
        """
        Calculate setup time based on current and new material.
        
        Args:
            machine_id: ID of the machine
            new_material: New material to be processed
            
        Returns:
            float: Setup time
        """
        machine_idx = self.machine_id_to_idx[machine_id]
        current_material = self.current_machine_material[machine_idx]
        
        # If machine hasn't processed any material yet, no setup time
        if current_material == "":
            return 0
        
        # If material remains the same, standard setup time
        if current_material == new_material:
            return self.setupTimes[machine_id]["standard"]
        else:
            # For material change, higher setup time
            return self.setupTimes[machine_id]["materialChange"]
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Index of the job to process next
            
        Returns:
            observation: New observation after taking the action
            reward: Reward received
            done: Whether the episode is finished
            info: Additional information
        """
        # Increment step counter
        self.episode_steps += 1
        
        # Check if action is valid
        if action >= self.num_jobs:
            return self._get_observation(), -10.0, False, {"error": "Invalid job index"}
        
        job_idx = action
        
        # Check if job is already completed
        if self.job_progress[job_idx] >= len(self.jobs[job_idx]["operations"]):
            return self._get_observation(), -10.0, False, {"error": "Job already completed"}
        
        # Get next operation of the job
        op_idx = self.job_progress[job_idx]
        op = self.jobs[job_idx]["operations"][op_idx]
        machine_id = op["machineId"]
        machine_idx = self.machine_id_to_idx[machine_id]
        proc_time = op["processingTime"]
        material = op["material"]
        
        # Check if all predecessors are completed
        if not self._check_predecessors(job_idx, op_idx):
            return self._get_observation(), -10.0, False, {"error": "Predecessors not completed"}
        
        # Calculate setup time
        setup_time = self._calculate_setup_time(machine_id, material)
        
        # Earliest possible start time (max of machine availability and current time)
        start_time = max(self.machine_times[machine_idx], self.current_time)
        
        # Add setup time
        start_time += setup_time
        
        # Execute operation
        end_time = start_time + proc_time
        self.machine_times[machine_idx] = end_time
        
        # Update current material of the machine
        old_material = self.current_machine_material[machine_idx]
        self.current_machine_material[machine_idx] = material
        self.machine_material_idx[machine_idx] = self.material_to_idx.get(material, 0)
        
        # Log material change if it occurred
        if old_material != material:
            material_change = {
                'step': self.episode_steps,
                'machine_id': machine_id,
                'machine_idx': machine_idx,
                'old_material': old_material,
                'new_material': material,
                'setup_time': setup_time,
                'time': self.current_time
            }
            self.material_changes.append(material_change)
            
            if self.enable_logging:
                self.logger.info(f"Material change on machine {machine_id}: {old_material} -> {material}, setup time: {setup_time}")
        
        # Update job progress
        self.job_progress[job_idx] += 1
        
        # Record operation execution
        operation_record = {
            'step': self.episode_steps,
            'job_id': self.idx_to_job_id[job_idx],
            'job_idx': job_idx,
            'operation_idx': op_idx,
            'machine_id': machine_id,
            'machine_idx': machine_idx,
            'start_time': start_time,
            'end_time': end_time,
            'processing_time': proc_time,
            'setup_time': setup_time,
            'material': material
        }
        self.operation_history.append(operation_record)
        
        # Update machine utilization record
        machine_util = {
            'step': self.episode_steps,
            'machine_id': machine_id,
            'start_time': start_time,
            'end_time': end_time,
            'busy_time': proc_time,
            'setup_time': setup_time,
            'idle_time': start_time - max(self.current_time, self.machine_times[machine_idx]) if start_time > max(self.current_time, self.machine_times[machine_idx]) else 0
        }
        self.machine_utilization[machine_idx].append(machine_util)
        
        if self.enable_logging:
            self.logger.info(f"Executed operation: Job {self.idx_to_job_id[job_idx]}, Op {op_idx}, Machine {machine_id}, "
                           f"Start: {start_time}, End: {end_time}, Setup: {setup_time}")
        
        # Update current time to the maximum machine time
        prev_time = self.current_time
        self.current_time = max(self.machine_times)
        
        # Check if job is completed
        job_completed = False
        if self.job_progress[job_idx] >= len(self.jobs[job_idx]["operations"]):
            job_completed = True
            self.completed_jobs += 1
            
            # Record job completion time
            job_id = self.idx_to_job_id[job_idx]
            deadline = self.jobs[job_idx]["deadline"]
            deadline_met = self.current_time <= deadline
            self.job_completion_times[job_id] = {
                'completion_time': self.current_time,
                'deadline': deadline,
                'deadline_met': deadline_met,
                'priority': self.jobs[job_idx]["priority"]
            }
            
            # Check if deadline is met
            if deadline_met:
                self.episode_met_deadlines += 1
                
            if self.enable_logging:
                self.logger.info(f"Job {job_id} completed at time {self.current_time}, deadline: {deadline}, "
                               f"met: {deadline_met}, priority: {self.jobs[job_idx]['priority']}")
        
        # Check if all jobs are completed
        done = self.completed_jobs >= self.num_jobs
        
        # Calculate reward
        reward = self._calculate_reward(job_idx, job_completed, setup_time, 
                                       prev_time, self.current_time)
        
        # Update episode statistics
        self.episode_reward += reward
        self.episode_makespan = max(self.machine_times)
        
        # Get new observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            "makespan": self.episode_makespan,
            "completed_jobs": self.completed_jobs,
            "met_deadlines": self.episode_met_deadlines,
            "job_completed": job_completed,
            "setup_time": setup_time
        }
        
        return observation, reward, done, info
    
    def _calculate_reward(self, job_idx, job_completed, setup_time, prev_time, current_time):
        """
        Calculate reward based on various factors.
        
        Args:
            job_idx: Index of the job
            job_completed: Whether the job was completed
            setup_time: Setup time for the operation
            prev_time: Previous current time
            current_time: New current time
            
        Returns:
            float: Reward
        """
        # Initialize reward components
        total_reward = 0.0
        
        # Get operation details
        op_idx = self.job_progress[job_idx] - 1  # The operation that was just completed
        if op_idx >= 0:  # Make sure we have a valid operation
            operation = self.jobs[job_idx]["operations"][op_idx]
            machine_id = operation["machineId"]
            machine_idx = self.machine_id_to_idx[machine_id]
            proc_time = operation["processingTime"]
            
            # Berechne aktuelle Zielfunktion: makespan + makespan * (1 - timelines)
            current_makespan = max(self.machine_times)
            timelines_ratio = self.episode_met_deadlines / max(1, self.completed_jobs) if self.completed_jobs > 0 else 0
            current_objective = current_makespan + current_makespan * (1 - timelines_ratio)
            
            # Priorität des aktuellen Jobs (immer berücksichtigen)
            priority = self.jobs[job_idx]["priority"]
            priority_factor = priority / 10.0  # Normalisiert auf 0-1
            
            # 1. Makespan-Optimierung mit Berücksichtigung der Zielfunktion
            # Berechne durchschnittliche Prozesszeit für diese Maschine
            machine_ops = [op for job in self.jobs for op in job["operations"] if op["machineId"] == machine_id]
            avg_proc_time = sum(op["processingTime"] for op in machine_ops) / len(machine_ops) if machine_ops else proc_time
            
            # Belohne effiziente Operationen
            time_efficiency = (avg_proc_time - proc_time) / avg_proc_time if avg_proc_time > 0 else 0
            
            # Kritischer Pfad: Bewertung basierend auf Zielfunktion
            if self.machine_times[machine_idx] < current_makespan:
                # Operation ist nicht auf dem kritischen Pfad
                makespan_reward = 3.0 + time_efficiency
            else:
                # Operation ist auf dem kritischen Pfad
                # Höhere Bestrafung für Jobs mit niedriger Priorität auf dem kritischen Pfad
                makespan_reward = -2.0 + time_efficiency + priority_factor * 2.0
            
            # 2. Umrüstzeit-Optimierung - mit Prioritätsberücksichtigung
            if setup_time == 0:
                setup_reward = 1.5 * (1 + priority_factor)  # Bonus für keine Umrüstung, verstärkt durch Priorität
            elif setup_time <= self.setupTimes[machine_id]["standard"]:
                setup_reward = 0.7 * (1 + priority_factor)  # Moderater Bonus für Standard-Umrüstung
            else:
                # Bestrafung für Materialwechsel, gemildert durch hohe Priorität
                setup_reward = -1.5 + priority_factor
            
            # 3. Maschinenauslastung - mit Prioritätsberücksichtigung
            machine_idle_time = 0.0
            if prev_time > self.machine_times[machine_idx]:
                machine_idle_time = prev_time - self.machine_times[machine_idx]
                
            # Bestrafung für Leerlaufzeiten, gemildert durch hohe Priorität
            if machine_idle_time > 0:
                idle_penalty = -1.5 * min(1.0, machine_idle_time / (avg_proc_time * 2.0)) * (1 - priority_factor * 0.5)
            else:
                idle_penalty = 0.8 * (1 + priority_factor * 0.5)  # Bonus für keine Leerlaufzeit
            
            # 4. Maschinenbalance - mit Prioritätsberücksichtigung
            mean_machine_time = sum(self.machine_times) / self.num_machines
            machine_time_variance = sum((t - mean_machine_time) ** 2 for t in self.machine_times) / self.num_machines
            machine_time_std = machine_time_variance ** 0.5
            
            if current_time > 0:
                relative_imbalance = machine_time_std / current_time
                # Jobs mit hoher Priorität dürfen mehr Ungleichgewicht verursachen
                balance_reward = 0.6 * (1.0 - min(1.0, relative_imbalance * (1 - priority_factor * 0.5)))
            else:
                balance_reward = 0.0
            
            # 5. Deadline-Einhaltung - DRASTISCH erhöhte Gewichtung mit Prioritätsberücksichtigung
            deadline_reward = 0.0
            job_deadline = self.jobs[job_idx]["deadline"]
            
            # Schätze die verbleibende Zeit für diesen Job
            remaining_ops = len(self.jobs[job_idx]["operations"]) - self.job_progress[job_idx]
            estimated_finish_time = current_time
            
            if remaining_ops > 0:
                avg_op_time = sum(op["processingTime"] for op in self.jobs[job_idx]["operations"]) / len(self.jobs[job_idx]["operations"])
                estimated_finish_time += remaining_ops * avg_op_time
            
            # Drastisch erhöhte Belohnung für Deadline-Einhaltung
            if job_completed:
                completion_time = self.machine_times[machine_idx]
                
                if completion_time <= job_deadline:
                    # Massive Belohnung für eingehaltene Deadlines, verstärkt durch Priorität
                    deadline_reward = 12.0 * (1 + priority_factor * 0.5)
                else:
                    # Starke Bestrafung für verpasste Deadlines, gemildert durch niedrige Priorität
                    overdue_ratio = (completion_time - job_deadline) / job_deadline
                    deadline_reward = -7.0 * min(1.0, overdue_ratio) * (1 - priority_factor * 0.3)
            else:
                # Belohne auch Fortschritt bei Jobs, die voraussichtlich die Deadline einhalten
                time_margin = job_deadline - estimated_finish_time
                if time_margin > 0:
                    deadline_reward = 3.0 * (self.job_progress[job_idx] / len(self.jobs[job_idx]["operations"])) * (1 + priority_factor * 0.5)
                else:
                    # Bestrafung für Jobs, die voraussichtlich die Deadline nicht einhalten
                    deadline_reward = -1.5 * (1.0 - self.job_progress[job_idx] / len(self.jobs[job_idx]["operations"])) * (1 - priority_factor * 0.3)
            
            # 6. Prioritätsbasierte Belohnung - STARK erhöht
            # Direkte Belohnung basierend auf Priorität
            priority_reward = 2.5 * priority_factor
            
            # 7. Fortschrittsbelohnung - mit Prioritätsberücksichtigung
            progress_ratio = self.job_progress[job_idx] / len(self.jobs[job_idx]["operations"])
            progress_reward = 0.5 * progress_ratio * (1 + priority_factor * 0.5)
            
            # 8. Jobabschluss-Bonus - mit Prioritätsberücksichtigung
            completion_reward = 0.0
            if job_completed:
                completion_reward = 4.0 * (1 + priority_factor)  # Starker Bonus für Jobabschluss
            
            # 9. Kritische Jobs bevorzugen - mit Prioritätsberücksichtigung
            critical_job_reward = 0.0
            if not job_completed:
                # Berechne Dringlichkeit basierend auf verbleibender Zeit und verbleibenden Operationen
                if remaining_ops > 0:
                    urgency = (job_deadline - current_time) / (remaining_ops * avg_op_time)
                    if urgency < 1.0:  # Sehr kritischer Job
                        critical_job_reward = 4.0 * (1 + priority_factor * 0.5)
                    elif urgency < 1.5:  # Kritischer Job
                        critical_job_reward = 2.0 * (1 + priority_factor * 0.3)
            
            # 10. Globaler Fortschritt - mit Prioritätsberücksichtigung
            global_progress = sum(self.job_progress) / sum(len(job["operations"]) for job in self.jobs)
            global_progress_reward = 0.5 * global_progress * (1 + priority_factor * 0.2)
            
            # 11. Zielfunktionsverbesserung - NEU
            # Belohne Aktionen, die die Zielfunktion verbessern
            if hasattr(self, 'previous_objective'):
                objective_improvement = self.previous_objective - current_objective
                if objective_improvement > 0:
                    objective_reward = 3.0 * min(1.0, objective_improvement / current_objective) * (1 + priority_factor * 0.5)
                else:
                    objective_reward = -1.0 * min(1.0, -objective_improvement / current_objective) * (1 - priority_factor * 0.3)
            else:
                objective_reward = 0.0
            
            # Speichere aktuelle Zielfunktion für nächsten Vergleich
            self.previous_objective = current_objective
            
            # Kombiniere alle Belohnungskomponenten mit angepassten Gewichtungen
            total_reward = (
                makespan_reward * 3.5 +           # Erhöht
                setup_reward * 1.2 +              # Erhöht
                idle_penalty * 1.2 +              # Erhöht
                balance_reward * 0.7 +            # Leicht erhöht
                deadline_reward * 6.0 +           # Drastisch erhöht
                priority_reward * 3.0 +           # Stark erhöht
                progress_reward * 0.5 +           # Leicht erhöht
                completion_reward * 2.5 +         # Erhöht
                critical_job_reward * 2.5 +       # Erhöht
                global_progress_reward * 0.5 +    # Leicht erhöht
                objective_reward * 4.0            # NEU: Zielfunktionsverbesserung
            )
        
        # Erlaube größere Belohnungen und Bestrafungen
        total_reward = max(min(total_reward, 20.0), -15.0)
        
        return total_reward
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
            
        Returns:
            None
        """
        if mode == 'human': 
            print(f"Current Time: {self.current_time}")
            print(f"Job Progress: {self.job_progress}")
            print(f"Machine Times: {self.machine_times}")
            print(f"Completed Jobs: {self.completed_jobs}/{self.num_jobs}")
            print(f"Met Deadlines: {self.episode_met_deadlines}/{self.num_jobs}")
            print(f"Current Makespan: {max(self.machine_times)}")
            
            # Print machine utilization
            print("\nMachine Utilization:")
            for machine_idx in range(self.num_machines):
                machine_id = self.idx_to_machine_id[machine_idx]
                if self.current_time > 0:
                    utilization = self.machine_times[machine_idx] / self.current_time
                    print(f"  Machine {machine_id}: {utilization:.2f}")
                else:
                    print(f"  Machine {machine_id}: 0.00")
            
            # Print material on each machine
            print("\nCurrent Materials:")
            for machine_idx in range(self.num_machines):
                machine_id = self.idx_to_machine_id[machine_idx]
                material = self.current_machine_material[machine_idx]
                print(f"  Machine {machine_id}: {material}")
            
            print("---")
    
    def close(self):
        """
        Clean up resources.
        """
        if self.enable_logging:
            # Log final statistics
            self.logger.info(f"Environment closed. Final statistics:")
            self.logger.info(f"  Makespan: {max(self.machine_times)}")
            self.logger.info(f"  Completed Jobs: {self.completed_jobs}/{self.num_jobs}")
            self.logger.info(f"  Met Deadlines: {self.episode_met_deadlines}/{self.num_jobs}")
            
            # Close log handlers
            for handler in self.logger.handlers:
                handler.close()
                self.logger.removeHandler(handler)
    
    def get_machine_utilization_stats(self):
        """
        Get detailed machine utilization statistics.
        
        Returns:
            dict: Dictionary containing machine utilization statistics
        """
        stats = {}
        
        for machine_idx in range(self.num_machines):
            machine_id = self.idx_to_machine_id[machine_idx]
            machine_records = self.machine_utilization[machine_idx]
            
            if not machine_records:
                stats[machine_id] = {
                    'utilization': 0.0,
                    'setup_time_ratio': 0.0,
                    'idle_time_ratio': 0.0,
                    'processing_time_ratio': 0.0
                }
                continue
            
            # Calculate total times
            total_busy_time = sum(record['busy_time'] for record in machine_records)
            total_setup_time = sum(record['setup_time'] for record in machine_records)
            total_idle_time = sum(record['idle_time'] for record in machine_records)
            
            # Calculate ratios
            if self.current_time > 0:
                utilization = self.machine_times[machine_idx] / self.current_time
                setup_time_ratio = total_setup_time / self.current_time
                idle_time_ratio = total_idle_time / self.current_time
                processing_time_ratio = total_busy_time / self.current_time
            else:
                utilization = 0.0
                setup_time_ratio = 0.0
                idle_time_ratio = 0.0
                processing_time_ratio = 0.0
            
            stats[machine_id] = {
                'utilization': utilization,
                'setup_time_ratio': setup_time_ratio,
                'idle_time_ratio': idle_time_ratio,
                'processing_time_ratio': processing_time_ratio,
                'total_busy_time': total_busy_time,
                'total_setup_time': total_setup_time,
                'total_idle_time': total_idle_time
            }
        
        return stats
    
    def get_material_change_stats(self):
        """
        Get statistics about material changes.
        
        Returns:
            dict: Dictionary containing material change statistics
        """
        stats = {}
        
        for machine_idx in range(self.num_machines):
            machine_id = self.idx_to_machine_id[machine_idx]
            
            # Filter material changes for this machine
            machine_changes = [change for change in self.material_changes 
                              if change['machine_idx'] == machine_idx]
            
            # Count changes by material type
            material_counts = {}
            for change in machine_changes:
                new_material = change['new_material']
                if new_material in material_counts:
                    material_counts[new_material] += 1
                else:
                    material_counts[new_material] = 1
            
            # Calculate total setup time
            total_setup_time = sum(change['setup_time'] for change in machine_changes)
            
            stats[machine_id] = {
                'total_changes': len(machine_changes),
                'total_setup_time': total_setup_time,
                'material_counts': material_counts
            }
        
        return stats
    
    def get_job_completion_stats(self):
        """
        Get statistics about job completions.
        
        Returns:
            dict: Dictionary containing job completion statistics
        """
        # Calculate average completion time
        if self.job_completion_times:
            avg_completion_time = sum(job['completion_time'] for job in self.job_completion_times.values()) / len(self.job_completion_times)
        else:
            avg_completion_time = 0.0
        
        # Calculate deadline statistics
        met_deadlines = sum(1 for job in self.job_completion_times.values() if job['deadline_met'])
        deadline_ratio = met_deadlines / len(self.job_completion_times) if self.job_completion_times else 0.0
        
        # Calculate priority-weighted statistics
        priority_weighted_completion = sum(job['completion_time'] * job['priority'] for job in self.job_completion_times.values()) if self.job_completion_times else 0.0
        total_priority = sum(job['priority'] for job in self.job_completion_times.values()) if self.job_completion_times else 0.0
        priority_weighted_avg = priority_weighted_completion / total_priority if total_priority > 0 else 0.0
        
        # High priority jobs (priority >= 7)
        high_priority_jobs = {job_id: job for job_id, job in self.job_completion_times.items() if job['priority'] >= 7}
        high_priority_met = sum(1 for job in high_priority_jobs.values() if job['deadline_met'])
        high_priority_ratio = high_priority_met / len(high_priority_jobs) if high_priority_jobs else 0.0
        
        return {
            'completed_jobs': len(self.job_completion_times),
            'avg_completion_time': avg_completion_time,
            'met_deadlines': met_deadlines,
            'deadline_ratio': deadline_ratio,
            'priority_weighted_avg_completion': priority_weighted_avg,
            'high_priority_met_ratio': high_priority_ratio
        }
