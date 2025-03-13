import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Any, Dict, List, Union


class JSPLogger:
    """
    Logger for Job Shop Scheduling environment to track detailed metrics
    during training and testing.
    """
    
    def __init__(self, log_dir="logs", experiment_name=None):
        """
        Initialize the JSP Logger.
        
        Args:
            log_dir: Directory to store logs
            experiment_name: Name of the experiment (default: timestamp)
        """
        self.log_dir = log_dir
        
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = f"jsp_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        # Create log directory
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize log data structures
        self.episode_data = []
        self.step_data = []
        self.machine_data = []
        self.material_change_data = []
        self.job_completion_data = []
        
        # Current episode tracking
        self.current_episode = 0
        self.current_step = 0
        
        print(f"JSP Logger initialized. Logs will be saved to: {self.experiment_dir}")
    
    def log_episode_start(self, episode_num, jsp_data=None):
        """
        Log the start of a new episode.
        
        Args:
            episode_num: Episode number
            jsp_data: JSP problem data (optional)
        """
        self.current_episode = episode_num
        self.current_step = 0
        
        # Save JSP data for the first episode
        if episode_num == 1 and jsp_data is not None:
            with open(os.path.join(self.experiment_dir, "jsp_data.json"), "w") as f:
                json.dump(jsp_data, f, indent=2)
    
    def log_step(self, state, action, reward, next_state, done, info):
        """
        Log data for a single step in the environment.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            info: Additional information
        """
        self.current_step += 1
        
        # Extract relevant information
        job_idx = action
        job_progress = next_state['job_progress'][job_idx]
        machine_times = next_state['machine_times']
        current_time = next_state['current_time'][0]
        makespan = max(machine_times)
        
        # Log step data
        step_info = {
            'episode': self.current_episode,
            'step': self.current_step,
            'job_idx': job_idx,
            'reward': reward,
            'makespan': makespan,
            'current_time': current_time,
            'setup_time': info.get('setup_time', 0),
            'job_completed': info.get('job_completed', False)
        }
        self.step_data.append(step_info)
        
        # Log machine utilization
        for machine_idx, machine_time in enumerate(machine_times):
            machine_info = {
                'episode': self.current_episode,
                'step': self.current_step,
                'machine_idx': machine_idx,
                'machine_time': machine_time,
                'current_time': current_time,
                'utilization': machine_time / current_time if current_time > 0 else 0
            }
            self.machine_data.append(machine_info)
        
        # Log material changes if available in info
        if 'material_change' in info:
            material_info = {
                'episode': self.current_episode,
                'step': self.current_step,
                'machine_idx': info.get('machine_idx', -1),
                'old_material': info.get('old_material', ''),
                'new_material': info.get('new_material', ''),
                'setup_time': info.get('setup_time', 0)
            }
            self.material_change_data.append(material_info)
        
        # Log job completion
        if info.get('job_completed', False):
            completion_info = {
                'episode': self.current_episode,
                'step': self.current_step,
                'job_idx': job_idx,
                'completion_time': current_time,
                'deadline_met': info.get('deadline_met', False)
            }
            self.job_completion_data.append(completion_info)
    
    def log_episode_end(self, final_state, episode_reward, episode_steps):
        """
        Log the end of an episode.
        
        Args:
            final_state: Final state of the environment
            episode_reward: Total reward for the episode
            episode_steps: Number of steps in the episode
        """
        # Calculate metrics
        makespan = max(final_state['machine_times'])
        completed_jobs = sum([1 for progress in final_state['job_progress'] 
                             if progress > 0])  # Count jobs with any progress
        fully_completed_jobs = sum([1 for i, progress in enumerate(final_state['job_progress']) 
                                  if i < len(final_state['job_deadlines']) and 
                                  progress >= 1])  # Count fully completed jobs
        
        # Calculate machine utilization
        machine_times = final_state['machine_times']
        current_time = final_state['current_time'][0]
        if current_time > 0:
            avg_machine_util = sum(machine_times) / (current_time * len(machine_times))
            machine_util_std = np.std([m_time / current_time for m_time in machine_times])
        else:
            avg_machine_util = 0
            machine_util_std = 0
        
        # Calculate deadline metrics
        met_deadlines = 0
        for i, progress in enumerate(final_state['job_progress']):
            if i < len(final_state['job_deadlines']) and progress >= 1:
                if current_time <= final_state['job_deadlines'][i]:
                    met_deadlines += 1
        
        deadline_ratio = met_deadlines / max(1, fully_completed_jobs)
        
        # Log episode data
        episode_info = {
            'episode': self.current_episode,
            'total_reward': episode_reward,
            'steps': episode_steps,
            'makespan': makespan,
            'completed_jobs': completed_jobs,
            'fully_completed_jobs': fully_completed_jobs,
            'met_deadlines': met_deadlines,
            'deadline_ratio': deadline_ratio,
            'avg_machine_util': avg_machine_util,
            'machine_util_std': machine_util_std
        }
        self.episode_data.append(episode_info)
    
    def save_logs(self):
        """
        Save all logs to CSV files.
        """
        # Convert data to DataFrames
        episode_df = pd.DataFrame(self.episode_data)
        step_df = pd.DataFrame(self.step_data)
        machine_df = pd.DataFrame(self.machine_data)
        material_change_df = pd.DataFrame(self.material_change_data)
        job_completion_df = pd.DataFrame(self.job_completion_data)
        
        # Save DataFrames to CSV
        episode_df.to_csv(os.path.join(self.experiment_dir, "episode_logs.csv"), index=False)
        step_df.to_csv(os.path.join(self.experiment_dir, "step_logs.csv"), index=False)
        machine_df.to_csv(os.path.join(self.experiment_dir, "machine_logs.csv"), index=False)
        
        if not material_change_df.empty:
            material_change_df.to_csv(os.path.join(self.experiment_dir, "material_change_logs.csv"), index=False)
        
        if not job_completion_df.empty:
            job_completion_df.to_csv(os.path.join(self.experiment_dir, "job_completion_logs.csv"), index=False)
        
        print(f"Logs saved to: {self.experiment_dir}")
    
    def generate_reports(self):
        """
        Generate summary reports and visualizations.
        """
        if not self.episode_data:
            print("No data to generate reports.")
            return
        
        # Create reports directory
        reports_dir = os.path.join(self.experiment_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Convert data to DataFrames
        episode_df = pd.DataFrame(self.episode_data)
        machine_df = pd.DataFrame(self.machine_data)
        
        # 1. Episode performance over time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(episode_df['episode'], episode_df['total_reward'])
        plt.title('Total Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(episode_df['episode'], episode_df['makespan'])
        plt.title('Makespan per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Makespan')
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(episode_df['episode'], episode_df['deadline_ratio'])
        plt.title('Deadline Met Ratio per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Ratio')
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(episode_df['episode'], episode_df['avg_machine_util'])
        plt.title('Average Machine Utilization per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Utilization')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "episode_performance.png"))
        
        # 2. Machine utilization analysis
        # Group by episode and machine_idx
        machine_util_by_episode = machine_df.groupby(['episode', 'machine_idx'])['utilization'].mean().reset_index()
        
        # Get unique episodes and machines
        episodes = machine_util_by_episode['episode'].unique()
        machines = machine_util_by_episode['machine_idx'].unique()
        
        # Create a pivot table for heatmap
        machine_util_pivot = machine_util_by_episode.pivot(index='episode', columns='machine_idx', values='utilization')
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        for machine in machines:
            machine_data = machine_util_by_episode[machine_util_by_episode['machine_idx'] == machine]
            plt.plot(machine_data['episode'], machine_data['utilization'], label=f'Machine {machine}')
        
        plt.title('Machine Utilization Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Utilization')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.imshow(machine_util_pivot, aspect='auto', cmap='viridis')
        plt.colorbar(label='Utilization')
        plt.title('Machine Utilization Heatmap')
        plt.xlabel('Machine Index')
        plt.ylabel('Episode')
        plt.tight_layout()
        
        plt.savefig(os.path.join(reports_dir, "machine_utilization.png"))
        
        # 3. Generate summary statistics
        summary = {
            'total_episodes': len(episode_df),
            'avg_reward': episode_df['total_reward'].mean(),
            'avg_makespan': episode_df['makespan'].mean(),
            'avg_deadline_ratio': episode_df['deadline_ratio'].mean(),
            'avg_machine_util': episode_df['avg_machine_util'].mean(),
            'best_episode': episode_df.loc[episode_df['total_reward'].idxmax()]['episode'],
            'best_reward': episode_df['total_reward'].max(),
            'best_makespan': episode_df['makespan'].min(),
            'best_deadline_ratio': episode_df['deadline_ratio'].max(),
            'best_machine_util': episode_df['avg_machine_util'].max()
        }
        
        with open(os.path.join(reports_dir, "summary.json"), "w") as f:
            json.dump(convert_numpy_types(summary), f, indent=2)
        
        print(f"Reports generated in: {reports_dir}")


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def detailed_machine_analysis(log_dir, experiment_name):
    """
    Perform detailed analysis of machine utilization and material changes.
    
    Args:
        log_dir: Directory containing logs
        experiment_name: Name of the experiment
    """
    experiment_dir = os.path.join(log_dir, experiment_name)
    reports_dir = os.path.join(experiment_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load data
    machine_df = pd.read_csv(os.path.join(experiment_dir, "machine_logs.csv"))
    
    try:
        material_change_df = pd.read_csv(os.path.join(experiment_dir, "material_change_logs.csv"))
        has_material_data = True
    except:
        has_material_data = False
    
    # 1. Machine idle time analysis
    machine_idle_df = machine_df.copy()
    machine_idle_df['idle_ratio'] = 1 - machine_idle_df['utilization']
    
    # Group by episode and machine
    idle_by_episode_machine = machine_idle_df.groupby(['episode', 'machine_idx'])['idle_ratio'].mean().reset_index()
    
    # Create pivot table
    idle_pivot = idle_by_episode_machine.pivot(index='episode', columns='machine_idx', values='idle_ratio')
    
    plt.figure(figsize=(12, 6))
    plt.imshow(idle_pivot, aspect='auto', cmap='coolwarm')
    plt.colorbar(label='Idle Ratio')
    plt.title('Machine Idle Time Heatmap')
    plt.xlabel('Machine Index')
    plt.ylabel('Episode')
    plt.savefig(os.path.join(reports_dir, "machine_idle_heatmap.png"))
    
    # 2. Material change analysis
    if has_material_data:
        # Count material changes per machine
        material_changes_count = material_change_df.groupby(['episode', 'machine_idx']).size().reset_index(name='changes')
        
        # Average setup time per material change
        avg_setup_time = material_change_df.groupby(['episode', 'machine_idx'])['setup_time'].mean().reset_index()
        
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        for machine in material_changes_count['machine_idx'].unique():
            machine_data = material_changes_count[material_changes_count['machine_idx'] == machine]
            plt.plot(machine_data['episode'], machine_data['changes'], label=f'Machine {machine}')
        
        plt.title('Material Changes per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Changes')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        for machine in avg_setup_time['machine_idx'].unique():
            machine_data = avg_setup_time[avg_setup_time['machine_idx'] == machine]
            plt.plot(machine_data['episode'], machine_data['setup_time'], label=f'Machine {machine}')
        
        plt.title('Average Setup Time per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Setup Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(reports_dir, "material_change_analysis.png"))
    
    print(f"Detailed machine analysis completed. Reports saved to: {reports_dir}")
