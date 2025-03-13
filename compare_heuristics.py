import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from collections import deque
from gym_environment import JSPGymEnvironment
from torch_ppo_agent import TorchPPOAgent

# Load JSP data
def load_jsp_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Heuristic implementations
def fifo_heuristic(state, jsp_data):
    """First In First Out - selects the job with the lowest index that can be processed"""
    valid_actions_mask = state['valid_actions_mask']
    for job_idx, is_valid in enumerate(valid_actions_mask):
        if is_valid == 1:
            return job_idx
    return 0  # Fallback

def lifo_heuristic(state, jsp_data):
    """Last In First Out - selects the job with the highest index that can be processed"""
    valid_actions_mask = state['valid_actions_mask']
    for job_idx in range(len(valid_actions_mask) - 1, -1, -1):
        if valid_actions_mask[job_idx] == 1:
            return job_idx
    return 0  # Fallback

def spt_heuristic(state, jsp_data):
    """Shortest Processing Time - selects the job with the shortest next operation"""
    valid_actions_mask = state['valid_actions_mask']
    job_progress = state['job_progress']
    
    min_time = float('inf')
    selected_job = 0
    
    for job_idx, is_valid in enumerate(valid_actions_mask):
        if is_valid == 1:
            op_idx = job_progress[job_idx]
            if op_idx < len(jsp_data["jobs"][job_idx]["operations"]):
                proc_time = jsp_data["jobs"][job_idx]["operations"][op_idx]["processingTime"]
                if proc_time < min_time:
                    min_time = proc_time
                    selected_job = job_idx
    
    return selected_job

def random_heuristic(state, jsp_data):
    """Random selection from valid jobs"""
    valid_actions_mask = state['valid_actions_mask']
    valid_jobs = [job_idx for job_idx, is_valid in enumerate(valid_actions_mask) if is_valid == 1]
    
    if valid_jobs:
        return random.choice(valid_jobs)
    return 0  # Fallback

def run_heuristic(env, heuristic_func, jsp_data, num_episodes=10):
    """Run a heuristic for multiple episodes and return average metrics"""
    makespans = []
    utilizations = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = heuristic_func(state, jsp_data)
            state, reward, done, info = env.step(action)
        
        # Calculate metrics
        makespan = max(env.machine_times)
        
        # Calculate machine utilization
        total_processing_time = sum(env.machine_times)
        total_possible_time = makespan * len(env.machine_times)
        utilization = total_processing_time / total_possible_time if total_possible_time > 0 else 0
        
        makespans.append(makespan)
        utilizations.append(utilization)
    
    return np.mean(makespans), np.mean(utilizations)

def run_ppo_agent(env, model_path, jsp_data, num_episodes=10):
    """Run the PPO agent with a loaded model for multiple episodes"""
    # Initialize agent
    agent = TorchPPOAgent(len(jsp_data["jobs"]), jsp_data)
    
    # Load the model state to check dimensions
    model_state = torch.load(model_path)
    
    # Adjust all dimensions to match the saved model
    if 'node_embedding' in model_state:
        # Get embedding dimension from saved model
        embedding_dim = model_state['node_embedding']['weight'].shape[0]
        agent.embedding_dim = embedding_dim
        
        # Get hidden dimension from saved model
        hidden_dim = model_state['graph_layer1']['weight'].shape[0]
        agent.hidden_dim = hidden_dim
        
        # Recreate all layers with correct dimensions
        node_features = 7  # From the original code
        agent.node_embedding = torch.nn.Linear(node_features, agent.embedding_dim)
        agent.graph_layer1 = torch.nn.Linear(agent.embedding_dim, agent.hidden_dim)
        agent.graph_layer2 = torch.nn.Linear(agent.hidden_dim, agent.hidden_dim)
        agent.output_layer = torch.nn.Linear(agent.hidden_dim, agent.num_jobs)
    
    # Now load the model with adjusted architecture
    agent.load_model(model_path)
    
    makespans = []
    utilizations = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action, _ = agent.select_action(state)
            state, reward, done, info = env.step(action)
        
        # Calculate metrics
        makespan = max(env.machine_times)
        
        # Calculate machine utilization
        total_processing_time = sum(env.machine_times)
        total_possible_time = makespan * len(env.machine_times)
        utilization = total_processing_time / total_possible_time if total_possible_time > 0 else 0
        
        makespans.append(makespan)
        utilizations.append(utilization)
    
    return np.mean(makespans), np.mean(utilizations)

def compare_heuristics(jsp_data_path, model_path, num_episodes=10):
    """Compare different heuristics and the PPO agent"""
    jsp_data = load_jsp_data(jsp_data_path)
    env = JSPGymEnvironment(jsp_data)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run each heuristic
    fifo_makespan, fifo_util = run_heuristic(env, fifo_heuristic, jsp_data, num_episodes)
    lifo_makespan, lifo_util = run_heuristic(env, lifo_heuristic, jsp_data, num_episodes)
    spt_makespan, spt_util = run_heuristic(env, spt_heuristic, jsp_data, num_episodes)
    random_makespan, random_util = run_heuristic(env, random_heuristic, jsp_data, num_episodes)
    
    # Run PPO agent
    ppo_makespan, ppo_util = run_ppo_agent(env, model_path, jsp_data, num_episodes)
    
    # Prepare data for plotting
    heuristics = ['FIFO', 'LIFO', 'SPT', 'RANDOM', 'PPO']
    makespans = [fifo_makespan, lifo_makespan, spt_makespan, random_makespan, ppo_makespan]
    utilizations = [fifo_util, lifo_util, spt_util, random_util, ppo_util]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot makespan (lower is better)
    ax1.bar(heuristics, makespans, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax1.set_title('Makespan Comparison (lower is better)')
    ax1.set_ylabel('Makespan')
    
    # Add values on top of bars
    for i, v in enumerate(makespans):
        ax1.text(i, v + 5, f'{v:.1f}', ha='center')
    
    # Plot utilization (higher is better)
    ax2.bar(heuristics, utilizations, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax2.set_title('Machine Utilization Comparison (higher is better)')
    ax2.set_ylabel('Utilization')
    
    # Add values on top of bars
    for i, v in enumerate(utilizations):
        ax2.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('heuristic_comparison.png')
    plt.show()
    
    # Print results
    print("\nResults Summary:")
    print("-" * 50)
    print(f"{'Heuristic':<10} {'Makespan':<15} {'Utilization':<15}")
    print("-" * 50)
    for h, m, u in zip(heuristics, makespans, utilizations):
        print(f"{h:<10} {m:<15.2f} {u:<15.2f}")

if __name__ == "__main__":
    jsp_data_path = "/Users/paulmill/Desktop/Reinforcement Learning/data.json"
    model_path = "/Users/paulmill/Desktop/Reinforcement Learning/results/models/gym_ppo_checkpoint_ep50.pt"
    
    # Run comparison with 10 episodes per heuristic
    compare_heuristics(jsp_data_path, model_path, num_episodes=10)