import json
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from torch_ppo_agent import TorchPPOAgent
from gym_environment import JSPGymEnvironment
from jsp_logger import JSPLogger, convert_numpy_types



def train_gym_ppo(jsp_data_path, num_episodes=500, verbose=True, save_interval=50, batch_size=32, log_dir="logs"):
    """
    Train a PyTorch-based PPO agent for the JSP problem using a Gym environment.
    
    Args:
        jsp_data_path: Path to the JSP data file
        num_episodes: Number of training episodes
        verbose: Whether to output progress messages
        save_interval: Interval for saving checkpoints
        batch_size: Batch size for PPO updates
        log_dir: Directory to store logs
    
    Returns:
        agent: The trained agent
        env: The environment
        logger: The JSP logger
    """
    # Load JSP data
    with open(jsp_data_path, 'r') as f:
        jsp_data = json.load(f)
    
    # Create Gym environment with logging enabled, agent, and logger
    env = JSPGymEnvironment(jsp_data, enable_logging=True)
    agent = TorchPPOAgent(len(jsp_data["jobs"]), jsp_data)
    
    # Initialize logger
    experiment_name = f"jsp_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = JSPLogger(log_dir=log_dir, experiment_name=experiment_name)
    
    # Tracking for visualization
    episode_rewards = []
    episode_makespans = []
    episode_losses = []
    episode_priorities = []  # Average priority of completed jobs
    episode_deadlines = []   # Number of met deadlines
    episode_util = []        # Machine utilization
    
    # Training
    for episode in range(1, num_episodes + 1):
        # Log episode start
        logger.log_episode_start(episode, jsp_data if episode == 1 else None)
        
        state = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0
        
        # Run one episode
        while not done:
            episode_steps += 1
            
            # Select action
            action, action_prob = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Enhance info with material change data for logging
            if 'setup_time' in info and info['setup_time'] > 0:
                # Get operation details for logging
                job_idx = action
                op_idx = env.job_progress[job_idx] - 1  # The operation that was just completed
                if op_idx >= 0:
                    operation = env.jobs[job_idx]["operations"][op_idx]
                    machine_id = operation["machineId"]
                    machine_idx = env.machine_id_to_idx[machine_id]
                    new_material = operation["material"]
                    old_material = env.current_machine_material[machine_idx] if machine_idx < len(env.current_machine_material) else ""
                    
                    # Add material change info
                    info['material_change'] = True
                    info['machine_idx'] = machine_idx
                    info['old_material'] = old_material
                    info['new_material'] = new_material
            
            # Log step data
            logger.log_step(state, action, reward, next_state, done, info)
            
            # Better reward based on makespan improvement
            makespan_reward = agent.get_makespan_reward(state, action, next_state)
            
            # Store experience
            agent.store_experience(state, action, action_prob, makespan_reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
        
        # Update after each episode with potentially larger batch size
        loss = agent.update(batch_size=batch_size)
        
        # Calculate makespan
        makespan = max(state['machine_times'])
        
        # Calculate additional metrics
        completed_jobs = sum([1 for i, progress in enumerate(state['job_progress']) 
                            if progress >= len(agent.jsp_data["jobs"][i]["operations"])])
        
        # Priorities of completed jobs
        priorities = []
        for i, progress in enumerate(state['job_progress']):
            if progress >= len(agent.jsp_data["jobs"][i]["operations"]):
                priorities.append(agent.jsp_data["jobs"][i]["priority"])
        avg_priority = sum(priorities) / len(priorities) if priorities else 0
        
        # Met deadlines
        met_deadlines = 0
        for i, progress in enumerate(state['job_progress']):
            if progress >= len(agent.jsp_data["jobs"][i]["operations"]) and \
               state['current_time'][0] <= agent.jsp_data["jobs"][i]["deadline"]:
                met_deadlines += 1
        
        # Calculate machine utilization
        if state['current_time'][0] > 0:
            machine_util = sum(state['machine_times']) / (state['current_time'][0] * len(state['machine_times']))
        else:
            machine_util = 0
        
        # Log episode end
        logger.log_episode_end(state, total_reward, episode_steps)
        
        # Tracking for visualization
        episode_rewards.append(total_reward)
        episode_makespans.append(makespan)
        episode_losses.append(loss)
        episode_priorities.append(avg_priority)
        episode_deadlines.append(met_deadlines)
        episode_util.append(machine_util)
        
        # Show progress
        if verbose and episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {total_reward:.2f}, Makespan: {makespan}, "
                  f"Loss: {loss:.4f}, Deadlines: {met_deadlines}/{len(agent.jsp_data['jobs'])}, "
                  f"Util: {machine_util:.2f}")
        
        # Save checkpoint
        if episode % save_interval == 0:
            # Ensure directories exist
            os.makedirs('results/models', exist_ok=True)
            
            # Save model
            checkpoint_path = f"results/models/gym_ppo_checkpoint_ep{episode}.pt"
            agent.save_model(checkpoint_path)
            if verbose:
                print(f"Checkpoint saved at: {checkpoint_path}")
    
    # Ensure directories exist
    os.makedirs('results/models', exist_ok=True)
    
    # Current date and time for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"results/models/gym_ppo_model_{timestamp}.pt"
    
    # Save model
    agent.save_model(model_path)
    print(f"Model saved at: {model_path}")
    
    # Save and generate logs
    logger.save_logs()
    logger.generate_reports()
    
    # Save detailed machine utilization statistics
    machine_stats = env.get_machine_utilization_stats()
    material_stats = env.get_material_change_stats()
    job_stats = env.get_job_completion_stats()
    
    # Save detailed statistics to JSON
    os.makedirs('results/stats', exist_ok=True)
    
    with open(f'results/stats/machine_stats_{timestamp}.json', 'w') as f:
        json.dump(convert_numpy_types(machine_stats), f, indent=2)
    
    with open(f'results/stats/material_stats_{timestamp}.json', 'w') as f:
        json.dump(convert_numpy_types(material_stats), f, indent=2)
    
    with open(f'results/stats/job_stats_{timestamp}.json', 'w') as f:
        json.dump(convert_numpy_types(job_stats), f, indent=2)
    
    # Generate additional visualizations for machine utilization
    plt.figure(figsize=(15, 10))
    
    # Machine utilization breakdown
    plt.subplot(2, 2, 1)
    machine_ids = list(machine_stats.keys())
    utilization = [stats['utilization'] for stats in machine_stats.values()]
    setup_ratios = [stats['setup_time_ratio'] for stats in machine_stats.values()]
    idle_ratios = [stats['idle_time_ratio'] for stats in machine_stats.values()]
    processing_ratios = [stats['processing_time_ratio'] for stats in machine_stats.values()]
    
    x = range(len(machine_ids))
    width = 0.2
    
    plt.bar([i - 1.5*width for i in x], utilization, width, label='Total Utilization')
    plt.bar([i - 0.5*width for i in x], processing_ratios, width, label='Processing')
    plt.bar([i + 0.5*width for i in x], setup_ratios, width, label='Setup')
    plt.bar([i + 1.5*width for i in x], idle_ratios, width, label='Idle')
    
    plt.xlabel('Machine')
    plt.ylabel('Ratio')
    plt.title('Machine Time Breakdown')
    plt.xticks(x, machine_ids)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Material changes per machine
    plt.subplot(2, 2, 2)
    changes = [stats['total_changes'] for stats in material_stats.values()]
    setup_times = [stats['total_setup_time'] for stats in material_stats.values()]
    
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    ax1.bar(machine_ids, changes, color='blue', alpha=0.7, label='Changes')
    ax1.set_xlabel('Machine')
    ax1.set_ylabel('Number of Changes', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2.plot(machine_ids, setup_times, 'r-', marker='o', label='Setup Time')
    ax2.set_ylabel('Total Setup Time', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('Material Changes and Setup Times')
    plt.grid(True, axis='y')
    
    # Job completion vs deadline
    plt.subplot(2, 2, 3)
    plt.bar(['All Jobs', 'High Priority'], 
            [job_stats['deadline_ratio'], job_stats['high_priority_met_ratio']], 
            color=['blue', 'green'])
    plt.ylabel('Deadline Met Ratio')
    plt.title('Deadline Performance')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    
    # Priority-weighted completion time
    plt.subplot(2, 2, 4)
    plt.bar(['Average', 'Priority-Weighted'], 
            [job_stats['avg_completion_time'], job_stats['priority_weighted_avg_completion']])
    plt.ylabel('Completion Time')
    plt.title('Job Completion Times')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'results/images/detailed_stats_{timestamp}.png', dpi=300)
    
    print(f"Detailed statistics saved to results/stats/ directory")
    
    # Visualize learning progress with extended metrics
    plt.figure(figsize=(15, 10))
    
    # Moving average for smoother curves
    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Rewards
    plt.subplot(2, 3, 1)
    plt.plot(range(1, num_episodes + 1), episode_rewards, 'b-', alpha=0.3)
    if num_episodes > 10:
        ma_rewards = moving_average(episode_rewards)
        plt.plot(range(10, num_episodes + 1), ma_rewards, 'b-', linewidth=2)
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Makespan
    plt.subplot(2, 3, 2)
    plt.plot(range(1, num_episodes + 1), episode_makespans, 'r-', alpha=0.3)
    if num_episodes > 10:
        ma_makespans = moving_average(episode_makespans)
        plt.plot(range(10, num_episodes + 1), ma_makespans, 'r-', linewidth=2)
    plt.title('Makespan per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Makespan')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Loss
    plt.subplot(2, 3, 3)
    plt.plot(range(1, num_episodes + 1), episode_losses, 'g-', alpha=0.3)
    if num_episodes > 10:
        ma_losses = moving_average(episode_losses)
        plt.plot(range(10, num_episodes + 1), ma_losses, 'g-', linewidth=2)
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Priorities
    plt.subplot(2, 3, 4)
    plt.plot(range(1, num_episodes + 1), episode_priorities, 'm-', alpha=0.3)
    if num_episodes > 10:
        ma_priorities = moving_average(episode_priorities)
        plt.plot(range(10, num_episodes + 1), ma_priorities, 'm-', linewidth=2)
    plt.title('Average Priority of Completed Jobs')
    plt.xlabel('Episode')
    plt.ylabel('Priority')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Deadlines
    plt.subplot(2, 3, 5)
    plt.plot(range(1, num_episodes + 1), episode_deadlines, 'c-', alpha=0.3)
    if num_episodes > 10:
        ma_deadlines = moving_average(episode_deadlines)
        plt.plot(range(10, num_episodes + 1), ma_deadlines, 'c-', linewidth=2)
    plt.title('Met Deadlines')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Machine utilization
    plt.subplot(2, 3, 6)
    plt.plot(range(1, num_episodes + 1), episode_util, 'y-', alpha=0.3)
    if num_episodes > 10:
        ma_util = moving_average(episode_util)
        plt.plot(range(10, num_episodes + 1), ma_util, 'y-', linewidth=2)
    plt.title('Machine Utilization')
    plt.xlabel('Episode')
    plt.ylabel('Utilization')
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Ensure directory exists
    os.makedirs('results/images', exist_ok=True)
    
    # Current date and time for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/images/gym_training_progress_{timestamp}.png'
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    
    print(f"Training progress saved at: {filename}")
    
    return agent, env, logger


def test_gym_ppo(agent, env, log_dir="logs"):
    """
    Test a trained agent on the JSP problem.
    
    Args:
        agent: The trained agent
        env: The environment
        log_dir: Directory to store logs
    
    Returns:
        final_state: The final state
        actions: The executed actions
        test_logger: The logger used for testing
    """
    # Initialize test logger and environment with logging enabled
    test_experiment_name = f"jsp_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_logger = JSPLogger(log_dir=log_dir, experiment_name=test_experiment_name)
    test_logger.log_episode_start(1, agent.jsp_data)
    
    # Enable logging in the environment
    env.enable_logging = True
    
    state = env.reset()
    done = False
    actions = []
    step = 0
    
    print("\nTest run with trained agent:")
    while not done:
        step += 1
        action, _ = agent.select_action(state)
        actions.append(action)
        next_state, reward, done, info = env.step(action)
        
        # Enhance info with material change data for logging
        if 'setup_time' in info and info['setup_time'] > 0:
            # Get operation details for logging
            job_idx = action
            op_idx = env.job_progress[job_idx] - 1  # The operation that was just completed
            if op_idx >= 0:
                operation = env.jobs[job_idx]["operations"][op_idx]
                machine_id = operation["machineId"]
                machine_idx = env.machine_id_to_idx[machine_id]
                new_material = operation["material"]
                old_material = env.current_machine_material[machine_idx] if machine_idx < len(env.current_machine_material) else ""
                
                # Add material change info
                info['material_change'] = True
                info['machine_idx'] = machine_idx
                info['old_material'] = old_material
                info['new_material'] = new_material
        
        # Log step data
        test_logger.log_step(state, action, reward, next_state, done, info)
        
        # Update state
        state = next_state
    
    makespan = max(state['machine_times'])
    print(f"Final Makespan: {makespan}")
    print(f"Action sequence: {actions}")
    
    # Additional metrics
    completed_jobs = sum([1 for i, progress in enumerate(state['job_progress']) 
                        if progress >= len(agent.jsp_data["jobs"][i]["operations"])])
    
    met_deadlines = 0
    for i, progress in enumerate(state['job_progress']):
        if progress >= len(agent.jsp_data["jobs"][i]["operations"]) and \
           state['current_time'][0] <= agent.jsp_data["jobs"][i]["deadline"]:
            met_deadlines += 1
    
    print(f"Completed Jobs: {completed_jobs}/{len(agent.jsp_data['jobs'])}")
    print(f"Met Deadlines: {met_deadlines}/{len(agent.jsp_data['jobs'])}")
    
    if state['current_time'][0] > 0:
        machine_util = sum(state['machine_times']) / (state['current_time'][0] * len(state['machine_times']))
        print(f"Machine Utilization: {machine_util:.2f}")
    
    # Log episode end and generate reports
    test_logger.log_episode_end(state, sum([info.get('reward', 0) for info in test_logger.step_data]), step)
    test_logger.save_logs()
    test_logger.generate_reports()
    
    # Get and save detailed statistics
    machine_stats = env.get_machine_utilization_stats()
    material_stats = env.get_material_change_stats()
    job_stats = env.get_job_completion_stats()
    
    # Save detailed statistics to JSON
    os.makedirs(os.path.join(log_dir, test_experiment_name, 'stats'), exist_ok=True)
    
    with open(os.path.join(log_dir, test_experiment_name, 'stats', 'machine_stats.json'), 'w') as f:
        json.dump(convert_numpy_types(machine_stats), f, indent=2)
    
    with open(os.path.join(log_dir, test_experiment_name, 'stats', 'material_stats.json'), 'w') as f:
        json.dump(convert_numpy_types(material_stats), f, indent=2)
    
    with open(os.path.join(log_dir, test_experiment_name, 'stats', 'job_stats.json'), 'w') as f:
        json.dump(convert_numpy_types(job_stats), f, indent=2)
    
    # Print detailed machine utilization
    print("\nDetailed Machine Utilization:")
    for machine_id, stats in machine_stats.items():
        print(f"Machine {machine_id}:")
        print(f"  Total Utilization: {stats['utilization']:.2f}")
        print(f"  Processing Time: {stats['processing_time_ratio']:.2f} ({stats['total_busy_time']:.1f})")
        print(f"  Setup Time: {stats['setup_time_ratio']:.2f} ({stats['total_setup_time']:.1f})")
        print(f"  Idle Time: {stats['idle_time_ratio']:.2f} ({stats['total_idle_time']:.1f})")
    
    # Print material changes
    print("\nMaterial Changes:")
    for machine_id, stats in material_stats.items():
        print(f"Machine {machine_id}: {stats['total_changes']} changes, {stats['total_setup_time']:.1f} setup time")
        if stats['material_counts']:
            print("  Materials processed:")
            for material, count in stats['material_counts'].items():
                print(f"    {material}: {count} times")
    
    # Print job statistics
    print("\nJob Completion Statistics:")
    print(f"  Completed Jobs: {job_stats['completed_jobs']}/{len(agent.jsp_data['jobs'])}")
    print(f"  Average Completion Time: {job_stats['avg_completion_time']:.1f}")
    print(f"  Met Deadlines: {job_stats['met_deadlines']}/{job_stats['completed_jobs']} ({job_stats['deadline_ratio']*100:.1f}%)")
    print(f"  High Priority Jobs Met Deadlines: {job_stats['high_priority_met_ratio']*100:.1f}%")
    
    return state, actions, test_logger


if __name__ == "__main__":
    # Path to JSP data file
    jsp_data_path = "data.json"
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results/stats", exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train and test a PPO agent for JSP')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--save-interval', type=int, default=50, help='Interval for saving checkpoints')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for PPO updates')
    parser.add_argument('--test-only', action='store_true', help='Only test, no training')
    parser.add_argument('--model-path', type=str, help='Path to model for testing')
    args = parser.parse_args()
    
    if args.test_only and args.model_path:
        # Load JSP data
        with open(jsp_data_path, 'r') as f:
            jsp_data = json.load(f)
        
        # Create environment and agent
        env = JSPGymEnvironment(jsp_data, enable_logging=True)
        agent = TorchPPOAgent(len(jsp_data["jobs"]), jsp_data)
        
        # Load model
        agent.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
        
        # Test trained agent
        final_state, actions, test_logger = test_gym_ppo(agent, env)
        
        # Perform detailed machine analysis
        from jsp_logger import detailed_machine_analysis
        detailed_machine_analysis("logs", test_logger.experiment_name)
    else:
        # Train agent
        trained_agent, env, logger = train_gym_ppo(jsp_data_path, num_episodes=args.episodes, 
                                                save_interval=args.save_interval, batch_size=args.batch_size)
        
        # Test trained agent
        final_state, actions, test_logger = test_gym_ppo(trained_agent, env)
        
        # Perform detailed machine analysis
        from jsp_logger import detailed_machine_analysis
        detailed_machine_analysis("logs", test_logger.experiment_name)
