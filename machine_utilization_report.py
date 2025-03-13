#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Utilization Report Generator

This script generates detailed markdown reports about machine utilization,
material changes, and job completion statistics from the JSP environment.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def generate_machine_utilization_report(stats_dir, output_file):
    """
    Generate a detailed markdown report about machine utilization.
    
    Args:
        stats_dir: Directory containing the statistics JSON files
        output_file: Path to the output markdown file
    """
    # Load statistics
    with open(os.path.join(stats_dir, 'machine_stats.json'), 'r') as f:
        machine_stats = json.load(f)
    
    with open(os.path.join(stats_dir, 'material_stats.json'), 'r') as f:
        material_stats = json.load(f)
    
    with open(os.path.join(stats_dir, 'job_stats.json'), 'r') as f:
        job_stats = json.load(f)
    
    # Create report
    with open(output_file, 'w') as f:
        # Header
        f.write("# Machine Utilization Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Table of Contents
        f.write("## Table of Contents\n\n")
        f.write("1. [Executive Summary](#executive-summary)\n")
        f.write("2. [Machine Utilization](#machine-utilization)\n")
        f.write("3. [Material Changes](#material-changes)\n")
        f.write("4. [Job Completion Statistics](#job-completion-statistics)\n")
        f.write("5. [Recommendations](#recommendations)\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        
        # Calculate overall statistics
        avg_utilization = sum(stats['utilization'] for stats in machine_stats.values()) / len(machine_stats)
        avg_setup_ratio = sum(stats['setup_time_ratio'] for stats in machine_stats.values()) / len(machine_stats)
        avg_idle_ratio = sum(stats['idle_time_ratio'] for stats in machine_stats.values()) / len(machine_stats)
        avg_processing_ratio = sum(stats['processing_time_ratio'] for stats in machine_stats.values()) / len(machine_stats)
        
        total_material_changes = sum(stats['total_changes'] for stats in material_stats.values())
        total_setup_time = sum(stats['total_setup_time'] for stats in material_stats.values())
        
        f.write(f"This report provides a detailed analysis of machine utilization, material changes, and job completion statistics.\n\n")
        f.write("### Key Findings:\n\n")
        f.write(f"- **Average Machine Utilization**: {avg_utilization:.2%}\n")
        f.write(f"- **Total Material Changes**: {total_material_changes}\n")
        f.write(f"- **Total Setup Time**: {total_setup_time:.1f} time units\n")
        f.write(f"- **Completed Jobs**: {job_stats['completed_jobs']}\n")
        f.write(f"- **Deadline Met Ratio**: {job_stats['deadline_ratio']:.2%}\n")
        f.write(f"- **High Priority Jobs Deadline Met Ratio**: {job_stats['high_priority_met_ratio']:.2%}\n\n")
        
        # Machine Utilization
        f.write("## Machine Utilization\n\n")
        f.write("### Overall Utilization\n\n")
        f.write("| Machine | Total Utilization | Processing Time | Setup Time | Idle Time |\n")
        f.write("|---------|-------------------|----------------|------------|----------|\n")
        
        for machine_id, stats in machine_stats.items():
            f.write(f"| {machine_id} | {stats['utilization']:.2%} | {stats['processing_time_ratio']:.2%} | {stats['setup_time_ratio']:.2%} | {stats['idle_time_ratio']:.2%} |\n")
        
        f.write("\n### Detailed Time Breakdown\n\n")
        f.write("| Machine | Total Busy Time | Total Setup Time | Total Idle Time |\n")
        f.write("|---------|----------------|-----------------|----------------|\n")
        
        for machine_id, stats in machine_stats.items():
            f.write(f"| {machine_id} | {stats['total_busy_time']:.1f} | {stats['total_setup_time']:.1f} | {stats['total_idle_time']:.1f} |\n")
        
        # Material Changes
        f.write("\n## Material Changes\n\n")
        f.write("### Overview\n\n")
        f.write("| Machine | Total Changes | Total Setup Time | Avg Setup Time per Change |\n")
        f.write("|---------|--------------|-----------------|---------------------------|\n")
        
        for machine_id, stats in material_stats.items():
            avg_setup = stats['total_setup_time'] / stats['total_changes'] if stats['total_changes'] > 0 else 0
            f.write(f"| {machine_id} | {stats['total_changes']} | {stats['total_setup_time']:.1f} | {avg_setup:.1f} |\n")
        
        f.write("\n### Material Change Details\n\n")
        
        for machine_id, stats in material_stats.items():
            f.write(f"#### Machine {machine_id}\n\n")
            
            if not stats['material_counts']:
                f.write("No material changes recorded.\n\n")
                continue
            
            f.write("| Material | Count |\n")
            f.write("|----------|-------|\n")
            
            for material, count in stats['material_counts'].items():
                material_name = material if material else "Empty"
                f.write(f"| {material_name} | {count} |\n")
            
            f.write("\n")
        
        # Job Completion Statistics
        f.write("## Job Completion Statistics\n\n")
        f.write("### Overview\n\n")
        f.write(f"- **Completed Jobs**: {job_stats['completed_jobs']}\n")
        f.write(f"- **Average Completion Time**: {job_stats['avg_completion_time']:.1f} time units\n")
        f.write(f"- **Met Deadlines**: {job_stats['met_deadlines']} out of {job_stats['completed_jobs']} ({job_stats['deadline_ratio']:.2%})\n")
        f.write(f"- **Priority-Weighted Average Completion Time**: {job_stats['priority_weighted_avg_completion']:.1f} time units\n")
        f.write(f"- **High Priority Jobs Met Deadlines**: {job_stats['high_priority_met_ratio']:.2%}\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        # Identify machines with low utilization
        low_util_machines = [machine_id for machine_id, stats in machine_stats.items() if stats['utilization'] < avg_utilization * 0.8]
        
        # Identify machines with high setup times
        high_setup_machines = [machine_id for machine_id, stats in machine_stats.items() if stats['setup_time_ratio'] > avg_setup_ratio * 1.2]
        
        # Generate recommendations
        if low_util_machines:
            f.write("### Underutilized Machines\n\n")
            f.write(f"The following machines have utilization below 80% of the average ({avg_utilization:.2%}):\n\n")
            for machine in low_util_machines:
                f.write(f"- **{machine}**: {machine_stats[machine]['utilization']:.2%} utilization\n")
            
            f.write("\n**Recommendations:**\n")
            f.write("- Redistribute workload to better balance machine utilization\n")
            f.write("- Consider assigning more operations to these machines\n")
            f.write("- Evaluate if these machines have limited capabilities that restrict their use\n\n")
        
        if high_setup_machines:
            f.write("### High Setup Time Machines\n\n")
            f.write(f"The following machines spend more than 120% of the average ({avg_setup_ratio:.2%}) time on setups:\n\n")
            for machine in high_setup_machines:
                f.write(f"- **{machine}**: {machine_stats[machine]['setup_time_ratio']:.2%} of time spent on setups\n")
            
            f.write("\n**Recommendations:**\n")
            f.write("- Group similar materials to reduce the number of setups\n")
            f.write("- Review the setup time requirements for these machines\n")
            f.write("- Consider technological improvements to reduce setup times\n\n")
        
        if job_stats['deadline_ratio'] < 0.9:
            f.write("### Deadline Performance\n\n")
            f.write(f"Only {job_stats['deadline_ratio']:.2%} of jobs met their deadlines.\n\n")
            
            f.write("**Recommendations:**\n")
            f.write("- Review job priorities and deadlines for feasibility\n")
            f.write("- Consider adjusting the scheduling algorithm to prioritize jobs with tight deadlines\n")
            f.write("- Analyze the critical path of jobs that missed deadlines\n\n")
        
        if job_stats['high_priority_met_ratio'] < job_stats['deadline_ratio']:
            f.write("### High Priority Job Performance\n\n")
            f.write(f"High priority jobs have a lower deadline met ratio ({job_stats['high_priority_met_ratio']:.2%}) than the overall average ({job_stats['deadline_ratio']:.2%}).\n\n")
            
            f.write("**Recommendations:**\n")
            f.write("- Adjust the scheduling algorithm to give higher weight to high priority jobs\n")
            f.write("- Review the deadlines of high priority jobs for feasibility\n")
            f.write("- Consider dedicating specific machines to high priority jobs\n\n")


def generate_visualizations(stats_dir, output_dir):
    """
    Generate visualizations for machine utilization statistics.
    
    Args:
        stats_dir: Directory containing the statistics JSON files
        output_dir: Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load statistics
    with open(os.path.join(stats_dir, 'machine_stats.json'), 'r') as f:
        machine_stats = json.load(f)
    
    with open(os.path.join(stats_dir, 'material_stats.json'), 'r') as f:
        material_stats = json.load(f)
    
    with open(os.path.join(stats_dir, 'job_stats.json'), 'r') as f:
        job_stats = json.load(f)
    
    # Prepare data for visualizations
    machine_ids = list(machine_stats.keys())
    
    # Machine utilization breakdown
    utilization_data = []
    for machine_id, stats in machine_stats.items():
        utilization_data.append({
            'Machine': machine_id,
            'Type': 'Processing',
            'Value': stats['processing_time_ratio']
        })
        utilization_data.append({
            'Machine': machine_id,
            'Type': 'Setup',
            'Value': stats['setup_time_ratio']
        })
        utilization_data.append({
            'Machine': machine_id,
            'Type': 'Idle',
            'Value': stats['idle_time_ratio']
        })
    
    df_utilization = pd.DataFrame(utilization_data)
    
    # Material changes
    material_data = []
    for machine_id, stats in material_stats.items():
        for material, count in stats['material_counts'].items():
            material_name = material if material else "Empty"
            material_data.append({
                'Machine': machine_id,
                'Material': material_name,
                'Count': count
            })
    
    if material_data:
        df_materials = pd.DataFrame(material_data)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Machine utilization breakdown
    ax = plt.subplot(2, 2, 1)
    sns.barplot(x='Machine', y='Value', hue='Type', data=df_utilization, ax=ax)
    ax.set_title('Machine Time Breakdown')
    ax.set_ylabel('Ratio')
    ax.set_ylim(0, 1)
    
    # Material changes per machine
    ax = plt.subplot(2, 2, 2)
    changes = [stats['total_changes'] for stats in material_stats.values()]
    setup_times = [stats['total_setup_time'] for stats in material_stats.values()]
    
    ax1 = ax
    ax2 = ax.twinx()
    
    ax1.bar(machine_ids, changes, color='blue', alpha=0.7, label='Changes')
    ax1.set_xlabel('Machine')
    ax1.set_ylabel('Number of Changes', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2.plot(machine_ids, setup_times, 'r-', marker='o', label='Setup Time')
    ax2.set_ylabel('Total Setup Time', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax.set_title('Material Changes and Setup Times')
    
    # Job completion vs deadline
    ax = plt.subplot(2, 2, 3)
    ax.bar(['All Jobs', 'High Priority'], 
           [job_stats['deadline_ratio'], job_stats['high_priority_met_ratio']], 
           color=['blue', 'green'])
    ax.set_ylabel('Deadline Met Ratio')
    ax.set_title('Deadline Performance')
    ax.set_ylim(0, 1)
    
    # Priority-weighted completion time
    ax = plt.subplot(2, 2, 4)
    ax.bar(['Average', 'Priority-Weighted'], 
           [job_stats['avg_completion_time'], job_stats['priority_weighted_avg_completion']])
    ax.set_ylabel('Completion Time')
    ax.set_title('Job Completion Times')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'machine_utilization_summary.png'), dpi=300)
    
    # Material distribution per machine (if we have material data)
    if material_data:
        plt.figure(figsize=(14, 8))
        if len(df_materials) > 0:
            sns.countplot(x='Machine', hue='Material', data=df_materials)
            plt.title('Material Distribution per Machine')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'material_distribution.png'), dpi=300)


def main():
    """
    Main function to generate machine utilization reports.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate machine utilization reports')
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory containing logs')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    args = parser.parse_args()
    
    if args.experiment:
        # Generate report for a specific experiment
        stats_dir = os.path.join(args.log_dir, args.experiment, 'stats')
        output_dir = os.path.join(args.log_dir, args.experiment, 'reports')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'machine_utilization_report.md')
        
        generate_machine_utilization_report(stats_dir, output_file)
        generate_visualizations(stats_dir, output_dir)
        
        print(f"Report generated: {output_file}")
        print(f"Visualizations saved to: {output_dir}")
    else:
        # Find all experiments with stats directories
        experiments = []
        for exp in os.listdir(args.log_dir):
            stats_dir = os.path.join(args.log_dir, exp, 'stats')
            if os.path.isdir(stats_dir) and os.path.exists(os.path.join(stats_dir, 'machine_stats.json')):
                experiments.append(exp)
        
        if not experiments:
            print("No experiments with statistics found.")
            return
        
        print(f"Found {len(experiments)} experiments with statistics.")
        
        # Generate reports for all experiments
        for exp in experiments:
            stats_dir = os.path.join(args.log_dir, exp, 'stats')
            output_dir = os.path.join(args.log_dir, exp, 'reports')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, 'machine_utilization_report.md')
            
            generate_machine_utilization_report(stats_dir, output_file)
            generate_visualizations(stats_dir, output_dir)
            
            print(f"Report generated for {exp}: {output_file}")


if __name__ == "__main__":
    main()
