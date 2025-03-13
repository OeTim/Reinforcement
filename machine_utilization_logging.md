# Machine Utilization Logging in JSP Environment

This document explains the detailed machine utilization logging capabilities implemented in the Job Shop Scheduling (JSP) Gym Environment. The logging system tracks machine utilization, material changes, and job completion statistics during training and testing.

## Table of Contents

1. [Overview](#overview)
2. [Key Metrics Tracked](#key-metrics-tracked)
3. [Enabling Logging](#enabling-logging)
4. [Data Collection](#data-collection)
5. [Statistics and Reports](#statistics-and-reports)
6. [Visualization](#visualization)
7. [Integration with Training](#integration-with-training)

## Overview

The machine utilization logging system provides detailed insights into how machines are utilized during the scheduling process. It tracks various metrics related to machine utilization, material changes, and job completion, allowing for comprehensive analysis of the scheduling algorithm's performance.

## Key Metrics Tracked

### Machine Utilization
- **Total Utilization**: The ratio of time a machine is busy (processing + setup) to the total time
- **Processing Time Ratio**: The ratio of time a machine is actively processing operations
- **Setup Time Ratio**: The ratio of time a machine spends on setup operations
- **Idle Time Ratio**: The ratio of time a machine is idle

### Material Changes
- **Total Changes**: The number of material changes on each machine
- **Setup Time**: The total time spent on setups due to material changes
- **Material Distribution**: The count of each material type processed by each machine

### Job Completion
- **Completion Time**: The time at which each job is completed
- **Deadline Performance**: Whether jobs met their deadlines
- **Priority-Weighted Statistics**: Statistics weighted by job priorities
- **High Priority Job Performance**: Special tracking for high priority jobs (priority ≥ 7)

## Enabling Logging

Logging can be enabled when creating the JSP Gym Environment:

```python
# Enable logging in the environment
env = JSPGymEnvironment(jsp_data, enable_logging=True, log_level=logging.INFO)
```

The `enable_logging` parameter enables detailed logging, and the `log_level` parameter sets the logging level (default: INFO).

## Data Collection

The logging system collects data at various points during the scheduling process:

### Operation Execution
When an operation is executed, the following data is recorded:
- Job and operation identifiers
- Machine identifier
- Start and end times
- Processing and setup times
- Material being processed

### Material Changes
When a material change occurs on a machine, the following data is recorded:
- Machine identifier
- Old and new materials
- Setup time required for the change
- Time at which the change occurred

### Job Completion
When a job is completed, the following data is recorded:
- Job identifier
- Completion time
- Deadline
- Whether the deadline was met
- Job priority

## Statistics and Reports

The environment provides methods to retrieve detailed statistics:

### Machine Utilization Statistics
```python
machine_stats = env.get_machine_utilization_stats()
```

This returns a dictionary with detailed statistics for each machine, including:
- Utilization ratios (total, processing, setup, idle)
- Total times (busy, setup, idle)

### Material Change Statistics
```python
material_stats = env.get_material_change_stats()
```

This returns a dictionary with statistics about material changes for each machine, including:
- Total number of changes
- Total setup time
- Count of each material type processed

### Job Completion Statistics
```python
job_stats = env.get_job_completion_stats()
```

This returns a dictionary with statistics about job completions, including:
- Number of completed jobs
- Average completion time
- Deadline performance
- Priority-weighted statistics

## Visualization

The system includes visualization capabilities to generate charts and graphs of the collected data:

### Built-in Visualizations
The training script automatically generates visualizations after training and testing:
- Machine time breakdown (utilization, processing, setup, idle)
- Material changes and setup times per machine
- Deadline performance (all jobs vs. high priority)
- Job completion times (average vs. priority-weighted)

### Custom Reports
The `machine_utilization_report.py` script can generate comprehensive markdown reports with detailed analysis and recommendations:

```bash
python machine_utilization_report.py --log-dir logs --experiment experiment_name
```

This generates a report with:
- Executive summary
- Detailed machine utilization statistics
- Material change analysis
- Job completion statistics
- Recommendations for improvement

## Integration with Training

The logging system is integrated with the training script (`train_gym_ppo.py`):

### During Training
- The environment logs detailed information about each step
- The JSPLogger tracks episode and step data
- Statistics are saved after training

### During Testing
- The environment continues logging during testing
- Detailed statistics are generated and saved
- Visualizations are created to analyze performance

### Command-Line Options
The training script supports command-line options for logging:
```bash
python train_gym_ppo.py --episodes 300 --save-interval 50
```

For testing only:
```bash
python train_gym_ppo.py --test-only --model-path models/model_name.pt
```

## Example Output

### Console Output
```
Detailed Machine Utilization:
Machine M1:
  Total Utilization: 0.85
  Processing Time: 0.70 (350.5)
  Setup Time: 0.15 (75.0)
  Idle Time: 0.15 (75.0)

Material Changes:
Machine M1: 10 changes, 75.0 setup time
  Materials processed:
    Material1: 3 times
    Material2: 4 times
    Material3: 3 times

Job Completion Statistics:
  Completed Jobs: 15/15
  Average Completion Time: 425.5
  Met Deadlines: 12/15 (80.0%)
  High Priority Jobs Met Deadlines: 90.0%
```

### Generated Files
- `machine_stats.json`: Detailed machine utilization statistics
- `material_stats.json`: Material change statistics
- `job_stats.json`: Job completion statistics
- `detailed_stats.png`: Visualizations of key metrics
- `machine_utilization_report.md`: Comprehensive markdown report

## Conclusion

The machine utilization logging system provides valuable insights into the performance of scheduling algorithms in the JSP environment. By tracking detailed metrics and generating comprehensive reports, it helps identify bottlenecks, optimize machine utilization, and improve overall scheduling performance.
