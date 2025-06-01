import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Plotting Style (Optional but recommended) ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

# --- Helper function to load data ---
def load_iteration_log(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Iteration log file not found: {filepath}")
        return pd.DataFrame()
    # Specify dtypes for columns that might be read incorrectly
    # Dtypes based on the headers in your Approach 3 iteration log
    dtype_spec_iter = {
        'hpo_approach_id': str, 'run_id': str, 'macro_iteration': int,
        'iteration_start_time_abs': float, 'iteration_end_time_abs': float, 'iteration_duration_s': float,
        'num_configs_screened_cpu_total': 'Int64', # Use nullable Int64 if N/A possible
        'num_configs_promoted_to_gpu': 'Int64',
        'num_configs_evaluated_gpu': 'Int64', # For A3, A1/A2 use num_configs_evaluated
        'num_configs_evaluated': 'Int64',     # For A1/A2
        'best_fitness_this_iteration_gpu': float, # For A3
        'best_fitness_this_iteration': float,     # For A1/A2
        'best_fitness_so_far_gpu': float,         # For A3
        'best_fitness_so_far': float,             # For A1/A2
        'wall_clock_time_to_best_so_far_s': float,
        'total_cpu_screen_time_s_iter': str, # Read as str due to "N/A", convert later
        'approx_cpu_busy_time_s': float,     # For A1/A2
        'total_gpu_eval_time_s_iter': float, # For A3
        'approx_gpu_busy_time_s': float      # For A1/A2
    }
    try:
        df = pd.read_csv(filepath, dtype=dtype_spec_iter)
        # Convert time columns for A3 if they were string "N/A"
        if 'total_cpu_screen_time_s_iter' in df.columns:
             df['total_cpu_screen_time_s_iter'] = pd.to_numeric(df['total_cpu_screen_time_s_iter'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()


def load_config_log(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Config log file not found: {filepath}")
        return pd.DataFrame()
    # Dtypes based on your Approach 3 config log
    dtype_spec_cfg = {
        'hpo_approach_id': str, 'run_id': str, 'macro_iteration': int, 'origin_cpu_id': 'Int64',
        'config_screened_hash': str, 'screened_lr': float, 'screened_num_layers': int,
        'screened_hidden_size': int, 'screened_dropout': float, 'screened_l2': float,
        'gpu_device_used': str, 'gpu_processing_start_time_abs': float, 'gpu_processing_duration_s': float,
        'gpu_test_accuracy': float, 'gpu_train_accuracy': float, 'gpu_fitness_score': float,
        'gpu_run_num_layers': int, 'gpu_run_hidden_size': int,
        'screen_loss_decrease': float, 'screen_accuracy_1epoch': float, 'cpu_screen_duration_s': float,
        # For A1/A2 logs (they have slightly different headers)
        'iteration': int, 'config_original_idx': 'Int64', 'lr': float, 'num_layers': int,
        'hidden_size': int, 'dropout_rate': float, 'weight_decay': float, 'device_used': str,
        'processing_start_time_abs': float, 'processing_duration_s': float,
        'test_accuracy': float, 'train_accuracy': float, 'fitness_score': float
    }
    try:
        return pd.read_csv(filepath, dtype=dtype_spec_cfg)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return pd.DataFrame()

# --- File Paths (Update these with your actual filenames/paths if they differ) ---
# Assuming you have multiple run_ids per approach, you might load them in a loop
# Or, for simplicity, specify one representative run per approach for now.
# Example:
# run_id_a1 = "xxxxxxxx" (replace with your actual run_id from filename)
# run_id_a2 = "yyyyyyyy"
# run_id_a3 = "zzzzzzzz"

# You'll need to manually find your run_ids from the filenames generated.
# For now, let's hardcode example filenames.
# In a real scenario, you'd discover these or pass them as arguments.

# Placeholder for a function to find the latest run_id for an approach if needed
def find_latest_run_id(approach_prefix, log_type_prefix):
    # Example: approach_prefix = "A1", log_type_prefix = "hpo_iteration_log_"
    # This is a simplified example; real discovery might be more complex.
    all_files = [f for f in os.listdir('.') if f.startswith(log_type_prefix + approach_prefix)]
    if not all_files: return None
    # Assuming filename format like: hpo_iteration_log_A1_RUNID.csv
    # Extract RUNID and sort by modification time or RUNID itself if it's sortable (like a timestamp)
    # This part needs robust implementation based on your exact filenames.
    # For now, let's assume we will manually set run_ids.
    return None # Needs proper implementation


# Manually set these for the specific runs you want to compare
# Replace "YOUR_RUN_ID_A1" etc. with the actual IDs from your filenames
RUN_ID_A1 = "7c7346df" # e.g., "d2ef5b01"
RUN_ID_A2 = "c6468262"
RUN_ID_A3 = "912a125b" # e.g., "78eb9302"

# Construct file paths
# Iteration Logs
iter_log_a1_path = f"hpo_iteration_log_A1_{RUN_ID_A1}.csv"
iter_log_a2_path = f"hpo_iteration_log_A2_{RUN_ID_A2}.csv"
iter_log_a3_path = f"hpo_iteration_log_A3_{RUN_ID_A3}.csv"

# Config Logs (GPU evaluated for A3, all for A1/A2)
config_log_a1_path = f"hpo_config_log_A1_{RUN_ID_A1}.csv"
config_log_a2_path = f"hpo_config_log_A2_{RUN_ID_A2}.csv"
config_log_a3_path = f"hpo_config_log_A3_{RUN_ID_A3}.csv"

# CPU Screen Log (Only for A3)
cpu_screen_log_a3_path = f"hpo_cpuscreen_log_A3_{RUN_ID_A3}.csv"


# Load the data
df_iter_a1 = load_iteration_log(iter_log_a1_path)
df_iter_a2 = load_iteration_log(iter_log_a2_path)
df_iter_a3 = load_iteration_log(iter_log_a3_path)

df_config_a1 = load_config_log(config_log_a1_path)
df_config_a2 = load_config_log(config_log_a2_path)
df_config_a3_gpu = load_config_log(config_log_a3_path) # GPU evaluated configs for A3
df_config_a3_cpu = load_config_log(cpu_screen_log_a3_path) # CPU screened configs for A3


# Combine iteration data for easier plotting if multiple runs per approach (not done here yet)
# For now, assuming one run per approach loaded.
# If you have multiple runs, you'd load all and then group by 'hpo_approach_id' and 'iteration'
# and calculate mean/std for metrics.

# Add a 'wall_clock_time_cumulative' column to iteration logs
# This assumes iteration_start_time_abs is the start of the HPO process for the first iteration.
# For subsequent iterations, it's the start of that specific iteration.
# We need cumulative time from the VERY start of the HPO run.
# The 'wall_clock_time_to_best_so_far_s' is already relative to HPO start.
# We need a general cumulative time for plotting on x-axis.

for df_iter in [df_iter_a1, df_iter_a2, df_iter_a3]:
    if not df_iter.empty:
        # Ensure 'iteration_start_time_abs' is float
        df_iter['iteration_start_time_abs'] = pd.to_numeric(df_iter['iteration_start_time_abs'], errors='coerce')
        # Calculate cumulative time from the start of the first iteration for this run
        if not df_iter.empty:
            first_iter_start_time = df_iter['iteration_start_time_abs'].min()
            df_iter['cumulative_wall_clock_time_s'] = df_iter['iteration_end_time_abs'] - first_iter_start_time


# --- Placeholder for DataFrame dictionary for easier access ---
iter_dfs = {
    "Approach 1 (Static)": df_iter_a1,
    "Approach 2 (Dynamic)": df_iter_a2,
    "Approach 3 (MultiStage)": df_iter_a3
}
config_dfs = { # These are the main evaluated configs (GPU for A3)
    "Approach 1 (Static)": df_config_a1,
    "Approach 2 (Dynamic)": df_config_a2,
    "Approach 3 (MultiStage)": df_config_a3_gpu # Use GPU evaluated for A3 main comparison
}

def plot_fitness_vs_wallclock(iter_dfs_dict, output_filename="fitness_vs_wallclock.png"):
    plt.figure()
    for approach_name, df in iter_dfs_dict.items():
        if df.empty:
            print(f"Skipping '{approach_name}' for fitness vs wallclock: No data.")
            continue
        
        # Use the 'best_fitness_so_far' or 'best_fitness_so_far_gpu' column
        # And 'wall_clock_time_to_best_so_far_s' which is already relative to HPO start
        time_col = 'wall_clock_time_to_best_so_far_s'
        
        if 'best_fitness_so_far_gpu' in df.columns: # For Approach 3
            fitness_col = 'best_fitness_so_far_gpu'
        elif 'best_fitness_so_far' in df.columns: # For Approach 1 & 2
            fitness_col = 'best_fitness_so_far'
        else:
            print(f"Could not find suitable fitness column for {approach_name}")
            continue
            
        if time_col not in df.columns or fitness_col not in df.columns:
            print(f"Missing required columns '{time_col}' or '{fitness_col}' for {approach_name}")
            continue

        # Sort by time to ensure the line plot is correct
        df_sorted = df.sort_values(by=time_col)
        
        # Create a step plot: start at (0, initial_fitness_or_negative_inf)
        # and then each point where the best_fitness_so_far changes
        plot_times = [0]
        plot_fitness = [-np.inf] # Start from a very low fitness or the first recorded one

        last_fitness = -np.inf
        for _, row in df_sorted.iterrows():
            current_best_fitness = row[fitness_col]
            time_to_current_best = row[time_col]
            if current_best_fitness > last_fitness : #or plot_times[-1] != time_to_current_best:
                 # Add previous point to make it a step
                if plot_times[-1] < time_to_current_best :
                    plot_times.append(time_to_current_best)
                    plot_fitness.append(last_fitness)
                
                # Add new best point
                plot_times.append(time_to_current_best)
                plot_fitness.append(current_best_fitness)
                last_fitness = current_best_fitness
        
        # Ensure the line extends to the end of the experiment for that approach
        if 'cumulative_wall_clock_time_s' in df_sorted.columns and not df_sorted.empty:
            max_time_for_approach = df_sorted['cumulative_wall_clock_time_s'].max()
            if plot_times[-1] < max_time_for_approach:
                plot_times.append(max_time_for_approach)
                plot_fitness.append(last_fitness)


        plt.plot(plot_times, plot_fitness, marker='.', linestyle='-', label=approach_name, drawstyle='steps-post')

    plt.xlabel("Wall-Clock Time (seconds)")
    plt.ylabel("Best Fitness Score Achieved So Far")
    plt.title("HPO Convergence: Best Fitness vs. Wall-Clock Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved plot: {output_filename}")
    plt.show()

# Call the function
plot_fitness_vs_wallclock(iter_dfs)

def plot_fitness_vs_evals(iter_dfs_dict, config_dfs_dict, output_filename="fitness_vs_evals.png"):
    plt.figure()
    for approach_name, df_iter in iter_dfs_dict.items():
        df_config = config_dfs_dict.get(approach_name)
        if df_iter.empty or (df_config is None or df_config.empty) :
            print(f"Skipping '{approach_name}' for fitness vs evals: No iteration or config data.")
            continue

        # Determine fitness and config count columns based on approach
        # For A1/A2, 'num_configs_evaluated' is in iter_log, fitness is from config_log
        # For A3, 'num_configs_evaluated_gpu' is in iter_log, fitness is from config_log_A3_gpu

        # We need to iterate through the config log, sort by processing_start_time_abs,
        # and track cumulative evals and best fitness found up to that eval.
        
        eval_count_col_iter = None
        fitness_col_config = None
        time_col_config = None # To sort configs chronologically

        if approach_name == "Approach 3 (MultiStage)":
            fitness_col_config = 'gpu_fitness_score'
            time_col_config = 'gpu_processing_start_time_abs'
        else: # Approach 1 & 2
            fitness_col_config = 'fitness_score' # Assuming this is test_accuracy or similar
            time_col_config = 'processing_start_time_abs'

        if fitness_col_config not in df_config.columns or time_col_config not in df_config.columns:
            print(f"Missing required columns in config log for {approach_name}")
            continue
        
        # Sort configs by when they started processing
        df_config_sorted = df_config.sort_values(by=time_col_config).copy()
        df_config_sorted.dropna(subset=[fitness_col_config], inplace=True) # Drop if fitness is NaN

        if df_config_sorted.empty:
            print(f"No valid config data after sorting/dropping NaN for {approach_name}")
            continue
            
        cumulative_evals = []
        best_fitness_at_eval = []
        current_max_fitness = -np.inf

        for i, row in enumerate(df_config_sorted.iterrows()):
            idx, data = row
            cumulative_evals.append(i + 1)
            if data[fitness_col_config] > current_max_fitness:
                current_max_fitness = data[fitness_col_config]
            best_fitness_at_eval.append(current_max_fitness)
        
        if cumulative_evals: # Only plot if there's data
            plt.plot(cumulative_evals, best_fitness_at_eval, marker='.', linestyle='-', label=approach_name, markersize=3)

    plt.xlabel("Number of Configurations Fully Evaluated")
    plt.ylabel("Best Fitness Score Achieved So Far")
    plt.title("HPO Sample Efficiency: Best Fitness vs. Evaluations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved plot: {output_filename}")
    plt.show()

# Call the function
plot_fitness_vs_evals(iter_dfs, config_dfs)

def plot_iteration_durations_and_busy_times(iter_dfs_dict, output_filename_prefix="iteration_analysis"):
    if all(df.empty for df in iter_dfs_dict.values()):
        print("No iteration data to plot for durations/busy times.")
        return

    # Plot 1: Iteration Durations
    plt.figure()
    for approach_name, df in iter_dfs_dict.items():
        if df.empty or 'iteration_duration_s' not in df.columns:
            print(f"Skipping '{approach_name}' for iter duration: No data or missing column.")
            continue
        # Use 'macro_iteration' for A3, 'iteration' for A1/A2 as x-axis
        iter_col = 'macro_iteration' if 'macro_iteration' in df.columns else 'iteration'
        if iter_col not in df.columns: continue

        plt.plot(df[iter_col], df['iteration_duration_s'], marker='o', linestyle='-', label=f"{approach_name} Iter Duration")
    
    plt.xlabel("Iteration / Macro-Iteration Number")
    plt.ylabel("Duration (seconds)")
    plt.title("Duration per HPO Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_filename_prefix}_durations.png")
    print(f"Saved plot: {output_filename_prefix}_durations.png")
    plt.show()

    # Plot 2: GPU vs CPU Busy Time (Stacked Bar or Lines) per Iteration
    # This requires specific columns:
    # A1/A2: 'approx_gpu_busy_time_s', 'approx_cpu_busy_time_s'
    # A3: 'total_gpu_eval_time_s_iter', 'total_cpu_screen_time_s_iter' (ensure this is summed correctly)
    
    plt.figure(figsize=(14, 8)) # Wider for stacked/grouped bars
    num_approaches = len([df for df in iter_dfs_dict.values() if not df.empty])
    bar_width = 0.8 / num_approaches if num_approaches > 0 else 0.8
    
    plotted_approaches = 0
    for i, (approach_name, df) in enumerate(iter_dfs_dict.items()):
        if df.empty: continue

        iter_col = 'macro_iteration' if 'macro_iteration' in df.columns else 'iteration'
        if iter_col not in df.columns: continue
        
        gpu_busy_col, cpu_busy_col = None, None
        if 'approx_gpu_busy_time_s' in df.columns and 'approx_cpu_busy_time_s' in df.columns: # A1/A2
            gpu_busy_col = 'approx_gpu_busy_time_s'
            cpu_busy_col = 'approx_cpu_busy_time_s'
        elif 'total_gpu_eval_time_s_iter' in df.columns and 'total_cpu_screen_time_s_iter' in df.columns: # A3
            gpu_busy_col = 'total_gpu_eval_time_s_iter'
            cpu_busy_col = 'total_cpu_screen_time_s_iter' # Ensure this is correctly populated
        
        if not gpu_busy_col or not cpu_busy_col: 
            print(f"Skipping busy time plot for '{approach_name}': Missing busy time columns.")
            continue

        # Ensure busy times are numeric and handle NaN
        df[gpu_busy_col] = pd.to_numeric(df[gpu_busy_col], errors='coerce').fillna(0)
        df[cpu_busy_col] = pd.to_numeric(df[cpu_busy_col], errors='coerce').fillna(0)

        # Grouped Bar Chart
        x_indices = np.arange(len(df[iter_col])) # Iteration numbers as x
        
        plt.bar(x_indices - bar_width/2 + i*bar_width*0.9, df[gpu_busy_col], width=bar_width*0.4, label=f"{approach_name} GPU Busy", align='center')
        plt.bar(x_indices + bar_width/2 + i*bar_width*0.9, df[cpu_busy_col], width=bar_width*0.4, label=f"{approach_name} CPU Busy", align='center', alpha=0.7)
        
        # Or Line plot (simpler if bars are too cluttered)
        # plt.plot(df[iter_col], df[gpu_busy_col], marker='s', linestyle='--', label=f"{approach_name} GPU Busy")
        # plt.plot(df[iter_col], df[cpu_busy_col], marker='x', linestyle=':', label=f"{approach_name} CPU Busy (sum)")
        
        plotted_approaches +=1

    if plotted_approaches > 0:
        plt.xlabel("Iteration / Macro-Iteration Number")
        plt.ylabel("Approx. Busy Time (seconds)")
        plt.title("Approximate GPU vs CPU Busy Time per Iteration")
        plt.xticks(x_indices, df[iter_col]) # Set x-axis ticks to iteration numbers if using bars
        plt.legend(loc='upper left', bbox_to_anchor=(1,1)) # Move legend out
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.savefig(f"{output_filename_prefix}_busy_times.png")
        print(f"Saved plot: {output_filename_prefix}_busy_times.png")
        plt.show()
    else:
        print("No data plotted for busy times.")


# Call the function
plot_iteration_durations_and_busy_times(iter_dfs)

def plot_config_gantt_proxy(config_dfs_dict, hpo_start_time, output_filename="gantt_proxy.png"):
    plt.figure(figsize=(15, 8)) # Wider for Gantt
    
    y_ticks_labels = []
    y_tick_positions = []
    current_y = 0

    for approach_name, df_config in config_dfs_dict.items():
        if df_config.empty:
            print(f"Skipping Gantt for '{approach_name}': No config data.")
            continue

        time_start_col, duration_col, device_col = None, None, None
        # Determine column names based on approach (A1/A2 vs A3)
        if 'gpu_processing_start_time_abs' in df_config.columns: # A3 (GPU eval part)
            time_start_col = 'gpu_processing_start_time_abs'
            duration_col = 'gpu_processing_duration_s'
            device_col = 'gpu_device_used' # Or 'origin_cpu_id' if you want to track that
        elif 'processing_start_time_abs' in df_config.columns: # A1/A2
            time_start_col = 'processing_start_time_abs'
            duration_col = 'processing_duration_s'
            device_col = 'device_used'
        
        if not all([time_start_col, duration_col, device_col]):
            print(f"Skipping Gantt for '{approach_name}': Missing required time/device columns.")
            continue
        
        # Filter for valid data and convert to numeric
        df_plot = df_config[[time_start_col, duration_col, device_col]].copy()
        df_plot[time_start_col] = pd.to_numeric(df_plot[time_start_col], errors='coerce')
        df_plot[duration_col] = pd.to_numeric(df_plot[duration_col], errors='coerce')
        df_plot.dropna(inplace=True)

        if df_plot.empty:
            print(f"No valid data for Gantt plot for {approach_name}")
            continue

        # Create a unique worker ID for plotting (e.g., "A1_GPU", "A1_CPU_0", "A1_CPU_1")
        # For simplicity, we'll just use 'device_used' combined with approach name.
        # A more detailed Gantt would track individual CPU worker PIDs if logged.
        
        # Assign distinct y-levels for each device within each approach
        # Add approach name to y-axis labels to distinguish
        unique_devices = df_plot[device_col].unique()
        
        for device_name in unique_devices:
            device_specific_df = df_plot[df_plot[device_col] == device_name]
            label = f"{approach_name} - {device_name}"
            y_ticks_labels.append(label)
            y_tick_positions.append(current_y)
            
            # Plot bars: (start_time_relative, duration)
            # start_time_relative should be from the beginning of that HPO run.
            # The config log has absolute start times. We need a common reference.
            # For simplicity, let's assume the first 'processing_start_time_abs' in the
            # *entire loaded data* (across all approaches) can be a reference,
            # or pass the hpo_overall_start_time.
            # Here, using the passed hpo_start_time (which should be min across all runs for comparison)

            bars = []
            for _, row in device_specific_df.iterrows():
                start_rel = row[time_start_col] - hpo_start_time
                bars.append((start_rel, row[duration_col]))
            
            if bars:
                plt.broken_barh(bars, (current_y - 0.4, 0.8), facecolors=sns.color_palette()[len(y_tick_positions)%len(sns.color_palette())])
            current_y += 1
        current_y += 1 # Add a small gap between approaches
        

    if not y_tick_positions:
        print("No data to plot for Gantt.")
        plt.close()
        return

    plt.yticks(y_tick_positions, y_ticks_labels)
    plt.xlabel("Time since HPO Start (seconds)")
    plt.ylabel("Worker/Device")
    plt.title("Proxy Gantt Chart: Configuration Processing Activity")
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved plot: {output_filename}")
    plt.show()

# To call Gantt, you need a common start time for all approaches if plotting on same chart.
# Find the minimum start time from all iteration logs.
all_start_times = []
if not df_iter_a1.empty: all_start_times.append(df_iter_a1['iteration_start_time_abs'].min())
if not df_iter_a2.empty: all_start_times.append(df_iter_a2['iteration_start_time_abs'].min())
if not df_iter_a3.empty: all_start_times.append(df_iter_a3['iteration_start_time_abs'].min())
global_hpo_start_time_ref = min(all_start_times) if all_start_times else time.time() # Fallback

gantt_config_dfs = {
    "Approach 1 (Static)": df_config_a1,
    "Approach 2 (Dynamic)": df_config_a2,
    "Approach 3 (MultiStage) GPU Evals": df_config_a3_gpu,
    # Optionally add CPU screening for A3 to Gantt, would need to load and process cpu_screen_log_a3
}
plot_config_gantt_proxy(gantt_config_dfs, global_hpo_start_time_ref)