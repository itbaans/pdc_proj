import pandas as pd
import numpy as np
import os
import glob
import torch.multiprocessing as mp
import torch # To find log files

# --- Configuration ---
# Manually list the RUN_IDs for each approach from your filenames
# Or implement a discovery mechanism if you have many runs.
# Example:
# RUN_IDS_A1 = ["d2ef5b01", "another_a1_run"] # List of run_ids for Approach 1
# RUN_IDS_A2 = ["your_a2_id"]
# RUN_IDS_A3 = ["78eb9302"]

# For simplicity, let's assume one representative run_id per approach for now.
# *** REPLACE THESE WITH YOUR ACTUAL RUN IDS ***
REPRESENTATIVE_RUN_IDS = {
    "StaticSplit_GPUx3": "7c7346df",         # e.g., "d2ef5b01"
    "DynamicPull_GPUmicrobatch": "c6468262", # e.g., "abcdef12"
    "MultiStage_CPUScreen_GPUFull": "912a125b" # e.g., "78eb9302"
}

# --- Helper Functions to Load Data (Slightly more robust) ---
def load_log_data(pattern, dtype_spec=None):
    """Loads and concatenates all CSVs matching a pattern."""
    all_dfs = []
    found_files = glob.glob(pattern)
    if not found_files:
        print(f"Warning: No files found for pattern: {pattern}")
        return pd.DataFrame()
        
    print(f"Found files for {pattern}: {found_files}")
    for f_path in found_files:
        try:
            df = pd.read_csv(f_path, dtype=dtype_spec if dtype_spec else None)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {f_path}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)

def safe_to_numeric(series, default=0.0):
    return pd.to_numeric(series, errors='coerce').fillna(default) if isinstance(series, pd.Series) else default

# --- Main Data Processing and Benchmark Calculation ---
def calculate_benchmarks(run_ids_dict):
    benchmark_results = []

    for approach_id_name, run_id in run_ids_dict.items(): # Changed variable name for clarity
        if run_id.startswith("YOUR_"): 
             print(f"Skipping {approach_id_name}: Placeholder RUN_ID not replaced.")
             continue

        print(f"\nProcessing Approach: {approach_id_name}, Run ID: {run_id}")

        approach_short_name = "A1" if "StaticSplit" in approach_id_name else \
                              "A2" if "DynamicPull" in approach_id_name else \
                              "A3" if "MultiStage" in approach_id_name else "Unknown"

        # Define dtype specifications to handle potential variations and ensure numeric conversion
        iter_dtype_spec = {
            # Common fields
            'hpo_approach_id': str, 'run_id': str, 
            'iteration_start_time_abs': float, 'iteration_end_time_abs': float, 'iteration_duration_s': float,
            'best_fitness_this_iteration': float, 'best_fitness_so_far': float,
            'wall_clock_time_to_best_so_far_s': float,
            # A1 specific (or general if A2/A3 don't have the detailed ones)
            'iteration': 'Int64', 'num_configs_evaluated': 'Int64', 
            # A2 specific (might also be in a more detailed A1 log)
            'num_configs_gpu': 'Int64', 'num_configs_cpu': 'Int64', 
            'approx_gpu_busy_time_s': float, 'approx_cpu_busy_time_s': float,
            # A3 specific
            'macro_iteration': 'Int64', 'num_configs_screened_cpu_total': 'Int64', 
            'num_configs_promoted_to_gpu': 'Int64', 'num_configs_evaluated_gpu': 'Int64',
            'best_fitness_this_iteration_gpu': float, 'best_fitness_so_far_gpu': float,
            'total_cpu_screen_time_s_iter': float, 'total_gpu_eval_time_s_iter': float
        }
        
        config_dtype_spec = { # Assuming a common superset of columns for config logs
            'device_used': str, 'gpu_device_used': str, # Handle both possible names
            'processing_duration_s': float, 'gpu_processing_duration_s': float
        }


        iter_log_pattern = f"hpo_iteration_log_{approach_short_name}_{run_id}.csv"
        config_log_pattern = f"hpo_config_log_{approach_short_name}_{run_id}.csv"
        cpu_screen_log_pattern = f"hpo_cpuscreen_log_{approach_short_name}_{run_id}.csv"

        df_iter = load_log_data(iter_log_pattern, dtype_spec=iter_dtype_spec)
        df_config = load_log_data(config_log_pattern, dtype_spec=config_dtype_spec) 
        df_cpu_screen = pd.DataFrame() # Only for A3
        if approach_short_name == "A3":
            df_cpu_screen = load_log_data(cpu_screen_log_pattern) # Dtype spec can be added if needed

        if df_iter.empty:
            print(f"  No iteration data found for {approach_id_name} run {run_id}. Skipping.")
            continue

        # --- Basic Run Info ---
        df_iter['iteration_start_time_abs'] = safe_to_numeric(df_iter['iteration_start_time_abs'])
        df_iter['iteration_end_time_abs'] = safe_to_numeric(df_iter['iteration_end_time_abs'])
        total_wall_clock_time_s = (df_iter['iteration_end_time_abs'].max() - df_iter['iteration_start_time_abs'].min())
        
        iter_col_name = 'macro_iteration' if 'macro_iteration' in df_iter.columns else 'iteration'
        num_iterations_completed = safe_to_numeric(df_iter[iter_col_name]).max() if iter_col_name in df_iter.columns else 0
        
        # --- Best Fitness and Time to Best ---
        best_fitness_col = 'best_fitness_so_far_gpu' if 'best_fitness_so_far_gpu' in df_iter.columns else \
                           'best_fitness_so_far' if 'best_fitness_so_far' in df_iter.columns else None
        
        final_best_fitness = -float('inf')
        best_fitness_achieved_time_s = 0.0

        if best_fitness_col and best_fitness_col in df_iter.columns:
            df_iter[best_fitness_col] = safe_to_numeric(df_iter[best_fitness_col], default=-float('inf'))
            final_best_fitness = df_iter[best_fitness_col].max()
            time_to_best_col = 'wall_clock_time_to_best_so_far_s'
            if time_to_best_col in df_iter.columns and not df_iter[df_iter[best_fitness_col] == final_best_fitness].empty:
                # Get time for the first occurrence of the final best fitness
                best_fitness_achieved_time_s = safe_to_numeric(df_iter.loc[df_iter[df_iter[best_fitness_col] == final_best_fitness][time_to_best_col].idxmin(), time_to_best_col])


        # --- Evaluation Counts & Throughput & Utilization ---
        total_primary_evals = 0 # GPU evals for A3, total for A1/A2 (or GPU part if distinguishable)
        gpu_eval_count_total = 0
        cpu_eval_count_total = 0 # For A1/A2 generic CPU workers, A3 specific screening
        
        total_gpu_busy_s_from_iter = 0
        total_cpu_busy_s_from_iter = 0 # For A1/A2 generic, for A3 screening

        if approach_short_name == "A1" or approach_short_name == "A2":
            # Get counts from config log as it's more reliable for device split
            if not df_config.empty:
                device_col_cfg = 'device_used' if 'device_used' in df_config.columns else 'gpu_device_used' # A3 uses gpu_device_used
                if device_col_cfg in df_config.columns:
                    gpu_eval_count_total = len(df_config[df_config[device_col_cfg] == 'cuda'])
                    cpu_eval_count_total = len(df_config[df_config[device_col_cfg] == 'cpu'])
                total_primary_evals = gpu_eval_count_total + cpu_eval_count_total # All evals are primary for A1/A2
            else: # Fallback to iteration log if config log is empty
                if 'num_configs_evaluated' in df_iter.columns:
                    total_primary_evals = safe_to_numeric(df_iter['num_configs_evaluated']).sum()
                if 'num_configs_gpu' in df_iter.columns: # A2 might have this
                    gpu_eval_count_total = safe_to_numeric(df_iter['num_configs_gpu']).sum()
                if 'num_configs_cpu' in df_iter.columns: # A2 might have this
                    cpu_eval_count_total = safe_to_numeric(df_iter['num_configs_cpu']).sum()
            
            # Get busy times from iteration log for A1/A2
            if 'approx_gpu_busy_time_s' in df_iter.columns:
                total_gpu_busy_s_from_iter = safe_to_numeric(df_iter['approx_gpu_busy_time_s']).sum()
            if 'approx_cpu_busy_time_s' in df_iter.columns:
                total_cpu_busy_s_from_iter = safe_to_numeric(df_iter['approx_cpu_busy_time_s']).sum()

        elif approach_short_name == "A3":
            gpu_eval_count_total = safe_to_numeric(df_iter.get('num_configs_evaluated_gpu', 0)).sum()
            cpu_eval_count_total = safe_to_numeric(df_iter.get('num_configs_screened_cpu_total', 0)).sum() # This is screen_evals
            total_primary_evals = gpu_eval_count_total # Primary evals are GPU evals for A3
            
            total_gpu_busy_s_from_iter = safe_to_numeric(df_iter.get('total_gpu_eval_time_s_iter', 0)).sum()
            total_cpu_busy_s_from_iter = safe_to_numeric(df_iter.get('total_cpu_screen_time_s_iter', 0)).sum() # This is screen busy time

        throughput = total_primary_evals / total_wall_clock_time_s if total_wall_clock_time_s > 0 else 0
        
        # Approximate number of CPU worker processes from your HPO script setup
        # This is a simplification; ideally, log this or get from config.
        num_cpu_worker_procs_approx = 3 # Default assumption
        if approach_short_name == "A1" or approach_short_name == "A2":
            # Based on mp.cpu_count() - (1 if cuda else 0), assume at least 1.
            num_cpu_worker_procs_approx = max(1, mp.cpu_count() - (1 if gpu_eval_count_total > 0 else 0))
        elif approach_short_name == "A3":
             num_cpu_worker_procs_approx = max(1, mp.cpu_count() - 1 if torch.cuda.is_available() else mp.cpu_count())


        gpu_utilization_pct = (total_gpu_busy_s_from_iter / total_wall_clock_time_s) * 100 if total_wall_clock_time_s > 0 else 0
        
        cpu_utilization_pct = (total_cpu_busy_s_from_iter / (total_wall_clock_time_s * num_cpu_worker_procs_approx)) * 100 \
            if total_wall_clock_time_s > 0 and num_cpu_worker_procs_approx > 0 else 0

        configs_screened_per_gpu_eval_a3 = cpu_eval_count_total / gpu_eval_count_total \
            if approach_short_name == "A3" and gpu_eval_count_total > 0 else float('nan')

        benchmark_results.append({
            "Approach": approach_id_name, # Use the full name from dict keys
            "Run ID": run_id,
            "Total Wall-Clock Time (s)": round(total_wall_clock_time_s, 2),
            "Iterations Completed": int(num_iterations_completed),
            "Total Primary Evals": int(total_primary_evals), # GPU for A3, All for A1/A2
            "Total GPU Evals (Actual)": int(gpu_eval_count_total),
            "Total CPU Evals/Screens": int(cpu_eval_count_total),
            "Best Fitness Score": round(final_best_fitness, 4),
            "Time to Best Fitness (s)": round(best_fitness_achieved_time_s, 2),
            "Throughput (Primary Evals/s)": round(throughput, 2),
            "GPU Utilization (%)": round(gpu_utilization_pct, 2),
            "Avg CPU Worker Util (%)": round(cpu_utilization_pct, 2),
            "CPU Screens per GPU Eval (A3)": round(configs_screened_per_gpu_eval_a3,2) if approach_short_name == "A3" else "N/A"
        })

    return pd.DataFrame(benchmark_results)

if __name__ == "__main__":
    # --- Calculate Benchmarks ---
    benchmarks_df = calculate_benchmarks(REPRESENTATIVE_RUN_IDS)

    if not benchmarks_df.empty:
        print("\n\n--- Benchmark Results Table ---")
        # Transpose for better readability if few approaches, many metrics
        # Or keep as is if many approaches.
        print(benchmarks_df.set_index("Approach").T.to_string()) 
        
        # Save to CSV
        benchmarks_df.to_csv("hpo_benchmark_summary.csv", index=False)
        print("\nBenchmark summary saved to hpo_benchmark_summary.csv")
    else:
        print("No benchmark data was generated. Check RUN_IDs and file paths.")

    # --- Plotting (using the functions from previous response) ---
    # You would call your plotting functions here, passing the loaded DataFrames
    # Example:
    # print("\nGenerating plots...")
    # Make sure iter_dfs and config_dfs are populated correctly for plotting functions
    
    iter_dfs_plot = {}
    config_dfs_plot = {} # For main evaluated configs
    config_dfs_cpu_screen_plot = {} # For A3 CPU screens

    for approach_id_prefix, run_id in REPRESENTATIVE_RUN_IDS.items():
        if run_id == f"YOUR_{approach_id_prefix.split('_')[0]}_RUN_ID": continue # Skip placeholders
        
        approach_short_name = "A1" if "StaticSplit" in approach_id_prefix else \
                              "A2" if "DynamicPull" in approach_id_prefix else \
                              "A3" if "MultiStage" in approach_id_prefix else "Unknown"
        
        iter_df = load_log_data(f"hpo_iteration_log_{approach_short_name}_{run_id}.csv")
        config_df = load_log_data(f"hpo_config_log_{approach_short_name}_{run_id}.csv")
        
        # Preprocess for plotting: add cumulative wall clock
        if not iter_df.empty:
            iter_df['iteration_start_time_abs'] = safe_to_numeric(iter_df['iteration_start_time_abs'])
            iter_df['iteration_end_time_abs'] = safe_to_numeric(iter_df['iteration_end_time_abs'])
            first_iter_start_time = iter_df['iteration_start_time_abs'].min()
            iter_df['cumulative_wall_clock_time_s'] = iter_df['iteration_end_time_abs'] - first_iter_start_time
        
        iter_dfs_plot[approach_id_prefix] = iter_df
        config_dfs_plot[approach_id_prefix] = config_df
        
        if approach_short_name == "A3":
            df_cpu_s_temp = load_log_data(f"hpo_cpuscreen_log_{approach_short_name}_{run_id}.csv")
            config_dfs_cpu_screen_plot[approach_id_prefix] = df_cpu_s_temp


    # --- Call plotting functions (ensure these are defined in this script or imported) ---
    # Example plotting calls (assuming plotting functions are defined as in previous response):
    
    # from plot_hpo_results_functions import plot_fitness_vs_wallclock, plot_fitness_vs_evals # If in separate file
    # For simplicity, let's assume they are in this file. You'd need to copy them here.
    
    # plot_fitness_vs_wallclock(iter_dfs_plot, output_filename="benchmark_fitness_vs_wallclock.png")
    # plot_fitness_vs_evals(iter_dfs_plot, config_dfs_plot, output_filename="benchmark_fitness_vs_evals.png")
    # plot_iteration_durations_and_busy_times(iter_dfs_plot, output_filename_prefix="benchmark_iteration_analysis")

    # For Gantt, find global start time
    # all_start_times_plot = [df['iteration_start_time_abs'].min() for df in iter_dfs_plot.values() if not df.empty and 'iteration_start_time_abs' in df.columns and df['iteration_start_time_abs'].notna().any()]
    # global_hpo_start_ref_plot = min(all_start_times_plot) if all_start_times_plot else time.time()
    # plot_config_gantt_proxy(config_dfs_plot, global_hpo_start_ref_plot, output_filename="benchmark_gantt_proxy.png")

    print("\nAnalysis script finished. Check for generated table and plots.")