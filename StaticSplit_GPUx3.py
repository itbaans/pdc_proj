import time
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
import csv 
import os  
import uuid 
from queue import Empty # For result queue timeout

# --- Experiment Setup & Logging Constants ---
HPO_APPROACH_ID = "StaticSplit_GPUx3" 
RUN_ID = str(uuid.uuid4())[:8]       

CONFIG_LOG_FILENAME = f"hpo_config_log_A1_{RUN_ID}.csv"
ITERATION_LOG_FILENAME = f"hpo_iteration_log_A1_{RUN_ID}.csv"

# --- HPO Budget ---
MAX_HPO_WALL_CLOCK_SECONDS = 900  # e.g., 15 minutes for testing
# MAX_HPO_WALL_CLOCK_SECONDS = 60  # For very quick testing
MAX_ITERATIONS_SAFETY_CAP = 1000 # Prevent infinite loops

# --- HPO Configuration Parameters for Approach 1 ---
# NUM_ITERATIONS is now controlled by time/safety cap
INITIAL_POOL_SIZE = 20 
TOP_KEEP_RATIO = 0.3   
MUTATION_RATE = 0.25
GPU_SET_MULTIPLIER = 3 
NUM_EPOCHS_PER_EVAL = 3 # Reduced for faster iterations with time limit

# --- Dataset Configuration ---
TOTAL_SAMPLES_FOR_HPO_EXPERIMENT = 50000 # Reduced for faster iterations
HPO_INTERNAL_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
FLATTENED_DATA_SIZE = 54 
NUM_CLASSES = 7          

def prepare_shared_data(): # Unchanged from your provided version
    print(f"Fetching Covertype. Will use a total of ~{TOTAL_SAMPLES_FOR_HPO_EXPERIMENT} samples for this HPO experiment.")
    covtype = fetch_covtype()
    X_full = covtype.data.astype(np.float32)
    y_full = covtype.target.astype(np.int64)
    y_full = y_full - 1 

    if TOTAL_SAMPLES_FOR_HPO_EXPERIMENT < len(X_full):
        X_experiment_subset, _, y_experiment_subset, _ = train_test_split(
            X_full, y_full, train_size=TOTAL_SAMPLES_FOR_HPO_EXPERIMENT,
            random_state=RANDOM_STATE, stratify=y_full)
        print(f"Using a subset of {len(X_experiment_subset)} samples for the HPO experiment.")
    else:
        X_experiment_subset, y_experiment_subset = X_full, y_full
        print(f"Using the full dataset of {len(X_experiment_subset)} samples.")

    X_hpo_train, X_hpo_test, y_hpo_train, y_hpo_test = train_test_split(
        X_experiment_subset, y_experiment_subset, test_size=HPO_INTERNAL_TEST_SPLIT_RATIO,
        random_state=RANDOM_STATE, stratify=y_experiment_subset)

    scaler = StandardScaler()
    X_hpo_train_scaled = scaler.fit_transform(X_hpo_train)
    X_hpo_test_scaled = scaler.transform(X_hpo_test)

    hpo_train_data_shared = torch.from_numpy(X_hpo_train_scaled).share_memory_()
    hpo_train_labels_shared = torch.from_numpy(y_hpo_train).share_memory_()
    hpo_test_data_shared = torch.from_numpy(X_hpo_test_scaled).share_memory_()
    hpo_test_labels_shared = torch.from_numpy(y_hpo_test).share_memory_()

    print(f"HPO Train data shape (shared for workers): {hpo_train_data_shared.shape}")
    print(f"HPO Test data shape (shared for workers): {hpo_test_data_shared.shape}")
    
    return ((hpo_train_data_shared, hpo_train_labels_shared),
            (hpo_test_data_shared, hpo_test_labels_shared))

class DynamicNN(torch.nn.Module): # Unchanged
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.config = config; layers = OrderedDict(); current_dim = input_size
        for i in range(config['num_layers']):
            layers[f'fc{i+1}'] = torch.nn.Linear(current_dim, config['hidden_size'])
            layers[f'relu{i+1}'] = torch.nn.ReLU()
            dr = config.get('dropout_rate',0.0)
            if dr > 0: layers[f'dropout{i+1}'] = torch.nn.Dropout(dr)
            current_dim = config['hidden_size']
        layers['fc_out'] = torch.nn.Linear(current_dim, output_size)
        self.model_layers = torch.nn.Sequential(layers)
    def forward(self, x): return self.model_layers(x.view(-1, FLATTENED_DATA_SIZE)) # Ensure view is present

def worker(worker_id, task_queue, result_queue, shared_train, shared_test, device_str): # Unchanged from your logged A1
    device = torch.device(device_str)
    train_dataset = TensorDataset(*shared_train)
    test_dataset = TensorDataset(*shared_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    while True:
        hpo_config_batch_with_ids = task_queue.get() 
        if hpo_config_batch_with_ids is None: 
            result_queue.put(None); break
        
        worker_batch_results = [] 
        for config_original_idx, config_dict in hpo_config_batch_with_ids:
            processing_start_time_abs = time.time() 
            config_processing_start_mono = time.monotonic() 
            test_accuracy = 0.0
            try:
                model = DynamicNN(FLATTENED_DATA_SIZE, NUM_CLASSES, config_dict).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['lr'], 
                                             weight_decay=config_dict.get('weight_decay', 0.0))
                criterion = torch.nn.CrossEntropyLoss()
                model.train()
                for epoch in range(NUM_EPOCHS_PER_EVAL):
                    for data, targets in train_loader:
                        data, targets = data.to(device), targets.to(device)
                        optimizer.zero_grad()
                        outputs = model(data) # .view(-1, FLATTENED_DATA_SIZE) is in model.forward
                        loss = criterion(outputs, targets); loss.backward(); optimizer.step()
                model.eval(); correct, total = 0, 0
                with torch.no_grad():
                    for data, targets in test_loader:
                        data, targets = data.to(device), targets.to(device)
                        outputs = model(data); _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0); correct += (predicted == targets).sum().item()
                test_accuracy = correct / total if total > 0 else 0.0
                del model, optimizer, criterion
                if device_str == 'cuda': torch.cuda.empty_cache()
            except Exception as e:
                print(f"!!! ERROR Worker {worker_id} ({device_str}) cfg_idx {config_original_idx}: {type(e).__name__} {e}")
                test_accuracy = -1.0 
            config_processing_end_mono = time.monotonic()
            duration_s = config_processing_end_mono - config_processing_start_mono
            worker_batch_results.append((
                (config_original_idx, config_dict), test_accuracy, 
                processing_start_time_abs, duration_s, device_str ))
        result_queue.put(worker_batch_results)

def generate_initial_configs(): # Unchanged from your logged A1
    configs = []
    for _ in range(INITIAL_POOL_SIZE):
        configs.append({
            'lr': 10**np.random.uniform(-4.0, -1.5), 'num_layers': random.randint(1, 3),
            'hidden_size': int(2**np.random.uniform(5, 8)), 'dropout_rate': np.random.uniform(0.0, 0.5),
            'weight_decay': 10**np.random.uniform(-7, -4)})
    return configs

def mutate_config(config_tuple_to_mutate): # Unchanged from your logged A1
    if isinstance(config_tuple_to_mutate, tuple) and isinstance(config_tuple_to_mutate[0], tuple):
        config = config_tuple_to_mutate[0][1]
    elif isinstance(config_tuple_to_mutate, dict): config = config_tuple_to_mutate
    else: config = config_tuple_to_mutate[0]
    mutated_config = config.copy(); param = random.choice(list(mutated_config.keys()))
    rate = MUTATION_RATE
    if param == 'lr': mutated_config['lr'] = np.clip(mutated_config['lr'] * (1 + np.random.uniform(-rate,rate)*random.choice([-1,1])), 1e-6, 1e-1)
    elif param == 'num_layers': mutated_config['num_layers'] = np.clip(mutated_config['num_layers'] + random.choice([-1,0,1]), 1, 4)
    elif param == 'hidden_size':
        hs = mutated_config['hidden_size']; hs = int(hs*(1+np.random.uniform(-rate,rate)*random.choice([-1,1])))
        mutated_config['hidden_size'] = np.clip(max(16, (hs//8)*8), 16, 512)
    elif param == 'dropout_rate': mutated_config['dropout_rate'] = np.clip(config.get('dropout_rate',0) + np.random.uniform(-rate/2,rate/2), 0.0, 0.7)
    elif param == 'weight_decay': mutated_config['weight_decay'] = np.clip(config.get('weight_decay',0) * (1 + np.random.uniform(-rate,rate)*random.choice([-1,1])), 1e-8, 1e-2)
    return mutated_config

def split_configurations_static(configs_with_indices, num_cpu_workers, num_gpu_workers, gpu_multiplier): # Unchanged from your logged A1
    total_configs = len(configs_with_indices)
    if total_configs == 0: return [], []
    gpu_batch = []; cpu_batches = [[] for _ in range(num_cpu_workers if num_cpu_workers > 0 else 1)] # Ensure cpu_batches is list even if num_cpu_workers is 0
    
    # If no CPU workers but GPU worker exists, all go to GPU
    if num_cpu_workers == 0 and num_gpu_workers > 0:
        gpu_batch = configs_with_indices
        return gpu_batch, []
    # If no GPU workers but CPU workers exist, all go to CPUs
    if num_gpu_workers == 0 and num_cpu_workers > 0:
        for i, cfg_item in enumerate(configs_with_indices):
            cpu_batches[i % num_cpu_workers].append(cfg_item)
        return [], [b for b in cpu_batches if b]
    # If both exist
    if num_gpu_workers > 0 and num_cpu_workers > 0:
        total_units = (gpu_multiplier * num_gpu_workers) + num_cpu_workers
        unit_batch_size = max(1, total_configs // total_units if total_units > 0 else total_configs)
        gpu_batch_target_size = unit_batch_size * gpu_multiplier * num_gpu_workers
        gpu_batch = configs_with_indices[:gpu_batch_target_size]
        remaining_configs = configs_with_indices[gpu_batch_target_size:]
        for i, cfg_item in enumerate(remaining_configs):
            cpu_batches[i % num_cpu_workers].append(cfg_item)
        return gpu_batch, [b for b in cpu_batches if b]
    # If no workers at all (should not happen if main logic prevents it)
    if num_gpu_workers == 0 and num_cpu_workers == 0:
        return [], []
        
    return gpu_batch, [b for b in cpu_batches if b]


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # --- Initialize Log Files ---
    config_log_file_exists = os.path.exists(CONFIG_LOG_FILENAME)
    config_log_f = open(CONFIG_LOG_FILENAME, 'a', newline='')
    config_csv_writer = csv.writer(config_log_f)
    if not config_log_file_exists:
        config_csv_writer.writerow([
            "hpo_approach_id", "run_id", "iteration", "config_original_idx",
            "lr", "num_layers", "hidden_size", "dropout_rate", "weight_decay",
            "device_used", "processing_start_time_abs", "processing_duration_s",
            "test_accuracy", "fitness_score", "train_accuracy" # Added train_accuracy
        ])

    iteration_log_file_exists = os.path.exists(ITERATION_LOG_FILENAME)
    iteration_log_f = open(ITERATION_LOG_FILENAME, 'a', newline='')
    iteration_csv_writer = csv.writer(iteration_log_f)
    if not iteration_log_file_exists:
        iteration_csv_writer.writerow([
            "hpo_approach_id", "run_id", "iteration",
            "iteration_start_time_abs", "iteration_end_time_abs", "iteration_duration_s",
            "num_configs_evaluated_total", "num_configs_gpu", "num_configs_cpu", # More detailed counts
            "best_fitness_this_iteration", "best_fitness_so_far", 
            "wall_clock_time_to_best_so_far_s",
            "approx_gpu_busy_time_s", "approx_cpu_busy_time_s" # Sum of durations for each device type
        ])
    # --- End Log File Init ---

    hpo_overall_start_time_abs = time.time()
    print(f"Starting HPO Approach: {HPO_APPROACH_ID}, Run ID: {RUN_ID}")
    print(f"Target HPO Duration: {MAX_HPO_WALL_CLOCK_SECONDS} seconds.")
    print(f"Config log: {CONFIG_LOG_FILENAME}\nIteration log: {ITERATION_LOG_FILENAME}")

    shared_hpo_train_data, shared_hpo_test_data = prepare_shared_data()
    print("Shared data prepared for HPO.")

    current_configs_pool = generate_initial_configs()
    overall_best_fitness = -float('inf')
    time_of_overall_best_fitness_abs = hpo_overall_start_time_abs
    
    hpo_total_gpu_busy_s = 0
    hpo_total_cpu_busy_s = 0
    iteration_actual_count = 0

    # --- Main HPO Loop (Time-Based) ---
    while True:
        current_hpo_duration_s = time.time() - hpo_overall_start_time_abs
        if current_hpo_duration_s >= MAX_HPO_WALL_CLOCK_SECONDS:
            print(f"Main: Reached max HPO duration of {MAX_HPO_WALL_CLOCK_SECONDS:.0f}s. Stopping.")
            break
        if iteration_actual_count >= MAX_ITERATIONS_SAFETY_CAP:
            print(f"Main: Reached safety cap of {MAX_ITERATIONS_SAFETY_CAP} iterations. Stopping.")
            break

        iteration_actual_count += 1
        iter_start_abs_timestamp = time.time()
        iter_start_mono_timestamp = time.monotonic()

        print(f"\n=== Iteration {iteration_actual_count} ({HPO_APPROACH_ID}) - Time: {current_hpo_duration_s:.0f}s / {MAX_HPO_WALL_CLOCK_SECONDS}s ===")

        if not current_configs_pool:
            print("Config pool empty, generating new initial pool for safety.")
            current_configs_pool = generate_initial_configs()
        
        configs_to_eval_this_iter_with_ids = [(idx, cfg) for idx, cfg in enumerate(current_configs_pool)]
        print(f"Evaluating {len(configs_to_eval_this_iter_with_ids)} configurations.")

        task_q = mp.Queue()
        result_q = mp.Queue()
        active_worker_procs = []
        num_gpu_procs = 0; worker_id_gen = 0

        if torch.cuda.is_available():
            p_gpu = mp.Process(target=worker, args=(worker_id_gen, task_q, result_q, 
                                                  shared_hpo_train_data, shared_hpo_test_data, 'cuda'))
            active_worker_procs.append(p_gpu); num_gpu_procs = 1; worker_id_gen+=1; p_gpu.start()
            print("CUDA available, GPU worker started.")
        
        # Determine number of CPU workers
        # If GPU exists, num_cpu = total_cores - 1 (for GPU process). If no GPU, num_cpu = total_cores.
        # However, for A1, if no GPU, the "GPU batch" is processed by the first CPU.
        num_cpu_procs_to_start = 0
        if num_gpu_procs > 0: # GPU exists
            num_cpu_procs_to_start = max(0, mp.cpu_count() - 1) # Leave one for main/GPU
        else: # No dedicated GPU, first CPU worker will handle "GPU batch"
            num_cpu_procs_to_start = max(0, mp.cpu_count() - 1) # One CPU for "GPU", others for "CPU tasks"
                                                                # If only 1 core, this will be 0.

        print(f"Creating {num_cpu_procs_to_start} dedicated CPU workers.")
        for _ in range(num_cpu_procs_to_start):
            p_cpu = mp.Process(target=worker, args=(worker_id_gen, task_q, result_q, 
                                                  shared_hpo_train_data, shared_hpo_test_data, 'cpu'))
            active_worker_procs.append(p_cpu); worker_id_gen+=1; p_cpu.start()
        
        # If no dedicated GPU and no dedicated CPU workers (e.g. 1 core total), the "GPU" task runs on main CPU process context
        # This is complex for A1. Let's simplify: if no GPU, the first worker in active_worker_procs
        # (which would be a CPU worker if mp.cpu_count() >=1) takes the GPU_BATCH.
        # If mp.cpu_count() is 1 and no GPU, only one CPU worker is started.
        if not torch.cuda.is_available() and not active_worker_procs and mp.cpu_count() >=1:
            print("No dedicated GPU. Simulating GPU tasks on a CPU worker.")
            # This CPU worker will get the "gpu_batch_tasks"
            p_sim_gpu = mp.Process(target=worker, args=(worker_id_gen, task_q, result_q, 
                                                  shared_hpo_train_data, shared_hpo_test_data, 'cpu'))
            active_worker_procs.append(p_sim_gpu); worker_id_gen+=1; p_sim_gpu.start()
            # num_gpu_procs remains 0, but split_config will give work to "gpu_batch" if num_gpu_procs=0 and there is a worker

        num_active_procs_started = len(active_worker_procs)
        if num_active_procs_started == 0: print("CRITICAL: No workers started!"); break

        # Static split
        # The num_cpu_workers arg for split_configurations_static is the number of workers that will receive CPU batches.
        # If no GPU, the first worker in active_worker_procs handles gpu_batch, so effective num_cpu_workers for cpu_batches is reduced.
        
        # For split_configurations_static:
        # num_gpu_workers_for_split: 1 if torch.cuda.is_available() else 0 (or 1 if simulating on CPU)
        # num_cpu_workers_for_split: len(active_worker_procs) - num_gpu_workers_for_split
        
        # Simplified: split assumes a "GPU worker" exists if num_gpu_procs=1 (real or simulated)
        # and remaining active_worker_procs are "CPU workers".
        
        _num_gpu_for_split = 1 if torch.cuda.is_available() or (not torch.cuda.is_available() and num_active_procs_started > num_cpu_procs_to_start) else 0
        _num_cpu_for_split = num_active_procs_started - _num_gpu_for_split
        
        gpu_batch_tasks, cpu_batches_tasks = split_configurations_static(
            configs_to_eval_this_iter_with_ids, _num_cpu_for_split, _num_gpu_for_split, GPU_SET_MULTIPLIER
        )
        
        # Dispatch tasks
        dispatched_to_gpu_worker = False
        if gpu_batch_tasks:
            task_q.put(gpu_batch_tasks)
            dispatched_to_gpu_worker = True
            print(f"Sent {len(gpu_batch_tasks)} configs to GPU-designated worker.")
        
        # If no dedicated GPU worker was started (num_gpu_procs=0) but there's a gpu_batch,
        # it means the first CPU worker is handling it.
        # The cpu_batches_tasks should be for the *other* CPU workers.
        
        cpu_worker_task_idx = 0
        for cpu_batch in cpu_batches_tasks:
            if cpu_batch:
                # Ensure we don't try to send to a worker that doesn't exist or is already the "GPU" worker
                # This dispatch logic for A1 is tricky when simulating GPU on CPU.
                # Simplest: gpu_batch goes to worker 0 if active_worker_procs[0] exists.
                # cpu_batches go to worker 1, 2, ...
                # If active_worker_procs[0] is the GPU worker:
                target_worker_queue_idx = (1 if dispatched_to_gpu_worker and torch.cuda.is_available() else 0) + cpu_worker_task_idx
                if target_worker_queue_idx < num_active_procs_started : # Redundant check as task_queue is shared
                    task_q.put(cpu_batch)
                    print(f"Sent {len(cpu_batch)} configs to a CPU worker.")
                    cpu_worker_task_idx +=1
                else:
                    print(f"Warning: No CPU worker available for a CPU batch of size {len(cpu_batch)}. Adding to GPU queue if possible or dropping.")
                    if gpu_batch_tasks : task_q.put(cpu_batch) # Add to general queue, GPU worker might pick it up
                    elif active_worker_procs : task_q.put(cpu_batch) # Or first available if no GPU batch

        for _ in range(num_active_procs_started): task_q.put(None)

        # --- Collect Results ---
        iter_results_list = []; iter_finished_procs = 0
        iter_gpu_busy_s = 0; iter_cpu_busy_s = 0
        iter_gpu_cfg_count = 0; iter_cpu_cfg_count = 0

        while iter_finished_procs < num_active_procs_started:
            try:
                res_batch_from_worker = result_q.get(timeout=300)
                if res_batch_from_worker is None: iter_finished_procs += 1
                else:
                    iter_results_list.extend(res_batch_from_worker)
                    for _res_item in res_batch_from_worker:
                        _dur = _res_item[3]; _dev = _res_item[4]
                        if _dev == 'cuda': iter_gpu_busy_s += _dur; iter_gpu_cfg_count +=1
                        else: iter_cpu_busy_s += _dur; iter_cpu_cfg_count+=1
            except Empty:
                print(f"Timeout A1 results. Finished: {iter_finished_procs}/{num_active_procs_started}"); break # Check aliveness if breaking

        iter_end_abs_timestamp = time.time(); iter_end_mono_timestamp = time.monotonic()
        iter_duration_s_val = iter_end_mono_timestamp - iter_start_mono_timestamp
        hpo_total_gpu_busy_s += iter_gpu_busy_s; hpo_total_cpu_busy_s += iter_cpu_busy_s

        iter_best_fitness_val = -float('inf')
        ga_pool_this_iter = []

        for res_item_tuple in iter_results_list:
            (cfg_tuple, t_acc, proc_abs, dur, dev) = res_item_tuple
            cfg_orig_idx, cfg_d = cfg_tuple
            fitness = t_acc if t_acc >= 0 else -float('inf') # Simple fitness for A1

            config_csv_writer.writerow([
                HPO_APPROACH_ID, RUN_ID, iteration_actual_count, cfg_orig_idx,
                f"{cfg_d['lr']:.6e}", cfg_d['num_layers'], cfg_d['hidden_size'],
                f"{cfg_d.get('dropout_rate',0):.4f}", f"{cfg_d.get('weight_decay',0):.6e}",
                dev, f"{proc_abs:.3f}", f"{dur:.3f}",
                f"{t_acc:.4f}", f"{fitness:.4f}", "N/A" # No train_acc logged by A1 worker
            ])
            if fitness > iter_best_fitness_val: iter_best_fitness_val = fitness
            if fitness > overall_best_fitness:
                overall_best_fitness = fitness; time_of_overall_best_fitness_abs = time.time()
            if t_acc >=0: ga_pool_this_iter.append( (cfg_tuple, fitness) )
        config_log_f.flush()

        iteration_csv_writer.writerow([
            HPO_APPROACH_ID, RUN_ID, iteration_actual_count,
            f"{iter_start_abs_timestamp:.3f}", f"{iter_end_abs_timestamp:.3f}", f"{iter_duration_s_val:.3f}",
            len(iter_results_list), iter_gpu_cfg_count, iter_cpu_cfg_count,
            f"{iter_best_fitness_val:.4f}", f"{overall_best_fitness:.4f}",
            f"{(time_of_overall_best_fitness_abs - hpo_overall_start_time_abs):.3f}",
            f"{iter_gpu_busy_s:.3f}", f"{iter_cpu_busy_s:.3f}"
        ])
        iteration_log_f.flush()

        # --- GA ---
        if not ga_pool_this_iter: current_configs_pool = generate_initial_configs()
        else:
            ga_pool_this_iter.sort(key=lambda x: x[1], reverse=True)
            keep_n = int(len(ga_pool_this_iter) * TOP_KEEP_RATIO)
            if keep_n == 0 and ga_pool_this_iter: keep_n = 1
            top_k = ga_pool_this_iter[:keep_n]
            next_gen_cfgs = []
            if top_k: next_gen_cfgs.append(top_k[0][0][1]) # Elitism
            if top_k:
                for _ in range(INITIAL_POOL_SIZE // 2 - len(next_gen_cfgs) + (INITIAL_POOL_SIZE % 2)):
                    next_gen_cfgs.append(mutate_config(random.choice(top_k)))
            num_rand_needed = INITIAL_POOL_SIZE - len(next_gen_cfgs)
            for _ in range(max(0, num_rand_needed)): next_gen_cfgs.append(generate_initial_configs()[0])
            current_configs_pool = next_gen_cfgs[:INITIAL_POOL_SIZE]; random.shuffle(current_configs_pool)

        if ga_pool_this_iter : print(f"Iter {iteration_actual_count} Best This Iter: Fitness={iter_best_fitness_val:.4f}")
        else: print(f"Iter {iteration_actual_count} no valid results.")
        
        print("Joining iter workers for A1...")
        for p in active_worker_procs:
            if p.is_alive(): p.join(timeout=10)
            if p.is_alive(): print(f"Worker {p.pid} hanging, terminating."); p.terminate()
        print(f"Iter {iteration_actual_count} workers joined/terminated.")


    # --- HPO Run Finished ---
    print(f"\n=== HPO Run ({HPO_APPROACH_ID}, {RUN_ID}) Finished after {iteration_actual_count} iterations ===")
    hpo_total_wall_clock_duration_s = time.time() - hpo_overall_start_time_abs
    print(f"Total HPO wall-clock duration: {hpo_total_wall_clock_duration_s:.2f} seconds.")
    print(f"Overall best fitness achieved: {overall_best_fitness:.4f}")
    print(f"Achieved at wall-clock time: {(time_of_overall_best_fitness_abs - hpo_overall_start_time_abs):.2f}s from HPO start.")
    print(f"Approx total GPU busy time: {hpo_total_gpu_busy_s:.2f}s")
    print(f"Approx total CPU busy time: {hpo_total_cpu_busy_s:.2f}s")
    
    config_log_f.close(); iteration_log_f.close()
    print(f"\nLogs saved: {CONFIG_LOG_FILENAME}, {ITERATION_LOG_FILENAME}")