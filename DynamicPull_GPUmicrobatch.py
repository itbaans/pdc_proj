import time
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from collections import OrderedDict
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from queue import Empty # For worker task pulling
import csv
import os
import uuid

# --- Experiment Setup & Logging Constants ---
HPO_APPROACH_ID = "DynamicPull_GPUmicrobatch" 
RUN_ID = str(uuid.uuid4())[:8]          

CONFIG_LOG_FILENAME = f"hpo_config_log_A2_{RUN_ID}.csv"
ITERATION_LOG_FILENAME = f"hpo_iteration_log_A2_{RUN_ID}.csv"

# --- HPO Budget ---
MAX_HPO_WALL_CLOCK_SECONDS = 900  # e.g., 15 minutes for testing
# MAX_HPO_WALL_CLOCK_SECONDS = 60  # For very quick testing
MAX_ITERATIONS_SAFETY_CAP = 1000 

# --- HPO Configuration Parameters for Approach 2 ---
# NUM_ITERATIONS is now controlled by time/safety cap
INITIAL_POOL_SIZE = 20 
TOP_KEEP_RATIO = 0.3
MUTATION_RATE = 0.25
NUM_EPOCHS_PER_EVAL = 3 # Reduced for faster iterations with time limit
GPU_WORKER_PULL_BATCH_SIZE = 1 # GPU worker grabs this many tasks if available

# Fitness function parameters
OVERFITTING_PENALTY_ALPHA = 0.5
OVERFITTING_TOLERANCE = 0.05

# --- Dataset Configuration ---
TOTAL_SAMPLES_FOR_HPO_EXPERIMENT = 50000 # Reduced for faster iterations
HPO_INTERNAL_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
FLATTENED_DATA_SIZE = 54 
NUM_CLASSES = 7          

TRAIN_EVAL_SUBSET_SIZE = 5000 # For worker's train_acc evaluation subset

def prepare_shared_data(): # Unchanged
    print(f"Fetching Covertype. Will use a total of ~{TOTAL_SAMPLES_FOR_HPO_EXPERIMENT} samples for this HPO experiment.")
    covtype = fetch_covtype()
    X_full = covtype.data.astype(np.float32); y_full = covtype.target.astype(np.int64) - 1
    if TOTAL_SAMPLES_FOR_HPO_EXPERIMENT < len(X_full):
        X_exp, _, y_exp, _ = train_test_split(X_full, y_full, train_size=TOTAL_SAMPLES_FOR_HPO_EXPERIMENT, random_state=RANDOM_STATE, stratify=y_full)
        print(f"Using subset: {len(X_exp)} samples.")
    else: X_exp, y_exp = X_full, y_full; print(f"Using full dataset: {len(X_exp)} samples.")
    X_train, X_test, y_train, y_test = train_test_split(X_exp, y_exp, test_size=HPO_INTERNAL_TEST_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=y_exp)
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    train_d = torch.from_numpy(X_train_s).share_memory_(); train_l = torch.from_numpy(y_train).share_memory_()
    test_d = torch.from_numpy(X_test_s).share_memory_(); test_l = torch.from_numpy(y_test).share_memory_()
    print(f"HPO Train data shape: {train_d.shape}, HPO Test data shape: {test_d.shape}")
    return ((train_d, train_l), (test_d, test_l))

class DynamicNN(torch.nn.Module): # Unchanged
    def __init__(self, input_size, output_size, config):
        super().__init__(); self.config = config; layers = OrderedDict(); current_dim = input_size
        for i in range(config['num_layers']):
            layers[f'fc{i+1}'] = torch.nn.Linear(current_dim, config['hidden_size'])
            layers[f'relu{i+1}'] = torch.nn.ReLU()
            dr = config.get('dropout_rate', 0.0)
            if dr > 0: layers[f'dropout{i+1}'] = torch.nn.Dropout(dr)
            current_dim = config['hidden_size']
        layers['fc_out'] = torch.nn.Linear(current_dim, output_size); self.model_layers = torch.nn.Sequential(layers)
    def forward(self, x): return self.model_layers(x.view(-1, FLATTENED_DATA_SIZE))

def _evaluate_model_metrics(model, test_loader, train_eval_loader, device, criterion): # Unchanged
    model.eval(); test_correct, test_total = 0, 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data); _, predicted = torch.max(outputs.data, 1)
            test_total += targets.size(0); test_correct += (predicted == targets).sum().item()
    test_accuracy = test_correct / test_total if test_total > 0 else 0.0
    train_correct, train_total = 0, 0
    if train_eval_loader:
        with torch.no_grad():
            for data, targets in train_eval_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data); _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0); train_correct += (predicted == targets).sum().item()
    train_accuracy = train_correct / train_total if train_total > 0 else 0.0
    return test_accuracy, train_accuracy

def worker(worker_id, task_queue, result_queue,
           shared_hpo_train_data, shared_hpo_train_labels,
           shared_hpo_test_data, shared_hpo_test_labels,
           device_str): # Unchanged from your A2 logged version
    device = torch.device(device_str); is_gpu_worker = (device_str == 'cuda')
    hpo_train_dataset = TensorDataset(shared_hpo_train_data, shared_hpo_train_labels)
    hpo_test_dataset = TensorDataset(shared_hpo_test_data, shared_hpo_test_labels)
    train_loader_full = DataLoader(hpo_train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader_eval = DataLoader(hpo_test_dataset, batch_size=256, shuffle=False, num_workers=0)
    train_eval_loader = None
    if TRAIN_EVAL_SUBSET_SIZE > 0 and len(hpo_train_dataset) > 0:
        num_train_samples = len(hpo_train_dataset); eval_subset_size = min(TRAIN_EVAL_SUBSET_SIZE, num_train_samples)
        train_indices = torch.randperm(num_train_samples)[:eval_subset_size]
        train_eval_dataset_subset = Subset(hpo_train_dataset, train_indices)
        train_eval_loader = DataLoader(train_eval_dataset_subset, batch_size=256, shuffle=False, num_workers=0)

    while True:
        tasks_for_this_worker_iteration = []
        if is_gpu_worker:
            for _ in range(GPU_WORKER_PULL_BATCH_SIZE): 
                try: task = task_queue.get(timeout=0.005)
                except Empty: break 
                if task is None: tasks_for_this_worker_iteration.append(None); break
                tasks_for_this_worker_iteration.append(task)
            if not tasks_for_this_worker_iteration: 
                try: task = task_queue.get(timeout=0.1); tasks_for_this_worker_iteration.append(task)
                except Empty: tasks_for_this_worker_iteration.append(None) 
        else: 
            try: task = task_queue.get(); tasks_for_this_worker_iteration.append(task)
            except Exception: tasks_for_this_worker_iteration.append(None) 

        if not tasks_for_this_worker_iteration or tasks_for_this_worker_iteration[0] is None:
            result_queue.put(None); break 

        batch_results_to_main = []
        for task_item in tasks_for_this_worker_iteration:
            if task_item is None: continue
            config_original_idx, config_dict = task_item
            proc_start_abs = time.time(); proc_start_mono = time.monotonic()
            test_acc, train_acc = 0.0, 0.0 
            try:
                model = DynamicNN(FLATTENED_DATA_SIZE, NUM_CLASSES, config_dict).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=config_dict['lr'], 
                                             weight_decay=config_dict.get('weight_decay', 0.0))
                criterion = torch.nn.CrossEntropyLoss()
                model.train()
                for _epoch in range(NUM_EPOCHS_PER_EVAL):
                    for data, targets in train_loader_full:
                        data, targets = data.to(device), targets.to(device)
                        optimizer.zero_grad(); outputs = model(data); loss = criterion(outputs, targets)
                        loss.backward(); optimizer.step()
                test_acc, train_acc = _evaluate_model_metrics(model, test_loader_eval, train_eval_loader, device, criterion)
                del model, optimizer, criterion
                if is_gpu_worker: torch.cuda.empty_cache()
            except Exception as e:
                print(f"!!! ERROR Worker {worker_id} ({device_str}) config {config_original_idx}: {type(e).__name__} {e}")
                test_acc, train_acc = -1.0, -1.0 
            proc_end_mono = time.monotonic(); duration_s = proc_end_mono - proc_start_mono
            batch_results_to_main.append(((config_original_idx, config_dict), test_acc, train_acc,
                                          proc_start_abs, duration_s, device_str))
        if batch_results_to_main: result_queue.put(batch_results_to_main)

def calculate_complexity(config): # Unchanged
    complexity = config.get('num_layers',1) * (config.get('hidden_size',32) ** 1.5) 
    complexity += (FLATTENED_DATA_SIZE * config.get('hidden_size',32) + config.get('hidden_size',32) * NUM_CLASSES) * 0.01 
    return complexity

def generate_initial_configs(): # Unchanged
    configs = []
    for _ in range(INITIAL_POOL_SIZE):
        configs.append({'lr': 10**np.random.uniform(-4.5,-2.0),'num_layers':random.randint(1,4),
                        'hidden_size': int(2**np.random.uniform(5,9)),'dropout_rate':np.random.uniform(0.0,0.6),
                        'weight_decay':10**np.random.uniform(-7,-3.5)})
    return configs

def mutate_config(config_to_mutate): # Unchanged
    mutated_config = config_to_mutate.copy(); param = random.choice(list(mutated_config.keys())); rate = MUTATION_RATE
    if param == 'lr': mutated_config['lr'] = np.clip(mutated_config['lr'] * (1 + np.random.uniform(-rate,rate)*random.choice([-1,1])), 1e-6, 1e-1)
    elif param == 'num_layers': mutated_config['num_layers'] = np.clip(mutated_config['num_layers'] + random.choice([-1,0,1]), 1, 5)
    elif param == 'hidden_size':
        hs = mutated_config['hidden_size']; hs = int(hs * (1 + np.random.uniform(-rate,rate)*random.choice([-1,1])))
        mutated_config['hidden_size'] = np.clip(max(16, (hs//8)*8), 16, 1024)
    elif param == 'dropout_rate': mutated_config['dropout_rate'] = np.clip(config_to_mutate.get('dropout_rate',0) + np.random.uniform(-rate/2,rate/2), 0.0, 0.7)
    elif param == 'weight_decay': mutated_config['weight_decay'] = np.clip(config_to_mutate.get('weight_decay',0) * (1 + np.random.uniform(-rate,rate)*random.choice([-1,1])), 1e-8, 1e-2)
    return mutated_config

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # --- Log File Initialization ---
    config_log_fexists = os.path.exists(CONFIG_LOG_FILENAME)
    config_log_f = open(CONFIG_LOG_FILENAME, 'a', newline='')
    config_writer = csv.writer(config_log_f)
    if not config_log_fexists:
        config_writer.writerow(["hpo_approach_id", "run_id", "iteration", "config_original_idx",
                                "lr", "num_layers", "hidden_size", "dropout_rate", "weight_decay",
                                "device_used", "processing_start_time_abs", "processing_duration_s",
                                "test_accuracy", "train_accuracy", "fitness_score"])

    iter_log_fexists = os.path.exists(ITERATION_LOG_FILENAME)
    iter_log_f = open(ITERATION_LOG_FILENAME, 'a', newline='')
    iter_writer = csv.writer(iter_log_f)
    if not iter_log_fexists:
        iter_writer.writerow(["hpo_approach_id", "run_id", "iteration",
                              "iteration_start_time_abs", "iteration_end_time_abs", "iteration_duration_s",
                              "num_configs_evaluated", "num_configs_gpu", "num_configs_cpu", # Detailed counts
                              "best_fitness_this_iteration", "best_fitness_so_far", 
                              "wall_clock_time_to_best_so_far_s",
                              "approx_gpu_busy_time_s", "approx_cpu_busy_time_s"])
    # --- End Log File Init ---

    hpo_overall_start_time_abs = time.time() # For the entire HPO run
    print(f"Starting HPO Approach: {HPO_APPROACH_ID}, Run ID: {RUN_ID}")
    print(f"Target HPO Duration: {MAX_HPO_WALL_CLOCK_SECONDS} seconds.")
    print(f"Config log: {CONFIG_LOG_FILENAME}\nIteration log: {ITERATION_LOG_FILENAME}")

    shared_train_data, shared_test_data = prepare_shared_data()
    print("Shared data prepared.")

    current_configs_pool = generate_initial_configs()
    overall_best_fitness_achieved = -float('inf')
    time_of_overall_best_fitness_abs = hpo_overall_start_time_abs
    
    hpo_total_gpu_busy_time_s = 0
    hpo_total_cpu_busy_time_s = 0
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
            print("Config pool empty, generating new for safety.")
            current_configs_pool = generate_initial_configs()
        
        configs_this_gen_with_ids = [(idx, cfg) for idx, cfg in enumerate(current_configs_pool)]
        configs_this_gen_with_ids.sort(key=lambda item: calculate_complexity(item[1]), reverse=True)
        
        num_tasks_this_gen = len(configs_this_gen_with_ids)
        print(f"Evaluating {num_tasks_this_gen} configs, individually queued.")

        task_q = mp.Queue(maxsize=num_tasks_this_gen + mp.cpu_count() + 10) # Increased buffer slightly
        result_q = mp.Queue()
        
        worker_procs = []; gpu_worker_count = 0; worker_id_gen = 0
        worker_args_base = (task_q, result_q, 
                            shared_train_data[0], shared_train_data[1], 
                            shared_test_data[0], shared_test_data[1])

        if torch.cuda.is_available():
            p_gpu = mp.Process(target=worker, args=(worker_id_gen, *worker_args_base, 'cuda'))
            p_gpu.start(); worker_procs.append(p_gpu); gpu_worker_count = 1; worker_id_gen+=1
            print("CUDA available, GPU worker started.")
        
        cpu_worker_count_to_start = max(1, mp.cpu_count() - gpu_worker_count)
        print(f"Creating {cpu_worker_count_to_start} CPU workers.")
        for _ in range(cpu_worker_count_to_start):
            p_cpu = mp.Process(target=worker, args=(worker_id_gen, *worker_args_base, 'cpu'))
            p_cpu.start(); worker_procs.append(p_cpu); worker_id_gen+=1
        
        num_active_workers_started = len(worker_procs)
        if num_active_workers_started == 0: print("CRITICAL: No workers started!"); break


        for cfg_task_tuple in configs_this_gen_with_ids: task_q.put(cfg_task_tuple)
        for _ in range(num_active_workers_started): task_q.put(None)
        print(f"All {num_tasks_this_gen} tasks and {num_active_workers_started} term signals enqueued.")

        iter_raw_results_list = []; iter_finished_procs = 0
        iter_gpu_busy_s_this_iter = 0; iter_cpu_busy_s_this_iter = 0
        iter_gpu_cfg_count_this_iter = 0; iter_cpu_cfg_count_this_iter = 0

        while iter_finished_procs < num_active_workers_started:
            try:
                res_list_from_worker = result_q.get(timeout=300) # 5 min timeout
                if res_list_from_worker is None: iter_finished_procs += 1
                else:
                    iter_raw_results_list.extend(res_list_from_worker)
                    for _res_item_tuple in res_list_from_worker:
                        _dur = _res_item_tuple[4]; _dev = _res_item_tuple[5]
                        if _dev == 'cuda': iter_gpu_busy_s_this_iter += _dur; iter_gpu_cfg_count_this_iter +=1
                        else: iter_cpu_busy_s_this_iter += _dur; iter_cpu_cfg_count_this_iter +=1
            except Empty:
                print(f"Timeout A2 results. Finished: {iter_finished_procs}/{num_active_workers_started}"); break # Check aliveness
        
        iter_end_abs_timestamp = time.time(); iter_end_mono_timestamp = time.monotonic()
        iter_duration_s_val = iter_end_mono_timestamp - iter_start_mono_timestamp
        hpo_total_gpu_busy_time_s += iter_gpu_busy_s_this_iter
        hpo_total_cpu_busy_time_s += iter_cpu_busy_s_this_iter

        iter_best_fitness_val = -float('inf')
        ga_pool_this_iter = []

        for res_item_tuple_full in iter_raw_results_list:
            (cfg_id_tuple_full, t_acc, tr_acc, proc_abs_start_full, dur_s_full, dev_s_full) = res_item_tuple_full
            cfg_orig_idx_full, cfg_d_full = cfg_id_tuple_full
            
            fitness = -float('inf')
            if t_acc >= 0 and tr_acc >=0: 
                overfit_gap = tr_acc - t_acc
                penalty = OVERFITTING_PENALTY_ALPHA * max(0, overfit_gap - OVERFITTING_TOLERANCE)
                fitness = t_acc - penalty
            
            config_writer.writerow([
                HPO_APPROACH_ID, RUN_ID, iteration_actual_count, cfg_orig_idx_full,
                f"{cfg_d_full['lr']:.6e}", cfg_d_full['num_layers'], cfg_d_full['hidden_size'],
                f"{cfg_d_full.get('dropout_rate',0):.4f}", f"{cfg_d_full.get('weight_decay',0):.6e}",
                dev_s_full, f"{proc_abs_start_full:.3f}", f"{dur_s_full:.3f}",
                f"{t_acc:.4f}", f"{tr_acc:.4f}", f"{fitness:.4f}"
            ])
            if fitness > iter_best_fitness_val: iter_best_fitness_val = fitness
            if fitness > overall_best_fitness_achieved:
                overall_best_fitness_achieved = fitness; time_of_overall_best_fitness_abs = time.time()
            if t_acc >= 0: ga_pool_this_iter.append( (cfg_id_tuple_full, fitness) )
        config_log_f.flush()

        iter_writer.writerow([
            HPO_APPROACH_ID, RUN_ID, iteration_actual_count,
            f"{iter_start_abs_timestamp:.3f}", f"{iter_end_abs_timestamp:.3f}", f"{iter_duration_s_val:.3f}",
            len(iter_raw_results_list), iter_gpu_cfg_count_this_iter, iter_cpu_cfg_count_this_iter,
            f"{iter_best_fitness_val:.4f}", f"{overall_best_fitness_achieved:.4f}",
            f"{(time_of_overall_best_fitness_abs - hpo_overall_start_time_abs):.3f}",
            f"{iter_gpu_busy_s_this_iter:.3f}", f"{iter_cpu_busy_s_this_iter:.3f}"
        ])
        iter_log_f.flush()

        # --- GA ---
        if not ga_pool_this_iter: current_configs_pool = generate_initial_configs()
        else:
            ga_pool_this_iter.sort(key=lambda x: x[1], reverse=True)
            keep_n = int(len(ga_pool_this_iter) * TOP_KEEP_RATIO)
            if keep_n == 0 and ga_pool_this_iter: keep_n = 1
            top_k = ga_pool_this_iter[:keep_n]
            next_gen_cfgs = []
            if top_k: next_gen_cfgs.append(top_k[0][0][1]) 
            if top_k:
                for _ in range(INITIAL_POOL_SIZE // 2 - len(next_gen_cfgs) + (INITIAL_POOL_SIZE % 2)):
                    parent_cfg_tuple_ga, _ = random.choice(top_k)
                    next_gen_cfgs.append(mutate_config(parent_cfg_tuple_ga[1]))
            num_rand_needed = INITIAL_POOL_SIZE - len(next_gen_cfgs)
            for _ in range(max(0, num_rand_needed)): next_gen_cfgs.append(generate_initial_configs()[0])
            current_configs_pool = next_gen_cfgs[:INITIAL_POOL_SIZE]; random.shuffle(current_configs_pool)

        if ga_pool_this_iter:
            # Find the full result package for the best of this iteration to print details
            best_of_iter_cfg_tuple_print, best_of_iter_fit_print = ga_pool_this_iter[0]
            for res_tuple_full_print in iter_raw_results_list:
                if res_tuple_full_print[0] == best_of_iter_cfg_tuple_print:
                    _t_p, _tr_p, _s_p, _d_p, _dev_p = res_tuple_full_print[1], res_tuple_full_print[2], res_tuple_full_print[3], res_tuple_full_print[4], res_tuple_full_print[5]
                    print(f"Iter {iteration_actual_count} Best Fitness: {best_of_iter_fit_print:.4f} (Test: {_t_p:.4f}, Train: {_tr_p:.4f}, Dev: {_dev_p}, Dur: {_d_p:.2f}s)")
                    break
        else: print(f"Iter {iteration_actual_count} had no successful GA candidates.")
        
        print("Joining iter workers for A2...")
        for p_final in worker_procs:
            if p_final.is_alive(): p_final.join(timeout=10)
            if p_final.is_alive(): print(f"Worker {p_final.pid} hanging, terminating."); p_final.terminate()
        print(f"Iter {iteration_actual_count} workers joined/terminated.")


    # --- HPO Run Finished ---
    print(f"\n=== HPO Run ({HPO_APPROACH_ID}, {RUN_ID}) Finished after {iteration_actual_count} iterations ===")
    hpo_total_wall_clock_duration_s = time.time() - hpo_overall_start_time_abs
    print(f"Total HPO wall-clock duration: {hpo_total_wall_clock_duration_s:.2f} seconds.")
    print(f"Overall best fitness achieved: {overall_best_fitness_achieved:.4f}")
    print(f"Achieved at wall-clock time: {(time_of_overall_best_fitness_abs - hpo_overall_start_time_abs):.2f}s from HPO start.")
    print(f"Approx total GPU busy time: {hpo_total_gpu_busy_time_s:.2f}s")
    print(f"Approx total CPU busy time: {hpo_total_cpu_busy_time_s:.2f}s")
    
    config_log_f.close(); iter_log_f.close()
    print(f"\nLogs saved: {CONFIG_LOG_FILENAME}, {ITERATION_LOG_FILENAME}")