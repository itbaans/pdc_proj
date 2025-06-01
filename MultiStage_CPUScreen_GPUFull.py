import time
import random
import torch
import torch.multiprocessing as mp
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Subset
from collections import OrderedDict, defaultdict 
from queue import Empty, Full 
import csv
import os
import copy 
import uuid 
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler

# --- Experiment Setup & Logging Constants ---
HPO_APPROACH_ID = "MultiStage_CPUScreen_GPUFull"
RUN_ID = str(uuid.uuid4())[:8]

CONFIG_LOG_FILENAME_A3 = f"hpo_config_log_A3_{RUN_ID}.csv"
CPU_SCREEN_LOG_FILENAME_A3 = f"hpo_cpuscreen_log_A3_{RUN_ID}.csv"
ITERATION_LOG_FILENAME_A3 = f"hpo_iteration_log_A3_{RUN_ID}.csv"

# --- HPO Budget ---
MAX_HPO_WALL_CLOCK_SECONDS = 900  # e.g., 15 minutes for testing
# MAX_HPO_WALL_CLOCK_SECONDS = 60  # For very quick testing
MAX_ITERATIONS_SAFETY_CAP = 1000 # Safety cap on macro-iterations

# --- HPO Configuration Parameters for Approach 3 ---
# NUM_MACRO_ITERATIONS is now controlled by time/safety cap
CPU_WORKER_CONFIG_BATCH_SIZE = 8 
CPU_WORKER_PROMOTE_TOP_N = 4    
INITIAL_CPU_POPULATION_SIZE = 6 

NUM_CPU_SCREEN_EPOCHS = 1 
NUM_GPU_FULL_EPOCHS = 3 # Reduced for faster iterations with time limit 

CPU_GA_MUTATION_RATE = 0.4
CPU_GA_TOP_KEEP_RATIO = 0.5 

OVERFITTING_PENALTY_ALPHA = 0.5
OVERFITTING_TOLERANCE = 0.05
PBT_EXPLOIT_PROBABILITY = 0.2 

# --- Dataset Configuration ---
TOTAL_SAMPLES_FOR_HPO_EXPERIMENT = 50000 # Reduced for faster iterations
HPO_INTERNAL_TEST_SPLIT_RATIO = 0.2
CPU_SCREEN_TRAIN_SUBSET_SIZE = 2000 
GPU_TRAIN_EVAL_SUBSET_SIZE = 5000 
RANDOM_STATE = 42
FLATTENED_DATA_SIZE = 54 
NUM_CLASSES = 7          

def prepare_shared_data(): # Unchanged
    print(f"Fetching Covertype. Will use ~{TOTAL_SAMPLES_FOR_HPO_EXPERIMENT} samples.")
    covtype = fetch_covtype(); X_full = covtype.data.astype(np.float32); y_full = covtype.target.astype(np.int64) - 1
    if TOTAL_SAMPLES_FOR_HPO_EXPERIMENT < len(X_full):
        X_exp, _, y_exp, _ = train_test_split(X_full, y_full, train_size=TOTAL_SAMPLES_FOR_HPO_EXPERIMENT, random_state=RANDOM_STATE, stratify=y_full)
        print(f"Using subset: {len(X_exp)} samples.")
    else: X_exp, y_exp = X_full, y_full; print(f"Using full dataset: {len(X_exp)} samples.")
    X_train, X_test, y_train, y_test = train_test_split(X_exp, y_exp, test_size=HPO_INTERNAL_TEST_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=y_exp)
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_test_s = scaler.transform(X_test)
    train_d = torch.from_numpy(X_train_s).share_memory_(); train_l = torch.from_numpy(y_train).share_memory_()
    test_d = torch.from_numpy(X_test_s).share_memory_(); test_l = torch.from_numpy(y_test).share_memory_()
    print(f"HPO Train: {train_d.shape}, HPO Test: {test_d.shape}")
    return ((train_d, train_l), (test_d, test_l))

class DynamicNN(torch.nn.Module): # Unchanged
    def __init__(self, input_size, output_size, config):
        super().__init__(); self.config = config; layers = OrderedDict(); current_dim = input_size
        for i in range(config['num_layers']):
            layers[f'fc{i+1}'] = torch.nn.Linear(current_dim, config['hidden_size'])
            layers[f'relu{i+1}'] = torch.nn.ReLU()
            dr = config.get('dropout_rate',0.0) 
            if dr > 0: layers[f'dropout{i+1}'] = torch.nn.Dropout(dr)
            current_dim = config['hidden_size']
        layers['fc_out'] = torch.nn.Linear(current_dim, output_size); self.model_layers = torch.nn.Sequential(layers)
    def forward(self, x): return self.model_layers(x.view(-1, FLATTENED_DATA_SIZE))

def _evaluate_model_metrics_internal(model, loader, device, criterion=None): # Unchanged
    model.eval(); correct, total, total_loss = 0, 0, 0.0
    if not loader: return 0.0, float('inf') 
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data); _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0); correct += (predicted == targets).sum().item()
            if criterion: total_loss += criterion(outputs, targets).item() * data.size(0)
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 and criterion else float('inf')
    return accuracy, avg_loss

def _generate_random_config_a3(): # Unchanged
    return {'lr':10**np.random.uniform(-4.5,-2.0),'num_layers':random.randint(1,3),
            'hidden_size':int(2**np.random.uniform(4,7)),'dropout_rate':np.random.uniform(0.0,0.5),
            'weight_decay':10**np.random.uniform(-7,-4)}

def _mutate_config_a3(config, mutation_rate_factor=1.0): # Unchanged
    mutated_config=copy.deepcopy(config);param=random.choice(list(mutated_config.keys()));rate=CPU_GA_MUTATION_RATE*mutation_rate_factor
    if param=='lr':mutated_config['lr']=np.clip(mutated_config['lr']*(1+np.random.uniform(-rate,rate)*random.choice([-1,1])),1e-6,1e-1)
    elif param=='num_layers':mutated_config['num_layers']=np.clip(mutated_config['num_layers']+random.choice([-1,0,1]),1,3)
    elif param=='hidden_size':
        hs=mutated_config['hidden_size'];hs=int(hs*(1+np.random.uniform(-rate,rate)*random.choice([-1,1])))
        mutated_config['hidden_size']=np.clip(max(16,(hs//8)*8),16,256)
    elif param=='dropout_rate':mutated_config['dropout_rate']=np.clip(mutated_config.get('dropout_rate',0)+np.random.uniform(-rate/2,rate/2),0.0,0.7)
    elif param=='weight_decay':mutated_config['weight_decay']=np.clip(mutated_config.get('weight_decay',0)*(1+np.random.uniform(-rate,rate)*random.choice([-1,1])),1e-8,1e-2)
    return mutated_config

def cpu_screener_worker(worker_id, control_q, feedback_q_ignored, gpu_submission_q, shared_data_tuple,
                        cpu_screen_log_filepath, cpu_screen_log_lock): # Unchanged (from fixed version)
    device=torch.device('cpu');(s_train_data,s_train_labels),_=shared_data_tuple;train_dataset=TensorDataset(s_train_data,s_train_labels)
    screen_subset_indices=torch.randperm(len(train_dataset))[:min(len(train_dataset),CPU_SCREEN_TRAIN_SUBSET_SIZE)]
    screen_dataset=Subset(train_dataset,screen_subset_indices);screen_loader=DataLoader(screen_dataset,batch_size=128,shuffle=True)
    population=[{'config':_generate_random_config_a3(),'fitness':-float('inf'),'macro_iter_gpu_eval':-1} for _ in range(INITIAL_CPU_POPULATION_SIZE)]
    current_macro_iter_num=0;print(f"[CPU Worker {worker_id}] Started. Screen subset size: {len(screen_dataset)}")
    while True:
        cmd_data_tuple=None
        try:
            cmd_data_tuple=control_q.get(timeout=1800) 
            if cmd_data_tuple is None:break
            cmd,data=cmd_data_tuple
            if cmd=="START_SCREENING":
                current_macro_iter_num=data['macro_iteration'];configs_for_screening=[]
                population.sort(key=lambda x:x['fitness'],reverse=True)
                num_elites=int(len(population)*CPU_GA_TOP_KEEP_RATIO)
                if len(population)>0 and num_elites==0:num_elites=1
                elites=[p['config'] for p in population[:num_elites]]
                for _ in range(CPU_WORKER_CONFIG_BATCH_SIZE):
                    if elites and random.random()>0.3:configs_for_screening.append(_mutate_config_a3(random.choice(elites)))
                    else:configs_for_screening.append(_generate_random_config_a3())
                batch_screen_results_list=[]
                for cfg_batch_idx,cfg_to_screen in enumerate(configs_for_screening):
                    screen_start_abs=time.time();screen_start_mono=time.monotonic()
                    loss_dec=-float('inf');final_loss_1_epoch_val=float('inf');acc_1ep_val=0.0
                    try:
                        model=DynamicNN(FLATTENED_DATA_SIZE,NUM_CLASSES,cfg_to_screen).to(device)
                        opt=torch.optim.Adam(model.parameters(),lr=cfg_to_screen['lr'])
                        crit=torch.nn.CrossEntropyLoss()
                        _,initial_loss=_evaluate_model_metrics_internal(model,screen_loader,device,crit)
                        if initial_loss==float('inf'):initial_loss=999.0
                        model.train()
                        for d_batch,t_batch in screen_loader:
                            opt.zero_grad();out=model(d_batch.to(device));l=crit(out,t_batch.to(device));l.backward();opt.step()
                        acc_1ep_val,final_loss_1_epoch_val=_evaluate_model_metrics_internal(model,screen_loader,device,crit)
                        loss_dec=initial_loss-final_loss_1_epoch_val;del model,opt,crit
                    except Exception as e_scr:print(f"[CPU{worker_id}] Error screening cfg {cfg_batch_idx}: {e_scr}")
                    screen_end_mono=time.monotonic();screen_dur_s=screen_end_mono-screen_start_mono
                    batch_screen_results_list.append({'config':cfg_to_screen,'loss_decrease':loss_dec, 
                                                    'accuracy_1epoch':acc_1ep_val,'final_loss_1epoch':final_loss_1_epoch_val,
                                                    'duration_s':screen_dur_s,'start_time_abs':screen_start_abs,
                                                    'config_batch_idx':cfg_batch_idx})
                batch_screen_results_list.sort(key=lambda x:x['loss_decrease'],reverse=True)
                promoted_this_worker_count=0
                for i_promo in range(min(CPU_WORKER_PROMOTE_TOP_N,len(batch_screen_results_list))):
                    res_to_promote=batch_screen_results_list[i_promo]
                    if res_to_promote['loss_decrease']>-0.5:
                        gpu_submission_q.put({'config':res_to_promote['config'],'origin_cpu_id':worker_id,
                                            'screen_loss_decrease':res_to_promote['loss_decrease'], 
                                            'screen_accuracy_1epoch':res_to_promote['accuracy_1epoch'],
                                            'macro_iteration':current_macro_iter_num,
                                            'cpu_screen_duration_s':res_to_promote['duration_s'], 
                                            'cpu_screen_time_abs':res_to_promote['start_time_abs']})
                        promoted_this_worker_count+=1
                with cpu_screen_log_lock:
                    with open(cpu_screen_log_filepath,'a',newline='') as log_f:
                        temp_csv_writer=csv.writer(log_f)
                        for idx_res,screened_res_item in enumerate(batch_screen_results_list):
                            cfg_screened=screened_res_item['config'];was_promoted=idx_res<promoted_this_worker_count
                            temp_csv_writer.writerow([HPO_APPROACH_ID,RUN_ID,current_macro_iter_num,worker_id,screened_res_item['config_batch_idx'],
                                                    f"{cfg_screened['lr']:.6e}",cfg_screened['num_layers'],cfg_screened['hidden_size'],
                                                    f"{cfg_screened.get('dropout_rate',0):.4f}",f"{cfg_screened.get('weight_decay',0):.6e}",
                                                    f"{screened_res_item['start_time_abs']:.3f}",f"{screened_res_item['duration_s']:.3f}",
                                                    f"{screened_res_item['accuracy_1epoch']:.4f}",f"{screened_res_item['final_loss_1epoch']:.4f}", 
                                                    f"{screened_res_item['loss_decrease']:.4f}",was_promoted])
            elif cmd=="GPU_RESULT":
                orig_cfg=data['config'];fitness_gpu=data['fitness']
                population.append({'config':orig_cfg,'fitness':fitness_gpu,'macro_iter_gpu_eval':current_macro_iter_num})
                population.sort(key=lambda x:x['fitness'],reverse=True);population=population[:INITIAL_CPU_POPULATION_SIZE]
            elif cmd=="GLOBAL_BEST_CONFIG":
                gb_cfg=data['config'];gb_fit=data['fitness']
                if population and (population[0]['fitness'] is None or population[0]['fitness']<gb_fit-0.02) and random.random()<PBT_EXPLOIT_PROBABILITY:
                    new_cfg=_mutate_config_a3(copy.deepcopy(gb_cfg),0.5)
                    population.append({'config':new_cfg,'fitness':-float('inf'),'macro_iter_gpu_eval':current_macro_iter_num})
                    population.sort(key=lambda x:x['fitness'],reverse=True);population=population[:INITIAL_CPU_POPULATION_SIZE]
            control_q.task_done()
        except Empty:print(f"[CPU{worker_id}] ControlQ timeout.");continue
        except Exception as e:
            print(f"!!! FATAL CPU{worker_id} (Cmd: {cmd if 'cmd' in locals() else 'Unknown'}): {type(e).__name__} {e}")
            if cmd_data_tuple:
                try:
                    control_q.task_done()
                except (ValueError, AttributeError):
                    pass
    print(f"[CPU Worker {worker_id}] Terminating.")

def gpu_evaluator_worker(gpu_submission_q, main_results_q, shared_data_tuple,
                         config_log_filepath, config_log_lock): # Unchanged (from fixed version)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print(f"[GPU Worker] Started on {device}")
    (s_train_data,s_train_labels),(s_test_data,s_test_labels)=shared_data_tuple
    train_full_ds=TensorDataset(s_train_data,s_train_labels);test_ds=TensorDataset(s_test_data,s_test_labels)
    gpu_train_loader=DataLoader(train_full_ds,batch_size=128,shuffle=True,num_workers=0)
    gpu_test_loader=DataLoader(test_ds,batch_size=256,shuffle=False,num_workers=0)
    gpu_train_eval_loader=None
    if GPU_TRAIN_EVAL_SUBSET_SIZE>0 and len(train_full_ds)>0:
        subset_indices=torch.randperm(len(train_full_ds))[:min(len(train_full_ds),GPU_TRAIN_EVAL_SUBSET_SIZE)]
        gpu_train_eval_loader=DataLoader(Subset(train_full_ds,subset_indices),batch_size=256,shuffle=False,num_workers=0)
    while True:
        submission=None
        try:
            submission=gpu_submission_q.get(timeout=1800)
            if submission is None:break
            cfg_screened=submission['config'];origin_cpu=submission['origin_cpu_id']
            macro_iter=submission['macro_iteration'];screen_loss_dec=submission['screen_loss_decrease']
            screen_acc_1ep=submission['screen_accuracy_1epoch'];cpu_screen_dur=submission['cpu_screen_duration_s']
            cpu_screen_abs_time=submission['cpu_screen_time_abs']
            gpu_proc_start_abs=time.time();gpu_proc_start_mono=time.monotonic()
            t_acc,tr_acc,fit=0.0,0.0,-float('inf');gpu_cfg_to_run=copy.deepcopy(cfg_screened)
            if screen_loss_dec>0.05 or screen_acc_1ep>(1.0/NUM_CLASSES+0.05):
                gpu_cfg_to_run['hidden_size']=min(1024,int(gpu_cfg_to_run['hidden_size']*random.uniform(1.2,2.0)))
                gpu_cfg_to_run['num_layers']=min(5,gpu_cfg_to_run['num_layers']+random.choice([0,1,1,2]))
            try:
                model=DynamicNN(FLATTENED_DATA_SIZE,NUM_CLASSES,gpu_cfg_to_run).to(device)
                opt=torch.optim.Adam(model.parameters(),lr=gpu_cfg_to_run['lr'],weight_decay=gpu_cfg_to_run.get('weight_decay',0))
                crit=torch.nn.CrossEntropyLoss();model.train()
                for _ep in range(NUM_GPU_FULL_EPOCHS):
                    for d_batch,t_batch in gpu_train_loader:opt.zero_grad();out=model(d_batch.to(device));l=crit(out,t_batch.to(device));l.backward();opt.step()
                t_acc,_=_evaluate_model_metrics_internal(model,gpu_test_loader,device)
                tr_acc,_=_evaluate_model_metrics_internal(model,gpu_train_eval_loader,device,crit)
                gap=tr_acc-t_acc;penalty=OVERFITTING_PENALTY_ALPHA*max(0,gap-OVERFITTING_TOLERANCE);fit=t_acc-penalty
                del model,opt,crit;
                if device.type=='cuda':torch.cuda.empty_cache()
            except Exception as e_gpu:print(f"!!! GPU Error (cfg from CPU{origin_cpu}): {type(e_gpu).__name__} {e_gpu}");t_acc=-1.0;tr_acc=-1.0;fit=-float('inf')
            gpu_proc_end_mono=time.monotonic();gpu_dur_s=gpu_proc_end_mono-gpu_proc_start_mono
            with config_log_lock:
                with open(config_log_filepath,'a',newline='') as log_f:
                    temp_csv_writer=csv.writer(log_f)
                    temp_csv_writer.writerow([HPO_APPROACH_ID,RUN_ID,macro_iter,origin_cpu,hash(str(cfg_screened)),
                                            f"{cfg_screened['lr']:.6e}",cfg_screened['num_layers'],cfg_screened['hidden_size'],
                                            f"{cfg_screened.get('dropout_rate',0):.4f}",f"{cfg_screened.get('weight_decay',0):.6e}",
                                            device.type,f"{gpu_proc_start_abs:.3f}",f"{gpu_dur_s:.3f}",
                                            f"{t_acc:.4f}",f"{tr_acc:.4f}",f"{fit:.4f}",
                                            gpu_cfg_to_run['num_layers'],gpu_cfg_to_run['hidden_size'],
                                            f"{screen_loss_dec:.4f}",f"{screen_acc_1ep:.4f}",f"{cpu_screen_dur:.3f}"])
            main_results_q.put({'config_screened':cfg_screened,'gpu_config_run':gpu_cfg_to_run,
                                'test_acc':t_acc,'train_acc':tr_acc,'fitness':fit,
                                'origin_cpu_id':origin_cpu,'gpu_duration_s':gpu_dur_s,
                                'macro_iteration':macro_iter,'gpu_eval_time_abs':gpu_proc_start_abs,
                                'screen_loss_decrease':screen_loss_dec,'screen_accuracy_1epoch':screen_acc_1ep,
                                'cpu_screen_duration_s':cpu_screen_dur,'cpu_screen_time_abs':cpu_screen_abs_time})
            gpu_submission_q.task_done()
        except Empty:continue
        except Exception as e:print(f"!!! FATAL GPU Worker: {type(e).__name__} {e}");break
    print(f"[GPU Worker] Terminating.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    hpo_overall_start_time_abs = time.time()

    # --- Log File Initialization ---
    cfg_log_fexists_a3 = os.path.exists(CONFIG_LOG_FILENAME_A3)
    with open(CONFIG_LOG_FILENAME_A3, 'a', newline='') as cfg_log_f_a3:
        cfg_writer_a3 = csv.writer(cfg_log_f_a3)
        if not cfg_log_fexists_a3:
            cfg_writer_a3.writerow(["hpo_approach_id","run_id","macro_iteration","origin_cpu_id","config_screened_hash",
                                    "screened_lr","screened_num_layers","screened_hidden_size","screened_dropout","screened_l2",
                                    "gpu_device_used","gpu_processing_start_time_abs","gpu_processing_duration_s",
                                    "gpu_test_accuracy","gpu_train_accuracy","gpu_fitness_score",
                                    "gpu_run_num_layers","gpu_run_hidden_size", "screen_loss_decrease",
                                    "screen_accuracy_1epoch","cpu_screen_duration_s"])
    config_writer_lock = mp.Lock()
    cpu_screen_log_fexists_a3 = os.path.exists(CPU_SCREEN_LOG_FILENAME_A3)
    with open(CPU_SCREEN_LOG_FILENAME_A3, 'a', newline='') as cpu_screen_log_f_a3:
        cpu_screen_writer_a3 = csv.writer(cpu_screen_log_f_a3)
        if not cpu_screen_log_fexists_a3:
            cpu_screen_writer_a3.writerow(["hpo_approach_id","run_id","macro_iteration","cpu_worker_id","config_batch_idx",
                                           "lr","num_layers","hidden_size","dropout","l2",
                                           "screen_time_abs","screen_duration_s",
                                           "screen_accuracy_1epoch","screen_final_loss_1epoch","screen_loss_decrease","promoted_to_gpu"])
    cpu_screen_writer_lock = mp.Lock()
    iter_log_f_a3 = open(ITERATION_LOG_FILENAME_A3, 'a', newline='')
    iter_writer_a3 = csv.writer(iter_log_f_a3)
    if not os.path.exists(ITERATION_LOG_FILENAME_A3) or os.path.getsize(ITERATION_LOG_FILENAME_A3) == 0:
        iter_writer_a3.writerow(["hpo_approach_id","run_id","macro_iteration",
                                 "iteration_start_time_abs","iteration_end_time_abs","iteration_duration_s",
                                 "num_configs_screened_cpu_total","num_configs_promoted_to_gpu","num_configs_evaluated_gpu",
                                 "best_fitness_this_iteration_gpu","best_fitness_so_far_gpu",
                                 "wall_clock_time_to_best_so_far_s",
                                 "total_cpu_screen_time_s_iter","total_gpu_eval_time_s_iter"])
    # --- End Log File Init ---
    print(f"Starting HPO Approach: {HPO_APPROACH_ID}, Run ID: {RUN_ID}")
    print(f"Target HPO Duration: {MAX_HPO_WALL_CLOCK_SECONDS} seconds.") # Added this print
    print(f"GPU Config Log: {CONFIG_LOG_FILENAME_A3}\nCPU Screen Log: {CPU_SCREEN_LOG_FILENAME_A3}\nIteration Log: {ITERATION_LOG_FILENAME_A3}")

    shared_data = prepare_shared_data()
    num_actual_cpu_workers = max(1, mp.cpu_count() - (1 if torch.cuda.is_available() else 0))
    print(f"Main: Using {num_actual_cpu_workers} CPU screener workers.")

    gpu_submit_q = mp.JoinableQueue(); main_collect_q = mp.Queue()
    cpu_ctrl_queues = [mp.JoinableQueue() for _ in range(num_actual_cpu_workers)]
    cpu_procs = [mp.Process(target=cpu_screener_worker, 
                            args=(i,cpu_ctrl_queues[i],None,gpu_submit_q,shared_data,
                                  CPU_SCREEN_LOG_FILENAME_A3,cpu_screen_writer_lock)) for i in range(num_actual_cpu_workers)]
    for p in cpu_procs: p.start()
    gpu_proc = mp.Process(target=gpu_evaluator_worker, 
                          args=(gpu_submit_q,main_collect_q,shared_data,
                                CONFIG_LOG_FILENAME_A3,config_writer_lock))
    gpu_proc.start()

    overall_best_gpu_eval = {'fitness': -float('inf'), 'full_package': None}
    time_of_overall_best_abs = hpo_overall_start_time_abs
    total_hpo_cpu_screen_time = 0; total_hpo_gpu_eval_time = 0
    macro_iteration_actual_count = 0

    # --- Main HPO Loop (Time-Based) for Approach 3 ---
    while True:
        current_hpo_duration_s = time.time() - hpo_overall_start_time_abs
        if current_hpo_duration_s >= MAX_HPO_WALL_CLOCK_SECONDS:
            print(f"Main: Reached max HPO duration of {MAX_HPO_WALL_CLOCK_SECONDS:.0f}s. Stopping.")
            break
        if macro_iteration_actual_count >= MAX_ITERATIONS_SAFETY_CAP:
            print(f"Main: Reached safety cap of {MAX_ITERATIONS_SAFETY_CAP} macro-iterations. Stopping.")
            break
        
        macro_iteration_actual_count += 1
        iter_start_abs_timestamp = time.time()
        iter_start_mono_timestamp = time.monotonic()
        print(f"\n--- Main: Macro Iteration {macro_iteration_actual_count} ({HPO_APPROACH_ID}) - Time: {current_hpo_duration_s:.0f}s / {MAX_HPO_WALL_CLOCK_SECONDS}s ---")

        iter_total_cpu_screen_duration_this_iter = 0
        iter_total_gpu_eval_duration_this_iter = 0
        num_configs_screened_total_this_iter = CPU_WORKER_CONFIG_BATCH_SIZE * num_actual_cpu_workers

        for i in range(num_actual_cpu_workers):
            cpu_ctrl_queues[i].put(("START_SCREENING", {'macro_iteration': macro_iteration_actual_count, 'num_cpu_workers': num_actual_cpu_workers}))
        
        for i in range(num_actual_cpu_workers): cpu_ctrl_queues[i].join()
        print(f"Main: All CPU workers finished screening for iter {macro_iteration_actual_count}.")
        
        num_promoted_to_gpu_this_iter = gpu_submit_q.qsize()
        print(f"Main: {num_promoted_to_gpu_this_iter} configs promoted to GPU queue.")
        if num_promoted_to_gpu_this_iter == 0 and macro_iteration_actual_count > 1 : 
            print("Main: No configs promoted. Ending HPO early."); break
        
        gpu_submit_q.join()
        print(f"Main: GPU finished evaluating {num_promoted_to_gpu_this_iter} configs for iter {macro_iteration_actual_count}.")

        gpu_results_collected_this_iter = []
        while not main_collect_q.empty():
            try: gpu_results_collected_this_iter.append(main_collect_q.get_nowait())
            except Empty: break
        
        num_gpu_evaluated_this_iter = len(gpu_results_collected_this_iter)
        print(f"Main: Collected {num_gpu_evaluated_this_iter} results from GPU.")

        iter_best_gpu_fitness_this_iter = -float('inf')
        for gpu_res_pkg_item in gpu_results_collected_this_iter:
            iter_total_gpu_eval_duration_this_iter += gpu_res_pkg_item['gpu_duration_s']
            iter_total_cpu_screen_duration_this_iter += gpu_res_pkg_item['cpu_screen_duration_s']
            fitness = gpu_res_pkg_item['fitness']
            if fitness > iter_best_gpu_fitness_this_iter: iter_best_gpu_fitness_this_iter = fitness
            if fitness > overall_best_gpu_eval['fitness']:
                overall_best_gpu_eval['fitness'] = fitness
                overall_best_gpu_eval['full_package'] = copy.deepcopy(gpu_res_pkg_item)
                time_of_overall_best_abs = time.time()
            
            origin_cpu_id_feedback = gpu_res_pkg_item['origin_cpu_id']
            feedback_payload_to_cpu = {'config': gpu_res_pkg_item['config_screened'], 'fitness': fitness, 
                                       'test_acc': gpu_res_pkg_item['test_acc'], 'train_acc': gpu_res_pkg_item['train_acc']}
            cpu_ctrl_queues[origin_cpu_id_feedback].put(("GPU_RESULT", feedback_payload_to_cpu))

        if overall_best_gpu_eval['full_package']:
            best_cfg_for_pbt_broadcast = overall_best_gpu_eval['full_package']['gpu_config_run']
            for i in range(num_actual_cpu_workers):
                cpu_ctrl_queues[i].put(("GLOBAL_BEST_CONFIG", 
                                         {'config': copy.deepcopy(best_cfg_for_pbt_broadcast), 
                                          'fitness': overall_best_gpu_eval['fitness']}))
        
        iter_end_abs_timestamp = time.time(); iter_end_mono_timestamp = time.monotonic()
        iter_duration_s_val = iter_end_mono_timestamp - iter_start_mono_timestamp
        total_hpo_cpu_screen_time += iter_total_cpu_screen_duration_this_iter
        total_hpo_gpu_eval_time += iter_total_gpu_eval_duration_this_iter

        iter_writer_a3.writerow([
            HPO_APPROACH_ID, RUN_ID, macro_iteration_actual_count,
            f"{iter_start_abs_timestamp:.3f}", f"{iter_end_abs_timestamp:.3f}", f"{iter_duration_s_val:.3f}",
            num_configs_screened_total_this_iter, num_promoted_to_gpu_this_iter, num_gpu_evaluated_this_iter,
            f"{iter_best_gpu_fitness_this_iter:.4f}", f"{overall_best_gpu_eval['fitness']:.4f}",
            f"{(time_of_overall_best_abs - hpo_overall_start_time_abs):.3f}",
            f"{iter_total_cpu_screen_duration_this_iter:.3f}", 
            f"{iter_total_gpu_eval_duration_this_iter:.3f}"
        ])
        iter_log_f_a3.flush()

        if overall_best_gpu_eval['full_package']:
            pkg_iter_best = overall_best_gpu_eval['full_package']
            print(f"Main: Iter {macro_iteration_actual_count} Current Overall Best Fit: {pkg_iter_best['fitness']:.4f} (CPU {pkg_iter_best['origin_cpu_id']})")
        else:
            print(f"Main: Iter {macro_iteration_actual_count} - No overall best package yet or no GPU results.")

    # --- HPO Run Finished ---
    print(f"\n=== HPO Run ({HPO_APPROACH_ID}, {RUN_ID}) Finished after {macro_iteration_actual_count} macro-iterations ===")
    
    # --- Cleanup Workers ---
    print("Main: Terminating persistent workers for Approach 3.")
    for i_term in range(num_actual_cpu_workers): cpu_ctrl_queues[i_term].put(None)
    gpu_submit_q.put(None)
    for p_term_cpu in cpu_procs: p_term_cpu.join(timeout=20)
    if gpu_proc: gpu_proc.join(timeout=20)
    for i_term_check, p_term_check_cpu in enumerate(cpu_procs): 
        if p_term_check_cpu.is_alive(): print(f"CPU{i_term_check} hanging, terminating."); p_term_check_cpu.terminate()
    if gpu_proc and gpu_proc.is_alive(): print("GPU hanging, terminating."); gpu_proc.terminate()
    # --- End Worker Cleanup ---

    hpo_total_wall_clock_duration_s = time.time() - hpo_overall_start_time_abs
    print(f"Total HPO wall-clock duration: {hpo_total_wall_clock_duration_s:.2f} seconds.")
    print(f"Overall best fitness achieved: {overall_best_gpu_eval['fitness']:.4f}")
    print(f"Achieved at wall-clock time: {(time_of_overall_best_abs - hpo_overall_start_time_abs):.2f}s from HPO start.")
    print(f"Approx total CPU screening time (sum of durations): {total_hpo_cpu_screen_time:.2f}s")
    print(f"Approx total GPU evaluation time (sum of durations): {total_hpo_gpu_eval_time:.2f}s")
    
    iter_log_f_a3.close() 
    print(f"\nLog files saved. Check filenames defined: {CONFIG_LOG_FILENAME_A3}, {CPU_SCREEN_LOG_FILENAME_A3}, {ITERATION_LOG_FILENAME_A3}")
    if overall_best_gpu_eval['full_package']:
        bp_final = overall_best_gpu_eval['full_package']
        print("\n=== Final Best Overall Result (Approach 3) ===")
        print(f"Fitness: {bp_final['fitness']:.4f} (Test: {bp_final['test_acc']:.4f}, Train: {bp_final['train_acc']:.4f})")
        print(f"Achieved in Macro Iteration: {bp_final['macro_iteration']}, by original CPU Worker: {bp_final['origin_cpu_id']}")
        print(f"Screened Config: {bp_final['config_screened']}")
        print(f"GPU Evaluated Config: {bp_final['gpu_config_run']}")
        print(f"GPU Eval Duration: {bp_final['gpu_duration_s']:.2f}s")
    else:
        print("No configurations yielded a best result.")