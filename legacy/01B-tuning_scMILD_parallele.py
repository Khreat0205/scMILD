import sys, os
import torch
sys.path.append(r"src")
from train_scMILD import train_scMILD
import itertools
from datetime import datetime
from multiprocessing import Process
import subprocess


# data_dir = 'data_quantized_all_datasets/'
# prefix_dataset = f'scMILDQ_all'
# dataset_suffices = ['SCP1884', 'Skin3', 'Skin2', 'PCD', 'Skin']
# n_gene = 6000
# device_num = 5
# device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
# torch.set_num_threads(64)
# print("INFO: Using device: {}".format(device))

# ## hyperparameter set
# hyperparams = {
#     'mil_latent_dim': [32, 64, 128],
#     'cell_batch_size': [1024],
#     'sample_learning_rate': [1e-4, 5e-4, 1e-3],
#     'cell_learning_rate': [5e-4, 1e-3],
#     'scMILD_stuOpt': [3, 1]
#     'opl': [True]
# }

# # 2차 실험을 위한 수정된 Paired epoch and patience
# epoch_patience_pairs = [
#     (30, 5),
#     (70, 10),
# ]

# # Generate combinations
# base_combinations = list(itertools.product(*hyperparams.values()))
# all_combinations = []

# for base_combo in base_combinations:
#     mil_dim, batch_size, sample_lr, cell_lr, stuOpt = base_combo
    
#     for epoch, patience in epoch_patience_pairs:        
#         full_combo = base_combo + (epoch, patience)
#         all_combinations.append(full_combo)

# param_keys = list(hyperparams.keys()) + ['scMILD_epoch', 'scMILD_patience']
# print(f"Total combinations: {len(all_combinations)}")


# for dataset_suffix in dataset_suffices:
#     res_dir = f'/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_{dataset_suffix}/'
#     os.makedirs(res_dir, exist_ok=True)
    
#     # Run experiments
#     for i, combo in enumerate(all_combinations, 1):
#         params = dict(zip(param_keys, combo))
        
#         print(f"\n[{i}/{len(all_combinations)}] Running with params:")
#         print(params)
        
#         try:
#             train_scMILD(
#                 data_dir=data_dir,
#                 dataset=f"{prefix_dataset}",
#                 device_num=device_num,
#                 data_dim=n_gene,
#                 mil_latent_dim=params['mil_latent_dim'],
#                 cell_batch_size=params['cell_batch_size'],
#                 sample_learning_rate=params['sample_learning_rate'],
#                 cell_learning_rate=params['cell_learning_rate'],
#                 scMILD_epoch=params['scMILD_epoch'],
#                 scMILD_stuOpt=params['scMILD_stuOpt'],
#                 scMILD_patience=params['scMILD_patience'],
#                 opl = params['opl'],
#                 use_loss= True,
#                 train_stud=True,
#                 dataset_suffix=dataset_suffix,
#                 res_dir=res_dir
#             )
#             print(f"✓ Completed {i}/{len(all_combinations)}")
#             torch.cuda.empty_cache()
#         except Exception as e:
#             print(f"✗ Failed {i}/{len(all_combinations)}: {e}")
#             torch.cuda.empty_cache()
#             continue
    


def get_free_gpu():
    """Get GPU with most free memory"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                          capture_output=True, text=True)
    free_memory = [int(x) for x in result.stdout.strip().split('\n')]
    return free_memory.index(max(free_memory))

def run_dataset(dataset_suffix, gpu_id, all_combinations, param_keys):
    """Run experiments for one dataset on specific GPU"""
    torch.set_num_threads(16)  # 각 프로세
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    
    data_dir = 'data_quantized_all_datasets/'
    prefix_dataset = f'scMILDQ_all'
    n_gene = 6000
    res_dir = f'/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_all_{dataset_suffix}/'
    os.makedirs(res_dir, exist_ok=True)
    
    print(f"[GPU {gpu_id}] Processing {dataset_suffix}")
    
    for i, combo in enumerate(all_combinations, 1):
        params = dict(zip(param_keys, combo))
        print(f"[GPU {gpu_id}] [{i}/{len(all_combinations)}] {dataset_suffix}")
        
        try:
            train_scMILD(
                data_dir=data_dir,
                dataset=prefix_dataset,
                device_num=gpu_id,
                data_dim=n_gene,
                mil_latent_dim=params['mil_latent_dim'],
                cell_batch_size=params['cell_batch_size'],
                sample_learning_rate=params['sample_learning_rate'],
                cell_learning_rate=params['cell_learning_rate'],
                scMILD_epoch=params['scMILD_epoch'],
                scMILD_stuOpt=params['scMILD_stuOpt'],
                scMILD_patience=params['scMILD_patience'],
                opl=params['opl'],
                use_loss=True,
                train_stud=True,
                dataset_suffix=dataset_suffix,
                res_dir=res_dir
            )
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[GPU {gpu_id}] Failed: {e}")
            torch.cuda.empty_cache()
            continue


if __name__ == '__main__':
    # Setup hyperparameters
    hyperparams = {
        'mil_latent_dim': [32, 64, 128],
        'cell_batch_size': [1024],
        'sample_learning_rate': [1e-4, 5e-4, 1e-3],
        'cell_learning_rate': [5e-4, 1e-3],
        'scMILD_stuOpt': [1],
        'opl': [True]
    }
    
    epoch_patience_pairs = [(30, 5), (70, 10)]
    
    # Generate combinations
    base_combinations = list(itertools.product(*hyperparams.values()))
    all_combinations = []
    
    for base_combo in base_combinations:
        for epoch, patience in epoch_patience_pairs:
            full_combo = base_combo + (epoch, patience)
            all_combinations.append(full_combo)
    
    param_keys = list(hyperparams.keys()) + ['scMILD_epoch', 'scMILD_patience']
    print(f"Total combinations: {len(all_combinations)}")
    
    # Dataset and GPU assignment
    dataset_suffices = ['SCP1884', 'Skin3', 'Skin2', 'PCD', 'Skin']
    gpu_assignments = [7, 1, 1, 6, 5]  # Assign specific GPUs
    # dataset_suffices = ['SCP1884', 'Skin3', 'Skin2','PCD']
    # gpu_assignments = [5, 1, 4]  # Assign specific GPUs
    torch.set_num_threads(16*5)

    # Create processes
    processes = []
    for dataset, gpu_id in zip(dataset_suffices, gpu_assignments):
        p = Process(target=run_dataset, args=(dataset, gpu_id, all_combinations, param_keys))
        p.start()
        processes.append(p)
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    print("All experiments completed!")

