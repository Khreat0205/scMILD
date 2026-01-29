# 02-tuning_scMILD_conditional.py
import sys, os
import torch
sys.path.append(r"src")
from train_scMILD import train_scMILD
import itertools
from multiprocessing import Process
import subprocess
import argparse

# Usage:
# python 02-tuning_scMILD_conditional.py --dataset SCP1884 --gpus auto --n_gpus 4 --ratio_reg_lambda 0.05
# python 02-tuning_scMILD_conditional.py --dataset Skin3 --gpus auto --n_gpus 4 --ratio_reg_lambda 0.05
# python 02-tuning_scMILD_conditional.py --dataset SCP1884 --gpus auto --n_gpus 4 --ratio_reg_lambda 0.05

# v3 
# python 02-tuning_scMILD_conditional.py --dataset SCP1884 --gpus auto --n_gpus 1 --ratio_reg_lambda 0.1
# python 02-tuning_scMILD_conditional.py --dataset SCP1884 --gpus auto --n_gpus 1 --ratio_reg_lambda 0.0
# python 02-tuning_scMILD_conditional.py --dataset SCP1884 --gpus auto --n_gpus 1 --ratio_reg_lambda 0.05
# python 02-tuning_scMILD_conditional.py --dataset SCP1884 --gpus auto --n_gpus 1 --ratio_reg_lambda 0.2

# python 02-tuning_scMILD_conditional.py --dataset Skin3 --gpus auto --n_gpus 1 --ratio_reg_lambda 0.0
# python 02-tuning_scMILD_conditional.py --dataset Skin3 --gpus auto --n_gpus 1 --ratio_reg_lambda 0.05
# python 02-tuning_scMILD_conditional.py --dataset Skin3 --gpus auto --n_gpus 1 --ratio_reg_lambda 0.1 & python 02-tuning_scMILD_conditional.py --dataset Skin3 --gpus auto --n_gpus 1 --ratio_reg_lambda 0.2
# python 02-tuning_scMILD_conditional.py --dataset SCP1884 --gpus auto --n_gpus 4
# python 02-tuning_scMILD_conditional.py --dataset Skin3 --gpus auto --n_gpus 4

def get_free_gpus(n=4):
    """Get n GPUs with most free memory"""
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'], 
                          capture_output=True, text=True)
    free_memory = [(int(mem), idx) for idx, mem in enumerate(result.stdout.strip().split('\n'))]
    free_memory.sort(reverse=True)
    return [idx for _, idx in free_memory[:n]]

def run_combinations(dataset_suffix, gpu_id, combinations, param_keys, conditional_ae_dir, res_dir):
    """Run specific combinations on one GPU"""
    torch.set_num_threads(16)
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(gpu_id)
    
    # Classification용 데이터는 기존 경로
    data_dir = 'data_quantized_all_datasets/'  # 기존 classification dataset
    prefix_dataset = f'scMILDQ_all'
    n_gene = 6000
    os.makedirs(res_dir, exist_ok=True)
    
    print(f"[GPU {gpu_id}] Processing {len(combinations)} combinations for {dataset_suffix}")
    
    print(f"[GPU {gpu_id}] Classification data: {data_dir}")
    print(f"[GPU {gpu_id}] Conditional AE: {conditional_ae_dir}")
    
    for i, combo in enumerate(combinations, 1):
        params = dict(zip(param_keys, combo))
        print(f"[GPU {gpu_id}] [{i}/{len(combinations)}] Running combination")
        
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
                res_dir=res_dir,
                freeze_encoder=True,
                use_whole_ae=True,
                use_projection=True,
                use_conditional_ae=True,
                conditional_ae_dir=conditional_ae_dir,
                ratio_reg_lambda=params['ratio_reg_lambda'],
            )
            print(f"[GPU {gpu_id}] ✓ Completed {i}/{len(combinations)}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[GPU {gpu_id}] ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SCP1884', help='Dataset suffix')
    parser.add_argument('--gpus', type=str, default='auto', 
                       help='GPU IDs comma-separated (e.g., "0,1,2,3") or "auto" for automatic selection')
    parser.add_argument('--n_gpus', type=int, default=4, 
                       help='Number of GPUs to use when gpus="auto"')
    # parser.add_argument('--ratio_reg_lambda', type=float, default=0.05,
    #                    help='Disease ratio regularization strength (0 to disable)')
    parser.add_argument('--conditional_ae_dir', type=str, default='data_conditional/',
                       help='Directory with conditional VQ-AENB')
    args = parser.parse_args()
    
    dataset_suffix = args.dataset
    conditional_ae_dir = args.conditional_ae_dir
    # res_dir = f'/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_{dataset_suffix}/'
    # res_dir = f'/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_{dataset_suffix}_v2/'
    res_dir = f'/home/bmi-user/workspace/data/HSvsCD/scMILDQ/res_scMILDQ_conditional_v3_{dataset_suffix}/'
    
    
    # Hyperparameters
    # hyperparams = {
    #     'mil_latent_dim': [64, 128],
    #     'cell_batch_size': [1024],
    #     'sample_learning_rate': [1e-4, 5e-4, 1e-3],
    #     'cell_learning_rate': [5e-4, 1e-3],
    #     'scMILD_stuOpt': [1],
    #     'opl': [True]
    # }
    # epoch_patience_pairs = [(30, 5)]
    # Hyperparameters V2 for SCP1884
    # hyperparams = {
    #     'mil_latent_dim': [128],
    #     'cell_batch_size': [1024],
    #     'sample_learning_rate': [1e-4, 5e-4],
    #     'cell_learning_rate': [5e-4, 1e-3],
    #     'scMILD_stuOpt': [1],
    #     'opl': [True]
    # }
    # epoch_patience_pairs = [(100, 15)]
    # v2의 Best Performance Fixed (고정)
    # hyperparams = {
    #     'mil_latent_dim': [128],
    #     'cell_batch_size': [1024], 
    #     'sample_learning_rate': [1e-4], # ✅ Winner
    #     'cell_learning_rate': [5e-4],   # ✅ Winner
    #     'scMILD_stuOpt': [1],
    #     'opl': [True],
    #     'ratio_reg_lambda': [0.0, 0.2, 0.4]
    # }
    # epoch_patience_pairs = [(100, 15)]
    # v3 다양하게     
    hyperparams = {
        'mil_latent_dim': [32, 64, 128],
        'cell_batch_size': [1024], 
        'sample_learning_rate': [1e-4, 5e-4, 1e-3], 
        'cell_learning_rate': [5e-4],   
        'scMILD_stuOpt': [1],
        'opl': [True],
        'ratio_reg_lambda': [0.0, 0.2, 0.4, 0.6]
    }
    epoch_patience_pairs = [(30, 5)]
    
    
    
    # Generate all combinations
    base_combinations = list(itertools.product(*hyperparams.values()))
    all_combinations = []
    
    for base_combo in base_combinations:
        for epoch, patience in epoch_patience_pairs:
            full_combo = base_combo + (epoch, patience)
            all_combinations.append(full_combo)
    
    param_keys = list(hyperparams.keys()) + ['scMILD_epoch', 'scMILD_patience']
    
    print(f"="*60)
    print(f"Conditional scMILD Training")
    print(f"="*60)
    print(f"Dataset: {dataset_suffix}")
    # print(f"Data dir: {data_dir}")
    print(f"Results dir: {res_dir}")
    # print(f"Ratio reg lambda: {args.ratio_reg_lambda}")
    print(f"Total combinations: {len(all_combinations)}")
    print(f"="*60)
    
    # Get GPUs
    if args.gpus == 'auto':
        available_gpus = get_free_gpus(args.n_gpus)
        print(f"Auto-selected GPUs: {available_gpus}")
    else:
        available_gpus = [int(g.strip()) for g in args.gpus.split(',')]
        print(f"Using specified GPUs: {available_gpus}")
    
    n_gpus = len(available_gpus)
    
    # Split combinations across GPUs
    n_per_gpu = len(all_combinations) // n_gpus
    remainder = len(all_combinations) % n_gpus
    
    processes = []
    start_idx = 0
    
    for i, gpu_id in enumerate(available_gpus):
        n_combos = n_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + n_combos
        gpu_combinations = all_combinations[start_idx:end_idx]
        
        print(f"GPU {gpu_id}: {len(gpu_combinations)} combinations")
        
        # p = Process(target=run_combinations, 
        #        args=(dataset_suffix, gpu_id, gpu_combinations, param_keys, 
        #              args.ratio_reg_lambda, conditional_ae_dir, res_dir))
        p = Process(target=run_combinations, 
               args=(dataset_suffix, gpu_id, gpu_combinations, param_keys, 
                    conditional_ae_dir, res_dir))
        p.start()
        processes.append(p)
        
        start_idx = end_idx
    
    for p in processes:
        p.join()
    
    print(f"\n✅ All experiments for {dataset_suffix} completed!")