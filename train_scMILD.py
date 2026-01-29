import torch
import argparse
from src.utils import load_ae_hyperparameters, load_and_process_datasets, load_dataloaders, load_model_and_optimizer
import os
from src.optimizer import Optimizer

# from torch.utils.tensorboard import SummaryWriter
def train_scMILD(data_dir="data/MyData", dataset="MyData", device_num=6, data_dim=2000, mil_latent_dim=64, 
                 cell_batch_size=256, sample_learning_rate=1e-3, cell_learning_rate=1e-3, encoder_learning_rate=1e-3, 
                 scMILD_epoch=100, scMILD_neg_weight=0.3, scMILD_stuOpt=3, scMILD_patience=15, val_combined_metric=False, 
                 add_suffix="reported", n_exp=8, exp=None, opl=False, use_loss=False, op_lambda=0.5, train_stud=True, 
                 op_gamma=0.5, epoch_warmup=0, opl_warmup=0, opl_comps=[2], res_dir="results", dataset_suffix=None, 
                 freeze_encoder=False, use_whole_ae=False, use_projection=False, 
                 use_conditional_ae=False, conditional_ae_dir=None, ratio_reg_lambda=0.0):
    
    # Conditional AE 사용 시 별도 경로
    if use_conditional_ae and conditional_ae_dir:
        ae_dir = f'{conditional_ae_dir}/AE/'
        print(f"INFO: Using Conditional VQ-AENB from {ae_dir}")
    else:
        ae_dir = f'{data_dir}/AE/'

    # Load hyperparameters including model type information
    ae_latent_dim, ae_hidden_layers, model_type, vq_num_codes, vq_commitment_weight = load_ae_hyperparameters(ae_dir)

    device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: Using device: {device}")
    print(f"INFO: Detected pretrained model type: {model_type}")
    if model_type == 'VQ-AENB':
        print(f"INFO: VQ-AENB configuration - num_codes: {vq_num_codes}, commitment_weight: {vq_commitment_weight}")

    if exp is None:
        exps = range(1, n_exp + 1)
    else:
        exps = [exp]

    for exp in exps:
        print(f'Experiment {exp}')
        
        # [MODIFIED] use_conditional_ae, conditional_ae_dir 전달
        instance_train_dl, bag_train, bag_val, bag_test = load_and_process_datasets(
            data_dir, exp, device, cell_batch_size, 
            suffix=dataset_suffix,
            use_conditional_ae=use_conditional_ae,
            conditional_ae_dir=conditional_ae_dir
        )
        
        bag_train_dl, bag_val_dl, bag_test_dl = load_dataloaders(bag_train, bag_val, bag_test)
        print("loaded all dataset")
        del bag_train, bag_val, bag_test
        
        # Pass model type and VQ parameters to load_model_and_optimizer
        model_sample, model_cell, model_encoder, optimizer_sample, optimizer_cell, optimizer_encoder = load_model_and_optimizer(
            data_dim, ae_latent_dim, ae_hidden_layers, device, ae_dir, exp, mil_latent_dim,
            sample_learning_rate, cell_learning_rate, encoder_learning_rate,
            model_type=model_type, vq_num_codes=vq_num_codes, vq_commitment_weight=vq_commitment_weight,
            freeze_encoder=freeze_encoder, use_whole_ae=use_whole_ae, use_projection=use_projection
        )
        
        # freeze_encoder면 student 학습 무의미
        if freeze_encoder and not use_projection:
            train_stud = False
            print("INFO: train_stud automatically set to False (encoder frozen, no projection)")
        elif freeze_encoder and use_projection:
            print("INFO: Encoder frozen but projection layer will be trained")
        print(f"loaded all model and optimizer (using {model_type} encoder)")
        
        exp_writer = None
        
        # Disease ratio 로드 (ratio regularization용)
        code_disease_ratio = None
        if ratio_reg_lambda > 0 and conditional_ae_dir:
            import json
            ratio_path = f'{conditional_ae_dir}/disease_ratio_exp{exp}_{dataset_suffix}.json'
            if os.path.exists(ratio_path):
                with open(ratio_path, 'r') as f:
                    code_disease_ratio = {int(k): v for k, v in json.load(f).items()}
                print(f"Loaded disease ratio for {len(code_disease_ratio)} codes (exp {exp})")
            else:
                print(f"WARNING: {ratio_path} not found, ratio regularization disabled")
        
        # [MODIFIED] use_conditional_ae 전달
        test_optimizer = Optimizer(
            exp, model_sample, model_cell, model_encoder,
            optimizer_sample, optimizer_cell, optimizer_encoder,
            bag_train_dl, bag_val_dl, bag_test_dl,
            instance_train_dl,
            scMILD_epoch,
            device,
            val_combined_metric=val_combined_metric,
            stuOptPeriod=scMILD_stuOpt,
            stu_loss_weight_neg=scMILD_neg_weight,
            writer=exp_writer,
            patience=scMILD_patience,
            csv=f'{res_dir}/{dataset}_model_ae_ed{ae_latent_dim}_md{mil_latent_dim}_cb{cell_batch_size}_lr{sample_learning_rate}_elr{cell_learning_rate}_epoch{scMILD_epoch}_{scMILD_stuOpt}_{scMILD_patience}_{add_suffix}_useopl{opl}_lam{ratio_reg_lambda}.csv',
            saved_path=f'{res_dir}/{dataset}_model_ae_ed{ae_latent_dim}_md{mil_latent_dim}_cb{cell_batch_size}_lr{sample_learning_rate}_elr{cell_learning_rate}_epoch{scMILD_epoch}_{scMILD_stuOpt}_{scMILD_patience}_{add_suffix}_useopl{opl}_lam{ratio_reg_lambda}',
            train_stud=train_stud,
            opl=opl,
            use_loss=use_loss,
            op_lambda=op_lambda,
            op_gamma=op_gamma,
            epoch_warmup=epoch_warmup,
            opl_warmup=opl_warmup,
            opl_comps=opl_comps,
            code_disease_ratio=code_disease_ratio,
            ratio_reg_lambda=ratio_reg_lambda,
            use_conditional_ae=use_conditional_ae
        )
        test_optimizer.optimize()
        torch.cuda.empty_cache()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scMILD Training')
    parser.add_argument('--data_dir', type=str, default="data/MyData", help='Data directory')
    parser.add_argument('--dataset', type=str, default="MyData", help='Dataset name')
    parser.add_argument('--device_num', type=int, default=6, help='CUDA device number')
    parser.add_argument('--data_dim', type=int, default=2000, help='Data dimension')
    parser.add_argument('--mil_latent_dim', type=int, default=64, help='Latent dimension for MIL')
    parser.add_argument('--cell_batch_size', type=int, default=256, help='Batch size for cell')
    parser.add_argument('--sample_learning_rate', type=float, default=1e-3, help='Learning rate for sample')
    parser.add_argument('--cell_learning_rate', type=float, default=1e-3, help='Learning rate for cell')
    parser.add_argument('--encoder_learning_rate', type=float, default=1e-3, help='Learning rate for encoder')
    parser.add_argument('--scMILD_epoch', type=int, default=500, help='Number of epochs for scMILD')
    parser.add_argument('--scMILD_neg_weight', type=float, default=0.3, help='Negative weight for scMILD')
    parser.add_argument('--scMILD_stuOpt', type=int, default=3, help='cell optimization period')
    parser.add_argument('--scMILD_patience', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--val_combined_metric', type=bool, default=False, help='Use combined metric for validation')
    parser.add_argument('--add_suffix', type=str, default="reported", help='Suffix for output files')
    parser.add_argument('--n_exp', type=int, default=8, help='Number of experiments')
    parser.add_argument('--exp', type=int, default=None, help='Experiment number (if None, all experiments will be run)')
    parser.add_argument('--opl', type=bool, default=False, help='Use OPL for cell')
    parser.add_argument('--use_loss', type=bool, default=True, help='Use loss for early stopping')
    parser.add_argument('--op_lambda', type=float, default=0.5, help='Lambda for orthogonal projection loss')
    parser.add_argument('--op_gamma', type=float, default=0.5, help='Gamma for orthogonal projection loss')
    parser.add_argument('--res_dir', type=str, default="results", help='Results directory')
    parser.add_argument('--dataset_suffix', type=str, default=None, help='Dataset suffix')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder')
    parser.add_argument('--use_whole_ae', action='store_true', help='Use whole-data pretrained AE')
    parser.add_argument('--use_projection', action='store_true', help='Use projection layer after encoder')
    
    # [NEW] Conditional AE 관련 인자
    parser.add_argument('--use_conditional_ae', action='store_true', help='Use Conditional VQ-AENB')
    parser.add_argument('--conditional_ae_dir', type=str, default=None, help='Directory for conditional AE files')
    parser.add_argument('--ratio_reg_lambda', type=float, default=0.0, help='Lambda for disease ratio regularization')
    
    args = parser.parse_args()
    torch.set_num_threads(32)
    train_scMILD(**vars(args))