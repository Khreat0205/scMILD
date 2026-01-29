import torch
import argparse
from src.utils import load_ae_hyperparameters, load_and_process_datasets, load_dataloaders, load_model_and_optimizer
from src.optimizer import Optimizer

# from torch.utils.tensorboard import SummaryWriter

def train_scMILD(data_dir="data/MyData", dataset="MyData", device_num=6, data_dim=2000, mil_latent_dim=64, 
                 cell_batch_size=256, sample_learning_rate=1e-3, cell_learning_rate=1e-3, encoder_learning_rate=1e-3, 
                 scMILD_epoch=100, scMILD_neg_weight=0.3,scMILD_stuOpt=3, scMILD_patience=15, val_combined_metric = False, add_suffix="reported", 
                 n_exp=8, exp=None, opl= True, use_loss = True, op_lambda=0.5, train_stud=True, op_gamma=0.5, epoch_warmup=0,opl_warmup=0, opl_comps=[2]):
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
        
        instance_train_dl, bag_train, bag_val, bag_test = load_and_process_datasets(data_dir, exp, device, cell_batch_size)
        
        bag_train_dl, bag_val_dl, bag_test_dl = load_dataloaders(bag_train, bag_val, bag_test)
        print("loaded all dataset")
        del(bag_train, bag_val, bag_test)
        
        # Pass model type and VQ parameters to load_model_and_optimizer
        model_sample, model_cell, model_encoder, optimizer_sample, optimizer_cell, optimizer_encoder = load_model_and_optimizer(
            data_dim, ae_latent_dim, ae_hidden_layers, device, ae_dir, exp, mil_latent_dim,
            sample_learning_rate, cell_learning_rate, encoder_learning_rate,
            model_type=model_type, vq_num_codes=vq_num_codes, vq_commitment_weight=vq_commitment_weight
        )
        print(f"loaded all model and optimizer (using {model_type} encoder)")
        
        exp_writer = None
        test_optimizer= Optimizer(exp, model_sample, model_cell, model_encoder, 
                                  optimizer_sample, optimizer_cell, optimizer_encoder, 
                                  bag_train_dl, bag_val_dl, bag_test_dl, 
                                  instance_train_dl, 
                                  scMILD_epoch, 
                                  device, 
                                  val_combined_metric=val_combined_metric, 
                                  stuOptPeriod=scMILD_stuOpt,
                                  stu_loss_weight_neg= scMILD_neg_weight, 
                                  writer=exp_writer,
                                  patience=scMILD_patience, 
                                  csv=f'results/{dataset}_ae_ed{ae_latent_dim}_md{mil_latent_dim}_lr{sample_learning_rate}_{scMILD_epoch}_{scMILD_neg_weight}_{scMILD_stuOpt}_{scMILD_patience}_{add_suffix}.csv', 
                                  saved_path=f'results/{dataset}_model_ae_ed{ae_latent_dim}_md{mil_latent_dim}_lr{sample_learning_rate}_{scMILD_epoch}_{scMILD_neg_weight}_{scMILD_stuOpt}_{scMILD_patience}_{add_suffix}', 
                                  train_stud=train_stud,
                                  opl=opl,
                                  use_loss = use_loss,
                                  op_lambda=op_lambda,
                                  op_gamma=op_gamma, epoch_warmup=epoch_warmup,opl_warmup=opl_warmup,opl_comps=opl_comps)
        test_optimizer.optimize()
        # print(test_optimizer.evaluate_sample(400, test=True))
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
   parser.add_argument('--gmm', type=bool, default=False, help='Use GMM for cell')
   parser.add_argument('--use_loss', type=bool, default=True, help='Use loss for early stopping')
   parser.add_argument('--op_lambda', type=float, default=0.5, help='Lambda for orthogonal projection loss')
   parser.add_argument('--op_gamma', type=float, default=0.5, help='Gamma for orthogonal projection loss')

   args = parser.parse_args()
   torch.set_num_threads(16)
   train_scMILD(**vars(args))
