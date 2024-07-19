import sys
import os
import torch
import argparse
import json
from src.trainer_ae import optimizer_ae
from tqdm import tqdm as tdqm

def pretraining_autoencoder(data_dir="data/MyData", device_num=6, ae_learning_rate=1e-3, ae_epochs=15, ae_patience=3, 
                            ae_latent_dim=128, ae_hidden_layers=[512, 256, 128], ae_batch_size=128, data_dim=2000, n_exp=8):
    base_path = data_dir
    target_dir = f'{base_path}/AE2/'
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    hyperparameters_dict = {
        "ae_learning_rate": ae_learning_rate,
        "ae_epochs": ae_epochs,
        "ae_patience": ae_patience,
        "ae_latent_dim": ae_latent_dim,
        "ae_hidden_layers": ae_hidden_layers,
        "ae_batch_size": ae_batch_size
    }

    ### Save hyperparameters
    hyperparameter_file_path = os.path.join(target_dir, 'hyperparameters_pretraining_autoencoder.json')
    if not os.path.exists(hyperparameter_file_path):
        with open(hyperparameter_file_path, 'w') as file:
            json.dump(hyperparameters_dict, file, indent=4)

    device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
    print("INFO: Using device: {}".format(device))

    exps = range(1, n_exp + 1)
    for exp in tdqm(exps, desc="Experiment"):
        optimizer_ae(base_path, exp, device, data_dim, ae_latent_dim, ae_hidden_layers,
                     ae_batch_size, ae_learning_rate, ae_epochs, ae_patience, target_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretraining Autoencoder')
    parser.add_argument('--data_dir', type=str, default="data/MyData", help='Data directory')
    parser.add_argument('--device_num', type=int, default=6, help='CUDA device number')
    parser.add_argument('--ae_learning_rate', type=float, default=1e-3, help='Learning rate for autoencoder')
    parser.add_argument('--ae_epochs', type=int, default=15, help='Number of epochs for autoencoder training')
    parser.add_argument('--ae_patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--ae_latent_dim', type=int, default=128, help='Latent dimension for autoencoder')
    parser.add_argument('--ae_hidden_layers', type=int, nargs='+', default=[512, 256, 128], help='Hidden layers for autoencoder')
    parser.add_argument('--ae_batch_size', type=int, default=128, help='Batch size for autoencoder')
    parser.add_argument('--data_dim', type=int, default=2000, help='Data dimension')
    parser.add_argument('--n_exp', type=int, default=8, help='Number of experiments')

    args = parser.parse_args()
    pretraining_autoencoder(**vars(args))