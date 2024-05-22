
import sys
import os 
import torch
import argparse
import json
from src.trainer_ae import optimizer_ae
from tqdm import tqdm as tdqm

dataset = "MyData"
base_path = f"data/{dataset}"
target_dir = f'{base_path}/AE/'

if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=False)

### Hyperparameters for the Autoencoder
device_num = 6

ae_learning_rate = 1e-3
ae_epochs = 25
ae_patience = 3
ae_latent_dim = 128
ae_hidden_layers = [512, 256, 128]
ae_batch_size = 128

data_dim = 2000
n_exp = 8

exps = range(1, n_exp + 1)


hyperparameters_dict = {
    "ae_learning_rate": ae_learning_rate,
    "ae_epochs": ae_epochs,
    "ae_patience": ae_patience,
    "ae_latent_dim": ae_latent_dim,
    "ae_hidden_layers": ae_hidden_layers,
    "ae_batch_size": ae_batch_size
}

### Save hyperparameters
hyperparameter_file_path = os.path.join(target_dir, 'hyperparameters_ae.json')
if not os.path.exists(hyperparameter_file_path):
    with open(hyperparameter_file_path, 'w') as file:
        json.dump(hyperparameters_dict, file, indent=4)
        
device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
print("INFO: Using device: {}".format(device))

for exp in tdqm(exps, desc="Experiment"):
    optimizer_ae(base_path, exp, device, data_dim, ae_latent_dim, ae_hidden_layers, ae_batch_size, ae_learning_rate, ae_epochs, ae_patience, target_dir)



