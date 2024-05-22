
import sys
import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse
import json
# import copy
# import pandas
from src.utils import set_random_seed, load_dataset_and_preprocessors
from src.model import AENB
from src.trainer_ae import train_ae, test
from tqdm import tqdm as tdqm
import time

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

hyperparameter_file_path = os.path.join(target_dir, 'hyperparameters_ae.json')
if not os.path.exists(hyperparameter_file_path):
    with open(hyperparameter_file_path, 'w') as file:
        json.dump(hyperparameters_dict, file, indent=4)
        
device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
print("INFO: Using device: {}".format(device))


for exp in tdqm(exps,  desc="Experiment"):
    ################################## Load Dataset - Instance ###############
    train_dataset, val_dataset, test_dataset, label_encoder = load_dataset_and_preprocessors(base_path, exp, device)
    
    train_dl = DataLoader(train_dataset, batch_size=ae_batch_size, shuffle=False, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=round(ae_batch_size/2), shuffle=False, drop_last=False)
    test_dl = DataLoader(test_dataset, batch_size=round(ae_batch_size/2), shuffle=False, drop_last=False)
    del train_dataset, val_dataset, test_dataset
    set_random_seed(exp)
    ################################## Set Encoding Model ####################
    ae = AENB(input_dim=data_dim, latent_dim=ae_latent_dim, 
                        device=device, hidden_layers=ae_hidden_layers, 
                        activation_function=nn.Sigmoid).to(device)

    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=ae_learning_rate)
    ################################## Training VAE ####################
    
    ae = train_ae(ae, train_dl, val_dl, ae_optimizer, device, n_epochs=ae_epochs, patience= ae_patience, model_save_path=f"{target_dir}/aenb_{exp}.pth")


    test_recon = test(model=ae, optimizer=None, dataloader=test_dl, device=device, csv_path=f"{target_dir}/aenb_test.csv")
    del train_dl, val_dl, test_dl, ae, ae_optimizer, test_recon
    torch.cuda.empty_cache()
    

