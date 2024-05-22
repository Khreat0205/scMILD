import sys
import os 
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
import random
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.dataset import MilDataset, InstanceDataset, InstanceDataset2, collate, update_instance_labels_with_bag_labels
from src.utils import set_random_seed, load_dataset_and_preprocessors
from src.model import AENB, AttentionModule, TeacherBranch, StudentBranch
from src.optimizer import Optimizer

from torch.utils.tensorboard import SummaryWriter
base_path = 'data/PBMC'
ae_dir = f'{base_path}/AE/'

device_num = 3
device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
print("INFO: Using device: {}".format(device))

hyperparameter_file_path = os.path.join(ae_dir, f'hyperparameters_ae.json')

with open(hyperparameter_file_path, 'r') as file:
    loaded_args_dict = json.load(file)       
    
loaded_args = argparse.Namespace(**loaded_args_dict)


ae_learning_rate = loaded_args.ae_learning_rate
ae_epochs = loaded_args.ae_epochs
ae_patience = loaded_args.ae_patience
ae_latent_dim = loaded_args.ae_latent_dim
ae_hidden_layers = loaded_args.ae_hidden_layers
ae_batch_size = loaded_args.ae_batch_size


for exp in range(1, 9):
    print(f'Experiment {exp}')
    train_dataset, val_dataset, test_dataset, label_encoder = load_dataset_and_preprocessors(base_path, exp, device)

    instance_train_dataset = update_instance_labels_with_bag_labels(train_dataset, device=device)
    instance_val_dataset = update_instance_labels_with_bag_labels(val_dataset, device=device)
    instance_test_dataset = update_instance_labels_with_bag_labels(test_dataset, device=device)
    
    
    set_random_seed(exp)
    
    vae_batch_size = 512
    instance_train_dl = DataLoader(instance_train_dataset, batch_size=vae_batch_size, shuffle=True, drop_last=False)
    instance_val_dl = DataLoader(instance_val_dataset, batch_size=vae_batch_size, shuffle=True, drop_last=False)
    instance_test_dl = DataLoader(instance_test_dataset, batch_size=round(vae_batch_size/2), shuffle=False, drop_last=False)
    del instance_train_dataset, instance_val_dataset, instance_test_dataset
    
    bag_train = MilDataset(train_dataset.data.to(device), train_dataset.ids.unsqueeze(0).to(device), train_dataset.labels.to(device), train_dataset.instance_labels.to(device))
    bag_val = MilDataset(val_dataset.data.to(device), val_dataset.ids.unsqueeze(0).to(device), val_dataset.labels.to(device), val_dataset.instance_labels.to(device))
    bag_test = MilDataset(test_dataset.data.to(device), test_dataset.ids.unsqueeze(0).to(device), test_dataset.labels.to(device), test_dataset.instance_labels.to(device))
    del train_dataset, val_dataset, test_dataset
    
    bag_train_dl = DataLoader(bag_train,batch_size = 14, shuffle=False, drop_last=False,collate_fn=collate)
    bag_val_dl = DataLoader(bag_val,batch_size = 15, shuffle=False, drop_last=False,collate_fn=collate)
    bag_test_dl = DataLoader(bag_test,batch_size = 15, shuffle=False, drop_last=False,collate_fn=collate)
    del bag_train, bag_val, bag_test
    print("loaded all dataset")
    
    

    ae = AENB(input_dim=2000, latent_dim=ae_latent_dim, 
                            device=device, hidden_layers=ae_hidden_layers, 
                            activation_function=nn.Sigmoid).to(device)
    ae.load_state_dict(torch.load(f"{ae_dir}/aenb_{exp}.pth"))
    print("loaded pretrained autoencoder")

    mil_latent_dim = 64
    encoder_dim = ae_latent_dim
    model_encoder = ae.features
    attention_module = AttentionModule(L=encoder_dim, D=encoder_dim, K=1).to(device)

    model_teacher = TeacherBranch(input_dims = encoder_dim, latent_dims=mil_latent_dim, 
                            attention_module=attention_module, num_classes=2, activation_function=nn.Tanh)

    model_student = StudentBranch(input_dims = encoder_dim, latent_dims=mil_latent_dim, num_classes=2, activation_function=nn.Tanh)
     
    
    model_teacher.to(device)
    model_student.to(device)
    teacher_learning_rate = 1e-4
    student_learning_rate = 1e-3
    encoder_learning_rate = 1e-3
    optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=teacher_learning_rate)

    optimizer_student = torch.optim.Adam(model_student.parameters(), lr=student_learning_rate)

    optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=encoder_learning_rate)
    scMILD_epoch = 500
    scMILD_neg_weight = 0.1
    scMILD_stuOpt = 5
    scMILD_patience = 15
    add_suffix = "replicate"
    exp_writer = SummaryWriter(f'runs/PBMC')
    #default patience = 15 
    
    test_optimizer= Optimizer(exp, model_teacher, model_student, model_encoder, optimizer_teacher, optimizer_student, optimizer_encoder, bag_train_dl, bag_val_dl, bag_test_dl, instance_train_dl, instance_val_dl, instance_test_dl,  scMILD_epoch, device, val_combined_metric=False, stuOptPeriod=scMILD_stuOpt,stu_loss_weight_neg= scMILD_neg_weight, writer=exp_writer,
                              patience=scMILD_patience, csv=f'results/PBMC_ae_ed{encoder_dim}_md{mil_latent_dim}_lr{teacher_learning_rate}_{scMILD_epoch}_{scMILD_neg_weight}_{scMILD_stuOpt}_{scMILD_patience}_{add_suffix}.csv', saved_path=f'results/model_PBMC_ae_ed{encoder_dim}_md{mil_latent_dim}_lr{teacher_learning_rate}_{scMILD_epoch}_{scMILD_neg_weight}_{scMILD_stuOpt}_{scMILD_patience}_{add_suffix}')
    print("running scMILD")
    test_optimizer.optimize()
    
    print(test_optimizer.evaluate_teacher(400, test=True))
    torch.cuda.empty_cache()
    del test_optimizer, model_teacher, model_student, model_encoder, optimizer_teacher, optimizer_student, optimizer_encoder, bag_train_dl, bag_val_dl, bag_test_dl, instance_train_dl, instance_val_dl, instance_test_dl, exp_writer
