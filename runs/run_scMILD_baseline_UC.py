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
# from utils import *
# from dataset import *
# from model import *
# from optimizer import *

from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc
base_path = 'data/UC'
ae_dir = f'{base_path}/AE/'

device_num = 4
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

    vae_batch_size = 256
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

    

    ae = AENB(input_dim=2000, latent_dim=ae_latent_dim, 
                            device=device, hidden_layers=ae_hidden_layers, 
                            activation_function=nn.Sigmoid).to(device)
    ae.load_state_dict(torch.load(f"{ae_dir}/aenb_{exp}.pth"))
    

    mil_latent_dim = 64
    # mil_learning_rate = 1e-3
    # mode 1
    # attention_module = AttentionModule(L=vae_latent_dim, D=vae_latent_dim //4 , K=1).to(device)
    # mode 2
    encoder_dim = ae_latent_dim
    model_encoder = ae.features
    # encoder_dim = 64
    # model_encoder = EncoderBranch(proto_vae, encoder_dim, activation_function=nn.LeakyReLU).to(device)
    attention_module = AttentionModule(L=encoder_dim, D=encoder_dim, K=1).to(device)
    # attention_module = GatedAttentionModule(L=encoder_dim, D=encoder_dim, K=1).to(device)

    model_teacher = TeacherBranch(input_dims = encoder_dim, latent_dims=mil_latent_dim, 
                            attention_module=attention_module, num_classes=2, activation_function=nn.Tanh)

    model_student = StudentBranch(input_dims = encoder_dim, latent_dims=mil_latent_dim, num_classes=2, activation_function=nn.Tanh)
     
    
    # model_encoder.to(device)
    model_teacher.to(device)
    model_student.to(device)
    teacher_learning_rate = 1e-3
    student_learning_rate = 1e-3
    encoder_learning_rate = 1e-3
    optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=teacher_learning_rate)

    optimizer_student = torch.optim.Adam(model_student.parameters(), lr=student_learning_rate)

    optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=encoder_learning_rate)
    ### 지금 이 아래 세팅이 best model 
    scMILD_epoch = 100
    scMILD_neg_weight = 0.3
    scMILD_stuOpt = 100
    scMILD_patience = 15
    add_suffix = "baseline_rep2"
    exp_writer = SummaryWriter(f'runs/UC')
    #default patience = 15 
    test_optimizer= Optimizer(exp, model_teacher, model_student, model_encoder, optimizer_teacher, optimizer_student, optimizer_encoder, bag_train_dl, bag_val_dl, bag_test_dl, instance_train_dl, instance_val_dl, instance_test_dl,  scMILD_epoch, device, val_combined_metric=False, stuOptPeriod=scMILD_stuOpt,stu_loss_weight_neg= scMILD_neg_weight, writer=exp_writer,
                              patience=scMILD_patience, csv=f'results/UC_ae_ed{encoder_dim}_md{mil_latent_dim}_lr{teacher_learning_rate}_{scMILD_epoch}_{scMILD_neg_weight}_{scMILD_stuOpt}_{scMILD_patience}_{add_suffix}.csv', saved_path=f'results/model_UC_ae_ed{encoder_dim}_md{mil_latent_dim}_lr{teacher_learning_rate}_{scMILD_epoch}_{scMILD_neg_weight}_{scMILD_stuOpt}_{scMILD_patience}_{add_suffix}',train_stud=False)
    test_optimizer.optimize()
    print(test_optimizer.evaluate_teacher(400, test=True))
    torch.cuda.empty_cache()
    del test_optimizer, model_teacher, model_student, model_encoder, optimizer_teacher, optimizer_student, optimizer_encoder, bag_train_dl, bag_val_dl, bag_test_dl, instance_train_dl, instance_val_dl, instance_test_dl, exp_writer
