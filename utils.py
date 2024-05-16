import torch
import pickle
import numpy as np
import random
def load_dataset_and_preprocessors(base_path, exp, device):
    train_dataset = torch.load(f"{base_path}/train_dataset_exp{exp}.pt", map_location= device)
    val_dataset = torch.load(f"{base_path}/val_dataset_exp{exp}.pt", map_location= device)
    test_dataset = torch.load(f"{base_path}/test_dataset_exp{exp}.pt",map_location = device)
    
    with open(f"{base_path}/label_encoder_exp{exp}.pkl", 'rb') as f:
        label_encoder = pickle.load(f)

    return train_dataset, val_dataset, test_dataset, label_encoder


def set_random_seed(exp):
    torch.manual_seed(exp)
    torch.cuda.manual_seed(exp)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(exp)
    random.seed(exp)
    torch.cuda.manual_seed_all(exp)
# git test