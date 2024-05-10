import torch
import pickle

def load_dataset_and_preprocessors(base_path, exp, device):
    train_dataset = torch.load(f"{base_path}/train_dataset_exp{exp}_HVG_count_noflt_only2.pt", map_location= device)
    val_dataset = torch.load(f"{base_path}/val_dataset_exp{exp}_HVG_count_noflt_only2.pt", map_location= device)
    test_dataset = torch.load(f"{base_path}/test_dataset_exp{exp}_HVG_count_noflt_only2.pt",map_location = device)
    
    with open(f"{base_path}/label_encoder_exp{exp}_HVG_count_noflt_only2.pkl", 'rb') as f:
        label_encoder = pickle.load(f)
    with open(f"{base_path}/scaler_exp{exp}_HVG_count_noflt_only2.pkl", 'rb') as f:
        scaler = pickle.load(f)

    return train_dataset, val_dataset, test_dataset, label_encoder, scaler

# git test