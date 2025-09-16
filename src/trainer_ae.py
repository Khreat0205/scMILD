import os
import torch
from torch import nn
import time
import copy
import pandas
from torch.utils.data import DataLoader
from src.utils import load_dataset_and_preprocessors, set_random_seed
from src.model import AENB, VQ_AENB

def negative_binomial_loss(y_pred, theta, y_true, eps=1e-10):
    # adapted from https://github.com/uhlerlab/STACI/blob/master/gae/gae/optimizer.py
    nbloss1=torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
    nbloss2=(theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
    nbloss=nbloss1+nbloss2
    return torch.mean(nbloss)

def _train_or_test(model=None, dataloader=None, optimizer=None, device='cuda'):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    # Check if model is VQ-AENB
    is_vq_model = isinstance(model, VQ_AENB)

    start = time.time()
    
    n_batches = 0
    
    total_recons_loss = 0
    total_commitment_loss = 0 if is_vq_model else None
    
    for i, (data, _, instance_labels) in enumerate(dataloader):
        input_data = data.to(device)
        target = instance_labels.to(device)
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # Handle VQ-AENB vs AENB forward pass
            if is_vq_model:
                forward_output = model(input_data, target, is_train=is_train)
                if is_train:
                    mu_recon, theta_recon, commitment_loss = forward_output
                    total_commitment_loss += commitment_loss.item()
                else:
                    mu_recon, theta_recon = forward_output
                    commitment_loss = 0
            else:
                mu_recon, theta_recon = model(input_data, target, is_train=is_train)
                commitment_loss = 0
            
            recons_loss = negative_binomial_loss(mu_recon, theta_recon, input_data)
            
            # Accumulate loss values
            n_batches += 1
            
            total_recons_loss += recons_loss.item()
            
            if is_train:
                # Combine losses for VQ-AENB
                if is_vq_model:
                    total_loss = recons_loss + commitment_loss
                else:
                    total_loss = recons_loss
                    
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            del input_data
            del mu_recon
            del theta_recon

    end = time.time()
    
    avg_recons_loss = total_recons_loss / n_batches
    print(f'\tTime: {end - start:.2f}s')
    print(f'\tReconstruction Loss: {avg_recons_loss:.4f}')
    
    if is_vq_model and is_train:
        avg_commitment_loss = total_commitment_loss / n_batches
        print(f'\tCommitment Loss: {avg_commitment_loss:.4f}')
        print(f'\tTotal Loss: {avg_recons_loss + avg_commitment_loss:.4f}')
    
    return avg_recons_loss


def train(model, optimizer=None, dataloader=None, device='cuda'):
    assert(optimizer is not None)
    print('\ttrain')
    model.train()
    return _train_or_test(model, optimizer=optimizer, dataloader=dataloader,device=device)


def test(model, optimizer=None, dataloader=None, device='cuda', csv_path=None):
    print('\ttest')
    model.eval()
    performance_metrics = _train_or_test(model, optimizer=optimizer, dataloader=dataloader,device=device)
    if csv_path is not None:
        data = {
            'Reconstruction Loss': [performance_metrics],
        }
        df = pandas.DataFrame(data)
        if not os.path.isfile(csv_path):
            df.to_csv(csv_path, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)
            
    return performance_metrics


def train_ae(ae, train_dl, val_dl, optimizer, device, n_epochs=25, patience= 10, model_save_path='best_ae.pth'):
    best_loss = float('inf')
    no_improvement = 0
    
    for epoch in range(n_epochs):
        print('epoch: \t{0}'.format(epoch))
        train_recon = train(model=ae, dataloader=train_dl,optimizer=optimizer, device=device)
        # train_recon_addition = train(model=ae, dataloader=train_dl_addition,optimizer=optimizer, device=device)
        print('\n')
        val_recon = test(model=ae, dataloader=val_dl, device=device)
        avg_val_loss = val_recon
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(ae.state_dict())
            torch.save(best_model_wts, model_save_path)
            no_improvement = 0
            
        else:
            no_improvement += 1
            if no_improvement == patience:
                break
    ae.load_state_dict(best_model_wts)  
    
    return ae


def optimizer_ae(base_path, exp, device, data_dim, ae_latent_dim, ae_hidden_layers, ae_batch_size,
                 ae_learning_rate, ae_epochs, ae_patience, target_dir, model_type='AENB',
                 vq_num_codes=256, vq_commitment_weight=0.25):
    ################################## Load Dataset - Instance ###############
    train_dataset, val_dataset, test_dataset, label_encoder = load_dataset_and_preprocessors(base_path, exp, device)
    
    # train_dl = DataLoader(train_dataset, batch_size=ae_batch_size, shuffle=False, drop_last=False)
    train_dl = DataLoader(train_dataset, batch_size=ae_batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_dataset, batch_size=round(ae_batch_size/2), shuffle=False, drop_last=False)
    test_dl = DataLoader(test_dataset, batch_size=round(ae_batch_size/2), shuffle=False, drop_last=False)
    del train_dataset, val_dataset, test_dataset
    set_random_seed(exp)

    ################################## Set Model ####################
    if model_type == 'VQ-AENB':
        ae = VQ_AENB(input_dim=data_dim, latent_dim=ae_latent_dim,
                     device=device, hidden_layers=ae_hidden_layers,
                     num_codes=vq_num_codes, commitment_weight=vq_commitment_weight,
                     activation_function=nn.Sigmoid).to(device)
        
        # Initialize codebook with training data
        print(f"Initializing VQ-AENB codebook with {vq_num_codes} codes...")
        ae.init_codebook(train_dl, method="kmeans", num_samples=min(10000, len(train_dl.dataset)))
        
        model_save_name = f"vq_aenb_{exp}.pth"
        csv_name = f"vq_aenb_test.csv"
    else:
        ae = AENB(input_dim=data_dim, latent_dim=ae_latent_dim,
                  device=device, hidden_layers=ae_hidden_layers,
                  activation_function=nn.Sigmoid).to(device)
        
        model_save_name = f"aenb_{exp}.pth"
        csv_name = f"aenb_test.csv"

    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=ae_learning_rate)

    ################################## Training AE ####################
    ae = train_ae(ae, train_dl, val_dl, ae_optimizer, device, n_epochs=ae_epochs,
                  patience=ae_patience, model_save_path=f"{target_dir}/{model_save_name}")

    test_recon = test(model=ae, optimizer=None, dataloader=test_dl, device=device,
                      csv_path=f"{target_dir}/{csv_name}")
    
    # Print codebook usage statistics for VQ-AENB
    if model_type == 'VQ-AENB':
        codebook_stats = ae.get_codebook_usage()
        print(f"Codebook usage: {codebook_stats['num_active']}/{codebook_stats['total_codes']} codes active")
    
    del train_dl, val_dl, test_dl, ae, ae_optimizer, test_recon
    torch.cuda.empty_cache()