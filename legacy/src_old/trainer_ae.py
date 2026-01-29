import os
import torch
from torch import nn
import time
import copy
import pandas
import numpy as np
from torch.utils.data import DataLoader
from src.utils import load_dataset_and_preprocessors, set_random_seed, load_whole_dataset
from src.model import AENB, VQ_AENB, VQ_AENB_Conditional


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

def _train_or_test_conditional(model=None, dataloader=None, optimizer=None, device='cuda'):
    """Training/testing loop for Conditional VQ-AENB"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    start = time.time()
    n_batches = 0
    total_recons_loss = 0
    total_commitment_loss = 0

    for i, (data, study_ids, instance_labels) in enumerate(dataloader):
        input_data = data.to(device)
        study_ids = study_ids.to(device)
        target = instance_labels.to(device)
        
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            if is_train:
                mu_recon, theta_recon, commitment_loss = model(input_data, study_ids, target, is_train=True)
                total_commitment_loss += commitment_loss.item()
            else:
                mu_recon, theta_recon = model(input_data, study_ids, target, is_train=False)
                commitment_loss = 0
            
            recons_loss = negative_binomial_loss(mu_recon, theta_recon, input_data)
            n_batches += 1
            total_recons_loss += recons_loss.item()
            
            if is_train:
                total_loss = recons_loss + commitment_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            del input_data, mu_recon, theta_recon

    end = time.time()
    avg_recons_loss = total_recons_loss / n_batches
    
    print(f'\tTime: {end - start:.2f}s')
    print(f'\tReconstruction Loss: {avg_recons_loss:.4f}')
    
    if is_train:
        avg_commitment_loss = total_commitment_loss / n_batches
        print(f'\tCommitment Loss: {avg_commitment_loss:.4f}')
        print(f'\tTotal Loss: {avg_recons_loss + avg_commitment_loss:.4f}')
    
    return avg_recons_loss


def train_conditional(model, optimizer=None, dataloader=None, device='cuda'):
    assert(optimizer is not None)
    print('\ttrain')
    model.train()
    return _train_or_test_conditional(model, optimizer=optimizer, dataloader=dataloader, device=device)


def test_conditional(model, optimizer=None, dataloader=None, device='cuda'):
    print('\ttest')
    model.eval()
    return _train_or_test_conditional(model, optimizer=optimizer, dataloader=dataloader, device=device)

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


# def train_ae(ae, train_dl, val_dl, optimizer, device, n_epochs=25, patience= 10, model_save_path='best_ae.pth'):
#     best_loss = float('inf')
#     no_improvement = 0
    
#     for epoch in range(n_epochs):
#         print('epoch: \t{0}'.format(epoch))
#         train_recon = train(model=ae, dataloader=train_dl,optimizer=optimizer, device=device)
#         # train_recon_addition = train(model=ae, dataloader=train_dl_addition,optimizer=optimizer, device=device)
#         print('\n')
#         val_recon = test(model=ae, dataloader=val_dl, device=device)
#         avg_val_loss = val_recon
        
#         if avg_val_loss < best_loss:
#             best_loss = avg_val_loss
#             best_model_wts = copy.deepcopy(ae.state_dict())
#             torch.save(best_model_wts, model_save_path)
#             no_improvement = 0
            
#         else:
#             no_improvement += 1
#             if no_improvement == patience:
#                 break
#     ae.load_state_dict(best_model_wts)  
    
#     return ae

def train_ae(ae, train_dl, val_dl, optimizer, device, n_epochs=25, patience=10, 
             model_save_path='best_ae.pth', use_train_plateau=False, plateau_threshold=1e-4):
    """
    use_train_plateau: True면 val_dl 없이 train loss plateau로 early stopping
    plateau_threshold: loss 변화량이 이 값 이하면 plateau로 판단
    """
    best_loss = float('inf')
    no_improvement = 0
    prev_train_loss = float('inf')
    
    for epoch in range(n_epochs):
        print('epoch: \t{0}'.format(epoch))
        train_recon = train(model=ae, dataloader=train_dl, optimizer=optimizer, device=device)
        print('\n')
        
        if use_train_plateau:
            # Train loss plateau 기반 early stopping
            loss_change = abs(prev_train_loss - train_recon)
            if loss_change < plateau_threshold:
                no_improvement += 1
            else:
                no_improvement = 0
                best_model_wts = copy.deepcopy(ae.state_dict())
                torch.save(best_model_wts, model_save_path)
            prev_train_loss = train_recon
            
            if no_improvement >= patience:
                print(f"Train loss plateau reached at epoch {epoch}")
                break
        else:
            # 기존 val loss 기반 early stopping
            val_recon = test(model=ae, dataloader=val_dl, device=device)
            avg_val_loss = val_recon
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model_wts = copy.deepcopy(ae.state_dict())
                torch.save(best_model_wts, model_save_path)
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    ae.load_state_dict(best_model_wts)
    return ae
def train_ae_conditional(ae, train_dl, val_dl, optimizer, device, n_epochs=25, patience=10, 
                         model_save_path='best_ae.pth', use_train_plateau=False, plateau_threshold=1e-4):
    """
    Training loop for Conditional VQ-AENB.
    use_train_plateau: True면 val_dl 없이 train loss plateau로 early stopping
    """
    best_loss = float('inf')
    no_improvement = 0
    prev_train_loss = float('inf')
    best_model_wts = None
    
    for epoch in range(n_epochs):
        print('epoch: \t{0}'.format(epoch))
        train_recon = train_conditional(model=ae, dataloader=train_dl, optimizer=optimizer, device=device)
        print('\n')
        
        if use_train_plateau:
            # Train loss plateau 기반 early stopping
            loss_change = abs(prev_train_loss - train_recon)
            if loss_change < plateau_threshold:
                no_improvement += 1
            else:
                no_improvement = 0
                best_model_wts = copy.deepcopy(ae.state_dict())
                torch.save(best_model_wts, model_save_path)
            prev_train_loss = train_recon
            
            if no_improvement >= patience:
                print(f"Train loss plateau reached at epoch {epoch}")
                break
        else:
            # Val loss 기반 early stopping
            val_recon = test_conditional(model=ae, dataloader=val_dl, device=device)
            
            if val_recon < best_loss:
                best_loss = val_recon
                best_model_wts = copy.deepcopy(ae.state_dict())
                torch.save(best_model_wts, model_save_path)
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    if best_model_wts is not None:
        ae.load_state_dict(best_model_wts)
    
    return ae
def optimizer_ae(base_path, exp, device, data_dim, ae_latent_dim, ae_hidden_layers, ae_batch_size,
                 ae_learning_rate, ae_epochs, ae_patience, target_dir, model_type='AENB',
                 vq_num_codes=256, vq_commitment_weight=0.25, whole_data=False):
    
    ################################## Load Dataset - Instance ###############
    if whole_data:
        # 전체 데이터 로드 (split 없이)
        from src.utils import load_whole_dataset  # 새로 만들어야 함
        whole_dataset, label_encoder = load_whole_dataset(base_path, device)
        train_dl = DataLoader(whole_dataset, batch_size=ae_batch_size, shuffle=True, drop_last=False)
        val_dl = None
        set_random_seed(42)  # whole data는 고정 seed
    else:
        train_dataset, val_dataset, test_dataset, label_encoder = load_dataset_and_preprocessors(base_path, exp, device)
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
        
        print(f"Initializing VQ-AENB codebook with {vq_num_codes} codes...")
        ae.init_codebook(train_dl, method="kmeans", num_samples=min(20000, len(train_dl.dataset)))
        
        model_save_name = "vq_aenb_whole.pth" if whole_data else f"vq_aenb_{exp}.pth"
        csv_name = "vq_aenb_test.csv"
    else:
        ae = AENB(input_dim=data_dim, latent_dim=ae_latent_dim,
                  device=device, hidden_layers=ae_hidden_layers,
                  activation_function=nn.Sigmoid).to(device)
        
        model_save_name = "aenb_whole.pth" if whole_data else f"aenb_{exp}.pth"
        csv_name = "aenb_test.csv"

    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=ae_learning_rate)

    ################################## Training AE ####################
    ae = train_ae(ae, train_dl, val_dl, ae_optimizer, device, n_epochs=ae_epochs,
                  patience=ae_patience, model_save_path=f"{target_dir}/{model_save_name}",
                  use_train_plateau=whole_data)

    # Test only if not whole_data mode
    if not whole_data:
        test_recon = test(model=ae, optimizer=None, dataloader=test_dl, device=device,
                          csv_path=f"{target_dir}/{csv_name}")
    
    if model_type == 'VQ-AENB':
        codebook_stats = ae.get_codebook_usage()
        print(f"Codebook usage: {codebook_stats['num_active']}/{codebook_stats['total_codes']} codes active")
    
    torch.cuda.empty_cache()

def optimizer_ae_conditional(base_path, exp, device, data_dim, ae_latent_dim, ae_hidden_layers, 
                              ae_batch_size, ae_learning_rate, ae_epochs, ae_patience, target_dir,
                              n_studies, study_emb_dim=16,
                              vq_num_codes=256, vq_commitment_weight=0.25, whole_data=False):
    """
    Optimizer for Conditional VQ-AENB.
    Requires dataset with study_ids.
    """
    from src.utils import load_dataset_and_preprocessors_conditional, load_whole_dataset_conditional
    
    ################################## Load Dataset ###############
    if whole_data:
        whole_dataset, _ = load_whole_dataset_conditional(base_path, device)
        train_dl = DataLoader(whole_dataset, batch_size=ae_batch_size, shuffle=True, drop_last=False)
        val_dl = None
        set_random_seed(42)
    else:
        train_dataset, val_dataset, test_dataset, _ = load_dataset_and_preprocessors_conditional(base_path, exp, device)
        train_dl = DataLoader(train_dataset, batch_size=ae_batch_size, shuffle=True, drop_last=False)
        val_dl = DataLoader(val_dataset, batch_size=round(ae_batch_size/2), shuffle=False, drop_last=False)
        test_dl = DataLoader(test_dataset, batch_size=round(ae_batch_size/2), shuffle=False, drop_last=False)
        del train_dataset, val_dataset, test_dataset
        set_random_seed(exp)

    ################################## Set Model ####################
    ae = VQ_AENB_Conditional(
        input_dim=data_dim, 
        latent_dim=ae_latent_dim,
        device=device, 
        hidden_layers=ae_hidden_layers,
        n_studies=n_studies,
        study_emb_dim=study_emb_dim,
        num_codes=vq_num_codes, 
        commitment_weight=vq_commitment_weight,
        activation_function=nn.Sigmoid
    ).to(device)
    
    # Initialize codebook
    print(f"Initializing Conditional VQ-AENB codebook with {vq_num_codes} codes...")
    ae.init_codebook(train_dl, method="kmeans", num_samples=min(20000, len(train_dl.dataset)))
    
    model_save_name = "vq_aenb_conditional_whole.pth" if whole_data else f"vq_aenb_conditional_{exp}.pth"

    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=ae_learning_rate)

    ################################## Training AE ####################
    ae = train_ae_conditional(
        ae, train_dl, val_dl, ae_optimizer, device, 
        n_epochs=ae_epochs, patience=ae_patience, 
        model_save_path=f"{target_dir}/{model_save_name}",
        use_train_plateau=whole_data
    )

    # Test (if not whole_data mode)
    if not whole_data:
        test_recon = test_conditional(model=ae, dataloader=test_dl, device=device)
    
    # Print codebook usage
    codebook_stats = ae.get_codebook_usage()
    print(f"Codebook usage: {codebook_stats['num_active']}/{codebook_stats['total_codes']} codes active")
    
    # Save study embeddings
    study_embs = ae.get_study_embeddings()
    np.save(f"{target_dir}/study_embeddings{'_whole' if whole_data else f'_{exp}'}.npy", study_embs)
    print(f"Study embeddings saved: {study_embs.shape}")
    
    del train_dl, ae_optimizer
    if not whole_data:
        del val_dl, test_dl
    torch.cuda.empty_cache()