import os
import torch
from torch import nn
import time
import copy
import pandas


def negative_binomial_loss(y_pred, theta, y_true, eps=1e-10):
    # adapted from https://github.com/uhlerlab/STACI/blob/master/gae/gae/optimizer.py
    nbloss1=torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
    nbloss2=(theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
    nbloss=nbloss1+nbloss2
    return torch.mean(nbloss)

def _train_or_test(model=None, dataloader=None, optimizer=None, device='cuda'):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    start = time.time()
    
    n_batches = 0
    
    total_recons_loss = 0
    for i, (data, _, instance_labels) in enumerate(dataloader):
        input_data = data.to(device)
        target = instance_labels.to(device)
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            mu_recon, theta_recon = model(input_data, target, is_train=is_train)
            recons_loss = negative_binomial_loss(mu_recon,theta_recon, input_data)
           
            # Accumulate loss values
            n_batches += 1
            
            total_recons_loss += recons_loss.item()
            if is_train:
                optimizer.zero_grad()
                recons_loss.backward()
                optimizer.step()

            del input_data
            del mu_recon
            del theta_recon

    end = time.time()
    
    avg_recons_loss = total_recons_loss / n_batches
    print(f'\tTime: {end - start:.2f}s')
    print(f'\tReconstruction Loss: {avg_recons_loss:.4f}')
    
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