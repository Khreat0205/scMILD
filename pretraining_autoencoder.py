import sys
import os
import torch
import argparse
import json
from src.trainer_ae import optimizer_ae, optimizer_ae_conditional
from tqdm import tqdm as tdqm

def pretraining_autoencoder(data_dir="data/MyData", device_num=6, ae_learning_rate=1e-3, ae_epochs=15, ae_patience=3,
                            ae_latent_dim=128, ae_hidden_layers=[512, 256, 128], ae_batch_size=128, data_dim=2000, n_exp=8,
                            model_type='AENB', vq_num_codes=256, vq_commitment_weight=0.25, whole_data=False):
    base_path = data_dir
    target_dir = f'{base_path}/AE/'
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    hyperparameters_dict = {
        "ae_learning_rate": ae_learning_rate,
        "ae_epochs": ae_epochs,
        "ae_patience": ae_patience,
        "ae_latent_dim": ae_latent_dim,
        "ae_hidden_layers": ae_hidden_layers,
        "ae_batch_size": ae_batch_size,
        "model_type": model_type
    }
    
    # Add VQ-AENB specific parameters if model_type is VQ-AENB
    if model_type == 'VQ-AENB':
        hyperparameters_dict["vq_num_codes"] = vq_num_codes
        hyperparameters_dict["vq_commitment_weight"] = vq_commitment_weight

    ### Save hyperparameters
    hyperparameter_file_path = os.path.join(target_dir, 'hyperparameters_pretraining_autoencoder.json')
    if not os.path.exists(hyperparameter_file_path):
        with open(hyperparameter_file_path, 'w') as file:
            json.dump(hyperparameters_dict, file, indent=4)

    device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: Using device: {device}")
    print(f"INFO: Model type: {model_type}")
    if model_type == 'VQ-AENB':
        print(f"INFO: VQ-AENB with {vq_num_codes} codes, commitment weight: {vq_commitment_weight}")

    if whole_data:
        # Whole data: 단일 AE 학습
        optimizer_ae(base_path, exp=1, device=device, data_dim=data_dim, ae_latent_dim=ae_latent_dim,
                     ae_hidden_layers=ae_hidden_layers, ae_batch_size=ae_batch_size,
                     ae_learning_rate=ae_learning_rate, ae_epochs=ae_epochs, ae_patience=ae_patience,
                     target_dir=target_dir, model_type=model_type, vq_num_codes=vq_num_codes,
                     vq_commitment_weight=vq_commitment_weight, whole_data=True)
    else:
        # 기존: exp별 AE 학습
        exps = range(1, n_exp + 1)
        for exp in tdqm(exps, desc="Experiment"):
            optimizer_ae(base_path, exp, device, data_dim, ae_latent_dim, ae_hidden_layers,
                         ae_batch_size, ae_learning_rate, ae_epochs, ae_patience, target_dir,
                         model_type=model_type, vq_num_codes=vq_num_codes,
                         vq_commitment_weight=vq_commitment_weight, whole_data=False)

def pretraining_autoencoder_conditional(data_dir="data/MyData", device_num=6, ae_learning_rate=1e-3, 
                                         ae_epochs=15, ae_patience=3, ae_latent_dim=128, 
                                         ae_hidden_layers=[512, 256, 128], ae_batch_size=128, 
                                         data_dim=2000, n_exp=8, vq_num_codes=256, 
                                         vq_commitment_weight=0.25, study_emb_dim=16, whole_data=False):
    """
    Pretraining Conditional VQ-AENB with study/batch information.
    """
    import json
    
    base_path = data_dir
    target_dir = f'{base_path}/AE/'
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    # Load n_studies from metadata
    metadata_path = f"{base_path}/conditional_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        n_studies = metadata['n_studies']
        print(f"Loaded metadata: {n_studies} studies")
    else:
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Run load_and_save_datasets_adata_conditional first.")

    hyperparameters_dict = {
        "ae_learning_rate": ae_learning_rate,
        "ae_epochs": ae_epochs,
        "ae_patience": ae_patience,
        "ae_latent_dim": ae_latent_dim,
        "ae_hidden_layers": ae_hidden_layers,
        "ae_batch_size": ae_batch_size,
        "model_type": "VQ-AENB-Conditional",
        "vq_num_codes": vq_num_codes,
        "vq_commitment_weight": vq_commitment_weight,
        "n_studies": n_studies,
        "study_emb_dim": study_emb_dim,
    }
    
    ### Save hyperparameters
    hyperparameter_file_path = os.path.join(target_dir, 'hyperparameters_pretraining_autoencoder.json')
    with open(hyperparameter_file_path, 'w') as file:
        json.dump(hyperparameters_dict, file, indent=4)
    print(f"Hyperparameters saved to {hyperparameter_file_path}")

    device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
    print(f"INFO: Using device: {device}")
    print(f"INFO: Model type: VQ-AENB-Conditional")
    print(f"INFO: {n_studies} studies, embedding dim: {study_emb_dim}")
    print(f"INFO: Codebook: {vq_num_codes} codes")

    if whole_data:
        # Whole data: 단일 AE 학습
        print("\n[Whole Data Mode] Training single Conditional VQ-AENB...")
        optimizer_ae_conditional(
            base_path, exp=1, device=device, data_dim=data_dim, 
            ae_latent_dim=ae_latent_dim, ae_hidden_layers=ae_hidden_layers,
            ae_batch_size=ae_batch_size, ae_learning_rate=ae_learning_rate,
            ae_epochs=ae_epochs, ae_patience=ae_patience, target_dir=target_dir,
            n_studies=n_studies, study_emb_dim=study_emb_dim,
            vq_num_codes=vq_num_codes, vq_commitment_weight=vq_commitment_weight,
            whole_data=True
        )
    else:
        # exp별 AE 학습
        exps = range(1, n_exp + 1)
        for exp in tdqm(exps, desc="Experiment"):
            optimizer_ae_conditional(
                base_path, exp, device, data_dim, ae_latent_dim, ae_hidden_layers,
                ae_batch_size, ae_learning_rate, ae_epochs, ae_patience, target_dir,
                n_studies=n_studies, study_emb_dim=study_emb_dim,
                vq_num_codes=vq_num_codes, vq_commitment_weight=vq_commitment_weight,
                whole_data=False
            )
    
    print("\n✅ Conditional VQ-AENB pretraining complete!")

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
    
    # VQ-AENB specific arguments
    parser.add_argument('--model_type', type=str, default='AENB', 
                        choices=['AENB', 'VQ-AENB', 'VQ-AENB-Conditional'],
                        help='Model type to use')
    parser.add_argument('--vq_num_codes', type=int, default=256,
                        help='Number of codes in VQ-AENB codebook')
    parser.add_argument('--vq_commitment_weight', type=float, default=0.25,
                        help='Commitment loss weight for VQ-AENB')
    parser.add_argument('--whole_data', action='store_true', 
                        help='Train single AE on whole dataset')
    
    # Conditional specific arguments
    parser.add_argument('--study_emb_dim', type=int, default=16,
                        help='Study embedding dimension (for Conditional model)')

    args = parser.parse_args()
    
    if args.model_type == 'VQ-AENB-Conditional':
        pretraining_autoencoder_conditional(
            data_dir=args.data_dir,
            device_num=args.device_num,
            ae_learning_rate=args.ae_learning_rate,
            ae_epochs=args.ae_epochs,
            ae_patience=args.ae_patience,
            ae_latent_dim=args.ae_latent_dim,
            ae_hidden_layers=args.ae_hidden_layers,
            ae_batch_size=args.ae_batch_size,
            data_dim=args.data_dim,
            n_exp=args.n_exp,
            vq_num_codes=args.vq_num_codes,
            vq_commitment_weight=args.vq_commitment_weight,
            study_emb_dim=args.study_emb_dim,
            whole_data=args.whole_data
        )
    else:
        pretraining_autoencoder(
            data_dir=args.data_dir,
            device_num=args.device_num,
            ae_learning_rate=args.ae_learning_rate,
            ae_epochs=args.ae_epochs,
            ae_patience=args.ae_patience,
            ae_latent_dim=args.ae_latent_dim,
            ae_hidden_layers=args.ae_hidden_layers,
            ae_batch_size=args.ae_batch_size,
            data_dim=args.data_dim,
            n_exp=args.n_exp,
            model_type=args.model_type,
            vq_num_codes=args.vq_num_codes,
            vq_commitment_weight=args.vq_commitment_weight,
            whole_data=args.whole_data
        )