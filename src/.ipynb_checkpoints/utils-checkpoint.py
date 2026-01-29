import os 
import torch
import pickle
import numpy as np
import random
import json
import argparse
import anndata
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse
from src.dataset import MilDataset, InstanceDataset, collate, update_instance_labels_with_bag_labels
from src.model import AENB, VQ_AENB, AttentionModule, TeacherBranch, StudentBranch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

class InstanceDatasetConditional(torch.utils.data.Dataset):
    """
    Dataset for Conditional VQ-AENB.
    Returns (data, study_id, label) tuples.
    """
    def __init__(self, data, study_ids, labels):
        self.data = data
        self.study_ids = study_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.study_ids[idx], self.labels[idx]
        
def preprocess_adata(adata, data_dim=2000):
    """
    Preprocess the anndata object by filtering cells, normalizing, and selecting highly variable genes.
    """
    print(f'Initial shape: {adata.shape}')
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Filter cells with too few genes
    sc.pp.filter_cells(adata, min_genes=200)
    print(adata)
    adata.obs_names_make_unique()
    adata_raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    print(f'Normalization Complete: {adata.shape}')
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=data_dim, batch_key='study')
    adata = adata_raw[:, adata.var.highly_variable]
    print(f'Highly variable gene selection Complete: {adata.shape}')
    del adata_raw
    return adata

def load_mil_dataset_from_adata(adata, device='cpu', label_encoder=None, is_train=True, use_cell_types=False):
    features = adata.X
    features = features.toarray() if issparse(features) else features
    sample_ids = adata.obs['sample_id_numeric'].values
    labels = adata.obs['disease_numeric'].values
    
    if use_cell_types:
        instance_labels = adata.obs['cell_type'].values
        if is_train:
            if label_encoder is None:
                label_encoder = LabelEncoder()
                instance_labels = label_encoder.fit_transform(instance_labels)
            else:
                instance_labels = label_encoder.transform(instance_labels)
        else:
            if label_encoder is None:
                raise ValueError("label_encoder must be provided for test set when using cell types")
            try:
                instance_labels = label_encoder.transform(instance_labels)
            except ValueError:
                valid_labels = [label for label in instance_labels if label in label_encoder.classes_]
                mask = adata.obs['cell_type'].isin(valid_labels)
                adata = adata[mask]
                features = features[mask]
                sample_ids = sample_ids[mask]
                labels = labels[mask]
                instance_labels = label_encoder.transform(adata.obs['cell_type'].values)
    else:
        instance_labels = np.zeros(len(adata), dtype=int)
        label_encoder = None

    # PyTorch Tensor 변환
    tensor_data = torch.tensor(features, dtype=torch.float).to(device)
    tensor_ids = torch.tensor(sample_ids, dtype=torch.long).to(device)
    tensor_labels = torch.tensor(labels, dtype=torch.long).to(device)
    tensor_instance_labels = torch.tensor(instance_labels, dtype=torch.long).to(device)

    # 백 레이블 계산
    unique_bag_ids = torch.unique(tensor_ids)
    bag_labels = torch.stack([max(tensor_labels[tensor_ids == i]) for i in unique_bag_ids]).long()
    bag_labels = bag_labels.cpu().long()

    instance_dataset = InstanceDataset(tensor_data, tensor_ids, bag_labels, tensor_instance_labels)
    return instance_dataset, label_encoder


def save_preprocessors(base_path, label_encoder, exp):
    with open(f"{base_path}/label_encoder_exp{exp}.pkl", 'wb') as f:
        pickle.dump(label_encoder, f)

# def load_and_save_datasets_adata(base_path, exps, device, adata):
#     '''
#     base_path: str
#         path to save the datasets
#     exps: list
#         list of integers for experiment numbers
#     device: str
#         device to save the datasets
#     adata: AnnData
#         AnnData object to load the datasets
        
#     '''
#     for exp in exps:
#         label_encoder = LabelEncoder()
        
#         print(f"Experiment {exp}")
#         sample_labels = adata.obs[['disease_numeric', 'sample_id_numeric']].drop_duplicates()
#         split_ratio = [0.5, 0.25, 0.25]
        
#         train_val_set, test_set = train_test_split(sample_labels, test_size=split_ratio[2], random_state=exp, stratify=sample_labels['disease_numeric'])
#         train_set, val_set = train_test_split(train_val_set, test_size=split_ratio[1] / (1 - split_ratio[2]), random_state=exp,stratify=train_val_set['disease_numeric'])
#         print(f"Experiment {exp} Train-Val-Test Split Complete!")
        
#         train_data = adata[adata.obs['sample_id_numeric'].isin(train_set['sample_id_numeric'])]
#         val_data = adata[adata.obs['sample_id_numeric'].isin(val_set['sample_id_numeric'])]
#         test_data = adata[adata.obs['sample_id_numeric'].isin(test_set['sample_id_numeric'])]

#         train_dataset, label_encoder = load_mil_dataset_from_adata(adata=train_data, device=device, is_train=True, label_encoder=label_encoder)
#         save_preprocessors(base_path, label_encoder, exp)
#         torch.save(train_dataset, f"{base_path}/train_dataset_exp{exp}.pt")
#         print(f"Experiment {exp} Train Dataset Saved!")
        
#         val_dataset, _ = load_mil_dataset_from_adata(adata=val_data, is_train=False, label_encoder=label_encoder)
#         torch.save(val_dataset, f"{base_path}/val_dataset_exp{exp}.pt")
#         print(f"Experiment {exp} Val Dataset Saved!")
        
#         test_dataset, _ = load_mil_dataset_from_adata(adata=test_data, is_train=False, label_encoder=label_encoder)
#         torch.save(test_dataset, f"{base_path}/test_dataset_exp{exp}.pt")
#         print(f"Experiment {exp} Test Dataset Saved!")
        
#         del train_dataset, label_encoder, val_dataset, test_dataset, train_data, val_data, test_data
def adaptive_subsample_size(cell_count, base_size=1000):
    """Calculate adaptive subsample size based on original cell count"""
    if cell_count <= base_size:
        return cell_count
    
    # Log-scale based adaptive sizing
    return int(base_size * (1 + np.log10(cell_count/base_size)))

def adaptive_repeat_count(cell_count, base_repeat=10, min_repeat=3):
    """Calculate adaptive repeat count based on original cell count"""
    if cell_count <= 1000:
        return base_repeat
        
    # Reduce repeat count for larger samples to manage computational load
    return max(min_repeat, 
              int(base_repeat * (1 - np.log10(cell_count/1000)/np.log10(275000/1000))))

def improved_subsample_adata(adata, base_size=1000, base_repeat=10, random_state=42):
    """
    Improved subsampling function that adapts the subsample size and repeat count
    based on the original cell count in each sample.
    - Uses adaptive subsample size and repeat count
    - Ensures that smaller samples are not excessively subsampled
    Parameters:
    -----------
    adata : AnnData
        서브샘플링할 데이터
    base_size : int
        기본 서브샘플 크기 (default: 1000)
    base_repeat : int
        기본 반복 횟수 (default: 10)
    random_state : int
        난수 생성을 위한 시드값 (default: 42)
    
    """
    np.random.seed(random_state)
    subsampled_adatas = []
    new_sample_id = max(adata.obs['sample_id_numeric'].astype(int)) + 1
    
    for sample_id in adata.obs['sample_id_numeric'].unique():
        sample_data = adata[adata.obs['sample_id_numeric'] == sample_id]
        cell_count = len(sample_data)
        
        # Calculate adaptive parameters
        subsample_size = adaptive_subsample_size(cell_count, base_size)
        repeat_count = adaptive_repeat_count(cell_count, base_repeat)
        
        for i in range(repeat_count):
            indices = np.random.choice(len(sample_data), 
                                    size=subsample_size,
                                    replace=len(sample_data) < subsample_size)
            
            subset = sample_data[indices].copy()
            subset.obs['sample_id_numeric'] = new_sample_id
            subset.obs['subsample_weight'] = cell_count / (subsample_size * repeat_count)
            new_sample_id += 1
            subsampled_adatas.append(subset)
    
    return anndata.concat(subsampled_adatas, join='outer', index_unique='-')
        
def subsample_adata(adata, n_subsample=400, n_repeat=10, random_state=42):
    np.random.seed(random_state)
    
    subsampled_adatas = []
    new_sample_id = max(adata.obs['sample_id_numeric'].astype(int)) + 1
    
    for sample_id in adata.obs['sample_id_numeric'].unique():
        sample_data = adata[adata.obs['sample_id_numeric'] == sample_id]
        
        for i in range(n_repeat):
            if len(sample_data) < n_subsample:
                # If sample size is smaller than desired, sample with replacement
                indices = np.random.choice(len(sample_data), size=n_subsample, replace=True)
            else:
                # If sample size is larger, sample without replacement
                indices = np.random.choice(len(sample_data), size=n_subsample, replace=False)
            
            subset = sample_data[indices].copy()
            subset.obs['sample_id_numeric'] = new_sample_id
            new_sample_id += 1
            subsampled_adatas.append(subset)
    
    return anndata.concat(subsampled_adatas, join='outer', index_unique='-')
## 2025 0204에 수정함  - used_cell type 관련
def load_and_save_datasets_adata(base_path, exps, device, adata, do_subsample=False, n_subsample=400, suffix = None, split_ratio = [0.5, 0.25, 0.25]):
    for exp in exps:
        label_encoder = LabelEncoder()
        
        print(f"Experiment {exp}")
        sample_labels = adata.obs[['disease_numeric', 'sample_id_numeric']].drop_duplicates()
        
        train_val_set, test_set = train_test_split(sample_labels, test_size=split_ratio[2], random_state=exp, stratify=sample_labels['disease_numeric'])
        train_set, val_set = train_test_split(train_val_set, test_size=split_ratio[1] / (1 - split_ratio[2]), random_state=exp, stratify=train_val_set['disease_numeric'])
        print(f"Experiment {exp} Train-Val-Test Split Complete!")
        
        train_data = adata[adata.obs['sample_id_numeric'].isin(train_set['sample_id_numeric'])]
        val_data = adata[adata.obs['sample_id_numeric'].isin(val_set['sample_id_numeric'])]
        test_data = adata[adata.obs['sample_id_numeric'].isin(test_set['sample_id_numeric'])]

        if do_subsample:
            train_data = improved_subsample_adata(train_data, n_subsample, random_state=exp)
            print(f'Experiment {exp} Train Data Subsampled! {train_data.shape[0]} cells')
            # val_data = subsample_adata(val_data, n_subsample, random_state=exp)
            # print(f'Experiment {exp} Val Data Subsampled! {val_data.shape[0]} cells')
            # test_data = subsample_adata(test_data, n_subsample, random_state=exp)
            # print(f'Experiment {exp} Test Data Subsampled! {test_data.shape[0]} cells')

        # Ensure sample_id_numeric is int64
        train_data.obs['sample_id_numeric'] = train_data.obs['sample_id_numeric'].astype('int64')
        val_data.obs['sample_id_numeric'] = val_data.obs['sample_id_numeric'].astype('int64')
        test_data.obs['sample_id_numeric'] = test_data.obs['sample_id_numeric'].astype('int64')
        
        train_dataset, label_encoder = load_mil_dataset_from_adata(adata=train_data, device=device,  is_train=True, label_encoder=label_encoder, use_cell_types=False)
        
        val_dataset, _ = load_mil_dataset_from_adata(adata=val_data, is_train=False, label_encoder=label_encoder, use_cell_types=False)
        test_dataset, _ = load_mil_dataset_from_adata(adata=test_data, is_train=False, label_encoder=label_encoder, use_cell_types=False)
        if suffix is not None:
            torch.save(train_dataset, f"{base_path}/train_dataset_exp{exp}_{suffix}.pt")
            torch.save(val_dataset, f"{base_path}/val_dataset_exp{exp}_{suffix}.pt")
            torch.save(test_dataset, f"{base_path}/test_dataset_exp{exp}_{suffix}.pt")
            
        else:  
            torch.save(train_dataset, f"{base_path}/train_dataset_exp{exp}.pt")
            torch.save(val_dataset, f"{base_path}/val_dataset_exp{exp}.pt")
            torch.save(test_dataset, f"{base_path}/test_dataset_exp{exp}.pt")
        
        
        del train_dataset, label_encoder, val_dataset, test_dataset, train_data, val_data, test_data

def load_and_save_datasets_adata(base_path, exps, device, adata, do_subsample=False, n_subsample=400, 
                                  suffix=None, split_ratio=[0.5, 0.25, 0.25],
                                  save_study_mapping=True, study_col='study'):
    """
    Dataset 저장 + sample_to_study 매핑 동시 생성
    
    Args:
        save_study_mapping: True면 sample_to_study json도 저장
        study_col: study 정보가 있는 컬럼명
    """
    import json
    from sklearn.preprocessing import LabelEncoder
    
    os.makedirs(base_path, exist_ok=True)
    
    # Study ID 인코딩 (전체 adata 기준)
    if save_study_mapping and study_col in adata.obs.columns:
        study_encoder = LabelEncoder()
        adata.obs['study_id_numeric'] = study_encoder.fit_transform(adata.obs[study_col])
        
        # study_mapping.json 저장 (이미 있으면 스킵)
        study_mapping_path = f"{base_path}/study_mapping{'_' + suffix if suffix else ''}.json"
        study_mapping = {str(i): name for i, name in enumerate(study_encoder.classes_)}
        with open(study_mapping_path, 'w') as f:
            json.dump(study_mapping, f, indent=2)
        print(f"Study mapping saved: {study_mapping}")
    
    for exp in exps:
        label_encoder = LabelEncoder()
        
        print(f"Experiment {exp}")
        sample_labels = adata.obs[['disease_numeric', 'sample_id_numeric']].drop_duplicates()
        
        train_val_set, test_set = train_test_split(
            sample_labels, test_size=split_ratio[2], 
            random_state=exp, stratify=sample_labels['disease_numeric']
        )
        train_set, val_set = train_test_split(
            train_val_set, test_size=split_ratio[1] / (1 - split_ratio[2]),
            random_state=exp, stratify=train_val_set['disease_numeric']
        )
        print(f"Experiment {exp} Train-Val-Test Split Complete!")
        
        train_data = adata[adata.obs['sample_id_numeric'].isin(train_set['sample_id_numeric'])]
        val_data = adata[adata.obs['sample_id_numeric'].isin(val_set['sample_id_numeric'])]
        test_data = adata[adata.obs['sample_id_numeric'].isin(test_set['sample_id_numeric'])]

        if do_subsample:
            train_data = improved_subsample_adata(train_data, n_subsample, random_state=exp)
            print(f'Experiment {exp} Train Data Subsampled! {train_data.shape[0]} cells')

        # Ensure sample_id_numeric is int64
        train_data.obs['sample_id_numeric'] = train_data.obs['sample_id_numeric'].astype('int64')
        val_data.obs['sample_id_numeric'] = val_data.obs['sample_id_numeric'].astype('int64')
        test_data.obs['sample_id_numeric'] = test_data.obs['sample_id_numeric'].astype('int64')
        
        train_dataset, label_encoder = load_mil_dataset_from_adata(
            adata=train_data, device=device, is_train=True, 
            label_encoder=label_encoder, use_cell_types=False
        )
        val_dataset, _ = load_mil_dataset_from_adata(
            adata=val_data, is_train=False, 
            label_encoder=label_encoder, use_cell_types=False
        )
        test_dataset, _ = load_mil_dataset_from_adata(
            adata=test_data, is_train=False, 
            label_encoder=label_encoder, use_cell_types=False
        )
        
        # 저장
        if suffix is not None:
            torch.save(train_dataset, f"{base_path}/train_dataset_exp{exp}_{suffix}.pt")
            torch.save(val_dataset, f"{base_path}/val_dataset_exp{exp}_{suffix}.pt")
            torch.save(test_dataset, f"{base_path}/test_dataset_exp{exp}_{suffix}.pt")
        else:  
            torch.save(train_dataset, f"{base_path}/train_dataset_exp{exp}.pt")
            torch.save(val_dataset, f"{base_path}/val_dataset_exp{exp}.pt")
            torch.save(test_dataset, f"{base_path}/test_dataset_exp{exp}.pt")
        
        # [NEW] sample_to_study 매핑 저장
        if save_study_mapping and study_col in adata.obs.columns:
            # 모든 split의 sample -> study 매핑
            all_split_data = anndata.concat([train_data, val_data, test_data])
            sample_study_df = all_split_data.obs[['sample_id_numeric', 'study_id_numeric']].drop_duplicates()
            sample_to_study = dict(zip(
                sample_study_df['sample_id_numeric'].astype(int),
                sample_study_df['study_id_numeric'].astype(int)
            ))
            
            mapping_fname = f"sample_to_study_exp{exp}{'_' + suffix if suffix else ''}.json"
            with open(f"{base_path}/{mapping_fname}", 'w') as f:
                json.dump({str(k): v for k, v in sample_to_study.items()}, f, indent=2)
            print(f"  Saved: {mapping_fname} ({len(sample_to_study)} samples)")
        
        del train_dataset, label_encoder, val_dataset, test_dataset, train_data, val_data, test_data

def load_dataset_and_preprocessors(base_path, exp, device, used_cell_types=False, suffix=None):
    if suffix is not None:
        train_dataset = torch.load(f"{base_path}/train_dataset_exp{exp}_{suffix}.pt", map_location=device, weights_only=False)
        val_dataset = torch.load(f"{base_path}/val_dataset_exp{exp}_{suffix}.pt", map_location=device, weights_only=False)
        test_dataset = torch.load(f"{base_path}/test_dataset_exp{exp}_{suffix}.pt", map_location=device, weights_only=False)
    else:
        train_dataset = torch.load(f"{base_path}/train_dataset_exp{exp}.pt", map_location= device, weights_only=False)
        val_dataset = torch.load(f"{base_path}/val_dataset_exp{exp}.pt", map_location= device, weights_only=False)
        test_dataset = torch.load(f"{base_path}/test_dataset_exp{exp}.pt",map_location = device, weights_only=False)
    if used_cell_types:
        with open(f"{base_path}/label_encoder_exp{exp}.pkl", 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        label_encoder = None

    return train_dataset, val_dataset, test_dataset, label_encoder

def load_whole_dataset(base_path, device, suffix=None):
    """exp=1 기준으로 train/val/test 합쳐서 whole dataset 생성"""
    train_dataset, val_dataset, test_dataset, _ = load_dataset_and_preprocessors(base_path, exp=1, device=device, suffix=suffix)
    
    # ConcatDataset으로 합치기
    from torch.utils.data import ConcatDataset
    whole_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    
    return whole_dataset, None

def load_dataset_and_preprocessors_conditional(base_path, exp, device, suffix=None):
    """
    Load dataset with study_ids for Conditional VQ-AENB.
    Expects datasets saved with study_id information.
    """
    if suffix is not None:
        train_dataset = torch.load(f"{base_path}/train_dataset_conditional_exp{exp}_{suffix}.pt", map_location=device, weights_only=False)
        val_dataset = torch.load(f"{base_path}/val_dataset_conditional_exp{exp}_{suffix}.pt", map_location=device, weights_only=False)
        test_dataset = torch.load(f"{base_path}/test_dataset_conditional_exp{exp}_{suffix}.pt", map_location=device, weights_only=False)
    else:
        train_dataset = torch.load(f"{base_path}/train_dataset_conditional_exp{exp}.pt", map_location=device, weights_only=False)
        val_dataset = torch.load(f"{base_path}/val_dataset_conditional_exp{exp}.pt", map_location=device, weights_only=False)
        test_dataset = torch.load(f"{base_path}/test_dataset_conditional_exp{exp}.pt", map_location=device, weights_only=False)
    
    return train_dataset, val_dataset, test_dataset, None


def load_whole_dataset_conditional(base_path, device, suffix=None):
    """
    Load whole dataset (train+val+test combined) with study_ids.
    """
    from torch.utils.data import ConcatDataset
    
    train_dataset, val_dataset, test_dataset, _ = load_dataset_and_preprocessors_conditional(
        base_path, exp=1, device=device, suffix=suffix
    )
    whole_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    
    return whole_dataset, None

def set_random_seed(exp, enable_cudnn_benchmark=True):
    """
    Set random seed for reproducibility.
    
    Args:
        exp: Experiment number for seed
        enable_cudnn_benchmark: If True, enable cudnn benchmark for better performance (default: True)
                                Note: This may cause slight non-determinism but improves speed significantly
    """
    torch.manual_seed(exp)
    torch.cuda.manual_seed(exp)
    np.random.seed(exp)
    random.seed(exp)
    torch.cuda.manual_seed_all(exp)
    
    if enable_cudnn_benchmark:
        # Enable cudnn benchmark for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("CuDNN benchmark enabled for better GPU utilization (may cause slight non-determinism)")
    else:
        # Strict determinism (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_ae_hyperparameters(ae_dir):
    hyperparameter_file_path = os.path.join(ae_dir, f'hyperparameters_pretraining_autoencoder.json')
    with open(hyperparameter_file_path, 'r') as file:
        loaded_args_dict = json.load(file)
        
    loaded_args = argparse.Namespace(**loaded_args_dict)
    
    # Return basic parameters plus model type info if available
    model_type = getattr(loaded_args, 'model_type', 'AENB')  # Default to AENB for backward compatibility
    vq_num_codes = getattr(loaded_args, 'vq_num_codes', 256)
    vq_commitment_weight = getattr(loaded_args, 'vq_commitment_weight', 0.25)
    
    return loaded_args.ae_latent_dim, loaded_args.ae_hidden_layers, model_type, vq_num_codes, vq_commitment_weight

# def load_and_process_datasets(data_dir, exp, device, student_batch_size):
#     train_dataset, val_dataset, test_dataset, _ = load_dataset_and_preprocessors(data_dir, exp, device=torch.device('cpu'))
#     instance_train_dataset = update_instance_labels_with_bag_labels(train_dataset, device=torch.device('cpu'))
    
#     set_random_seed(exp)
    
#     #### add 06 04 _ sampler
#     # Calculate weights for each combined label
#     label_counts = instance_train_dataset.bag_labels.bincount()
#     label_weights = 1.0 / label_counts
#     instance_weights = label_weights[instance_train_dataset.bag_labels]
#     # Define your DataLoader
#     num_workers = 1
#     pin_memory = True
#     prefetch_factor_student=4
#     # Create a WeightedRandomSampler
#     sampler = WeightedRandomSampler(instance_weights, len(instance_train_dataset))
    
#     instance_train_dl = DataLoader(instance_train_dataset, batch_size=student_batch_size, sampler=sampler, drop_last=False, pin_memory=pin_memory, num_workers=num_workers, prefetch_factor=prefetch_factor_student) 
    
#     bag_train = MilDataset(train_dataset.data.to(device), train_dataset.ids.unsqueeze(0).to(device), train_dataset.labels.to(device), train_dataset.instance_labels.to(device))
#     bag_val = MilDataset(val_dataset.data.to(device), val_dataset.ids.unsqueeze(0).to(device), val_dataset.labels.to(device), val_dataset.instance_labels.to(device))
#     bag_test = MilDataset(test_dataset.data.to(device), test_dataset.ids.unsqueeze(0).to(device), test_dataset.labels.to(device), test_dataset.instance_labels.to(device))
    
#     return instance_train_dl, bag_train, bag_val, bag_test

   
# def load_and_process_datasets(data_dir, exp, device, student_batch_size, suffix=None):
#     train_dataset, val_dataset, test_dataset, _ = load_dataset_and_preprocessors(data_dir, exp, device=torch.device('cpu'), suffix=suffix)


#     # 2. [NEW] Conditional AE 사용 시 Study ID 주입
#     if use_conditional_ae and conditional_ae_dir:
#         import json
#         import torch
        
#         # 매핑 파일 로드 (sample_id -> study_id)
#         # 이 파일이 없다면 train_scMILD.py 실행 전 한 번 생성해야 합니다. (아래 3단계 참고)
#         mapping_path = f"{conditional_ae_dir}/sample_to_study_exp{exp}.json"
        
#         if os.path.exists(mapping_path):
#             with open(mapping_path, 'r') as f:
#                 # json 키는 문자열이므로 int로 변환
#                 sample_to_study = {int(k): int(v) for k, v in json.load(f).items()}
            
#             print(f"INFO: Injecting Study IDs from {mapping_path}")
            
#             # Helper function to inject IDs
#             def inject_study_ids(dataset, mapping):
#                 # dataset.ids는 sample_id들임
#                 s_ids = [mapping.get(int(s_id), 0) for s_id in dataset.ids.numpy()]
#                 dataset.study_ids = torch.tensor(s_ids, dtype=torch.long)
#                 return dataset

#             train_dataset = inject_study_ids(train_dataset, sample_to_study)
#             val_dataset = inject_study_ids(val_dataset, sample_to_study)
#             test_dataset = inject_study_ids(test_dataset, sample_to_study)
#         else:
#             print(f"WARNING: Mapping file {mapping_path} not found. Conditional mode disabled.")

#     # 3. Instance Dataset 생성 (Study ID 전달)
#     # update_instance_labels_with_bag_labels 함수도 study_ids를 보존하도록 수정 필요하지만,
#     # 여기서는 새로 생성하므로 직접 넣어줍니다.
#     instance_train_dataset = update_instance_labels_with_bag_labels(train_dataset, device=torch.device('cpu'))
    
#     if hasattr(train_dataset, 'study_ids') and train_dataset.study_ids is not None:
#         instance_train_dataset.study_ids = train_dataset.study_ids
    
#     set_random_seed(exp)
    
#     #### add 06 04 _ sampler
#     # Calculate weights for each combined label
#     label_counts = instance_train_dataset.bag_labels.bincount()
#     label_weights = 1.0 / label_counts
#     instance_weights = label_weights[instance_train_dataset.bag_labels]
#     # Define your DataLoader
#     num_workers = 8
#     pin_memory = True
#     prefetch_factor_student=4
#     # Create a WeightedRandomSampler
#     sampler = WeightedRandomSampler(instance_weights, len(instance_train_dataset))

#     # DataLoader 생성
#     instance_train_dl = DataLoader(instance_train_dataset, batch_size=student_batch_size, sampler=sampler, 
#                                    drop_last=False, pin_memory=pin_memory, num_workers=num_workers, prefetch_factor=prefetch_factor_student) 
    
#     # 4. Bag Dataset 생성 (study_ids 전달)
#     bag_train = MilDataset(train_dataset.data.to(device), train_dataset.ids.unsqueeze(0).to(device), 
#                            train_dataset.labels.to(device), train_dataset.instance_labels.to(device), 
#                            study_ids=getattr(train_dataset, 'study_ids', None)) # 전달
                           
#     bag_val = MilDataset(val_dataset.data.to(device), val_dataset.ids.unsqueeze(0).to(device), 
#                          val_dataset.labels.to(device), val_dataset.instance_labels.to(device),
#                          study_ids=getattr(val_dataset, 'study_ids', None)) # 전달
                         
#     bag_test = MilDataset(test_dataset.data.to(device), test_dataset.ids.unsqueeze(0).to(device), 
#                           test_dataset.labels.to(device), test_dataset.instance_labels.to(device),
#                           study_ids=getattr(test_dataset, 'study_ids', None)) # 전달
    
#     return instance_train_dl, bag_train, bag_val, bag_test
    
    # instance_train_dl = DataLoader(instance_train_dataset, batch_size=student_batch_size, sampler=sampler, drop_last=False, pin_memory=pin_memory, num_workers=num_workers, prefetch_factor=prefetch_factor_student) 
    
    # bag_train = MilDataset(train_dataset.data.to(device), train_dataset.ids.unsqueeze(0).to(device), train_dataset.labels.to(device), train_dataset.instance_labels.to(device))
    # bag_val = MilDataset(val_dataset.data.to(device), val_dataset.ids.unsqueeze(0).to(device), val_dataset.labels.to(device), val_dataset.instance_labels.to(device))
    # bag_test = MilDataset(test_dataset.data.to(device), test_dataset.ids.unsqueeze(0).to(device), test_dataset.labels.to(device), test_dataset.instance_labels.to(device))
    
    # return instance_train_dl, bag_train, bag_val, bag_test



def load_and_process_datasets(data_dir, exp, device, student_batch_size, suffix=None,
                              use_conditional_ae=False, conditional_ae_dir=None):
    """
    Load and process datasets for scMILD training.
    
    Args:
        data_dir: Directory containing dataset files
        exp: Experiment number
        device: torch device
        student_batch_size: Batch size for instance dataloader
        suffix: Dataset suffix (e.g., 'SCP1884', 'Skin3')
        use_conditional_ae: Whether to use conditional VQ-AENB
        conditional_ae_dir: Directory containing conditional AE files and mappings
    """
    train_dataset, val_dataset, test_dataset, _ = load_dataset_and_preprocessors(
        data_dir, exp, device=torch.device('cpu'), suffix=suffix
        )
    # [CONDITIONAL] Study ID 주입
    if use_conditional_ae and conditional_ae_dir:
        import json
        
        # suffix 포함한 경로 먼저 시도
        if suffix:
            mapping_path = f"{data_dir}/sample_to_study_exp{exp}_{suffix}.json"
        else:
            mapping_path = f"{data_dir}/sample_to_study_exp{exp}.json"
        
        # suffix 있는 파일 없으면 suffix 없는 버전 시도
        if not os.path.exists(mapping_path) and suffix:
            fallback_path = f"{data_dir}/sample_to_study_exp{exp}.json"
            if os.path.exists(fallback_path):
                mapping_path = fallback_path
                print(f"INFO: Using fallback mapping {fallback_path}")
        
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r') as f:
                sample_to_study = {int(k): int(v) for k, v in json.load(f).items()}
            
            print(f"INFO: Injecting Study IDs from {mapping_path}")
            
            def inject_study_ids(dataset, mapping):
                """Helper function to inject study_ids into dataset."""
                s_ids = [mapping.get(int(s_id), 0) for s_id in dataset.ids.numpy()]
                dataset.study_ids = torch.tensor(s_ids, dtype=torch.long)
                return dataset

            train_dataset = inject_study_ids(train_dataset, sample_to_study)
            val_dataset = inject_study_ids(val_dataset, sample_to_study)
            test_dataset = inject_study_ids(test_dataset, sample_to_study)
        else:
            print(f"WARNING: Mapping file not found at {mapping_path}. Study IDs will be None.")

    # Instance Dataset 생성
    instance_train_dataset = update_instance_labels_with_bag_labels(train_dataset, device=torch.device('cpu'))
    
    # Study IDs 보존
    if hasattr(train_dataset, 'study_ids') and train_dataset.study_ids is not None:
        instance_train_dataset.study_ids = train_dataset.study_ids
    
    set_random_seed(exp)
    
    # Weighted sampler 설정
    label_counts = instance_train_dataset.bag_labels.bincount()
    label_weights = 1.0 / label_counts
    instance_weights = label_weights[instance_train_dataset.bag_labels]
    
    num_workers = 8
    pin_memory = True
    prefetch_factor_student = 4
    
    sampler = WeightedRandomSampler(instance_weights, len(instance_train_dataset))
    instance_train_dl = DataLoader(
        instance_train_dataset, 
        batch_size=student_batch_size, 
        sampler=sampler,
        drop_last=False, 
        pin_memory=pin_memory, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor_student
    )
    
    # Bag Dataset 생성 (study_ids 포함)
    def get_study_ids_on_device(dataset, target_device):
        """study_ids를 device로 이동, 없으면 None 반환"""
        if hasattr(dataset, 'study_ids') and dataset.study_ids is not None:
            return dataset.study_ids.to(target_device)
        return None
    
    bag_train = MilDataset(
        train_dataset.data.to(device), 
        train_dataset.ids.unsqueeze(0).to(device),
        train_dataset.labels.to(device), 
        train_dataset.instance_labels.to(device),
        study_ids=get_study_ids_on_device(train_dataset, device)
    )
    bag_val = MilDataset(
        val_dataset.data.to(device), 
        val_dataset.ids.unsqueeze(0).to(device),
        val_dataset.labels.to(device), 
        val_dataset.instance_labels.to(device),
        study_ids=get_study_ids_on_device(val_dataset, device)
    )
    bag_test = MilDataset(
        test_dataset.data.to(device), 
        test_dataset.ids.unsqueeze(0).to(device),
        test_dataset.labels.to(device), 
        test_dataset.instance_labels.to(device),
        study_ids=get_study_ids_on_device(test_dataset, device)
    )
    
    return instance_train_dl, bag_train, bag_val, bag_test

    
def load_dataloaders(bag_train, bag_val, bag_test):
    # bag_train_dl = DataLoader(bag_train,batch_size = 14, shuffle=False, drop_last=False,collate_fn=collate)
    bag_train_dl = DataLoader(bag_train,batch_size = 28, shuffle=False, drop_last=False,collate_fn=collate)
    bag_val_dl = DataLoader(bag_val,batch_size = 15, shuffle=False, drop_last=False,collate_fn=collate)
    bag_test_dl = DataLoader(bag_test,batch_size = 15, shuffle=False, drop_last=False,collate_fn=collate)
    return bag_train_dl, bag_val_dl, bag_test_dl

# class VQEncoderWrapper(nn.Module):
#     """Wrapper for VQ-AENB to only return features when called."""
#     def __init__(self, vq_aenb_model):
#         super().__init__()
#         self.vq_model = vq_aenb_model
#         self.input_dims = vq_aenb_model.latent_dim  # For compatibility with optimizer.py
    
#     def forward(self, x):
#         # Return only the quantized features, not the full reconstruction
#         return self.vq_model.features(x)
    
#     def parameters(self):
#         # Return only encoder parameters for optimization
#         return self.vq_model.encoder.parameters()

class VQEncoderWrapper(nn.Module):
    """Wrapper for VQ-AENB to only return features when called."""
    def __init__(self, vq_aenb_model, use_projection=False, projection_dim=None):
        super().__init__()
        self.vq_model = vq_aenb_model
        self.input_dims = vq_aenb_model.latent_dim  # For compatibility with optimizer.py
        self.use_projection = use_projection
        
        if use_projection:
            proj_dim = projection_dim or vq_aenb_model.latent_dim
            self.projection = nn.Linear(vq_aenb_model.latent_dim, proj_dim)
            self.input_dims = proj_dim  # Update for downstream compatibility
        else:
            self.projection = None
    
    def forward(self, x):
        encoded = self.vq_model.features(x)
        if hasattr(self, 'projection') and self.projection is not None:
            encoded = self.projection(encoded)
        return encoded
    
    def parameters(self, trainable_only=False):
        """
        Return parameters for optimization.
        If trainable_only=True and projection exists, return only projection params.
        """
        if trainable_only and self.projection is not None:
            return self.projection.parameters()
        else:
            # Return all parameters (encoder + projection if exists)
            return super().parameters()
    
    def freeze_encoder(self):
        """Freeze VQ-AENB encoder parameters, keep projection trainable."""
        for param in self.vq_model.parameters():
            param.requires_grad = False
        if self.projection is not None:
            for param in self.projection.parameters():
                param.requires_grad = True

class VQEncoderWrapperConditional(nn.Module):
    """Wrapper for Conditional VQ-AENB - requires study_ids in forward."""
    def __init__(self, vq_aenb_conditional_model, use_projection=False, projection_dim=None):
        super().__init__()
        self.vq_model = vq_aenb_conditional_model
        self.input_dims = vq_aenb_conditional_model.latent_dim
        self.use_projection = use_projection
        
        if use_projection:
            proj_dim = projection_dim or vq_aenb_conditional_model.latent_dim
            self.projection = nn.Linear(vq_aenb_conditional_model.latent_dim, proj_dim)
            self.input_dims = proj_dim
        else:
            self.projection = None
    
    def forward(self, x, study_ids=None):
        """
        Forward pass.
        Args:
            x: input tensor
            study_ids: required for conditional model, optional fallback to 0
        """
        if study_ids is None:
            # Fallback: use study_id=0 for all samples
            study_ids = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            print("WARNING: study_ids not provided to conditional encoder, using 0")
        
        encoded = self.vq_model.features(x, study_ids)
        
        if self.projection is not None:
            encoded = self.projection(encoded)
        return encoded
    
    def freeze_encoder(self):
        """Freeze VQ-AENB encoder parameters, keep projection trainable."""
        for param in self.vq_model.parameters():
            param.requires_grad = False
        if self.projection is not None:
            for param in self.projection.parameters():
                param.requires_grad = True
                
def load_model_and_optimizer(data_dim, ae_latent_dim, ae_hidden_layers, device, ae_dir, exp, mil_latent_dim,
                            teacher_learning_rate, student_learning_rate, encoder_learning_rate,
                            model_type='AENB', vq_num_codes=256, vq_commitment_weight=0.25, 
                            freeze_encoder=False, use_whole_ae=False, use_projection=False):
    """
    Load pretrained autoencoder and create MIL models and optimizers.
    Supports: AENB, VQ-AENB, VQ-AENB-Conditional
    """
    from src.model import AENB, VQ_AENB, VQ_AENB_Conditional, AttentionModule, TeacherBranch, StudentBranch
    
    model_encoder = None  # Initialize to avoid UnboundLocalError
    
    # VQ-AENB-Conditional
    if model_type == 'VQ-AENB-Conditional':
        if use_whole_ae:
            model_path = f"{ae_dir}/vq_aenb_conditional_whole.pth"
        else:
            model_path = f"{ae_dir}/vq_aenb_conditional_{exp}.pth"
        
        if os.path.exists(model_path):
            # Load metadata to get n_studies
            import json
            metadata_path = os.path.join(os.path.dirname(ae_dir.rstrip('/')), 'conditional_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                n_studies = metadata.get('n_studies', 11)
            else:
                n_studies = 11  # fallback
                print(f"WARNING: metadata not found at {metadata_path}, using n_studies={n_studies}")
            
            ae = VQ_AENB_Conditional(
                input_dim=data_dim, 
                latent_dim=ae_latent_dim,
                n_studies=n_studies,
                device=device, 
                hidden_layers=ae_hidden_layers,
                num_codes=vq_num_codes, 
                commitment_weight=vq_commitment_weight,
                activation_function=nn.Sigmoid
            ).to(device)
            ae.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            print(f"Loaded VQ-AENB-Conditional from {model_path} (n_studies={n_studies})")
            model_encoder = VQEncoderWrapperConditional(ae, use_projection=use_projection)
        else:
            raise FileNotFoundError(f"Conditional model not found at {model_path}")
    
    # VQ-AENB (non-conditional)
    elif model_type == 'VQ-AENB':
        if use_whole_ae:
            vq_model_path = f"{ae_dir}/vq_aenb_whole.pth"
        else:
            vq_model_path = f"{ae_dir}/vq_aenb_{exp}.pth"
        
        if os.path.exists(vq_model_path):
            ae = VQ_AENB(
                input_dim=data_dim, 
                latent_dim=ae_latent_dim,
                device=device, 
                hidden_layers=ae_hidden_layers,
                num_codes=vq_num_codes, 
                commitment_weight=vq_commitment_weight,
                activation_function=nn.Sigmoid
            ).to(device)
            ae.load_state_dict(torch.load(vq_model_path, map_location=device, weights_only=False))
            print(f"Loaded VQ-AENB model from {vq_model_path}")
            model_encoder = VQEncoderWrapper(ae, use_projection=use_projection)
        else:
            print(f"VQ-AENB model file not found at {vq_model_path}, falling back to AENB")
            model_type = 'AENB'
    
    # AENB (fallback or explicit)
    if model_type == 'AENB':
        if use_whole_ae:
            aenb_model_path = f"{ae_dir}/aenb_whole.pth"
        else:
            aenb_model_path = f"{ae_dir}/aenb_{exp}.pth"
        
        if os.path.exists(aenb_model_path):
            ae = AENB(
                input_dim=data_dim, 
                latent_dim=ae_latent_dim,
                device=device, 
                hidden_layers=ae_hidden_layers,
                activation_function=nn.Sigmoid
            ).to(device)
            ae.load_state_dict(torch.load(aenb_model_path, map_location=device, weights_only=False))
            print(f"Loaded AENB model from {aenb_model_path}")
            model_encoder = ae.features
        else:
            raise FileNotFoundError(f"No model file found at {aenb_model_path}")

    # Freeze encoder if specified
    if freeze_encoder:
        if hasattr(model_encoder, 'freeze_encoder'):
            model_encoder.freeze_encoder()
            print("INFO: Encoder frozen, projection layer trainable" if use_projection else "INFO: Encoder frozen")
        else:
            for param in model_encoder.parameters():
                param.requires_grad = False
            print("INFO: Encoder frozen - parameters will not be updated")
    
    model_encoder.to(device)
        
    encoder_dim = ae_latent_dim
    attention_module = AttentionModule(L=encoder_dim, D=encoder_dim, K=1).to(device)
    model_teacher = TeacherBranch(
        input_dims=encoder_dim, 
        latent_dims=mil_latent_dim,
        attention_module=attention_module, 
        num_classes=2, 
        activation_function=nn.Tanh
    )
    model_student = StudentBranch(
        input_dims=encoder_dim, 
        latent_dims=mil_latent_dim,
        num_classes=2, 
        activation_function=nn.Tanh
    )
    
    model_teacher.to(device)
    model_student.to(device)
    
    optimizer_teacher = torch.optim.Adam(model_teacher.parameters(), lr=teacher_learning_rate)
    optimizer_student = torch.optim.Adam(model_student.parameters(), lr=student_learning_rate)
    
    # Optimizer setup
    if freeze_encoder:
        if use_projection and hasattr(model_encoder, 'projection') and model_encoder.projection is not None:
            optimizer_encoder = torch.optim.Adam(model_encoder.projection.parameters(), lr=encoder_learning_rate)
            print("INFO: Optimizer set for projection layer only")
        else:
            optimizer_encoder = None
    else:
        optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=encoder_learning_rate)
    
    return model_teacher, model_student, model_encoder, optimizer_teacher, optimizer_student, optimizer_encoder
    
def dataset_to_dl(dataset, device):
    bag_external = MilDataset(dataset[0].data.to(device), dataset[0].ids.unsqueeze(0).to(device), dataset[0].labels.to(device), dataset[0].instance_labels.to(device))
    external_dl  = DataLoader(bag_external,batch_size = 1, shuffle=False, drop_last=False,collate_fn=collate)
    return external_dl

def add_cell_scores(adata, model_encoder, model_teacher, model_student, device, model_key=None, exp=None, dataset=None):
    if dataset is None:
        dataset, _ = load_mil_dataset_from_adata(adata, device=device)
    
    model_encoder.eval() 
    model_teacher.eval()
    model_student.eval()
    
    # Define column suffix
    parts = []
    if model_key is not None:
        parts.append(model_key)
    if exp is not None:
        parts.append(f'exp{exp}')
    suffix = '_' + '_'.join(parts) if parts else ''    
    
    with torch.no_grad():    
        features = model_encoder(dataset.data.clone().detach().to(device))[:, :model_teacher.input_dims]
        cell_score_teacher = model_teacher.attention_module(features).squeeze(0)
        cell_score_student = model_student(features).squeeze(0)
        cell_score_student = torch.softmax(cell_score_student, dim=1)
    
    cell_score_student = cell_score_student[:, 1]
    features_np = features.cpu().numpy()
    cell_score_teacher_np = cell_score_teacher.cpu().detach().numpy()
    cell_score_teacher_np_minmax = (cell_score_teacher_np - cell_score_teacher_np.min()) / (cell_score_teacher_np.max() - cell_score_teacher_np.min())
    cell_score_student_np = cell_score_student.cpu().detach().numpy()
    cell_score_student_np_minmax = (cell_score_student_np - cell_score_student_np.min()) / (cell_score_student_np.max() - cell_score_student_np.min())

    # Add to adata.obs with suffix
    adata.obs[f'cell_score{suffix}'] = cell_score_teacher_np
    adata.obs[f'cell_score_minmax{suffix}'] = cell_score_teacher_np_minmax
    adata.obs[f'cell_score_cb{suffix}'] = cell_score_student_np
    adata.obs[f'cell_score_cb_minmax{suffix}'] = cell_score_student_np_minmax
    adata.obsm[f'X_scMILD{suffix}'] = features_np
    
    # Create dataframe with suffix
    # import pandas as pd
    # df = pd.DataFrame(features_np, columns=[f'feature_{i}{suffix}' for i in range(features_np.shape[1])], index=adata.obs.index)
    # df[f'cell_score{suffix}'] = cell_score_teacher_np
    # df[f'cell_score_minmax{suffix}'] = cell_score_teacher_np_minmax
    # df[f'cell_score_cb{suffix}'] = cell_score_student_np
    # df[f'cell_score_cb_minmax{suffix}'] = cell_score_student_np_minmax

    return adata # , df
def load_and_save_datasets_adata_conditional(data_dir, exps, device, adata, split_ratio=[0.9, 0.05, 0.05], 
                                              study_col='study', label_col='disease_numeric', sample_col='sample_id_numeric'):
    """
    Save datasets with study_id for Conditional VQ-AENB.
    
    Args:
        data_dir: Directory to save datasets
        exps: List of experiment numbers (for different random splits)
        device: torch device
        adata: AnnData object with expression data
        split_ratio: [train, val, test] ratios
        study_col: Column name for study/batch information
        label_col: Column name for disease labels
        sample_col: Column name for sample IDs
    """
    import os
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Encode study labels to integers
    study_encoder = LabelEncoder()
    study_ids = study_encoder.fit_transform(adata.obs[study_col].values)
    n_studies = len(study_encoder.classes_)
    
    # Save study encoder mapping
    study_mapping = {i: name for i, name in enumerate(study_encoder.classes_)}
    import json
    with open(f"{data_dir}/study_mapping.json", 'w') as f:
        json.dump(study_mapping, f, indent=2)
    print(f"Study mapping saved: {n_studies} studies")
    print(study_mapping)
    
    # Get expression data
    if hasattr(adata.X, 'toarray'):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    labels = adata.obs[label_col].values.astype(int)
    sample_ids = adata.obs[sample_col].values
    
    # Get unique samples for splitting
    unique_samples = np.unique(sample_ids)
    sample_labels = np.array([labels[sample_ids == s][0] for s in unique_samples])
    
    for exp in exps:
        print(f"\nProcessing exp {exp}...")
        np.random.seed(exp)
        
        # Split samples (not cells) to avoid data leakage
        train_samples, temp_samples, _, temp_labels = train_test_split(
            unique_samples, sample_labels, 
            train_size=split_ratio[0], 
            stratify=sample_labels,
            random_state=exp
        )
        
        val_ratio = split_ratio[1] / (split_ratio[1] + split_ratio[2])
        val_samples, test_samples = train_test_split(
            temp_samples,
            train_size=val_ratio,
            stratify=temp_labels,
            random_state=exp
        )
        
        # Get cell indices for each split
        train_idx = np.isin(sample_ids, train_samples)
        val_idx = np.isin(sample_ids, val_samples)
        test_idx = np.isin(sample_ids, test_samples)
        
        # Create datasets
        train_dataset = InstanceDatasetConditional(
            data=torch.tensor(X[train_idx], dtype=torch.float32),
            study_ids=torch.tensor(study_ids[train_idx], dtype=torch.long),
            labels=torch.tensor(labels[train_idx], dtype=torch.long)
        )
        
        val_dataset = InstanceDatasetConditional(
            data=torch.tensor(X[val_idx], dtype=torch.float32),
            study_ids=torch.tensor(study_ids[val_idx], dtype=torch.long),
            labels=torch.tensor(labels[val_idx], dtype=torch.long)
        )
        
        test_dataset = InstanceDatasetConditional(
            data=torch.tensor(X[test_idx], dtype=torch.float32),
            study_ids=torch.tensor(study_ids[test_idx], dtype=torch.long),
            labels=torch.tensor(labels[test_idx], dtype=torch.long)
        )
        
        # Save
        torch.save(train_dataset, f"{data_dir}/train_dataset_conditional_exp{exp}.pt")
        torch.save(val_dataset, f"{data_dir}/val_dataset_conditional_exp{exp}.pt")
        torch.save(test_dataset, f"{data_dir}/test_dataset_conditional_exp{exp}.pt")
        
        print(f"  Train: {len(train_dataset)} cells")
        print(f"  Val: {len(val_dataset)} cells")
        print(f"  Test: {len(test_dataset)} cells")
    
    # Save metadata
    metadata = {
        'n_studies': n_studies,
        'n_genes': X.shape[1],
        'n_cells': X.shape[0],
        'study_col': study_col,
    }
    with open(f"{data_dir}/conditional_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Conditional datasets saved to {data_dir}")
    return n_studies, study_encoder

def calculate_smoothed_disease_ratio(adata, code_col='vq_code', disease_col='disease_numeric', 
                                      alpha=1.0, beta=1.0):
    """
    Bayesian smoothed disease ratio per code.
    Uses Beta prior (alpha, beta) for smoothing.
    
    smoothed_ratio = (n_case + alpha) / (n_total + alpha + beta)
    
    Args:
        adata: AnnData with code assignments
        code_col: column name for code indices
        disease_col: column name for disease labels
        alpha, beta: Beta prior parameters (default: uniform prior)
    
    Returns:
        dict: {code_id: smoothed_ratio}
    """
    code_stats = adata.obs.groupby(code_col)[disease_col].agg(['sum', 'count'])
    code_stats.columns = ['n_case', 'n_total']
    
    # Bayesian smoothing
    code_stats['smoothed_ratio'] = (code_stats['n_case'] + alpha) / (code_stats['n_total'] + alpha + beta)
    
    return code_stats['smoothed_ratio'].to_dict()

# def calculate_smoothed_disease_ratio_from_dataloader(dataloader, model_encoder, device, alpha=1.0, beta=1.0):
#     """
#     Calculate smoothed disease ratio per code from training dataloader.
#     Uses only training split to avoid leakage.
    
#     Args:
#         dataloader: training instance dataloader
#         model_encoder: encoder with get_codebook_indices method
#         device: torch device
#         alpha, beta: Beta prior parameters for smoothing
    
#     Returns:
#         dict: {code_id: smoothed_ratio}
#     """
#     import torch
#     from collections import defaultdict
    
#     code_case_count = defaultdict(int)
#     code_total_count = defaultdict(int)
    
#     model_encoder.eval()
    
#     with torch.no_grad():
#         for batch in dataloader:
#             # instance_train_dl: (data, bag_ids, bag_labels, instance_labels)
#             data = batch[0].to(device)
#             instance_labels = batch[3].to(device)  # disease label at instance level
            
#             # Get code indices
#             if hasattr(model_encoder, 'vq_model'):
#                 # VQEncoderWrapper
#                 code_indices = model_encoder.vq_model.get_codebook_indices(data)
#             elif hasattr(model_encoder, 'get_codebook_indices'):
#                 code_indices = model_encoder.get_codebook_indices(data)
#             else:
#                 print("WARNING: Cannot get code indices, returning None")
#                 return None
            
#             # Count per code
#             for code, label in zip(code_indices.cpu().numpy(), instance_labels.cpu().numpy()):
#                 code_total_count[int(code)] += 1
#                 if label == 1:
#                     code_case_count[int(code)] += 1
    
#     # Bayesian smoothing
#     smoothed_ratio = {}
#     for code in code_total_count:
#         n_case = code_case_count[code]
#         n_total = code_total_count[code]
#         smoothed_ratio[code] = (n_case + alpha) / (n_total + alpha + beta)
    
#     return smoothed_ratio

def calculate_smoothed_disease_ratio_from_dataloader(dataloader, model_encoder, device, alpha=1.0, beta=1.0):
    """
    Calculate smoothed disease ratio per code from training dataloader.
    Uses only training split to avoid leakage.
    
    Args:
        dataloader: training instance dataloader
        model_encoder: encoder with get_codebook_indices method
        device: torch device
        alpha, beta: Beta prior parameters for smoothing
    
    Returns:
        dict: {code_id: smoothed_ratio}
    """
    import torch
    from collections import defaultdict
    
    code_case_count = defaultdict(int)
    code_total_count = defaultdict(int)
    
    model_encoder.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            # 배치 언패킹: (data, bag_ids, bag_labels, instance_labels) 또는 (+ study_ids)
            data = batch[0].to(device)
            # batch[1]: bag_ids, batch[2]: bag_labels (InstanceDL 기준)
            
            # Instance Label 위치 확인 (보통 3번째)
            if len(batch) >= 4:
                instance_labels = batch[3].to(device)
            else:
                # InstanceDataset 구조에 따라 다를 수 있으므로 확인 필요
                # 여기서는 InstanceDataset의 __getitem__ 순서를 따름
                instance_labels = batch[2].to(device) 

            # [FIX] Conditional 처리를 위한 Study ID 추출
            study_ids = None
            if len(batch) == 5: # (data, bag_id, instance_label, bag_label, study_id) 가정
                study_ids = batch[4].to(device)
            elif len(batch) == 4 and isinstance(batch[3], torch.Tensor) and batch[3].dtype == torch.long:
                 # InstanceDataset 구조가 (data, bag_id, instance_label, study_id) 인 경우 등 체크 필요
                 # 안전하게는 collate 함수 구조를 따름. 
                 # 만약 study_id가 있다면 보통 마지막에 위치함.
                 pass

            # Get code indices
            if hasattr(model_encoder, 'vq_model'):
                # VQEncoderWrapper or VQEncoderWrapperConditional
                if study_ids is not None:
                    # [FIX] Study ID 전달
                    code_indices = model_encoder.vq_model.get_codebook_indices(data, study_ids)
                else:
                    # Study ID가 없으면 data만 전달 (Non-conditional or Fallback)
                    # Conditional 모델인데 study_ids가 없으면 내부에서 0으로 처리하거나 에러 발생
                    if 'study_ids' in model_encoder.vq_model.get_codebook_indices.__code__.co_varnames:
                         # Conditional 모델인데 study_id가 없는 경우
                         print("WARNING: Conditional model detected but no study_ids in batch.")
                         # 임시로 0 전달 혹은 data만 전달 (모델 구현에 따름)
                         code_indices = model_encoder.vq_model.get_codebook_indices(data, torch.zeros(data.size(0), dtype=torch.long, device=device))
                    else:
                        code_indices = model_encoder.vq_model.get_codebook_indices(data)

            elif hasattr(model_encoder, 'get_codebook_indices'):
                 if study_ids is not None:
                    code_indices = model_encoder.get_codebook_indices(data, study_ids)
                 else:
                    code_indices = model_encoder.get_codebook_indices(data)
            else:
                print("WARNING: Cannot get code indices, returning None")
                return None
            
            # Count per code
            for code, label in zip(code_indices.cpu().numpy(), instance_labels.cpu().numpy()):
                code_total_count[int(code)] += 1
                if label == 1:
                    code_case_count[int(code)] += 1
    
    # Bayesian smoothing
    smoothed_ratio = {}
    for code in code_total_count:
        n_case = code_case_count[code]
        n_total = code_total_count[code]
        smoothed_ratio[code] = (n_case + alpha) / (n_total + alpha + beta)
    
    return smoothed_ratio