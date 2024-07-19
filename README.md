# scMILD: Single-cell Multiple Instance Learning for Sample Classification and Associated Subpopulation Discovery

We propose scMILD, a weakly supervised learning framework based on Multiple Instance Learning (MIL), which leverages sample-level labels to identify condition-associated cell subpopulations. By treating samples as bags and cells as instances, scMILD effectively learns cell-level representations and improves sample classification performance.

This repository contains the implementation of scMILD. The pipeline consists of three main steps: data preprocessing, autoencoder pretraining, and scMILD training.

## Data Preparation

The input data should be in the form of a single-cell gene expression matrix stored in the AnnData format (`.h5ad` file). The AnnData object should have the following structure:

- `adata.X`: The gene expression matrix, where rows correspond to cells and columns correspond to genes.
- `adata.obs`: A dataframe containing cell metadata, including the following columns:
  - Sample label column (e.g., 'Health'): Indicates the sample label for each cell (e.g., 'Healthy', 'Inflamed').
  - Sample ID column (e.g., 'Subject'): Indicates the sample ID for each cell.
  - Cell type column (e.g., 'Cluster'): Indicates the cell type for each cell.

Place the `.h5ad` file in the `data` directory with a specific dataset name (e.g., `data/MyData/adata.h5ad`).

## Pipeline

### 1. Data Preprocessing

Run the `preprocess_adata.py` script to preprocess the data and save split datasets:

```bash
python preprocess_data.py --data_dir data/MyData --data_dim 2000 --obs_name_sample_label Health --obs_name_sample_id Subject --obs_name_cell_type Cluster --sample_label_negative Healthy --sample_label_positive Inflamed --device_num 6 --n_exp 8
```
This script performs the following steps:
- Normalization and log transformation of the gene expression data.
- Selection of highly variable genes.
- Subset the data to include only the specified sample labels.
- One-hot encoding of sample labels.
- Splitting the data into train, validation, and test sets for multiple experiments.

The preprocessed datasets will be saved in the `data/MyData` directory.

#### Data Preprocessing Script

`preprocess_adata.py`: This script preprocesses the data and saves split datasets.

**Usage**
```bash
python preprocess_data.py [--data_dir DATA_DIR] [--data_dim DATA_DIM] 
                          [--obs_name_sample_label OBS_NAME_SAMPLE_LABEL]
                          [--obs_name_sample_id OBS_NAME_SAMPLE_ID] 
                          [--obs_name_cell_type OBS_NAME_CELL_TYPE]
                          [--sample_label_negative SAMPLE_LABEL_NEGATIVE]
                          [--sample_label_positive SAMPLE_LABEL_POSITIVE]
                          [--device_num DEVICE_NUM] [--n_exp N_EXP]
```

**Arguments**
- `--data_dir`: Directory containing adata.h5ad (default: "data/MyData")
- `--data_dim`: Number of top Highly varialbe genes to select (default: 2000)
- `--obs_name_sample_label`: Obs column name for sample labels (default: 'Health')
- `--obs_name_sample_id`: Obs column name for sample IDs (default: 'Subject')
- `--obs_name_cell_type`: Obs column name for cell types (default: 'Cluster')
- `--sample_label_negative`: Negative sample label (default: 'Healthy')
- `--sample_label_positive`: Positive sample label (default: 'Inflamed')
- `--device_num`: CUDA device number (default: 6)
- `--n_exp`: Number of experiments (default: 8)


**Example**
```bash
python preprocess_data.py --data_dir data/MyData --data_dim 2000 --obs_name_sample_label Health --obs_name_sample_id Subject --obs_name_cell_type Cluster --sample_label_negative Healthy --sample_label_positive Inflamed --device_num 6 --n_exp 8
```

### 2. Autoencoder Pretraining

Run the `pretraining_autoencoder.py` script to pretrain an autoencoder model:

```bash
python pretraining_autoencoder.py --data_dir data/MyData --device_num 6 --ae_learning_rate 1e-3 --ae_epochs 25 --ae_patience 3 --ae_latent_dim 128 --ae_hidden_layers 512 256 128 --ae_batch_size 128 --data_dim 2000 --n_exp 8
```

This script trains an autoencoder model on the preprocessed data. The trained autoencoder model will be saved in the `data/MyData/AE` directory.

#### Autoencoder Pretraining Script

`pretraining_autoencoder.py`: This script performs pretraining of an autoencoder model and saves the trained model.

**Usage**
```bash
python pretraining_autoencoder.py [--data_dir DATA_DIR] [--device_num DEVICE_NUM]
                                  [--ae_learning_rate AE_LEARNING_RATE] [--ae_epochs AE_EPOCHS]
                                  [--ae_patience AE_PATIENCE] [--ae_latent_dim AE_LATENT_DIM]
                                  [--ae_hidden_layers AE_HIDDEN_LAYERS [AE_HIDDEN_LAYERS ...]]
                                  [--ae_batch_size AE_BATCH_SIZE] [--data_dim DATA_DIM]
                                  [--n_exp N_EXP]
```

**Arguments**
- `--data_dir`: Data directory (default: "data/MyData")
- `--device_num`: CUDA device number (default: 6)
- `--ae_learning_rate`: Learning rate for autoencoder (default: 1e-3)
- `--ae_epochs`: Number of epochs for autoencoder training (default: 25)
- `--ae_patience`: Patience for early stopping (default: 3)
- `--ae_latent_dim`: Latent dimension for autoencoder (default: 128)
- `--ae_hidden_layers`: Hidden layers for autoencoder (default: [512, 256, 128])
- `--ae_batch_size`: Batch size for autoencoder (default: 128)
- `--data_dim`: Data dimension (default: 2000)
- `--n_exp`: Number of experiments (default: 8)

**Example**
```bash
python pretraining_autoencoder.py --data_dir data/MyData --device_num 6 --ae_learning_rate 1e-3 --ae_epochs 25 --ae_patience 3 --ae_latent_dim 128 --ae_hidden_layers 512 256 128 --ae_batch_size 128 --data_dim 2000 --n_exp 8
```

### 3. scMILD Training

Run the `train_scMILD.py` script to train the scMILD model:

```bash
python train_scMILD.py --data_dir data/MyData --dataset MyData --device_num 6 --data_dim 2000 --mil_latent_dim 64 --cell_batch_size 256 --sample_learning_rate 1e-3 --cell_learning_rate 1e-3 --encoder_learning_rate 1e-3 --scMILD_epoch 500 --scMILD_neg_weight 0.3 --scMILD_stuOpt 3 --scMILD_patience 15 --add_suffix reported --n_exp 8
```

This script trains the scMILD model using the pretrained autoencoder. The trained scMILD model and evaluation results will be saved in the `results` directory.

#### scMILD Training Script

`train_scMILD.py`: This script performs training of the scMILD model and saves the trained model.

**Usage**
```bash
python train_scMILD.py [--data_dir DATA_DIR] [--dataset DATASET] 
                       [--device_num DEVICE_NUM] [--data_dim DATA_DIM]
                       [--mil_latent_dim MIL_LATENT_DIM] [--cell_batch_size cell_BATCH_SIZE]
                       [--sample_learning_rate SAMPLE_LEARNING_RATE]
                       [--cell_learning_rate CELL_LEARNING_RATE]
                       [--encoder_learning_rate ENCODER_LEARNING_RATE]
                       [--scMILD_epoch SCMILD_EPOCH] [--scMILD_neg_weight SCMILD_NEG_WEIGHT]
                       [--scMILD_stuOpt SCMILD_STUOPT] [--scMILD_patience SCMILD_PATIENCE]
                       [--add_suffix ADD_SUFFIX] [--n_exp N_EXP] [--exp EXP] [--gmm GMM] [--op_lambda OP_LAMBDA]
```

**Arguments**
- `--data_dir`: Data directory (default: "data/MyData")
- `--dataset`: Dataset name (default: "MyData")
- `--device_num`: CUDA device number (default: 6)
- `--data_dim`: Data dimension (default: 2000)
- `--mil_latent_dim`: Latent dimension for MIL (default: 64)
- `--cell_batch_size`: Batch size for cell (default: 256)
- `--sample_learning_rate`: Learning rate for sample (default: 1e-3)
- `--cell_learning_rate`: Learning rate for cell (default: 1e-3)
- `--encoder_learning_rate`: Learning rate for encoder (default: 1e-3)
- `--scMILD_epoch`: Number of epochs for scMILD (default: 500)
- `--scMILD_neg_weight`: Negative weight for scMILD (default: 0.3)
- `--scMILD_stuOpt`: cell optimization period (default: 3)
- `--scMILD_patience`: Patience for early stopping (default: 15)
- `--add_suffix`: Suffix for output files (default: "reported")
- `--n_exp`: Number of experiments (default: 8)
- `--exp`: Experiment number (if None, all experiments will be run) (default: None)
- `--gmm`: Use GMM (default: True)
- `--op_lambda`: Lambda for orthogonal projection loss (default: 0.5)

**Example**
Run all experiments:
```bash
python train_scMILD.py --data_dir data/MyData --dataset MyData --device_num 6 --data_dim 2000 --mil_latent_dim 64 --cel_batch_size 256 --sample_learning_rate 1e-3 --cel_learning_rate 1e-3 --encoder_learning_rate 1e-3 --scMILD_epoch 500 --scMILD_neg_weight 0.3 --scMILD_stuOpt 3 --scMILD_patience 15 --add_suffix reported --n_exp 8
```

Run a single experiment (e.g., experiment 3):
```bash
python train_scMILD.py --data_dir data/MyData --dataset MyData --device_num 6 --data_dim 2000 --mil_latent_dim 64 --cel_batch_size 256 --sample_learning_rate 1e-3 --cell_learning_rate 1e-3 --encoder_learning_rate 1e-3 --scMILD_epoch 500 --scMILD_neg_weight 0.3 --scMILD_stuOpt 3 --scMILD_patience 15 --add_suffix reported --n_exp 8 --exp 3
```

# Contact
- Kyeonghun Jeong, scientist0205@snu.ac.kr
