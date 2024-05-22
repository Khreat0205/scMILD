# scMILD: Single-cell Multiple Instance Learning for Disease classification and associated subpopulation Discovery

We propose scMILD, a weakly supervised learning framework based on Multiple Instance Learning (MIL), which leverages sample-level labels to identify condition-associated cell subpopulations. By treating samples as bags and cells as instances, scMILD effectively learns cell-level representations and improves sample classification performance.

## Data Preprocessing (*.h5ad)

`preprocess_adata.py`: This script preprocesses the data and saves split datasets.

### Usage
```bash
python preprocess_data.py [--data_dir DATA_DIR] [--data_dim DATA_DIM] 
                          [--obs_name_sample_label OBS_NAME_SAMPLE_LABEL]
                          [--obs_name_sample_id OBS_NAME_SAMPLE_ID] 
                          [--obs_name_cell_type OBS_NAME_CELL_TYPE]
                          [--sample_label_negative SAMPLE_LABEL_NEGATIVE]
                          [--sample_label_positive SAMPLE_LABEL_POSITIVE]
                          [--device_num DEVICE_NUM] [--n_exp N_EXP]
```

### Arguments
- `--data_dir`: Directory containing adata.h5ad (default: "data/MyData")
- `--data_dim`: Number of top Highly varialbe genes to select (default: 2000)
- `--obs_name_sample_label`: Obs column name for sample labels (default: 'Health')
- `--obs_name_sample_id`: Obs column name for sample IDs (default: 'Subject')
- `--obs_name_cell_type`: Obs column name for cell types (default: 'Cluster')
- `--sample_label_negative`: Negative sample label (default: 'Healthy')
- `--sample_label_positive`: Positive sample label (default: 'Inflamed')
- `--device_num`: CUDA device number (default: 6)
- `--n_exp`: Number of experiments (default: 8)

### Example
```bash
python preprocess_data.py --data_dir data/MyData --data_dim 2000 --obs_name_sample_label Health --obs_name_sample_id Subject --obs_name_cell_type Cluster --sample_label_negative Healthy --sample_label_positive Inflamed --device_num 6 --n_exp 8
```

## Pretraining Autoencoder
`pretraining_autoencoder.py`: This script performs pretraining of an autoencoder model and saves the trained model.

### Usage
```bash
python pretraining_autoencoder.py [--dataset DATASET] [--device_num DEVICE_NUM]
                                  [--ae_learning_rate AE_LEARNING_RATE] [--ae_epochs AE_EPOCHS]
                                  [--ae_patience AE_PATIENCE] [--ae_latent_dim AE_LATENT_DIM]
                                  [--ae_hidden_layers AE_HIDDEN_LAYERS [AE_HIDDEN_LAYERS ...]]
                                  [--ae_batch_size AE_BATCH_SIZE] [--data_dim DATA_DIM]
                                  [--n_exp N_EXP]
```

### Arguments
- `--dataset`: Dataset name (default: "MyData")
- `--device_num`: CUDA device number (default: 6)
- `--ae_learning_rate`: Learning rate for autoencoder (default: 1e-3)
- `--ae_epochs`: Number of epochs for autoencoder training (default: 25)
- `--ae_patience`: Patience for early stopping (default: 3)
- `--ae_latent_dim`: Latent dimension for autoencoder (default: 128)
- `--ae_hidden_layers`: Hidden layers for autoencoder (default: [512, 256, 128])
- `--ae_batch_size`: Batch size for autoencoder (default: 128)
- `--data_dim`: Data dimension (default: 2000)
- `--n_exp`: Number of experiments (default: 8)

### Example
```bash
python pretraining_autoencoder.py --dataset MyData --device_num 6 --ae_learning_rate 1e-3 --ae_epochs 25 --ae_patience 3 --ae_latent_dim 128 --ae_hidden_layers 512 256 128 --ae_batch_size 128 --data_dim 2000 --n_exp 8
```



# Contact
- scientist0205@snu.ac.kr

