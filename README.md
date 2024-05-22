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
- `--data_dim`: Number of top genes to select (default: 2000)
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




# Contact
scientist0205@snu.ac.kr

