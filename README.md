# scMILD: Single-cell Multiple Instance Learning for Sample Classification and Associated Subpopulation Discovery

scMILD는 Multiple Instance Learning (MIL) 기반의 약지도 학습 프레임워크로, 샘플 레벨 라벨을 활용하여 질병 관련 세포 하위집단을 식별합니다. 샘플을 bag으로, 세포를 instance로 처리하여 세포 레벨 표현을 학습하고 샘플 분류 성능을 향상시킵니다.

## 주요 특징

- **VQ-AENB-Conditional**: Vector Quantized Autoencoder with Negative Binomial loss, study/batch 조건부 임베딩 지원
- **Teacher-Student 구조**: 샘플 레벨 분류(Teacher)와 세포 레벨 점수(Student) 동시 학습
- **LOOCV 지원**: Leave-One-Out Cross Validation으로 소규모 샘플 데이터셋 평가
- **Cross-disease 일반화**: 한 질병에서 학습한 모델로 다른 질병 평가
- **Disease Ratio Regularization**: VQ 코드북 기반 attention score 정규화

## 프로젝트 구조

```
scMILD/
├── config/                   # YAML 설정 파일
│   ├── default.yaml          # 기본 설정
│   ├── skin3.yaml            # Skin3 (HS 분류) 설정
│   └── scp1884.yaml          # SCP1884 (CD 분류) 설정
│
├── src/                      # 소스 코드
│   ├── config.py             # 설정 관리
│   ├── models/               # 모델 정의
│   │   ├── attention.py      # Gated Attention Module
│   │   ├── branches.py       # Teacher/Student Branch
│   │   ├── autoencoder.py    # VQ-AENB, VQ-AENB-Conditional
│   │   ├── encoder_wrapper.py # Encoder 래퍼
│   │   └── quantizer.py      # Vector Quantizer
│   ├── data/                 # 데이터 처리
│   │   ├── dataset.py        # MilDataset, InstanceDataset
│   │   ├── splitter.py       # LOOCV, StratifiedKFold
│   │   └── preprocessing.py  # AnnData 전처리
│   └── training/             # 학습 모듈
│       ├── trainer.py        # MILTrainer
│       ├── trainer_ae.py     # AETrainer
│       ├── metrics.py        # 평가 메트릭
│       └── disease_ratio.py  # Disease Ratio Regularization
│
├── scripts/                  # 실행 스크립트
│   ├── 01_pretrain_encoder.py    # Encoder pretrain
│   ├── 02_train_loocv.py         # LOOCV 학습
│   ├── 03_finalize_model.py      # Final model 학습
│   ├── 04_cross_disease_eval.py  # Cross-disease 평가
│   └── 05_cell_scoring.py        # Cell-level 점수 계산
│
├── legacy/                   # 기존 코드 (참조용)
├── results/                  # 결과 저장 (gitignore)
└── docs/                     # 문서
```

## 설치

```bash
# 저장소 클론
git clone https://github.com/Khreat0205/scMILD.git
cd scMILD

# 환경 설정 (conda 권장)
conda create -n scmild python=3.12
conda activate scmild

# 의존성 설치
pip install torch torchvision scanpy anndata scikit-learn pandas numpy pyyaml
```

## 데이터 준비

입력 데이터는 AnnData 형식(`.h5ad`)이어야 합니다:

```python
adata.X        # 유전자 발현 매트릭스 (cells × genes)
adata.obs      # 세포 메타데이터:
  - sample_id_numeric   # 샘플 ID (숫자)
  - disease_numeric     # 질병 라벨 (0: Control, 1: Disease)
  - study_id_numeric    # Study ID (conditional encoder용)
  - sample              # 샘플 이름 (문자열)
  - Status              # 질병 상태 (문자열)
```

## 사용법

### 1. 설정 파일 준비

데이터셋별 설정 파일 생성 (`config/your_dataset.yaml`):

```yaml
_base_: "default.yaml"

paths:
  data_root: "/path/to/your/data"
  pretrained_encoder: "${paths.data_root}/AE/vq_aenb_conditional_whole.pth"

data:
  adata_path: "${paths.data_root}/adata.h5ad"

  conditional_embedding:
    column: "study"                    # 또는 "Organ"
    encoded_column: "study_id_numeric"
```

### 2. Encoder Pretrain

```bash
python scripts/01_pretrain_encoder.py \
    --config config/your_dataset.yaml \
    --gpu 0
```

### 3. LOOCV 학습

```bash
python scripts/02_train_loocv.py \
    --config config/your_dataset.yaml \
    --gpu 0
```

### 4. Final Model 학습

```bash
python scripts/03_finalize_model.py \
    --config config/your_dataset.yaml \
    --gpu 0
```

### 5. Cross-disease 평가

```bash
python scripts/04_cross_disease_eval.py \
    --model_dir results/final_model_xxx \
    --source_config config/skin3.yaml \
    --target_config config/scp1884.yaml \
    --gpu 0
```

### 6. Cell-level 점수 계산

```bash
python scripts/05_cell_scoring.py \
    --model_dir results/final_model_xxx \
    --config config/your_dataset.yaml \
    --gpu 0
```

## 주요 설정 옵션

### MIL 학습 설정 (`config/default.yaml`)

```yaml
mil:
  freeze_encoder: true        # Encoder 고정 여부
  use_projection: true        # Projection layer 사용

  training:
    epochs: 100
    learning_rate: 0.0001
    use_early_stopping: false  # LOOCV에서는 false 권장

  loss:
    negative_weight: 0.3
    disease_ratio_reg:
      enabled: false           # Disease ratio regularization
      lambda_weight: 0.1
```

### Disease Ratio Regularization

VQ 코드북의 각 코드별 질병 비율을 타겟으로 attention score를 정규화:

```yaml
mil:
  loss:
    disease_ratio_reg:
      enabled: true
      lambda_weight: 0.1      # 0.05 ~ 0.2 권장
      alpha: 1.0              # Beta prior smoothing
      beta: 1.0
```

## 출력 결과

### LOOCV 결과 (`results/loocv_*/`)
```
loocv_20250129_123456/
├── results.csv           # 폴드별 메트릭 (AUC, Accuracy, F1)
└── models/               # 폴드별 모델 체크포인트
```

### Cell Scoring 결과 (`results/cell_scores_*/`)
```
cell_scores_20250129_123456/
├── cell_scores.csv       # 세포별 attention score, prediction
└── sample_summary.csv    # 샘플별 요약
```

## 참고

- 기존 스크립트(`preprocess_adata.py`, `train_scMILD.py` 등)는 `legacy/` 폴더에 보관
- 새 파이프라인은 YAML 설정 기반으로 단순화됨

## Contact

- Kyeonghun Jeong, scientist0205@snu.ac.kr
