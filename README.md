# scMILD: Single-cell Multiple Instance Learning for Disease Classification

scMILD는 Multiple Instance Learning (MIL) 기반의 약지도 학습 프레임워크로, 샘플 레벨 라벨을 활용하여 질병 분류 및 질병 관련 세포 하위집단을 식별합니다.

## 주요 특징

- **VQ-AENB-Conditional**: Vector Quantized Autoencoder with Negative Binomial loss, study/batch 조건부 임베딩
- **Teacher-Student MIL**: Bag(샘플) 레벨 분류 + Instance(세포) 레벨 점수 동시 학습
- **LOOCV 지원**: Leave-One-Out Cross Validation으로 소규모 샘플 평가
- **Cross-disease 일반화**: 한 질병에서 학습 → 다른 질병 평가
- **Disease Ratio Regularization**: VQ 코드북 기반 attention 정규화
- **Grid Search 튜닝**: 하이퍼파라미터 자동 최적화

## 프로젝트 구조

```
scMILD/
├── config/                   # YAML 설정 파일
│   ├── default.yaml          # 기본 설정
│   ├── skin3.yaml            # Skin3 (HS) 설정
│   └── scp1884.yaml          # SCP1884 (CD) 설정
│
├── src/                      # 소스 코드
│   ├── config.py             # 설정 관리 (dataclass + YAML)
│   ├── models/               # 모델 정의
│   │   ├── attention.py      # Gated Attention Module
│   │   ├── branches.py       # Teacher/Student Branch
│   │   ├── autoencoder.py    # VQ-AENB-Conditional
│   │   ├── encoder_wrapper.py
│   │   └── quantizer.py
│   ├── data/                 # 데이터 처리
│   │   ├── dataset.py        # MilDataset, InstanceDataset
│   │   ├── splitter.py       # LOOCV, StratifiedKFold
│   │   └── preprocessing.py  # AnnData 로딩/전처리
│   └── training/             # 학습 모듈
│       ├── trainer.py        # MILTrainer
│       ├── trainer_ae.py     # AETrainer (pretrain)
│       ├── metrics.py        # 평가 메트릭
│       └── disease_ratio.py  # Disease Ratio Regularization
│
├── scripts/                  # 실행 스크립트
│   ├── 01_pretrain_encoder.py    # Encoder pretrain (전체 데이터)
│   ├── 02_train_loocv.py         # LOOCV 학습 (기본 파라미터)
│   ├── 03_finalize_model.py      # Final model 학습 (best params)
│   ├── 04_cross_disease_eval.py  # Cross-disease 평가
│   ├── 05_cell_scoring.py        # Cell-level scoring
│   └── 06_tune_hyperparams.py    # Grid Search 튜닝
│
├── notebooks/                # 분석 노트북
│   ├── 01_vq_embedding_analysis.ipynb   # VQ code/embedding 추출
│   └── 02_codebook_visualization.ipynb  # Codebook 시각화 및 클러스터링
│
├── CLAUDE.md                 # Claude AI 컨텍스트
├── legacy/                   # 기존 코드 (참조용)
└── results/                  # 결과 저장 (gitignore)
```

## 설치

```bash
# 저장소 클론
git clone -b Quant_Conditioned https://github.com/Khreat0205/scMILD.git
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
  - Status              # 질병 상태 (HS, CD, ctrl_skin 등)
```

### 지원 데이터셋

| Dataset | Disease | Studies | Samples | Cells |
|---------|---------|---------|---------|-------|
| Skin3 | HS (Hidradenitis Suppurativa) | GSE175990, GSE220116 | 19 | ~36k |
| SCP1884 | CD (Crohn's Disease) | SCP1884 | 34 | ~290k |

## 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         전체 파이프라인 흐름                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Pretrain Encoder (전체 데이터, unsupervised)                         │
│         ↓                                                               │
│  2. LOOCV 학습 (기본 파라미터로 baseline 확인)                            │
│         ↓                                                               │
│  3. 하이퍼파라미터 튜닝 (LOOCV 기반 grid search)                          │
│         ↓ best_params.yaml                                              │
│  4. Finalize Model (best params로 전체 subset 학습)                      │
│         ↓ final_model                                                   │
│  ┌──────┴──────┐                                                        │
│  ↓             ↓                                                        │
│  5a. Cell Scoring     5b. Cross-disease 평가                            │
│  (학습 데이터)         (다른 질병 데이터)                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Cell Scoring 주의사항

| 데이터 유형 | 사용 모델 | 이유 |
|-------------|----------|------|
| **학습 데이터** (예: Skin3) | LOOCV fold 모델들 | 각 샘플은 해당 샘플 제외 모델로 scoring (data leakage 방지) |
| **평가 데이터** (예: SCP1884) | Final model | 학습에 포함되지 않았으므로 final model 사용 |

## 사용법

### Quick Start

```bash
# Pretrained encoder가 있는 경우, 바로 LOOCV 학습
python scripts/02_train_loocv.py --config config/skin3.yaml --gpu 0
```

### 전체 파이프라인

#### 1. Encoder Pretrain (필수 - 처음 한 번)

```bash
python scripts/01_pretrain_encoder.py --config config/default.yaml --gpu 0
```

Pretrain 완료 후, 결과물을 `pretrained/` 폴더로 복사:

```bash
# 모델 파일 복사
cp results/pretrain_YYYYMMDD_HHMMSS/vq_aenb_conditional.pth \
   results/pretrained/vq_aenb_conditional_whole.pth

# study_mapping.json 복사 (중요! subset 학습 시 필요)
cp results/pretrain_YYYYMMDD_HHMMSS/study_mapping.json \
   results/pretrained/study_mapping.json
```

#### 2. LOOCV 학습 (Baseline)

```bash
# Skin3 (HS 분류)
python scripts/02_train_loocv.py --config config/skin3.yaml --gpu 0

# SCP1884 (CD 분류)
python scripts/02_train_loocv.py --config config/scp1884.yaml --gpu 0
```

#### 3. 하이퍼파라미터 튜닝

```bash
python scripts/06_tune_hyperparams.py --config config/skin3.yaml --gpu 0 --verbose
```

출력: `results/skin3/tuning_YYYYMMDD_HHMMSS/best_params.yaml`

#### 4. Final Model 학습

```bash
# best_params.yaml을 사용하여 전체 데이터로 학습
python scripts/03_finalize_model.py \
    --config config/skin3.yaml \
    --best_params results/skin3/tuning_YYYYMMDD_HHMMSS/best_params.yaml \
    --gpu 0
```

출력: `results/skin3/final_model_YYYYMMDD_HHMMSS/`

#### 5. Cell Scoring

```bash
# 학습 데이터 (Skin3) - LOOCV 모델 사용
python scripts/05_cell_scoring.py \
    --loocv_dir results/skin3/loocv_YYYYMMDD_HHMMSS \
    --config config/skin3.yaml \
    --gpu 0

# 평가 데이터 (SCP1884) - Final model 사용
python scripts/05_cell_scoring.py \
    --model_dir results/skin3/final_model_YYYYMMDD_HHMMSS \
    --config config/scp1884.yaml \
    --gpu 0
```

#### 6. Cross-disease 평가

```bash
# Skin3에서 학습한 모델로 SCP1884 평가
python scripts/04_cross_disease_eval.py \
    --model_dir results/skin3/final_model_YYYYMMDD_HHMMSS \
    --test_config config/scp1884.yaml \
    --gpu 0
```

## 설정 시스템

### Config 상속

```yaml
# config/skin3.yaml
_base_: "default.yaml"  # default.yaml 설정 상속

paths:
  output_root: "${paths.data_root}/results/skin3"

data:
  subset:
    enabled: true
    values: ["GSE175990", "GSE220116"]  # 이 study만 사용
```

### 주요 설정 옵션

```yaml
# MIL 학습 설정
mil:
  freeze_encoder: true
  use_projection: true

  training:
    learning_rate: 0.0001
    encoder_learning_rate: 0.0005
    epochs: 100
    use_early_stopping: false  # LOOCV에서는 false 권장

  loss:
    negative_weight: 0.3
    disease_ratio_reg:
      enabled: false
      lambda_weight: 0.1

# 하이퍼파라미터 튜닝 설정
tuning:
  enabled: false
  learning_rate: [0.001, 0.0001]
  encoder_learning_rate: [0.001, 0.0001]
  epochs: [10, 30]  # Transfer learning이므로 작은 epoch 권장
  disease_ratio_lambda: [0.0, 0.05, 0.1]
  metric: "auc"
  save_top_k: 3  # 상위 K개 조합 모델 저장
```

## 출력 결과

### 디렉토리 구조

```
results/
├── pretrained/                        # Pretrained encoder (수동 배치)
│   ├── vq_aenb_conditional_whole.pth
│   └── study_mapping.json
│
├── skin3/
│   ├── loocv_YYYYMMDD_HHMMSS/         # LOOCV 결과
│   │   ├── results.csv                # Fold별 정보
│   │   ├── overall_results.csv        # 전체 AUROC
│   │   ├── predictions.csv            # 샘플별 예측
│   │   └── models/                    # Fold별 모델 (cell scoring용)
│   │       ├── model_teacher_fold0.pt
│   │       ├── model_student_fold0.pt
│   │       └── ...
│   │
│   ├── tuning_YYYYMMDD_HHMMSS/        # 튜닝 결과
│   │   ├── tuning_results.csv         # 전체 결과 (실시간 저장)
│   │   ├── best_params.yaml           # 최적 하이퍼파라미터
│   │   └── models/                    # Top-K 모델
│   │
│   ├── final_model_YYYYMMDD_HHMMSS/   # Final model
│   │   ├── model_teacher_fold0.pt
│   │   ├── model_student_fold0.pt
│   │   ├── model_encoder_fold0.pt
│   │   └── model_info.json
│   │
│   └── cell_scores_YYYYMMDD_HHMMSS/   # Cell scoring 결과
│       ├── cell_scores.csv
│       └── cell_scores.h5ad
│
├── scp1884/
│   └── ...
│
└── vq_analysis/                       # VQ 분석 결과 (노트북)
    ├── adata_with_vq.h5ad
    ├── codebook_stats_whole.csv
    └── figures/
```

### LOOCV 평가 메트릭

LOOCV에서는 각 fold의 테스트 샘플이 1개이므로 fold별 AUC가 무의미합니다.
대신 **전체 예측을 concatenate하여 AUROC를 계산**합니다:

- `overall_results.csv`: 전체 AUROC, Accuracy, F1 Score
- `predictions.csv`: 각 샘플의 예측 확률

## 아키텍처

```
[scRNA-seq Data]
       ↓
[VQ-AENB-Conditional Encoder] ← Pretrained (frozen)
       ↓
[Projection Layer] ← Trainable
       ↓
[Gated Attention MIL]
    ↓         ↓
[Teacher]  [Student]
(Bag-level) (Instance-level)
```

## 분석 노트북

### 01_vq_embedding_analysis.ipynb
- 전체 adata에 VQ code 및 embedding 추출
- Codebook statistics 계산 (disease ratio, organ ratio 등)
- Subset별 분석

### 02_codebook_visualization.ipynb
- Codebook PCA/UMAP 시각화
- Codebook leiden clustering → cell-level 적용
- Special codes 식별 (cross-tissue, disease-specific 등)
- Code-Sample heatmap

## 참고

- **CLAUDE.md**: Claude AI가 프로젝트를 이해하기 위한 컨텍스트 파일
- **legacy/**: 기존 스크립트 (참조용)

## Contact

- Kyeonghun Jeong, scientist0205@snu.ac.kr
