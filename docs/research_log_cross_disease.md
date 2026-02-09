# scMILD Cross-Disease Analysis Research Log

> **Last Updated:** 2026-02-04
> **Project:** scMILD (Single-cell Multiple Instance Learning for Disease classification)
> **Goal:** HS와 CD의 shared/specific signature 발굴을 위한 cross-disease analysis framework

---

## Table of Contents

1. [연구 목표](#1-연구-목표)
2. [데이터셋 구성](#2-데이터셋-구성)
3. [파이프라인 설계](#3-파이프라인-설계)
4. [실험 결과](#4-실험-결과)
5. [주요 논의사항](#5-주요-논의사항)
6. [실행 명령어 레퍼런스](#6-실행-명령어-레퍼런스)
7. [TODO & Next Steps](#7-todo--next-steps)

---

## 1. 연구 목표

### Primary Goal
- Cross-disease transferability를 통한 **HS(Hidradenitis Suppurativa)와 CD(Crohn's Disease)의 shared/specific signature 발굴**

### Framework Contribution
- 단일 질병 데이터로 학습한 classifier를 다른 질병에 적용하는 분석 프레임워크 제시
- MIL 기반 attention score를 활용한 disease-relevant cell 식별

### Target
- 종합지 (Biology/Discovery 중심: Nature Communications, Cell Reports 등)

---

## 2. 데이터셋 구성

### 2.1 Pretraining Dataset (전체 데이터)

| Dataset | Total | Case | Control | Organ | Note |
|---------|-------|------|---------|-------|------|
| **SCP1884** | **34** | **18** | **16** | Colon | CD main dataset |
| **PCD** (5 studies) | **55** | **49** | **6** | Colon | CD pooled |
| ├─ GSE225199 | 7 | 6 | 1 | Colon | |
| ├─ GSE260842 | 34 | 34 | 0 | Colon | Case only |
| ├─ GSE277387 | 9 | 9 | 0 | Colon | Case only |
| ├─ GSE114374 | 2 | 0 | 2 | Colon | Control only |
| └─ GSE116222 | 3 | 0 | 3 | Colon | Control only |
| **Skin3** (2 studies) | **16** | **5** | **11** | Skin | HS main dataset |
| ├─ GSE175990 | 4 | 3 | 1 | Skin | |
| └─ GSE220116 | 12 | 2 | 10 | Skin | |
| **GSE154775** | **3** | **3** | **0** | Skin | Case only |
| **GSE212721** | **9** | **6** | **3** | Skin | CD45-sorted (immune) |

### 2.2 Classification Dataset (학습용)

| Model | Training Data | Samples | Case/Ctrl |
|-------|--------------|---------|-----------|
| **CD-trained** | SCP1884 | 34 | 18/16 |
| **HS-trained** | Skin3 | 16 | 5/11 |

### 2.3 데이터셋 특성 및 제약

**Cross-dataset transfer 제약사항:**
- Skin3 외 다른 HS 데이터셋: cell type composition이 현저히 다름
- SCP1884 외 다른 CD 데이터셋: 마찬가지로 composition 차이
- GSE212721: CD45-sorted (immune cell enriched) → 전체 tissue와 다름

---

## 3. 파이프라인 설계

### 3.1 전체 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Pretrain (Unsupervised)                                │
│   - Data: ALL datasets (SCP1884 + PCD + Skin3 + GSE154775 + ...) │
│   - Task: VQ-AENB reconstruction                                │
│   - Output: General cell representation encoder                 │
│   - Label 사용: NO                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Classification Training (Supervised)                   │
│   - Data: Single disease dataset only (e.g., SCP1884 for CD)    │
│   - Task: Patient-level MIL classification                      │
│   - Output: Disease-specific classifier                         │
│   - Hyperparameter tuning via Stratified K-Fold CV              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Cross-Disease Evaluation                               │
│   - Data: Different disease dataset (e.g., Skin3 for HS)        │
│   - Task: Zero-shot classification                              │
│   - Analysis: Shared vs specific signature identification       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Information Leakage 검토

**결론: Leakage 아님**

| 단계 | Disease B 데이터 | Disease B Label |
|------|-----------------|-----------------|
| Pretrain | 사용됨 (reconstruction) | **미사용** |
| Classification | 미사용 | 미사용 |
| Cross-disease test | 사용됨 (inference) | 평가용 |

- Pretrain은 unsupervised (label 미사용)
- Classification은 Disease A 데이터만 사용
- Standard transfer learning protocol과 일치 (GPT/BERT 방식)

### 3.3 Conditional Embedding 역할

- `study` 컬럼 기반 embedding → batch effect 흡수
- Disease 정보가 아닌 **기술적 변이**만 모델링
- Cross-disease test 시에도 동일 mapping 사용

---

## 4. 실험 결과

### 4.1 CD-trained Model (SCP1884)

#### Hyperparameter Tuning (5-Fold Stratified CV)

```
Best hyperparameters (by mean AUC):
  - learning_rate: 0.0001
  - encoder_learning_rate: 0.0001
  - epochs: 50
  - disease_ratio_lambda: 0.0
  - CV AUC: 0.9000
```

**Tuning Results:**
| Rank | lr | enc_lr | epochs | ratio_lambda | AUC |
|------|-----|--------|--------|--------------|-----|
| 1 | 0.0001 | 0.0001 | 50 | 0.0 | 0.9000 |
| 2 | 0.001 | 0.0001 | 50 | 0.1 | 0.8333 |
| 3 | 0.0001 | 0.0001 | 50 | 0.1 | 0.7667 |
| 4 | 0.001 | 0.0001 | 50 | 0.0 | 0.6500 |

#### Cross-Disease Evaluation

**Model Path:** `results/scp1884_skfold_512/final_model_20260204_003135/`

| Test Dataset | AUC | Accuracy | F1 | Precision | Recall | Result Path |
|--------------|-----|----------|-----|-----------|--------|-------------|
| **Skin3 (HS)** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `cross_eval_20260204_003957/` |
| **PCD (CD)** | **0.9111** | 0.7778 | 0.8462 | 1.0000 | 0.7333 | `cross_eval_20260204_0759xx/` |
| GSE212721 (HS) | 0.2222 | 0.3333 | 0.0000 | 0.0000 | 0.0000 | `cross_eval_20260204_075919/` |

**Key Finding:** CD-trained model이 HS(Skin3)를 완벽하게 분류 → **Strong shared signature**

---

### 4.2 HS-trained Model (Skin3)

#### Hyperparameter Tuning (5-Fold Stratified CV)

```
Best hyperparameters (by mean AUC):
  - learning_rate: 0.0005
  - encoder_learning_rate: 0.0001
  - epochs: 100
  - disease_ratio_lambda: 0.0
  - CV AUC: 1.0000
```

**Tuning Results (Top 5):**
| Rank | lr | enc_lr | epochs | ratio_lambda | AUC |
|------|-----|--------|--------|--------------|-----|
| 1 | 0.0005 | 0.0001 | 100 | 0.0 | 1.0000 |
| 2 | 0.0001 | 0.001 | 100 | 0.0 | 1.0000 |
| 3 | 0.0001 | 0.0001 | 100 | 0.0 | 0.9500 |
| 4 | 0.0005 | 0.0001 | 10 | 0.0 | 0.9000 |
| 5 | 0.0001 | 0.0001 | 10 | 0.0 | 0.9000 |

#### Cross-Disease Evaluation (Multiple Configs)

**Config 1:** lr=0.0005, enc_lr=0.0001 (Best CV AUC=1.0)
- **Model Path:** `results/skin3_skfold_512/final_model_20260204_064903/`

| Test Dataset | AUC | Accuracy | F1 | Precision | Recall | Result Path |
|--------------|-----|----------|-----|-----------|--------|-------------|
| SCP1884 (CD) | 0.7188 | 0.7353 | 0.6897 | 0.9091 | 0.5556 | `cross_eval_20260204_065107/` |
| PCD (CD) | 0.8111 | 0.9444 | 0.9677 | 0.9375 | 1.0000 | `cross_eval_20260204_075335/` |
| GSE212721 (HS) | 0.0000 | 0.3333 | 0.0000 | 0.0000 | 0.0000 | `cross_eval_20260204_075723/` |

**Config 2:** lr=0.0001, enc_lr=0.001 (Tied Best CV AUC=1.0)
- **Model Path:** `results/skin3_skfold_512/final_model_20260204_065704/`

| Test Dataset | AUC | Accuracy | F1 | Precision | Recall | Result Path |
|--------------|-----|----------|-----|-----------|--------|-------------|
| SCP1884 (CD) | 0.6250 | 0.6765 | 0.6207 | 0.8182 | 0.5000 | `cross_eval_20260204_065852/` |
| PCD (CD) | **1.0000** | **1.0000** | **1.0000** | 1.0000 | 1.0000 | `cross_eval_20260204_0759xx/` |

**Config 3:** lr=0.0001, enc_lr=0.0001 (3rd CV AUC=0.95)
- **Model Path:** `results/skin3_skfold_512/final_model_20260204_065948/`

| Test Dataset | AUC | Accuracy | F1 | Precision | Recall | Result Path |
|--------------|-----|----------|-----|-----------|--------|-------------|
| SCP1884 (CD) | 0.6528 | 0.7353 | 0.6897 | 0.9091 | 0.5556 | `cross_eval_20260204_0659xx/` |
| PCD (CD) | 0.9000 | 0.7500 | 0.8235 | 1.0000 | 0.7000 | `cross_eval_20260204_075157/` |
| GSE212721 (HS) | 0.2222 | 0.3333 | 0.0000 | 0.0000 | 0.0000 | `cross_eval_20260204_075816/` |

---

### 4.3 결과 요약

#### Cross-Disease Transfer Matrix (AUC)

| Train → Test | SCP1884 (CD) | Skin3 (HS) | PCD (CD) | GSE212721 (HS) |
|--------------|--------------|------------|----------|----------------|
| **SCP1884 (CD)** | CV: 0.90 | **1.00** | 0.91 | 0.22 |
| **Skin3 (HS) Config1** | 0.72 | CV: 1.00 | 0.81 | 0.00 |
| **Skin3 (HS) Config2** | 0.63 | CV: 1.00 | **1.00** | - |
| **Skin3 (HS) Config3** | 0.65 | CV: 0.95 | 0.90 | 0.22 |

#### Key Observations

1. **CD → HS transfer가 HS → CD보다 강함**
   - SCP1884-trained → Skin3: AUC 1.00
   - Skin3-trained → SCP1884: AUC 0.65~0.72

2. **GSE212721 (CD45-sorted)은 모든 모델에서 실패**
   - Cell type composition이 근본적으로 다름 (immune cell enriched)
   - 이는 limitation이자 method의 sensitivity 증거

3. **PCD (pooled CD datasets)는 양방향 transfer 성공**
   - CD-trained → PCD: 0.91
   - HS-trained → PCD: 0.81~1.00 (Config2에서 완벽)

4. **HS-trained 모델의 config별 차이가 큼**
   - Cross-disease (SCP1884) 성능: 0.63~0.72
   - PCD 성능: 0.81~1.00
   - CV 성능과 cross-disease 성능이 반드시 일치하지 않음

---

## 5. 주요 논의사항

### 5.1 Information Leakage 우려에 대한 대응

**Methods 섹션에 명시할 내용:**
> "Pretrain 단계에서 cross-disease test set의 gene expression은 사용되었으나,
> disease label은 사용되지 않았다. 이는 standard transfer learning protocol을 따른 것이며,
> classification 단계에서 Disease B 정보는 완전히 배제되었다."

### 5.2 Ablation 실험 필요성

**결론: 우선순위 낮음**

- 연구 목표가 "모델 성능 증명"이 아닌 "생물학적 발견"
- Conditional embedding 구조상 완전한 ablation이 까다로움
- 대신 생물학적 검증에 집중:
  - High-attention cell types 분석
  - Shared signature의 pathway enrichment
  - 기존 문헌과의 일치성

### 5.3 Cross-Disease만으로 충분한가?

**현실적 제약:**
- 같은 질병 다른 데이터셋: cell type composition 차이로 직접 비교 어려움
- Within-disease transfer (Skin3 → 다른 HS) 실험이 까다로움

**전략:**
1. Main story: Cross-disease analysis (Skin3 ↔ SCP1884)
2. Supplementary에서 limitation 명시
3. 가능하다면 특정 cell type만 추출하여 partial validation

### 5.4 종합지 타겟 전략

**강조할 포인트:**
1. Cross-disease transferability 자체가 생물학적 발견
2. CD → HS 완벽 transfer의 의미: 강력한 shared pathology
3. Transfer 방향성의 비대칭성: 생물학적 해석 필요

**필요한 추가 분석:**
- Attention score 기반 cell type 분석
- Shared signature의 pathway/GO enrichment
- Disease-specific signature 식별

---

## 6. 실행 명령어 레퍼런스

### 6.1 SCP1884 (CD) Model Pipeline

```bash
# Hyperparameter Tuning
python scripts/06_tune_hyperparams.py --config config/scp1884_skfold_512.yaml --gpu 0 --verbose
# Output: results/scp1884_skfold_512/tuning_stratified_kfold_20260203_064707/

# Final Model Training
python scripts/03_finalize_model.py \
    --config config/scp1884_skfold_512.yaml \
    --best_params results/scp1884_skfold_512/tuning_stratified_kfold_20260203_064707/best_params.yaml \
    --gpu 0
# Output: results/scp1884_skfold_512/final_model_20260204_003135/

# Cross-Disease Evaluation
python scripts/04_cross_disease_eval.py \
    --model_dir results/scp1884_skfold_512/final_model_20260204_003135/ \
    --test_config config/skin3_skfold_512.yaml \
    --gpu 0

python scripts/04_cross_disease_eval.py \
    --model_dir results/scp1884_skfold_512/final_model_20260204_003135/ \
    --test_config config/pcd.yaml \
    --gpu 0

# Cell Scoring
python scripts/05_cell_scoring.py \
    --model_dir results/scp1884_skfold_512/final_model_20260204_003135 \
    --config config/skin3_skfold_512.yaml \
    --output_dir results/scp1884_skfold_512/cross_eval_skin3_cell_scores \
    --gpu 0
```

### 6.2 Skin3 (HS) Model Pipeline

```bash
# Hyperparameter Tuning
python scripts/06_tune_hyperparams.py --config config/skin3_skfold_512.yaml --gpu 0 --verbose
# Output: results/skin3_skfold_512/tuning_stratified_kfold_20260204_055734/

# Final Model Training (Best Config: lr=0.0005, enc_lr=0.0001, epochs=100)
python scripts/03_finalize_model.py \
    --config config/skin3_skfold_512.yaml \
    --best_params results/skin3_skfold_512/tuning_stratified_kfold_20260204_055734/best_params.yaml \
    --gpu 0
# Output: results/skin3_skfold_512/final_model_20260204_064903/

# Cross-Disease Evaluation
python scripts/04_cross_disease_eval.py \
    --model_dir results/skin3_skfold_512/final_model_20260204_064903/ \
    --test_config config/scp1884_skfold_512.yaml \
    --gpu 0

python scripts/04_cross_disease_eval.py \
    --model_dir results/skin3_skfold_512/final_model_20260204_064903/ \
    --test_config config/pcd.yaml \
    --gpu 0
```

### 6.3 CV Mode Cell Scoring

```bash
# CV 모드 (fold별 test sample scoring)
python scripts/05_cell_scoring.py \
    --cv_dir results/scp1884_skfold_512/cv_stratified_kfold_YYYYMMDD_HHMMSS \
    --config config/scp1884_skfold_512.yaml \
    --output_dir results/scp1884_skfold_512/cell_scores_cv \
    --gpu 0

# Tuning 모드 (top-k config별 scoring)
python scripts/05_cell_scoring.py \
    --tuning_dir results/scp1884_skfold_512/tuning_stratified_kfold_20260203_064707 \
    --config config/scp1884_skfold_512.yaml \
    --output_dir results/scp1884_skfold_512/cell_scores_skfold \
    --gpu 0
```

---

## 7. TODO & Next Steps

### High Priority

- [ ] **Attention score 기반 cell type 분석**
  - High-attention cells의 cell type distribution
  - CD-trained vs HS-trained 모델 비교

- [ ] **Shared signature pathway analysis**
  - High-attention cells에서 enriched pathway
  - GO/KEGG enrichment

- [ ] **Disease-specific signature 식별**
  - CD-trained에서만 high attention인 cells
  - HS-trained에서만 high attention인 cells

### Medium Priority

- [ ] **Transfer 방향성 비대칭 분석**
  - 왜 CD → HS가 HS → CD보다 잘 되는가?
  - Sample size 차이? (34 vs 16)
  - Disease mechanism 차이?

- [ ] **GSE212721 실패 원인 심층 분석**
  - CD45-sorted의 cell type composition
  - 다른 데이터셋과의 비교

### Low Priority

- [ ] Partial validation: 특정 cell type만 추출하여 cross-dataset 비교
- [ ] Study embedding 분석 (t-SNE/UMAP)

---

## Appendix

### A. Model Paths

```
Pretrained Encoder:
  /home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/results/pretrained_skfold_512/vq_aenb_conditional_whole.pth

CD-trained Final Model:
  /home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/results/scp1884_skfold_512/final_model_20260204_003135/

HS-trained Final Models:
  Config 1 (best): /home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/results/skin3_skfold_512/final_model_20260204_064903/
  Config 2: /home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/results/skin3_skfold_512/final_model_20260204_065704/
  Config 3: /home/bmi-user/workspace/data/HSvsCD/scMILDQ_Cond/results/skin3_skfold_512/final_model_20260204_065948/
```

### B. Config Files

```
Main Configs:
  config/scp1884_skfold_512.yaml  # CD classification
  config/skin3_skfold_512.yaml    # HS classification

Test Configs:
  config/pcd.yaml                 # Pooled CD datasets
  config/gse212721.yaml           # CD45-sorted HS
```

### C. Key Hyperparameters

| Parameter | CD Model | HS Model |
|-----------|----------|----------|
| learning_rate | 0.0001 | 0.0005 |
| encoder_learning_rate | 0.0001 | 0.0001 |
| epochs | 50 | 100 |
| disease_ratio_lambda | 0.0 | 0.0 |
| CV AUC | 0.90 | 1.00 |
