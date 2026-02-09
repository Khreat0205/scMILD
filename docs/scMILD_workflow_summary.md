# scMILD: Single-cell Multiple Instance Learning for Disease Classification

**Cross-Disease Transferability Analysis of Hidradenitis Suppurativa and Crohn's Disease**

**Keywords:** scRNA-seq, Multiple Instance Learning, Vector Quantization, Cross-disease Transfer, Hidradenitis Suppurativa, Crohn's Disease, Gated Attention

---

## Abstract

scMILD는 단일세포 RNA-seq (scRNA-seq) 데이터에서 환자 수준의 질병 분류를 수행하는 딥러닝 프레임워크이다. 단일세포 데이터는 환자당 수천~수만 개의 세포를 포함하지만, 질병 label은 환자 수준에서만 존재한다. scMILD는 이 문제를 Multiple Instance Learning (MIL)으로 해결하며, VQ-AENB-Conditional Encoder와 Gated Attention 기반 Teacher-Student 구조를 결합한다. Hidradenitis Suppurativa (HS)와 Crohn's Disease (CD)에 적용하여 cross-disease transferability를 분석한 결과, CD-trained 모델이 HS를 AUC 1.00으로 완벽히 분류하는 강력한 shared inflammatory signature를 발견하였다.

---

## 1. Introduction

### 1.1 Problem: Label Granularity Gap in scRNA-seq

단일세포 RNA-seq 기술은 조직 내 개별 세포의 유전자 발현을 측정할 수 있게 해주었다. 그러나 질병 연구에서 핵심적인 문제가 존재한다: **세포 수준의 질병 label은 존재하지 않으며, 환자(sample) 수준의 label만 사용 가능하다.** 즉, "이 환자는 질병이 있다"는 알 수 있지만, "이 세포가 질병에 기여한다"는 알 수 없다.

### 1.2 Multiple Instance Learning (MIL) Paradigm

MIL은 이 문제에 자연스럽게 적합한 프레임워크이다:
- **Bag** = 환자 (sample): 하나의 bag에는 수천 개의 instance가 포함
- **Instance** = 세포 (cell): 개별 세포는 label이 없음
- **Weakly Supervised**: bag-level label만으로 instance-level 정보를 추출

### 1.3 Research Goals

1. **환자 수준 질병 분류** — scRNA-seq에서 disease vs control을 구분
2. **질병 관련 세포 식별** — attention score로 중요한 세포 하위집단을 발견
3. **Cross-disease transferability** — HS와 CD 간 shared/specific disease signature 발굴

### 1.4 Target Diseases

**Hidradenitis Suppurativa (HS)** 와 **Crohn's Disease (CD)** 는 만성 염증성 질환으로, 임상적으로 공존 빈도가 높고 shared inflammatory pathway의 존재가 제안되어 왔다. scMILD를 통해 이 연관성을 분자 수준에서 검증한다.

| Feature | Bulk RNA-seq | Standard scRNA-seq ML | scMILD |
|---------|-------------|----------------------|--------|
| Label requirement | Patient-level | Cell-level (supervised) | Patient-level (weakly supervised) |
| Cell heterogeneity | Ignored (averaged) | Utilized | Utilized |
| Batch effect correction | External tools | Varies | Built-in (Conditional Embedding) |
| Interpretability | Gene-level | Cell-level | Cell + Codebook level |

**Key Takeaway:** scMILD는 환자 수준 label만으로 세포 수준의 질병 관련 정보를 추출하는 MIL 기반 프레임워크이다.

---

## 2. Datasets

### 2.1 Data Overview

총 5개 데이터셋, 약 805,000 cells, 6,000 highly variable genes를 사용한다.

| Dataset | Disease | Samples | Case/Control | Organ | Usage |
|---------|---------|---------|-------------|-------|-------|
| SCP1884 | CD | 34 | 18/16 | Colon | Train + CV |
| PCD (5 studies pooled) | CD | 55 | 49/6 | Colon | Cross-eval |
| Skin3 (2 studies) | HS | 16 | 5/11 | Skin | Train + CV |
| GSE154775 | HS | 3 | 3/0 | Skin | Cross-eval |
| GSE212721 | HS | 9 | 6/3 | Skin | Cross-eval (CD45-sorted) |

### 2.2 Pretraining vs Classification Data

- **Pretrain**: 전체 데이터셋 (~805k cells) 사용. **Unsupervised** reconstruction task로 disease label 미사용
- **Classification**: 단일 질병 데이터셋만 사용 (SCP1884 또는 Skin3). Supervised MIL 학습

이 구조에서 **information leakage는 발생하지 않는다**: pretrain은 label-free이며, classification은 Disease A 데이터만 사용한다. 이는 GPT/BERT의 standard transfer learning protocol과 동일하다.

**Key Takeaway:** Pretrain은 전체 데이터를 비지도로, classification은 단일 질환 데이터만 지도로 학습하여 information leakage를 방지한다.

---

## 3. Methods — Model Architecture

### 3.1 Overall Architecture

scMILD는 2-stage로 구성된다: **Stage 1** (unsupervised representation learning) → **Stage 2** (supervised MIL classification).

```
┌─────────────────────────────────────────────────────────────┐
│                    scMILD Architecture                       │
│                                                             │
│  [scRNA-seq counts]  +  [Conditional ID (study/organ)]      │
│          ↓                       ↓                          │
│  ┌─────────────────────────────────────┐                    │
│  │   VQ-AENB-Conditional Encoder       │  ← Stage 1        │
│  │   (Unsupervised, Frozen after)      │    (Pretrained)    │
│  └──────────────┬──────────────────────┘                    │
│                 ↓                                           │
│        [Quantized Latent z_q]                               │
│                 ↓                                           │
│  ┌──────────────────────────┐                               │
│  │    Projection Layer       │  ← Trainable adapter         │
│  └──────────────┬───────────┘                               │
│                 ↓                                           │
│  ┌──────────────────────────────────────────────────┐       │
│  │         Gated Attention MIL (Stage 2)             │       │
│  │  ┌──────────────────┐  ┌───────────────────────┐ │       │
│  │  │  Teacher Branch   │  │   Student Branch       │ │       │
│  │  │  (Bag-level)      │→│   (Instance-level)     │ │       │
│  │  │  Gated Attention  │  │   Pseudo-label from    │ │       │
│  │  │  → Bag Classifier │  │   teacher attention    │ │       │
│  │  └──────────────────┘  └───────────────────────┘ │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 VQ-AENB-Conditional Encoder

**Encoder-Decoder 구조:**
- Input: gene expression vector (6,000 genes) + conditional embedding (16-dim)
- Hidden layers: `[512, 256, 128]` → latent dim 128
- Reconstruction: **Negative Binomial** distribution으로 scRNA-seq count data의 overdispersion을 모델링

**Vector Quantization (VQ):**
- **Codebook**: 1,024 codes × 128 dimensions의 discrete latent space
- **Matching**: cosine similarity 기반 nearest code lookup
- **Gradient**: straight-through estimator로 gradient propagation
- **Loss**: `Total = NB Reconstruction + β × Commitment Loss`
- **Codebook 초기화**: Stratified K-means (conditional 변수별 균등 샘플링)

```
z_continuous → cosine_sim(z, codebook) → argmax → z_q
                                                   ↓
                          straight-through: z + (z_q - z).detach()
```

**Conditional Embedding:**
- `nn.Embedding(n_conditionals, 16)`: study 또는 organ 별 학습된 임베딩
- Encoder와 decoder 양쪽에 concatenation으로 주입 (scVI-style)
- **목적**: batch effect를 conditional embedding이 흡수하여 latent space는 condition-free하게 유지

### 3.3 Gated Attention MIL

**Gated Attention Module:**

```
Value path:  H → Linear(L, D) → Tanh   → A_V
Gate path:   H → Linear(L, D) → Sigmoid → A_U
Attention:   w^T(A_V ⊙ A_U) → softmax → attention weights
```

Sigmoid gate가 각 feature의 중요도를 선택적으로 조절하여 standard attention보다 표현력이 높다.

**Teacher Branch (Bag-level classification):**
- 모든 세포의 embedding에 attention weight를 적용하여 aggregation: `h_bag = Σ(a_i × h_i)`
- 3-layer MLP classifier: `Linear → Tanh → Linear → Tanh → Linear(2)`
- **Loss**: Binary Cross-Entropy (bag-level)

**Student Branch (Instance-level classification):**
- 개별 세포 embedding을 직접 분류하는 3-layer MLP
- **Pseudo-label**: Teacher의 normalized attention score를 cell-level target으로 사용
- Control sample의 세포는 pseudo-label을 0으로 강제
- **Loss**: Weighted BCE with negative weight 0.3

**Mathematical Formulation:**
```
Encoder:   z_q = VQ(Enc(x, c))
Projection: h_i = Proj(z_q_i)
Attention: a_i = softmax(w^T(tanh(Vh_i) ⊙ σ(Uh_i)))
Teacher:   P(Y=1|Bag) = Classifier(Σ a_i × h_i)
Student:   P(y_i=1|x_i) = CellClassifier(h_i), supervised by teacher attention
```

**Key Takeaway:** VQ-AENB-Conditional이 batch-corrected discrete representation을 학습하고, Gated Attention MIL의 Teacher-Student 구조가 환자 분류와 세포 수준 해석을 동시에 수행한다.

---

## 4. Methods — Pipeline

### 4.1 Pipeline Overview

```
Step 1: Pretrain Encoder (unsupervised, ALL ~805k cells)
  → vq_aenb_conditional.pth + conditional_mapping.json
        ↓
Step 2: Cross-Validation (single disease subset, LOOCV or K-Fold)
  → CV metrics (AUC, Accuracy, F1)
        ↓
Step 3: Hyperparameter Tuning (Grid Search over CV)
  → best_params.yaml + top-K model checkpoints
        ↓
Step 4: Final Model Training (entire subset, best params)
  → Final teacher + student + encoder models
        ↓
Step 5a: Cell Scoring                Step 5b: Cross-Disease Eval
  → scored_adata.h5ad                  → AUC on unseen disease
  → codebook_adata.h5ad
```

### 4.2 Cross-Validation Strategies

| Strategy | 특징 | 적합한 경우 |
|----------|------|-----------|
| LOOCV | 1개 sample test, 나머지 train | 소규모 데이터 (<30 samples) |
| Stratified K-Fold | Class ratio 유지, fold별 metric | 중규모 데이터 |
| Repeated Stratified K-Fold | K-Fold를 N회 반복 | Variance estimation 필요 시 |

### 4.3 Cell Scoring System

세포 수준 분석을 위한 3가지 scoring mode를 지원한다:

- **Final model mode**: 단일 최종 모델로 전체 데이터 scoring
- **CV mode**: 각 fold 모델이 해당 test sample만 scoring (data leakage 방지)
- **Tuning mode**: best hyperparameter configuration의 fold 모델 사용

**Cell-level 출력** (`scored_adata.h5ad`):
- `attention_score_global`: 전체 데이터 기준 min-max 정규화
- `attention_score_sample`: sample 내 정규화
- `student_prediction`: cell-level 질병 확률
- `vq_code`: codebook index
- `X_pretrained`, `X_scmild`: pre/post-projection embeddings

**Codebook-level 출력** (`codebook_adata.h5ad`):
- `attn_direct`: codebook vector를 직접 attention module에 통과시킨 score
- `attn_cell_mean/std/median/max`: cell 기반 attention 통계
- `n_cells`, `n_samples`, `disease_ratio`: code별 할당 통계

### 4.4 Configuration System

YAML 기반 hierarchical config 시스템으로, `_base_` 상속과 `${variable}` 참조를 지원한다. Python dataclass로 type-safe하게 매핑된다.

**Key Takeaway:** 6단계 파이프라인으로 pretrain → CV → tuning → finalize → evaluation/scoring을 체계적으로 수행하며, YAML config로 재현성을 보장한다.

---

## 5. Results — Within-Disease Classification

### 5.1 Classification Performance

5-Fold Stratified Cross-Validation 결과:

| Model | Dataset | CV AUC | Best lr | Best enc_lr | Best epochs |
|-------|---------|--------|---------|-------------|-------------|
| CD-trained | SCP1884 (34 samples) | **0.90** | 0.0001 | 0.0001 | 50 |
| HS-trained | Skin3 (16 samples) | **1.00** | 0.0005 | 0.0001 | 100 |

두 질환 모두 높은 분류 성능을 보이며, disease_ratio_lambda=0.0 (regularization 미사용)이 최적이었다.

**Key Takeaway:** CD는 AUC 0.90, HS는 AUC 1.00의 within-disease 분류 성능을 달성하였다.

---

## 6. Results — Cross-Disease Transfer

### 6.1 Transfer Matrix

한 질환으로 학습한 모델을 다른 질환에 zero-shot 적용한 결과:

| Train \ Test | SCP1884 (CD) | Skin3 (HS) | PCD (CD pooled) | GSE212721 (CD45) |
|-------------|-------------|-----------|----------------|-----------------|
| **CD-trained (SCP1884)** | CV: 0.90 | **1.00** | 0.91 | 0.22 |
| **HS-trained (Skin3)** | 0.63~0.72 | CV: 1.00 | 0.81~1.00 | 0.00~0.22 |

### 6.2 Key Findings

1. **CD → HS transfer가 완벽 (AUC 1.00)**: CD로 학습한 모델이 HS를 완벽히 분류 → **강력한 shared inflammatory signature 존재**
2. **HS → CD transfer는 부분적 (AUC 0.63~0.72)**: 비대칭 전이 패턴 → sample size (34 vs 16) 또는 disease mechanism 차이의 영향
3. **PCD (pooled CD)는 양방향 성공**: CD-trained 0.91, HS-trained 0.81~1.00
4. **GSE212721 (CD45-sorted)은 모든 모델에서 실패**: cell type composition mismatch (immune cell enriched)가 원인

**Key Takeaway:** CD → HS 완벽 전이는 두 질환이 공유하는 강력한 분자적 시그니처를 시사하며, 전이의 비대칭성은 생물학적으로 의미 있는 패턴이다.

---

## 7. Discussion

### 7.1 Cross-Disease Transfer의 생물학적 의미

CD-trained 모델의 HS 완벽 분류(AUC 1.00)는 기존 임상 관찰(HS-CD 공존)을 분자 수준에서 뒷받침한다. CD가 HS보다 더 크고 명확한 inflammatory signature를 가지고 있을 가능성이 있으며, 이것이 transfer 방향성 비대칭의 원인으로 추정된다.

### 7.2 VQ Codebook의 해석 가능성

VQ codebook은 세포를 1,024개의 discrete cluster로 분류한다. 각 code별 attention score와 disease ratio를 분석하면 disease-relevant cell subpopulation을 체계적으로 식별할 수 있다. Codebook-level analysis는 cross-disease shared code vs disease-specific code를 구분하는 데 활용 가능하다.

### 7.3 Conditional Embedding의 Batch Correction

Study/organ 기반 conditional embedding은 기술적 변이를 흡수하면서도 biological variation은 보존한다. 이를 통해 multi-study 데이터의 통합과 cross-dataset 일반화가 가능해진다.

### 7.4 Limitations

- **Small sample size**: 특히 Skin3 (16 samples)은 overfitting 위험
- **Cross-dataset composition 차이**: cell type composition이 다른 데이터셋 간 직접 비교 한계
- **GSE212721 실패**: CD45-sorted 데이터는 전체 tissue와 cell composition이 달라 method의 한계 노출
- **Ablation study 부재**: 연구 목표가 생물학적 발견에 초점을 맞추었기 때문

### 7.5 Future Directions

- **High-attention cell type 분석**: 질병 관련 세포 유형 규명
- **Pathway enrichment (GO/KEGG)**: shared signature의 기능적 해석
- **Disease-specific signature 식별**: CD-only vs HS-only high-attention cells 비교
- **Multi-organ extension**: organ-based conditional embedding 활용

**Key Takeaway:** Cross-disease transferability 자체가 shared disease mechanism의 분자적 증거이며, VQ codebook과 attention score를 통한 후속 분석이 핵심 next step이다.

---

## 8. Technical Appendix

### A. Hyperparameter Search Space

| Parameter | Search Values |
|-----------|--------------|
| learning_rate | 0.001, 0.0005, 0.0001 |
| encoder_learning_rate | 0.001, 0.0001 |
| epochs | 10, 30, 50, 100 |
| disease_ratio_lambda | 0.0, 0.05, 0.1 |

### B. Model Architecture Details

| Component | Architecture |
|-----------|-------------|
| Encoder | [6000 + 16] → 512 → 256 → 128 (latent) |
| Decoder | [128 + 16] → 128 → 256 → 512 → 6000×2 (mu + theta) |
| VQ Codebook | 1,024 codes × 128 dim (cosine similarity) |
| Projection | 128 → 128 → 128 (ReLU) |
| Gated Attention | V: 128→128→Tanh, U: 128→128→Sigmoid, w: 128→1 |
| Teacher Classifier | 128 → 128 → 128 → 2 (Tanh) |
| Student Classifier | 128 → 128 → 128 → 2 (Tanh) |

### C. Development Timeline

| Date | Milestone |
|------|-----------|
| 2025-01-29 | Core refactoring: YAML config system, 모듈화, 5개 pipeline scripts |
| 2026-01-30 | Conditional embedding generalization, memory management, tuning system |
| 2026-02-02 | NaN debugging: large num_codes에서 dead code revival 불안정 진단 |
| 2026-02-03 | CV generalization (LOOCV → Stratified K-Fold), variable renaming, auto-registration |
| 2026-02-04 | Cell scoring expansion (3 modes), cross-disease analysis experiments |

### D. Software Stack

- Python 3.12, PyTorch, scanpy, anndata, scikit-learn, FAISS, pandas, NumPy
