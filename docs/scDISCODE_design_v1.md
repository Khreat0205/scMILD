# scDISCODE: Disease-associated Code Discovery via VQ Representation Learning
> 설계 문서 v1.1 — 2026-04-02
> 시나리오 2: Organ별 독립 codebook + cross-organ transfer evaluation

---

## 0. 한 줄 요약

**Organ별로 독립적으로 disease-aware VQ codebook을 end-to-end로 학습하고, 한쪽 organ의 codebook을 다른 organ에 적용하여 cross-organ transferability를 평가하며, shared/specific disease gene program을 체계적으로 발굴한다.**

---

## 1. 동기 및 배경

### 1.1 기존 방법의 한계

| 방법 | 한계 |
|------|------|
| scMILD (MIL) | MIL attention이 organ-specific code에 과집중 → cross-organ transfer 실패 (0.64~0.76) |
| MMIL | EM 기반 cell-level label refinement, cross-organ 미다룸 |
| HiDDEN | Continuous space에서 cell-level perturbation detection, cross-organ 미다룸 |
| Code-level Transformer | Within CV underfitting, cross 0.6~0.7 |
| Disease ratio zero-shot | Pelka cross-organ 0.8+ — **학습 없이 가장 높은 성능** |

### 1.2 핵심 관찰

> 학습을 시킬수록 오히려 zero-shot보다 나빠진다 → 학습 과정에서 organ-specific noise를 fitting

### 1.3 scDISCODE의 방향 전환

- MIL을 버리고, **VQ codebook 자체를 disease-aware하게 학습**
- Organ별 독립 학습 → **cross-organ eval이 자연스럽게 정의**
- Code = gene program의 단위 → **추가 분석 없이 disease program 직접 해석**

---

## 2. 전체 파이프라인 개요

```
╔══════════════════════════════════════════════════════════════════╗
║                      scDISCODE Pipeline                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  [Phase 1] Organ-specific Disease-Aware Codebook 학습 (End-to-End)║
║                                                                  ║
║    Organ A cells ──→ VQ-AENB + Disease Aux Loss (동시 학습)       ║
║                      L = L_recon + λ_vq·L_vq + λ_dis·L_disease  ║
║                      Disease ratio: EMA로 매 step 업데이트        ║
║                      Pseudo-label: 매 N epoch 정제 (EM 내장)      ║
║                      ──→ 수렴 시 Codebook_A + disease score 확정  ║
║                                                                  ║
║    Organ B cells ──→ (동일 과정) ──→ Codebook_B + disease score   ║
║                                                                  ║
║  [Phase 2] Cross-Organ Transfer Evaluation                       ║
║                                                                  ║
║    Organ B cells ──→ Codebook_A에 quantize                       ║
║                  ──→ A의 disease score로 sample 분류              ║
║                  ──→ Cross-organ AUC                              ║
║                  (역방향도 동일)                                    ║
║                                                                  ║
║  [Phase 3] Shared / Specific Signature 발굴                       ║
║                                                                  ║
║    Codebook_A disease codes의 top genes                           ║
║         ∩                                                        ║
║    Codebook_B disease codes의 top genes                           ║
║         = Shared disease signature                                ║
║         차집합 = Organ-specific signature                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 3. Phase 1: Disease-Aware Codebook 학습 (End-to-End)

### 3.1 아키텍처

```
[scRNA-seq x_i] + [batch_id]
        ↓
    Encoder
        ↓
      z_i (continuous latent)
        ↓
    VQ (hard quantization)
        ↓
     z_q_i ──────────────────→ Decoder(z_q_i + batch_id) → x̂_i
        ↓                              ↓
   code index k_i                   L_recon
        ↓
   Disease Aux Head(codebook[k_i])
        ↓
   disease_pred_k
        ↓
   L_disease (vs EMA disease ratio target)
```

**Loss function (매 step)**:
```
L_total = L_recon + λ_vq * L_vq + λ_disease * L_disease
```

모든 것이 **하나의 학습 loop에서 동시에** 최적화됨.

### 3.2 Disease Auxiliary Loss 상세

#### Core Mechanism: EMA Disease Ratio + Aux Head

```python
class DiseaseRatioEMA:
    """학습 중 code별 disease ratio를 EMA로 추적"""
    def __init__(self, n_codes, decay=0.99):
        self.case_count = torch.zeros(n_codes)   # running case count
        self.total_count = torch.zeros(n_codes)  # running total count
        self.decay = decay

    def update(self, code_indices, sample_labels):
        """매 batch마다 호출"""
        for k in range(self.n_codes):
            mask_k = (code_indices == k)
            n_k = mask_k.sum()
            if n_k == 0:
                continue
            n_case_k = sample_labels[mask_k].sum()

            self.total_count[k] = self.decay * self.total_count[k] + (1 - self.decay) * n_k
            self.case_count[k] = self.decay * self.case_count[k] + (1 - self.decay) * n_case_k

    @property
    def disease_ratio(self):
        """Bayesian smoothed ratio"""
        alpha, beta = 1.0, 1.0
        return (self.case_count + alpha) / (self.total_count + alpha + beta)


class DiseaseAuxHead(nn.Module):
    """Codebook vector → disease score 예측"""
    def __init__(self, latent_dim, hidden_dim=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, codebook_vectors):
        return self.head(codebook_vectors).squeeze(-1)
```

#### Training Step (하나의 loop)

```python
def training_step(batch):
    x, batch_ids, sample_labels = batch

    # 1. Forward: Encoder → VQ → Decoder
    z = encoder(x, batch_ids)
    z_q, code_indices, vq_loss = quantizer(z)
    x_hat = decoder(z_q, batch_ids)
    recon_loss = nb_loss(x_hat, x)

    # 2. Disease ratio EMA 업데이트
    disease_ratio_ema.update(code_indices, sample_labels)

    # 3. Disease aux loss
    #    codebook vector에서 disease score 예측 → EMA ratio와 비교
    active_codes = code_indices.unique()
    codebook_vecs = quantizer.codebook[active_codes]
    disease_pred = disease_aux_head(codebook_vecs)
    disease_target = disease_ratio_ema.disease_ratio[active_codes].detach()
    disease_loss = F.mse_loss(disease_pred, disease_target)

    # 4. Total loss
    loss = recon_loss + λ_vq * vq_loss + λ_disease * disease_loss

    return loss
```

### 3.3 Iterative Pseudo-label Refinement (학습 중 내장)

Codebook이 고정될 때까지 기다리지 않고, **학습 중에 pseudo-label을 점진적으로 정제**:

```python
def refine_pseudo_labels(epoch):
    """매 N epoch마다 pseudo-label 업데이트 (학습 loop 내에서)"""
    if epoch % refine_every != 0:
        return

    # Disease aux head의 현재 prediction으로 pseudo-label 업데이트
    with torch.no_grad():
        for code_k in range(n_codes):
            pred_k = disease_aux_head(quantizer.codebook[code_k])

            # Case sample 세포: prediction으로 업데이트
            # Control sample 세포: 0 유지 (anchor)
            cells_in_k = get_cells_in_code(code_k)
            for cell_i in cells_in_k:
                if sample_label[cell_i] == 1:  # case
                    pseudo_label[cell_i] = pred_k.item()
                else:  # control — anchor
                    pseudo_label[cell_i] = 0.0

    # 업데이트된 pseudo-label로 disease ratio 재계산
    disease_ratio_ema.reset_from_pseudo_labels(pseudo_label)
```

**학습 흐름 요약**:
```
Epoch 1~N:
  매 batch: L_recon + L_vq + L_disease (EMA ratio 기반)
  매 N epoch: pseudo-label 정제 → EMA ratio 업데이트
  → Codebook, Encoder, Disease Head 모두 동시 수렴
```

### 3.4 Conditional Embedding: Batch 보정

시나리오 2에서는 organ별 독립 학습이므로, conditional embedding은 **organ이 아닌 study/batch**에만 사용:

```python
# Organ A 데이터 내에서 batch effect가 있을 경우
x + study_id → Encoder → z → VQ → z_q → Decoder(z_q + study_id) → x̂
```

단일 study라면 conditional embedding 불필요.

### 3.5 Config

```yaml
scdiscode:
  # VQ-AENB 기본 설정 (기존과 동일)
  encoder:
    latent_dim: 128
    num_codes: 512
    hidden_layers: [512, 256, 128]
    conditional_emb_dim: 16

  # Disease Aux (새로 추가)
  disease_aux:
    enabled: true
    loss_weight: 0.1              # λ_disease — 핵심 hyperparameter
    head_hidden_dim: 64
    head_n_layers: 1
    ratio_ema_decay: 0.99
    ratio_smoothing_alpha: 1.0    # Bayesian prior
    ratio_smoothing_beta: 1.0

  # Iterative Refinement (학습 중 내장)
  refinement:
    enabled: true
    refine_every: 5               # 매 5 epoch마다 pseudo-label 정제
    control_anchor: true          # control sample 세포 = 0 고정

  # Training
  training:
    batch_size: 256
    learning_rate: 0.001
    epochs: 100
    patience: 10
```

---

## 4. Phase 2: Cross-Organ Transfer Evaluation

### 4.1 평가 절차

```python
# === A → B 방향 ===

# 1. Organ B 세포를 Codebook_A에 quantize
for cell_b in organ_B_cells:
    z_b = Encoder_A(cell_b)             # Encoder_A로 인코딩
    k_b = argmin_k ||z_b - codebook_A[k]||  # nearest code

# 2. A에서 학습한 disease score로 sample-level 분류
for sample_b in organ_B_samples:
    codes_in_sample = [k_b for cell_b in sample_b]
    code_histogram = Counter(codes_in_sample)

    # Zero-shot: disease score weighted sum
    sample_score = sum(disease_score_A[k] * count_k / n_total
                       for k, count_k in code_histogram.items())

# 3. AUC 계산
auc_A_to_B = roc_auc_score(organ_B_labels, organ_B_scores)
```

### 4.2 Quantization Error 모니터링

```python
# Organ B → Codebook_A 의 quantization error
qe_cross = mean(||z_b - codebook_A[k_b]||^2)  # cross-organ
qe_within = mean(||z_a - codebook_A[k_a]||^2)  # within (baseline)

# qe_cross >> qe_within → codebook이 다른 organ을 잘 표현 못함
# qe_cross ≈ qe_within → codebook이 cross-organ에서도 유효

# Code별 QE → organ-specific code 식별에 활용 가능
qe_per_code[k] = mean(||z_b - codebook_A[k]||^2 for z_b assigned to k)
```

### 4.3 Encoder 문제: Organ A Encoder로 Organ B 인코딩

**이슈**: Encoder_A는 Organ A 데이터로만 학습됨.

| 방안 | 설명 | 장단점 |
|------|------|--------|
| **A. Encoder_A 그대로 사용** | Organ B → Encoder_A → VQ_A | 가장 엄밀한 transfer test. Encoding quality 저하 가능 |
| **B. Codebook만 공유** | Organ B용 별도 encoder 학습 (codebook_A 고정) | Encoder는 organ-specific, codebook만 transfer |
| **C. 공통 HVG** | 양쪽 organ의 공통 HVG로 입력 통일 | 실용적, A와 결합하면 가장 fair |

**권장: A + C** — 공통 HVG 사용 + Encoder_A 그대로 적용

### 4.4 평가 지표 체계

| 지표 | 설명 | 수준 |
|------|------|------|
| **Cross-organ AUC** | A의 disease score로 B 분류 | Sample-level |
| **Within-organ AUC** | A의 disease score로 A 분류 (LOOCV) | Sample-level (sanity check) |
| **QE ratio** | qe_cross / qe_within | Codebook-level |
| **Code utilization overlap** | A와 B에서 공통 사용 code 비율 | Codebook-level |
| **Disease score correlation** | Codebook_A vs Codebook_B gene program 유사도 | Interpretation |

---

## 5. Phase 3: Shared / Specific Signature 발굴

### 5.1 Code-level Disease Classification

```python
# Codebook_A에서
disease_codes_A = {k for k if disease_score_A[k] > threshold_high}
agnostic_codes_A = {k for k if disease_score_A[k] < threshold_low}
# Codebook_B에서도 동일
```

### 5.2 Shared vs Specific: Gene Program Overlap

두 codebook의 code는 직접 대응이 안 되므로(독립 학습), **gene program 수준에서 비교**:

```python
# 각 disease code의 top gene program 추출
for code_k in disease_codes_A:
    cells_in_k = cells assigned to code k
    top_genes_k = differential_expression(cells_in_k vs rest)
    gene_program_A[k] = set(top_genes_k[:50])

# Gene program overlap (Jaccard)
for k in disease_codes_A:
    for j in disease_codes_B:
        jaccard = |gene_program_A[k] ∩ gene_program_B[j]| / |gene_program_A[k] ∪ gene_program_B[j]|

# Shared: high overlap pairs → shared genes = intersection
# Specific: no counterpart → organ-specific genes
```

### 5.3 정량 검증: Shared/Specific 분리의 품질

```python
# Shared code만으로 cross-organ 분류 → 높아야 함
auc_shared_cross = classify_with_codes(shared_codes, organ_B)

# Specific code만으로 within-organ 분류 → 높아야 함
auc_specific_within = classify_with_codes(specific_A_codes, organ_A)

# Specific code로 cross-organ 분류 → 낮아야 함 (organ-specific이니까)
auc_specific_cross = classify_with_codes(specific_A_codes, organ_B)
```

### 5.4 추가 해석

| 분석 | 방법 | 목적 |
|------|------|------|
| Pathway enrichment | Shared genes → GO/KEGG | 생물학적 의미 |
| Cell type composition | Code별 cell type 비율 | Code가 어떤 세포에서 유래 |
| TCGA bulk validation | Shared signature → bulk 생존 분석 | Clinical relevance |

---

## 6. 구현 계획

### 6.1 기존 scMILD 코드 재활용

| 모듈 | 재활용 | 수정 |
|------|--------|------|
| `models/autoencoder.py` (VQ-AENB) | ✅ 그대로 | Disease aux head 추가 |
| `models/quantizer.py` | ✅ 그대로 | — |
| `training/trainer_ae.py` | ✅ 기반 | Disease loss + EMA + refinement 통합 |
| `training/disease_ratio.py` | ✅ 재활용 | EMA 방식으로 전환 |
| `data/preprocessing.py` | ✅ 그대로 | Organ별 분리 로직 추가 |
| MIL 관련 (attention, branches) | ❌ 불필요 | — |

### 6.2 새로 구현

| 모듈 | 설명 | 난이도 |
|------|------|--------|
| `DiseaseAuxHead` | Codebook → disease score (MLP) | 낮음 |
| `DiseaseRatioEMA` | Running disease ratio (EMA) | 낮음 |
| `training_step` 수정 | L_recon + L_vq + L_disease 동시 학습 | 낮음 |
| Pseudo-label refinement logic | 매 N epoch 정제 (training loop 내) | 낮음 |
| `CrossOrganEvaluator` | Codebook_A → Organ B quantize → AUC | 중간 |
| `SignatureDiscovery` | Gene program overlap 비교 | 중간 |

### 6.3 실험 우선순위

| 순서 | 실험 | 목적 | 데이터 |
|------|------|------|--------|
| 1 | Organ별 독립 pretrain + zero-shot baseline 재현 | 기준선 | Pelka |
| 2 | Disease aux loss end-to-end 학습 | 핵심 실험 | Pelka |
| 3 | Pseudo-label refinement 효과 | 정제 효과 검증 | Pelka |
| 4 | Shared/Specific gene program 분석 | 해석 | Pelka |
| 5 | Yoshida 적용 | 실패 케이스 개선 | Yoshida |
| 6 | 추가 데이터셋 | Generalizability | PBMCpedia / Pan-GI |

### 6.4 Baseline Comparison

| 방법 | 비교 포인트 |
|------|-------------|
| Disease ratio zero-shot | scDISCODE가 0.8+를 이기는가? |
| scMILD (OPL) | MIL 대비 개선? |
| MMIL | 같은 EM인데 VQ 유무 차이 |
| HiDDEN | Continuous vs discrete |
| Standard DEG | 기존 분석 대비 추가 발견? |

---

## 7. 논문 기대 Contribution

### Method
1. **VQ discrete bottleneck + disease-aware end-to-end learning** — pretrain에서 disease signal을 code-level로 직접 학습하는 최초의 프레임워크
2. **In-training pseudo-label refinement on discrete codes** — MMIL EM을 VQ codebook 위에서 end-to-end로 수행

### Analysis Framework
3. **Cross-organ transferability test** — organ별 독립 학습 → 다른 organ 적용 → 평가. 이 세팅 자체가 새로움
4. **Shared/Specific signature의 체계적 분해** — codebook gene program overlap 기반

### Biological
5. 데이터셋별 shared/specific disease gene program 발굴

---

## 8. 리스크 및 대응

| 리스크 | 가능성 | 대응 |
|--------|--------|------|
| Disease aux loss가 within은 올리지만 cross를 망침 | 중간 | λ_disease 작게 시작, cross AUC 모니터링 |
| Organ별 독립 codebook의 gene space 차이 | 낮음 | 공통 HVG 고정 |
| Pelka에서만 작동, Yoshida 실패 | 높음 | 실패 분석 자체가 contribution |
| Zero-shot 0.8을 못 이김 | 중간 | 해석 프레임워크 contribution으로 전환 |
| "VQ + ratio인데?" | 높음 | End-to-end disease-aware learning + cross-organ eval framework novelty |

---

## 9. 타임라인 (안)

| 주차 | 할 일 |
|------|-------|
| W1 | Organ별 독립 pretrain + baseline zero-shot 재현 |
| W2 | DiseaseAuxHead + EMA 구현, end-to-end 학습 통합 |
| W3 | Pseudo-label refinement + Pelka 실험 |
| W4 | CrossOrganEvaluator + shared/specific 분석 |
| W5 | Yoshida + 비교 실험 (MMIL, HiDDEN) |
| W6 | 결과 정리 + 논문 초고 |
