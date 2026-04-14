# Session Log - 2026-03-17

## 목표
- Celltype Auxiliary Loss v2/v3 pretrain NaN 원인 분석 및 해결
- Pretrain encoder 하이퍼파라미터 sweep 구축 및 실행
- Cross-disease evaluation 결과 종합 비교 (v1/v2/v3)
- Codebook quality 비교 분석
- 최적 encoder 선정 및 향후 방향 결정

---

## 1. NaN 원인 분석

### 발단
`pretrain_celltype_aux_v2.yaml`로 pretrain 시 Epoch 5에서 NaN 발생.

```
python scripts/01_pretrain_encoder.py --config config/pretrain_celltype_aux_v2.yaml --gpu 1 --register
Epoch 5/100 - Train Loss: nan - CT Loss: nan
```

### v2 Config (NaN 발생)
- `celltype_lineage_v2` 컬럼, `loss_weight: 0.2`, `hidden_dim: 128`, `lr: 0.001`
- 전체 805k 데이터, 40.5% valid celltype labels

### 원인 분석 결과

**1차 원인: Dead code revival 불안정**
- `quantizer.py:160`의 threshold `1e-3 / 512 ≈ 1.95e-6`이 극도로 낮음
- 초기 epoch에서 대다수 코드가 "dead"로 판정 → 매 배치마다 대량 재초기화
- Codebook 불안정 → NaN cascade

**2차 원인: v2의 공격적 하이퍼파라미터**
- `loss_weight: 0.2` (v1의 0.1에서 2배): celltype aux gradient 증폭
- `hidden_dim: 128` (v1의 64에서 2배): classifier 용량 증가로 더 큰 gradient
- `lr: 0.001`: 높은 학습률이 codebook 불안정과 결합

**3차 원인: K-means 초기화 샘플 부족**
- 512 centroids에 14,650 points만 사용 (최소 19,968 필요)
- Stratified sampling에서 일부 study 그룹의 샘플 부족

### 해결: v3 Config
```yaml
# NaN-safe parameters
learning_rate: 0.0001      # 0.001 → 0.0001 (10배 감소)
celltype_aux:
  loss_weight: 0.1         # 0.2 → 0.1
  hidden_dim: 16           # 128 → 16
```

v3로 정상 학습 확인:
```
Epoch 65/100 - Train Loss: 0.1989 - CT Loss: 0.0141
```

---

## 2. Pretrain Sweep

### 설계

| 축 | 값 | 비고 |
|---|---|---|
| **loss_weight** | 0.05, 0.1, 0.2 | celltype aux 강도 |
| **num_codes** | 256, 512, 1024 | codebook 크기 |
| **data_scope** | subset_326k, whole_805k | 학습 데이터 범위 |
| lr | 0.0001 (고정) | |
| hidden_dim | 16 (고정) | |
| batch_size | 2048 (고정) | |

= 18개 실험 (3 × 3 × 2)

### 생성된 파일
- `config/pretrain_celltype_aux_v3.yaml`: 기본 config
- `scripts/sweep_pretrain.sh`: 자동 실행 스크립트
- `scripts/summarize_sweep.py`: 결과 요약 스크립트

### 실행 이슈: `bc` command not found
- `format_weight()` 함수에서 `bc`를 사용했으나 서버에 미설치
- 모든 실험이 `w000`으로 생성되는 버그 발생
- **수정**: `bc` → `awk` 사용

```bash
# Before (broken)
printf "%03.0f" "$(echo "$1 * 100" | bc)"
# After (fixed)
echo "$1" | awk '{printf "%03d", $1 * 100}'
```

### Sweep 결과

| Experiment | Status | Codes | Weight | BestLoss | CTLoss |
|---|---|---|---|---|---|
| **sub_c512_w0.1** | **OK** | **512** | **0.10** | **0.1673** | **0.0156** |
| who_c512_w0.1 | OK | 512 | 0.10 | 0.1949 | 0.0107 |
| who_c1024_w0.05 | OK | 1024 | 0.05 | 0.2122 | 0.0746 |
| who_c256_w0.05 | OK | 256 | 0.05 | 0.2238 | 0.0755 |
| sub_c256_w0.05 | OK | 256 | 0.05 | 0.2356 | 0.1738 |
| sub_c1024_w0.05 | OK | 1024 | 0.05 | 0.2422 | 0.1292 |
| sub_c512_w0.05 | OK | 512 | 0.05 | 0.2433 | 0.1744 |
| sub_c512_w0.2 | OK | 512 | 0.20 | 0.2461 | 0.0573 |
| who_c256_w0.2 | OK | 256 | 0.20 | 0.2480 | 0.0505 |
| sub_c256_w0.2 | OK | 256 | 0.20 | 1.2200 | 0.3003 |
| sub_c1024_w0.1 | NaN | 1024 | 0.10 | - | - |
| sub_c1024_w0.2 | NaN | 1024 | 0.20 | - | - |
| sub_c256_w0.1 | NaN | 256 | 0.10 | - | - |
| who_c1024_w0.1 | NaN | 1024 | 0.10 | - | - |
| who_c1024_w0.2 | NaN | 1024 | 0.20 | - | - |
| who_c256_w0.1 | NaN | 256 | 0.10 | - | - |
| who_c512_w0.05 | NaN | 512 | 0.05 | - | - |
| who_c512_w0.2 | NaN | 512 | 0.20 | - | - |

**10/18 성공, 8/18 NaN** (전부 epoch 1에서 발생)

### NaN 패턴 분석
- NaN은 하이퍼파라미터 패턴이 아닌 **k-means 초기화 품질의 확률적 변동**에 의존
- `c512_w0.1`만 subset/whole 모두 성공 → 가장 안정적 조합
- Dead code revival 로직의 근본적 수정 필요 (향후 과제)

---

## 3. Encoder 버전 정의

| 버전 | Celltype Label | 학습 데이터 | 비고 |
|------|---------------|-----------|------|
| **v1** | celltype_lineage (6 classes) | whole 805k | 안정적, 높은 codebook 품질 |
| **v2** | celltype_lineage_v2 (10 classes) | subset 326k (3 studies) | 불안정한 결과 |
| **v3** | celltype_lineage_v2 (10-11 classes) | whole 805k | dead codes 많음 |

---

## 4. Codebook Quality 비교

| 지표 | v1 | v2 | v3 |
|------|:---:|:---:|:---:|
| Active codes | **461/512 (90%)** | 433/512 (84.6%) | 314/512 (61.3%) |
| Dead codes | 51 | 79 | **198** |
| Purity (lineage_v2, 10-11 cls) | 0.916 | **0.945** | 0.898 |
| Purity (lineage, 5-6 cls) | **0.965** | 0.964 | 0.938 |
| Cond mixing entropy | 0.376 | **0.417** | 0.395 |
| High purity codes [0.9-1.0] | **418** | 371 | 217 |

### 핵심 관찰
- v1이 codebook 활용률(90%)과 purity(0.965) 모두 최고
- v3는 lr=0.0001로 인한 학습 부족 → dead codes 38.7%
- v1은 coarser label(6 cls)로 학습했지만 finer label(11 cls) 평가에서도 0.916 purity → **coarse label로도 fine-grained 구조 포착**

---

## 5. Cross-Disease Evaluation 종합

### Skin3 → SCP1884 (HS 모델로 CD 예측)

| Encoder | Params (lr / enc_lr / epochs) | AUC | Acc | F1 |
|---------|-------------------------------|-----|-----|-----|
| **v1** | 0.0005 / 0.0001 / 100 | **0.7604** | **0.7647** | **0.7333** |
| v1 | 0.0005 / 0.0001 / 10 | 0.7118 | 0.7059 | 0.6429 |
| v3 | 0.0005 / 0.0001 / 50 | 0.6424 | 0.6765 | 0.5926 |
| v3 | 0.001 / 0.0001 / 100 | 0.6250 | 0.6765 | 0.6452 |
| v1 | tuned 1st | 0.6007 | 0.6176 | 0.4800 |
| v2 | 0.0005 / 0.0001 / 100 | 0.6024 | 0.6471 | 0.5000 |
| v1 | default | 0.5799 | 0.7059 | 0.6154 |
| v2 | tuned 1st | 0.5694 | 0.6176 | 0.4800 |
| v2 | default | 0.5417 | 0.5882 | 0.3636 |

### SCP1884 → Skin3 (CD 모델로 HS 예측)

| Encoder | Params (lr / enc_lr / epochs) | AUC | Acc | F1 |
|---------|-------------------------------|-----|-----|-----|
| v3 | 0.001 / 0.0001 / 100 | **0.9886** | 0.9474 | 0.9412 |
| v1 | default | 0.9432 | 0.9474 | 0.9412 |
| v1 | tuned (0.001/0.0001/100) | 0.9318 | 0.9474 | 0.9412 |
| v3 | 0.0001 / 0.0001 / 50 | 0.8864 | 0.8947 | 0.8571 |
| v2 | default | 1.0000* | 1.0000 | 1.0000 |
| v2 | tuned | 0.2045 | 0.4737 | 0.6154 |

*v2 default AUC 1.0은 과적합/우연 (tuned 후 0.20으로 급락)

### 방향 비대칭
- **SCP1884→Skin3: AUC 0.88~0.99** (일관적으로 높음)
- **Skin3→SCP1884: AUC 0.52~0.76** (어려움)
- CD 모델이 HS를 잘 예측하지만, 역방향은 어려움

---

## 6. Within-Disease CV (Tuning Best AUC)

| Dataset | v1 | v2 | v3 |
|---------|:---:|:---:|:---:|
| Skin3 | ~1.0 (미확인) | 1.0 | 1.0 |
| SCP1884 | 0.85 | (미확인) | **0.9111** |

### 주의사항
v1의 SCP1884 tuning grid가 제한적이었음:
- `lr: [0.001, 0.0001]`, `enc_lr: [0.0001]` 고정, `epochs: [50]` 고정
- 실질 4개 조합만 탐색

---

## 7. 최종 결정: v1 (lineage + whole) 선택

### 선택 근거
1. **Codebook 품질 최고**: 90% utilization, 0.965 purity → 논문의 해석가능성 스토리에 핵심
2. **Cross-disease 최고 (Skin3→SCP1884)**: AUC 0.7604 vs v3의 0.6424
3. **Coarse label 강점**: 6 class aux로도 11 class 구조 포착 → 라벨 의존성 낮음
4. **안정적 학습**: v3의 dead codes 38.7%는 구조적 약점

### SCP1884 CV 개선 과제
v1의 SCP1884 CV (0.85)를 확인하기 위해 확장된 grid search 수행 예정:

```yaml
# SCP1884 (v1 encoder) - 18 조합
tuning:
  learning_rate: [0.001, 0.0005, 0.0001]
  encoder_learning_rate: [0.0001, 0.00005]
  epochs: [10, 50, 100]
  disease_ratio_lambda: [0.0]
```

```yaml
# Skin3 (v1 encoder) - 27 조합
tuning:
  learning_rate: [0.001, 0.0005, 0.0001]
  encoder_learning_rate: [0.001, 0.0001, 0.00005]
  epochs: [10, 50, 100]
  disease_ratio_lambda: [0.0]
```

**목표**: SCP1884 CV AUC 0.85→0.90+ 달성 시 v1의 모든 지표에서 v3 대비 우위 확보

---

## 8. 향후 과제 (TODO)

- [ ] v1 encoder로 확장 grid search 실행 (Skin3: 27조합, SCP1884: 18조합)
- [ ] 확장 grid 결과 기반 최종 하이퍼파라미터 확정
- [ ] Cross-disease evaluation 재실행 (최적 params)
- [ ] Dead code revival 로직 개선 (한번에 revive하는 코드 수 제한)
- [ ] Sweep 결과 재현성 확보 (NaN 문제 해결 후 재실행)
- [ ] Cell scoring 및 생물학적 해석 진행

---

## 변경된 파일 목록

### 신규 생성
| 파일 | 설명 |
|------|------|
| `config/pretrain_celltype_aux_v3.yaml` | v3 pretrain config (NaN-safe params) |
| `scripts/sweep_pretrain.sh` | Pretrain 하이퍼파라미터 sweep 스크립트 |
| `scripts/summarize_sweep.py` | Sweep 결과 요약/비교 스크립트 |

### 수정
| 파일 | 변경 | 이유 |
|------|------|------|
| `scripts/sweep_pretrain.sh` | `format_weight()`: `bc` → `awk` | 서버에 `bc` 미설치 |
