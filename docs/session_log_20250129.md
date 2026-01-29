# scMILD 리팩토링 세션 로그

**날짜:** 2025-01-29
**목적:** 코드 구조 단순화, LOOCV 구현, 모듈화 개선

---

## 세션 요약

scMILD 코드베이스의 대규모 리팩토링을 진행했습니다. 복잡했던 코드 구조를 모듈화하고, LOOCV 분할 전략을 구현했으며, YAML 기반 설정 시스템을 도입했습니다.

---

## 완료된 작업

### 1. 설정 시스템 구현 (YAML + Dataclass)

| 파일 | 설명 |
|------|------|
| `config/default.yaml` | 기본 설정 파일 |
| `config/skin3.yaml` | Skin3 (HS) 데이터셋 설정 |
| `config/scp1884.yaml` | SCP1884 (CD) 데이터셋 설정 |
| `src/config.py` | YAML 로더 + Dataclass 정의 |

**특징:**
- 변수 참조 해결 (`${paths.data_root}` 형태)
- `_base_` 키를 통한 설정 상속
- 타입 안전한 dataclass 기반 설정

### 2. 모듈 구조 재구성

```
src/
├── config.py                 # 설정 관리
├── models/
│   ├── __init__.py
│   ├── attention.py         # AttentionModule, GatedAttentionModule
│   ├── branches.py          # TeacherBranch, StudentBranch
│   ├── autoencoder.py       # AENB, VQ_AENB, VQ_AENB_Conditional
│   ├── encoder_wrapper.py   # VQEncoderWrapper, VQEncoderWrapperConditional
│   └── quantizer.py         # Quantizer (Vector Quantization)
│
├── data/
│   ├── __init__.py
│   ├── dataset.py           # MilDataset, InstanceDataset
│   ├── splitter.py          # LOOCVSplitter, StratifiedKFoldSplitter
│   └── preprocessing.py     # preprocess_adata, encode_labels
│
└── training/
    ├── __init__.py
    ├── trainer.py           # MILTrainer (from optimizer.py)
    ├── trainer_ae.py        # AETrainer
    └── metrics.py           # compute_metrics, MetricsLogger
```

### 3. LOOCV Splitter 구현

```python
# src/data/splitter.py

class LOOCVSplitter:
    """Leave-One-Out Cross Validation"""
    def split(sample_ids, labels, sample_names) -> Iterator[FoldInfo]:
        # 각 샘플을 한 번씩 테스트 세트로 사용

class StratifiedKFoldSplitter:
    """Stratified K-Fold (Repeated 지원)"""
    ...
```

### 4. 통합 스크립트 생성

| 스크립트 | 용도 |
|----------|------|
| `scripts/01_pretrain_encoder.py` | VQ-AENB-Conditional pretrain |
| `scripts/02_train_loocv.py` | LOOCV 기반 MIL 학습 |
| `scripts/03_finalize_model.py` | 전체 데이터로 최종 모델 학습 |
| `scripts/04_cross_disease_eval.py` | Cross-disease 일반화 평가 |
| `scripts/05_cell_scoring.py` | 세포 레벨 점수 계산 |

### 5. Conditional Embedding 설정 추가

`config/default.yaml`에 조건부 임베딩 컬럼 설정 추가:

```yaml
data:
  conditional_embedding:
    column: "study"                      # 원본 컬럼명 (study, Organ 등)
    encoded_column: "study_id_numeric"   # 인코딩된 컬럼명
```

`src/config.py`에 `ConditionalEmbeddingConfig` dataclass 추가:

```python
@dataclass
class ConditionalEmbeddingConfig:
    """Conditional encoder용 임베딩 컬럼 설정"""
    column: str = "study"
    encoded_column: str = "study_id_numeric"
```

이를 통해 'study' 또는 'Organ' 등 다양한 조건부 임베딩 컬럼을 유연하게 사용 가능

### 6. Disease Ratio Regularization (Phase 2)

VQ 코드북의 각 코드별 질병 비율을 타겟으로 하여 attention score가 이 비율과 유사해지도록 정규화합니다.

**새로 추가된 파일:**
- `src/training/disease_ratio.py` - 질병 비율 계산 및 정규화 유틸리티

**수정된 파일:**
- `src/training/trainer.py` - `disease_ratio`, `ratio_reg_lambda` 파라미터 추가
- `src/config.py` - `DiseaseRatioRegConfig` dataclass 추가
- `config/default.yaml` - `disease_ratio_reg` 설정 섹션 추가
- `scripts/02_train_loocv.py` - disease ratio 계산 및 trainer에 전달

**설정 예시 (`config/default.yaml`):**
```yaml
mil:
  loss:
    disease_ratio_reg:
      enabled: true           # 활성화
      lambda_weight: 0.1      # regularization 강도
      alpha: 1.0              # Beta prior smoothing
      beta: 1.0
```

### 7. Legacy 파일 정리

기존 파일들을 `legacy/` 폴더로 이동:
- `01A-tuning_scMILD_conditional.py`
- `01B-tuning_scMILD_parallele.py`
- `02-testing_scMILDQ_conditional.py`
- `03-cell_score_scMILDQ_conditional.py`
- `preprocess_adata.py`
- `pretraining_autoencoder.py`
- `train_scMILD.py`
- `runs/` (데이터셋별 스크립트들)

---

## 주요 개선사항

### Before (문제점)
1. 같은 이름의 함수가 2개 정의됨 (`load_and_save_datasets_adata`)
2. Split 로직이 3곳에서 반복
3. 8단계 이상의 중첩 조건 분기
4. 하드코딩된 경로 10개 이상
5. 실행 순서 파악 어려움

### After (개선)
1. 명확한 모듈 분리 (models, data, training)
2. 단일 책임 원칙 적용
3. YAML 기반 설정으로 경로 관리
4. 단일 진입점 스크립트 (`02_train_loocv.py`)
5. Type hints 및 docstring 추가

---

## 새로운 폴더 구조

```
scMILD/
├── config/                   # YAML 설정 파일들
│   ├── default.yaml
│   ├── skin3.yaml
│   └── scp1884.yaml
│
├── src/                      # 리팩토링된 소스 코드
│   ├── config.py
│   ├── models/
│   ├── data/
│   └── training/
│
├── scripts/                  # 실행 스크립트
│   ├── 01_pretrain_encoder.py
│   ├── 02_train_loocv.py
│   ├── 03_finalize_model.py
│   ├── 04_cross_disease_eval.py
│   └── 05_cell_scoring.py
│
├── legacy/                   # 기존 코드 (참조용)
│   ├── 01A-tuning_scMILD_conditional.py
│   ├── optimizer.py (원본)
│   └── ...
│
├── results/                  # 결과 저장 (gitignore)
└── docs/                     # 문서
    └── session_log_20250129.md
```

---

## 파이프라인 실행 방법

### Step 1: Encoder Pretrain (한 번만)
```bash
python scripts/01_pretrain_encoder.py \
    --adata_path /path/to/whole_data.h5ad \
    --output_dir /path/to/output \
    --gpu 0
```

### Step 2: LOOCV 학습
```bash
# Skin3 (HS 분류)
python scripts/02_train_loocv.py --config config/skin3.yaml --gpu 0

# SCP1884 (CD 분류)
python scripts/02_train_loocv.py --config config/scp1884.yaml --gpu 0
```

### Step 3: Final Model 학습 (전체 데이터)
```bash
python scripts/03_finalize_model.py \
    --config config/skin3.yaml \
    --gpu 0
```

### Step 4: Cross-disease 평가
```bash
# Skin3 모델로 SCP1884 평가
python scripts/04_cross_disease_eval.py \
    --model_dir results/final_model_xxx \
    --source_config config/skin3.yaml \
    --target_config config/scp1884.yaml \
    --gpu 0
```

### Step 5: Cell-level 점수 계산
```bash
python scripts/05_cell_scoring.py \
    --model_dir results/final_model_xxx \
    --config config/skin3.yaml \
    --gpu 0
```

---

## TODO (다음 세션)

### Phase 1 - 완료 ✅
- [x] `scripts/03_finalize_model.py` - 최적 HP로 final model 학습
- [x] `scripts/04_cross_disease_eval.py` - Cross-disease 평가
- [x] `scripts/05_cell_scoring.py` - Cell-level 분석
- [x] Conditional embedding 컬럼 설정 추가 (study/Organ 선택 가능)

### Phase 2 - 완료 ✅
- [x] Disease ratio regularization 구현

### Phase 3 - 나중에
- [ ] Repeated Stratified K-Fold 옵션 추가
- [ ] MLflow/W&B 통합

---

## 데이터셋 정보 (참고)

| Dataset | 용도 | 샘플 수 | 세포 수 | 질병 분포 |
|---------|------|---------|---------|-----------|
| **SCP1884** | CD 분류 | 34 | 289,730 | CD 18 / Ctrl 16 |
| **Skin3** | HS 분류 | 19 | 36,595 | HS 8 / Ctrl 11 |

---

## 병원망 경로 (참고)

```yaml
# config/default.yaml 기준
paths:
  data_root: "/home/bmi-user/workspace/data/HSvsCD"
  pretrained_encoder: "${paths.data_root}/data_conditional/AE/vq_aenb_conditional_whole.pth"
  output_root: "${paths.data_root}/results"
  conda_env: "/home/bmi-user/workspace/data/scvi-env"
```

---

## 다음 세션 첨부 권장 파일

### 필수
1. 이 로그 파일 (`docs/session_log_20250129.md`)
2. 이전 로그 파일 (있다면)

### 결과 확인용 (있다면)
3. `results/loocv_*/results.csv`

### 참조용
4. `config/skin3.yaml` 또는 `config/scp1884.yaml`

---

## 핵심 코드 위치 요약

| 기능 | 파일 |
|------|------|
| 설정 로드 | `src/config.py` → `load_config()` |
| LOOCV 분할 | `src/data/splitter.py` → `LOOCVSplitter` |
| MIL 학습 | `src/training/trainer.py` → `MILTrainer.train_fold()` |
| Encoder 래핑 | `src/models/encoder_wrapper.py` → `VQEncoderWrapperConditional` |
| 메트릭 계산 | `src/training/metrics.py` → `compute_metrics()` |
| Conditional 임베딩 | `src/config.py` → `ConditionalEmbeddingConfig` |

---

## 출력 결과물 설명

### LOOCV 학습 결과 (`results/loocv_*/`)
```
loocv_20250129_123456/
├── results.csv           # 폴드별 메트릭 (AUC, Accuracy, F1)
└── models/               # 폴드별 저장된 모델 (optional)
    ├── model_teacher_fold0.pt
    ├── model_student_fold0.pt
    └── model_encoder_fold0.pt
```

### Final Model 결과 (`results/final_model_*/`)
```
final_model_20250129_123456/
├── model_teacher_fold0.pt    # 전체 데이터로 학습된 teacher
├── model_student_fold0.pt    # 전체 데이터로 학습된 student
├── model_encoder_fold0.pt    # 전체 데이터로 학습된 encoder wrapper
└── training_config.yaml      # 사용된 설정 백업
```

### Cross-disease 평가 결과 (`results/cross_disease_*/`)
```
cross_disease_eval_20250129_123456/
├── evaluation_results.json   # 전체 메트릭
├── sample_predictions.csv    # 샘플별 예측
└── cell_scores.csv           # 세포별 점수 (optional)
```

### Cell Scoring 결과 (`results/cell_scores_*/`)
```
cell_scores_20250129_123456/
├── cell_scores.csv           # 세포별 attention score, student prediction
├── sample_summary.csv        # 샘플별 요약 통계
└── cell_embeddings.npy       # 세포 임베딩 (optional, large)
```

---

*Generated: 2025-01-29*
*Updated: 2025-01-29 (Phase 1 완료, 전체 파이프라인 스크립트 추가)*
