# Session Log - 2026-01-30

## 주요 작업 내용

### 1. study_ids가 MIL 학습 시 encoder에 제대로 전달되는지 확인 및 수정

#### 문제점
- Subset 데이터(예: Skin3)로 MIL 학습 시 `study_id_numeric` 컬럼이 캐시된 adata에 없음
- Conditional encoder에 study_ids가 전달되지 않아 경고 발생:
  ```
  WARNING: study_ids not provided to conditional encoder, using 0
  ```

#### 해결책
1. **Pretrain 시 `study_mapping.json` 저장** (`scripts/01_pretrain_encoder.py`)
   - `{study_id: study_name}` 형태로 저장
   - 예: `{"0": "GSE175990", "1": "GSE220116", "2": "SCP1884"}`

2. **MIL 학습 시 매핑 파일 로드 후 적용** (`scripts/02_train_loocv.py`, `06_tune_hyperparams.py`)
   - `study_mapping.json`을 로드하여 `{study_name: study_id}` 매핑 생성
   - adata.obs의 `study` 컬럼을 매핑하여 `study_id_numeric` 컬럼 생성

#### 수정된 파일
- `scripts/01_pretrain_encoder.py`: study_mapping.json 생성 및 저장
- `scripts/02_train_loocv.py`: study_mapping.json 로드 및 적용
- `scripts/06_tune_hyperparams.py`: study_mapping.json 로드 및 적용
- `config/default.yaml`: `conditional_embedding.mapping_path` 설정 추가
- `src/config.py`: `ConditionalEmbeddingConfig`에 `mapping_path` 필드 추가
- `src/data/__init__.py`: `load_study_mapping`, `save_study_mapping` export 추가

---

### 2. LOOCV에서 fold별 prob을 concat해서 전체 AUROC 계산하는 로직 구현

#### 문제점
- LOOCV에서 각 fold의 테스트 샘플이 1개이므로 fold별 AUC 계산이 무의미
- 기존 코드는 fold별 AUC를 평균하는 방식 사용

#### 해결책
1. **`FoldResult`에 예측 결과 저장** (`src/training/trainer.py`)
   - `y_true`, `y_pred_proba` 필드 추가

2. **모든 fold의 예측을 concatenate하여 전체 AUROC 계산**
   ```python
   all_y_true = np.concatenate([r.y_true for r in all_results])
   all_y_pred_proba = np.concatenate([r.y_pred_proba for r in all_results])
   overall_auc = roc_auc_score(all_y_true, all_y_pred_proba)
   ```

3. **결과 파일 저장** (`scripts/02_train_loocv.py`)
   - `overall_results.csv`: 전체 AUROC, Accuracy, F1 Score
   - `predictions.csv`: 각 샘플의 예측 확률

#### 수정된 파일
- `src/training/trainer.py`: FoldResult에 y_true, y_pred_proba 필드 추가
- `scripts/02_train_loocv.py`: 전체 AUROC 계산 및 결과 파일 저장
- `scripts/06_tune_hyperparams.py`: 전체 AUROC 계산 방식으로 변경

---

### 3. LOOCV에서 불필요한 fold별 메트릭 계산 제거

#### 문제점
- LOOCV에서 fold별 AUC, accuracy 등을 계산하면 sklearn 경고 발생
  - `Only one class is present in y_true. ROC AUC score is not defined in that case.`
- 불필요한 함수 호출로 인한 오버헤드

#### 해결책
1. **`train_fold()` 메서드에 `skip_fold_metrics` 파라미터 추가** (`src/training/trainer.py`)
   - `skip_fold_metrics=True`일 때 `_evaluate()`, `compute_metrics()` 호출 건너뜀
   - 예측값(`y_true`, `y_pred_proba`)만 반환

2. **스크립트에서 `skip_fold_metrics=True` 전달**
   - `scripts/02_train_loocv.py`
   - `scripts/06_tune_hyperparams.py`

3. **fold별 출력 변경**
   - 기존: `AUC: 0.5000, Acc: 1.0000, F1: 0.0000` (무의미한 메트릭)
   - 변경: `prob=0.1906 (pred=Control, true=Control) ✓` (예측 결과)

#### 수정된 파일
- `src/training/trainer.py`: `skip_fold_metrics` 파라미터 추가
- `scripts/02_train_loocv.py`: `skip_fold_metrics=True` 전달, fold별 출력 변경
- `scripts/06_tune_hyperparams.py`: `skip_fold_metrics=True` 전달

---

### 4. 하이퍼파라미터 튜닝 개선 (`06_tune_hyperparams.py`)

#### 문제점
- 튜닝 중 크래시 시 모든 결과 손실 (중간 저장 없음)
- 모델 저장 없음 - 최적 하이퍼파라미터를 찾아도 다시 학습해야 함
- best_params가 터미널에만 출력됨

#### 해결책
1. **실시간 CSV 저장**: 각 조합 완료 시마다 `tuning_results.csv` 업데이트
2. **Top-K 모델 저장**: 상위 K개 조합의 fold별 모델 저장 (Min-heap 사용)
3. **best_params.yaml**: 최적 하이퍼파라미터 별도 파일 저장

#### 출력 구조
```
tuning_{timestamp}/
├── tuning_results.csv      # 실시간 업데이트 (각 조합 완료마다)
├── best_params.yaml        # 최적 하이퍼파라미터
└── models/                 # Top-K 조합의 모델들
    ├── config_005/
    │   ├── params.yaml
    │   ├── fold_00/
    │   │   ├── teacher.pth
    │   │   ├── student.pth
    │   │   ├── encoder.pth
    │   │   └── info.yaml
    │   └── fold_01/
    ...
```

#### 수정된 파일
- `scripts/06_tune_hyperparams.py`: TopKTracker 클래스, 실시간 저장 로직 추가
- `src/config.py`: `TuningConfig.save_top_k` 필드 추가

---

### 5. 기본값 수정

#### Student optimize_period
- **변경 전**: `optimize_period: 3`
- **변경 후**: `optimize_period: 1` (매 epoch마다 student 최적화)

#### 튜닝 epochs
- **변경 전**: `epochs: [100, 50]`
- **변경 후**: `epochs: [10, 30]` (Transfer learning이므로 작은 epoch)

#### 수정된 파일
- `config/default.yaml`
- `src/config.py`

---

## 실행 흐름 (수정 후)

```bash
# 1. Encoder Pretrain (처음 한 번, 전체 데이터)
python scripts/01_pretrain_encoder.py --config config/default.yaml

# 1.1 Pretrain 결과물 배치 (수동) - 중요!
cp results/pretrain_*/vq_aenb_conditional.pth results/pretrained/vq_aenb_conditional_whole.pth
cp results/pretrain_*/study_mapping.json results/pretrained/study_mapping.json

# 2. MIL 학습 (LOOCV) - subset 데이터 사용
python scripts/02_train_loocv.py --config config/skin3.yaml --gpu 0

# 3. 하이퍼파라미터 튜닝 (선택)
python scripts/06_tune_hyperparams.py --config config/skin3.yaml --gpu 0 --verbose
```

---

## Config 변경사항

### `config/default.yaml`
```yaml
data:
  conditional_embedding:
    column: "study"
    encoded_column: "study_id_numeric"
    mapping_path: "${paths.project_root}/results/pretrained/study_mapping.json"  # 추가됨
```

---

## 출력 파일 구조

```
results/
├── pretrained/
│   ├── vq_aenb_conditional_whole.pth  # 모델 가중치
│   └── study_mapping.json              # Study ID 매핑 (필수!)
├── pretrain_YYYYMMDD_HHMMSS/
│   ├── vq_aenb_conditional.pth
│   ├── study_mapping.json              # 자동 생성
│   └── training_history.json
└── skin3/
    ├── loocv_YYYYMMDD_HHMMSS/
    │   ├── results.csv                 # Fold별 메트릭
    │   ├── overall_results.csv         # 전체 AUROC (NEW)
    │   ├── predictions.csv             # 샘플별 예측 확률 (NEW)
    │   └── models/
    └── tuning_YYYYMMDD_HHMMSS/
        ├── tuning_results.csv          # 실시간 업데이트
        ├── best_params.yaml            # 최적 하이퍼파라미터 (NEW)
        └── models/                     # Top-K 모델 (NEW)
            ├── config_XXX/
            │   ├── params.yaml
            │   └── fold_XX/
            ...
```

---

## 주의사항

1. **Pretrain 후 반드시 `study_mapping.json` 복사**
   - 이 파일이 없으면 MIL 학습 시 study_ids가 올바르게 전달되지 않음

2. **Subset 캐시 재사용 시**
   - 캐시된 adata에는 `study_id_numeric` 컬럼이 없음
   - 런타임에 `study_mapping.json`을 사용하여 생성

3. **LOOCV 평가 시**
   - fold별 AUC는 의미 없음 (테스트 샘플 1개)
   - `overall_results.csv`의 전체 AUROC 사용
   - `skip_fold_metrics=True`로 불필요한 계산 건너뜀

4. **하이퍼파라미터 튜닝 시**
   - Transfer learning이므로 작은 epoch 사용 (10, 30 권장)
   - `save_top_k` 설정으로 상위 K개 조합 모델 저장
   - 중간에 크래시 발생해도 `tuning_results.csv`에 결과 보존

---

# Session Log - 2026-01-30 (3차)

## 주요 작업 내용

### 1. `03_finalize_model.py`에 `--best_params` 옵션 추가

#### 목적
- 튜닝 스크립트(`06_tune_hyperparams.py`)에서 생성된 `best_params.yaml`을 자동으로 로드하여 config 오버라이드

#### 구현 내용

1. **`load_best_params()` 함수 추가**
   ```python
   def load_best_params(best_params_path: str) -> dict:
       """Load best hyperparameters from YAML file."""
       with open(best_params_path, 'r') as f:
           data = yaml.safe_load(f)
       return data.get('best_hyperparameters', {})
   ```

2. **`apply_best_params()` 함수 추가**
   - 지원하는 파라미터와 config 경로 매핑:
   ```python
   param_mapping = {
       'learning_rate': ('mil', 'training', 'learning_rate'),
       'encoder_learning_rate': ('mil', 'training', 'encoder_learning_rate'),
       'attention_dim': ('mil', 'attention_dim'),
       'latent_dim': ('mil', 'latent_dim'),
       'projection_dim': ('mil', 'projection_dim'),
       'negative_weight': ('mil', 'loss', 'negative_weight'),
       'student_optimize_period': ('mil', 'student', 'optimize_period'),
       'epochs': ('mil', 'training', 'epochs'),
       'disease_ratio_lambda': ('mil', 'loss', 'disease_ratio_reg', 'lambda_weight'),
   }
   ```

3. **Disease Ratio Regularization 지원**
   - `disease_ratio_lambda > 0`일 때 disease ratio 계산 후 trainer에 전달
   ```python
   if disease_ratio_lambda > 0:
       disease_ratio = calculate_disease_ratio_from_dataloader(
           instance_dl, model_encoder, device, ...
       )
   trainer = MILTrainer(..., disease_ratio=disease_ratio, ratio_reg_lambda=disease_ratio_lambda)
   ```

4. **사용법**
   ```bash
   python scripts/03_finalize_model.py \
       --config config/skin3.yaml \
       --best_params results/skin3/tuning_*/best_params.yaml \
       --gpu 0
   ```

#### 수정된 파일
- `scripts/03_finalize_model.py`

---

### 2. 전체 스크립트 데이터 로딩 통일

#### 문제점
- `03_finalize_model.py`, `04_cross_disease_eval.py`, `05_cell_scoring.py`가 구버전 API 사용
  - `load_adata()` 대신 `load_adata_with_subset()` 사용해야 함
  - `config.data.adata_path` 존재하지 않음 → `config.data.whole_adata_path` 사용

#### 해결책
모든 스크립트를 `02_train_loocv.py`, `06_tune_hyperparams.py`와 동일한 패턴으로 통일:

1. **데이터 로딩**
   ```python
   adata = load_adata_with_subset(
       whole_adata_path=config.data.whole_adata_path,
       subset_enabled=config.data.subset.enabled,
       subset_column=config.data.subset.column,
       subset_values=config.data.subset.values,
       cache_dir=config.data.subset.cache_dir,
       use_cache=config.data.subset.use_cache,
   )
   ```

2. **Study ID Mapping**
   - 각 스크립트의 dataloader 생성 함수 내에서 study mapping 적용
   ```python
   if embedding_col not in adata.obs.columns and embedding_source_col in adata.obs.columns:
       if embedding_mapping_path and Path(embedding_mapping_path).exists():
           id_to_name = load_study_mapping(embedding_mapping_path)
           name_to_id = {v: k for k, v in id_to_name.items()}
           adata.obs[embedding_col] = adata.obs[embedding_source_col].map(name_to_id)
   ```

3. **Embedding IDs 직접 컬럼 사용**
   - 기존: `embedding_mapping` dict를 통해 sample_id → study_id 변환
   - 변경: 직접 `adata.obs[embedding_col]`에서 가져옴
   ```python
   if embedding_col in adata.obs.columns:
       embedding_ids = torch.tensor(
           adata.obs[embedding_col].values.astype(int),
           dtype=torch.long, device=device
       )
   ```

#### 수정된 파일
- `scripts/03_finalize_model.py`: `create_full_dataloaders()` 함수 수정
- `scripts/04_cross_disease_eval.py`: `create_test_dataloader()` 함수 수정, import 변경
- `scripts/05_cell_scoring.py`: `ensure_embedding_column()` 함수 추가, `compute_cell_scores()` 수정

---

### 3. 문서 업데이트

#### CLAUDE.md
- 변경 이력에 3차 세션 내용 추가
- TODO 항목 상태 업데이트 (완료)

---

## 수정된 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `scripts/03_finalize_model.py` | `--best_params` 옵션, disease ratio 지원, 데이터 로딩 통일 |
| `scripts/04_cross_disease_eval.py` | 데이터 로딩 통일, study mapping 로직 추가 |
| `scripts/05_cell_scoring.py` | 데이터 로딩 통일, `ensure_embedding_column()` 함수 추가 |
| `CLAUDE.md` | 변경 이력 추가 |
| `docs/session_log_20260130.md` | 세션 로그 업데이트 |

---

## 파이프라인 실행 예시 (전체)

```bash
# 1. Encoder Pretrain (처음 한 번)
python scripts/01_pretrain_encoder.py --config config/default.yaml --gpu 0

# 1.1 결과물 배치 (수동)
cp results/pretrain_*/vq_aenb_conditional.pth results/pretrained/vq_aenb_conditional_whole.pth
cp results/pretrain_*/study_mapping.json results/pretrained/study_mapping.json

# 2. LOOCV Baseline
python scripts/02_train_loocv.py --config config/skin3.yaml --gpu 0

# 3. 하이퍼파라미터 튜닝
python scripts/06_tune_hyperparams.py --config config/skin3.yaml --gpu 0 --verbose
# → results/skin3/tuning_*/best_params.yaml 생성

# 4. Final Model 학습 (best params 적용)
python scripts/03_finalize_model.py \
    --config config/skin3.yaml \
    --best_params results/skin3/tuning_*/best_params.yaml \
    --gpu 0
# → results/skin3/final_model_*/

# 5a. Cell Scoring (학습 데이터)
python scripts/05_cell_scoring.py \
    --model_dir results/skin3/final_model_* \
    --config config/skin3.yaml \
    --gpu 0

# 5b. Cross-disease 평가
python scripts/04_cross_disease_eval.py \
    --model_dir results/skin3/final_model_* \
    --test_config config/scp1884.yaml \
    --gpu 0
```

---

## best_params.yaml 형식

```yaml
best_hyperparameters:
  disease_ratio_lambda: 0.05
  encoder_learning_rate: 0.0001
  epochs: 30
  learning_rate: 0.001
best_score:
  metric: auc
  value: 1.0
timestamp: '2026-01-30T01:33:46.123456'
```
