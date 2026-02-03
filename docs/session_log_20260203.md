# Session Log - 2026-02-03

## 개요
scMILD 프로젝트의 코드 개선 작업을 진행했습니다. YAML 관리 개선, CV 스크립트 일반화, 변수명 범용화 등 다양한 개선 사항을 구현했습니다.

---

## 수정 사항

### 1. `02_train_cv.py` 신규 생성 (기존 `02_train_loocv.py` 대체)

**문제점:**
- 기존 스크립트가 LOOCV만 지원
- `overall_results.csv`가 생성되지 않는 오류 발생
- Repeated K-Fold 등 다른 CV 전략 사용 불가

**해결:**
- `02_train_cv.py` 신규 생성
- LOOCV, Stratified K-Fold, Repeated Stratified K-Fold 지원
- `config.splitting.strategy`에 따라 자동으로 splitter 선택
- 예외 처리 추가: 학습 중 오류 발생 시 부분 결과 저장
- 출력 디렉토리명: `cv_{strategy}_{timestamp}`

**수정 파일:**
- `scripts/02_train_cv.py` (신규)
- `scripts/02_train_loocv.py` (deprecated 경고 추가)

---

### 2. Pretrain 결과물 자동 배치

**문제점:**
- Pretrain 후 수동으로 `results/pretrain_*/` → `results/pretrained/`로 복사 필요
- mapping.json 복사 누락 가능

**해결:**
- `--register` 플래그 추가
- `--pretrained_dir` 옵션으로 등록 경로 지정 가능
- 기존 파일 자동 백업 후 덮어쓰기

**사용 예:**
```bash
python scripts/01_pretrain_encoder.py --config config/default.yaml --gpu 0 --register
```

**수정 파일:**
- `scripts/01_pretrain_encoder.py`

---

### 3. K-means Codebook Stratified Sampling

**문제점:**
- `init_codebook()`에서 처음 N개 샘플만 사용
- Conditional 변수 분포가 불균형하면 일부 조건이 codebook에 반영 안 됨

**해결:**
- `VQ_AENB_Conditional.init_codebook()`에 `stratify` 파라미터 추가
- Conditional 모델에서 자동으로 stratified sampling 사용
- 각 conditional 그룹에서 균등하게 샘플링

**수정 파일:**
- `src/models/autoencoder.py`
- `src/training/trainer_ae.py`

---

### 4. 변수명 범용화 (하위 호환성 유지)

**문제점:**
- `study_emb_dim`, `n_studies` 등 변수명이 "study"를 암시
- 실제로는 study, organ 등 다양한 조건 변수에 사용됨

**해결:**

| 기존 | 신규 | 비고 |
|-----|-----|-----|
| `study_emb_dim` | `conditional_emb_dim` | Config, 모델, 스크립트 |
| `n_studies` | `n_conditionals` | 모델, 스크립트 |
| `study_embedding` | `conditional_embedding` | 모델 내부 |

- 하위 호환성: 기존 파라미터명도 계속 지원
- 기존 checkpoint 파일도 정상 로드 가능

**수정 파일:**
- `src/config.py`
- `src/models/autoencoder.py`
- `scripts/01_pretrain_encoder.py`
- `scripts/02_train_cv.py`
- `scripts/03_finalize_model.py`
- `scripts/04_cross_disease_eval.py`
- `scripts/05_cell_scoring.py`
- `scripts/06_tune_hyperparams.py`
- `config/default.yaml`
- `config/pretrain_epoch100_organ_ver00.yaml`

---

### 5. `tuning.enabled` 필드 제거

**문제점:**
- `tuning.enabled` 필드가 스크립트에서 사용되지 않음
- 튜닝 여부는 `06_tune_hyperparams.py` 실행 여부로 결정됨

**해결:**
- `TuningConfig.enabled` 필드 제거
- `config/*.yaml`에서 `tuning.enabled` 라인 제거

**수정 파일:**
- `src/config.py`
- `config/default.yaml`
- `config/pretrain_epoch100_organ_ver00.yaml`

---

### 6. Config 업데이트

**변경 사항:**
- `default.yaml`: `n_repeats: 3` → `n_repeats: 1` (기본값)
- `default.yaml`: `conditional_emb_dim: 16` 명시적 추가
- `pretrain_epoch100_organ_ver00.yaml`: `study_emb_dim: 1` → `conditional_emb_dim: 1`

---

## 수정된 파일 전체 목록

### 신규 파일
- `scripts/02_train_cv.py`

### 수정된 파일
- `scripts/01_pretrain_encoder.py`
- `scripts/02_train_loocv.py` (deprecated 경고)
- `scripts/03_finalize_model.py`
- `scripts/04_cross_disease_eval.py`
- `scripts/05_cell_scoring.py`
- `scripts/06_tune_hyperparams.py`
- `src/config.py`
- `src/models/autoencoder.py`
- `src/training/trainer_ae.py`
- `config/default.yaml`
- `config/pretrain_epoch100_organ_ver00.yaml`
- `CLAUDE.md`

---

## 테스트 가이드

### 1. LOOCV 테스트
```bash
python scripts/02_train_cv.py --config config/skin3.yaml --gpu 0
```

### 2. Pretrain 자동 등록 테스트
```bash
python scripts/01_pretrain_encoder.py --config config/default.yaml --gpu 0 --register
```

### 3. Stratified K-Fold 테스트
```yaml
# config/skin3.yaml에서 수정
splitting:
  strategy: "stratified_kfold"
  n_splits: 5
```
```bash
python scripts/02_train_cv.py --config config/skin3.yaml --gpu 0
```

### 4. Repeated K-Fold 테스트
```yaml
# config/skin3.yaml에서 수정
splitting:
  strategy: "repeated_stratified_kfold"
  n_splits: 5
  n_repeats: 3
```
```bash
python scripts/02_train_cv.py --config config/skin3.yaml --gpu 0
```

### 5. Hyperparameter Tuning with K-Fold
```yaml
# config/skin3.yaml에서 수정
splitting:
  strategy: "stratified_kfold"
  n_splits: 5
```
```bash
python scripts/06_tune_hyperparams.py --config config/skin3.yaml --gpu 0 --verbose
```

---

### 7. `06_tune_hyperparams.py` K-Fold 지원 추가

**문제점:**
- Hyperparameter tuning 스크립트가 LOOCV만 지원
- `LOOCVSplitter` 하드코딩되어 있음
- `02_train_cv.py`와 달리 K-Fold 사용 불가

**해결:**
- `LOOCVSplitter` → `create_splitter()` 변경
- `run_loocv_for_hyperparams()` → `run_cv_for_hyperparams()` 함수명 변경
- `config.splitting.strategy`에 따라 splitter 동적 생성
- `skip_fold_metrics` 동적 설정:
  - LOOCV: `True` (단일 샘플이므로 fold별 메트릭 무의미)
  - K-Fold: `False` (fold별 메트릭 계산)
- 메트릭 계산 분리:
  - LOOCV: 전체 prediction concatenate 후 메트릭 계산
  - K-Fold: fold별 메트릭의 평균/표준편차 계산
- 출력 디렉토리명: `tuning_{strategy}_{timestamp}`

**사용 예:**
```yaml
# config/skin3.yaml
splitting:
  strategy: "stratified_kfold"  # or "loocv", "repeated_stratified_kfold"
  n_splits: 5
  n_repeats: 3  # repeated_stratified_kfold인 경우에만 사용
```
```bash
python scripts/06_tune_hyperparams.py --config config/skin3.yaml --gpu 0
```

**수정 파일:**
- `scripts/06_tune_hyperparams.py`

---

## 향후 작업 (TODO)

- [ ] `05_cell_scoring.py`: `--loocv_dir` 옵션으로 LOOCV 모드 지원
- [ ] Pretrain: `num_codes` 증가 시 NaN 방지 개선
- [ ] MLflow/W&B 통합
- [ ] Optuna 기반 최적화

---

## 참고사항

### 하위 호환성
- 기존 config 파일 (`study_emb_dim` 사용)도 정상 동작
- 기존 checkpoint 파일도 정상 로드 가능
- `02_train_loocv.py`는 deprecated 경고와 함께 계속 동작

### 주요 변경된 CLI 인자
- `--study_emb_dim` → `--conditional_emb_dim` (deprecated alias 유지)
- `--register` 플래그 추가 (01_pretrain_encoder.py)
