# CLAUDE.md - scMILD Project Context

이 파일은 Claude가 scMILD 프로젝트를 이해하고 효과적으로 도울 수 있도록 하는 컨텍스트 파일입니다.

## 프로젝트 개요

**scMILD** (Single-cell Multiple Instance Learning for Disease classification)는 단일세포 RNA-seq 데이터를 활용하여 환자 수준의 질병을 분류하는 딥러닝 프레임워크입니다.

### 핵심 문제
- 단일세포 데이터는 환자(샘플)당 수천~수만 개의 세포를 포함
- 세포 수준 레이블은 없고, 환자 수준 레이블(질병/정상)만 존재
- → **Multiple Instance Learning (MIL)** 접근법 사용

### 주요 질병 타겟
- **HS (Hidradenitis Suppurativa)**: 피부 질환, Skin3 데이터셋
- **CD (Crohn's Disease)**: 장 질환, SCP1884 데이터셋

## 아키텍처

```
[Raw scRNA-seq] → [VQ-AENB-Conditional Encoder] → [Latent Space]
                                                        ↓
                                              [Gated Attention MIL]
                                                   ↓         ↓
                                            [Teacher]   [Student]
                                            (Bag-level) (Instance-level)
```

### 1. VQ-AENB-Conditional (Pretrained Encoder)
- **VQ**: Vector Quantization - 이산적 잠재 공간
- **AENB**: Autoencoder with Negative Binomial loss - scRNA-seq 카운트 데이터에 적합
- **Conditional**: Study/Organ 조건부 임베딩 - 배치 효과 보정

### 2. Teacher-Student MIL
- **Teacher Branch**: Bag(환자) 수준 분류, Gated Attention으로 중요 세포 가중치 학습
- **Student Branch**: Instance(세포) 수준 분류, Teacher의 attention을 pseudo-label로 활용

### 3. Disease Ratio Regularization (선택적)
- VQ 코드별 질병 비율을 계산하여 attention score를 유도
- 생물학적으로 의미 있는 세포 선별 유도

## 디렉토리 구조

```
scMILD/
├── src/
│   ├── config.py              # 설정 시스템 (dataclass + YAML)
│   ├── models/
│   │   ├── attention.py       # GatedAttentionModule
│   │   ├── branches.py        # TeacherBranch, StudentBranch
│   │   ├── autoencoder.py     # VQ_AENB_Conditional
│   │   ├── encoder_wrapper.py # VQEncoderWrapperConditional
│   │   └── quantizer.py       # VectorQuantizer
│   ├── training/
│   │   ├── trainer.py         # MILTrainer
│   │   ├── trainer_ae.py      # AETrainer (pretrain용)
│   │   ├── metrics.py         # 평가 메트릭
│   │   └── disease_ratio.py   # Disease ratio regularization
│   └── data/
│       ├── dataset.py         # MilDataset, InstanceDataset
│       ├── splitter.py        # LOOCVSplitter, StratifiedKFoldSplitter
│       └── preprocessing.py   # load_adata_with_subset 등
├── scripts/
│   ├── 01_pretrain_encoder.py    # Encoder pretrain (전체 데이터)
│   ├── 02_train_loocv.py         # LOOCV 학습 (기본 파라미터)
│   ├── 03_finalize_model.py      # Final model 학습 (best params)
│   ├── 04_cross_disease_eval.py  # Cross-disease 평가
│   ├── 05_cell_scoring.py        # Cell-level scoring
│   └── 06_tune_hyperparams.py    # Grid Search 튜닝
├── notebooks/
│   ├── 01_vq_embedding_analysis.ipynb   # VQ code/embedding 추출
│   └── 02_codebook_visualization.ipynb  # Codebook 시각화
└── config/
    ├── default.yaml           # 기본 설정
    ├── skin3.yaml             # Skin3 (HS) 설정
    └── scp1884.yaml           # SCP1884 (CD) 설정
```

## 파이프라인 흐름

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
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Cell Scoring 로직 (중요!)

| 데이터 유형 | 사용 모델 | 이유 |
|-------------|----------|------|
| **학습 데이터** (예: Skin3) | LOOCV fold 모델들 | 각 샘플은 해당 샘플 제외 모델로 scoring (data leakage 방지) |
| **평가 데이터** (예: SCP1884) | Final model | 학습에 포함되지 않았으므로 final model 사용 |

## 핵심 실행 흐름

```bash
# 1. Encoder Pretrain (처음 한 번, 전체 데이터)
python scripts/01_pretrain_encoder.py --config config/default.yaml --gpu 0

# 1.1 Pretrain 결과물 배치 (수동)
cp results/pretrain_*/vq_aenb_conditional.pth results/pretrained/vq_aenb_conditional_whole.pth
cp results/pretrain_*/study_mapping.json results/pretrained/study_mapping.json

# 2. LOOCV 학습 (baseline)
python scripts/02_train_loocv.py --config config/skin3.yaml --gpu 0

# 3. 하이퍼파라미터 튜닝
python scripts/06_tune_hyperparams.py --config config/skin3.yaml --gpu 0 --verbose
# → results/skin3/tuning_*/best_params.yaml 생성

# 4. Final model 학습
python scripts/03_finalize_model.py \
    --config config/skin3.yaml \
    --best_params results/skin3/tuning_*/best_params.yaml \
    --gpu 0

# 5a. Cell scoring (학습 데이터 - LOOCV 모델 사용)
python scripts/05_cell_scoring.py \
    --loocv_dir results/skin3/loocv_* \
    --config config/skin3.yaml --gpu 0

# 5b. Cell scoring (평가 데이터 - Final model 사용)
python scripts/05_cell_scoring.py \
    --model_dir results/skin3/final_model_* \
    --config config/scp1884.yaml --gpu 0

# 6. Cross-disease 평가
python scripts/04_cross_disease_eval.py \
    --model_dir results/skin3/final_model_* \
    --test_config config/scp1884.yaml --gpu 0
```

## 데이터 구조

### AnnData (.h5ad) 형식
```python
adata.X          # 유전자 발현 매트릭스 (cells × genes)
adata.obs        # 세포 메타데이터
  - sample_id_numeric   # 샘플 ID (정수)
  - disease_numeric     # 질병 레이블 (0: 정상, 1: 질병)
  - study_id_numeric    # Study ID (조건부 임베딩용)
  - sample              # 샘플 이름 (문자열)
  - Status              # 원본 레이블 (HS, ctrl_skin 등)
```

### 데이터셋 정보
| Dataset | Disease | Studies | Samples | Cells |
|---------|---------|---------|---------|-------|
| Skin3 | HS | GSE175990, GSE220116 | 19 (8 HS, 11 ctrl) | ~36k |
| SCP1884 | CD | SCP1884 | 34 (18 CD, 16 ctrl) | ~290k |

### Subset 로직
- **Pretrain**: 전체 데이터 (`Whole_SCP_PCD_Skin_805k_6k.h5ad`)
  - `study_id_numeric` 컬럼 생성 및 `study_mapping.json` 저장
- **MIL 학습**: Study별 subset (config의 `data.subset.values`)
  - `study_mapping.json`을 사용하여 pretrain과 동일한 study ID 매핑 적용
  - **중요**: subset 캐시에는 `study_id_numeric`이 없으므로 런타임에 매핑 적용

## Config 시스템

### 상속 구조
```yaml
# skin3.yaml
_base_: "default.yaml"  # default.yaml 설정 상속

paths:
  output_root: "${paths.data_root}/results/skin3"  # 변수 참조

data:
  subset:
    enabled: true
    values: ["GSE175990", "GSE220116"]  # 이 study만 사용
```

### 주요 설정 경로
- `config.paths.pretrained_encoder`: Pretrained encoder 경로
- `config.data.subset`: 데이터 subset 설정
- `config.data.conditional_embedding.mapping_path`: Study ID 매핑 파일 경로 (**중요**)
- `config.mil.training`: 학습 하이퍼파라미터
- `config.tuning`: Grid search 설정

## 자주 사용하는 패턴

### Config 로드
```python
from src.config import load_config
config = load_config("config/skin3.yaml")
```

### 데이터 로드 (Subset 적용)
```python
from src.data import load_adata_with_subset
adata = load_adata_with_subset(
    whole_adata_path=config.data.whole_adata_path,
    subset_enabled=config.data.subset.enabled,
    subset_column=config.data.subset.column,
    subset_values=config.data.subset.values,
)
```

### 모델 생성
```python
from src.models import VQEncoderWrapperConditional, TeacherBranch, StudentBranch
# Encoder wrapper가 pretrained encoder를 로드하고 projection layer 추가
```

## 개발 시 주의사항

### 1. 데이터 분리
- Pretrained encoder: **전체 데이터**로 학습 (unsupervised)
- MIL 모델: **Subset 데이터**로 학습 (study별)

### 2. LOOCV 사용 시
- Early stopping **비활성화** 권장 (`use_early_stopping: false`)
- 샘플 수가 적어서 validation set 분리가 어려움
- **AUROC 계산**: 각 fold의 예측을 concatenate하여 전체 AUROC 계산
  - fold별 AUC는 테스트 샘플이 1개라 의미 없음
  - `overall_results.csv`에 전체 메트릭 저장

### 3. Cell Scoring 모델 선택
- **학습 데이터**: LOOCV fold 모델 사용 (data leakage 방지)
- **평가 데이터**: Final model 사용

### 4. 병원망 환경
- 외부 패키지 설치 제한
- conda 환경 경로: `config.paths.conda_env`

### 5. GPU 메모리
- SCP1884는 세포 수가 많아 subsampling 필요할 수 있음
- `config.mil.subsampling.max_cells_per_sample`

## 트러블슈팅

### Import 에러
```python
# 프로젝트 루트를 path에 추가
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### CUDA Out of Memory
- `config.mil.subsampling.enabled: true`
- `config.mil.training.batch_size` 감소

### 캐시 문제
- Subset 캐시 위치: `config.data.subset.cache_dir`
- 캐시 삭제 후 재실행하면 새로 생성
- **주의**: Subset 캐시에는 `study_id_numeric`이 없음 → `study_mapping.json` 필요

### study_ids 전달 관련
- Conditional encoder는 `study_ids`를 입력으로 받아 배치 효과 보정
- `study_mapping.json`이 없으면 경고 발생: `WARNING: study_ids not provided to conditional encoder`
- 해결: pretrain 후 `study_mapping.json`을 `results/pretrained/`에 복사

## 코드 수정 시 체크리스트

1. [ ] `src/` 모듈 수정 시 `__init__.py`에 export 추가
2. [ ] 새 config 옵션 추가 시 `src/config.py`의 dataclass도 수정
3. [ ] 새 스크립트 추가 시 `scripts/` 디렉토리에 번호 체계 유지
4. [ ] 모델 구조 변경 시 pretrained encoder와 호환성 확인
5. [ ] Pretrain 후 `study_mapping.json` 복사 확인
6. [ ] Subset 데이터 사용 시 `conditional_embedding.mapping_path` 설정 확인

## 분석 노트북

### notebooks/01_vq_embedding_analysis.ipynb
- 전체 adata에 VQ code 및 embedding 추출
- `adata.obs['vq_code']`: 각 세포의 codebook index
- `adata.obsm['X_vq']`: 각 세포의 quantized embedding
- Codebook statistics (disease ratio, organ ratio 등)

### notebooks/02_codebook_visualization.ipynb
- Codebook PCA/UMAP + dendrogram
- Codebook leiden clustering → cell-level 적용
- Code cluster vs Cell cluster 비교 (ARI, NMI)
- Special codes 식별 (cross-tissue, disease-specific 등)

## 변경 이력

### 2026-01-30 (3차)
- **`03_finalize_model.py` 개선**
  - `--best_params` 옵션 추가: `best_params.yaml` 로드하여 config 오버라이드
  - `load_best_params()`, `apply_best_params()` 함수 추가
  - `disease_ratio_lambda` 지원: lambda > 0일 때 disease ratio 계산 및 적용
  - `model_info.json`에 사용된 하이퍼파라미터 기록

- **전체 스크립트 데이터 로딩 통일**
  - `03_finalize_model.py`, `04_cross_disease_eval.py`, `05_cell_scoring.py` 수정
  - `load_adata` → `load_adata_with_subset` 변경
  - `config.data.adata_path` → `config.data.whole_adata_path` + subset 시스템 사용
  - 모든 스크립트에서 동일한 study mapping 로직 적용

- **지원 하이퍼파라미터 (best_params.yaml)**
  - `learning_rate`, `encoder_learning_rate`, `epochs`
  - `attention_dim`, `latent_dim`, `projection_dim`
  - `negative_weight`, `student_optimize_period`
  - `disease_ratio_lambda` → `config.mil.loss.disease_ratio_reg.lambda_weight`

### 2026-01-30 (2차)
- **문서 정비**
  - README.md, CLAUDE.md 실제 스크립트 구조와 동기화
  - 파이프라인 흐름 명확화 (튜닝 → finalize → cell scoring/cross-disease)
  - Cell scoring 로직 설명 추가 (LOOCV vs Final model)

- **노트북 추가**
  - `notebooks/01_vq_embedding_analysis.ipynb`: VQ code/embedding 추출
  - `notebooks/02_codebook_visualization.ipynb`: Codebook 시각화

- **버그 수정**
  - `trainer.py`: attention score squeeze 차원 오류 수정 (`squeeze(-1)` → `squeeze(0)`)

### 2026-01-30 (1차)
- **study_ids 전달 문제 수정**
  - Pretrain 시 `study_mapping.json` 자동 생성
  - MIL 학습 시 매핑 파일 로드하여 `study_id_numeric` 컬럼 생성
  - `config.data.conditional_embedding.mapping_path` 설정 추가

- **LOOCV AUROC 계산 방식 개선**
  - fold별 AUC 평균 → 전체 예측 concatenate 후 AUROC 계산
  - `FoldResult`에 `y_true`, `y_pred_proba` 필드 추가
  - `overall_results.csv`, `predictions.csv` 출력 추가

- **LOOCV fold별 메트릭 계산 제거**
  - `train_fold()`에 `skip_fold_metrics` 파라미터 추가
  - LOOCV에서 불필요한 `_evaluate()`, `compute_metrics()` 호출 건너뜀

- **하이퍼파라미터 튜닝 개선** (`06_tune_hyperparams.py`)
  - 실시간 CSV 저장
  - Top-K 모델 저장
  - `best_params.yaml` 저장

- **기본값 수정**
  - `student.optimize_period`: 3 → 1
  - `tuning.epochs`: [100, 50] → [10, 30]

## TODO (미구현)

- [x] `03_finalize_model.py`: `--best_params` 옵션으로 best_params.yaml 자동 로드 (2026-01-30 3차 완료)
- [ ] `05_cell_scoring.py`: `--loocv_dir` 옵션으로 LOOCV 모드 지원
