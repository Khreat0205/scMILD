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

- **VQ-AENB-Conditional**: Vector Quantization + Negative Binomial loss + Conditional embedding (배치 효과 보정)
- **Teacher Branch**: Bag(환자) 수준 분류, Gated Attention으로 중요 세포 가중치 학습
- **Student Branch**: Instance(세포) 수준 분류, Teacher의 attention을 pseudo-label로 활용

## 디렉토리 구조

```
scMILD/
├── src/
│   ├── config.py              # 설정 시스템 (dataclass + YAML)
│   ├── models/                # GatedAttention, TeacherBranch, StudentBranch, VQ_AENB_Conditional
│   ├── training/              # MILTrainer, AETrainer, metrics
│   └── data/                  # MilDataset, LOOCVSplitter, preprocessing
├── scripts/
│   ├── 01_pretrain_encoder.py    # Encoder pretrain
│   ├── 02_train_loocv.py         # LOOCV 학습
│   ├── 03_finalize_model.py      # Final model 학습
│   ├── 04_cross_disease_eval.py  # Cross-disease 평가
│   ├── 05_cell_scoring.py        # Cell-level scoring
│   └── 06_tune_hyperparams.py    # Grid Search 튜닝
└── config/
    ├── default.yaml              # Study 기반 기본 설정
    ├── skin3.yaml, scp1884.yaml  # Study 기반 subset 설정
    └── *_organ_*.yaml            # Organ 기반 설정
```

## 파이프라인 흐름

```
1. Pretrain Encoder (전체 데이터, unsupervised)
       ↓
2. LOOCV 학습 (기본 파라미터로 baseline)
       ↓
3. 하이퍼파라미터 튜닝 (LOOCV 기반 grid search)
       ↓ best_params.yaml
4. Finalize Model (best params로 전체 subset 학습)
       ↓
5a. Cell Scoring / 5b. Cross-disease 평가
```

## 핵심 실행 흐름

```bash
# 1. Pretrain (Study 또는 Organ 기반)
python scripts/01_pretrain_encoder.py --config config/default.yaml --gpu 0

# 1.1 결과물 배치 (수동)
cp results/pretrain_*/vq_aenb_conditional.pth results/pretrained/
cp results/pretrain_*/{study,organ}_mapping.json results/pretrained/

# 2-4. 학습 파이프라인
python scripts/02_train_loocv.py --config config/skin3.yaml --gpu 0
python scripts/06_tune_hyperparams.py --config config/skin3.yaml --gpu 0
python scripts/03_finalize_model.py --config config/skin3.yaml --best_params results/skin3/tuning_*/best_params.yaml --gpu 0

# 5. 평가
python scripts/04_cross_disease_eval.py --model_dir results/skin3/final_model_* --test_config config/scp1884.yaml --gpu 0
```

## Conditional Embedding 설정

| 설정 | Study 기반 | Organ 기반 |
|------|-----------|-----------|
| `conditional_embedding.column` | `study` | `Organ` |
| `conditional_embedding.encoded_column` | `study_id_numeric` | `Organ_id_numeric` |
| `conditional_embedding.mapping_path` | `study_mapping.json` | `organ_mapping.json` |

## 주요 Config 경로

- `config.paths.pretrained_encoder`: Pretrained encoder 경로
- `config.data.conditional_embedding`: Conditional 임베딩 설정 (column, encoded_column, mapping_path)
- `config.encoder`: latent_dim, num_codes, conditional_emb_dim, pretrain.*
- `config.mil.training`: learning_rate, encoder_learning_rate, epochs
- `config.tuning`: Grid search 설정
- `config.splitting`: strategy (loocv, stratified_kfold, repeated_stratified_kfold), n_splits, n_repeats

## 개발 시 주의사항

1. **Pretrain 후 `--register` 플래그 사용**: 자동으로 pretrained 디렉토리에 등록
2. **LOOCV/K-Fold 평가**: `02_train_cv.py` 사용, strategy에 따라 메트릭 계산 방식이 다름
3. **Config 수정 시**: `src/config.py`의 dataclass와 `_dict_to_config()` 함수 모두 수정
4. **스크립트에서 "study" 하드코딩 금지**: `config.data.conditional_embedding.column` 사용
5. **GPU 메모리**: CV fold 간 `gc.collect()`, `torch.cuda.empty_cache()` 자동 호출
6. **변수명 범용화**: `study_emb_dim` → `conditional_emb_dim`, `n_studies` → `n_conditionals` (하위 호환성 유지)

## 트러블슈팅

- **Import 에러**: `sys.path.insert(0, str(Path(__file__).parent.parent))`
- **CUDA OOM**: `config.mil.subsampling.enabled: true` 또는 batch_size 감소
- **Mapping 경고**: pretrain 후 생성된 `{column}_mapping.json`을 config 경로에 복사 (또는 `--register` 사용)
- **Pretrain NaN 발생** (`num_codes` 증가 시):
  - 원인: Dead code revival 로직이 대량의 코드를 동시에 재초기화 → codebook 불안정
  - 해결: `encoder.pretrain.batch_size` 증가 (512→2048 권장) 또는 dead code revival 비활성화
  - 해결: Stratified sampling 사용 (자동 활성화됨)

## 변경 이력

상세 변경 이력은 `docs/session_log_*.md` 참조

### 최근 주요 변경 (2026-02-04)
- `05_cell_scoring.py` 대폭 확장:
  - 3가지 모드 지원: `--model_dir` (final), `--cv_dir` (CV), `--tuning_dir` (tuning)
  - CV 모드: 각 fold 모델로 해당 test sample만 scoring, fold 정보 포함
  - h5ad 출력 지원 (`scored_adata.h5ad`, `codebook_adata.h5ad`)
  - Cell-level: `vq_code`, `X_pretrained`, `X_scmild` 추가
  - Codebook-level: `attn_direct` (codebook 직접 통과), `attn_cell_*` (cell 기반 통계)
  - CV 모드 fold별 attention 통계 지원

### 이전 변경 (2026-02-03)
- `02_train_loocv.py` → `02_train_cv.py` 일반화: LOOCV, Stratified K-Fold, Repeated K-Fold 지원
- Pretrain 자동 배치: `--register` 플래그 추가
- K-means Codebook 초기화: Stratified sampling 지원 (conditional 변수 기준)
- 변수명 범용화: `study_*` → `conditional_*` (하위 호환성 유지)
- `tuning.enabled` 필드 제거 (불필요)

### 이전 변경 (2026-02-02)
- Gradient clipping 추가 (`trainer_ae.py`): `max_norm=1.0`
- NaN 발생 원인 분석: `num_codes` 증가 시 dead code revival 로직 불안정
- 해결 방안 정리: batch size 증가, dead code revival 완화 등

### 이전 변경 (2026-01-30)
- Conditional Embedding 일반화: Study/Organ 등 임의의 categorical 컬럼 지원
- Pretrain 스크립트: CLI 기본값 버그 수정 (config 값 우선 적용)
- 메모리 관리: LOOCV fold 간 GPU 메모리 정리 추가
- Config: `EncoderConfig.study_emb_dim` 필드 추가

## TODO
- [x] `05_cell_scoring.py`: CV/Tuning 모드 지원 (완료)
- [ ] Pretrain: `num_codes` 증가 시 NaN 방지 (batch size 증가 또는 dead code revival 개선)
