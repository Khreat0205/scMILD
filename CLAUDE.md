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
- `config.encoder`: latent_dim, num_codes, study_emb_dim, pretrain.*
- `config.mil.training`: learning_rate, encoder_learning_rate, epochs
- `config.tuning`: Grid search 설정

## 개발 시 주의사항

1. **Pretrain 후 매핑 파일 복사 필수**: `{column}_mapping.json`이 없으면 MIL 학습 시 경고 발생
2. **LOOCV 평가**: fold별 AUC는 무의미, `overall_results.csv`의 전체 AUROC 사용
3. **Config 수정 시**: `src/config.py`의 dataclass와 `_dict_to_config()` 함수 모두 수정
4. **스크립트에서 "study" 하드코딩 금지**: `config.data.conditional_embedding.column` 사용
5. **GPU 메모리**: LOOCV fold 간 `gc.collect()`, `torch.cuda.empty_cache()` 자동 호출

## 트러블슈팅

- **Import 에러**: `sys.path.insert(0, str(Path(__file__).parent.parent))`
- **CUDA OOM**: `config.mil.subsampling.enabled: true` 또는 batch_size 감소
- **Mapping 경고**: pretrain 후 생성된 `{column}_mapping.json`을 config 경로에 복사

## 변경 이력

상세 변경 이력은 `docs/session_log_*.md` 참조

### 최근 주요 변경 (2026-01-30)
- Conditional Embedding 일반화: Study/Organ 등 임의의 categorical 컬럼 지원
- Pretrain 스크립트: CLI 기본값 버그 수정 (config 값 우선 적용)
- 메모리 관리: LOOCV fold 간 GPU 메모리 정리 추가
- Config: `EncoderConfig.study_emb_dim` 필드 추가

## TODO
- [ ] `05_cell_scoring.py`: `--loocv_dir` 옵션으로 LOOCV 모드 지원
