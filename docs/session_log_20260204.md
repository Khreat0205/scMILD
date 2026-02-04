# Session Log - 2026-02-04

## 목표
- `05_cell_scoring.py` 확장: CV/Tuning 모드 지원, h5ad 출력, codebook adata 생성

## 주요 변경사항

### 1. `05_cell_scoring.py` 대폭 확장

#### 3가지 실행 모드
- `--model_dir`: Final model 모드 (단일 모델로 전체 데이터 scoring)
- `--cv_dir`: CV 모드 (각 fold 모델로 해당 test set만 scoring)
- `--tuning_dir`: Tuning 모드 (best params의 fold 모델 사용)

#### Cell-level 출력 (`scored_adata.h5ad`)
```
adata.obs:
  - attention_score_raw      # 원본 attention score
  - attention_score_global   # 전체 기준 min-max 정규화
  - attention_score_sample   # sample 내 min-max 정규화
  - student_prediction       # Student branch 예측 확률
  - vq_code                  # Codebook index (int)
  - fold                     # [CV 모드] test였던 fold 번호

adata.obsm:
  - X_pretrained             # Projection 이전 embedding
  - X_scmild                 # Projection 이후 embedding

adata.uns:
  - codebook                 # Codebook matrix
  - model_info               # 모델 메타정보
```

#### Codebook-level 출력 (`codebook_adata.h5ad`)
```
adata_codebook.X = codebook  # (num_codes, latent_dim)

adata_codebook.obs:
  # 기본 통계
  - code_idx, n_cells, n_samples, disease_ratio

  # Codebook 직접 통과 attention (방식 1)
  - attn_direct              # codebook → projection → attention

  # Cell 기반 attention 통계 (방식 2)
  - attn_cell_mean, attn_cell_std, attn_cell_median, attn_cell_max, attn_cell_n

  # [CV 모드] fold별 통계
  - attn_direct_fold{N}
  - attn_cell_mean_fold{N}, attn_cell_std_fold{N}
```

### 2. Attention 정규화 방식 변경

기존: `attention_score_norm` (sample 내 정규화만 제공)

변경:
- `attention_score_global`: 전체 cell 기준 min-max 정규화 (code 간 비교 가능)
- `attention_score_sample`: sample 내 min-max 정규화 (sample 내 상대적 중요도)

**이유**: 같은 VQ code를 가진 cell들이 동일한 codebook embedding을 사용하므로, 전체 기준 정규화 시 같은 code의 cell들은 동일한 attention score를 가져야 함. Sample별 정규화는 이 특성을 깨뜨림.

### 3. Tuning 모드 구현

Tuning 디렉토리 구조:
```
tuning_xxx/
├── tuning_results.csv      # config별 성능
├── best_params.yaml
└── models/
    └── config_XXX/         # top-k configs
        ├── params.yaml
        └── fold_YY/
            ├── encoder.pth
            ├── teacher.pth
            ├── student.pth
            └── info.yaml
```

Tuning 모드 로직:
1. `tuning_results.csv`에서 best config 찾기
2. Splitter를 재생성하여 sample → fold 매핑 복원
3. 각 fold 모델로 해당 test sample scoring

## 사용법

```bash
# Final model 모드
python scripts/05_cell_scoring.py \
    --model_dir results/final_model_xxx \
    --config config/skin3.yaml \
    --gpu 0

# CV 모드
python scripts/05_cell_scoring.py \
    --cv_dir results/cv_stratified_kfold_xxx \
    --config config/scp1884.yaml \
    --output_dir results/cell_scores_cv \
    --gpu 0

# Tuning 모드
python scripts/05_cell_scoring.py \
    --tuning_dir results/tuning_xxx \
    --config config/scp1884.yaml \
    --output_dir results/cell_scores_tuning \
    --gpu 0

# Cross-disease (Final model로 다른 데이터셋 scoring)
python scripts/05_cell_scoring.py \
    --model_dir results/scp1884/final_model_xxx \
    --config config/skin3.yaml \
    --output_dir results/cross_eval_skin3_cell_scores \
    --gpu 0
```

## 버그 수정

1. `create_codebook_adata`에서 `label_col` 인자 대신 고정 컬럼명 `disease_label` 사용
2. `add_scores_to_adata`에서 컬럼명 업데이트 (`attention_score_norm` → `attention_score_global`/`attention_score_sample`)
3. `main()` summary 부분 컬럼명 업데이트

## 관련 파일
- `scripts/05_cell_scoring.py`: 주요 변경
- `CLAUDE.md`: 문서 업데이트
