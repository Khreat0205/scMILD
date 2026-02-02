# Session Log - 2026-02-02

## 주요 이슈: VQ-AENB Pretrain 시 NaN 발생

### 문제 상황

`num_codes` 값에 따라 pretrain 중 NaN loss 발생:

| Config | num_codes | 결과 |
|--------|-----------|------|
| ver01 | 64 | 정상 (epoch 100 완료) |
| ver02 | 256 | NaN (epoch 25에서 발생) |

두 config의 유일한 차이는 `num_codes` 값.

### 원인 분석

#### 1. Dead Code Revival 로직 (`quantizer.py:151-190`)

```python
dead_codes = self.code_usage < (1e-3 / self.num_codes)
```

- **Threshold 계산**:
  - num_codes=64: `1e-3 / 64 = 1.56e-5`
  - num_codes=256: `1e-3 / 256 = 3.9e-6`

- **문제점**:
  1. num_codes가 크면 threshold가 매우 낮아짐
  2. 학습 초기에 대부분의 코드가 "dead"로 판정됨
  3. 매 batch마다 수십~수백 개 코드가 동시에 재초기화됨
  4. Codebook이 급격히 변함 → Encoder 출력과 mismatch → Loss 폭발 → NaN

#### 2. `code_usage` EMA 업데이트 메커니즘

```python
avg_probs = torch.mean(one_hot, dim=0)  # 이번 batch에서 각 코드 사용 비율
self.code_usage.mul_(self.decay).add_(avg_probs, alpha=1 - self.decay)  # EMA 업데이트
```

- 균등 사용 시 기대값: `1 / num_codes`
- Threshold는 기대값의 0.1%로 설정됨
- 그러나 학습 초기에는 EMA가 수렴하지 않아 대부분 코드가 threshold 미만

#### 3. Batch Size와의 관계

| Batch Size | 한 batch에서 사용되는 코드 수 | Dead code 판정 |
|------------|------------------------------|----------------|
| 512 (현재) | ~10-50개 (추정) | 나머지 200+개가 매번 "dead"로 판정 |
| 2048 | ~40-150개 (추정) | 더 많은 코드가 "alive" |
| 8192 | ~100-200개 (추정) | 대부분 코드가 커버됨 |

**→ Batch size를 늘리면 dead code revival 빈도가 줄어들어 안정화될 수 있음**

### 적용된 수정

#### Gradient Clipping 추가 (`src/training/trainer_ae.py`)

```python
# Backward
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 추가됨
optimizer.step()
```

**결과**: Gradient clipping만으로는 NaN 해결 안 됨 (ver02에서 여전히 epoch 25에서 NaN)

### 추가 해결 방안 (미적용, 추후 검토)

#### 방안 1: Batch Size 증가 (권장)

```yaml
encoder:
  pretrain:
    batch_size: 2048  # 512 → 2048
```

- 장점: 근본적 해결, 코드 수정 불필요
- 단점: GPU 메모리 증가

#### 방안 2: Dead Code Revival 비활성화 또는 완화

```python
# 옵션 A: 완전 비활성화
def _revive_dead_codes(self, z, similarity):
    return

# 옵션 B: 고정 threshold
dead_codes = self.code_usage < 1e-5  # num_codes와 무관

# 옵션 C: Revival 빈도 제한
if batch_idx % 100 == 0:
    self._revive_dead_codes(...)
```

#### 방안 3: Gradient Clipping 강화

```python
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # 1.0 → 0.5
```

#### 방안 4: Learning Rate 감소

```yaml
encoder:
  pretrain:
    learning_rate: 0.0005  # 0.001 → 0.0005
```

### K-means 초기화 관련 논의

#### 현재 구현

- `num_samples = num_codes * 40`으로 샘플링
- DataLoader 순서대로 앞에서부터 가져옴 → **study/organ 비율이 불균형할 수 있음**

#### 균등 샘플링 개선안 (미적용, 추후 검토)

Conditional column 기준으로 stratified sampling:

```python
def init_codebook(
    self,
    dataloader,
    method: str = "kmeans",
    num_samples: int = None,
    stratify_column_values: np.ndarray = None,
    stratify: bool = False
):
    if stratify and stratify_column_values is not None:
        unique_ids = np.unique(stratify_column_values)
        samples_per_group = num_samples // len(unique_ids)
        # 각 그룹에서 균등하게 샘플링
        ...
```

Config 확장:
```yaml
encoder:
  pretrain:
    codebook_init:
      method: "kmeans"
      stratify: true  # conditional column 기준 균등 샘플링
      num_samples: null
```

---

## 코드 변경 요약

| 파일 | 변경 내용 |
|------|----------|
| `src/training/trainer_ae.py` | Gradient clipping 추가 (`max_norm=1.0`) |

---

## TODO

- [ ] Batch size 증가 테스트 (512 → 2048)
- [ ] Dead code revival 로직 개선 검토
- [ ] K-means 초기화 시 stratified sampling 구현
