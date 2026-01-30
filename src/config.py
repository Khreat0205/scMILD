"""
Configuration management for scMILD.

YAML 설정 파일을 로드하고 Python dataclass로 변환합니다.
"""

import os
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Literal


# ============================================================================
# Dataclasses for type-safe configuration
# ============================================================================

@dataclass
class PathsConfig:
    """경로 설정"""
    data_root: str = "/home/bmi-user/workspace/data/HSvsCD"
    pretrained_encoder: Optional[str] = None
    output_root: str = "./results"
    conda_env: str = "/home/bmi-user/workspace/data/scvi-env"


@dataclass
class ColumnsConfig:
    """데이터 컬럼명 설정"""
    sample_id: str = "sample_id_numeric"
    sample_name: str = "sample"
    disease_label: str = "disease_numeric"
    status: str = "Status"


@dataclass
class ConditionalEmbeddingConfig:
    """Conditional encoder용 임베딩 컬럼 설정"""
    column: str = "study"                      # 원본 컬럼명 (study, Organ 등)
    encoded_column: str = "study_id_numeric"   # 인코딩된 컬럼명
    mapping_path: Optional[str] = None         # Pretrain 시 생성된 study_name -> study_id 매핑 파일


@dataclass
class PreprocessingConfig:
    """전처리 설정"""
    n_top_genes: int = 6000
    normalize_total: int = 10000
    log_transform: bool = True


@dataclass
class SubsetConfig:
    """데이터 subset 설정 (whole_adata에서 특정 study만 필터링)"""
    enabled: bool = False
    column: str = "study"           # subset 기준 컬럼
    values: List[str] = field(default_factory=list)  # 포함할 값들
    cache_dir: Optional[str] = None
    use_cache: bool = True


@dataclass
class DataConfig:
    """데이터 설정"""
    whole_adata_path: Optional[str] = None  # 전체 데이터 경로
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    conditional_embedding: ConditionalEmbeddingConfig = field(default_factory=ConditionalEmbeddingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


@dataclass
class SplittingConfig:
    """데이터 분할 설정"""
    strategy: Literal["loocv", "stratified_kfold", "repeated_stratified_kfold"] = "loocv"
    n_splits: int = 5
    n_repeats: int = 3
    random_seed: int = 42


@dataclass
class EncoderPretrainConfig:
    """Encoder pretrain 설정"""
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs: int = 50
    patience: int = 5


@dataclass
class EncoderConfig:
    """Encoder 설정"""
    type: str = "VQ_AENB_Conditional"
    latent_dim: int = 128
    num_codes: int = 1024
    study_emb_dim: int = 16  # Conditional embedding dimension
    pretrain: EncoderPretrainConfig = field(default_factory=EncoderPretrainConfig)


@dataclass
class MILTrainingConfig:
    """MIL 학습 설정"""
    batch_size: int = 2
    learning_rate: float = 0.0001
    encoder_learning_rate: float = 0.0005
    epochs: int = 100
    patience: int = 15
    use_early_stopping: bool = False


@dataclass
class StudentConfig:
    """Student branch 설정"""
    enabled: bool = True
    optimize_period: int = 1  # 매 N epoch마다 student 최적화 (default: 1)


@dataclass
class DiseaseRatioRegConfig:
    """Disease Ratio Regularization 설정"""
    enabled: bool = False
    lambda_weight: float = 0.1
    alpha: float = 1.0  # Beta prior alpha (smoothing)
    beta: float = 1.0   # Beta prior beta (smoothing)


@dataclass
class LossConfig:
    """손실 함수 설정"""
    negative_weight: float = 0.3
    orthogonal_projection_lambda: float = 0.2
    disease_ratio_reg: DiseaseRatioRegConfig = field(default_factory=DiseaseRatioRegConfig)


@dataclass
class SubsamplingConfig:
    """서브샘플링 설정"""
    enabled: bool = False
    max_cells_per_sample: int = 5000


@dataclass
class MILConfig:
    """MIL 전체 설정"""
    latent_dim: int = 128
    attention_dim: int = 128
    num_classes: int = 2
    freeze_encoder: bool = True
    use_projection: bool = True
    projection_dim: int = 128
    training: MILTrainingConfig = field(default_factory=MILTrainingConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    subsampling: SubsamplingConfig = field(default_factory=SubsamplingConfig)


@dataclass
class EvaluationConfig:
    """평가 설정"""
    metrics: List[str] = field(default_factory=lambda: ["auc", "accuracy", "f1_score"])
    save_attention_scores: bool = True
    save_predictions: bool = True


@dataclass
class LoggingConfig:
    """로깅 설정"""
    level: str = "INFO"
    save_format: str = "csv"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10


@dataclass
class HardwareConfig:
    """하드웨어 설정"""
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 4


@dataclass
class TuningConfig:
    """하이퍼파라미터 튜닝 설정"""
    enabled: bool = False
    # Search space
    learning_rate: List[float] = field(default_factory=lambda: [0.0001])
    encoder_learning_rate: List[float] = field(default_factory=lambda: [0.0005])
    epochs: List[int] = field(default_factory=lambda: [10, 30])  # Transfer learning이므로 작은 epoch
    disease_ratio_lambda: List[float] = field(default_factory=lambda: [0.0])
    # Evaluation
    metric: str = "auc"  # 최적화 기준 메트릭
    # Output
    results_file: str = "tuning_results.csv"
    # Model saving
    save_top_k: int = 3  # Top K 조합의 모델 저장 (0이면 저장 안함)


@dataclass
class ScMILDConfig:
    """scMILD 전체 설정"""
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    mil: MILConfig = field(default_factory=MILConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)


# ============================================================================
# YAML Loading utilities
# ============================================================================

def _resolve_variables(config: dict, root: dict = None, max_iterations: int = 10) -> dict:
    """
    ${paths.data_root} 같은 변수 참조를 해결합니다.
    중첩된 변수 참조 (예: ${paths.project_root}가 ${paths.data_root}를 참조)를 지원합니다.
    """
    if root is None:
        root = config

    def _get_nested(d: dict, key_path: str):
        """점(.)으로 구분된 키 경로에서 값을 가져옵니다."""
        keys = key_path.split(".")
        value = d
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    def _has_variables(s: str) -> bool:
        """문자열에 ${...} 패턴이 있는지 확인합니다."""
        return '${' in s

    def _resolve_string(s: str, current_root: dict) -> str:
        """문자열 내의 ${...} 패턴을 해결합니다."""
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, s)
        for match in matches:
            replacement = _get_nested(current_root, match)
            if replacement is not None:
                s = s.replace(f"${{{match}}}", str(replacement))
        return s

    def _resolve_dict(d: dict, current_root: dict) -> dict:
        """딕셔너리 내의 모든 변수를 해결합니다."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = _resolve_dict(value, current_root)
            elif isinstance(value, str):
                result[key] = _resolve_string(value, current_root)
            elif isinstance(value, list):
                result[key] = [
                    _resolve_string(v, current_root) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                result[key] = value
        return result

    def _has_unresolved_variables(d: dict) -> bool:
        """딕셔너리에 미해결 변수가 있는지 확인합니다."""
        for value in d.values():
            if isinstance(value, dict):
                if _has_unresolved_variables(value):
                    return True
            elif isinstance(value, str):
                if _has_variables(value):
                    return True
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, str) and _has_variables(v):
                        return True
        return False

    # 반복적으로 변수 해석 (중첩 변수 지원)
    result = config.copy()
    for _ in range(max_iterations):
        result = _resolve_dict(result, result)
        if not _has_unresolved_variables(result):
            break

    return result


def _merge_configs(base: dict, override: dict) -> dict:
    """
    두 설정 딕셔너리를 병합합니다. override가 base를 덮어씁니다.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> ScMILDConfig:
    """
    YAML 설정 파일을 로드하고 ScMILDConfig 객체로 변환합니다.

    Args:
        config_path: YAML 설정 파일 경로

    Returns:
        ScMILDConfig 객체
    """
    config_path = Path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # _base_ 설정이 있으면 기본 설정 로드 후 병합
    if "_base_" in config:
        base_path = config_path.parent / config["_base_"]
        with open(base_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)
        del config["_base_"]
        config = _merge_configs(base_config, config)

    # 변수 참조 해결
    config = _resolve_variables(config)

    # Dataclass로 변환
    return _dict_to_config(config)


def _dict_to_config(d: dict) -> ScMILDConfig:
    """딕셔너리를 ScMILDConfig 객체로 변환합니다."""

    # 중첩된 dataclass 생성을 위한 헬퍼
    def _make_dataclass(cls, data):
        if data is None:
            return cls()
        # 해당 클래스의 필드만 추출
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    # 각 섹션별로 dataclass 생성
    paths = _make_dataclass(PathsConfig, d.get("paths"))

    # Data config (nested)
    data_dict = d.get("data", {})
    columns = _make_dataclass(ColumnsConfig, data_dict.get("columns"))
    conditional_embedding = _make_dataclass(ConditionalEmbeddingConfig, data_dict.get("conditional_embedding"))
    preprocessing = _make_dataclass(PreprocessingConfig, data_dict.get("preprocessing"))
    subset = _make_dataclass(SubsetConfig, data_dict.get("subset"))
    data = DataConfig(
        whole_adata_path=data_dict.get("whole_adata_path"),
        subset=subset,
        columns=columns,
        conditional_embedding=conditional_embedding,
        preprocessing=preprocessing,
    )

    splitting = _make_dataclass(SplittingConfig, d.get("splitting"))

    # Encoder config (nested)
    encoder_dict = d.get("encoder", {})
    pretrain = _make_dataclass(EncoderPretrainConfig, encoder_dict.get("pretrain"))
    encoder = EncoderConfig(
        type=encoder_dict.get("type", "VQ_AENB_Conditional"),
        latent_dim=encoder_dict.get("latent_dim", 128),
        num_codes=encoder_dict.get("num_codes", 1024),
        study_emb_dim=encoder_dict.get("study_emb_dim", 16),
        pretrain=pretrain,
    )

    # MIL config (nested)
    mil_dict = d.get("mil", {})
    mil_training = _make_dataclass(MILTrainingConfig, mil_dict.get("training"))
    student = _make_dataclass(StudentConfig, mil_dict.get("student"))

    # Loss config (nested with disease_ratio_reg)
    loss_dict = mil_dict.get("loss", {})
    disease_ratio_reg = _make_dataclass(DiseaseRatioRegConfig, loss_dict.get("disease_ratio_reg"))
    loss = LossConfig(
        negative_weight=loss_dict.get("negative_weight", 0.3),
        orthogonal_projection_lambda=loss_dict.get("orthogonal_projection_lambda", 0.2),
        disease_ratio_reg=disease_ratio_reg,
    )

    subsampling = _make_dataclass(SubsamplingConfig, mil_dict.get("subsampling"))
    mil = MILConfig(
        latent_dim=mil_dict.get("latent_dim", 128),
        attention_dim=mil_dict.get("attention_dim", 128),
        num_classes=mil_dict.get("num_classes", 2),
        freeze_encoder=mil_dict.get("freeze_encoder", True),
        use_projection=mil_dict.get("use_projection", True),
        projection_dim=mil_dict.get("projection_dim", 128),
        training=mil_training,
        student=student,
        loss=loss,
        subsampling=subsampling,
    )

    # Evaluation config
    eval_dict = d.get("evaluation", {})
    evaluation = EvaluationConfig(
        metrics=eval_dict.get("metrics", ["auc", "accuracy", "f1_score"]),
        save_attention_scores=eval_dict.get("save_attention_scores", True),
        save_predictions=eval_dict.get("save_predictions", True),
    )

    logging_cfg = _make_dataclass(LoggingConfig, d.get("logging"))
    hardware = _make_dataclass(HardwareConfig, d.get("hardware"))
    tuning = _make_dataclass(TuningConfig, d.get("tuning"))

    return ScMILDConfig(
        paths=paths,
        data=data,
        splitting=splitting,
        encoder=encoder,
        mil=mil,
        evaluation=evaluation,
        logging=logging_cfg,
        hardware=hardware,
        tuning=tuning,
    )


def save_config(config: ScMILDConfig, path: str):
    """
    ScMILDConfig 객체를 YAML 파일로 저장합니다.
    """
    import dataclasses

    def _to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        return obj

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(_to_dict(config), f, default_flow_style=False, allow_unicode=True)


# ============================================================================
# Convenience functions
# ============================================================================

def get_config_path(name: str) -> Path:
    """설정 파일 경로를 반환합니다."""
    config_dir = Path(__file__).parent.parent / "config"
    return config_dir / f"{name}.yaml"


def list_configs() -> List[str]:
    """사용 가능한 설정 파일 목록을 반환합니다."""
    config_dir = Path(__file__).parent.parent / "config"
    return [p.stem for p in config_dir.glob("*.yaml")]
