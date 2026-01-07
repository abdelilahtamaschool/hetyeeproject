"""
Configuration classes for the Akte Classification Pipeline.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for the BERTje model."""
    model_name: str = "GroNLP/bert-base-dutch-cased"  # BERTje
    max_length: int = 512
    num_labels: int = 20  # Top 20 rechtsfeitcodes
    problem_type: str = "multi_label_classification"
    hidden_size: int = 768  # BERTje hidden size


@dataclass
class ChunkingConfig:
    """Configuration for document chunking strategy."""
    strategy: str = "sliding_window"
    chunk_size: int = 510  # Leave room for [CLS] and [SEP]
    overlap: int = 128  # Overlap between chunks for context continuity
    aggregation: str = "max"  # Options: max, mean


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    batch_size: int = 64  # H200 can handle large batches (140GB VRAM)
    learning_rate: float = 2e-5  # BERT standard
    num_epochs: int = 20  # More epochs for better convergence
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2  # Effective batch = 128
    max_grad_norm: float = 1.0  # Gradient clipping
    fp16: bool = True  # Mixed precision for GPU
    early_stopping_patience: int = 7  # More patience for convergence
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 50
    save_total_limit: int = 3
    metric_for_best_model: str = "f1_macro"  # Focus on balanced performance
    scheduler_type: str = "cosine"  # Better than linear for BERT
    label_smoothing: float = 0.1  # Helps with calibration
    use_class_weights: bool = True  # Enable class weighting
    use_focal_loss: bool = True  # Enable focal loss for imbalanced data
    focal_gamma: float = 2.0  # Focal loss gamma parameter
    focal_alpha: float = 0.25  # Focal loss alpha parameter


@dataclass
class DataConfig:
    """Configuration for data handling."""
    data_path: Path = field(default_factory=lambda: Path("ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl"))
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    top_n_labels: int = 20  # Focus on top 20 rechtsfeitcodes


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    threshold: float = 0.5  # Default threshold, will be optimized
    min_confidence_for_review: float = 0.3  # Flag for human review below this
    batch_size: int = 16


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("models"))
    results_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self):
        """Ensure directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
default_config = Config()
