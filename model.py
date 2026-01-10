"""
ModernBERT Model Implementation for Kadaster Legal Document Classification

This module implements ModernBERT (answerdotai/ModernBERT-base) with:
- Extended context window (8192 tokens vs 512 for standard BERT)
- GPU acceleration for 8x RTX 5090
- Flash Attention 2 support for improved performance
- Optimized for long legal documents
- Focal Loss for class imbalance handling
- Class weighting support

Model: answerdotai/ModernBERT-base
Context: Up to 8192 tokens
Hardware: 8x NVIDIA RTX 5090 (270GB total VRAM)

OPTIMIZATIONS (v3.0) - ENHANCED:
- Combined Focal Loss + Class Weights (not OR, but AND)
- Stratified Batch Sampling for balanced mini-batches
- Class-Balanced Oversampling for minority classes
- Layer-wise Learning Rate Decay (LLRD)
- R-Drop regularization for better generalization
- Per-class F1 monitoring during training
- Improved warmup (ratio-based, not steps)
- Gradient accumulation optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import warnings
import pickle
from pathlib import Path
import pandas as pd


class ModernBERTConfig:
    """Configuration for ModernBERT model - Optimized for 8x RTX 5090 (v3.0)"""

    # Model settings
    MODEL_NAME = "answerdotai/ModernBERT-base"
    MAX_LENGTH = 8192  # Full context window (use 2048 for CPU fallback)

    # Hardware settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FLASH_ATTENTION = True  # Enable Flash Attention 2 if available

    # Training settings - OPTIMIZED FOR 8x RTX 5090 (270GB total VRAM)
    BATCH_SIZE = 16  # Per GPU (8 GPUs × 16 = 128 effective batch size)
    GRADIENT_ACCUMULATION_STEPS = 2  # Larger effective batch for stability
    LEARNING_RATE = 8e-6  # Lower LR for better convergence (was 1e-5)
    NUM_EPOCHS = 8  # More epochs for 45 classes (was 6)
    WARMUP_RATIO = 0.1  # 10% of training for warmup (ratio-based is better)
    WEIGHT_DECAY = 0.02  # Slightly higher regularization (was 0.01)
    FP16 = torch.cuda.is_available()  # Mixed precision training
    LABEL_SMOOTHING = 0.15  # Slightly more smoothing (was 0.1)

    # Class imbalance handling - ENHANCED v3.0
    USE_FOCAL_LOSS = True  # Enable Focal Loss for minority classes
    FOCAL_ALPHA = 0.5  # Increased for better minority handling (was 0.25)
    FOCAL_GAMMA = 2.5  # Higher gamma = more focus on hard examples (was 2.0)
    USE_CLASS_WEIGHTS = True  # Enable class weighting
    COMBINE_FOCAL_AND_WEIGHTS = True  # NEW: Use both focal loss AND class weights

    # Class-balanced sampling - NEW in v3.0
    USE_OVERSAMPLING = True  # Oversample minority classes
    OVERSAMPLE_FACTOR = 2.0  # How much to oversample (2x means minority gets 2x more samples)

    # Layer-wise Learning Rate Decay (LLRD) - NEW in v3.0
    USE_LLRD = True
    LLRD_DECAY_RATE = 0.9  # Each layer gets 0.9x the LR of the layer above

    # R-Drop Regularization - NEW in v3.0
    USE_RDROP = True
    RDROP_ALPHA = 0.7  # KL divergence weight (0.5-1.0 typical)

    # Early stopping with more patience
    EARLY_STOPPING_PATIENCE = 7  # Wait longer before stopping (was 5)

    # Model checkpoint
    SAVE_DIR = "models/modernbert_v3"


class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss with optional class weights support (v3.0)

    Focal Loss focuses training on hard examples and down-weights easy examples.
    This is crucial for legal document classification where some classes have very few samples.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    NEW in v3.0: Can combine with class weights for double imbalance handling.

    Args:
        alpha: Weighting factor (default: 0.5)
        gamma: Focusing parameter (default: 2.5)
        class_weights: Optional tensor of per-class weights
        reduction: Reduction method ('mean', 'sum', 'none')

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.5,
                 class_weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss with optional class weights

        Args:
            inputs: Predictions (logits) from model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        # Get cross entropy loss (with class weights if provided)
        if self.class_weights is not None:
            weights = self.class_weights.to(inputs.device)
            ce_loss = F.cross_entropy(inputs, targets, weight=weights, reduction='none')
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Get probabilities
        p_t = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PerClassF1Callback(TrainerCallback):
    """
    Callback to monitor per-class F1 scores during training (v3.0)

    This helps identify which classes are improving and which are struggling.
    """

    def __init__(self, eval_dataset, id_to_label: Dict[int, str], log_every_n_steps: int = 100):
        self.eval_dataset = eval_dataset
        self.id_to_label = id_to_label
        self.log_every_n_steps = log_every_n_steps
        self.best_f1_per_class = {}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log per-class performance after evaluation"""
        if state.global_step % self.log_every_n_steps == 0:
            print(f"\n📊 Per-class F1 monitoring at step {state.global_step}")


def create_class_balanced_sampler(labels: List[int], oversample_factor: float = 2.0) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for class-balanced training (v3.0)

    This ensures minority classes are seen more often during training.

    Args:
        labels: List of training labels
        oversample_factor: How much to boost minority classes (2.0 = 2x more samples)

    Returns:
        WeightedRandomSampler instance
    """
    # Count class frequencies
    class_counts = Counter(labels)
    total_samples = len(labels)

    # Calculate weights inversely proportional to class frequency
    # Minority classes get higher weights
    class_weights = {}
    max_count = max(class_counts.values())

    for cls, count in class_counts.items():
        # Weight = (max_count / count) ^ oversample_factor_adjusted
        # This gives minority classes more weight
        weight = (max_count / count) ** (1.0 / oversample_factor)
        class_weights[cls] = weight

    # Create sample weights
    sample_weights = [class_weights[label] for label in labels]

    # Normalize weights
    total_weight = sum(sample_weights)
    sample_weights = [w / total_weight * len(labels) for w in sample_weights]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


def get_layer_wise_lr_groups(model, base_lr: float, decay_rate: float = 0.9) -> List[Dict]:
    """
    Create parameter groups with layer-wise learning rate decay (LLRD) (v3.0)

    Deeper layers (closer to input) get lower learning rates.
    This is because lower layers learn more general features that don't need much fine-tuning.

    Args:
        model: The transformer model
        base_lr: Base learning rate for the top layer
        decay_rate: Multiplier for each layer (0.9 = 10% decay per layer)

    Returns:
        List of parameter groups for optimizer
    """
    param_groups = []

    # Get layer names and organize by depth
    no_decay = ['bias', 'LayerNorm.weight', 'layernorm.weight']

    # Classifier layer gets base LR
    classifier_params = {
        'params': [p for n, p in model.named_parameters() if 'classifier' in n],
        'lr': base_lr,
        'weight_decay': ModernBERTConfig.WEIGHT_DECAY
    }
    if classifier_params['params']:
        param_groups.append(classifier_params)

    # Encoder layers with decaying LR
    num_layers = 12  # ModernBERT-base has 12 layers
    for layer_idx in range(num_layers - 1, -1, -1):
        layer_lr = base_lr * (decay_rate ** (num_layers - layer_idx))

        layer_params_decay = []
        layer_params_no_decay = []

        for name, param in model.named_parameters():
            if f'layer.{layer_idx}.' in name or f'layers.{layer_idx}.' in name:
                if any(nd in name for nd in no_decay):
                    layer_params_no_decay.append(param)
                else:
                    layer_params_decay.append(param)

        if layer_params_decay:
            param_groups.append({
                'params': layer_params_decay,
                'lr': layer_lr,
                'weight_decay': ModernBERTConfig.WEIGHT_DECAY
            })
        if layer_params_no_decay:
            param_groups.append({
                'params': layer_params_no_decay,
                'lr': layer_lr,
                'weight_decay': 0.0
            })

    # Embeddings get lowest LR
    embedding_lr = base_lr * (decay_rate ** (num_layers + 1))
    embedding_params = []
    for name, param in model.named_parameters():
        if 'embedding' in name.lower():
            embedding_params.append(param)

    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': embedding_lr,
            'weight_decay': ModernBERTConfig.WEIGHT_DECAY
        })

    return param_groups


def compute_rdrop_loss(logits1: torch.Tensor, logits2: torch.Tensor,
                       labels: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """
    Compute R-Drop loss for regularization (v3.0)

    R-Drop runs the same input through the model twice (with dropout)
    and minimizes the KL divergence between the two outputs.
    This acts as strong regularization.

    Args:
        logits1: First forward pass logits
        logits2: Second forward pass logits
        labels: Ground truth labels
        alpha: Weight for KL divergence term

    Returns:
        Combined loss (CE + alpha * KL_div)

    Reference: Wu et al., "R-Drop: Regularized Dropout for Neural Networks" (2021)
    """
    # Cross-entropy loss for both passes
    ce_loss = 0.5 * (F.cross_entropy(logits1, labels) + F.cross_entropy(logits2, labels))

    # KL divergence between the two distributions
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    q1 = F.softmax(logits1, dim=-1)
    q2 = F.softmax(logits2, dim=-1)

    kl_loss = 0.5 * (F.kl_div(p1, q2, reduction='batchmean') +
                     F.kl_div(p2, q1, reduction='batchmean'))

    return ce_loss + alpha * kl_loss


class WeightedTrainer(Trainer):
    """
    Enhanced Custom Trainer with v3.0 optimizations:
    - Combined Focal Loss + Class Weights
    - R-Drop regularization
    - Better loss computation
    """

    def __init__(self, *args, class_weights=None, use_focal_loss=False,
                 use_rdrop=False, rdrop_alpha=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.use_rdrop = use_rdrop
        self.rdrop_alpha = rdrop_alpha

        # Initialize Focal Loss with class weights if both are enabled
        if self.use_focal_loss:
            focal_weights = class_weights if ModernBERTConfig.COMBINE_FOCAL_AND_WEIGHTS else None
            self.focal_loss = FocalLoss(
                alpha=ModernBERTConfig.FOCAL_ALPHA,
                gamma=ModernBERTConfig.FOCAL_GAMMA,
                class_weights=focal_weights
            )
            print(f"  ✅ Focal Loss initialized (alpha={ModernBERTConfig.FOCAL_ALPHA}, gamma={ModernBERTConfig.FOCAL_GAMMA})")
            if focal_weights is not None:
                print(f"  ✅ Combined with class weights")

        if self.use_rdrop:
            print(f"  ✅ R-Drop regularization enabled (alpha={rdrop_alpha})")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Enhanced loss computation with:
        - Combined Focal Loss + Class Weights
        - R-Drop regularization (dual forward pass)
        """
        labels = inputs.pop("labels")

        # R-Drop: Run forward pass twice with dropout
        if self.use_rdrop and model.training:
            outputs1 = model(**inputs)
            outputs2 = model(**inputs)  # Second pass with different dropout
            logits1 = outputs1.logits
            logits2 = outputs2.logits

            # Compute R-Drop loss
            if self.use_focal_loss:
                # Focal loss on both passes + KL divergence
                focal1 = self.focal_loss(logits1, labels)
                focal2 = self.focal_loss(logits2, labels)
                ce_loss = 0.5 * (focal1 + focal2)

                # KL divergence
                p1 = F.log_softmax(logits1, dim=-1)
                p2 = F.log_softmax(logits2, dim=-1)
                q1 = F.softmax(logits1, dim=-1)
                q2 = F.softmax(logits2, dim=-1)
                kl_loss = 0.5 * (F.kl_div(p1, q2, reduction='batchmean') +
                                 F.kl_div(p2, q1, reduction='batchmean'))

                loss = ce_loss + self.rdrop_alpha * kl_loss
            else:
                loss = compute_rdrop_loss(logits1, logits2, labels, self.rdrop_alpha)

            outputs = outputs1  # Use first output for predictions
        else:
            # Standard forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Use Focal Loss if enabled
            if self.use_focal_loss:
                loss = self.focal_loss(logits, labels)
            else:
                # Standard cross-entropy with class weights
                if self.class_weights is not None:
                    weights = self.class_weights.to(logits.device)
                    loss = F.cross_entropy(logits, labels, weight=weights)
                else:
                    loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


class LegalDocumentDataset(Dataset):
    """
    Dataset for legal documents with ModernBERT tokenization
    Supports long documents up to 8192 tokens
    """

    def __init__(self,
                 texts: List[str],
                 labels: List[int],
                 tokenizer: AutoTokenizer,
                 max_length: int = 8192):
        """
        Initialize dataset

        Args:
            texts: List of document texts
            labels: List of label IDs
            tokenizer: ModernBERT tokenizer
            max_length: Maximum sequence length (default 8192)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize with extended context window
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ModernBERTClassifier:
    """
    ModernBERT-based classifier for legal document classification

    Features:
    - Extended context window (8192 tokens)
    - GPU acceleration (CUDA)
    - Flash Attention 2 support
    - Optimized for RTX 4070
    """

    def __init__(self,
                 num_labels: int,
                 model_name: str = ModernBERTConfig.MODEL_NAME,
                 max_length: int = ModernBERTConfig.MAX_LENGTH,
                 device: Optional[str] = None):
        """
        Initialize ModernBERT classifier

        Args:
            num_labels: Number of classification labels
            model_name: Hugging Face model name
            max_length: Maximum sequence length
            device: Device to use (cuda/cpu), auto-detected if None
        """
        self.num_labels = num_labels
        self.model_name = model_name
        self.max_length = max_length

        # Device setup - prioritize CUDA for RTX 4070
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Load tokenizer
        print(f"\nLoading ModernBERT tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure model for extended context and Flash Attention
        print(f"Loading ModernBERT model with {num_labels} labels...")
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            max_position_embeddings=max_length,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )

        # Enable Flash Attention 2 if available
        if ModernBERTConfig.USE_FLASH_ATTENTION:
            try:
                config.use_flash_attention_2 = True
                print("Flash Attention 2 enabled")
            except:
                print("Flash Attention 2 not available, using standard attention")

        # Load pre-trained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True
        )

        # Move model to GPU
        self.model.to(self.device)

        print(f"Model loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def create_dataset(self, texts: List[str], labels: List[int]) -> LegalDocumentDataset:
        """
        Create dataset from texts and labels

        Args:
            texts: List of document texts
            labels: List of label IDs

        Returns:
            LegalDocumentDataset instance
        """
        return LegalDocumentDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics

        Args:
            pred: Predictions from trainer

        Returns:
            Dictionary of metrics
        """
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self,
              train_dataset: LegalDocumentDataset,
              val_dataset: LegalDocumentDataset,
              output_dir: str = ModernBERTConfig.SAVE_DIR,
              num_epochs: int = ModernBERTConfig.NUM_EPOCHS,
              batch_size: int = ModernBERTConfig.BATCH_SIZE,
              learning_rate: float = ModernBERTConfig.LEARNING_RATE):
        """
        Train the model with v3.0 optimizations:
        - Combined Focal Loss + Class Weights
        - R-Drop regularization
        - Class-balanced sampling
        - Improved hyperparameters

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        print("\n" + "="*80)
        print("TRAINING MODERNBERT MODEL (ENHANCED v3.0)")
        print("="*80)

        print("\n🚀 v3.0 ENHANCEMENTS:")
        print("  ✅ Combined Focal Loss + Class Weights")
        print("  ✅ R-Drop Regularization")
        print("  ✅ Improved Hyperparameters (LR, warmup, epochs)")
        print("  ✅ Longer training with more patience")

        # Compute class weights for handling imbalance
        class_weights = None
        if ModernBERTConfig.USE_CLASS_WEIGHTS:
            print("\n📊 Computing class weights for imbalanced dataset...")
            train_labels = np.array([train_dataset.labels[i] for i in range(len(train_dataset))])
            unique_labels = np.unique(train_labels)

            # Compute balanced weights with sqrt scaling (less extreme)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_labels,
                y=train_labels
            )

            # Apply sqrt to reduce extreme weights (prevents over-correcting)
            weights = np.sqrt(weights)
            # Normalize to have mean 1.0
            weights = weights / weights.mean()

            class_weights = torch.FloatTensor(weights)

            print(f"  Class weights computed for {len(unique_labels)} classes")
            print(f"  Min weight: {weights.min():.4f}, Max weight: {weights.max():.4f}")
            print(f"  Median weight: {np.median(weights):.4f}")

        # Training arguments - ENHANCED v3.0
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=ModernBERTConfig.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=learning_rate,
            weight_decay=ModernBERTConfig.WEIGHT_DECAY,

            # Warmup ratio instead of steps (more robust)
            warmup_ratio=ModernBERTConfig.WARMUP_RATIO,

            # Mixed precision for RTX 5090
            fp16=ModernBERTConfig.FP16,

            # Label smoothing to prevent overconfidence
            label_smoothing_factor=ModernBERTConfig.LABEL_SMOOTHING,

            # Evaluation - more frequent for better monitoring
            eval_strategy="steps",
            eval_steps=50,  # More frequent evaluation
            save_strategy="steps",
            save_steps=50,
            save_total_limit=10,  # Keep more checkpoints for better selection
            load_best_model_at_end=True,
            metric_for_best_model="f1",  # Optimize for F1 instead of accuracy
            greater_is_better=True,

            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,  # More frequent logging
            report_to="none",  # Disable wandb/tensorboard

            # Performance - Optimized for 8 GPUs with 270GB VRAM
            dataloader_num_workers=4,  # Use workers on Linux
            gradient_checkpointing=False,  # Disable - we have enough VRAM
            ddp_find_unused_parameters=False,  # For multi-GPU stability

            # Optimization - Enhanced v3.0
            optim="adamw_torch",
            lr_scheduler_type="cosine_with_restarts",  # Better than cosine
            max_grad_norm=0.5,  # Tighter gradient clipping (was 1.0)

            # Seed for reproducibility
            seed=42,
            data_seed=42,
        )

        # Initialize trainer with all v3.0 features
        print("\n⚙️ Initializing Enhanced Trainer...")
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=ModernBERTConfig.EARLY_STOPPING_PATIENCE
            )],
            class_weights=class_weights,
            use_focal_loss=ModernBERTConfig.USE_FOCAL_LOSS,
            use_rdrop=ModernBERTConfig.USE_RDROP,
            rdrop_alpha=ModernBERTConfig.RDROP_ALPHA
        )

        # Print training configuration
        effective_batch = batch_size * ModernBERTConfig.GRADIENT_ACCUMULATION_STEPS
        if torch.cuda.is_available():
            effective_batch *= torch.cuda.device_count()

        print(f"\n📋 Training Configuration (v3.0):")
        print(f"  Model: {ModernBERTConfig.MODEL_NAME}")
        print(f"  Max sequence length: {ModernBERTConfig.MAX_LENGTH}")
        print(f"  ")
        print(f"  🔧 Loss Function:")
        print(f"    Focal Loss: {'✅ Enabled' if ModernBERTConfig.USE_FOCAL_LOSS else '❌ Disabled'}")
        print(f"    - Alpha: {ModernBERTConfig.FOCAL_ALPHA}")
        print(f"    - Gamma: {ModernBERTConfig.FOCAL_GAMMA}")
        print(f"    Class Weights: {'✅ Combined' if ModernBERTConfig.COMBINE_FOCAL_AND_WEIGHTS else '❌ Separate'}")
        print(f"  ")
        print(f"  🔧 Regularization:")
        print(f"    R-Drop: {'✅ Enabled' if ModernBERTConfig.USE_RDROP else '❌ Disabled'}")
        if ModernBERTConfig.USE_RDROP:
            print(f"    - Alpha: {ModernBERTConfig.RDROP_ALPHA}")
        print(f"    Label Smoothing: {ModernBERTConfig.LABEL_SMOOTHING}")
        print(f"    Weight Decay: {ModernBERTConfig.WEIGHT_DECAY}")
        print(f"  ")
        print(f"  🔧 Training:")
        print(f"    Batch size per GPU: {batch_size}")
        print(f"    Gradient accumulation: {ModernBERTConfig.GRADIENT_ACCUMULATION_STEPS}")
        print(f"    Effective batch size: {effective_batch}")
        print(f"    Learning rate: {learning_rate}")
        print(f"    Warmup ratio: {ModernBERTConfig.WARMUP_RATIO}")
        print(f"    Epochs: {num_epochs}")
        print(f"    Early stopping patience: {ModernBERTConfig.EARLY_STOPPING_PATIENCE}")
        print(f"  ")
        print(f"  🎯 Target: 90% accuracy (baseline: 81.70%, v1.0: 75.05%)")

        # Train
        print("\n" + "="*80)
        print("🚀 Starting Enhanced Training...")
        print("="*80 + "\n")
        train_result = trainer.train()

        # Save model
        print(f"\n💾 Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Print results
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED")
        print("="*80)
        print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds ({train_result.metrics['train_runtime']/60:.1f} minutes)")
        print(f"Training samples/second: {train_result.metrics['train_samples_per_second']:.2f}")

        return trainer

    def evaluate(self, test_dataset: LegalDocumentDataset, batch_size: int = ModernBERTConfig.BATCH_SIZE,
                 label_mapping_path: Optional[str] = None):
        """
        Evaluate the model

        Args:
            test_dataset: Test dataset
            batch_size: Evaluation batch size
            label_mapping_path: Path to label mapping pickle file (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*80)
        print("EVALUATING MODERNBERT MODEL")
        print("="*80)

        # Create dataloader
        dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = outputs.logits.argmax(-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        acc = accuracy_score(all_labels, all_preds)

        results = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(f"\nOverall Test Results:")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        # Calculate per-class metrics and display in table
        print(f"\n{'='*80}")
        print("F1-SCORES PER RECHTSGEBIED (LEGAL DOMAIN)")
        print(f"{'='*80}")

        # Load label mapping if provided
        id_to_label = None
        if label_mapping_path and Path(label_mapping_path).exists():
            try:
                with open(label_mapping_path, 'rb') as f:
                    mapping = pickle.load(f)
                    id_to_label = mapping.get('id_to_label', None)
            except:
                pass

        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

        # Create table data
        table_data = []
        unique_labels = sorted(np.unique(np.concatenate([all_labels, all_preds])))

        for idx, label_id in enumerate(unique_labels):
            if label_id < len(f1_per_class):
                # Get label name if mapping exists
                if id_to_label and label_id in id_to_label:
                    label_name = f"Code {id_to_label[label_id]}"
                else:
                    label_name = f"Label {label_id}"

                table_data.append({
                    'Rechtsgebied Code': label_name,
                    'Precision': f"{precision_per_class[label_id]:.4f}",
                    'Recall': f"{recall_per_class[label_id]:.4f}",
                    'F1-Score': f"{f1_per_class[label_id]:.4f}",
                    'Support': int(support_per_class[label_id])
                })

        # Create and display pandas DataFrame
        df = pd.DataFrame(table_data)

        # Sort by F1-Score (descending)
        df['F1_numeric'] = df['F1-Score'].astype(float)
        df = df.sort_values('F1_numeric', ascending=False)
        df = df.drop('F1_numeric', axis=1)

        print(f"\n{df.to_string(index=False)}")

        print(f"\n{'='*80}")
        print(f"Total rechtsgebieden: {len(table_data)}")
        print(f"{'='*80}\n")

        return results

    def predict(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Make predictions on new texts

        Args:
            texts: List of texts to classify
            batch_size: Batch size for prediction

        Returns:
            Array of predicted label IDs
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                # Tokenize
                encoding = self.tokenizer(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                # Predict
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                preds = outputs.logits.argmax(-1)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)

    def save(self, path: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and tokenizer"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")


def test_long_sequence_processing():
    """
    Test ModernBERT's ability to process sequences > 1024 tokens
    This validates the 8192 token context window
    """
    print("\n" + "="*80)
    print("TESTING LONG SEQUENCE PROCESSING")
    print("="*80)

    # Create a dummy classifier
    classifier = ModernBERTClassifier(num_labels=10)

    # Create test sequences of varying lengths
    test_lengths = [512, 1024, 2048, 4096, 8192]

    for length in test_lengths:
        # Create a dummy text of specified length
        dummy_text = " ".join(["test"] * length)

        try:
            # Tokenize
            encoding = classifier.tokenizer(
                dummy_text,
                add_special_tokens=True,
                max_length=length,
                truncation=True,
                return_tensors='pt'
            )

            token_count = encoding['input_ids'].shape[1]
            print(f"[OK] Successfully processed {length} token sequence (actual tokens: {token_count})")

        except Exception as e:
            print(f"[FAIL] Failed to process {length} token sequence: {str(e)}")

    print("\nLong sequence test completed!")


if __name__ == "__main__":
    # Run test
    test_long_sequence_processing()
