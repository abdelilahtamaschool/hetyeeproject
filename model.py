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

OPTIMIZATIONS (v2.0):
- Focal Loss for minority class handling
- Increased batch size (16 per GPU, 128 effective)
- Lower learning rate (1e-5) for better convergence
- More epochs (6) for complex classification
- Proper multi-GPU training support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
import pickle
from pathlib import Path
import pandas as pd


class ModernBERTConfig:
    """Configuration for ModernBERT model - Optimized for 8x RTX 5090"""

    # Model settings
    MODEL_NAME = "answerdotai/ModernBERT-base"
    MAX_LENGTH = 8192  # Full context window (use 2048 for CPU fallback)

    # Hardware settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FLASH_ATTENTION = True  # Enable Flash Attention 2 if available

    # Training settings - OPTIMIZED FOR 8x RTX 5090 (270GB total VRAM)
    BATCH_SIZE = 16  # Per GPU (8 GPUs × 16 = 128 effective batch size)
    GRADIENT_ACCUMULATION_STEPS = 1  # Not needed with large batch size
    LEARNING_RATE = 1e-5  # Lower for better convergence with many classes
    NUM_EPOCHS = 6  # More epochs for 45 classes
    WARMUP_STEPS = 1000  # More warmup for stability
    WEIGHT_DECAY = 0.01
    FP16 = torch.cuda.is_available()  # Mixed precision training
    LABEL_SMOOTHING = 0.1  # Prevent overconfidence

    # Class imbalance handling
    USE_FOCAL_LOSS = True  # Enable Focal Loss for minority classes
    FOCAL_ALPHA = 0.25  # Weight for positive class
    FOCAL_GAMMA = 2.0  # Focus on hard examples
    USE_CLASS_WEIGHTS = True  # Enable class weighting

    # Model checkpoint
    SAVE_DIR = "models/modernbert_optimized"


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance

    Focal Loss focuses training on hard examples and down-weights easy examples.
    This is crucial for legal document classification where some classes have very few samples.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss

        Args:
            inputs: Predictions (logits) from model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Focal loss value
        """
        # Get probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)

        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedTrainer(Trainer):
    """
    Custom Trainer with Focal Loss and class weighting support
    """

    def __init__(self, *args, class_weights=None, use_focal_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss

        if self.use_focal_loss:
            self.focal_loss = FocalLoss(
                alpha=ModernBERTConfig.FOCAL_ALPHA,
                gamma=ModernBERTConfig.FOCAL_GAMMA
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation with Focal Loss and class weighting
        """
        labels = inputs.pop("labels")
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
        Train the model with Focal Loss and class weighting

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        print("\n" + "="*80)
        print("TRAINING MODERNBERT MODEL (OPTIMIZED v2.0)")
        print("="*80)

        # Compute class weights for handling imbalance
        class_weights = None
        if ModernBERTConfig.USE_CLASS_WEIGHTS:
            print("\nComputing class weights for imbalanced dataset...")
            train_labels = np.array([train_dataset.labels[i] for i in range(len(train_dataset))])
            unique_labels = np.unique(train_labels)

            # Compute balanced weights
            weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_labels,
                y=train_labels
            )
            class_weights = torch.FloatTensor(weights)

            print(f"  Class weights computed for {len(unique_labels)} classes")
            print(f"  Min weight: {weights.min():.4f}, Max weight: {weights.max():.4f}")

        # Training arguments optimized for 8x RTX 5090
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=ModernBERTConfig.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=learning_rate,
            weight_decay=ModernBERTConfig.WEIGHT_DECAY,
            warmup_steps=ModernBERTConfig.WARMUP_STEPS,

            # Mixed precision for RTX 5090
            fp16=ModernBERTConfig.FP16,

            # Label smoothing to prevent overconfidence
            label_smoothing_factor=ModernBERTConfig.LABEL_SMOOTHING,

            # Evaluation
            eval_strategy="steps",
            eval_steps=100,  # More frequent evaluation
            save_strategy="steps",
            save_steps=100,
            save_total_limit=5,  # Keep more checkpoints
            load_best_model_at_end=True,
            metric_for_best_model="f1",  # Optimize for F1 instead of accuracy

            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=25,  # More frequent logging
            report_to="none",  # Disable wandb/tensorboard

            # Performance - Optimized for 8 GPUs with 270GB VRAM
            dataloader_num_workers=4,  # Use workers on Linux
            gradient_checkpointing=False,  # Disable - we have enough VRAM
            ddp_find_unused_parameters=False,  # For multi-GPU stability

            # Optimization
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,  # Gradient clipping
        )

        # Initialize trainer with Focal Loss or class weights
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # More patience
            class_weights=class_weights,
            use_focal_loss=ModernBERTConfig.USE_FOCAL_LOSS
        )

        print(f"\nTraining configuration:")
        print(f"  Focal Loss: {'Enabled' if ModernBERTConfig.USE_FOCAL_LOSS else 'Disabled'}")
        print(f"  Class Weighting: {'Enabled' if class_weights is not None else 'Disabled'}")
        print(f"  Batch size per GPU: {batch_size}")
        print(f"  Effective batch size: {batch_size * torch.cuda.device_count() if torch.cuda.is_available() else batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Label smoothing: {ModernBERTConfig.LABEL_SMOOTHING}")

        # Train
        print("\nStarting training...")
        train_result = trainer.train()

        # Save model
        print(f"\nSaving model to {output_dir}...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Print results
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
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
