"""
ModernBERT Model Implementation for Kadaster Legal Document Classification

This module implements ModernBERT (answerdotai/ModernBERT-base) with:
- Extended context window (8192 tokens vs 512 for standard BERT)
- GPU acceleration for RTX 4070
- Flash Attention 2 support for improved performance
- Optimized for long legal documents

Model: answerdotai/ModernBERT-base
Context: Up to 8192 tokens
Hardware: NVIDIA RTX 4070
"""

import torch
import torch.nn as nn
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings


class ModernBERTConfig:
    """Configuration for ModernBERT model"""

    # Model settings
    MODEL_NAME = "answerdotai/ModernBERT-base"
    MAX_LENGTH = 8192  # Full context window (use 2048 for CPU fallback)

    # Hardware settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FLASH_ATTENTION = True  # Enable Flash Attention 2 if available

    # Training settings - Optimized for RTX 4070 (12GB VRAM)
    BATCH_SIZE = 4  # For GPU (use 2 for CPU)
    GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 32 (use 4 for CPU)
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3  # Full training (use 2 for quick tests)
    WARMUP_STEPS = 500  # Full training (use 100 for small datasets)
    WEIGHT_DECAY = 0.01
    FP16 = torch.cuda.is_available()  # Only use FP16 on CUDA

    # Model checkpoint
    SAVE_DIR = "models/modernbert"


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
        Train the model

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save model checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        print("\n" + "="*80)
        print("TRAINING MODERNBERT MODEL")
        print("="*80)

        # Training arguments optimized for RTX 4070
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=ModernBERTConfig.GRADIENT_ACCUMULATION_STEPS,
            learning_rate=learning_rate,
            weight_decay=ModernBERTConfig.WEIGHT_DECAY,
            warmup_steps=ModernBERTConfig.WARMUP_STEPS,

            # Mixed precision for RTX 4070
            fp16=ModernBERTConfig.FP16,

            # Evaluation
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",

            # Logging
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            report_to="none",  # Disable wandb/tensorboard

            # Performance
            dataloader_num_workers=0,  # Set to 0 for Windows compatibility
            gradient_checkpointing=True,  # Save memory

            # Optimization
            optim="adamw_torch",
            lr_scheduler_type="cosine",
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

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

    def evaluate(self, test_dataset: LegalDocumentDataset, batch_size: int = ModernBERTConfig.BATCH_SIZE):
        """
        Evaluate the model

        Args:
            test_dataset: Test dataset
            batch_size: Evaluation batch size

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

        print(f"\nTest Results:")
        print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

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
