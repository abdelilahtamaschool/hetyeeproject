"""
Training utilities for the Akte Classification Pipeline.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoTokenizer
)
from typing import Dict, Optional, Callable
from pathlib import Path

from ..config import TrainingConfig, Config
from ..data.dataset import AkteDataset, AkteChunkDataset, collate_fn, chunk_collate_fn
from ..models.classifier import AkteClassifier, SimpleAkteClassifier, FocalLoss
from .metrics import compute_multi_label_metrics


class WeightedTrainer(Trainer):
    """
    Custom Trainer with Focal Loss and class weights support.
    """

    def __init__(
        self,
        *args,
        class_weights: Optional[torch.Tensor] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Initialize loss function
        if use_focal_loss:
            self.loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma,
                pos_weight=class_weights
            )
        elif class_weights is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss with focal loss and class weights.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Move class weights to correct device
        if self.class_weights is not None and self.loss_fn.pos_weight is not None:
            self.loss_fn.pos_weight = self.loss_fn.pos_weight.to(logits.device)

        loss = self.loss_fn(logits, labels.float())

        return (loss, outputs) if return_outputs else loss


def create_training_args(
    config: TrainingConfig,
    output_dir: str = "./models/checkpoints"
) -> TrainingArguments:
    """
    Create HuggingFace TrainingArguments from config.

    Args:
        config: Training configuration
        output_dir: Output directory for checkpoints

    Returns:
        TrainingArguments instance
    """
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        report_to="none",  # Disable wandb/tensorboard by default
        dataloader_num_workers=0,  # Windows compatibility
        dataloader_pin_memory=False,  # Disable pin_memory for CPU/no accelerator
        remove_unused_columns=False,  # Keep doc_idx and chunk_idx for custom collator
    )


def create_trainer(
    model: SimpleAkteClassifier,
    train_dataset: AkteChunkDataset,
    val_dataset: AkteChunkDataset,
    config: TrainingConfig,
    output_dir: str = "./models/checkpoints",
    compute_metrics: Optional[Callable] = None,
    class_weights: Optional[torch.Tensor] = None
) -> Trainer:
    """
    Create HuggingFace Trainer for chunk-level training.

    Args:
        model: The classifier model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Training configuration
        output_dir: Output directory
        compute_metrics: Optional custom metrics function
        class_weights: Optional class weights for imbalanced data

    Returns:
        Configured Trainer instance (WeightedTrainer if class_weights or focal_loss enabled)
    """
    training_args = create_training_args(config, output_dir)

    # Use WeightedTrainer if class weights or focal loss are enabled
    use_focal = getattr(config, 'use_focal_loss', False)
    use_weights = getattr(config, 'use_class_weights', False) and class_weights is not None

    if use_focal or use_weights:
        trainer = WeightedTrainer(
            model=model.model,  # Use the inner HF model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=chunk_collate_fn,
            compute_metrics=compute_metrics or compute_multi_label_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
            ],
            class_weights=class_weights if use_weights else None,
            use_focal_loss=use_focal,
            focal_gamma=getattr(config, 'focal_gamma', 2.0),
            focal_alpha=getattr(config, 'focal_alpha', 0.25)
        )
    else:
        trainer = Trainer(
            model=model.model,  # Use the inner HF model
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=chunk_collate_fn,
            compute_metrics=compute_metrics or compute_multi_label_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
            ]
        )

    return trainer


class CustomTrainer:
    """
    Custom trainer for document-level training with chunk aggregation.

    Handles variable-length chunks per document.
    """

    def __init__(
        self,
        model: AkteClassifier,
        train_dataset: AkteDataset,
        val_dataset: AkteDataset,
        config: Config,
        device: str = None
    ):
        """
        Initialize the custom trainer.

        Args:
            model: AkteClassifier model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Full configuration
            device: Device to use (cuda/cpu)
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        total_steps = len(self.train_loader) * config.training.num_epochs
        warmup_steps = int(total_steps * config.training.warmup_ratio)

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )

        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.training.fp16 else None

        # Best model tracking
        self.best_metric = 0
        self.patience_counter = 0

    def train(self) -> Dict:
        """
        Run the full training loop.

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1_micro': []
        }

        for epoch in range(self.config.training.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")

            # Training
            train_loss = self._train_epoch()
            history['train_loss'].append(train_loss)

            # Validation
            val_metrics = self._validate()
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1_micro'].append(val_metrics['f1_micro'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val F1 Micro: {val_metrics['f1_micro']:.4f}")

            # Early stopping
            if val_metrics['f1_micro'] > self.best_metric:
                self.best_metric = val_metrics['f1_micro']
                self.patience_counter = 0
                self._save_checkpoint(f"best_model.pt")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return history

    def _train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            chunk_mask = batch['chunk_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        chunk_mask=chunk_mask,
                        labels=labels
                    )
                    loss = outputs['loss']

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    chunk_mask=chunk_mask,
                    labels=labels
                )
                loss = outputs['loss']
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate(self) -> Dict:
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                chunk_mask = batch['chunk_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    chunk_mask=chunk_mask,
                    labels=labels
                )

                total_loss += outputs['loss'].item()
                all_logits.append(outputs['logits'].cpu())
                all_labels.append(labels.cpu())

        # Compute metrics
        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        metrics = compute_multi_label_metrics((all_logits, all_labels))
        metrics['loss'] = total_loss / len(self.val_loader)

        return metrics

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        save_path = self.config.output_dir / filename
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")


def train_simple_model(
    model_name: str,
    train_docs: list,
    val_docs: list,
    label2id: dict,
    config: Config,
    output_dir: str = "./models"
) -> Trainer:
    """
    Train a simple chunk-level classifier.

    Args:
        model_name: HuggingFace model name
        train_docs: Training documents
        val_docs: Validation documents
        label2id: Label mapping
        config: Configuration
        output_dir: Output directory

    Returns:
        Trained Trainer instance
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create datasets
    train_dataset = AkteChunkDataset(
        train_docs,
        tokenizer,
        label2id,
        chunk_size=config.chunking.chunk_size,
        overlap=config.chunking.overlap
    )

    val_dataset = AkteChunkDataset(
        val_docs,
        tokenizer,
        label2id,
        chunk_size=config.chunking.chunk_size,
        overlap=config.chunking.overlap
    )

    # Create model
    model = SimpleAkteClassifier(
        model_name=model_name,
        num_labels=len(label2id)
    )

    # Create and run trainer
    trainer = create_trainer(
        model,
        train_dataset,
        val_dataset,
        config.training,
        output_dir
    )

    trainer.train()

    return trainer
