"""
Optimized training script for the Akte Classification Pipeline.

Key improvements over train.py:
1. Learning rate: 2e-5 (BERT standard) instead of 0.01
2. Class weights for imbalanced labels
3. Gradient clipping
4. Cosine learning rate scheduler
5. More epochs with early stopping
6. Label smoothing option

Usage:
    python train_optimized.py --data_path <path_to_jsonl>
"""
import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch
import torch.nn as nn

from src.config import Config
from src.data.loader import (
    load_jsonl,
    get_top_n_labels,
    get_label_mapping,
    filter_documents_by_labels,
    analyze_label_distribution,
    get_dataset_statistics
)
from src.data.dataset import AkteChunkDataset, chunk_collate_fn
from src.training.metrics import compute_multi_label_metrics


class WeightedBCETrainer(Trainer):
    """Custom Trainer with class-weighted BCE loss for imbalanced labels."""

    def __init__(self, class_weights=None, label_smoothing=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            smoothed_labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smoothed_labels = labels

        # Weighted BCE loss
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            loss_fct = nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, smoothed_labels.float())

        return (loss, outputs) if return_outputs else loss


def compute_class_weights(documents, label2id, method="inverse_freq"):
    """
    Compute class weights for imbalanced multi-label classification.

    Args:
        documents: List of documents with 'rechtsfeitcodes'
        label2id: Label to ID mapping
        method: 'inverse_freq', 'sqrt_inverse', or 'effective_samples'

    Returns:
        Tensor of class weights
    """
    num_labels = len(label2id)
    label_counts = Counter()
    total_docs = len(documents)

    for doc in documents:
        for code in doc.get('rechtsfeitcodes', []):
            if code in label2id:
                label_counts[label2id[code]] += 1

    weights = torch.ones(num_labels)

    for label_id in range(num_labels):
        count = label_counts.get(label_id, 1)  # Avoid division by zero

        if method == "inverse_freq":
            # Standard inverse frequency
            weights[label_id] = total_docs / (count + 1)
        elif method == "sqrt_inverse":
            # Sqrt for less aggressive weighting
            weights[label_id] = np.sqrt(total_docs / (count + 1))
        elif method == "effective_samples":
            # Effective number of samples (beta=0.999)
            beta = 0.999
            effective = (1 - beta**count) / (1 - beta)
            weights[label_id] = total_docs / (effective + 1)

    # Normalize weights to have mean 1
    weights = weights / weights.mean()

    return weights


def main():
    parser = argparse.ArgumentParser(description="Optimized Akte Classifier Training")
    parser.add_argument(
        "--data_path",
        type=str,
        default="ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl",
        help="Path to JSONL data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=20,
        help="Number of top labels to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,  # Reduced for stability
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,  # More epochs
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,  # BERT standard!
        help="Learning rate (2e-5 is optimal for BERT)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * this)"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping"
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor (0 = disabled)"
    )
    parser.add_argument(
        "--class_weight_method",
        type=str,
        default="sqrt_inverse",
        choices=["none", "inverse_freq", "sqrt_inverse", "effective_samples"],
        help="Method for computing class weights"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents (for testing)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=510,
        help="Chunk size in tokens"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts"],
        help="Learning rate scheduler type"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Akte Classification Pipeline - OPTIMIZED Training")
    print("=" * 70)
    print("\nKey optimizations:")
    print(f"  - Learning rate: {args.learning_rate} (BERT standard)")
    print(f"  - Class weights: {args.class_weight_method}")
    print(f"  - Label smoothing: {args.label_smoothing}")
    print(f"  - Scheduler: {args.scheduler_type}")
    print(f"  - Gradient clipping: {args.max_grad_norm}")
    print("=" * 70)

    # Initialize config
    config = Config()
    config.chunking.chunk_size = args.chunk_size
    config.chunking.overlap = args.overlap

    # Load data
    print(f"\n1. Loading data from {args.data_path}...")
    documents = load_jsonl(Path(args.data_path), limit=args.limit)
    print(f"   Loaded {len(documents)} documents")

    # Analyze labels
    print("\n2. Analyzing label distribution...")
    label_dist = analyze_label_distribution(documents)
    print(f"   Found {len(label_dist)} unique rechtsfeitcodes")
    print(f"\n   Top 10 codes:")
    print(label_dist.head(10).to_string())

    # Get top N labels
    print(f"\n3. Selecting top {args.num_labels} labels...")
    top_labels = get_top_n_labels(documents, args.num_labels)
    print(f"   Top {args.num_labels} codes: {top_labels}")

    coverage = label_dist[label_dist['rechtsfeitcode'].isin(top_labels)]['percentage'].sum()
    print(f"   Coverage: {coverage:.1f}%")

    # Create label mapping
    label2id, id2label = get_label_mapping(top_labels)
    print(f"   Label mapping created: {len(label2id)} labels")

    # Filter documents
    print("\n4. Filtering documents...")
    filtered_docs = filter_documents_by_labels(documents, top_labels)
    print(f"   {len(filtered_docs)} documents with at least one top label")

    # Dataset statistics
    stats = get_dataset_statistics(filtered_docs)
    print(f"\n   Dataset statistics:")
    print(f"   - Documents: {stats['num_documents']}")
    print(f"   - Avg text length: {stats['avg_text_length']:.0f} chars")
    print(f"   - Avg labels per doc: {stats['avg_labels_per_doc']:.2f}")

    # Compute class weights
    print("\n5. Computing class weights...")
    if args.class_weight_method != "none":
        class_weights = compute_class_weights(
            filtered_docs,
            label2id,
            method=args.class_weight_method
        )
        print(f"   Class weights computed using {args.class_weight_method}")
        print(f"   Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
    else:
        class_weights = None
        print("   No class weights (balanced training)")

    # Split data
    print("\n6. Splitting data (80/10/10)...")
    train_docs, temp_docs = train_test_split(
        filtered_docs,
        test_size=0.2,
        random_state=config.data.random_seed
    )
    val_docs, test_docs = train_test_split(
        temp_docs,
        test_size=0.5,
        random_state=config.data.random_seed
    )
    print(f"   Train: {len(train_docs)}")
    print(f"   Val: {len(val_docs)}")
    print(f"   Test: {len(test_docs)}")

    # Initialize tokenizer
    print(f"\n7. Loading tokenizer ({config.model.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Create datasets
    print("\n8. Creating datasets with chunking...")
    print(f"   Chunk size: {args.chunk_size}")
    print(f"   Overlap: {args.overlap}")

    train_dataset = AkteChunkDataset(
        train_docs,
        tokenizer,
        label2id,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    print(f"   Train chunks: {len(train_dataset)}")

    val_dataset = AkteChunkDataset(
        val_docs,
        tokenizer,
        label2id,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    print(f"   Val chunks: {len(val_dataset)}")

    # Initialize model
    print(f"\n9. Initializing model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model.model_name,
        num_labels=len(label2id),
        problem_type="multi_label_classification"
    )
    print(f"   Model: {config.model.model_name}")
    print(f"   Num labels: {len(label2id)}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Create training arguments
    print("\n10. Setting up trainer...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"   Effective batch size: {effective_batch_size}")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        lr_scheduler_type=args.scheduler_type,
    )

    # Create custom trainer with class weights
    trainer = WeightedBCETrainer(
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=chunk_collate_fn,
        compute_metrics=compute_multi_label_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ]
    )

    # Save configuration
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'effective_batch_size': effective_batch_size,
            'epochs': args.epochs,
            'warmup_ratio': args.warmup_ratio,
            'weight_decay': args.weight_decay,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'max_grad_norm': args.max_grad_norm,
            'label_smoothing': args.label_smoothing,
            'class_weight_method': args.class_weight_method,
            'scheduler_type': args.scheduler_type,
            'chunk_size': args.chunk_size,
            'overlap': args.overlap,
            'num_labels': args.num_labels,
            'fp16': args.fp16
        }, f, indent=2)
    print(f"   Config saved to {config_path}")

    # Save label mapping
    mapping_path = output_dir / "label_mapping.json"
    with open(mapping_path, 'w') as f:
        json.dump({
            'label2id': {str(k): v for k, v in label2id.items()},
            'id2label': {str(v): k for k, v in label2id.items()},
            'top_labels': top_labels
        }, f, indent=2)
    print(f"   Label mapping saved to {mapping_path}")

    # Train
    print("\n11. Starting training...")
    print("-" * 70)
    print(f"    Epochs: {args.epochs}")
    print(f"    Batch size: {args.batch_size} (effective: {effective_batch_size})")
    print(f"    Learning rate: {args.learning_rate}")
    print(f"    Scheduler: {args.scheduler_type}")
    print(f"    Warmup ratio: {args.warmup_ratio}")
    print(f"    Weight decay: {args.weight_decay}")
    print(f"    Max grad norm: {args.max_grad_norm}")
    print(f"    Label smoothing: {args.label_smoothing}")
    print(f"    Class weights: {args.class_weight_method}")
    print(f"    FP16: {args.fp16}")
    print(f"    Early stopping patience: {args.early_stopping_patience}")
    print("-" * 70)

    trainer.train()

    # Save final model
    print("\n12. Saving final model...")
    trainer.save_model(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    print(f"    Model saved to {output_dir / 'final_model'}")

    # Evaluate on test set
    print("\n13. Evaluating on test set...")
    test_dataset = AkteChunkDataset(
        test_docs,
        tokenizer,
        label2id,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )

    test_results = trainer.evaluate(test_dataset)
    print("\n   Test Results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.4f}")

    # Save test results
    results_path = output_dir / "test_results_optimized.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n   Results saved to {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print("\nKey metrics:")
    print(f"  - F1 Micro: {test_results.get('eval_f1_micro', 0):.4f}")
    print(f"  - F1 Macro: {test_results.get('eval_f1_macro', 0):.4f}")
    print(f"  - F1 Weighted: {test_results.get('eval_f1_weighted', 0):.4f}")
    print(f"  - Exact Match: {test_results.get('eval_exact_match_ratio', 0):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
