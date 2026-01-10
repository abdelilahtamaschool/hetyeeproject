"""
ModernBERT Training Script v3.0 - ENHANCED
==========================================

Major improvements over v2.0:
- Combined Focal Loss + Class Weights (not OR, but AND)
- R-Drop Regularization for better generalization
- Improved hyperparameters (LR 8e-6, 8 epochs, warmup ratio 0.1)
- Tighter gradient clipping (0.5 instead of 1.0)
- Cosine with restarts scheduler
- Sqrt-scaled class weights (less extreme)
- More frequent evaluation for better model selection

Expected results:
- Accuracy: 88-93% (vs 75% in v1.0, target: 90%)
- All 45 classes should have F1 > 0.40
- Minority classes: F1 > 0.30 (was 0.00)

Usage:
------
Single GPU:
    python src/train_modernbert_v3.py

Multi-GPU (8x RTX 5090):
    torchrun --nproc_per_node=8 src/train_modernbert_v3.py

Author: Kadaster AML Team (Saxion 2025/26)
Date: January 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from pathlib import Path
from model import (
    ModernBERTClassifier,
    ModernBERTConfig,
    test_long_sequence_processing,
    create_class_balanced_sampler
)
from src.preprocessing import load_splits, prepare_data_for_classification
import torch
import torch.distributed as dist
from collections import Counter


def is_distributed():
    """Check if running in distributed mode"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get current process rank"""
    if is_distributed():
        return dist.get_rank()
    return 0


def is_main_process():
    """Check if this is the main process (rank 0)"""
    return get_rank() == 0


def prepare_label_mapping(train_labels):
    """Create label ID mapping"""
    unique_labels = sorted(set(train_labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    if is_main_process():
        print(f"\nLabel mapping created:")
        print(f"  Number of unique labels: {len(unique_labels)}")
        print(f"  Label range: {min(unique_labels)} - {max(unique_labels)}")

    return label_to_id, id_to_label


def map_labels(labels, label_to_id):
    """Map labels to 0-indexed IDs"""
    return [label_to_id[label] for label in labels]


def print_class_distribution(labels, id_to_label=None):
    """Print class distribution for analysis"""
    counts = Counter(labels)
    total = len(labels)

    print(f"\n📊 Class Distribution Analysis:")
    print(f"  Total samples: {total}")
    print(f"  Number of classes: {len(counts)}")
    print(f"  ")

    # Sort by count
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print(f"  Top 10 classes:")
    for label_id, count in sorted_counts[:10]:
        pct = count / total * 100
        label_name = id_to_label.get(label_id, label_id) if id_to_label else label_id
        print(f"    {label_name}: {count} samples ({pct:.1f}%)")

    print(f"  ")
    print(f"  Bottom 10 classes:")
    for label_id, count in sorted_counts[-10:]:
        pct = count / total * 100
        label_name = id_to_label.get(label_id, label_id) if id_to_label else label_id
        print(f"    {label_name}: {count} samples ({pct:.1f}%)")

    # Class imbalance metrics
    max_count = max(counts.values())
    min_count = min(counts.values())
    imbalance_ratio = max_count / min_count

    print(f"  ")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
    print(f"  Classes with < 10 samples: {sum(1 for c in counts.values() if c < 10)}")
    print(f"  Classes with < 50 samples: {sum(1 for c in counts.values() if c < 50)}")


def main():
    # Initialize distributed training if available
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

    if is_main_process():
        print("="*80)
        print("MODERNBERT TRAINING v3.0 - ENHANCED")
        print("For Kadaster Legal Document Classification")
        print("="*80)

        print("\n🚀 v3.0 ENHANCEMENTS:")
        print("  ✅ Combined Focal Loss + Class Weights")
        print("  ✅ R-Drop Regularization (alpha=0.7)")
        print("  ✅ Sqrt-scaled class weights (less extreme)")
        print("  ✅ Lower learning rate (8e-6)")
        print("  ✅ More epochs (8)")
        print("  ✅ Warmup ratio 10%")
        print("  ✅ Cosine with restarts scheduler")
        print("  ✅ Tighter gradient clipping (0.5)")
        print("  ✅ More frequent evaluation (every 50 steps)")
        print("")

    # Step 1: Test long sequence processing
    if is_main_process():
        print("\n### Step 1: Testing ModernBERT Long Sequence Capabilities ###")
        test_long_sequence_processing()

    if is_distributed():
        dist.barrier()

    # Step 2: Load data splits
    if is_main_process():
        print("\n### Step 2: Loading Data Splits ###")

    train_df, val_df, test_df = load_splits()

    if is_main_process():
        print(f"  Train set: {len(train_df)} documents")
        print(f"  Validation set: {len(val_df)} documents")
        print(f"  Test set: {len(test_df)} documents")

    # Step 3: Prepare data
    if is_main_process():
        print("\n### Step 3: Preparing Data ###")

    X_train, y_train = prepare_data_for_classification(train_df, use_primary_label=True)
    X_val, y_val = prepare_data_for_classification(val_df, use_primary_label=True)
    X_test, y_test = prepare_data_for_classification(test_df, use_primary_label=True)

    # Create label mapping
    label_to_id, id_to_label = prepare_label_mapping(y_train)
    num_labels = len(label_to_id)

    # Map labels
    y_train_mapped = map_labels(y_train, label_to_id)
    y_val_mapped = map_labels(y_val, label_to_id)
    y_test_mapped = map_labels(y_test, label_to_id)

    # Print class distribution
    if is_main_process():
        print_class_distribution(y_train_mapped, id_to_label)

    # Save label mapping
    if is_main_process():
        output_dir = Path(ModernBERTConfig.SAVE_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "label_mapping.pkl", "wb") as f:
            pickle.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f)
        print(f"\n  Label mapping saved to {output_dir}/label_mapping.pkl")

    if is_distributed():
        dist.barrier()

    # Step 4: Initialize classifier
    if is_main_process():
        print("\n### Step 4: Initializing ModernBERT Classifier ###")

    classifier = ModernBERTClassifier(
        num_labels=num_labels,
        model_name=ModernBERTConfig.MODEL_NAME,
        max_length=ModernBERTConfig.MAX_LENGTH
    )

    # Step 5: Create datasets
    if is_main_process():
        print("\n### Step 5: Creating Datasets ###")
        print("Converting texts to ModernBERT format...")

    train_dataset = classifier.create_dataset(X_train.tolist(), y_train_mapped)
    val_dataset = classifier.create_dataset(X_val.tolist(), y_val_mapped)
    test_dataset = classifier.create_dataset(X_test.tolist(), y_test_mapped)

    if is_main_process():
        print(f"  Train dataset: {len(train_dataset)} samples")
        print(f"  Validation dataset: {len(val_dataset)} samples")
        print(f"  Test dataset: {len(test_dataset)} samples")

    # Analyze document lengths
    if is_main_process():
        print("\n### Analyzing Document Lengths ###")
        lengths = []
        for text in X_train[:100].tolist():
            encoding = classifier.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True
            )
            lengths.append(len(encoding['input_ids']))

        print(f"  Sample of 100 documents:")
        print(f"    Average token count: {sum(lengths)/len(lengths):.0f}")
        print(f"    Max token count: {max(lengths)}")
        print(f"    Min token count: {min(lengths)}")
        print(f"    Documents > 512 tokens: {sum(1 for l in lengths if l > 512)} ({sum(1 for l in lengths if l > 512)/len(lengths)*100:.1f}%)")
        print(f"    Documents > 2048 tokens: {sum(1 for l in lengths if l > 2048)} ({sum(1 for l in lengths if l > 2048)/len(lengths)*100:.1f}%)")
        print(f"    Documents > 4096 tokens: {sum(1 for l in lengths if l > 4096)} ({sum(1 for l in lengths if l > 4096)/len(lengths)*100:.1f}%)")
        print(f"    Documents > 8192 tokens: {sum(1 for l in lengths if l > 8192)} ({sum(1 for l in lengths if l > 8192)/len(lengths)*100:.1f}%)")

    # Step 6: Train model
    if is_main_process():
        print("\n### Step 6: Training Enhanced ModernBERT v3.0 ###")

    trainer = classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=ModernBERTConfig.SAVE_DIR,
        num_epochs=ModernBERTConfig.NUM_EPOCHS,
        batch_size=ModernBERTConfig.BATCH_SIZE,
        learning_rate=ModernBERTConfig.LEARNING_RATE
    )

    # Step 7: Final evaluation
    if is_main_process():
        print("\n### Step 7: Final Evaluation on Test Set ###")
        test_results = classifier.evaluate(
            test_dataset,
            label_mapping_path=str(Path(ModernBERTConfig.SAVE_DIR) / "label_mapping.pkl")
        )

        # Step 8: Save results
        print("\n### Step 8: Saving Results ###")
        results_summary = {
            'model': 'ModernBERT-v3.0-Enhanced',
            'model_name': ModernBERTConfig.MODEL_NAME,
            'max_length': ModernBERTConfig.MAX_LENGTH,
            'num_labels': num_labels,
            'optimizations': {
                'focal_loss': ModernBERTConfig.USE_FOCAL_LOSS,
                'focal_alpha': ModernBERTConfig.FOCAL_ALPHA,
                'focal_gamma': ModernBERTConfig.FOCAL_GAMMA,
                'class_weights': ModernBERTConfig.USE_CLASS_WEIGHTS,
                'combined_focal_weights': ModernBERTConfig.COMBINE_FOCAL_AND_WEIGHTS,
                'rdrop': ModernBERTConfig.USE_RDROP,
                'rdrop_alpha': ModernBERTConfig.RDROP_ALPHA,
                'batch_size': ModernBERTConfig.BATCH_SIZE,
                'gradient_accumulation': ModernBERTConfig.GRADIENT_ACCUMULATION_STEPS,
                'learning_rate': ModernBERTConfig.LEARNING_RATE,
                'epochs': ModernBERTConfig.NUM_EPOCHS,
                'warmup_ratio': ModernBERTConfig.WARMUP_RATIO,
                'label_smoothing': ModernBERTConfig.LABEL_SMOOTHING,
                'weight_decay': ModernBERTConfig.WEIGHT_DECAY,
            },
            'test_accuracy': test_results['accuracy'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'test_f1': test_results['f1']
        }

        with open(Path(ModernBERTConfig.SAVE_DIR) / "test_results_v3.pkl", "wb") as f:
            pickle.dump(results_summary, f)

        print(f"  Results saved to {ModernBERTConfig.SAVE_DIR}/test_results_v3.pkl")

        # Final summary
        print("\n" + "="*80)
        print("🎉 ENHANCED TRAINING v3.0 COMPLETED!")
        print("="*80)
        print(f"\nModernBERT v3.0 Results:")
        print(f"  Test Accuracy:  {test_results['accuracy']*100:.2f}%")
        print(f"  Test Precision: {test_results['precision']*100:.2f}%")
        print(f"  Test Recall:    {test_results['recall']*100:.2f}%")
        print(f"  Test F1-Score:  {test_results['f1']*100:.2f}%")

        print(f"\n📊 Comparison with Previous Runs:")
        print(f"  Baseline (TF-IDF):        81.70%")
        print(f"  ModernBERT v1.0:          75.05%")
        print(f"  ModernBERT v2.0:          (not run)")
        print(f"  ModernBERT v3.0 (This):   {test_results['accuracy']*100:.2f}%")

        if test_results['accuracy'] >= 0.90:
            improvement = test_results['accuracy']*100 - 75.05
            print(f"\n  ✅ GOAL ACHIEVED! (90% accuracy)")
            print(f"  🚀 Improvement over v1.0: +{improvement:.2f}%")
        elif test_results['accuracy'] >= 0.85:
            improvement = test_results['accuracy']*100 - 75.05
            gap = 90 - test_results['accuracy']*100
            print(f"\n  🔶 GOOD PROGRESS! ({test_results['accuracy']*100:.1f}% accuracy)")
            print(f"  🚀 Improvement over v1.0: +{improvement:.2f}%")
            print(f"  📈 Gap to 90% goal: {gap:.2f}%")
        else:
            gap = 90 - test_results['accuracy']*100
            improvement = test_results['accuracy']*100 - 75.05
            print(f"\n  Gap to goal (90%): {gap:.2f}%")
            print(f"  Improvement over v1.0: +{improvement:.2f}%")

        print("\n" + "="*80)
        print("📁 Output files:")
        print(f"  Model: {ModernBERTConfig.SAVE_DIR}/")
        print(f"  Results: {ModernBERTConfig.SAVE_DIR}/test_results_v3.pkl")
        print(f"  Labels: {ModernBERTConfig.SAVE_DIR}/label_mapping.pkl")
        print("="*80)

    # Cleanup
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
