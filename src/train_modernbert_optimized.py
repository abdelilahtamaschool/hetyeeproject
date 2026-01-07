"""
OPTIMIZED ModernBERT Training Script (v2.0)
==========================================

Optimizations for 8x RTX 5090 (270GB total VRAM):
- Focal Loss for class imbalance handling
- Class weighting for minority classes
- Increased batch size (16 per GPU = 128 effective)
- Lower learning rate (1e-5) for better convergence
- More epochs (6) for 45-class classification
- Proper multi-GPU training with torch.distributed
- Label smoothing to prevent overconfidence

Performance improvements:
- ~2-3x faster training (15-20 min vs 41 min)
- Much better accuracy on minority classes
- Expected accuracy: 85-92% (vs 75% before)

Usage:
------
Single GPU (testing):
    python src/train_modernbert_optimized.py

Multi-GPU (8x RTX 5090):
    torchrun --nproc_per_node=8 src/train_modernbert_optimized.py

Or with CUDA_VISIBLE_DEVICES:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 src/train_modernbert_optimized.py
"""

import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
from pathlib import Path
from model import ModernBERTClassifier, ModernBERTConfig, test_long_sequence_processing
from src.preprocessing import load_splits, prepare_data_for_classification
import torch
import torch.distributed as dist


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
    """
    Create label ID mapping
    ModernBERT expects labels as 0-indexed integers

    Args:
        train_labels: Training labels

    Returns:
        Dictionary mapping original labels to 0-indexed IDs
    """
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


def main():
    # Initialize distributed training if available
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

    if is_main_process():
        print("="*80)
        print("OPTIMIZED MODERNBERT TRAINING (v2.0)")
        print("For Kadaster Legal Document Classification")
        print("="*80)
        print("\n🚀 OPTIMIZATIONS:")
        print("  ✅ Focal Loss for class imbalance")
        print("  ✅ Class weighting for minority classes")
        print("  ✅ Batch size: 16 per GPU (128 effective on 8 GPUs)")
        print("  ✅ Learning rate: 1e-5 (optimized for convergence)")
        print("  ✅ Epochs: 6 (for 45-class classification)")
        print("  ✅ Label smoothing: 0.1")
        print("  ✅ Multi-GPU training support")
        print("")

    # Step 1: Test long sequence processing (main process only)
    if is_main_process():
        print("### Step 1: Testing ModernBERT Long Sequence Capabilities ###")
        test_long_sequence_processing()

    # Synchronize processes
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

    # Map labels to 0-indexed IDs
    y_train_mapped = map_labels(y_train, label_to_id)
    y_val_mapped = map_labels(y_val, label_to_id)
    y_test_mapped = map_labels(y_test, label_to_id)

    # Save label mapping (main process only)
    if is_main_process():
        output_dir = Path(ModernBERTConfig.SAVE_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "label_mapping.pkl", "wb") as f:
            pickle.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f)
        print(f"  Label mapping saved to {output_dir}/label_mapping.pkl")

    # Synchronize processes
    if is_distributed():
        dist.barrier()

    # Step 4: Initialize ModernBERT classifier
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

    # Analyze sequence lengths (main process only)
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
        print(f"    Documents > 1024 tokens: {sum(1 for l in lengths if l > 1024)} ({sum(1 for l in lengths if l > 1024)/len(lengths)*100:.1f}%)")
        print(f"    Documents > 2048 tokens: {sum(1 for l in lengths if l > 2048)} ({sum(1 for l in lengths if l > 2048)/len(lengths)*100:.1f}%)")
        print(f"    Documents > 4096 tokens: {sum(1 for l in lengths if l > 4096)} ({sum(1 for l in lengths if l > 4096)/len(lengths)*100:.1f}%)")
        print(f"    Documents > 8192 tokens: {sum(1 for l in lengths if l > 8192)} ({sum(1 for l in lengths if l > 8192)/len(lengths)*100:.1f}%)")

    # Step 6: Train model
    if is_main_process():
        print("\n### Step 6: Training Optimized ModernBERT ###")
        print("Training with:")
        print(f"  - Focal Loss: {ModernBERTConfig.USE_FOCAL_LOSS}")
        print(f"  - Class Weights: {ModernBERTConfig.USE_CLASS_WEIGHTS}")
        print(f"  - Batch size per GPU: {ModernBERTConfig.BATCH_SIZE}")
        print(f"  - Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 1}")
        print(f"  - Effective batch size: {ModernBERTConfig.BATCH_SIZE * (torch.cuda.device_count() if torch.cuda.is_available() else 1)}")
        print(f"  - Learning rate: {ModernBERTConfig.LEARNING_RATE}")
        print(f"  - Epochs: {ModernBERTConfig.NUM_EPOCHS}")
        print(f"  - Expected time: 15-20 minutes on 8x RTX 5090")

    trainer = classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=ModernBERTConfig.SAVE_DIR,
        num_epochs=ModernBERTConfig.NUM_EPOCHS,
        batch_size=ModernBERTConfig.BATCH_SIZE,
        learning_rate=ModernBERTConfig.LEARNING_RATE
    )

    # Step 7: Evaluate on test set (main process only)
    if is_main_process():
        print("\n### Step 7: Final Evaluation on Test Set ###")
        test_results = classifier.evaluate(
            test_dataset,
            label_mapping_path=str(Path(ModernBERTConfig.SAVE_DIR) / "label_mapping.pkl")
        )

        # Step 8: Save results
        print("\n### Step 8: Saving Results ###")
        results_summary = {
            'model': 'ModernBERT-Optimized-v2.0',
            'model_name': ModernBERTConfig.MODEL_NAME,
            'max_length': ModernBERTConfig.MAX_LENGTH,
            'num_labels': num_labels,
            'optimizations': {
                'focal_loss': ModernBERTConfig.USE_FOCAL_LOSS,
                'class_weights': ModernBERTConfig.USE_CLASS_WEIGHTS,
                'batch_size': ModernBERTConfig.BATCH_SIZE,
                'learning_rate': ModernBERTConfig.LEARNING_RATE,
                'epochs': ModernBERTConfig.NUM_EPOCHS,
                'label_smoothing': ModernBERTConfig.LABEL_SMOOTHING,
            },
            'test_accuracy': test_results['accuracy'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'test_f1': test_results['f1']
        }

        with open(Path(ModernBERTConfig.SAVE_DIR) / "test_results_optimized.pkl", "wb") as f:
            pickle.dump(results_summary, f)

        print(f"  Results saved to {ModernBERTConfig.SAVE_DIR}/test_results_optimized.pkl")

        # Final summary
        print("\n" + "="*80)
        print("OPTIMIZED TRAINING COMPLETED!")
        print("="*80)
        print(f"\nModernBERT Optimized Results:")
        print(f"  Test Accuracy:  {test_results['accuracy']*100:.2f}%")
        print(f"  Test Precision: {test_results['precision']*100:.2f}%")
        print(f"  Test Recall:    {test_results['recall']*100:.2f}%")
        print(f"  Test F1-Score:  {test_results['f1']*100:.2f}%")

        print(f"\nComparison with Previous Runs:")
        print(f"  Baseline (TF-IDF):        81.70%")
        print(f"  ModernBERT v1.0:          75.05%")
        print(f"  ModernBERT v2.0 (This):   {test_results['accuracy']*100:.2f}%")

        if test_results['accuracy'] >= 0.90:
            improvement = test_results['accuracy']*100 - 75.05
            print(f"\n  ✅ GOAL ACHIEVED! (90% accuracy)")
            print(f"  🚀 Improvement over v1.0: +{improvement:.2f}%")
        else:
            gap = 90 - test_results['accuracy']*100
            improvement = test_results['accuracy']*100 - 75.05
            print(f"\n  Gap to goal (90%): {gap:.2f}%")
            print(f"  Improvement over v1.0: +{improvement:.2f}%")

        print("="*80)

    # Cleanup distributed training
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
