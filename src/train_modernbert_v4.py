"""
ModernBERT Training Script v4.0 - COMPREHENSIVE
================================================

Major improvements over v3.0:
- Data augmentation for 13 CRITICAL classes (<20 samples)
- Text preprocessing (normalization, cleaning)
- Two-stage training strategy
- Threshold optimization for per-class decision boundaries
- Progressive training (start with balanced, fine-tune on full)

Key data insights addressed:
- 633:1 imbalance ratio (4432 vs 7 samples)
- 13 CRITICAL classes with <20 samples
- 11 LOW classes with <50 samples
- 99% of documents > 512 tokens
- 16% of documents > 8192 tokens

Expected results:
- Accuracy: 90-95% (vs 75% in v1.0, target: 90%)
- ALL 45 classes should have F1 > 0.30
- CRITICAL classes: F1 > 0.40 (was 0.00)

Usage:
------
Single GPU:
    python src/train_modernbert_v4.py

Multi-GPU (8x RTX 5090):
    torchrun --nproc_per_node=8 src/train_modernbert_v4.py

Author: Kadaster AML Team (Saxion 2025/26)
Date: January 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import torch
import torch.distributed as dist
from sklearn.metrics import precision_recall_fscore_support

from model import (
    ModernBERTClassifier,
    ModernBERTConfig,
    test_long_sequence_processing,
)
from src.preprocessing import load_splits, prepare_data_for_classification
from src.data_augmentation import augment_minority_classes, create_balanced_subset
from src.text_preprocessing import preprocess_batch


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfigV4:
    """Enhanced configuration for v4.0 training"""

    # Two-stage training
    USE_TWO_STAGE = True
    STAGE1_EPOCHS = 3  # Train on balanced subset
    STAGE1_LR = 1e-5  # Higher LR for initial learning
    STAGE2_EPOCHS = 5  # Fine-tune on full dataset
    STAGE2_LR = 5e-6  # Lower LR for fine-tuning

    # Data augmentation
    AUGMENT_MINORITY = True
    MIN_SAMPLES_FOR_AUG = 50  # Augment classes with fewer samples
    TARGET_SAMPLES = 100  # Target samples after augmentation

    # Text preprocessing
    PREPROCESS_TEXT = True

    # Threshold optimization
    OPTIMIZE_THRESHOLDS = True

    # Output
    SAVE_DIR = "models/modernbert_v4"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0

def is_main_process():
    return get_rank() == 0

def prepare_label_mapping(train_labels):
    unique_labels = sorted(set(train_labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    if is_main_process():
        print(f"\n  Label mapping: {len(unique_labels)} classes")
        print(f"  Label range: {min(unique_labels)} - {max(unique_labels)}")

    return label_to_id, id_to_label

def map_labels(labels, label_to_id):
    return [label_to_id[label] for label in labels]


def print_class_stats(labels, title="Class Distribution"):
    """Print class distribution statistics"""
    counts = Counter(labels)
    total = len(labels)

    print(f"\n📊 {title}:")
    print(f"  Total samples: {total}")
    print(f"  Number of classes: {len(counts)}")

    # Imbalance analysis
    max_count = max(counts.values())
    min_count = min(counts.values())
    critical = sum(1 for c in counts.values() if c < 20)
    low = sum(1 for c in counts.values() if 20 <= c < 50)

    print(f"  Imbalance ratio: {max_count/min_count:.1f}x")
    print(f"  CRITICAL classes (<20): {critical}")
    print(f"  LOW classes (20-50): {low}")


def optimize_thresholds(model, val_dataset, num_classes, default_threshold=0.5):
    """
    Optimize per-class decision thresholds on validation set.
    This can significantly improve minority class recall.
    """
    if not is_main_process():
        return [default_threshold] * num_classes

    print("\n🎯 Optimizing per-class thresholds...")

    model.model.eval()
    all_probs = []
    all_labels = []

    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels']

            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Optimize threshold per class
    thresholds = []
    for class_idx in range(num_classes):
        best_threshold = default_threshold
        best_f1 = 0

        class_probs = all_probs[:, class_idx]
        class_labels = (all_labels == class_idx).astype(int)

        # Skip if no samples of this class
        if class_labels.sum() == 0:
            thresholds.append(default_threshold)
            continue

        # Search for best threshold
        for threshold in np.arange(0.1, 0.9, 0.05):
            predictions = (class_probs >= threshold).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                class_labels, predictions, average='binary', zero_division=0
            )
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        thresholds.append(best_threshold)

    print(f"  Threshold range: {min(thresholds):.2f} - {max(thresholds):.2f}")
    print(f"  Average threshold: {np.mean(thresholds):.2f}")

    return thresholds


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

    if is_main_process():
        print("=" * 80)
        print("MODERNBERT TRAINING v4.0 - COMPREHENSIVE")
        print("For Kadaster Legal Document Classification")
        print("=" * 80)

        print("\n🚀 v4.0 ENHANCEMENTS:")
        print("  ✅ Data Augmentation for 13 CRITICAL classes")
        print("  ✅ Text Preprocessing (normalization, cleaning)")
        print("  ✅ Two-Stage Training (balanced → full)")
        print("  ✅ Threshold Optimization per class")
        print("  ✅ All v3.0 features (Focal Loss, R-Drop, etc.)")
        print("")

    # ========================================================================
    # STEP 1: Load and preprocess data
    # ========================================================================
    if is_main_process():
        print("\n" + "=" * 80)
        print("STEP 1: Loading and Preprocessing Data")
        print("=" * 80)

    train_df, val_df, test_df = load_splits()

    if is_main_process():
        print(f"  Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Prepare data
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

    if is_main_process():
        print_class_stats(y_train_mapped, "Original Training Data")

    # ========================================================================
    # STEP 2: Text Preprocessing
    # ========================================================================
    if TrainingConfigV4.PREPROCESS_TEXT and is_main_process():
        print("\n" + "=" * 80)
        print("STEP 2: Text Preprocessing")
        print("=" * 80)

        print("  Applying preprocessing pipeline...")
        X_train = preprocess_batch(X_train.tolist())
        X_val = preprocess_batch(X_val.tolist())
        X_test = preprocess_batch(X_test.tolist())

        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)

        print(f"  ✅ Preprocessed {len(X_train)} training documents")

    # ========================================================================
    # STEP 3: Data Augmentation
    # ========================================================================
    if TrainingConfigV4.AUGMENT_MINORITY and is_main_process():
        print("\n" + "=" * 80)
        print("STEP 3: Data Augmentation for Minority Classes")
        print("=" * 80)

        X_train_aug, y_train_aug = augment_minority_classes(
            texts=X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train),
            labels=y_train_mapped,
            min_samples=TrainingConfigV4.MIN_SAMPLES_FOR_AUG,
            target_samples=TrainingConfigV4.TARGET_SAMPLES
        )

        X_train = np.array(X_train_aug)
        y_train_mapped = y_train_aug

        print_class_stats(y_train_mapped, "After Augmentation")

    # ========================================================================
    # STEP 4: Save label mapping
    # ========================================================================
    if is_main_process():
        output_dir = Path(TrainingConfigV4.SAVE_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "label_mapping.pkl", "wb") as f:
            pickle.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f)

    if is_distributed():
        dist.barrier()

    # ========================================================================
    # STEP 5: Initialize Model
    # ========================================================================
    if is_main_process():
        print("\n" + "=" * 80)
        print("STEP 5: Initializing ModernBERT Classifier")
        print("=" * 80)

    classifier = ModernBERTClassifier(
        num_labels=num_labels,
        model_name=ModernBERTConfig.MODEL_NAME,
        max_length=ModernBERTConfig.MAX_LENGTH
    )

    # ========================================================================
    # STEP 6: Create Datasets
    # ========================================================================
    if is_main_process():
        print("\n" + "=" * 80)
        print("STEP 6: Creating Datasets")
        print("=" * 80)

    train_dataset = classifier.create_dataset(
        X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train),
        y_train_mapped
    )
    val_dataset = classifier.create_dataset(X_val.tolist(), y_val_mapped)
    test_dataset = classifier.create_dataset(X_test.tolist(), y_test_mapped)

    if is_main_process():
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

    # ========================================================================
    # STEP 7: Two-Stage Training
    # ========================================================================
    if TrainingConfigV4.USE_TWO_STAGE and is_main_process():
        print("\n" + "=" * 80)
        print("STEP 7: Two-Stage Training")
        print("=" * 80)

        # ----- STAGE 1: Train on balanced subset -----
        print("\n📍 STAGE 1: Training on Balanced Subset")
        print(f"  Epochs: {TrainingConfigV4.STAGE1_EPOCHS}")
        print(f"  Learning Rate: {TrainingConfigV4.STAGE1_LR}")

        # Create balanced subset
        X_balanced, y_balanced = create_balanced_subset(
            X_train.tolist() if hasattr(X_train, 'tolist') else list(X_train),
            y_train_mapped,
            samples_per_class=75  # Moderate balance
        )

        balanced_dataset = classifier.create_dataset(X_balanced, y_balanced)

        # Train stage 1
        trainer_s1 = classifier.train(
            train_dataset=balanced_dataset,
            val_dataset=val_dataset,
            output_dir=TrainingConfigV4.SAVE_DIR + "_stage1",
            num_epochs=TrainingConfigV4.STAGE1_EPOCHS,
            batch_size=ModernBERTConfig.BATCH_SIZE,
            learning_rate=TrainingConfigV4.STAGE1_LR
        )

        print("\n  ✅ Stage 1 complete")

        # ----- STAGE 2: Fine-tune on full dataset -----
        print("\n📍 STAGE 2: Fine-tuning on Full Dataset")
        print(f"  Epochs: {TrainingConfigV4.STAGE2_EPOCHS}")
        print(f"  Learning Rate: {TrainingConfigV4.STAGE2_LR}")

        # Continue training on full dataset
        trainer_s2 = classifier.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=TrainingConfigV4.SAVE_DIR,
            num_epochs=TrainingConfigV4.STAGE2_EPOCHS,
            batch_size=ModernBERTConfig.BATCH_SIZE,
            learning_rate=TrainingConfigV4.STAGE2_LR
        )

        print("\n  ✅ Stage 2 complete")

    else:
        # Standard single-stage training
        if is_main_process():
            print("\n" + "=" * 80)
            print("STEP 7: Training (Single Stage)")
            print("=" * 80)

        trainer = classifier.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=TrainingConfigV4.SAVE_DIR,
            num_epochs=ModernBERTConfig.NUM_EPOCHS,
            batch_size=ModernBERTConfig.BATCH_SIZE,
            learning_rate=ModernBERTConfig.LEARNING_RATE
        )

    # ========================================================================
    # STEP 8: Threshold Optimization
    # ========================================================================
    thresholds = None
    if TrainingConfigV4.OPTIMIZE_THRESHOLDS and is_main_process():
        print("\n" + "=" * 80)
        print("STEP 8: Threshold Optimization")
        print("=" * 80)

        thresholds = optimize_thresholds(classifier, val_dataset, num_labels)

        # Save thresholds
        with open(Path(TrainingConfigV4.SAVE_DIR) / "thresholds.pkl", "wb") as f:
            pickle.dump(thresholds, f)
        print(f"  Thresholds saved to {TrainingConfigV4.SAVE_DIR}/thresholds.pkl")

    # ========================================================================
    # STEP 9: Final Evaluation
    # ========================================================================
    if is_main_process():
        print("\n" + "=" * 80)
        print("STEP 9: Final Evaluation")
        print("=" * 80)

        test_results = classifier.evaluate(
            test_dataset,
            label_mapping_path=str(Path(TrainingConfigV4.SAVE_DIR) / "label_mapping.pkl")
        )

        # Save results
        results_summary = {
            'model': 'ModernBERT-v4.0-Comprehensive',
            'model_name': ModernBERTConfig.MODEL_NAME,
            'num_labels': num_labels,
            'config': {
                'two_stage': TrainingConfigV4.USE_TWO_STAGE,
                'augmentation': TrainingConfigV4.AUGMENT_MINORITY,
                'preprocessing': TrainingConfigV4.PREPROCESS_TEXT,
                'threshold_optimization': TrainingConfigV4.OPTIMIZE_THRESHOLDS,
            },
            'test_accuracy': test_results['accuracy'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'test_f1': test_results['f1'],
            'thresholds': thresholds
        }

        with open(Path(TrainingConfigV4.SAVE_DIR) / "test_results_v4.pkl", "wb") as f:
            pickle.dump(results_summary, f)

        # Final summary
        print("\n" + "=" * 80)
        print("🎉 TRAINING v4.0 COMPLETED!")
        print("=" * 80)

        print(f"\n📊 ModernBERT v4.0 Results:")
        print(f"  Test Accuracy:  {test_results['accuracy']*100:.2f}%")
        print(f"  Test Precision: {test_results['precision']*100:.2f}%")
        print(f"  Test Recall:    {test_results['recall']*100:.2f}%")
        print(f"  Test F1-Score:  {test_results['f1']*100:.2f}%")

        print(f"\n📈 Comparison:")
        print(f"  Baseline (TF-IDF):    81.70%")
        print(f"  ModernBERT v1.0:      75.05%")
        print(f"  ModernBERT v4.0:      {test_results['accuracy']*100:.2f}%")

        if test_results['accuracy'] >= 0.90:
            print(f"\n  ✅ GOAL ACHIEVED! 90% accuracy reached!")
        else:
            gap = 90 - test_results['accuracy']*100
            print(f"\n  Gap to 90% goal: {gap:.2f}%")

        print("\n" + "=" * 80)
        print(f"📁 Output: {TrainingConfigV4.SAVE_DIR}/")
        print("=" * 80)

    # Cleanup
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
