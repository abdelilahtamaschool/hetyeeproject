"""
Training script for ModernBERT model
Replaces baseline TF-IDF + Logistic Regression with ModernBERT

Features:
- 8192 token context window (vs 512 for standard BERT)
- GPU acceleration for RTX 4070
- Flash Attention 2 support
- Handles long legal documents
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

    print(f"\nLabel mapping created:")
    print(f"  Number of unique labels: {len(unique_labels)}")
    print(f"  Label range: {min(unique_labels)} - {max(unique_labels)}")

    return label_to_id, id_to_label


def map_labels(labels, label_to_id):
    """Map labels to 0-indexed IDs"""
    return [label_to_id[label] for label in labels]


def main():
    print("="*80)
    print("MODERNBERT TRAINING FOR KADASTER LEGAL DOCUMENT CLASSIFICATION")
    print("="*80)

    # Step 1: Test long sequence processing
    print("\n### Step 1: Testing ModernBERT Long Sequence Capabilities ###")
    test_long_sequence_processing()

    # Step 2: Load data splits
    print("\n### Step 2: Loading Data Splits ###")
    train_df, val_df, test_df = load_splits()

    print(f"  Train set: {len(train_df)} documents")
    print(f"  Validation set: {len(val_df)} documents")
    print(f"  Test set: {len(test_df)} documents")

    # Step 3: Prepare data
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

    # Save label mapping
    Path("models/modernbert").mkdir(parents=True, exist_ok=True)
    with open("models/modernbert/label_mapping.pkl", "wb") as f:
        pickle.dump({'label_to_id': label_to_id, 'id_to_label': id_to_label}, f)
    print("  Label mapping saved to models/modernbert/label_mapping.pkl")

    # Step 4: Initialize ModernBERT classifier
    print("\n### Step 4: Initializing ModernBERT Classifier ###")
    classifier = ModernBERTClassifier(
        num_labels=num_labels,
        model_name=ModernBERTConfig.MODEL_NAME,
        max_length=ModernBERTConfig.MAX_LENGTH
    )

    # Step 5: Create datasets
    print("\n### Step 5: Creating Datasets ###")
    print("Converting texts to ModernBERT format...")
    train_dataset = classifier.create_dataset(X_train.tolist(), y_train_mapped)
    val_dataset = classifier.create_dataset(X_val.tolist(), y_val_mapped)
    test_dataset = classifier.create_dataset(X_test.tolist(), y_test_mapped)

    print(f"  Train dataset: {len(train_dataset)} samples")
    print(f"  Validation dataset: {len(val_dataset)} samples")
    print(f"  Test dataset: {len(test_dataset)} samples")

    # Analyze sequence lengths
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
    print("\n### Step 6: Training ModernBERT ###")
    print("This may take several hours on RTX 4070...")

    trainer = classifier.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir="models/modernbert",
        num_epochs=ModernBERTConfig.NUM_EPOCHS,
        batch_size=ModernBERTConfig.BATCH_SIZE,
        learning_rate=ModernBERTConfig.LEARNING_RATE
    )

    # Step 7: Evaluate on test set
    print("\n### Step 7: Final Evaluation on Test Set ###")
    test_results = classifier.evaluate(test_dataset, label_mapping_path="models/modernbert/label_mapping.pkl")

    # Step 8: Save results
    print("\n### Step 8: Saving Results ###")
    results_summary = {
        'model': 'ModernBERT',
        'model_name': ModernBERTConfig.MODEL_NAME,
        'max_length': ModernBERTConfig.MAX_LENGTH,
        'num_labels': num_labels,
        'test_accuracy': test_results['accuracy'],
        'test_precision': test_results['precision'],
        'test_recall': test_results['recall'],
        'test_f1': test_results['f1']
    }

    with open("models/modernbert/test_results.pkl", "wb") as f:
        pickle.dump(results_summary, f)

    print("  Results saved to models/modernbert/test_results.pkl")

    # Final summary
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"\nModernBERT Results:")
    print(f"  Test Accuracy:  {test_results['accuracy']*100:.2f}%")
    print(f"  Test Precision: {test_results['precision']*100:.2f}%")
    print(f"  Test Recall:    {test_results['recall']*100:.2f}%")
    print(f"  Test F1-Score:  {test_results['f1']*100:.2f}%")

    print(f"\nComparison with Baseline (TF-IDF + Logistic Regression):")
    print(f"  Baseline Accuracy: 81.70%")
    print(f"  ModernBERT Accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"  Improvement: {test_results['accuracy']*100 - 81.70:.2f}%")

    print(f"\nGoal: 90% accuracy")
    if test_results['accuracy'] >= 0.90:
        print(f"  ✓ GOAL ACHIEVED!")
    else:
        print(f"  Gap to goal: {90 - test_results['accuracy']*100:.2f}%")

    print("="*80)


if __name__ == "__main__":
    main()
