"""
Main training script for the Akte Classification Pipeline.

Usage:
    python train.py --data_path <path_to_jsonl> --output_dir <output_dir>
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.config import Config
from src.data.loader import (
    load_jsonl,
    get_top_n_labels,
    get_label_mapping,
    filter_documents_by_labels,
    analyze_label_distribution,
    get_dataset_statistics
)
from src.data.dataset import AkteChunkDataset
from src.models.classifier import SimpleAkteClassifier
from src.training.trainer import create_trainer
from src.training.metrics import find_optimal_threshold, calculate_class_weights


def compute_label_counts(documents: list, label2id: dict) -> np.ndarray:
    """
    Count occurrences of each label in the dataset.

    Args:
        documents: List of documents with rechtsfeitcodes
        label2id: Mapping from label to index

    Returns:
        Array of counts per label
    """
    num_labels = len(label2id)
    counts = np.zeros(num_labels)

    for doc in documents:
        codes = doc.get('rechtsfeitcodes', [])
        if isinstance(codes, str):
            codes = [codes]
        for code in codes:
            if code in label2id:
                counts[label2id[code]] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Train Akte Classifier")
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
        default=64,
        help="Training batch size (H200 can handle 64+)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (2e-5 is standard for BERT)"
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
        help="Chunk size in tokens (default: 510)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Overlap between chunks (default: 128)"
    )

    args = parser.parse_args()

    # Initialize config
    config = Config()
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    config.data.top_n_labels = args.num_labels
    config.chunking.chunk_size = args.chunk_size
    config.chunking.overlap = args.overlap

    print("=" * 60)
    print("Akte Classification Pipeline - Training")
    print("=" * 60)

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

    # Split data
    print("\n5. Splitting data (80/10/10)...")
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
    print(f"\n6. Loading tokenizer ({config.model.model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    # Create datasets
    print("\n7. Creating datasets with chunking...")
    print(f"   Chunk size: {config.chunking.chunk_size}")
    print(f"   Overlap: {config.chunking.overlap}")

    train_dataset = AkteChunkDataset(
        train_docs,
        tokenizer,
        label2id,
        chunk_size=config.chunking.chunk_size,
        overlap=config.chunking.overlap
    )
    print(f"   Train chunks: {len(train_dataset)}")

    val_dataset = AkteChunkDataset(
        val_docs,
        tokenizer,
        label2id,
        chunk_size=config.chunking.chunk_size,
        overlap=config.chunking.overlap
    )
    print(f"   Val chunks: {len(val_dataset)}")

    # Initialize model
    print(f"\n8. Initializing model...")
    model = SimpleAkteClassifier(
        model_name=config.model.model_name,
        num_labels=len(label2id)
    )
    print(f"   Model: {config.model.model_name}")
    print(f"   Num labels: {len(label2id)}")

    # Compute class weights for imbalanced data
    print("\n9. Computing class weights...")
    label_counts = compute_label_counts(train_docs, label2id)
    total_samples = len(train_docs)

    print(f"   Label distribution:")
    for label, idx in sorted(label2id.items(), key=lambda x: label_counts[x[1]], reverse=True):
        count = int(label_counts[idx])
        pct = (count / total_samples) * 100
        print(f"   - {label}: {count} ({pct:.1f}%)")

    # Calculate class weights using effective number strategy
    class_weights = calculate_class_weights(label_counts, total_samples, strategy="sqrt_inverse")
    print(f"\n   Class weights computed (sqrt_inverse strategy)")
    print(f"   Weight range: {class_weights.min():.2f} - {class_weights.max():.2f}")

    # Create trainer
    print("\n10. Setting up trainer...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = create_trainer(
        model,
        train_dataset,
        val_dataset,
        config.training,
        str(output_dir / "checkpoints"),
        class_weights=class_weights
    )

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
    print(f"    Epochs: {config.training.num_epochs}")
    print(f"    Batch size: {config.training.batch_size}")
    print(f"    Effective batch: {config.training.batch_size * config.training.gradient_accumulation_steps}")
    print(f"    Learning rate: {config.training.learning_rate}")
    print(f"    FP16: {config.training.fp16}")
    print(f"    Focal Loss: {config.training.use_focal_loss} (gamma={config.training.focal_gamma})")
    print(f"    Class Weights: {config.training.use_class_weights}")
    print(f"    Best model metric: {config.training.metric_for_best_model}")
    print("-" * 60)

    trainer.train()

    # Save final model
    print("\n12. Saving final model...")
    trainer.save_model(str(output_dir / "final_model"))
    print(f"    Model saved to {output_dir / 'final_model'}")

    # Evaluate on test set
    print("\n13. Evaluating on test set...")
    test_dataset = AkteChunkDataset(
        test_docs,
        tokenizer,
        label2id,
        chunk_size=config.chunking.chunk_size,
        overlap=config.chunking.overlap
    )

    test_results = trainer.evaluate(test_dataset)
    print("\n   Test Results:")
    for key, value in test_results.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.4f}")

    # Save test results
    results_path = output_dir / "test_results.json"
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"\n   Results saved to {results_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
