"""
Multi-label evaluation metrics for the Akte Classification Pipeline.
"""
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    hamming_loss,
    classification_report,
    multilabel_confusion_matrix
)


def compute_multi_label_metrics(
    eval_pred,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive multi-label classification metrics.

    Args:
        eval_pred: Tuple of (logits, labels) from HuggingFace Trainer
        threshold: Classification threshold

    Returns:
        Dictionary of metric names and values
    """
    logits, labels = eval_pred

    # Convert logits to probabilities and predictions
    if isinstance(logits, torch.Tensor):
        probabilities = torch.sigmoid(logits).numpy()
    else:
        probabilities = 1 / (1 + np.exp(-logits))  # sigmoid

    predictions = (probabilities > threshold).astype(int)

    # Ensure labels are numpy arrays
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    labels = labels.astype(int)

    metrics = {
        # F1 scores
        "f1_micro": f1_score(labels, predictions, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "f1_samples": f1_score(labels, predictions, average="samples", zero_division=0),

        # Precision scores
        "precision_micro": precision_score(labels, predictions, average="micro", zero_division=0),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),

        # Recall scores
        "recall_micro": recall_score(labels, predictions, average="micro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),

        # Multi-label specific metrics
        "hamming_loss": hamming_loss(labels, predictions),
        "exact_match_ratio": accuracy_score(labels, predictions),  # All labels correct

        # Subset accuracy (at least one correct)
        "subset_accuracy": np.mean(np.any(predictions & labels, axis=1))
    }

    return metrics


def find_optimal_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray = None,
    metric: str = "f1_micro"
) -> Tuple[float, float]:
    """
    Find the optimal classification threshold.

    Args:
        logits: Model output logits
        labels: Ground truth labels
        thresholds: Array of thresholds to try
        metric: Metric to optimize

    Returns:
        Tuple of (best_threshold, best_score)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05)

    # Convert logits to probabilities
    if isinstance(logits, torch.Tensor):
        probabilities = torch.sigmoid(logits).numpy()
    else:
        probabilities = 1 / (1 + np.exp(-logits))

    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        predictions = (probabilities > threshold).astype(int)

        if metric == "f1_micro":
            score = f1_score(labels, predictions, average="micro", zero_division=0)
        elif metric == "f1_macro":
            score = f1_score(labels, predictions, average="macro", zero_division=0)
        elif metric == "exact_match":
            score = accuracy_score(labels, predictions)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def get_per_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    id2label: Dict[int, int]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each class individually.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
        id2label: Mapping from index to rechtsfeitcode

    Returns:
        Dictionary with metrics per class
    """
    per_class = {}

    for idx, code in id2label.items():
        y_true = labels[:, idx]
        y_pred = predictions[:, idx]

        per_class[str(code)] = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "support": int(y_true.sum())
        }

    return per_class


def get_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    id2label: Dict[int, int]
) -> str:
    """
    Generate a formatted classification report.

    Args:
        labels: Ground truth labels
        predictions: Model predictions
        id2label: Mapping from index to rechtsfeitcode

    Returns:
        Formatted report string
    """
    target_names = [str(id2label[i]) for i in range(len(id2label))]

    return classification_report(
        labels,
        predictions,
        target_names=target_names,
        zero_division=0
    )


def calculate_class_weights(
    label_counts: np.ndarray,
    total_samples: int,
    strategy: str = "inverse"
) -> torch.Tensor:
    """
    Calculate class weights for handling imbalanced data.

    Args:
        label_counts: Array of positive counts per class
        total_samples: Total number of samples
        strategy: Weighting strategy ('inverse', 'sqrt_inverse', 'effective')

    Returns:
        Tensor of pos_weights for BCEWithLogitsLoss
    """
    neg_counts = total_samples - label_counts

    if strategy == "inverse":
        # Simple inverse frequency
        pos_weight = neg_counts / (label_counts + 1e-6)

    elif strategy == "sqrt_inverse":
        # Square root dampened inverse
        pos_weight = np.sqrt(neg_counts / (label_counts + 1e-6))

    elif strategy == "effective":
        # Effective number of samples (from Class-Balanced Loss paper)
        beta = 0.999
        effective_num = 1.0 - np.power(beta, label_counts)
        pos_weight = (1.0 - beta) / (effective_num + 1e-6)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Clip extreme values
    pos_weight = np.clip(pos_weight, 0.1, 100)

    return torch.tensor(pos_weight, dtype=torch.float32)


def aggregate_chunk_predictions(
    chunk_predictions: List[np.ndarray],
    chunk_doc_ids: List[int],
    num_documents: int,
    aggregation: str = "max"
) -> np.ndarray:
    """
    Aggregate chunk-level predictions to document level.

    Args:
        chunk_predictions: List of prediction arrays per chunk
        chunk_doc_ids: Document ID for each chunk
        num_documents: Total number of documents
        aggregation: Aggregation strategy ('max', 'mean', 'any')

    Returns:
        Document-level predictions array
    """
    num_labels = chunk_predictions[0].shape[-1] if len(chunk_predictions[0].shape) > 1 else chunk_predictions[0].shape[0]

    # Initialize document predictions
    doc_predictions = [[] for _ in range(num_documents)]

    # Group chunks by document
    for chunk_pred, doc_id in zip(chunk_predictions, chunk_doc_ids):
        doc_predictions[doc_id].append(chunk_pred)

    # Aggregate
    aggregated = np.zeros((num_documents, num_labels))

    for doc_id, preds in enumerate(doc_predictions):
        if not preds:
            continue

        stacked = np.stack(preds)

        if aggregation == "max":
            aggregated[doc_id] = stacked.max(axis=0)
        elif aggregation == "mean":
            aggregated[doc_id] = stacked.mean(axis=0)
        elif aggregation == "any":
            aggregated[doc_id] = (stacked.sum(axis=0) > 0).astype(float)

    return aggregated
