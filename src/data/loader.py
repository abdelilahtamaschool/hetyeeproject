"""
Data loading utilities for the Akte Classification Pipeline.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import pandas as pd
from tqdm import tqdm


def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict]:
    """
    Load JSONL file with akte documents.

    Args:
        path: Path to the JSONL file
        limit: Optional limit on number of documents to load

    Returns:
        List of document dictionaries with keys: akteId, text, rechtsfeitcodes
    """
    documents = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Loading documents")):
            if limit and i >= limit:
                break
            doc = json.loads(line.strip())
            documents.append(doc)
    return documents


def analyze_label_distribution(documents: List[Dict]) -> pd.DataFrame:
    """
    Analyze the distribution of rechtsfeitcodes in the dataset.

    Args:
        documents: List of document dictionaries

    Returns:
        DataFrame with label statistics
    """
    all_codes = []
    for doc in documents:
        all_codes.extend(doc.get('rechtsfeitcodes', []))

    counter = Counter(all_codes)

    df = pd.DataFrame([
        {'rechtsfeitcode': code, 'count': count}
        for code, count in counter.most_common()
    ])

    df['percentage'] = df['count'] / len(documents) * 100
    df['cumulative_percentage'] = df['percentage'].cumsum()

    return df


def get_top_n_labels(documents: List[Dict], n: int = 20) -> List[int]:
    """
    Get the top N most frequent rechtsfeitcodes.

    Args:
        documents: List of document dictionaries
        n: Number of top labels to return

    Returns:
        List of top N rechtsfeitcodes sorted by frequency
    """
    all_codes = []
    for doc in documents:
        all_codes.extend(doc.get('rechtsfeitcodes', []))

    counter = Counter(all_codes)
    return [code for code, _ in counter.most_common(n)]


def get_label_mapping(labels: List[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create bidirectional mapping between rechtsfeitcodes and indices.

    Args:
        labels: List of unique rechtsfeitcodes

    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    label2id = {label: idx for idx, label in enumerate(sorted(labels))}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def filter_documents_by_labels(
    documents: List[Dict],
    valid_labels: List[int]
) -> List[Dict]:
    """
    Filter documents to only include those with at least one valid label.
    Also filter out invalid labels from rechtsfeitcodes.

    Args:
        documents: List of document dictionaries
        valid_labels: List of valid rechtsfeitcodes to keep

    Returns:
        Filtered list of documents
    """
    valid_set = set(valid_labels)
    filtered_docs = []

    for doc in documents:
        # Filter rechtsfeitcodes to only valid ones
        valid_codes = [c for c in doc.get('rechtsfeitcodes', []) if c in valid_set]

        if valid_codes:  # Only keep documents with at least one valid code
            filtered_doc = doc.copy()
            filtered_doc['rechtsfeitcodes'] = valid_codes
            filtered_docs.append(filtered_doc)

    return filtered_docs


def create_label_matrix(
    documents: List[Dict],
    label2id: Dict[int, int]
) -> 'np.ndarray':
    """
    Create a multi-hot label matrix for all documents.

    Args:
        documents: List of document dictionaries
        label2id: Mapping from rechtsfeitcode to index

    Returns:
        NumPy array of shape (num_documents, num_labels)
    """
    import numpy as np

    num_docs = len(documents)
    num_labels = len(label2id)

    label_matrix = np.zeros((num_docs, num_labels), dtype=np.float32)

    for i, doc in enumerate(documents):
        for code in doc.get('rechtsfeitcodes', []):
            if code in label2id:
                label_matrix[i, label2id[code]] = 1.0

    return label_matrix


def get_dataset_statistics(documents: List[Dict]) -> Dict:
    """
    Get comprehensive statistics about the dataset.

    Args:
        documents: List of document dictionaries

    Returns:
        Dictionary with dataset statistics
    """
    text_lengths = [len(doc.get('text', '')) for doc in documents]
    labels_per_doc = [len(doc.get('rechtsfeitcodes', [])) for doc in documents]

    all_codes = []
    for doc in documents:
        all_codes.extend(doc.get('rechtsfeitcodes', []))

    return {
        'num_documents': len(documents),
        'num_unique_labels': len(set(all_codes)),
        'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        'min_text_length': min(text_lengths) if text_lengths else 0,
        'max_text_length': max(text_lengths) if text_lengths else 0,
        'avg_labels_per_doc': sum(labels_per_doc) / len(labels_per_doc) if labels_per_doc else 0,
        'min_labels_per_doc': min(labels_per_doc) if labels_per_doc else 0,
        'max_labels_per_doc': max(labels_per_doc) if labels_per_doc else 0,
    }
