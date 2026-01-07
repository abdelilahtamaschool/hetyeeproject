"""
Data loader for Kadaster legal documents
Loads JSONL data and performs basic preprocessing
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load JSONL file containing legal documents

    Args:
        file_path: Path to JSONL file

    Returns:
        List of dictionaries with document data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def jsonl_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Convert JSONL to pandas DataFrame

    Args:
        file_path: Path to JSONL file

    Returns:
        DataFrame with columns: akteId, text, rechtsfeitcodes
    """
    data = load_jsonl(file_path)

    # Extract data
    df = pd.DataFrame({
        'akteId': [d['akteId'] for d in data],
        'text': [d['text'] for d in data],
        'rechtsfeitcodes': [d['rechtsfeitcodes'] for d in data]
    })

    return df


def get_data_statistics(df: pd.DataFrame) -> Dict:
    """
    Get basic statistics about the dataset

    Args:
        df: DataFrame with document data

    Returns:
        Dictionary with statistics
    """
    # Text length statistics
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()

    # Label statistics
    all_labels = []
    for labels in df['rechtsfeitcodes']:
        all_labels.extend(labels)

    label_counts = Counter(all_labels)

    # Number of labels per document
    df['num_labels'] = df['rechtsfeitcodes'].apply(len)

    stats = {
        'total_documents': len(df),
        'total_labels': len(all_labels),
        'unique_labels': len(label_counts),
        'label_distribution': label_counts,
        'avg_text_length': df['text_length'].mean(),
        'median_text_length': df['text_length'].median(),
        'min_text_length': df['text_length'].min(),
        'max_text_length': df['text_length'].max(),
        'avg_word_count': df['word_count'].mean(),
        'avg_labels_per_doc': df['num_labels'].mean(),
        'max_labels_per_doc': df['num_labels'].max(),
        'min_labels_per_doc': df['num_labels'].min()
    }

    return stats


def print_statistics(stats: Dict):
    """
    Print dataset statistics in a readable format

    Args:
        stats: Dictionary with statistics from get_data_statistics
    """
    print("="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"\nDocument Statistics:")
    print(f"  Total documents: {stats['total_documents']:,}")
    print(f"  Average text length: {stats['avg_text_length']:,.0f} characters")
    print(f"  Median text length: {stats['median_text_length']:,.0f} characters")
    print(f"  Min text length: {stats['min_text_length']:,} characters")
    print(f"  Max text length: {stats['max_text_length']:,} characters")
    print(f"  Average word count: {stats['avg_word_count']:,.0f} words")

    print(f"\nLabel Statistics:")
    print(f"  Total label occurrences: {stats['total_labels']:,}")
    print(f"  Unique labels: {stats['unique_labels']}")
    print(f"  Average labels per document: {stats['avg_labels_per_doc']:.2f}")
    print(f"  Min labels per document: {stats['min_labels_per_doc']}")
    print(f"  Max labels per document: {stats['max_labels_per_doc']}")

    print(f"\nLabel Distribution (Top 20):")
    print(f"{'Label':<15} {'Count':<10} {'Percentage':<10}")
    print("-"*40)

    total_labels = stats['total_labels']
    for label, count in stats['label_distribution'].most_common(20):
        percentage = (count / total_labels) * 100
        print(f"{str(label):<15} {count:<10} {percentage:.2f}%")

    print("="*80)


if __name__ == "__main__":
    # Test the data loader
    data_path = "ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl"

    print("Loading data...")
    df = jsonl_to_dataframe(data_path)

    print(f"Loaded {len(df)} documents\n")

    print("Calculating statistics...")
    stats = get_data_statistics(df)

    print_statistics(stats)

    print(f"\nFirst few rows:")
    print(df.head())
