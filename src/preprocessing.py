"""
Preprocessing utilities for Kadaster legal documents
Includes train/validation/test splitting and text preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle


def create_stratified_split(df: pd.DataFrame,
                            train_size: float = 0.70,
                            val_size: float = 0.15,
                            test_size: float = 0.15,
                            random_state: int = 42,
                            min_samples_per_class: int = 10) -> tuple:
    """
    Create stratified train/validation/test split

    For baseline model, we use only the first label (primary label)
    to ensure stratification works properly.

    Args:
        df: DataFrame with document data
        train_size: Proportion for training set (default 0.70)
        val_size: Proportion for validation set (default 0.15)
        test_size: Proportion for test set (default 0.15)
        random_state: Random seed for reproducibility
        min_samples_per_class: Minimum samples required for a class (default 10)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs((train_size + val_size + test_size) - 1.0) < 0.001, \
        "Split proportions must sum to 1.0"

    # For baseline: use primary (first) label only
    df['primary_label'] = df['rechtsfeitcodes'].apply(lambda x: x[0])

    # Filter out classes with too few samples
    label_counts = df['primary_label'].value_counts()
    valid_labels = label_counts[label_counts >= min_samples_per_class].index

    original_size = len(df)
    df_filtered = df[df['primary_label'].isin(valid_labels)].copy()

    if len(df_filtered) < original_size:
        print(f"Filtered out {original_size - len(df_filtered)} documents from rare classes (< {min_samples_per_class} samples)")
        print(f"Remaining documents: {len(df_filtered)} ({len(df_filtered)/original_size*100:.1f}%)")

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df_filtered,
        test_size=(val_size + test_size),
        stratify=df_filtered['primary_label'],
        random_state=random_state
    )

    # Second split: val vs test
    # Adjust the proportion for the second split
    val_proportion = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_proportion),
        stratify=temp_df['primary_label'],
        random_state=random_state
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df_filtered)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} ({len(val_df)/len(df_filtered)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df_filtered)*100:.1f}%)")

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                test_df: pd.DataFrame,
                output_dir: str = "Datasets/processed"):
    """
    Save train/val/test splits to disk

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Directory to save splits
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_df.to_pickle(f"{output_dir}/train.pkl")
    val_df.to_pickle(f"{output_dir}/val.pkl")
    test_df.to_pickle(f"{output_dir}/test.pkl")

    print(f"\nSaved splits to {output_dir}/")
    print(f"  train.pkl: {len(train_df)} documents")
    print(f"  val.pkl: {len(val_df)} documents")
    print(f"  test.pkl: {len(test_df)} documents")


def load_splits(input_dir: str = "Datasets/processed") -> tuple:
    """
    Load train/val/test splits from disk

    Args:
        input_dir: Directory containing splits

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = pd.read_pickle(f"{input_dir}/train.pkl")
    val_df = pd.read_pickle(f"{input_dir}/val.pkl")
    test_df = pd.read_pickle(f"{input_dir}/test.pkl")

    return train_df, val_df, test_df


def prepare_data_for_classification(df: pd.DataFrame, use_primary_label: bool = True):
    """
    Prepare data for classification

    Args:
        df: DataFrame with document data
        use_primary_label: If True, use only primary (first) label

    Returns:
        Tuple of (texts, labels)
    """
    texts = df['text'].values

    if use_primary_label:
        labels = df['primary_label'].values
    else:
        labels = df['rechtsfeitcodes'].values

    return texts, labels


if __name__ == "__main__":
    # Test the preprocessing pipeline
    from data_loader import jsonl_to_dataframe

    print("Loading data...")
    data_path = "ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl"
    df = jsonl_to_dataframe(data_path)

    print("\nCreating stratified splits...")
    train_df, val_df, test_df = create_stratified_split(df)

    print("\nChecking label distribution in splits:")
    print("\nTrain set - Top 10 labels:")
    print(train_df['primary_label'].value_counts().head(10))

    print("\nValidation set - Top 10 labels:")
    print(val_df['primary_label'].value_counts().head(10))

    print("\nTest set - Top 10 labels:")
    print(test_df['primary_label'].value_counts().head(10))

    print("\nSaving splits...")
    save_splits(train_df, val_df, test_df)

    print("\nDone!")
