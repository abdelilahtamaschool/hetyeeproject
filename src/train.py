"""
Training script for baseline model
TF-IDF + Logistic Regression
"""

import pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import pandas as pd
from preprocessing import load_splits, prepare_data_for_classification


class BaselineModel:
    """
    Baseline model using TF-IDF + Logistic Regression
    """

    def __init__(self,
                 max_features: int = 10000,
                 max_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize baseline model

        Args:
            max_features: Maximum number of TF-IDF features
            max_iter: Maximum iterations for Logistic Regression
            random_state: Random seed
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency (filter out very common words)
            strip_accents='unicode',
            lowercase=True,
            stop_words=None  # Keep all words for Dutch legal texts
        )

        self.classifier = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',  # Handle class imbalance
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )

        self.label_mapping = None

    def fit(self, X_train, y_train):
        """
        Train the model

        Args:
            X_train: Training texts
            y_train: Training labels
        """
        print("Vectorizing training data with TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        print(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
        print(f"Number of unique labels: {len(np.unique(y_train))}")

        print("\nTraining Logistic Regression classifier...")
        self.classifier.fit(X_train_tfidf, y_train)

        print("Training completed!")

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Input texts

        Returns:
            Predicted labels
        """
        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict(X_tfidf)

    def predict_proba(self, X):
        """
        Get prediction probabilities

        Args:
            X: Input texts

        Returns:
            Prediction probabilities
        """
        X_tfidf = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_tfidf)

    def save(self, model_dir: str = "models"):
        """
        Save model to disk

        Args:
            model_dir: Directory to save model
        """
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{model_dir}/baseline_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        with open(f"{model_dir}/baseline_classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)

        print(f"\nModel saved to {model_dir}/")

    def load(self, model_dir: str = "models"):
        """
        Load model from disk

        Args:
            model_dir: Directory containing model
        """
        with open(f"{model_dir}/baseline_vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

        with open(f"{model_dir}/baseline_classifier.pkl", "rb") as f:
            self.classifier = pickle.load(f)

        print(f"Model loaded from {model_dir}/")


def evaluate_model(model, X, y_true, dataset_name: str = "Test"):
    """
    Evaluate model performance

    Args:
        model: Trained model
        X: Input texts
        y_true: True labels
        dataset_name: Name of dataset (for display)

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating on {dataset_name} Set")
    print(f"{'='*80}")

    y_pred = model.predict(X)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} (weighted)")
    print(f"  Recall:    {recall:.4f} (weighted)")
    print(f"  F1-Score:  {f1:.4f} (weighted)")

    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred,
        'true_labels': y_true
    }

    return metrics


if __name__ == "__main__":
    print("="*80)
    print("BASELINE MODEL TRAINING - TF-IDF + Logistic Regression")
    print("="*80)

    # Load data splits
    print("\nLoading data splits...")
    train_df, val_df, test_df = load_splits()

    print(f"  Train set: {len(train_df)} documents")
    print(f"  Validation set: {len(val_df)} documents")
    print(f"  Test set: {len(test_df)} documents")

    # Prepare data
    print("\nPreparing data for classification...")
    X_train, y_train = prepare_data_for_classification(train_df)
    X_val, y_val = prepare_data_for_classification(val_df)
    X_test, y_test = prepare_data_for_classification(test_df)

    # Initialize and train model
    print("\nInitializing baseline model...")
    model = BaselineModel(max_features=10000, max_iter=1000)

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate on validation set
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")

    # Evaluate on test set
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    # Save model
    model.save()

    # Save metrics
    metrics_summary = {
        'validation': {
            'accuracy': val_metrics['accuracy'],
            'precision': val_metrics['precision'],
            'recall': val_metrics['recall'],
            'f1_score': val_metrics['f1_score']
        },
        'test': {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1_score': test_metrics['f1_score']
        }
    }

    with open("models/baseline_metrics.pkl", "wb") as f:
        pickle.dump(metrics_summary, f)

    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"\nFinal Results:")
    print(f"  Validation Accuracy: {val_metrics['accuracy']*100:.2f}%")
    print(f"  Test Accuracy:       {test_metrics['accuracy']*100:.2f}%")
    print(f"\n  Goal: 90% accuracy")
    print(f"  Gap:  {90 - test_metrics['accuracy']*100:.2f}%")
    print("="*80)
