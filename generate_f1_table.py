"""
Genereer F1-score tabel voor alle rechtsfeit codes
Gebruikt het getrainde ModernBERT model om predictions te maken en F1-scores te berekenen
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
import numpy as np
from model import ModernBERTClassifier, ModernBERTConfig
from src.preprocessing import load_splits, prepare_data_for_classification
from sklearn.metrics import classification_report, f1_score
from collections import Counter

print("="*100)
print("F1-SCORE TABEL - ALLE RECHTSFEIT CODES")
print("="*100)

# Load data
print("\nLaden van data...")
train_df, val_df, test_df = load_splits()
print(f"  Train: {len(train_df)} documenten")
print(f"  Test: {len(test_df)} documenten")

# Load label mapping
print("\nLaden van label mapping...")
with open("models/modernbert/label_mapping.pkl", "rb") as f:
    label_mapping = pickle.load(f)

label_to_id = label_mapping['label_to_id']
id_to_label = label_mapping['id_to_label']
print(f"  Aantal labels: {len(label_to_id)}")

# Prepare test data
print("\nVoorbereiden van test data...")
X_test, y_test = prepare_data_for_classification(test_df, use_primary_label=True)
y_test_mapped = [label_to_id[label] for label in y_test]

# Load model
print("\nLaden van getraind ModernBERT model...")
num_labels = len(label_to_id)
classifier = ModernBERTClassifier(
    num_labels=num_labels,
    model_name=ModernBERTConfig.MODEL_NAME,
    max_length=ModernBERTConfig.MAX_LENGTH
)

try:
    classifier.load("models/modernbert")
    print("  Model succesvol geladen!")
except Exception as e:
    print(f"  [ERROR] Kan model niet laden: {e}")
    print("  Zorg dat training is voltooid!")
    exit(1)

# Make predictions
print("\nMaken van predictions op test set...")
print("  Dit kan enkele minuten duren...")
y_pred = classifier.predict(X_test.tolist())

# Calculate overall metrics
print("\n" + "="*100)
print("OVERALL RESULTATEN")
print("="*100)

from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(y_test_mapped, y_pred)
precision = precision_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test_mapped, y_pred, average='weighted', zero_division=0)
f1_weighted = f1_score(y_test_mapped, y_pred, average='weighted', zero_division=0)

print(f"\nTest Set Performance:")
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision*100:.2f}%")
print(f"  Recall:    {recall*100:.2f}%")
print(f"  F1-Score:  {f1_weighted*100:.2f}%")

print(f"\nVergelijking met Baseline:")
print(f"  Baseline (TF-IDF): 81.70%")
print(f"  ModernBERT:        {accuracy*100:.2f}%")
print(f"  Verbetering:       {accuracy*100 - 81.70:+.2f}%")

# Get detailed classification report
report = classification_report(y_test_mapped, y_pred, output_dict=True, zero_division=0)

# Create per-class table
print("\n" + "="*100)
print("F1-SCORE PER RECHTSFEIT CODE")
print("="*100)

results = []
for label_id in sorted(id_to_label.keys()):
    rechtsfeit_code = id_to_label[label_id]

    # Get metrics
    if str(label_id) in report:
        metrics = report[str(label_id)]
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1-score']
        support = int(metrics['support'])
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        support = 0

    # Count in training data
    train_count = sum(1 for label in train_df['primary_label'] if label == rechtsfeit_code)

    results.append({
        'Code': rechtsfeit_code,
        'Train Samples': train_count,
        'Test Samples': support,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Precision %': f'{precision*100:.1f}%',
        'Recall %': f'{recall*100:.1f}%',
        'F1 %': f'{f1*100:.1f}%'
    })

# Create DataFrame
df_results = pd.DataFrame(results)

# Sort by rechtsfeit code
df_results = df_results.sort_values('Code')

# Print table
print("\n" + df_results[['Code', 'Train Samples', 'Test Samples', 'Precision %', 'Recall %', 'F1 %']].to_string(index=False))

# Summary statistics
print("\n" + "="*100)
print("SAMENVATTING STATISTICS")
print("="*100)

# Performance categories
excellent = df_results[df_results['F1-Score'] >= 0.90]
good = df_results[(df_results['F1-Score'] >= 0.70) & (df_results['F1-Score'] < 0.90)]
moderate = df_results[(df_results['F1-Score'] >= 0.50) & (df_results['F1-Score'] < 0.70)]
poor = df_results[df_results['F1-Score'] < 0.50]

print(f"\nPerformance Categorieën:")
print(f"  Excellent (F1 >= 90%):  {len(excellent)} codes")
print(f"  Goed (F1 >= 70%):       {len(good)} codes")
print(f"  Matig (F1 >= 50%):      {len(moderate)} codes")
print(f"  Slecht (F1 < 50%):      {len(poor)} codes")

# Best performers
print(f"\nTop 5 Best Performing Codes:")
top5 = df_results.nlargest(5, 'F1-Score')
print(top5[['Code', 'F1 %', 'Test Samples', 'Train Samples']].to_string(index=False))

# Worst performers
print(f"\nTop 5 Worst Performing Codes:")
bottom5 = df_results.nsmallest(5, 'F1-Score')
print(bottom5[['Code', 'F1 %', 'Test Samples', 'Train Samples']].to_string(index=False))

# Save to CSV
output_file = "models/modernbert/f1_scores_per_code.csv"
df_results.to_csv(output_file, index=False)
print(f"\n\nResultaten opgeslagen in: {output_file}")

print("\n" + "="*100)
print("KLAAR!")
print("="*100)
