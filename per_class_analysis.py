"""
Per rechtsfeit code accuracy analyse
Maakt een gedetailleerde tabel van alle rechtsfeit codes
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
import numpy as np
from model import ModernBERTClassifier, ModernBERTConfig
from src.preprocessing import load_splits, prepare_data_for_classification
from sklearn.metrics import classification_report, confusion_matrix

print("="*100)
print("PER RECHTSFEIT CODE ACCURACY ANALYSE")
print("="*100)

# Load test data
print("\nLaden van test data...")
train_df, val_df, test_df = load_splits()

# Load label mapping
with open("models/modernbert_sample/label_mapping.pkl", "rb") as f:
    label_mapping = pickle.load(f)

label_to_id = label_mapping['label_to_id']
id_to_label = label_mapping['id_to_label']

# Prepare test data
X_test, y_test = prepare_data_for_classification(test_df, use_primary_label=True)

# Filter only labels in our sample
test_mask = [label in label_to_id for label in y_test]
X_test_filtered = X_test[test_mask]
y_test_filtered = y_test[test_mask]

# Map to IDs
y_test_mapped = [label_to_id[label] for label in y_test_filtered]

print(f"Test samples: {len(y_test_mapped)}")
print(f"Unique labels: {len(set(y_test_mapped))}")

# Load model
print("\nLaden van getraind model...")
num_labels = len(label_to_id)
classifier = ModernBERTClassifier(
    num_labels=num_labels,
    model_name=ModernBERTConfig.MODEL_NAME,
    max_length=ModernBERTConfig.MAX_LENGTH
)

# Load trained weights
try:
    classifier.load("models/modernbert_sample")
    print("Model geladen!")
except:
    print("[ERROR] Model niet gevonden. Train eerst het model!")
    exit(1)

# Make predictions
print("\nMaken van predictions...")
y_pred = classifier.predict(X_test_filtered.tolist())

# Calculate per-class metrics
print("\n" + "="*100)
print("PER RECHTSFEIT CODE RESULTATEN")
print("="*100)

# Get classification report
report = classification_report(y_test_mapped, y_pred, output_dict=True, zero_division=0)

# Create detailed table
results = []
for label_id in sorted(id_to_label.keys()):
    rechtsfeit_code = id_to_label[label_id]

    # Get metrics for this class
    if str(label_id) in report:
        metrics = report[str(label_id)]
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score = metrics['f1-score']
        support = int(metrics['support'])
    else:
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        support = 0

    # Count in training data
    train_count = sum(1 for label in train_df['primary_label'] if label == rechtsfeit_code)

    results.append({
        'Rechtsfeit Code': rechtsfeit_code,
        'Test Samples': support,
        'Train Samples': train_count,
        'Precision': f'{precision*100:.1f}%',
        'Recall': f'{recall*100:.1f}%',
        'F1-Score': f'{f1_score*100:.1f}%',
        'Accuracy': f'{f1_score*100:.1f}%'  # F1 as proxy for accuracy
    })

# Create DataFrame
df_results = pd.DataFrame(results)

# Sort by rechtsfeit code
df_results = df_results.sort_values('Rechtsfeit Code')

# Print table
print("\n" + df_results.to_string(index=False))

# Summary statistics
print("\n" + "="*100)
print("SAMENVATTING")
print("="*100)

# Overall metrics
overall_acc = report['accuracy']
print(f"\nOverall Accuracy: {overall_acc*100:.2f}%")
print(f"Macro Avg F1: {report['macro avg']['f1-score']*100:.2f}%")
print(f"Weighted Avg F1: {report['weighted avg']['f1-score']*100:.2f}%")

# Best and worst performing classes
df_results['F1_numeric'] = [float(x.rstrip('%')) for x in df_results['F1-Score']]
df_best = df_results.nlargest(5, 'F1_numeric')
df_worst = df_results.nsmallest(5, 'F1_numeric')

print(f"\nBeste 5 Rechtsfeit Codes (hoogste F1):")
print(df_best[['Rechtsfeit Code', 'F1-Score', 'Test Samples', 'Train Samples']].to_string(index=False))

print(f"\nSlechtste 5 Rechtsfeit Codes (laagste F1):")
print(df_worst[['Rechtsfeit Code', 'F1-Score', 'Test Samples', 'Train Samples']].to_string(index=False))

# Save results
df_results.to_csv("models/modernbert_sample/per_class_results.csv", index=False)
print(f"\nResultaten opgeslagen in: models/modernbert_sample/per_class_results.csv")

print("\n" + "="*100)
print("CONCLUSIE: Sample te klein voor betrouwbare resultaten!")
print("Train op volledige dataset: python src/train_modernbert.py")
print("="*100)
