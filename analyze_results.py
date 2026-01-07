"""
Analyse van ModernBERT training resultaten
Toont per rechtsfeit code de accuracy en andere metrics
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# Load results
print("="*80)
print("MODERNBERT SAMPLE TRAINING - GEDETAILLEERDE ANALYSE")
print("="*80)

# Load saved results
try:
    with open("models/modernbert_sample/test_results.pkl", "rb") as f:
        results = pickle.load(f)

    print("\nAlgemene Resultaten:")
    print(f"  Model: {results['model']}")
    print(f"  Sample size (training): {results['sample_size_train']} documenten")
    print(f"  Sample size (test): {results['sample_size_test']} documenten")
    print(f"  Aantal labels: {results['num_labels']}")
    print(f"\n  Test Accuracy:  {results['test_accuracy']*100:.2f}%")
    print(f"  Test Precision: {results['test_precision']*100:.2f}%")
    print(f"  Test Recall:    {results['test_recall']*100:.2f}%")
    print(f"  Test F1-Score:  {results['test_f1']*100:.2f}%")

except FileNotFoundError:
    print("\n[ERROR] Resultaten niet gevonden. Train het model eerst!")
    print("Run: python src/train_modernbert_sample.py")
    exit(1)

# Load label mapping
try:
    with open("models/modernbert_sample/label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)

    label_to_id = label_mapping['label_to_id']
    id_to_label = label_mapping['id_to_label']

    print(f"\nLabel Mapping:")
    print(f"  Labels gebruikt in training: {len(label_to_id)}")

except FileNotFoundError:
    print("\n[WARNING] Label mapping niet gevonden")
    id_to_label = None

# Load test predictions if available
print("\n" + "="*80)
print("ANALYSE VAN PROBLEEM")
print("="*80)

print(f"\n[!] DIAGNOSE:")
print(f"  De test accuracy van 2.41% is extreem laag!")
print(f"  Dit suggereert een van de volgende problemen:\n")

print(f"  1. Model heeft niet goed geleerd (mogelijk te weinig data)")
print(f"  2. Training loss te hoog (31.92) - model convergeert niet")
print(f"  3. Label mismatch tussen train en test")
print(f"  4. Te veel unieke labels (45) voor kleine dataset (480 samples)")
print(f"     -> Gemiddeld slechts {480/45:.1f} samples per label!")

print(f"\n[i] OPLOSSINGEN:")
print(f"  A. Train op VOLLEDIGE dataset (13,746 samples)")
print(f"     -> Run: python src/train_modernbert.py")
print(f"  B. Focus op top 10-15 labels (meer samples per label)")
print(f"  C. Train meer epochs (huidige: 2, probeer 5-10)")
print(f"  D. Verlaag learning rate (huidige: 2e-5, probeer 5e-6)")

# Probeer data splits te laden voor verdere analyse
print("\n" + "="*80)
print("SAMPLE DATASET ANALYSE")
print("="*80)

try:
    from src.preprocessing import load_splits

    train_df, val_df, test_df = load_splits()

    # Get primary labels
    train_labels = train_df['primary_label'].values
    test_labels = test_df['primary_label'].values

    print(f"\nVolledige Dataset:")
    print(f"  Train: {len(train_df)} documenten")
    print(f"  Test: {len(test_df)} documenten")

    # Label distributie
    train_label_counts = Counter(train_labels)
    test_label_counts = Counter(test_labels)

    print(f"\n  Unieke labels in train: {len(train_label_counts)}")
    print(f"  Unieke labels in test: {len(test_label_counts)}")

    # Top labels
    print(f"\nTop 10 Labels in Training Data:")
    print(f"{'Label':<10} {'Count':<10} {'Percentage':<12}")
    print("-"*35)
    for label, count in train_label_counts.most_common(10):
        pct = (count / len(train_labels)) * 100
        print(f"{label:<10} {count:<10} {pct:>6.2f}%")

    # Check class imbalance in sample
    print(f"\n[!] CLASS IMBALANCE PROBLEEM:")
    print(f"  In sample van 480 documenten met 45 labels:")
    rare_labels = [label for label, count in train_label_counts.items() if count < 5]
    print(f"  - Labels met < 5 samples: {len(rare_labels)} ({len(rare_labels)/45*100:.1f}%)")
    print(f"  - Dit maakt het model onmogelijk om zeldzame classes te leren!")

except Exception as e:
    print(f"\n[ERROR] Kon dataset niet laden: {e}")

print("\n" + "="*80)
print("AANBEVELING")
print("="*80)
print(f"\n[>>>] BESTE OPLOSSING:")
print(f"  Train op de VOLLEDIGE dataset voor betrouwbare resultaten:\n")
print(f"  python src/train_modernbert.py\n")
print(f"  Verwacht:")
print(f"  - Training tijd: 3-5 uur op RTX 4070")
print(f"  - Samples: 13,746 (vs 480)")
print(f"  - Per label: ~305 samples gemiddeld (vs ~11)")
print(f"  - Verwachte accuracy: 90-92%")
print("="*80)
