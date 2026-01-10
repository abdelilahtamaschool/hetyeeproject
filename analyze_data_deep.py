"""
Deep Data Analysis for Kadaster Legal Document Classification
Comprehensive analysis to identify improvement opportunities
"""

import pickle
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
import re

def load_data():
    """Load train/val/test splits"""
    train_df = pd.read_pickle("Datasets/processed/train.pkl")
    val_df = pd.read_pickle("Datasets/processed/val.pkl")
    test_df = pd.read_pickle("Datasets/processed/test.pkl")
    return train_df, val_df, test_df

def analyze_class_distribution(train_df, val_df, test_df):
    """Analyze class distribution across splits"""
    print("=" * 100)
    print("1. CLASS DISTRIBUTION ANALYSIS")
    print("=" * 100)

    train_counts = Counter(train_df['primary_label'])
    val_counts = Counter(val_df['primary_label'])
    test_counts = Counter(test_df['primary_label'])

    total_train = len(train_df)

    # Create summary table
    all_labels = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))

    print(f"\nTotal classes: {len(all_labels)}")
    print(f"Total samples: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Detailed per-class analysis
    print(f"\n{'Label':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10} {'%Train':<10} {'Imbalance':<15}")
    print("-" * 85)

    max_count = max(train_counts.values())
    imbalance_issues = []

    for label in sorted(train_counts.keys(), key=lambda x: train_counts[x], reverse=True):
        train_n = train_counts.get(label, 0)
        val_n = val_counts.get(label, 0)
        test_n = test_counts.get(label, 0)
        total_n = train_n + val_n + test_n
        pct = train_n / total_train * 100
        imbalance = max_count / train_n if train_n > 0 else float('inf')

        status = ""
        if train_n < 20:
            status = "CRITICAL (<20)"
            imbalance_issues.append((label, train_n, "critical"))
        elif train_n < 50:
            status = "LOW (<50)"
            imbalance_issues.append((label, train_n, "low"))
        elif train_n < 100:
            status = "MODERATE"

        print(f"{label:<10} {train_n:<10} {val_n:<10} {test_n:<10} {total_n:<10} {pct:<10.2f} {status:<15}")

    print("\n" + "=" * 100)
    print("CLASS IMBALANCE SUMMARY")
    print("=" * 100)

    critical = [x for x in imbalance_issues if x[2] == "critical"]
    low = [x for x in imbalance_issues if x[2] == "low"]

    print(f"\nCRITICAL classes (<20 samples): {len(critical)}")
    for label, count, _ in critical:
        print(f"  - Label {label}: {count} samples")

    print(f"\nLOW classes (<50 samples): {len(low)}")
    for label, count, _ in low:
        print(f"  - Label {label}: {count} samples")

    print(f"\nImbalance ratio (max/min): {max(train_counts.values()) / min(train_counts.values()):.1f}x")

    return train_counts, imbalance_issues

def analyze_text_characteristics(train_df):
    """Analyze text characteristics"""
    print("\n" + "=" * 100)
    print("2. TEXT CHARACTERISTICS ANALYSIS")
    print("=" * 100)

    # Basic text stats
    train_df = train_df.copy()
    train_df['char_count'] = train_df['text'].str.len()
    train_df['word_count'] = train_df['text'].str.split().str.len()
    train_df['sentence_count'] = train_df['text'].str.count(r'[.!?]+')
    train_df['avg_word_length'] = train_df['text'].apply(lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0)

    print(f"\nDocument Length Statistics:")
    print(f"  Characters: min={train_df['char_count'].min():,}, max={train_df['char_count'].max():,}, mean={train_df['char_count'].mean():,.0f}, median={train_df['char_count'].median():,.0f}")
    print(f"  Words: min={train_df['word_count'].min():,}, max={train_df['word_count'].max():,}, mean={train_df['word_count'].mean():,.0f}, median={train_df['word_count'].median():,.0f}")
    print(f"  Sentences: min={train_df['sentence_count'].min():,}, max={train_df['sentence_count'].max():,}, mean={train_df['sentence_count'].mean():,.0f}")
    print(f"  Avg word length: {train_df['avg_word_length'].mean():.1f} characters")

    # Token estimation (approx 4 chars per token for Dutch)
    train_df['estimated_tokens'] = train_df['char_count'] / 4

    print(f"\nEstimated Token Distribution (assuming ~4 chars/token):")
    print(f"  Documents > 512 tokens: {(train_df['estimated_tokens'] > 512).sum()} ({(train_df['estimated_tokens'] > 512).mean()*100:.1f}%)")
    print(f"  Documents > 1024 tokens: {(train_df['estimated_tokens'] > 1024).sum()} ({(train_df['estimated_tokens'] > 1024).mean()*100:.1f}%)")
    print(f"  Documents > 2048 tokens: {(train_df['estimated_tokens'] > 2048).sum()} ({(train_df['estimated_tokens'] > 2048).mean()*100:.1f}%)")
    print(f"  Documents > 4096 tokens: {(train_df['estimated_tokens'] > 4096).sum()} ({(train_df['estimated_tokens'] > 4096).mean()*100:.1f}%)")
    print(f"  Documents > 8192 tokens: {(train_df['estimated_tokens'] > 8192).sum()} ({(train_df['estimated_tokens'] > 8192).mean()*100:.1f}%)")

    return train_df

def analyze_text_patterns(train_df):
    """Analyze common text patterns"""
    print("\n" + "=" * 100)
    print("3. TEXT PATTERN ANALYSIS")
    print("=" * 100)

    # Common patterns in legal documents
    patterns = {
        'dates': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        'money_amounts': r'[€$]\s*[\d.,]+|[\d.,]+\s*(euro|EUR)',
        'percentages': r'\d+[.,]?\d*\s*%',
        'article_references': r'artikel\s+\d+',
        'law_references': r'(BW|Rv|Sr|Sv|AWB|WvK)',
        'cadastral_numbers': r'\d{4}\s*[A-Z]{1,2}\s*\d+',
        'email_addresses': r'[\w\.-]+@[\w\.-]+',
        'phone_numbers': r'\b\d{2,4}[-\s]?\d{6,8}\b',
        'anonymized_markers': r'\[.*?\]|\<.*?\>',
        'headers_underscores': r'_{3,}',
        'repeated_spaces': r'\s{3,}',
        'all_caps_words': r'\b[A-Z]{4,}\b',
    }

    sample = train_df['text'].head(500)  # Sample for speed

    print("\nPattern frequency in sample of 500 documents:")
    for name, pattern in patterns.items():
        matches = sample.str.count(pattern).sum()
        docs_with_pattern = (sample.str.contains(pattern, regex=True, na=False)).sum()
        print(f"  {name:<25}: {matches:>6} occurrences in {docs_with_pattern:>3} docs ({docs_with_pattern/500*100:.1f}%)")

    # Check for potential data quality issues
    print("\n" + "=" * 100)
    print("POTENTIAL DATA QUALITY ISSUES")
    print("=" * 100)

    # Very short documents
    very_short = (train_df['text'].str.len() < 500).sum()
    print(f"\n  Very short documents (<500 chars): {very_short} ({very_short/len(train_df)*100:.1f}%)")

    # Documents with lots of special characters
    special_char_heavy = train_df['text'].str.count(r'[^\w\s]').mean()
    print(f"  Avg special characters per doc: {special_char_heavy:.0f}")

    # Documents with lots of numbers
    number_heavy = train_df['text'].str.count(r'\d').mean()
    print(f"  Avg digits per doc: {number_heavy:.0f}")

    # Check for empty or near-empty docs
    empty_docs = (train_df['text'].str.strip() == '').sum()
    print(f"  Empty documents: {empty_docs}")

def analyze_vocabulary(train_df):
    """Analyze vocabulary characteristics"""
    print("\n" + "=" * 100)
    print("4. VOCABULARY ANALYSIS")
    print("=" * 100)

    # Build vocabulary from sample
    all_words = []
    for text in train_df['text'].head(1000):  # Sample for speed
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)

    word_counts = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_counts)

    print(f"\nVocabulary Statistics (sample of 1000 docs):")
    print(f"  Total words: {total_words:,}")
    print(f"  Unique words: {unique_words:,}")
    print(f"  Type-token ratio: {unique_words/total_words:.4f}")

    print(f"\nTop 30 most common words:")
    for word, count in word_counts.most_common(30):
        print(f"  {word:<20}: {count:>6} ({count/total_words*100:.2f}%)")

    # Legal-specific terms
    legal_terms = ['akte', 'hypotheek', 'eigendom', 'kadaster', 'perceel', 'notaris',
                   'overdracht', 'recht', 'verklaring', 'partij', 'comparant', 'koopsom',
                   'registergoed', 'levering', 'zekerheid', 'schuld', 'erfpacht', 'opstal']

    print(f"\nLegal term frequency:")
    for term in legal_terms:
        count = word_counts.get(term, 0)
        if count > 0:
            print(f"  {term:<20}: {count:>6} ({count/total_words*100:.3f}%)")

def analyze_per_class_text_differences(train_df):
    """Analyze text differences between classes"""
    print("\n" + "=" * 100)
    print("5. PER-CLASS TEXT ANALYSIS")
    print("=" * 100)

    train_df = train_df.copy()
    train_df['word_count'] = train_df['text'].str.split().str.len()

    # Aggregate by class
    class_stats = train_df.groupby('primary_label').agg({
        'word_count': ['mean', 'std', 'min', 'max', 'count'],
        'text': lambda x: len(x)
    }).round(1)

    print(f"\nWord count statistics per class:")
    print(f"{'Label':<10} {'Count':<8} {'Mean':<10} {'Std':<10} {'Min':<8} {'Max':<10}")
    print("-" * 60)

    for label in sorted(train_df['primary_label'].unique()):
        subset = train_df[train_df['primary_label'] == label]
        count = len(subset)
        mean_wc = subset['word_count'].mean()
        std_wc = subset['word_count'].std()
        min_wc = subset['word_count'].min()
        max_wc = subset['word_count'].max()
        print(f"{label:<10} {count:<8} {mean_wc:<10.1f} {std_wc:<10.1f} {min_wc:<8} {max_wc:<10}")

def generate_recommendations(train_counts, imbalance_issues):
    """Generate specific recommendations"""
    print("\n" + "=" * 100)
    print("6. RECOMMENDATIONS FOR IMPROVEMENT")
    print("=" * 100)

    print("\n### A. DATA-LEVEL IMPROVEMENTS ###")

    critical = [x for x in imbalance_issues if x[2] == "critical"]
    low = [x for x in imbalance_issues if x[2] == "low"]

    if critical:
        print(f"""
1. CRITICAL: Oversample/Augment these {len(critical)} classes with <20 samples:
   - Use back-translation (Dutch -> English -> Dutch)
   - Apply synonym replacement for legal terms
   - Consider merging semantically similar classes if appropriate

   Classes needing augmentation: {[x[0] for x in critical]}
""")

    if low:
        print(f"""
2. LOW: Consider augmentation for these {len(low)} classes with <50 samples:
   - Apply moderate oversampling (2-3x)
   - Use EDA (Easy Data Augmentation) techniques

   Classes: {[x[0] for x in low]}
""")

    print("""
3. TEXT PREPROCESSING IMPROVEMENTS:
   - Normalize whitespace (multiple spaces -> single)
   - Remove/standardize headers and footers
   - Standardize date formats
   - Normalize currency amounts
   - Handle anonymization markers consistently
   - Consider lowercasing (already done in TF-IDF, not in BERT)
""")

    print("\n### B. MODEL-LEVEL IMPROVEMENTS ###")
    print("""
1. LOSS FUNCTION:
   - [IMPLEMENTED] Combined Focal Loss + Class Weights
   - Consider: Dice Loss for extreme imbalance
   - Consider: CB (Class-Balanced) Loss

2. SAMPLING STRATEGY:
   - [IMPLEMENTED] Class-balanced sampling
   - Consider: Curriculum learning (easy -> hard examples)
   - Consider: Progressive resizing (start with shorter sequences)

3. REGULARIZATION:
   - [IMPLEMENTED] R-Drop
   - Consider: Mixup data augmentation
   - Consider: Manifold Mixup (hidden layer interpolation)
   - Consider: Stochastic Weight Averaging (SWA)

4. ARCHITECTURE:
   - Consider: Adding a dense layer before classifier
   - Consider: Multi-sample dropout
   - Consider: Attention pooling instead of [CLS] token
""")

    print("\n### C. TRAINING STRATEGY ###")
    print("""
1. TWO-STAGE TRAINING:
   - Stage 1: Train on balanced subset (downsample majority)
   - Stage 2: Fine-tune on full dataset with class weights

2. ENSEMBLE APPROACH:
   - Train 3-5 models with different random seeds
   - Average predictions (or use voting)
   - Expected: +2-3% accuracy gain

3. CROSS-VALIDATION:
   - Use 5-fold stratified CV for robust evaluation
   - Helps identify unstable minority class predictions

4. THRESHOLD OPTIMIZATION:
   - Optimize per-class decision thresholds on validation set
   - Can significantly improve minority class recall
""")

    print("\n### D. SPECIFIC HYPERPARAMETER TUNING ###")
    print("""
Based on the data characteristics, consider:

1. Learning Rate:
   - Current: 8e-6 (good for 45 classes)
   - Try: 5e-6 for even more stable training

2. Warmup:
   - Current: 10% of training
   - Consider: 15% for better minority class learning

3. Epochs:
   - Current: 8
   - Consider: 10-12 with early stopping (patience 10)

4. Batch Size:
   - Current: 16 per GPU
   - Consider: 8 per GPU with gradient accumulation 4
     (smaller batches = more updates = better for minority classes)

5. Label Smoothing:
   - Current: 0.15
   - Consider: 0.2 for extreme imbalance
""")

def main():
    print("=" * 100)
    print("DEEP DATA ANALYSIS FOR KADASTER LEGAL DOCUMENT CLASSIFICATION")
    print("=" * 100)

    print("\nLoading data...")
    train_df, val_df, test_df = load_data()
    print(f"Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Run all analyses
    train_counts, imbalance_issues = analyze_class_distribution(train_df, val_df, test_df)
    train_df = analyze_text_characteristics(train_df)
    analyze_text_patterns(train_df)
    analyze_vocabulary(train_df)
    analyze_per_class_text_differences(train_df)
    generate_recommendations(train_counts, imbalance_issues)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
