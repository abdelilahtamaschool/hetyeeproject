"""
Data Augmentation for Kadaster Legal Document Classification
Specifically designed to address the severe class imbalance problem.

Key findings from analysis:
- 13 CRITICAL classes with <20 samples (7-19 samples each)
- 11 LOW classes with <50 samples
- Imbalance ratio: 633:1 (4432 vs 7 samples)
- These classes will NEVER learn without augmentation

Augmentation strategies implemented:
1. Synonym replacement for legal terms
2. Random word deletion (conservative)
3. Random word swap
4. Sentence shuffling (for long documents)
5. Back-translation simulation (word-level perturbation)
"""

import random
import re
from typing import List, Tuple
from collections import Counter
import numpy as np

# Dutch legal term synonyms for replacement
LEGAL_SYNONYMS = {
    'akte': ['document', 'oorkonde', 'stuk'],
    'hypotheek': ['zekerheidsrecht', 'hypothecaire inschrijving'],
    'eigendom': ['bezit', 'eigendomsrecht'],
    'perceel': ['grondstuk', 'kadastrale eenheid', 'onroerend goed'],
    'notaris': ['ambtenaar', 'notarieel ambtenaar'],
    'overdracht': ['levering', 'transport', 'eigendomsoverdracht'],
    'koopsom': ['aankoopprijs', 'koopprijs', 'verkoopprijs'],
    'verkoper': ['vervreemder', 'overdragende partij'],
    'koper': ['verkrijger', 'verwervende partij'],
    'schuldenaar': ['debiteur', 'schuldig'],
    'schuldeiser': ['crediteur', 'vordering hebbende'],
    'partij': ['contractant', 'belanghebbende'],
    'verklaring': ['mededeling', 'statement'],
    'recht': ['bevoegdheid', 'aanspraak'],
    'registergoed': ['onroerende zaak', 'vastgoed'],
    'erfpacht': ['erfpachtrecht', 'gebruiksrecht'],
    'opstal': ['opstalrecht', 'gebouw'],
    'zekerheid': ['garantie', 'waarborg'],
}

# Common Dutch words to potentially delete (low-information words)
DELETABLE_WORDS = {
    'de', 'het', 'een', 'en', 'van', 'te', 'in', 'op', 'voor', 'met',
    'aan', 'door', 'bij', 'tot', 'om', 'als', 'ook', 'zo', 'dan', 'nog',
    'wel', 'al', 'maar', 'toch', 'dus', 'reeds', 'thans', 'hierbij'
}


def synonym_replacement(text: str, n_replacements: int = 3) -> str:
    """
    Replace legal terms with their synonyms.
    Preserves meaning while creating variation.
    """
    words = text.split()
    new_words = words.copy()

    # Find words that have synonyms
    replaceable_indices = []
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,;:!?()')
        if word_lower in LEGAL_SYNONYMS:
            replaceable_indices.append(i)

    # Replace up to n words
    random.shuffle(replaceable_indices)
    for i in replaceable_indices[:n_replacements]:
        word = words[i]
        word_lower = word.lower().strip('.,;:!?()')
        if word_lower in LEGAL_SYNONYMS:
            synonym = random.choice(LEGAL_SYNONYMS[word_lower])
            # Preserve capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()
            new_words[i] = synonym

    return ' '.join(new_words)


def random_deletion(text: str, p: float = 0.05) -> str:
    """
    Randomly delete words with probability p.
    Very conservative (5%) to preserve legal meaning.
    Only deletes low-information words.
    """
    words = text.split()
    if len(words) < 50:  # Don't delete from short texts
        return text

    new_words = []
    for word in words:
        word_lower = word.lower().strip('.,;:!?()')
        # Only consider deleting low-information words
        if word_lower in DELETABLE_WORDS and random.random() < p:
            continue
        new_words.append(word)

    # Ensure we keep at least 90% of text
    if len(new_words) < len(words) * 0.9:
        return text

    return ' '.join(new_words)


def random_swap(text: str, n_swaps: int = 3) -> str:
    """
    Randomly swap adjacent words n times.
    Conservative to preserve legal sentence structure.
    """
    words = text.split()
    if len(words) < 10:
        return text

    new_words = words.copy()
    for _ in range(n_swaps):
        idx = random.randint(0, len(new_words) - 2)
        # Don't swap punctuation
        if not new_words[idx][-1].isalnum() or not new_words[idx + 1][-1].isalnum():
            continue
        new_words[idx], new_words[idx + 1] = new_words[idx + 1], new_words[idx]

    return ' '.join(new_words)


def sentence_shuffle(text: str, shuffle_ratio: float = 0.3) -> str:
    """
    Shuffle a portion of sentences in the document.
    For long legal documents where order is less critical.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) < 5:
        return text

    # Shuffle middle sentences only (preserve beginning and end)
    n_to_shuffle = max(1, int(len(sentences) * shuffle_ratio))

    # Don't touch first and last 2 sentences
    if len(sentences) > 4:
        start = sentences[:2]
        end = sentences[-2:]
        middle = sentences[2:-2]

        if len(middle) > 2:
            indices_to_shuffle = random.sample(range(len(middle)), min(n_to_shuffle, len(middle)))
            shuffled_middle = middle.copy()
            random.shuffle([shuffled_middle[i] for i in indices_to_shuffle])
            sentences = start + shuffled_middle + end

    return ' '.join(sentences)


def word_perturbation(text: str, p: float = 0.02) -> str:
    """
    Simulate back-translation by slightly perturbing words.
    Inserts, deletes, or swaps characters within words.
    Very conservative (2%) to preserve meaning.
    """
    words = text.split()
    new_words = []

    for word in words:
        if len(word) > 5 and random.random() < p:
            # Choose perturbation type
            perturb_type = random.choice(['swap', 'delete', 'duplicate'])

            chars = list(word)
            idx = random.randint(1, len(chars) - 2)  # Not first/last char

            if perturb_type == 'swap' and idx < len(chars) - 1:
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            elif perturb_type == 'delete':
                chars.pop(idx)
            elif perturb_type == 'duplicate':
                chars.insert(idx, chars[idx])

            word = ''.join(chars)

        new_words.append(word)

    return ' '.join(new_words)


def augment_text(text: str, augmentation_type: str = 'all') -> str:
    """
    Apply augmentation to a single text.

    Args:
        text: Input text
        augmentation_type: 'synonym', 'deletion', 'swap', 'shuffle', 'perturb', or 'all'

    Returns:
        Augmented text
    """
    if augmentation_type == 'synonym':
        return synonym_replacement(text)
    elif augmentation_type == 'deletion':
        return random_deletion(text)
    elif augmentation_type == 'swap':
        return random_swap(text)
    elif augmentation_type == 'shuffle':
        return sentence_shuffle(text)
    elif augmentation_type == 'perturb':
        return word_perturbation(text)
    elif augmentation_type == 'all':
        # Apply multiple augmentations in sequence
        text = synonym_replacement(text, n_replacements=2)
        text = random_deletion(text, p=0.03)
        text = random_swap(text, n_swaps=2)
        return text
    else:
        return text


def augment_minority_classes(
    texts: List[str],
    labels: List[int],
    min_samples: int = 50,
    target_samples: int = 100,
    augmentation_types: List[str] = ['synonym', 'deletion', 'swap', 'all']
) -> Tuple[List[str], List[int]]:
    """
    Augment minority classes to reach target_samples.

    Args:
        texts: List of document texts
        labels: List of corresponding labels
        min_samples: Only augment classes with fewer than this many samples
        target_samples: Target number of samples per minority class
        augmentation_types: List of augmentation methods to use

    Returns:
        Tuple of (augmented_texts, augmented_labels)
    """
    # Count samples per class
    label_counts = Counter(labels)

    # Identify minority classes
    minority_classes = {label for label, count in label_counts.items() if count < min_samples}

    print(f"\n📊 Data Augmentation Summary:")
    print(f"  Total classes: {len(label_counts)}")
    print(f"  Minority classes (<{min_samples} samples): {len(minority_classes)}")

    # Create augmented data
    augmented_texts = list(texts)
    augmented_labels = list(labels)

    for label in minority_classes:
        current_count = label_counts[label]
        needed = target_samples - current_count

        if needed <= 0:
            continue

        # Get all samples of this class
        class_indices = [i for i, l in enumerate(labels) if l == label]
        class_texts = [texts[i] for i in class_indices]

        print(f"  Class {label}: {current_count} -> {target_samples} (+{needed} augmented)")

        # Generate augmented samples
        for _ in range(needed):
            # Pick a random original sample
            original_text = random.choice(class_texts)

            # Pick a random augmentation type
            aug_type = random.choice(augmentation_types)

            # Augment
            augmented_text = augment_text(original_text, aug_type)

            augmented_texts.append(augmented_text)
            augmented_labels.append(label)

    print(f"\n  Total samples before: {len(texts)}")
    print(f"  Total samples after: {len(augmented_texts)}")
    print(f"  Added: {len(augmented_texts) - len(texts)} augmented samples")

    return augmented_texts, augmented_labels


def create_balanced_subset(
    texts: List[str],
    labels: List[int],
    samples_per_class: int = 100
) -> Tuple[List[str], List[int]]:
    """
    Create a balanced subset by downsampling majority classes
    and upsampling minority classes.

    Useful for two-stage training:
    - Stage 1: Train on balanced subset
    - Stage 2: Fine-tune on full dataset

    Args:
        texts: List of document texts
        labels: List of corresponding labels
        samples_per_class: Target samples per class

    Returns:
        Tuple of (balanced_texts, balanced_labels)
    """
    label_counts = Counter(labels)

    balanced_texts = []
    balanced_labels = []

    for label in label_counts.keys():
        class_indices = [i for i, l in enumerate(labels) if l == label]
        class_texts = [texts[i] for i in class_indices]
        current_count = len(class_texts)

        if current_count >= samples_per_class:
            # Downsample majority class
            selected = random.sample(class_texts, samples_per_class)
        else:
            # Upsample minority class with augmentation
            selected = class_texts.copy()
            while len(selected) < samples_per_class:
                original = random.choice(class_texts)
                aug_type = random.choice(['synonym', 'deletion', 'swap'])
                augmented = augment_text(original, aug_type)
                selected.append(augmented)

        balanced_texts.extend(selected)
        balanced_labels.extend([label] * len(selected))

    # Shuffle
    combined = list(zip(balanced_texts, balanced_labels))
    random.shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)

    print(f"\n📊 Balanced Subset Created:")
    print(f"  Target per class: {samples_per_class}")
    print(f"  Total samples: {len(balanced_texts)}")
    print(f"  Classes: {len(label_counts)}")

    return list(balanced_texts), list(balanced_labels)


if __name__ == "__main__":
    # Test augmentation
    test_text = """
    De notaris verklaart dat de akte van levering is opgemaakt.
    Het perceel is gelegen aan de Hoofdstraat 123 te Amsterdam.
    De koopsom bedraagt tweehonderdduizend euro.
    De koper en verkoper zijn beide aanwezig bij het tekenen.
    """

    print("Original text:")
    print(test_text)
    print("\n" + "=" * 50)

    print("\nSynonym replacement:")
    print(synonym_replacement(test_text))

    print("\nRandom deletion:")
    print(random_deletion(test_text))

    print("\nRandom swap:")
    print(random_swap(test_text))

    print("\nCombined augmentation:")
    print(augment_text(test_text, 'all'))
