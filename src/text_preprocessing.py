"""
Text Preprocessing for Kadaster Legal Documents
Optimized cleaning and normalization based on data analysis findings.

Key findings addressed:
- 100% of documents contain dates (need normalization)
- 78% contain money amounts (need normalization)
- 91% contain article references
- 99% contain ALL_CAPS words
- Average 575 special characters per document
- Average 291 digits per document
"""

import re
from typing import Optional


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace to single spaces.
    Removes tabs, multiple spaces, and normalizes newlines.
    """
    # Replace tabs and multiple spaces with single space
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r' {2,}', ' ', text)

    # Normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def normalize_dates(text: str) -> str:
    """
    Normalize date formats to consistent format.
    Preserves structure while removing variation.
    """
    # Pattern: DD-MM-YYYY or DD/MM/YYYY
    text = re.sub(r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})', r'\1-\2-\3', text)

    # Pattern: written dates like "1 januari 2024"
    months = {
        'januari': '01', 'februari': '02', 'maart': '03', 'april': '04',
        'mei': '05', 'juni': '06', 'juli': '07', 'augustus': '08',
        'september': '09', 'oktober': '10', 'november': '11', 'december': '12'
    }
    for month_name, month_num in months.items():
        pattern = rf'(\d{{1,2}})\s+{month_name}\s+(\d{{4}})'
        text = re.sub(pattern, rf'\1-{month_num}-\2', text, flags=re.IGNORECASE)

    return text


def normalize_money(text: str) -> str:
    """
    Normalize money amounts to consistent format.
    """
    # Euro symbol variations
    text = re.sub(r'€\s*', 'EUR ', text)
    text = re.sub(r'(\d+)[.,](\d{3})[.,](\d{2})\s*(euro|EUR)', r'EUR \1\2.\3', text, flags=re.IGNORECASE)

    # Simple amounts like "100.000 euro"
    text = re.sub(r'(\d+)[.,](\d{3})\s*(euro|EUR)', r'EUR \1\2', text, flags=re.IGNORECASE)

    return text


def normalize_percentages(text: str) -> str:
    """
    Normalize percentage formats.
    """
    # "5%" or "5 %" -> "5%"
    text = re.sub(r'(\d+[.,]?\d*)\s*%', r'\1%', text)
    return text


def normalize_article_references(text: str) -> str:
    """
    Normalize article references.
    """
    # "artikel 5" -> "ARTIKEL_5"
    text = re.sub(r'artikel\s+(\d+)', r'ARTIKEL_\1', text, flags=re.IGNORECASE)
    # "art. 5" -> "ARTIKEL_5"
    text = re.sub(r'art\.\s*(\d+)', r'ARTIKEL_\1', text, flags=re.IGNORECASE)
    return text


def clean_special_characters(text: str) -> str:
    """
    Clean up special characters while preserving meaning.
    """
    # Remove repeated underscores (often used as dividers)
    text = re.sub(r'_{3,}', '', text)

    # Remove repeated dashes
    text = re.sub(r'-{3,}', ' ', text)

    # Remove repeated equals signs
    text = re.sub(r'={3,}', '', text)

    # Normalize quotes
    text = re.sub(r'[""„]', '"', text)
    text = re.sub(r"['']", "'", text)

    # Remove non-breaking spaces
    text = text.replace('\xa0', ' ')

    return text


def remove_headers_footers(text: str, pattern: Optional[str] = None) -> str:
    """
    Remove common headers/footers from legal documents.
    """
    # Remove page numbers like "Pagina 1 van 10"
    text = re.sub(r'[Pp]agina\s+\d+\s+(van|/)\s+\d+', '', text)

    # Remove "blad X" patterns
    text = re.sub(r'[Bb]lad\s+\d+', '', text)

    # Remove standalone numbers that look like page numbers (at start/end of lines)
    text = re.sub(r'^\d{1,3}\s*$', '', text, flags=re.MULTILINE)

    return text


def normalize_legal_entities(text: str) -> str:
    """
    Normalize legal entity types for consistency.
    """
    # B.V. / BV / b.v.
    text = re.sub(r'[Bb]\.?\s*[Vv]\.?', 'BV', text)

    # N.V. / NV / n.v.
    text = re.sub(r'[Nn]\.?\s*[Vv]\.?', 'NV', text)

    return text


def truncate_to_max_tokens(text: str, max_tokens: int = 8192, chars_per_token: float = 4.0) -> str:
    """
    Truncate text to approximately max_tokens.
    Preserves complete sentences.
    """
    max_chars = int(max_tokens * chars_per_token)

    if len(text) <= max_chars:
        return text

    # Find a good break point (end of sentence)
    truncated = text[:max_chars]

    # Find last sentence end
    last_sentence_end = max(
        truncated.rfind('. '),
        truncated.rfind('.\n'),
        truncated.rfind('? '),
        truncated.rfind('! ')
    )

    if last_sentence_end > max_chars * 0.8:  # If we can keep at least 80%
        truncated = truncated[:last_sentence_end + 1]

    return truncated


def preprocess_legal_text(
    text: str,
    normalize_dates_flag: bool = True,
    normalize_money_flag: bool = True,
    normalize_percentages_flag: bool = True,
    normalize_articles_flag: bool = True,
    clean_special_flag: bool = True,
    remove_headers_flag: bool = True,
    normalize_entities_flag: bool = True,
    max_tokens: Optional[int] = None
) -> str:
    """
    Full preprocessing pipeline for legal documents.

    Args:
        text: Input text
        normalize_*: Flags to enable/disable specific normalizations
        max_tokens: Optional token limit for truncation

    Returns:
        Preprocessed text
    """
    # Always do whitespace normalization first
    text = normalize_whitespace(text)

    if remove_headers_flag:
        text = remove_headers_footers(text)

    if clean_special_flag:
        text = clean_special_characters(text)

    if normalize_dates_flag:
        text = normalize_dates(text)

    if normalize_money_flag:
        text = normalize_money(text)

    if normalize_percentages_flag:
        text = normalize_percentages(text)

    if normalize_articles_flag:
        text = normalize_article_references(text)

    if normalize_entities_flag:
        text = normalize_legal_entities(text)

    # Final whitespace cleanup
    text = normalize_whitespace(text)

    # Optional truncation
    if max_tokens:
        text = truncate_to_max_tokens(text, max_tokens)

    return text


def preprocess_batch(texts: list, **kwargs) -> list:
    """
    Preprocess a batch of texts.

    Args:
        texts: List of texts
        **kwargs: Arguments for preprocess_legal_text

    Returns:
        List of preprocessed texts
    """
    return [preprocess_legal_text(text, **kwargs) for text in texts]


if __name__ == "__main__":
    # Test preprocessing
    test_text = """
    AKTE VAN LEVERING
    _______________________________________________

    Pagina 1 van 10

    Heden, de 15 januari 2024, verscheen voor mij,
    mr. J. de Notaris, notaris te Amsterdam:

    De verkoper, handelend namens XYZ Vastgoed B.V.,
    verklaart te hebben verkocht aan de koper voor een
    koopsom van € 250.000,00 (tweehonderdvijftigduizend euro)
    het perceel gelegen aan de Hoofdstraat 123.

    Artikel 5 van de koopovereenkomst bepaalt dat...

    De hypotheekrente bedraagt 4,5% per jaar.

    Blad 2
    """

    print("Original text:")
    print(test_text)
    print("\n" + "=" * 60)

    print("\nPreprocessed text:")
    processed = preprocess_legal_text(test_text)
    print(processed)

    print("\n" + "=" * 60)
    print(f"\nOriginal length: {len(test_text)} chars")
    print(f"Processed length: {len(processed)} chars")
