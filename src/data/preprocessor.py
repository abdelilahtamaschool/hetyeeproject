"""
Text preprocessing and chunking utilities for long documents.
"""
import re
from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer


def clean_text(text: str) -> str:
    """
    Clean and normalize Dutch legal text.

    Args:
        text: Raw document text

    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")

    # Remove control characters but keep newlines for structure
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

    return text.strip()


def sliding_window_chunk(
    text: str,
    tokenizer: PreTrainedTokenizer,
    chunk_size: int = 510,
    overlap: int = 128,
    return_offsets: bool = False
) -> List[Dict]:
    """
    Split long documents into overlapping chunks using sliding window.

    For documents exceeding BERT's 512 token limit, this creates
    overlapping chunks to maintain context continuity.

    Args:
        text: Document text to chunk
        tokenizer: HuggingFace tokenizer
        chunk_size: Maximum tokens per chunk (excluding special tokens)
        overlap: Number of overlapping tokens between consecutive chunks
        return_offsets: Whether to return character offsets

    Returns:
        List of chunk dictionaries with input_ids, attention_mask, and metadata
    """
    # Tokenize the full text without special tokens
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=return_offsets
    )

    tokens = encoding['input_ids']

    # If text fits in single chunk, return as-is
    if len(tokens) <= chunk_size:
        full_encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=chunk_size + 2,  # +2 for [CLS] and [SEP]
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return [{
            'input_ids': full_encoding['input_ids'].squeeze(0),
            'attention_mask': full_encoding['attention_mask'].squeeze(0),
            'chunk_index': 0,
            'num_chunks': 1,
            'is_first': True,
            'is_last': True
        }]

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        # Add special tokens
        chunk_ids = [tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]

        # Pad if necessary
        padding_length = chunk_size + 2 - len(chunk_ids)
        attention_mask = [1] * len(chunk_ids) + [0] * padding_length
        chunk_ids = chunk_ids + [tokenizer.pad_token_id] * padding_length

        chunks.append({
            'input_ids': chunk_ids,
            'attention_mask': attention_mask,
            'chunk_index': chunk_index,
            'is_first': start == 0,
            'is_last': end >= len(tokens),
            'start_token': start,
            'end_token': end
        })

        # Move window
        start += chunk_size - overlap
        chunk_index += 1

        if end >= len(tokens):
            break

    # Add num_chunks to all chunks
    for chunk in chunks:
        chunk['num_chunks'] = len(chunks)

    return chunks


def first_last_chunk(
    text: str,
    tokenizer: PreTrainedTokenizer,
    first_size: int = 256,
    last_size: int = 256
) -> List[Dict]:
    """
    Alternative chunking strategy: take first and last portions.

    Useful for legal documents where key information often appears
    at the beginning (parties, subject) and end (signatures, dates).

    Args:
        text: Document text
        tokenizer: HuggingFace tokenizer
        first_size: Number of tokens from the beginning
        last_size: Number of tokens from the end

    Returns:
        List with two chunk dictionaries (first and last)
    """
    encoding = tokenizer(text, add_special_tokens=False)
    tokens = encoding['input_ids']

    chunks = []
    total_size = first_size + last_size

    if len(tokens) <= total_size:
        # Document fits, return single chunk
        full_encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return [{
            'input_ids': full_encoding['input_ids'].squeeze(0),
            'attention_mask': full_encoding['attention_mask'].squeeze(0),
            'chunk_index': 0,
            'num_chunks': 1,
            'is_first': True,
            'is_last': True
        }]

    # First chunk
    first_tokens = tokens[:first_size]
    first_ids = [tokenizer.cls_token_id] + first_tokens + [tokenizer.sep_token_id]
    padding_length = first_size + 2 - len(first_ids)
    first_mask = [1] * len(first_ids) + [0] * padding_length
    first_ids = first_ids + [tokenizer.pad_token_id] * padding_length

    chunks.append({
        'input_ids': first_ids,
        'attention_mask': first_mask,
        'chunk_index': 0,
        'num_chunks': 2,
        'is_first': True,
        'is_last': False,
        'position': 'first'
    })

    # Last chunk
    last_tokens = tokens[-last_size:]
    last_ids = [tokenizer.cls_token_id] + last_tokens + [tokenizer.sep_token_id]
    padding_length = last_size + 2 - len(last_ids)
    last_mask = [1] * len(last_ids) + [0] * padding_length
    last_ids = last_ids + [tokenizer.pad_token_id] * padding_length

    chunks.append({
        'input_ids': last_ids,
        'attention_mask': last_mask,
        'chunk_index': 1,
        'num_chunks': 2,
        'is_first': False,
        'is_last': True,
        'position': 'last'
    })

    return chunks


def estimate_token_count(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """
    Quickly estimate the number of tokens in a text.

    Args:
        text: Input text
        tokenizer: HuggingFace tokenizer

    Returns:
        Estimated token count
    """
    return len(tokenizer.encode(text, add_special_tokens=False))


def get_chunking_statistics(
    documents: List[Dict],
    tokenizer: PreTrainedTokenizer,
    chunk_size: int = 510,
    overlap: int = 128
) -> Dict:
    """
    Get statistics about chunking requirements for the dataset.

    Args:
        documents: List of document dictionaries
        tokenizer: HuggingFace tokenizer
        chunk_size: Chunk size for sliding window
        overlap: Overlap between chunks

    Returns:
        Dictionary with chunking statistics
    """
    from tqdm import tqdm

    token_counts = []
    chunk_counts = []

    for doc in tqdm(documents, desc="Analyzing documents"):
        text = doc.get('text', '')
        num_tokens = estimate_token_count(text, tokenizer)
        token_counts.append(num_tokens)

        # Calculate number of chunks needed
        if num_tokens <= chunk_size:
            num_chunks = 1
        else:
            num_chunks = 1 + (num_tokens - chunk_size - 1) // (chunk_size - overlap) + 1
        chunk_counts.append(num_chunks)

    exceeds_512 = sum(1 for t in token_counts if t > 512)

    return {
        'total_documents': len(documents),
        'documents_exceeding_512': exceeds_512,
        'percentage_exceeding_512': exceeds_512 / len(documents) * 100,
        'avg_tokens': sum(token_counts) / len(token_counts),
        'max_tokens': max(token_counts),
        'min_tokens': min(token_counts),
        'avg_chunks_per_doc': sum(chunk_counts) / len(chunk_counts),
        'max_chunks_per_doc': max(chunk_counts),
        'total_chunks': sum(chunk_counts)
    }
