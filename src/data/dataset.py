"""
PyTorch Dataset classes for the Akte Classification Pipeline.
"""
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Callable
from transformers import PreTrainedTokenizer
from tqdm import tqdm

from .preprocessor import sliding_window_chunk, clean_text


class AkteDataset(Dataset):
    """
    PyTorch Dataset for multi-label classification of Dutch legal documents.

    Handles document chunking for long texts and creates multi-hot label vectors.
    """

    def __init__(
        self,
        documents: List[Dict],
        tokenizer: PreTrainedTokenizer,
        label2id: Dict[int, int],
        chunk_size: int = 510,
        overlap: int = 128,
        max_chunks: Optional[int] = None,
        clean_text_fn: Optional[Callable] = None
    ):
        """
        Initialize the dataset.

        Args:
            documents: List of document dictionaries with 'text' and 'rechtsfeitcodes'
            tokenizer: HuggingFace tokenizer (BERTje)
            label2id: Mapping from rechtsfeitcode to index
            chunk_size: Maximum tokens per chunk
            overlap: Overlap between chunks
            max_chunks: Optional maximum number of chunks per document
            clean_text_fn: Optional text cleaning function
        """
        self.documents = documents
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_chunks = max_chunks
        self.clean_text_fn = clean_text_fn or clean_text

        # Pre-process all documents
        self._prepare_data()

    def _prepare_data(self):
        """Pre-process and chunk all documents."""
        self.processed_data = []

        for doc in tqdm(self.documents, desc="Processing documents"):
            text = self.clean_text_fn(doc.get('text', ''))

            # Create chunks
            chunks = sliding_window_chunk(
                text,
                self.tokenizer,
                chunk_size=self.chunk_size,
                overlap=self.overlap
            )

            # Limit chunks if specified
            if self.max_chunks and len(chunks) > self.max_chunks:
                chunks = chunks[:self.max_chunks]

            # Create label vector
            labels = torch.zeros(self.num_labels, dtype=torch.float32)
            for code in doc.get('rechtsfeitcodes', []):
                if code in self.label2id:
                    labels[self.label2id[code]] = 1.0

            self.processed_data.append({
                'akte_id': doc.get('akteId', ''),
                'chunks': chunks,
                'labels': labels,
                'num_chunks': len(chunks)
            })

    def __len__(self) -> int:
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single document with all its chunks.

        Returns:
            Dictionary with:
                - input_ids: Tensor of shape (num_chunks, seq_len)
                - attention_mask: Tensor of shape (num_chunks, seq_len)
                - labels: Tensor of shape (num_labels,)
                - num_chunks: int
                - akte_id: str
        """
        item = self.processed_data[idx]
        chunks = item['chunks']

        # Stack all chunk tensors
        input_ids = torch.tensor([c['input_ids'] for c in chunks], dtype=torch.long)
        attention_mask = torch.tensor([c['attention_mask'] for c in chunks], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': item['labels'],
            'num_chunks': item['num_chunks'],
            'akte_id': item['akte_id']
        }


class AkteChunkDataset(Dataset):
    """
    Alternative dataset that returns individual chunks.

    Each chunk is a separate sample, useful for simpler training loop
    but requires aggregation during inference.
    """

    def __init__(
        self,
        documents: List[Dict],
        tokenizer: PreTrainedTokenizer,
        label2id: Dict[int, int],
        chunk_size: int = 510,
        overlap: int = 128
    ):
        """
        Initialize the chunk-level dataset.

        Args:
            documents: List of document dictionaries
            tokenizer: HuggingFace tokenizer
            label2id: Mapping from rechtsfeitcode to index
            chunk_size: Maximum tokens per chunk
            overlap: Overlap between chunks
        """
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.num_labels = len(label2id)
        self.chunk_size = chunk_size
        self.overlap = overlap

        self.chunks = []
        self._prepare_chunks(documents)

    def _prepare_chunks(self, documents: List[Dict]):
        """Flatten all documents into individual chunks."""
        for doc_idx, doc in enumerate(tqdm(documents, desc="Creating chunks")):
            text = clean_text(doc.get('text', ''))

            # Create chunks
            chunks = sliding_window_chunk(
                text,
                self.tokenizer,
                chunk_size=self.chunk_size,
                overlap=self.overlap
            )

            # Create label vector
            labels = torch.zeros(self.num_labels, dtype=torch.float32)
            for code in doc.get('rechtsfeitcodes', []):
                if code in self.label2id:
                    labels[self.label2id[code]] = 1.0

            # Add each chunk as separate sample
            for chunk in chunks:
                # Handle both tensor and list input_ids (from preprocessor)
                input_ids = chunk['input_ids']
                attention_mask = chunk['attention_mask']

                # Convert to tensor if needed, avoiding warning
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.clone().detach()
                else:
                    input_ids = torch.tensor(input_ids, dtype=torch.long)

                if isinstance(attention_mask, torch.Tensor):
                    attention_mask = attention_mask.clone().detach()
                else:
                    attention_mask = torch.tensor(attention_mask, dtype=torch.long)

                self.chunks.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk['chunk_index'],
                    'num_chunks': chunk['num_chunks'],
                    'akte_id': doc.get('akteId', '')
                })

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single chunk."""
        return self.chunks[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching documents with variable chunk counts.

    Args:
        batch: List of document dictionaries from AkteDataset

    Returns:
        Batched dictionary with padded tensors
    """
    # Find max chunks in this batch
    max_chunks = max(item['num_chunks'] for item in batch)
    seq_len = batch[0]['input_ids'].shape[1]

    batch_size = len(batch)

    # Initialize tensors
    input_ids = torch.zeros(batch_size, max_chunks, seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_chunks, seq_len, dtype=torch.long)
    chunk_mask = torch.zeros(batch_size, max_chunks, dtype=torch.bool)
    labels = torch.stack([item['labels'] for item in batch])

    # Fill tensors
    for i, item in enumerate(batch):
        num_chunks = item['num_chunks']
        input_ids[i, :num_chunks] = item['input_ids']
        attention_mask[i, :num_chunks] = item['attention_mask']
        chunk_mask[i, :num_chunks] = True

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'chunk_mask': chunk_mask,
        'labels': labels,
        'num_chunks': torch.tensor([item['num_chunks'] for item in batch]),
        'akte_ids': [item['akte_id'] for item in batch]
    }


def chunk_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for chunk-level dataset.

    Args:
        batch: List of chunk dictionaries from AkteChunkDataset

    Returns:
        Batched dictionary with only model-required fields
    """
    # Only return fields that the model expects
    # doc_idx and chunk_idx are metadata for inference, not needed during training
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }
