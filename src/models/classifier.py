"""
Multi-label BERTje classifier for Dutch legal document classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig
from typing import Dict, Optional, Tuple


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced multi-label classification.

    Focuses training on hard-to-classify examples by down-weighting
    well-classified examples.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            pos_weight: Per-class positive weights for class imbalance
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Logits of shape (batch_size, num_labels)
            targets: Binary targets of shape (batch_size, num_labels)

        Returns:
            Focal loss value
        """
        # Compute BCE loss without reduction
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(inputs.device)
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=pos_weight, reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )

        # Compute probabilities
        probs = torch.sigmoid(inputs)

        # Compute pt (probability of correct class)
        pt = targets * probs + (1 - targets) * (1 - probs)

        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Combine
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AkteClassifier(nn.Module):
    """
    Multi-label classifier for Dutch legal documents using BERTje.

    Handles long documents via chunk aggregation strategies.
    """

    def __init__(
        self,
        model_name: str = "GroNLP/bert-base-dutch-cased",
        num_labels: int = 20,
        aggregation: str = "max",
        dropout: float = 0.1
    ):
        """
        Initialize the classifier.

        Args:
            model_name: HuggingFace model name (BERTje)
            num_labels: Number of rechtsfeitcodes to classify
            aggregation: Chunk aggregation strategy ('max', 'mean', 'attention')
            dropout: Dropout probability for classification head
        """
        super().__init__()

        self.num_labels = num_labels
        self.aggregation = aggregation

        # Load BERTje base model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)

        # Optional attention aggregation layer
        if aggregation == "attention":
            self.chunk_attention = nn.Sequential(
                nn.Linear(self.hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with chunk aggregation.

        Args:
            input_ids: Tensor of shape (batch_size, num_chunks, seq_len)
            attention_mask: Tensor of shape (batch_size, num_chunks, seq_len)
            chunk_mask: Tensor of shape (batch_size, num_chunks) indicating valid chunks
            labels: Optional tensor of shape (batch_size, num_labels) for training

        Returns:
            Dictionary with 'loss' (if labels provided) and 'logits'
        """
        batch_size, num_chunks, seq_len = input_ids.shape

        # Reshape for BERT: (batch_size * num_chunks, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)

        # Get BERT outputs
        outputs = self.bert(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask
        )

        # Get [CLS] token embeddings: (batch_size * num_chunks, hidden_size)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Reshape back: (batch_size, num_chunks, hidden_size)
        cls_embeddings = cls_embeddings.view(batch_size, num_chunks, -1)

        # Aggregate chunks
        if chunk_mask is None:
            chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool, device=input_ids.device)

        aggregated = self._aggregate_chunks(cls_embeddings, chunk_mask)

        # Classification
        pooled = self.dropout(aggregated)
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits
        }

    def _aggregate_chunks(
        self,
        embeddings: torch.Tensor,
        chunk_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate chunk embeddings into a single document representation.

        Args:
            embeddings: Tensor of shape (batch_size, num_chunks, hidden_size)
            chunk_mask: Tensor of shape (batch_size, num_chunks)

        Returns:
            Aggregated tensor of shape (batch_size, hidden_size)
        """
        # Expand mask for broadcasting
        mask = chunk_mask.unsqueeze(-1).float()  # (batch_size, num_chunks, 1)

        if self.aggregation == "max":
            # Masked max pooling
            embeddings = embeddings.masked_fill(~chunk_mask.unsqueeze(-1), float('-inf'))
            aggregated = embeddings.max(dim=1)[0]

        elif self.aggregation == "mean":
            # Masked mean pooling
            embeddings = embeddings * mask
            aggregated = embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        elif self.aggregation == "attention":
            # Attention-weighted aggregation
            attention_scores = self.chunk_attention(embeddings).squeeze(-1)  # (batch_size, num_chunks)
            attention_scores = attention_scores.masked_fill(~chunk_mask, float('-inf'))
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, num_chunks)
            aggregated = (embeddings * attention_weights.unsqueeze(-1)).sum(dim=1)

        else:
            raise ValueError(f"Unknown aggregation strategy: {self.aggregation}")

        return aggregated

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with threshold.

        Args:
            input_ids: Input tensor
            attention_mask: Attention mask
            chunk_mask: Optional chunk mask
            threshold: Classification threshold

        Returns:
            Tuple of (predictions, probabilities)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask, chunk_mask)
            logits = outputs['logits']
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > threshold).int()

        return predictions, probabilities


class SimpleAkteClassifier(nn.Module):
    """
    Simplified classifier for chunk-level training.

    Each chunk is classified independently; aggregation happens at inference.
    """

    def __init__(
        self,
        model_name: str = "GroNLP/bert-base-dutch-cased",
        num_labels: int = 20
    ):
        """
        Initialize the simple classifier.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of labels
        """
        super().__init__()

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single chunks.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            attention_mask: Tensor of shape (batch_size, seq_len)
            labels: Optional tensor of shape (batch_size, num_labels)

        Returns:
            Dictionary with 'loss' and 'logits'
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            'loss': outputs.loss,
            'logits': outputs.logits
        }


def load_classifier(
    checkpoint_path: str,
    num_labels: int = 20,
    aggregation: str = "max"
) -> AkteClassifier:
    """
    Load a trained classifier from checkpoint.

    Args:
        checkpoint_path: Path to the saved model
        num_labels: Number of labels
        aggregation: Aggregation strategy

    Returns:
        Loaded AkteClassifier model
    """
    model = AkteClassifier(
        num_labels=num_labels,
        aggregation=aggregation
    )

    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)

    return model
