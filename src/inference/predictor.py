"""
End-to-end inference pipeline for Dutch legal document classification.
"""
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

from ..config import Config, InferenceConfig
from ..data.preprocessor import sliding_window_chunk, clean_text
from ..models.classifier import AkteClassifier, SimpleAkteClassifier, load_classifier
from .ner_extractor import DutchLegalNER


class AktePredictor:
    """
    Production inference pipeline for Dutch legal document classification.

    Combines multi-label classification with rule-based NER for complete
    document analysis.
    """

    def __init__(
        self,
        classifier_path: str,
        label2id: Dict[int, int],
        id2label: Dict[int, int],
        config: Optional[Config] = None,
        threshold: float = 0.5,
        use_spacy_ner: bool = False,
        device: str = None
    ):
        """
        Initialize the predictor.

        Args:
            classifier_path: Path to trained classifier checkpoint
            label2id: Mapping from rechtsfeitcode to index
            id2label: Mapping from index to rechtsfeitcode
            config: Configuration (uses defaults if not provided)
            threshold: Classification threshold
            use_spacy_ner: Whether to use spaCy for NER
            device: Device to use (cuda/cpu)
        """
        self.config = config or Config()
        self.label2id = label2id
        self.id2label = id2label
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name
        )

        # Load classifier
        self.classifier = load_classifier(
            classifier_path,
            num_labels=len(label2id),
            aggregation=self.config.chunking.aggregation
        )
        self.classifier.to(self.device)
        self.classifier.eval()

        # Initialize NER
        self.ner = DutchLegalNER(use_spacy=use_spacy_ner)

    def predict(self, document: Dict) -> Dict:
        """
        Make predictions for a single document.

        Args:
            document: Dictionary with 'text' and optionally 'akteId'

        Returns:
            Prediction dictionary with classification and entities
        """
        text = document.get('text', '')
        akte_id = document.get('akteId', '')

        # Clean text
        cleaned_text = clean_text(text)

        # Classification
        classification = self._classify(cleaned_text)

        # NER
        entities = self.ner.extract_entities(text)
        entities_dict = self.ner.to_dict(entities)

        # Determine if review is needed
        requires_review, review_reason = self._should_flag_for_review(
            classification,
            entities_dict
        )

        return {
            "akte_id": akte_id,
            "classification": classification,
            "entities": entities_dict,
            "requires_review": requires_review,
            "review_reason": review_reason,
            "processing_timestamp": datetime.now().isoformat()
        }

    def _classify(self, text: str) -> Dict:
        """
        Classify the document.

        Args:
            text: Cleaned document text

        Returns:
            Classification results dictionary
        """
        # Create chunks
        chunks = sliding_window_chunk(
            text,
            self.tokenizer,
            chunk_size=self.config.chunking.chunk_size,
            overlap=self.config.chunking.overlap
        )

        # Prepare tensors
        input_ids = torch.tensor(
            [c['input_ids'] for c in chunks],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)  # Add batch dimension

        attention_mask = torch.tensor(
            [c['attention_mask'] for c in chunks],
            dtype=torch.long
        ).unsqueeze(0).to(self.device)

        chunk_mask = torch.ones(
            1, len(chunks),
            dtype=torch.bool,
            device=self.device
        )

        # Get predictions
        with torch.no_grad():
            outputs = self.classifier(
                input_ids=input_ids,
                attention_mask=attention_mask,
                chunk_mask=chunk_mask
            )
            logits = outputs['logits']
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # Get predictions above threshold
        predicted_indices = (probabilities > self.threshold).nonzero()[0].tolist()

        # Build result
        predicted_codes = [self.id2label[idx] for idx in predicted_indices]
        confidence_scores = {
            str(self.id2label[idx]): float(probabilities[idx])
            for idx in predicted_indices
        }

        # Calculate overall confidence
        if predicted_indices:
            avg_confidence = sum(probabilities[idx] for idx in predicted_indices) / len(predicted_indices)
        else:
            avg_confidence = 0.0

        return {
            "rechtsfeitcodes": predicted_codes,
            "confidence_scores": confidence_scores,
            "overall_confidence": float(avg_confidence),
            "threshold_used": self.threshold,
            "num_chunks_processed": len(chunks)
        }

    def _should_flag_for_review(
        self,
        classification: Dict,
        entities: Dict
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if the document should be flagged for human review.

        Args:
            classification: Classification results
            entities: Extracted entities

        Returns:
            Tuple of (should_flag, reason)
        """
        reasons = []

        # Low overall confidence
        if classification['overall_confidence'] < self.config.inference.min_confidence_for_review:
            reasons.append(
                f"Low confidence: {classification['overall_confidence']:.2f}"
            )

        # No rechtsfeitcodes predicted
        if not classification['rechtsfeitcodes']:
            reasons.append("No rechtsfeitcodes predicted")

        # No subjects found
        if not entities.get('subjects'):
            reasons.append("No subjects (persons) identified")

        # No cadastral references in property-related documents
        if not entities.get('properties'):
            # Check if any predicted code typically involves property
            property_related_codes = {537, 564, 585}  # Common property codes
            predicted_set = set(classification['rechtsfeitcodes'])
            if predicted_set & property_related_codes:
                reasons.append("Property-related code but no cadastral reference found")

        # Too many high-confidence predictions (unusual)
        high_conf_count = sum(
            1 for score in classification['confidence_scores'].values()
            if score > 0.9
        )
        if high_conf_count > 5:
            reasons.append(f"Unusually many high-confidence predictions: {high_conf_count}")

        if reasons:
            return True, "; ".join(reasons)
        return False, None

    def predict_batch(
        self,
        documents: List[Dict],
        batch_size: int = 16,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Make predictions for multiple documents.

        Args:
            documents: List of document dictionaries
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of prediction dictionaries
        """
        results = []

        iterator = tqdm(documents, desc="Processing documents") if show_progress else documents

        for doc in iterator:
            result = self.predict(doc)
            results.append(result)

        return results

    def export_results(
        self,
        results: List[Dict],
        output_path: str,
        model_version: str = "1.0.0"
    ):
        """
        Export results to JSON file in Kadaster format.

        Args:
            results: List of prediction results
            output_path: Output file path
            model_version: Model version string
        """
        # Calculate summary statistics
        total = len(results)
        flagged = sum(1 for r in results if r['requires_review'])
        avg_confidence = sum(
            r['classification']['overall_confidence'] for r in results
        ) / total if total > 0 else 0

        output = {
            "processing_timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "model_config": {
                "model_name": self.config.model.model_name,
                "threshold": self.threshold,
                "num_labels": len(self.label2id)
            },
            "summary": {
                "total_processed": total,
                "flagged_for_review": flagged,
                "flagged_percentage": flagged / total * 100 if total > 0 else 0,
                "average_confidence": avg_confidence
            },
            "documents": results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Results exported to {output_path}")
        print(f"  Total: {total}")
        print(f"  Flagged for review: {flagged} ({flagged/total*100:.1f}%)")
        print(f"  Average confidence: {avg_confidence:.3f}")


class SimplePredictorFromTrainer:
    """
    Simple predictor that loads a model saved by HuggingFace Trainer.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "GroNLP/bert-base-dutch-cased",
        label2id: Dict[int, int] = None,
        id2label: Dict[int, int] = None,
        threshold: float = 0.5,
        device: str = None
    ):
        """
        Initialize from Trainer checkpoint.

        Args:
            model_path: Path to saved model directory
            tokenizer_name: Tokenizer to use
            label2id: Label mapping
            id2label: Inverse label mapping
            threshold: Classification threshold
            device: Device to use
        """
        from transformers import AutoModelForSequenceClassification

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.label2id = label2id or {}
        self.id2label = id2label or {}
        self.threshold = threshold

        # NER
        self.ner = DutchLegalNER(use_spacy=False)

    def predict_chunk(self, text: str) -> Dict:
        """
        Predict for a single chunk of text (< 512 tokens).

        Args:
            text: Text to classify

        Returns:
            Prediction dictionary
        """
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).squeeze(0).cpu().numpy()

        predicted_indices = (probabilities > self.threshold).nonzero()[0].tolist()
        predicted_codes = [self.id2label.get(idx, idx) for idx in predicted_indices]

        return {
            "rechtsfeitcodes": predicted_codes,
            "probabilities": {
                str(self.id2label.get(i, i)): float(p)
                for i, p in enumerate(probabilities)
                if p > self.threshold
            }
        }


def load_predictor(
    checkpoint_dir: str,
    config_path: Optional[str] = None
) -> AktePredictor:
    """
    Load a predictor from a checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        config_path: Optional path to config file

    Returns:
        Configured AktePredictor
    """
    checkpoint_path = Path(checkpoint_dir)

    # Load label mappings
    label_mapping_path = checkpoint_path / "label_mapping.json"
    if label_mapping_path.exists():
        with open(label_mapping_path) as f:
            mapping = json.load(f)
            label2id = {int(k): v for k, v in mapping['label2id'].items()}
            id2label = {v: int(k) for k, v in mapping['label2id'].items()}
    else:
        raise FileNotFoundError(f"Label mapping not found at {label_mapping_path}")

    # Load config if provided
    config = None
    if config_path:
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            config = Config(**config_dict)

    # Find model checkpoint
    model_path = checkpoint_path / "best_model.pt"
    if not model_path.exists():
        model_path = checkpoint_path / "pytorch_model.bin"

    return AktePredictor(
        classifier_path=str(model_path),
        label2id=label2id,
        id2label=id2label,
        config=config
    )
