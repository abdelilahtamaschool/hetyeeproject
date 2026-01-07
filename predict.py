"""
Inference script for the Akte Classification Pipeline.

Usage:
    python predict.py --model_dir <model_dir> --input <input_file> --output <output_file>
"""
import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.data.loader import load_jsonl
from src.inference.predictor import SimplePredictorFromTrainer
from src.inference.ner_extractor import DutchLegalNER


def main():
    parser = argparse.ArgumentParser(description="Predict with Akte Classifier")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents"
    )
    parser.add_argument(
        "--use_spacy",
        action="store_true",
        help="Use spaCy for additional NER"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Akte Classification Pipeline - Inference")
    print("=" * 60)

    # Load label mapping
    model_dir = Path(args.model_dir)
    mapping_path = model_dir.parent / "label_mapping.json"

    if not mapping_path.exists():
        mapping_path = model_dir / "label_mapping.json"

    print(f"\n1. Loading label mapping from {mapping_path}...")
    with open(mapping_path) as f:
        mapping = json.load(f)
        label2id = {int(k): v for k, v in mapping['label2id'].items()}
        id2label = {v: int(k) for k, v in mapping['label2id'].items()}
    print(f"   {len(label2id)} labels loaded")

    # Initialize predictor
    print(f"\n2. Loading model from {args.model_dir}...")
    predictor = SimplePredictorFromTrainer(
        model_path=args.model_dir,
        label2id=label2id,
        id2label=id2label,
        threshold=args.threshold
    )
    print(f"   Model loaded on {predictor.device}")

    # Initialize NER
    print(f"\n3. Initializing NER extractor...")
    ner = DutchLegalNER(use_spacy=args.use_spacy)
    print(f"   spaCy: {args.use_spacy}")

    # Load documents
    print(f"\n4. Loading documents from {args.input}...")
    documents = load_jsonl(Path(args.input), limit=args.limit)
    print(f"   {len(documents)} documents loaded")

    # Process documents
    print(f"\n5. Processing documents...")
    results = []

    for doc in tqdm(documents, desc="Predicting"):
        text = doc.get('text', '')
        akte_id = doc.get('akteId', '')

        # Classification (using first 512 tokens for simple predictor)
        classification = predictor.predict_chunk(text[:10000])  # Limit text length

        # NER
        entities = ner.extract_entities(text)
        entities_dict = ner.to_dict(entities)

        # Determine review flag
        requires_review = False
        review_reason = None

        if not classification['rechtsfeitcodes']:
            requires_review = True
            review_reason = "No rechtsfeitcodes predicted"
        elif not entities_dict.get('subjects'):
            requires_review = True
            review_reason = "No subjects identified"

        results.append({
            "akte_id": akte_id,
            "classification": classification,
            "entities": entities_dict,
            "requires_review": requires_review,
            "review_reason": review_reason
        })

    # Export results
    print(f"\n6. Exporting results to {args.output}...")

    total = len(results)
    flagged = sum(1 for r in results if r['requires_review'])

    output = {
        "model_info": {
            "model_path": str(args.model_dir),
            "threshold": args.threshold,
            "num_labels": len(label2id)
        },
        "summary": {
            "total_processed": total,
            "flagged_for_review": flagged,
            "flagged_percentage": flagged / total * 100 if total > 0 else 0
        },
        "documents": results
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n   Results saved!")
    print(f"   - Total: {total}")
    print(f"   - Flagged: {flagged} ({flagged/total*100:.1f}%)")

    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
