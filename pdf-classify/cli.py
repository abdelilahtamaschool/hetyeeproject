"""
PDF Classifier - Command Line Interface
Classificeert juridische documenten met het getrainde BERTje model.

Usage:
    python cli.py document.pdf
    python cli.py document.pdf --model models/final_model --threshold 0.4
"""
import argparse
import json
import sys
import torch
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Rechtsfeit code descriptions
RECHTSFEIT_DESCRIPTIONS = {
    "606": "Levering van een onroerende zaak",
    "537": "Hypotheek",
    "585": "Verdeling",
    "545": "Erfpacht",
    "572": "Kwantitatieve splitsing",
    "564": "Vestiging erfdienstbaarheid",
    "527": "Doorhaling hypotheek",
    "532": "Wijziging hypotheek",
    "538": "Beslag",
    "580": "Verklaring van erfrecht",
    "581": "Huurkoop",
    "644": "Kwalitatieve verplichting",
    "543": "Opstal",
    "517": "Wijziging erfpacht",
    "516": "Einde erfpacht",
    "518": "Splitsing erfpacht",
    "652": "Publiekrechtelijke beperking",
    "579": "Legaat",
    "671": "Aanvulling/wijziging splitsing",
    "616": "Vruchtgebruik"
}


def load_model(model_dir: str):
    """Load the trained model and tokenizer."""
    model_path = Path(model_dir)

    # Load label mapping - check multiple locations
    mapping_path = model_path.parent / "label_mapping.json"
    if not mapping_path.exists():
        mapping_path = model_path.parent.parent / "label_mapping.json"
    if not mapping_path.exists():
        mapping_path = Path(__file__).parent.parent / "uitslag" / "models" / "label_mapping.json"

    if mapping_path.exists():
        with open(mapping_path) as f:
            mapping = json.load(f)
            id2label = {int(k): str(v) for k, v in mapping['id2label'].items()}
    else:
        id2label = None
        print("⚠️ Label mapping niet gevonden")

    # Load tokenizer from base model (BERTje)
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

    # Load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, id2label, device


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, tokenizer, chunk_size: int = 510, overlap: int = 128):
    """Split text into overlapping chunks."""
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(chunk_tokens)

        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def classify_document(text: str, model, tokenizer, device, id2label, threshold: float = 0.5):
    """Classify a document and return predictions."""
    chunks = chunk_text(text, tokenizer)

    if not chunks:
        return []

    all_probs = []

    with torch.no_grad():
        for chunk_tokens in chunks:
            input_ids = torch.tensor([[tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]])
            attention_mask = torch.ones_like(input_ids)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            all_probs.append(probs)

    # Aggregate probabilities (max pooling)
    aggregated_probs = np.max(all_probs, axis=0)

    # Get predictions
    results = []
    for idx, prob in enumerate(aggregated_probs):
        label = id2label.get(idx, f"Label_{idx}") if id2label else f"Label_{idx}"
        results.append({
            "label": label,
            "probability": float(prob),
            "predicted": prob >= threshold,
            "description": RECHTSFEIT_DESCRIPTIONS.get(label, "Onbekend")
        })

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Classificeer juridische PDF documenten")
    parser.add_argument("input", help="PDF bestand of tekst bestand")
    parser.add_argument("--model", default="../uitslag/models/checkpoints/checkpoint-9354", help="Model directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classificatie drempel")
    parser.add_argument("--json", action="store_true", help="Output als JSON")
    parser.add_argument("--all", action="store_true", help="Toon alle labels (niet alleen gedetecteerde)")

    args = parser.parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Bestand niet gevonden: {args.input}")
        sys.exit(1)

    # Load model
    print(f"📦 Model laden van {args.model}...")
    try:
        model, tokenizer, id2label, device = load_model(args.model)
        print(f"✅ Model geladen op {device.upper()}")
    except Exception as e:
        print(f"❌ Kon model niet laden: {e}")
        sys.exit(1)

    # Extract text
    print(f"📄 Verwerken: {args.input}")
    if input_path.suffix.lower() == ".pdf":
        text = extract_text_from_pdf(str(input_path))
    else:
        text = input_path.read_text(encoding="utf-8")

    print(f"   Tekst lengte: {len(text)} karakters")

    # Classify
    print(f"🔍 Classificeren (drempel: {args.threshold})...")
    results = classify_document(text, model, tokenizer, device, id2label, args.threshold)

    # Output
    if args.json:
        output = {
            "file": str(input_path),
            "threshold": args.threshold,
            "predictions": [r for r in results if r["predicted"] or args.all]
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print("\n" + "=" * 60)
        print("📊 CLASSIFICATIE RESULTATEN")
        print("=" * 60)

        predicted = [r for r in results if r["predicted"]]
        max_confidence = max(r["probability"] for r in results) if results else 0

        # Calculate entropy-like measure: if multiple high scores, model is uncertain
        high_scores = [r["probability"] for r in results if r["probability"] > 0.3]

        if predicted:
            print("\n✅ Gedetecteerde Rechtsfeiten:\n")
            for r in predicted:
                print(f"   [{r['label']}] {r['description']}")
                print(f"         Confidence: {r['probability']*100:.1f}%\n")

            # Warn if multiple high confidence predictions (unusual for clean documents)
            if len(high_scores) >= 3 and len(predicted) >= 2:
                print("   ⚠️  Let op: Meerdere hoge scores. Mogelijk past dit document")
                print("      niet goed in de top-20 categorieën, of bevat het meerdere rechtsfeiten.\n")
        elif max_confidence < 0.3:
            print("\n❓ NIET HERKEND")
            print("   Dit document lijkt niet te passen binnen de top-20 rechtsfeitcodes.")
            print("   Mogelijk betreft het een zeldzamer rechtsfeit dat niet in het model zit.")
            print(f"   (Hoogste score: {max_confidence*100:.1f}%)\n")
        else:
            print("\n⚠️  Geen rechtsfeiten gedetecteerd boven de drempel.\n")
            print(f"   Tip: Verlaag de drempel (nu {args.threshold}) om meer te zien.\n")

        if args.all:
            print("\n📋 Alle scores:\n")
            for r in results:
                status = "✅" if r["predicted"] else "  "
                print(f"   {status} [{r['label']}] {r['probability']*100:5.1f}% - {r['description']}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
