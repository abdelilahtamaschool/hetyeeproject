"""
PDF Classifier - Streamlit App
Classificeert juridische documenten met het getrainde BERTje model.
"""
import streamlit as st
import torch
import json
import fitz  # PyMuPDF
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Page config
st.set_page_config(
    page_title="Juridische Document Classifier",
    page_icon="📄",
    layout="wide"
)

@st.cache_resource
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
        st.warning(f"Label mapping niet gevonden")

    # Load tokenizer from base model (BERTje)
    tokenizer = AutoTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")

    # Load fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, tokenizer, id2label, device


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF."""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
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
    # Chunk the text
    chunks = chunk_text(text, tokenizer)

    if not chunks:
        return [], []

    all_probs = []

    with torch.no_grad():
        for chunk_tokens in chunks:
            # Prepare input
            input_ids = torch.tensor([[tokenizer.cls_token_id] + chunk_tokens + [tokenizer.sep_token_id]])
            attention_mask = torch.ones_like(input_ids)

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
            all_probs.append(probs)

    # Aggregate probabilities (max pooling across chunks)
    aggregated_probs = np.max(all_probs, axis=0)

    # Get predictions above threshold
    results = []
    for idx, prob in enumerate(aggregated_probs):
        label = id2label.get(idx, f"Label_{idx}") if id2label else f"Label_{idx}"
        results.append({
            "label": label,
            "probability": float(prob),
            "predicted": prob >= threshold
        })

    # Sort by probability
    results.sort(key=lambda x: x["probability"], reverse=True)

    return results, all_probs


# Rechtsfeit code descriptions (top 20)
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


def main():
    st.title("📄 Juridische Document Classifier")
    st.markdown("Upload een PDF om rechtsfeiten te classificeren met AI.")

    # Sidebar settings
    st.sidebar.header("⚙️ Instellingen")

    model_dir = st.sidebar.text_input(
        "Model directory",
        value="../uitslag/models/checkpoints/checkpoint-9354",
        help="Pad naar het getrainde model"
    )

    threshold = st.sidebar.slider(
        "Classificatie drempel",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Minimum confidence voor classificatie"
    )

    # Load model
    try:
        with st.spinner("Model laden..."):
            model, tokenizer, id2label, device = load_model(model_dir)
        st.sidebar.success(f"✅ Model geladen op {device.upper()}")
    except Exception as e:
        st.error(f"❌ Kon model niet laden: {e}")
        st.info("Zorg dat het model is gedownload naar de juiste directory.")
        return

    # File upload
    uploaded_file = st.file_uploader(
        "Upload een PDF document",
        type=["pdf"],
        help="Sleep een PDF hierheen of klik om te selecteren"
    )

    # Or paste text
    st.markdown("**Of plak tekst direct:**")
    text_input = st.text_area(
        "Tekst input",
        height=150,
        placeholder="Plak hier de tekst van het juridische document..."
    )

    # Process
    if uploaded_file is not None or text_input:
        if uploaded_file:
            with st.spinner("PDF verwerken..."):
                text = extract_text_from_pdf(uploaded_file)
            st.success(f"✅ PDF geladen: {len(text)} karakters")
        else:
            text = text_input

        # Show extracted text (collapsible)
        with st.expander("📝 Geëxtraheerde tekst", expanded=False):
            st.text(text[:2000] + "..." if len(text) > 2000 else text)

        # Classify
        if st.button("🔍 Classificeer Document", type="primary"):
            with st.spinner("Classificeren..."):
                results, chunk_probs = classify_document(
                    text, model, tokenizer, device, id2label, threshold
                )

            # Display results
            st.markdown("---")
            st.subheader("📊 Classificatie Resultaten")

            # Predicted labels
            predicted = [r for r in results if r["predicted"]]

            if predicted:
                st.markdown("### ✅ Gedetecteerde Rechtsfeiten")

                cols = st.columns(min(len(predicted), 3))
                for i, result in enumerate(predicted):
                    with cols[i % 3]:
                        label = result["label"]
                        prob = result["probability"]
                        desc = RECHTSFEIT_DESCRIPTIONS.get(label, "Onbekend rechtsfeit")

                        st.metric(
                            label=f"Code {label}",
                            value=f"{prob*100:.1f}%",
                            help=desc
                        )
                        st.caption(desc)
            else:
                st.warning("Geen rechtsfeiten gedetecteerd boven de drempel.")

            # All probabilities
            st.markdown("### 📈 Alle Scores")

            # Create bar chart data
            chart_data = {
                "Rechtsfeit": [],
                "Probability": [],
                "Status": []
            }

            for result in results[:10]:  # Top 10
                label = result["label"]
                desc = RECHTSFEIT_DESCRIPTIONS.get(label, "")
                chart_data["Rechtsfeit"].append(f"{label}: {desc[:20]}...")
                chart_data["Probability"].append(result["probability"])
                chart_data["Status"].append("Gedetecteerd" if result["predicted"] else "Onder drempel")

            st.bar_chart(
                data={r["label"]: r["probability"] for r in results[:10]},
                height=300
            )

            # Detailed table
            with st.expander("📋 Gedetailleerde scores", expanded=False):
                for result in results:
                    label = result["label"]
                    prob = result["probability"]
                    desc = RECHTSFEIT_DESCRIPTIONS.get(label, "Onbekend")
                    status = "✅" if result["predicted"] else "❌"
                    st.write(f"{status} **{label}** ({prob*100:.1f}%) - {desc}")

            # Chunk analysis
            if len(chunk_probs) > 1:
                with st.expander(f"📄 Chunk analyse ({len(chunk_probs)} chunks)", expanded=False):
                    st.write(f"Document verdeeld in {len(chunk_probs)} chunks voor analyse.")
                    for i, probs in enumerate(chunk_probs):
                        top_idx = np.argmax(probs)
                        top_label = id2label.get(top_idx, f"Label_{top_idx}") if id2label else f"Label_{top_idx}"
                        st.write(f"Chunk {i+1}: Top = {top_label} ({probs[top_idx]*100:.1f}%)")


if __name__ == "__main__":
    main()
