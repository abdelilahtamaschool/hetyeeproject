# PDF Classifier voor Juridische Documenten

Classificeert Nederlandse juridische documenten (aktes) met een getraind BERTje model.

## Installatie

```bash
cd pdf-classify
pip install -r requirements.txt
```

## Gebruik

### Web App (Streamlit)

```bash
streamlit run app.py
```

Open http://localhost:8501 in je browser.

### Command Line

```bash
# Basis gebruik
python cli.py document.pdf

# Met opties
python cli.py document.pdf --model ../models/final_model --threshold 0.4

# JSON output
python cli.py document.pdf --json

# Toon alle scores
python cli.py document.pdf --all
```

## Model Setup

Zorg dat het getrainde model beschikbaar is:

```
models/
├── final_model/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
└── label_mapping.json
```

## Rechtsfeiten Codes

| Code | Beschrijving |
|------|--------------|
| 606 | Levering van een onroerende zaak |
| 537 | Hypotheek |
| 585 | Verdeling |
| 545 | Erfpacht |
| 572 | Kwantitatieve splitsing |
| 564 | Vestiging erfdienstbaarheid |
| 527 | Doorhaling hypotheek |
| 532 | Wijziging hypotheek |
| 538 | Beslag |
| 580 | Verklaring van erfrecht |
