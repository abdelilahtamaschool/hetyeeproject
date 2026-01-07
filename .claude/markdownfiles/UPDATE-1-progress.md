# Update 1: Implementatie AI Pipeline voor Akte-classificatie

**Datum:** 2026-01-07
**Status:** Voltooid

---

## Samenvatting

Volledige implementatie van een AI/ML pipeline voor het classificeren van Nederlandse juridische aktes en het extraheren van entiteiten. De pipeline gebruikt BERTje (Nederlands BERT model) voor multi-label classificatie van rechtsfeitcodes en rule-based NER voor entiteitsextractie.

---

## Stappen uitgevoerd

### Stap 1: Analyse en Planning

1. **Gelezen bestanden:**
   - `CLAUDE-STACK.md` - Technische specificaties en requirements
   - `CLAUDE.md` - Instructies voor implementatie

2. **Dataset geanalyseerd:**
   - 19.744 documenten in JSONL formaat
   - 86 unieke rechtsfeitcodes (multi-label)
   - 99.2% documenten > 512 tokens (chunking nodig)
   - Top 20 codes dekken 93% van het volume

3. **Gebruikerskeuzes vastgelegd:**
   - Classificatie: Top 20 rechtsfeitcodes
   - NER: Rule-based met spaCy
   - Hardware: NVIDIA GPU met FP16

---

### Stap 2: Project Setup

1. **Directory structuur aangemaakt:**
   ```
   src/
   ├── data/
   ├── models/
   ├── training/
   └── inference/
   configs/
   models/
   outputs/
   ```

2. **requirements.txt aangemaakt met dependencies:**
   - torch, transformers, datasets, accelerate
   - spacy, pandas, numpy, scikit-learn
   - tqdm, PyYAML, matplotlib, seaborn

3. **`__init__.py` bestanden aangemaakt** voor alle modules

---

### Stap 3: Configuratie Module

**Bestand:** `src/config.py`

Dataclasses aangemaakt voor:
- `ModelConfig` - BERTje model instellingen
- `ChunkingConfig` - Sliding window parameters
- `TrainingConfig` - Hyperparameters (lr, batch size, epochs)
- `DataConfig` - Data paden en splits
- `InferenceConfig` - Threshold en review settings

---

### Stap 4: Data Pipeline

**Bestand:** `src/data/loader.py`
- `load_jsonl()` - JSONL bestand laden met progress bar
- `get_top_n_labels()` - Top N meest voorkomende rechtsfeitcodes
- `get_label_mapping()` - Bidirectionele label2id/id2label mapping
- `filter_documents_by_labels()` - Filter documenten op geldige labels
- `create_label_matrix()` - Multi-hot encoding voor labels
- `analyze_label_distribution()` - Statistieken over labels

**Bestand:** `src/data/preprocessor.py`
- `clean_text()` - Tekst normalisatie
- `sliding_window_chunk()` - Documenten splitsen in overlappende chunks
- `first_last_chunk()` - Alternatieve chunking strategie
- `estimate_token_count()` - Token telling
- `get_chunking_statistics()` - Analyse van chunking requirements

**Bestand:** `src/data/dataset.py`
- `AkteDataset` - PyTorch Dataset voor document-level training
- `AkteChunkDataset` - Dataset voor chunk-level training
- `collate_fn()` - Custom batching met variabele chunk aantallen
- `chunk_collate_fn()` - Batching voor chunk-level data

---

### Stap 5: Classification Model

**Bestand:** `src/models/classifier.py`

- `AkteClassifier` - Multi-label classifier met chunk aggregatie
  - BERTje als backbone
  - Aggregatie strategieën: max, mean, attention
  - BCEWithLogitsLoss voor multi-label

- `SimpleAkteClassifier` - Vereenvoudigde versie voor HuggingFace Trainer
  - Gebruikt AutoModelForSequenceClassification

- `load_classifier()` - Model laden van checkpoint

---

### Stap 6: Training Module

**Bestand:** `src/training/metrics.py`
- `compute_multi_label_metrics()` - F1 micro/macro, precision, recall
- `find_optimal_threshold()` - Threshold optimalisatie
- `get_per_class_metrics()` - Metrics per rechtsfeitcode
- `calculate_class_weights()` - Class weights voor imbalance
- `aggregate_chunk_predictions()` - Chunk naar document aggregatie

**Bestand:** `src/training/trainer.py`
- `create_training_args()` - HuggingFace TrainingArguments
- `create_trainer()` - Trainer setup voor chunk-level training
- `CustomTrainer` - Custom training loop voor document-level
- `train_simple_model()` - High-level training functie

---

### Stap 7: Inference Module

**Bestand:** `src/inference/ner_extractor.py`

`DutchLegalNER` class met regex patterns voor:
- **CADASTRAL** - Kadastrale aanduidingen (gemeente, sectie, nummer)
- **ORGANIZATION** - Rechtspersonen (B.V., N.V., Stichting)
- **PERSON** - Natuurlijke personen (comparant, erflater)
- **DATE** - Datums in Nederlandse formaten
- **MONEY** - Geldbedragen (EUR, euro)
- **ADDRESS** - Adressen met straat, nummer, postcode

**Bestand:** `src/inference/predictor.py`
- `AktePredictor` - End-to-end inference pipeline
  - Classificatie met chunk aggregatie
  - NER extractie
  - Human-in-the-loop flagging
  - JSON export in Kadaster formaat

- `SimplePredictorFromTrainer` - Predictor voor HF Trainer checkpoints
- `load_predictor()` - Helper voor model laden

---

### Stap 8: Main Scripts

**Bestand:** `train.py`
- Command-line interface voor training
- Data laden en splits maken
- Model initialisatie en training
- Evaluatie op test set
- Model en results opslaan

**Bestand:** `predict.py`
- Command-line interface voor inference
- Model en label mapping laden
- Batch prediction met progress bar
- JSON export met summary statistieken

---

### Stap 9: Documentatie

- `PLAN-1.md` bijgewerkt met voltooiingsstatus
- Gebruiksinstructies toegevoegd

---

## Bestanden aangemaakt

| Bestand | Regels | Beschrijving |
|---------|--------|--------------|
| `requirements.txt` | 20 | Dependencies |
| `src/config.py` | 85 | Configuratie classes |
| `src/data/loader.py` | 150 | Data loading utilities |
| `src/data/preprocessor.py` | 180 | Text preprocessing en chunking |
| `src/data/dataset.py` | 200 | PyTorch Dataset classes |
| `src/models/classifier.py` | 220 | BERTje classifier |
| `src/training/metrics.py` | 200 | Evaluation metrics |
| `src/training/trainer.py` | 250 | Training utilities |
| `src/inference/ner_extractor.py` | 350 | Rule-based NER |
| `src/inference/predictor.py` | 300 | Inference pipeline |
| `train.py` | 150 | Training script |
| `predict.py` | 120 | Inference script |

**Totaal:** ~2200 regels Python code

---

## Volgende stappen

1. **Dependencies installeren:** `pip install -r requirements.txt`
2. **Training starten:** `python train.py`
3. **Inference runnen:** `python predict.py --model_dir models/final_model --input data.jsonl --output results.json`
4. **Optioneel:** spaCy Nederlands model installeren voor betere NER: `python -m spacy download nl_core_news_lg`

---

## Technische details

- **Model:** GroNLP/bert-base-dutch-cased (BERTje)
- **Max sequence length:** 512 tokens
- **Chunking:** Sliding window (510 tokens, 128 overlap)
- **Aggregatie:** Max pooling over chunks
- **Loss:** BCEWithLogitsLoss (multi-label)
- **Optimizer:** AdamW met linear warmup
- **Mixed precision:** FP16 enabled
