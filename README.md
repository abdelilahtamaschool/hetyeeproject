# Kadaster - Automatisering Rechtsfeiten Herkenning (AML)

**Team:** Sam Streumer, Joost Leverink, Abdelilah Tama (Saxion 2025/26)

Automatische classificatie van rechtsfeiten in notariële aktes voor het Kadaster.

## Project Overzicht

Dit project ontwikkelt een AI/ML pipeline om rechtsfeiten (zoals overdracht, hypotheek) automatisch te herkennen in notariële aktes. Het doel is om de handmatige classificatie te automatiseren en de 'bewaarder' te ondersteunen met een streefwaarde van **90% accuracy**.

### Huidige Status: Fase 2 Voltooid

- **Baseline Model:** TF-IDF + Logistic Regression
- **Huidige Accuracy:** 81.70% op test set
- **Gap tot doel:** 8.30%

## Project Structuur

```
hetyeeproject/
│
├── Datasets/
│   ├── raw/                           # Ruwe data (niet in git)
│   └── processed/                     # Train/val/test splits
│       ├── train.pkl (13,746 docs)
│       ├── val.pkl   (2,946 docs)
│       └── test.pkl  (2,946 docs)
│
├── Notebooks/
│   └── 01_Exploratie.ipynb           # EDA notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Data inladen en statistieken
│   ├── preprocessing.py              # Data splitting en preprocessing
│   └── train.py                      # Baseline model training
│
├── models/
│   ├── baseline_vectorizer.pkl       # TF-IDF vectorizer
│   ├── baseline_classifier.pkl       # Logistic Regression model
│   └── baseline_metrics.pkl          # Evaluatie metrics
│
├── view_data.py                      # Script om data te inspecteren
├── requirements.txt                  # Python dependencies
├── PROJECT_PLAN.md                   # Gedetailleerd project plan
└── README.md                         # Dit bestand
```

## Dataset Statistieken

- **Totaal documenten:** 19,744 (19,638 na filtering van rare classes)
- **Gemiddelde tekstlengte:** ~22,000 karakters (~3,200 woorden)
- **Aantal labels:** 86 unieke rechtsfeiten
- **Labels per document:** Gemiddeld 1.26 (meeste documenten hebben 1 label)
- **Class imbalance:** Top 5 labels dekken >70% van data

### Top Rechtsfeiten

| Label | Count | Percentage |
|-------|-------|------------|
| 606   | 8,141 | 32.61%     |
| 537   | 4,827 | 19.34%     |
| 585   | 1,901 | 7.62%      |
| 545   | 1,844 | 7.39%      |
| 572   | 1,287 | 5.16%      |

## Installatie

```bash
# Clone repository
git clone <repository-url>
cd hetyeeproject

# Installeer dependencies
pip install -r requirements.txt

# (Optioneel) Installeer spacy Nederlands model
python -m spacy download nl_core_news_lg
```

## Gebruik

### 1. Data Verkennen

```bash
# Bekijk eerste 10 records met labels
python view_data.py 10

# Run EDA notebook
jupyter notebook Notebooks/01_Exploratie.ipynb
```

### 2. Data Preprocessing

```bash
# Maak train/val/test splits (70/15/15)
python src/preprocessing.py
```

### 3. Train Baseline Model

```bash
# Train TF-IDF + Logistic Regression
python src/train.py
```

### 4. Resultaten

Het getrainde model wordt opgeslagen in `models/` directory.

## Baseline Model Resultaten

### Overall Performance

| Metric    | Validation | Test   |
|-----------|------------|--------|
| Accuracy  | 80.99%     | 81.70% |
| Precision | 85.59%     | 86.00% |
| Recall    | 80.99%     | 81.70% |
| F1-Score  | 82.52%     | 83.13% |

### Per-Class Performance (Test Set)

**Uitstekende Performance (F1 > 0.90):**
- Label 537: 99% F1 (722 samples)
- Label 585: 99% F1 (285 samples)
- Label 517: 91% F1 (33 samples)

**Goede Performance (F1: 0.70-0.90):**
- Label 606: 85% F1 (950 samples) - meest voorkomende label
- Label 538: 89% F1 (79 samples)
- Label 581: 87% F1 (57 samples)

**Uitdagingen:**
- Rare classes met < 10 samples presteren slecht
- Label 572: 39% F1 (77 samples) - verwarring met andere labels
- Label 564: 28% F1 (22 samples)

## Belangrijkste Bevindingen

### ✓ Wat werkt goed:
1. **TF-IDF is effectief** - 81.7% accuracy met simpel model
2. **Frequent classes** presteren uitstekend (>95% F1)
3. **Stratified splitting** werkt goed voor class imbalance
4. **Signal aanwezig** - taak is duidelijk learnable

### ⚠ Uitdagingen:
1. **Class imbalance** - rare classes presteren slecht
2. **Tekstlengte** - Documenten zijn zeer lang (3200+ woorden), problematisch voor BERT
3. **Gap tot doel** - 8.3% verbetering nodig om 90% te bereiken

## Volgende Stappen (Fase 3)

### 1. Advanced Modeling
- **BERTje/RobBERT fine-tuning** - Nederlandse BERT modellen
- **Lange tekst strategieën:**
  - Eerste 512 tokens
  - Sliding window approach
  - Hierarchical models

### 2. Class Imbalance Aanpak
- Oversampling van rare classes (SMOTE)
- Focal loss voor training
- Ensemble methoden

### 3. Feature Engineering
- Domein-specifieke features (juridische keywords)
- Document structuur features
- Named entity features

### 4. Model Optimalisatie
- Hyperparameter tuning
- Cross-validation
- Ensemble van modellen

## Privacy & Security

⚠️ **Belangrijk:**
- Geen publieke cloud API's gebruiken (OpenAI, ChatGPT, etc.)
- Data bevat geanonimiseerde persoonlijke informatie
- Alle modellen draaien lokaal
- Data files zijn uitgesloten van git (zie `.gitignore`)

## Contact

Voor vragen of feedback, neem contact op met het team:
- Sam Streumer
- Joost Leverink
- Abdelilah Tama

Saxion University of Applied Sciences, 2025/26

---

**Status:** Fase 2 (Baseline Modelling) Voltooid ✓
**Volgende:** Fase 3 (Advanced Modelling met BERTje/RobBERT)
