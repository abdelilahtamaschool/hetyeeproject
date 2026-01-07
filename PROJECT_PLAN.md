# Project: Kadaster - Automatisering Rechtsfeiten Herkenning (AML)

**Datum:** 7 januari 2026  
**Status:** Initiatie & Planning  
**Team:** Sam Streumer, Joost Leverink, Abdelilah Tama (Saxion 2025/26)

---

## 1. Project Context

### 1.1 Probleemstelling
Het Kadaster verwerkt dagelijks grote hoeveelheden notariële aktes. Het handmatig herkennen en classificeren van rechtsfeiten (zoals overdracht, hypotheek), subjecten en objecten is tijdrovend en foutgevoelig. Notarissen hebben redactionele vrijheid, waardoor de structuur van teksten varieert (ongestructureerde data).

### 1.2 Doelstelling
Ontwikkeling van een AI/Machine Learning pipeline die:
*   **Rechtsfeiten classificeert** (focus op top 20, dekt 93% volume).
*   **Entiteiten extraheert** (Subjecten, Objecten).
*   **Resultaat:** Een gestructureerde output voor het Kadaster.
*   **Performance:** Streefwaarde van **90% accuracy**.

### 1.3 Randvoorwaarden & Constraints
*   **Privacy / Security:** Geen gebruik van publieke cloud API's (zoals OpenAI/ChatGPT) in verband met persoonsgegevens. De oplossing moet lokaal draaien.
*   **Input:** Geanonimiseerde akte-data (tekst).
*   **Human-in-the-loop:** Het model ondersteunt de 'bewaarder', maar neemt geen autonome eindbeslissing.
*   **Taal:** Juridisch Nederlands.

---

## 2. Technische Strategie

Gezien de eis voor lokale verwerking en de complexiteit van de Nederlandse juridische taal, is de volgende stack aanbevolen:

### 2.1 Model Keuze (NLP)
Omdat Engelse modellen vaak moeite hebben met Nederlandse nuances, focussen we op Dutch-specific Transformers:
*   **BERTje** (Universiteit Groningen): Getraind op grote hoeveelheden Nederlandse tekst.
*   **RobBERT** (KU Leuven): Gebaseerd op RoBERTa, vaak robuuster.

### 2.2 Technieken
1.  **Text Classification:** Om te bepalen welk *rechtsfeit* (label) bij een akte (of tekstsegment) hoort.
2.  **Named Entity Recognition (NER):** Om specifieke *subjecten* (wie) en *objecten* (wat/welk perceel) uit de tekst te lichten.

### 2.3 Technology Stack
*   **Taal:** Python 3.9+
*   **Libraries:**
    *   `pandas` & `numpy` (Data manipulatie)
    *   `scikit-learn` (Baseline modellen & metrics)
    *   `transformers` (Hugging Face - voor BERTje/RobBERT)
    *   `torch` / `tensorflow` (Deep Learning backend)
    *   `spacy` (Voor NLP preprocessing)

---

## 3. Gedetailleerd Implementatieplan

### Fase 1: Data Preparatie & Exploratie (Week 2)
*Doel: Begrijpen van de dataset en deze gereedmaken voor het model.*

1.  **Data Ingestie:**
    *   Scripts schrijven om de `.txt`, `.xml` of `.docx` bestanden in te lezen.
    *   Controle op encoding (UTF-8) en corrupte bestanden.
2.  **Exploratieve Data Analyse (EDA):**
    *   **Distributie check:** Hoe vaak komt "Hypotheek" voor vs. een zeldzaam rechtsfeit? (Class Imbalance identificeren).
    *   **Lengte analyse:** Hoe lang zijn de teksten? (BERT modellen hebben vaak een limiet van 512 tokens).
    *   **Kwaliteitscheck:** Zijn de geanonimiseerde labels consistent?
3.  **Preprocessing Pipeline:**
    *   Opschonen: Verwijderen van irrelevante headers/footers indien nodig.
    *   Normalisatie: Omgaan met afkortingen of juridisch jargon.
    *   **Splitting:** Data verdelen in Train (70%), Validation (15%), Test (15%). *Cruciaal: Zorg voor een 'stratified split' zodat zeldzame klassen in alle sets voorkomen.*

### Fase 2: Baseline Modellering (Begin Week 3)
*Doel: Een referentiepunt creëren voordat we zware AI inzetten.*

1.  **Bag-of-Words / TF-IDF aanpak:**
    *   Gebruik `TfidfVectorizer` om tekst naar getallen om te zetten.
    *   Train een simpel model: **Logistic Regression** of **Random Forest**.
2.  **Evaluatie Baseline:**
    *   Wat is de accuracy?
    *   Hiermee valideren we of er überhaupt genoeg signaal in de tekst zit.

### Fase 3: Advanced Modellering (Deep Learning)
*Doel: Het behalen van de 90% accuracy.*

1.  **Fine-tuning BERTje/RobBERT:**
    *   Laad een pre-trained Nederlands model.
    *   Voeg een classificatie-laag toe voor de rechtsfeiten.
    *   Train (Fine-tune) op de Kadaster dataset.
2.  **Omgaan met Lange Teksten:**
    *   Als aktes langer zijn dan 512 tokens, moeten we strategieën toepassen (bijv. Sliding Window of alleen de eerste/laatste secties gebruiken).
3.  **NER Training (Parallel):**
    *   Trainen van een token-classifier om entiteiten te labelen.

### Fase 4: Evaluatie & Validatie
*Doel: Aantonen dat het model werkt volgens de eisen.*

1.  **Metrics:**
    *   **Accuracy:** Algemeen percentage goed.
    *   **Precision & Recall:** Belangrijk omdat fout-positieven (onterecht een feit herkennen) andere kosten hebben dan fout-negatieven.
    *   **F1-Score:** Het harmonisch gemiddelde (beste maatstaf voor onbalans).
2.  **Foutenanalyse:**
    *   Waar gaat het mis? (Verwarringsmatrix/Confusion Matrix).
    *   Zijn er specifieke notarissen of stijlen die fout gaan?

---

## 4. Voorgestelde Projectstructuur

Om het project overzichtelijk te houden in VS Code:

```text
kadasterproject/
│
├── Datasets/               # Ruwe en verwerkte data (buiten git houden via .gitignore)
│   ├── raw/
│   └── processed/
│
├── Notebooks/              # Jupyter Notebooks voor experimenten
│   ├── 01_Exploratie.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Training_Baseline.ipynb
│
├── src/                    # Broncode (productie-waardige scripts)
│   ├── __init__.py
│   ├── data_loader.py      # Script om data in te laden
│   ├── preprocessing.py    # Tekst schoonmaak functies
│   ├── train.py            # Training loop
│   └── predict.py          # Script voor nieuwe voorspellingen
│
├── models/                 # Opgeslagen getrainde modellen
│
├── PROJECT_PLAN.md         # Dit bestand
├── PvA_AML_Kadaster.docx   # Origineel plan
├── requirements.txt        # Python dependencies
└── README.md               # Uitleg voor installatie
```


### model gebruik ###
ik wil graag dat je als eerst gebruik maakt van een moden bert model.

