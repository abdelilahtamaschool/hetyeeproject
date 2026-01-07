### DOEL

Organiseren van aktes dmv lokale AI.

### DOELSTELLING
Ontwikkeling van een AI/Machine Learning pipeline die:
*   **Rechtsfeiten classificeert** (focus op top 20, dekt 93% volume).
*   **Entiteiten extraheert** (Subjecten, Objecten).
*   **Resultaat:** Een gestructureerde output voor het Kadaster.
*   **Performance:** Streefwaarde van **90% accuracy**.

### RANDVOORWAARDEN & CONSTRAINTS
*   **Privacy / Security:** Geen gebruik van publieke cloud API's (zoals OpenAI/ChatGPT) in verband met persoonsgegevens. De oplossing moet lokaal draaien.
*   **Input:** Geanonimiseerde akte-data (tekst).
*   **Human-in-the-loop:** Het model ondersteunt de 'bewaarder', maar neemt geen autonome eindbeslissing.
*   **Taal:** Juridisch Nederlands.

### MODEL KEUZE (NLP)
Omdat Engelse modellen vaak moeite hebben met Nederlandse nuances, focussen we op Dutch-specific Transformers:
*   **BERTje** (Universiteit Groningen): Getraind op grote hoeveelheden Nederlandse tekst: https://github.com/wietsedv/bertje

### Technieken
1.  **Text Classification:** Om te bepalen welk *rechtsfeit* (label) bij een akte (of tekstsegment) hoort.
2.  **Named Entity Recognition (NER):** Om specifieke *subjecten* (wie) en *objecten* (wat/welk perceel) uit de tekst te lichten.

### Technology Stack
*   **Taal:** Python 3.9+
*   **Libraries:**
    *   `pandas` & `numpy` (Data manipulatie)
    *   `scikit-learn` (Baseline modellen & metrics)
    *   `transformers` (Hugging Face - voor BERTje/RobBERT)
    *   `torch` / `tensorflow` (Deep Learning backend)
    *   `spacy` (Voor NLP preprocessing)
 
### MOET GEPUSHT

Wat is NIET gepusht (te groot):
  - ❌ model.safetensors (416 MB)
  - ❌ optimizer.pt (833 MB)
  - ❌ Training data .jsonl
