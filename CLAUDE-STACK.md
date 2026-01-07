{
  "doel": "Organiseren van aktes dmv lokale AI.",
  "doelstelling": {
    "beschrijving": "Ontwikkeling van een AI/Machine Learning pipeline",
    "functionaliteiten": [
      {
        "naam": "Rechtsfeiten classificeren",
        "details": {
          "focus": "Top 20 rechtsfeiten",
          "dekking_volume_percentage": 93
        }
      },
      {
        "naam": "Entiteiten extraheren",
        "details": {
          "types": ["Subjecten", "Objecten"]
        }
      }
    ],
    "resultaat": "Gestructureerde output voor het Kadaster",
    "performance": {
      "streefwaarde_accuracy_percentage": 90
    }
  },
  "randvoorwaarden_en_constraints": {
    "privacy_en_security": "Geen gebruik van publieke cloud API's (zoals OpenAI/ChatGPT). Oplossing moet lokaal draaien.",
    "input": "Geanonimiseerde akte-data (tekst)",
    "human_in_the_loop": "Model ondersteunt de bewaarder, maar neemt geen autonome eindbeslissing.",
    "taal": "Juridisch Nederlands"
  },
  "model_keuze": {
    "categorie": "NLP",
    "motivatie": "Engelse modellen hebben moeite met Nederlandse nuances",
    "modellen": [
      {
        "naam": "BERTje",
        "organisatie": "Universiteit Groningen",
        "beschrijving": "Getraind op grote hoeveelheden Nederlandse tekst",
        "repository": "https://github.com/wietsedv/bertje"
      }
    ]
  },
  "technieken": [
    {
      "naam": "Text Classification",
      "doel": "Bepalen welk rechtsfeit (label) bij een akte of tekstsegment hoort"
    },
    {
      "naam": "Named Entity Recognition",
      "afkorting": "NER",
      "doel": "Extraheren van subjecten (wie) en objecten (wat/welk perceel)"
    }
  ],
  "technology_stack": {
    "taal": "Python",
    "versie": "3.9+",
    "libraries": [
      {
        "naam": "pandas",
        "doel": "Data manipulatie"
      },
      {
        "naam": "numpy",
        "doel": "Numerieke berekeningen"
      },
      {
        "naam": "scikit-learn",
        "doel": "Baseline modellen en metrics"
      },
      {
        "naam": "transformers",
        "doel": "Hugging Face models zoals BERTje en RobBERT"
      },
      {
        "naam": "torch",
        "doel": "Deep Learning backend"
      },
      {
        "naam": "tensorflow",
        "doel": "Alternatieve Deep Learning backend"
      },
      {
        "naam": "spacy",
        "doel": "NLP preprocessing"
      }
    ]
  }
}
