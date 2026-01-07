### DOEL

Lees @CLAUDE-STACK.md voor achtergrond en @CLAUDE.md voor instructies. Voer het doel uit, maar maak eerst een plan en vul hem hieronder in.


### PLAN

**Status: VOLTOOID**

#### Geïmplementeerde componenten:

```
src/
├── config.py                    ✅ Configuratie classes
├── data/
│   ├── loader.py               ✅ JSONL laden + label mapping
│   ├── preprocessor.py         ✅ Sliding window chunking
│   └── dataset.py              ✅ PyTorch Dataset classes
├── models/
│   └── classifier.py           ✅ BERTje multi-label classifier
├── training/
│   ├── trainer.py              ✅ HuggingFace Trainer wrapper
│   └── metrics.py              ✅ Multi-label metrics (F1, etc.)
└── inference/
    ├── ner_extractor.py        ✅ Rule-based NER
    └── predictor.py            ✅ End-to-end inference

train.py                        ✅ Training script
predict.py                      ✅ Inference script
requirements.txt                ✅ Dependencies
```

#### Gebruik:

**Training:**
```bash
pip install -r requirements.txt
python train.py --data_path ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl
```

**Inference:**
```bash
python predict.py --model_dir models/final_model --input data.jsonl --output results.json
```

#### Kenmerken:
- Multi-label classificatie met BERTje (top 20 rechtsfeitcodes)
- Sliding window chunking voor lange documenten (99.2% > 512 tokens)
- Rule-based NER voor entiteiten (personen, organisaties, kadastrale refs)
- FP16 mixed precision training
- Human-in-the-loop flagging bij lage confidence