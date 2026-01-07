### ALTIJD VOORDAT JE WERKT

1. Lees @CLAUDE-STACK.md voor achtergrond
2. Gebruik MCP server Github (BERTje) en Context7 voor documentatie
3. Datapath: C:\Users\wladl\Desktop\Dev\2. yeeproject\ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl


### HOE UIT TE VOEREN

1. Installeer dependencies:
  pip install -r requirements.txt
2. Start training:
  python train.py --data_path ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl
3. Run inference:
  python predict.py --model_dir models/final_model --input data.jsonl --output results.json