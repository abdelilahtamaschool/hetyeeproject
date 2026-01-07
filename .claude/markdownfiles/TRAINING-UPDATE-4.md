C:\Users\wladl\Desktop\Dev\2. yeeproject>python train_test.py
============================================================
TEST TRAINING - 2000 documenten, 3 epochs
============================================================
Command: C:\Users\wladl\AppData\Local\Programs\Python\Python312\python.exe train.py --data_path ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl --limit 2000 --epochs 3 --batch_size 8

============================================================
Akte Classification Pipeline - Training
============================================================

1. Loading data from ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl...
Loading documents: 2000it [00:00, 9070.84it/s]
   Loaded 2000 documents

2. Analyzing label distribution...
   Found 60 unique rechtsfeitcodes

   Top 10 codes:
   rechtsfeitcode  count  percentage  cumulative_percentage
0             606    788       39.40                  39.40
1             537    596       29.80                  69.20
2             545    207       10.35                  79.55
3             585    148        7.40                  86.95
4             572    134        6.70                  93.65
5             527     94        4.70                  98.35
6             564     77        3.85                 102.20
7             532     54        2.70                 104.90
8             538     52        2.60                 107.50
9             580     40        2.00                 109.50

3. Selecting top 20 labels...
   Top 20 codes: [606, 537, 545, 585, 572, 527, 564, 532, 538, 580, 581, 543, 516, 644, 671, 652, 651, 579, 518, 616]
   Coverage: 120.3%
   Label mapping created: 20 labels

4. Filtering documents...
   1922 documents with at least one top label

   Dataset statistics:
   - Documents: 1922
   - Avg text length: 23012 chars
   - Avg labels per doc: 1.25

5. Splitting data (80/10/10)...
   Train: 1537
   Val: 192
   Test: 193

6. Loading tokenizer (GroNLP/bert-base-dutch-cased)...

7. Creating datasets with chunking...
   Chunk size: 510
   Overlap: 128
Creating chunks:   0%|                                                                        | 0/1537 [00:00<?, ?it/s]Token indices sequence length is longer than the specified maximum sequence length for this model (901 > 512). Running this sequence through the model will result in indexing errors
Creating chunks: 100%|████████████████████████████████████████████████████████████| 1537/1537 [00:15<00:00, 101.51it/s]
   Train chunks: 21083
Creating chunks: 100%|██████████████████████████████████████████████████████████████| 192/192 [00:01<00:00, 111.42it/s]
   Val chunks: 2514

8. Initializing model...
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at GroNLP/bert-base-dutch-cased and are newly initialized: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
   Model: GroNLP/bert-base-dutch-cased
   Num labels: 20

9. Setting up trainer...
   Label mapping saved to models\label_mapping.json

10. Starting training...
    Epochs: 3
    Batch size: 8
    Learning rate: 0.01
    FP16: True
------------------------------------------------------------
  0%|                                                                              | 1/1977 [00:49<27:00:19, 49.20s/it]S


  ### PROBLEEM

  Zelfs de training met beperkt aantal files en pepochs duurt 27 uur. Onderzoek waarom dit zo lang duurt. Ik zie dat mijn GPU (gtx 1060 3gb) niet wordt gebruikt.