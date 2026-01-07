C:\Users\wladl\Desktop\Dev\2. yeeproject>python train.py --learning_rate 5e-5 --data_path ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl
============================================================
Akte Classification Pipeline - Training
============================================================

1. Loading data from ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl...
Loading documents: 19744it [00:02, 8649.91it/s]
   Loaded 19744 documents

2. Analyzing label distribution...
   Found 86 unique rechtsfeitcodes

   Top 10 codes:
   rechtsfeitcode  count  percentage  cumulative_percentage
0             606   8141   41.232780              41.232780
1             537   4827   24.447934              65.680713
2             585   1901    9.628241              75.308955
3             545   1844    9.339546              84.648501
4             572   1287    6.518436              91.166937
5             564    874    4.426661              95.593598
6             527    583    2.952796              98.546394
7             532    565    2.861629             101.408023
8             538    541    2.740073             104.148096
9             580    425    2.152553             106.300648

3. Selecting top 20 labels...
   Top 20 codes: [606, 537, 585, 545, 572, 564, 527, 532, 538, 580, 581, 644, 543, 516, 517, 652, 518, 579, 671, 616]
   Coverage: 117.9%
   Label mapping created: 20 labels

4. Filtering documents...
   18806 documents with at least one top label

   Dataset statistics:
   - Documents: 18806
   - Avg text length: 22252 chars
   - Avg labels per doc: 1.24

5. Splitting data (80/10/10)...
   Train: 15044
   Val: 1881
   Test: 1881

6. Loading tokenizer (GroNLP/bert-base-dutch-cased)...

7. Creating datasets with chunking...
   Chunk size: 510
   Overlap: 128
Creating chunks:   0%|                                                                                                                                          | 0/15044 [00:00<?, ?it/s]Token indices sequence length is longer than the specified maximum sequence length for this model (3345 > 512). Running this sequence through the model will result in indexing errors      
Creating chunks: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15044/15044 [02:29<00:00, 100.72it/s]
   Train chunks: 199505
Creating chunks: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1881/1881 [00:21<00:00, 88.71it/s] 
   Val chunks: 25280

8. Initializing model...
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at GroNLP/bert-base-dutch-cased and are newly initialized: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
   Model: GroNLP/bert-base-dutch-cased
   Num labels: 20

9. Setting up trainer...
   Label mapping saved to models\label_mapping.json

10. Starting training...
    Epochs: 10
    Batch size: 8
    Learning rate: 5e-05
    FP16: True
------------------------------------------------------------
  0%|                                                                                                                                                           | 0/62350 [00:00<?, ?it/s]



  ### PROBLEEM

  Het trainen begint niet. Onderzoek waarom niet.