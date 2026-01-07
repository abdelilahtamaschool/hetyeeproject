C:\Users\wladl\Desktop\Dev\2. yeeproject>python train.py --data_path ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl
============================================================
Akte Classification Pipeline - Training
============================================================

1. Loading data from ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl...
Loading documents: 19744it [00:10, 1800.73it/s]
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
tokenizer_config.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████| 254/254 [00:00<?, ?B/s]
C:\Users\wladl\AppData\Local\Programs\Python\Python312\Lib\site-packages\huggingface_hub\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\wladl\.cache\huggingface\hub\models--GroNLP--bert-base-dutch-cased. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
config.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 608/608 [00:00<?, ?B/s]
vocab.txt: 242kB [00:00, 52.9MB/s]
special_tokens_map.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 112/112 [00:00<?, ?B/s]

7. Creating datasets with chunking...
   Chunk size: 510
   Overlap: 128
Token indices sequence length is longer than the specified maximum sequence length for this model (3345 > 512). Running this sequence through the model will result in indexing errors
C:\Users\wladl\Desktop\Dev\2. yeeproject\src\data\dataset.py:173: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
  'input_ids': torch.tensor(chunk['input_ids'], dtype=torch.long),
C:\Users\wladl\Desktop\Dev\2. yeeproject\src\data\dataset.py:174: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
  'attention_mask': torch.tensor(chunk['attention_mask'], dtype=torch.long),



Traceback (most recent call last):
  File "C:\Users\wladl\Desktop\Dev\2. yeeproject\train.py", line 240, in <module>
    main()
  File "C:\Users\wladl\Desktop\Dev\2. yeeproject\train.py", line 147, in main
    train_dataset = AkteChunkDataset(
                    ^^^^^^^^^^^^^^^^^
  File "C:\Users\wladl\Desktop\Dev\2. yeeproject\src\data\dataset.py", line 149, in __init__
    self._prepare_chunks(documents)
  File "C:\Users\wladl\Desktop\Dev\2. yeeproject\src\data\dataset.py", line 157, in _prepare_chunks
    chunks = sliding_window_chunk(
             ^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wladl\Desktop\Dev\2. yeeproject\src\data\preprocessor.py", line 56, in sliding_window_chunk
    encoding = tokenizer(
               ^^^^^^^^^^
  File "C:\Users\wladl\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\tokenization_utils_base.py", line 3073, in __call__ 
    encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wladl\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\tokenization_utils_base.py", line 3183, in _call_one
    return self.encode_plus(
           ^^^^^^^^^^^^^^^^^
  File "C:\Users\wladl\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\tokenization_utils_base.py", line 3258, in encode_plus
    return self._encode_plus(
           ^^^^^^^^^^^^^^^^^^
  File "C:\Users\wladl\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\tokenization_utils_fast.py", line 627, in _encode_plus
    batched_output = self._batch_encode_plus(
                     ^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\wladl\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\tokenization_utils_fast.py", line 553, in _batch_encode_plus
    encodings = self._tokenizer.encode_batch(
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
forrtl: error (200): program aborting due to control-C event
Image              PC                Routine            Line        Source
KERNELBASE.dll     00007FF89C6AB99D  Unknown               Unknown  Unknown
KERNEL32.DLL       00007FF89EA4E8D7  Unknown               Unknown  Unknown
ntdll.dll          00007FF89F22C53C  Unknown               Unknown  Unknown

C:\Users\wladl\Desktop\Dev\2. yeeproject>

============================================================
ANALYSE - Training Run #1
============================================================

## Gestopt door: Ctrl+C (KeyboardInterrupt)

## Waarom was het zo traag?

### Kernprobleem: Inefficiënte document-voor-document tokenisatie

De `sliding_window_chunk()` functie in `src/data/preprocessor.py:56` tokeniseert
elk document afzonderlijk. Met deze dataset:

- 15044 trainingsdocumenten
- Gemiddeld 22252 karakters per document
- Geschatte ~3345 tokens per document (zie warning regel 56)

Dit betekent:
1. Elk document wordt individueel naar de tokenizer gestuurd
2. Geen batch-verwerking = geen GPU/parallel optimalisatie
3. Geen progress bar tijdens het chunken (onzichtbare voortgang)

### Geschatte workload:
- 15044 docs x ~3345 tokens = ~50+ miljoen tokens te verwerken
- Per document meerdere chunks (510 tokens met 128 overlap)
- Geschat ~6-10 chunks per document = 90.000-150.000 chunks totaal

### Aanbevelingen voor volgende run:

1. **Voeg progress bar toe** aan dataset creatie in `dataset.py`
2. **Batch tokenization** waar mogelijk
3. **Parallelisatie** met multiprocessing voor chunk creatie
4. **Eerste test** met subset (bijv. 1000 docs) om tijd te schatten