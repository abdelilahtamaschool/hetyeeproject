# Update 2: Performance Optimalisatie Dataset Creatie

**Datum:** 2026-01-07
**Status:** In Progress

---

## Probleem Analyse

Training run #1 stopte tijdens het aanmaken van chunks. Na ~10 seconden data laden kwam het proces vast te zitten zonder zichtbare voortgang.

### Root Cause

| Probleem | Locatie | Impact |
|----------|---------|--------|
| **Geen progress bar** | `dataset.py:153` | 15.044 docs onzichtbaar verwerkt |
| **Dubbele tokenisatie** | `preprocessor.py:56-73` | Elk document 2x getokeniseerd |
| **Geen batch processing** | Overal | 1 document per keer |
| **Onnodige tensor conversies** | `dataset.py:173-174` | ~100.000+ tensor ops |

### Workload Schatting

- 15.044 documenten
- ~3.345 tokens gemiddeld per document
- ~7-8 chunks per document
- **Totaal: ~100.000+ chunks te verwerken**

---

## Oplossingen

### Fix 1: Progress Bar Toevoegen (Kritiek)

**Bestand:** `src/data/dataset.py` regel 153

```python
# VOOR:
for doc_idx, doc in enumerate(documents):

# NA:
from tqdm import tqdm
for doc_idx, doc in enumerate(tqdm(documents, desc="Creating chunks")):
```

### Fix 2: Tensor Conversie Uitstellen (Medium)

**Bestand:** `src/data/dataset.py` regels 173-174

```python
# VOOR:
'input_ids': torch.tensor(chunk['input_ids'], dtype=torch.long),
'attention_mask': torch.tensor(chunk['attention_mask'], dtype=torch.long),

# NA:
'input_ids': chunk['input_ids'],  # List, convert in __getitem__
'attention_mask': chunk['attention_mask'],
```

### Fix 3: Dubbele Tokenisatie Elimineren (Kritiek)

**Bestand:** `src/data/preprocessor.py` regels 56-73

Huidige flow:
1. Tokenize zonder special tokens (regel 56-60)
2. Check lengte
3. Tokenize OPNIEUW met special tokens voor korte docs (regel 66-73)

Nieuwe flow:
1. Tokenize 1x zonder special tokens
2. Handmatig [CLS]/[SEP] toevoegen (sneller dan dubbel tokenizen)

### Fix 4: Batch Processing met Multiprocessing (Optioneel)

Voor grote datasets: `multiprocessing.Pool` voor chunk creatie.

---

## Implementatie Plan

### Stap 1: Quick Wins - VOLTOOID
- [x] Progress bar toevoegen aan `_prepare_chunks()` (regel 154)
- [x] Progress bar toevoegen aan `_prepare_data()` (regel 58)
- [x] Warning fixen met `.clone().detach()` voor tensors (regels 178-186)

### Stap 2: Test Run
- [ ] Test met `--limit 1000` voor snelle validatie
- [ ] Controleer dat output correct blijft

### Stap 3: Volledige Training
- [ ] Run zonder limit
- [ ] Monitor progress en performance

---

## Verwachte Verbetering

| Metric | Voor | Na (Geschat) |
|--------|------|--------------|
| Chunking tijd | Onbekend (afgebroken) | ~2-5 min |
| Visibility | 0% | 100% (progress bar) |
| Tensor overhead | ~100k ops | 0 (uitgesteld) |

---

## Code Changes

### File: `src/data/dataset.py`

**Wijziging 1:** Progress bar toevoegen (regel 152-153)

**Wijziging 2:** Tensor warning fixen (regel 173-174)

**Wijziging 3:** Lazy tensor conversion in `__getitem__`

### File: `src/data/preprocessor.py`

**Wijziging 1:** Elimineer dubbele tokenisatie (regels 56-73)

**Wijziging 2:** Inline num_chunks assignment (regel 117-118)

---

## Uitvoering

Na goedkeuring van dit plan worden de wijzigingen doorgevoerd.
