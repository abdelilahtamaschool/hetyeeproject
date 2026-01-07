# 🚀 ModernBERT v2.0 - Geoptimaliseerde Training

## Wat is er veranderd?

Je vorige training (v1.0) had **75.05% accuracy** - **6.65% slechter** dan de baseline!

**Belangrijkste problemen**:
- ❌ 33 van 45 klassen hadden F1 = 0.0000 (totaal gefaald)
- ❌ 4 onafhankelijke training processen in plaats van 1 gecoördineerd proces
- ❌ Geen Focal Loss of class weighting voor kleine klassen
- ❌ Te kleine batch size (4 ipv mogelijk 16)

**Nu geoptimaliseerd naar v2.0**:
- ✅ **Focal Loss** voor class imbalance
- ✅ **Class weighting** voor minority classes
- ✅ **Batch size 16 per GPU** (128 effectief op 8 GPUs)
- ✅ **Correct multi-GPU training** met torch.distributed
- ✅ **6 epochs** (ipv 3) voor betere convergentie
- ✅ **Learning rate 1e-5** (ipv 2e-5)

**Verwachte resultaten**:
- 🎯 **Accuracy: 85-92%** (vs 75% voorheen)
- ⏱️ **Training tijd: 15-20 min** (vs 41 min voorheen)
- 📊 **Alle 45 klassen: F1 > 0.50**

---

## 📁 Nieuwe Bestanden

### 1. `src/train_modernbert_optimized.py` (NIEUW)
Geoptimaliseerde training script met:
- Multi-GPU support (torch.distributed)
- Focal Loss en class weighting
- Betere logging en monitoring

### 2. `model.py` (AANGEPAST)
Toegevoegd:
- `FocalLoss` class voor class imbalance
- `WeightedTrainer` class voor custom loss
- Geoptimaliseerde `ModernBERTConfig` voor 8x RTX 5090
- Class weight berekening in `train()` methode

### 3. `OPTIMIZATION.md` (NIEUW)
Volledige documentatie met:
- Probleem analyse van v1.0 resultaten
- Uitleg van alle optimalisaties
- Verwachte resultaten
- Troubleshooting guide

### 4. `VASTAI-GUIDE.md` (AANGEPAST)
Stap 9 aangepast met:
- Correcte multi-GPU training commands
- Veelgemaakte fouten en oplossingen
- Checks om te valideren dat training correct loopt

---

## 🚀 Hoe te Gebruiken

### Upload Nieuwe Bestanden naar Vast.ai

```powershell
cd E:\Abdelilah\vscode\gitkad\hetyeeproject

# Upload alle nieuwe en aangepaste bestanden
scp -P 48101 -r src model.py root@84.2.196.163:/workspace/hetyeeproject/
```

### Start Geoptimaliseerde Training

**Op Vast.ai server:**

```bash
cd /workspace/hetyeeproject

# BELANGRIJK: Stop eventuele oude processen eerst
pkill -f train_modernbert
ps aux | grep train_modernbert  # Check dat alles gestopt is

# Start geoptimaliseerde training (8 GPUs)
torchrun --nproc_per_node=8 src/train_modernbert_optimized.py > training_optimized.log 2>&1 &

# Monitor progress
tail -f training_optimized.log

# Check GPU usage (alle 8 moeten actief zijn!)
watch -n 1 nvidia-smi
```

### Controleer dat het Correct Loopt

✅ **Je zou moeten zien**:
```bash
# In training_optimized.log:
🚀 OPTIMIZATIONS:
  ✅ Focal Loss for class imbalance
  ✅ Class weighting for minority classes
  ✅ Batch size: 16 per GPU (128 effective on 8 GPUs)

Training configuration:
  Number of GPUs: 8
  Effective batch size: 128
```

```bash
# In nvidia-smi:
8 python processen, elk op een GPU
GPU-Util: 90-100% op alle GPUs
Memory-Usage: ~20-25GB per GPU
```

❌ **FOUT als je ziet**:
- "Number of GPUs: 1" → Verkeerd commando gebruikt!
- Minder dan 8 processen → Niet alle GPUs worden gebruikt
- Sommige GPUs op 0% → Training loopt niet correct

---

## 📊 Resultaten Vergelijken

### Download Resultaten

```powershell
cd E:\Abdelilah\vscode\gitkad\hetyeeproject

# Download v2.0 model en resultaten
scp -P 48101 -r root@84.2.196.163:/workspace/hetyeeproject/models/modernbert_optimized ./models/
scp -P 48101 root@84.2.196.163:/workspace/hetyeeproject/training_optimized.log ./
```

### Bekijk Resultaten

Check `training_optimized.log` voor:
```
OPTIMIZED TRAINING COMPLETED!

ModernBERT Optimized Results:
  Test Accuracy:  XX.XX%
  Test Precision: XX.XX%
  Test Recall:    XX.XX%
  Test F1-Score:  XX.XX%

Comparison with Previous Runs:
  Baseline (TF-IDF):        81.70%
  ModernBERT v1.0:          75.05%
  ModernBERT v2.0 (This):   XX.XX%
```

En de **F1-scores per rechtsgebied tabel**!

---

## 📚 Documentatie

- **`OPTIMIZATION.md`**: Volledige analyse en uitleg van alle optimalisaties
- **`VASTAI-GUIDE.md`**: Stap-voor-stap guide (nu met correcte multi-GPU commands)
- **`src/train_modernbert_optimized.py`**: Code met inline documentatie

---

## ❓ Troubleshooting

### "Multiple training processes"
```bash
pkill -f train_modernbert
ps aux | grep train_modernbert  # Check dat alles weg is
torchrun --nproc_per_node=8 src/train_modernbert_optimized.py > training_optimized.log 2>&1 &
```

### "Only 1 GPU used"
```bash
# ❌ FOUT
python src/train_modernbert_optimized.py

# ✅ CORRECT
torchrun --nproc_per_node=8 src/train_modernbert_optimized.py
```

### "CUDA out of memory"
Edit `model.py`:
```python
# Verlaag batch size van 16 naar 12
BATCH_SIZE = 12
```

---

## 🎯 Success Criteria

De v2.0 training is succesvol als:
- ✅ Test Accuracy ≥ 85%
- ✅ Alle 45 klassen hebben F1 > 0.30
- ✅ Training duurt ≤ 25 minuten
- ✅ Alle 8 GPUs zijn ≥ 85% utilized

---

## 📞 Support

Voor vragen, check:
1. **OPTIMIZATION.md** - Volledige technische details
2. **VASTAI-GUIDE.md** - Stap-voor-stap instructies
3. Training logs - `tail -f training_optimized.log`

**Veel success met de geoptimaliseerde training!** 🚀

---

**Versie**: 2.0
**Datum**: 2026-01-07
**Status**: ✅ Klaar voor Deployment
