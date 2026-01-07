# ModernBERT Quick Start - RTX 4070

**Snelle instructies om ModernBERT training te draaien op je RTX 4070**

---

## ⚡ Snelste Route naar GPU Training

### 1. Installeer PyTorch met CUDA (5-10 min)
```bash
# Verwijder oude PyTorch
pip uninstall torch torchvision torchaudio -y

# Installeer met CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Verifieer GPU (30 sec)
```bash
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY - REINSTALL PYTORCH!')"
```

**Output moet zijn:** `GPU: NVIDIA GeForce RTX 4070`

### 3. Test ModernBERT (1 min)
```bash
cd E:\Abdelilah\vscode\gitkad\hetyeeproject
python model.py
```

**Check:** Output moet zeggen `Using device: cuda`

### 4. Run Sample Training (10-15 min)
```bash
python src/train_modernbert_sample.py
```

**Resultaat:** Test accuracy op 166 samples

### 5. (Optioneel) Full Training (3-5 uur)
```bash
python src/train_modernbert.py
```

**Resultaat:** 90-92% accuracy op volledige dataset

---

## 🚀 Performance Check

```bash
# Terminal 1: Start training
python src/train_modernbert_sample.py

# Terminal 2: Monitor GPU
nvidia-smi -l 2
```

**Verwacht:**
- GPU Util: 90-100%
- Memory: 10-11 GB / 12 GB
- Temp: 60-80°C

---

## 📊 Verwachte Resultaten

| Training Type | Time (RTX 4070) | Accuracy |
|---------------|-----------------|----------|
| Sample (500 docs) | 10-15 min | 70-75% |
| Full (13,746 docs) | 3-5 hours | 90-92% |
| **Baseline (TF-IDF)** | **2 min** | **81.70%** |

**Verbetering:** +8-10% accuracy door ModernBERT

---

## ⚙️ Configuratie

Alles staat al goed ingesteld in `model.py`:
- ✓ Max sequence length: 8192 tokens
- ✓ Batch size: 4 (optimaal voor 12GB VRAM)
- ✓ Mixed precision (FP16): Enabled
- ✓ Gradient accumulation: 8

**Geen wijzigingen nodig!**

---

## 🔧 Als iets niet werkt

**GPU niet herkend?**
→ Zie `GPU_SETUP_GUIDE.md` Stap 1-4

**Out of memory?**
→ Verlaag `BATCH_SIZE = 2` in `model.py` regel 44

**Te langzaam?**
→ Check GPU utilization met `nvidia-smi`

---

## 📚 Meer Info

- **Volledige setup:** `GPU_SETUP_GUIDE.md`
- **Technische details:** `ontwikkelstappen_modernbert.md`
- **Project overzicht:** `README.md`

---

**Team:** Sam, Joost, Abdelilah (Saxion 2025/26)
**Doel:** 90% accuracy rechtsfeiten classificatie
**Status:** Implementatie compleet ✓
