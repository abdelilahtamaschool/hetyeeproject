# RTX 4070 GPU Setup Guide voor ModernBERT Training

**Doel:** ModernBERT training draaien op je NVIDIA RTX 4070 GPU voor maximale performance

**Huidige Situatie:** PyTorch is geïnstalleerd zonder CUDA support, waardoor training op CPU draait (veel langzamer)

**Oplossing:** PyTorch opnieuw installeren met CUDA 12.x support

---

## Stap 1: Verifieer je GPU

Controleer of je RTX 4070 herkend wordt door Windows:

```bash
# Open PowerShell of Command Prompt
nvidia-smi
```

**Verwachte output:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 5xx.xx       Driver Version: 5xx.xx         CUDA Version: 12.x             |
|-------------------------------+----------------------+--------------------------------+
|   0  NVIDIA GeForce RTX 4070  |                      | 00000000:01:00.0               |
| ...                           |                      |                                 |
+-----------------------------------------------------------------------------------------+
```

Als je `nvidia-smi` niet werkt:
- Installeer de nieuwste NVIDIA drivers: https://www.nvidia.com/download/index.aspx
- Herstart je computer
- Probeer opnieuw

---

## Stap 2: Verwijder Huidige PyTorch (CPU-only)

```bash
pip uninstall torch torchvision torchaudio -y
```

---

## Stap 3: Installeer PyTorch met CUDA 12.x Support

**Optie A: Voor CUDA 12.1 (Aanbevolen voor RTX 4070)**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Optie B: Voor CUDA 11.8 (Als je een oudere driver hebt)**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Installatie duurt:** 5-10 minuten (PyTorch met CUDA is ~2GB)

---

## Stap 4: Verifieer CUDA Installatie

Voer dit Python commando uit:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('GPU Count:', torch.cuda.device_count())"
```

**Verwachte output (SUCCES):**
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4070
GPU Count: 1
```

**Als CUDA available = False:**
- Controleer of je de juiste CUDA versie hebt geïnstalleerd
- Herstart Python/terminal
- Check NVIDIA driver versie met `nvidia-smi`

---

## Stap 5: (Optioneel) Installeer Flash Attention 2

Flash Attention 2 maakt training **3x sneller** voor lange sequences (8192 tokens).

**Vereisten:**
- Visual Studio Build Tools (voor C++ compiler)
- CUDA Toolkit 12.x

**Installatie:**
```bash
pip install flash-attn --no-build-isolation
```

**Let op:** Dit kan 10-20 minuten duren omdat het from source compileert.

**Als installatie faalt:** Geen probleem! ModernBERT werkt ook zonder Flash Attention 2, alleen iets langzamer.

---

## Stap 6: Test ModernBERT met GPU

Voer de long sequence test uit:

```bash
cd E:\Abdelilah\vscode\gitkad\hetyeeproject
python model.py
```

**Verwachte output:**
```
Using device: cuda
GPU: NVIDIA GeForce RTX 4070
CUDA Version: 12.1
Available VRAM: 12.00 GB

Loading ModernBERT tokenizer...
Loading ModernBERT model with 10 labels...
Flash Attention 2 enabled  # (als geïnstalleerd)
Model loaded successfully!
Model parameters: 149,612,554
Trainable parameters: 149,612,554

[OK] Successfully processed 512 token sequence
[OK] Successfully processed 1024 token sequence
[OK] Successfully processed 2048 token sequence
[OK] Successfully processed 4096 token sequence
[OK] Successfully processed 8192 token sequence

Long sequence test completed!
```

**Als "Using device: cpu":** PyTorch CUDA installatie niet gelukt, herhaal Stap 3.

---

## Stap 7: Run Sample Training op GPU

Nu kun je de snelle sample training draaien:

```bash
python src/train_modernbert_sample.py
```

**Verwachte performance op RTX 4070:**
- Sample training (500 docs): **10-15 minuten**
- Full training (13,746 docs): **3-5 uur**

**CPU vs GPU verschil:**
- CPU: 30+ minuten voor sample
- GPU (RTX 4070): 10-15 minuten voor sample
- **~3x sneller!**

---

## Stap 8: (Optioneel) Full Training op Complete Dataset

Voor de beste resultaten, train op de volledige dataset:

```bash
python src/train_modernbert.py
```

**Training configuratie voor RTX 4070:**
- Batch size: 4
- Gradient accumulation: 8 (effectieve batch: 32)
- Max sequence length: 8192 tokens
- Mixed precision (FP16): Enabled
- Epochs: 3

**Verwachte tijd:** 3-5 uur
**Verwachte accuracy:** 90-92% (vs baseline 81.70%)

---

## Performance Monitoring tijdens Training

Je kunt GPU gebruik monitoren tijdens training:

**Terminal 1 (training):**
```bash
python src/train_modernbert_sample.py
```

**Terminal 2 (monitoring):**
```bash
# Update elke 2 seconden
nvidia-smi -l 2
```

**Wat te verwachten:**
- **GPU Utilization:** 90-100% (goed!)
- **Memory Usage:** 10-11 GB / 12 GB (bijna vol = optimaal)
- **Temperature:** 60-80°C (normaal)
- **Power:** 180-200W (max 200W voor RTX 4070)

---

## Troubleshooting

### Probleem: "CUDA out of memory"

**Oplossing:** Verlaag batch size in `model.py`:

```python
class ModernBERTConfig:
    BATCH_SIZE = 2  # Was: 4
    MAX_LENGTH = 4096  # Was: 8192
```

### Probleem: Training is erg langzaam op GPU

**Check:**
1. Verifieer dat FP16 enabled is: Check output "fp16=True"
2. Controleer GPU utilization met `nvidia-smi` (moet >80% zijn)
3. Close andere GPU-intensieve apps (games, video editing, etc.)

### Probleem: "RuntimeError: CUDA error"

**Oplossingen:**
1. Update NVIDIA drivers naar nieuwste versie
2. Herstart computer
3. Reinstall PyTorch met CUDA

### Probleem: Flash Attention 2 installeert niet

**Oplossing:** Skip deze stap. ModernBERT werkt prima zonder:
- Training is ~20% langzamer (nog steeds veel sneller dan CPU)
- Memory gebruik is iets hoger
- Accuracy blijft hetzelfde

---

## Performance Verwachtingen

### Sample Training (500 documenten)

| Hardware | Time | Speedup |
|----------|------|---------|
| CPU (Intel i7/i9) | 30-40 min | 1x |
| RTX 4070 (zonder Flash Attn) | 12-15 min | 2.5x |
| RTX 4070 (met Flash Attn 2) | 8-10 min | 3.5x |

### Full Training (13,746 documenten)

| Hardware | Time | Speedup |
|----------|------|---------|
| CPU | 10-15 hours | 1x |
| RTX 4070 (zonder Flash Attn) | 4-5 hours | 3x |
| RTX 4070 (met Flash Attn 2) | 3-4 hours | 3.5x |

---

## Model Accuracy Verwachtingen

| Dataset Size | Expected Accuracy | Notes |
|--------------|-------------------|-------|
| Sample (500) | 70-75% | Quick validation |
| Full (13,746) | 90-92% | Production ready |
| Baseline (TF-IDF) | 81.70% | For comparison |

**Gap verbetering:** +8-10 procentpunten door ModernBERT

---

## Volgende Stappen na Training

1. **Evalueer resultaten:**
   - Check `models/modernbert_sample/test_results.pkl`
   - Bekijk confusion matrix voor per-class performance

2. **Compare met baseline:**
   - Baseline: 81.70% accuracy
   - Target: 90% accuracy
   - ModernBERT sample: Zie test output

3. **Full training (als sample goed is):**
   ```bash
   python src/train_modernbert.py  # 3-5 uur
   ```

4. **Deploy model:**
   - Model opgeslagen in `models/modernbert/`
   - Gebruik voor inference: `classifier.predict(texts)`

---

## Contact & Support

**Documentatie:**
- Gedetailleerde technische uitleg: `ontwikkelstappen_modernbert.md`
- Project plan: `PROJECT_PLAN.md`
- README: `README.md`

**Team:** Sam Streumer, Joost Leverink, Abdelilah Tama (Saxion 2025/26)

---

**Status:** Guide compleet - Klaar voor GPU training!
**Verwachte totale setup tijd:** 30-45 minuten (incl. PyTorch download)
**Verwachte training tijd (sample):** 10-15 minuten op RTX 4070
