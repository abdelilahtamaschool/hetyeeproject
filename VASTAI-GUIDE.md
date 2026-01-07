# Vast.ai Handleiding - Stap voor Stap

Je hebt $10 gestort. Hier is precies wat je moet doen.

---

## Stap 1: SSH Key Aanmaken (op jouw PC)

Open PowerShell en voer uit:

```powershell
ssh-keygen -t ed25519 -C "vast.ai"
```

- Druk Enter voor default locatie (`C:\Users\wladl\.ssh\id_ed25519`)
- Optioneel: voer een wachtwoord in of druk Enter voor geen wachtwoord

Kopieer je public key:

```powershell
cat ~/.ssh/id_ed25519.pub
```

Kopieer de output (begint met `ssh-ed25519...`).

---

## Stap 2: SSH Key Toevoegen aan Vast.ai

1. Ga naar https://cloud.vast.ai/account/
2. Scroll naar **SSH Keys**
3. Plak je public key en klik **Add SSH Key**

---

## Stap 3: GPU Instance Selecteren

1. Ga naar https://cloud.vast.ai/create/
2. Filter op:
   - **GPU**: RTX 3090 of RTX 4090 (goede prijs/prestatie voor ModernBERT)
   - **VRAM**: Minimaal 16GB (aanbevolen: 24GB voor 8192 token context)
   - **Disk**: Minimaal 30GB
   - **CUDA**: 11.8 of hoger
3. Sorteer op prijs (laag naar hoog)
4. Kies een instance met goede "DLPerf" score

**Budget tip**: Met $10 kun je ~10-20 uur draaien op een RTX 3090.
**ModernBERT vereisten**: Flash Attention 2 werkt best op Ampere (RTX 30xx) of nieuwer.

---

## Stap 4: Template Kiezen

Bij het huren, kies een template:

- **PyTorch** (aanbevolen voor dit project)
- Of selecteer: `pytorch/pytorch:latest`

---

## Stap 5: Instance Starten

1. Klik **RENT** op je gekozen machine
2. Wacht tot status "Running" is (1-2 minuten)
3. Klik op de instance om details te zien

---

## Stap 6: Verbinden via SSH

In de instance details zie je een SSH commando. Het ziet er zo uit:

```bash
ssh -p 12345 root@ssh.vast.ai
```

Voer dit uit in PowerShell. Bij eerste keer: typ `yes` om te verbinden.

---

## Stap 7: Project Uploaden

**Optie A: Via SCP (aanbevolen)**

```powershell
# Vanuit je project folder
scp -P 12345 -r . root@ssh.vast.ai:/workspace/
```

**Optie B: Via Git (aanbevolen)**

Op de Vast.ai machine:
```bash
cd /workspace
git clone https://github.com/[jouw-username]/hetyeeproject.git
cd hetyeeproject
```

---

## Stap 8: Dependencies Installeren

**Optie A: Via pip (simpel)**

```bash
cd /workspace/hetyeeproject
pip install -r requirements.txt
```

**Optie B: Via Docker (aanbevolen)**

```bash
cd /workspace/hetyeeproject
docker build -t modernbert-training .
docker run --gpus all -v $(pwd):/workspace modernbert-training
```

---

## Stap 9: Training Starten

**BELANGRIJK**: Gebruik de **GEOPTIMALISEERDE** versie voor beste resultaten!

### Optie A: Geoptimaliseerde Training (AANBEVOLEN) ✅

**Voor 8 GPU's (8x RTX 5090):**

```bash
cd /workspace/hetyeeproject

# Multi-GPU training (CORRECT - 1 proces, 8 GPUs)
torchrun --nproc_per_node=8 src/train_modernbert_optimized.py > training_optimized.log 2>&1 &

# Monitor progress
tail -f training_optimized.log

# Check GPU usage (alle 8 moeten actief zijn!)
watch -n 1 nvidia-smi
```

**Verwacht**:
- ⏱️ Tijd: **15-20 minuten**
- 🎯 Accuracy: **85-92%**
- 💪 GPU Gebruik: **90-100% op alle 8 GPUs**

---

### Optie B: Snelle Test (1 GPU)

Voor testen of je maar 1 GPU hebt:

```bash
cd /workspace/hetyeeproject
python src/train_modernbert_optimized.py > training_test.log 2>&1 &
```

**Verwacht**:
- ⏱️ Tijd: ~1-2 uur
- Voor validatie dat alles werkt

---

### Optie C: Oude Versie (NIET AANBEVOLEN)

```bash
cd /workspace/hetyeeproject
python src/train_modernbert.py
```

⚠️ **Let op**: Deze versie heeft problemen:
- Geen Focal Loss (slecht voor kleine klassen)
- Geen class weighting
- Lagere accuracy (75% vs 85-92%)
- Gebruik alleen voor vergelijking!

---

### ✅ Controleer of Training Correct Loopt

**Check 1: GPU Usage**

```bash
nvidia-smi
```

Je zou moeten zien:
```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|  0      N/A  N/A    12345      C   python                            25GB   |
|  1      N/A  N/A    12346      C   python                            25GB   |
|  2      N/A  N/A    12347      C   python                            25GB   |
|  3      N/A  N/A    12348      C   python                            25GB   |
|  4      N/A  N/A    12349      C   python                            25GB   |
|  5      N/A  N/A    12350      C   python                            25GB   |
|  6      N/A  N/A    12351      C   python                            25GB   |
|  7      N/A  N/A    12352      C   python                            25GB   |
+-----------------------------------------------------------------------------+
```

✅ **GOED**: 8 python processen, elk op een GPU, ~90-100% GPU-Util
❌ **FOUT**: Minder dan 8 processen, of sommige GPUs op 0%

**Check 2: Training Log**

```bash
tail -f training_optimized.log
```

Je zou moeten zien:
```
🚀 OPTIMIZATIONS:
  ✅ Focal Loss for class imbalance
  ✅ Class weighting for minority classes
  ✅ Batch size: 16 per GPU (128 effective on 8 GPUs)

Training configuration:
  Focal Loss: Enabled
  Class Weighting: Enabled
  Effective batch size: 128
  Number of GPUs: 8
```

✅ **GOED**: "Number of GPUs: 8" en "Effective batch size: 128"
❌ **FOUT**: "Number of GPUs: 1" → Je gebruikt maar 1 GPU!

---

### 🚨 Veelgemaakte Fouten

**Fout 1: Meerdere Trainingen Tegelijk**

Als je per ongeluk het commando **meerdere keren** runt, krijg je:
- 4+ onafhankelijke processen
- GPUs inefficiënt verdeeld
- Training duurt 2-3x langer

**Oplossing**:
```bash
# Stop ALLE training processen
pkill -f train_modernbert

# Check dat alles gestopt is
ps aux | grep train_modernbert

# Start opnieuw (1x!)
torchrun --nproc_per_node=8 src/train_modernbert_optimized.py > training_optimized.log 2>&1 &
```

**Fout 2: Verkeerd Commando**

```bash
# ❌ FOUT - gebruikt maar 1 GPU
python src/train_modernbert_optimized.py

# ✅ CORRECT - gebruikt alle 8 GPUs
torchrun --nproc_per_node=8 src/train_modernbert_optimized.py
```

**Fout 3: Oude Script Gebruiken**

```bash
# ❌ FOUT - oude versie zonder optimalisaties
python src/train_modernbert.py

# ✅ CORRECT - geoptimaliseerde versie
torchrun --nproc_per_node=8 src/train_modernbert_optimized.py
```

---

## Stap 10: Resultaten Downloaden

Na training, op je lokale PC:

### Geoptimaliseerde Model (v2.0)

```powershell
cd E:\Abdelilah\vscode\gitkad\hetyeeproject

# Download het geoptimaliseerde model
scp -P [JOUW_PORT] -r root@[JOUW_IP]:/workspace/hetyeeproject/models/modernbert_optimized ./models/

# Download training logs
scp -P [JOUW_PORT] root@[JOUW_IP]:/workspace/hetyeeproject/training_optimized.log ./

# Download resultaten
scp -P [JOUW_PORT] root@[JOUW_IP]:/workspace/hetyeeproject/models/modernbert_optimized/test_results_optimized.pkl ./models/modernbert_optimized/
```

### Oude Model (v1.0) - Voor Vergelijking

```powershell
# Download het oude model
scp -P [JOUW_PORT] -r root@[JOUW_IP]:/workspace/hetyeeproject/models/modernbert ./models/

# Download oude training logs
scp -P [JOUW_PORT] root@[JOUW_IP]:/workspace/hetyeeproject/training.log ./
```

**Vervang**:
- `[JOUW_PORT]` → bijv. `48101`
- `[JOUW_IP]` → bijv. `84.2.196.163`

---

## Stap 11: Instance Stoppen

**BELANGRIJK**: Stop je instance als je klaar bent!

1. Ga naar https://cloud.vast.ai/instances/
2. Klik **STOP** of **DESTROY**
   - **STOP**: Data blijft, je betaalt storage
   - **DESTROY**: Alles weg, geen kosten meer

---

## Handige Commands

### Training Commands

| Actie | Command |
|-------|---------|
| **Start geoptimaliseerde training (8 GPUs)** | `torchrun --nproc_per_node=8 src/train_modernbert_optimized.py > training_optimized.log 2>&1 &` |
| **Start test training (1 GPU)** | `python src/train_modernbert_optimized.py > training_test.log 2>&1 &` |
| Stop alle training processen | `pkill -f train_modernbert` |
| Check training processes | `ps aux | grep train_modernbert` |

### Monitoring Commands

| Actie | Command |
|-------|---------|
| GPU checken | `nvidia-smi` |
| GPU usage monitoren (live) | `watch -n 1 nvidia-smi` |
| Logs bekijken (geoptimaliseerd) | `tail -f training_optimized.log` |
| Logs bekijken (laatste 100 regels) | `tail -100 training_optimized.log` |
| Logs doorzoeken op errors | `grep -i error training_optimized.log` |
| Check disk ruimte | `df -h` |
| Check process status | `ps aux | grep python` |

### Analyse Commands

| Actie | Command |
|-------|---------|
| Resultaten analyseren | `python analyze_results.py` |
| Per class analyse | `python per_class_analysis.py` |
| Compare v1.0 vs v2.0 | `python compare_models.py` |

---

## Kosten Inschatting

| GPU | Prijs/uur | $10 = uren | VRAM |
|-----|-----------|-----------|------|
| RTX 3090 | ~$0.30-0.50 | 20-33 uur | 24GB |
| RTX 4090 | ~$0.50-0.80 | 12-20 uur | 24GB |
| A100 | ~$1.50+ | ~6 uur | 40/80GB |

**Aanbeveling voor dit project**: RTX 3090 (beste prijs/prestatie, voldoende VRAM voor ModernBERT)

---

## Training Monitoren

**GPU usage in real-time:**
```bash
watch -n 1 nvidia-smi
```

**Training progress volgen:**
```bash
tail -f training.log
```

**Check of training nog loopt:**
```bash
ps aux | grep train_modernbert
```

---

## Troubleshooting

**SSH werkt niet?**
- Check of je SSH key correct is toegevoegd
- Wacht 1-2 minuten na instance start

**Out of memory tijdens training?**
- Verlaag batch size in `src/train_modernbert.py`
- Kies machine met meer VRAM (24GB aanbevolen)
- Check GPU memory: `nvidia-smi`

**Flash Attention 2 errors?**
- Update: `pip install flash-attn --no-build-isolation`
- Check CUDA versie: `nvcc --version`
- Vereist: CUDA 11.8+ en Ampere GPU of nieuwer

**Training crashes tijdens laden van data?**
- Check of data file aanwezig is: `ls -lh *.jsonl`
- Verifieer disk space: `df -h`

**Instance crashed?**
- Check logs in Vast.ai console
- Mogelijk te weinig disk space (vereist 30GB+)
- Check training log: `tail -100 training.log`

**ModernBERT import errors?**
- Install dependencies: `pip install -r requirements.txt`
- Update transformers: `pip install -U transformers`

---

## Quick Start Samenvatting

Als je ervaren bent met Vast.ai:

```bash
# 1. SSH key toevoegen aan Vast.ai account
# 2. Huur RTX 3090 instance (24GB VRAM, 30GB disk)
# 3. Verbind via SSH

# 4. Upload project
cd /workspace
git clone https://github.com/[jouw-username]/hetyeeproject.git
cd hetyeeproject

# 5. Installeer dependencies
pip install -r requirements.txt

# 6. Start training
nohup python src/train_modernbert.py > training.log 2>&1 &

# 7. Monitor progress
tail -f training.log
watch -n 1 nvidia-smi

# 8. Download resultaten (op lokale PC)
scp -P [PORT] -r root@ssh.vast.ai:/workspace/hetyeeproject/models ./models

# 9. Stop instance wanneer klaar!
```
