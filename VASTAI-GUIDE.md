# Vast.ai Handleiding - Stap voor Stap

Je hebt $5 gestort. Hier is precies wat je moet doen.

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
   - **GPU**: RTX 3090 of RTX 4090 (goede prijs/prestatie)
   - **VRAM**: Minimaal 16GB (voor BERTje training)
   - **Disk**: Minimaal 20GB
3. Sorteer op prijs (laag naar hoog)
4. Kies een instance met goede "DLPerf" score

**Budget tip**: Met $5 kun je ~5-10 uur draaien op een RTX 3090.

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

**Optie B: Via Git**

Op de Vast.ai machine:
```bash
cd /workspace
git clone <jouw-repo-url>
```

---

## Stap 8: Dependencies Installeren

Op de Vast.ai machine:

```bash
cd /workspace
pip install -r requirements.txt
```

---

## Stap 9: Training Starten

```bash
python train.py --data_path ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl
```

---

## Stap 10: Resultaten Downloaden

Na training, op je lokale PC:

```powershell
scp -P 12345 -r root@ssh.vast.ai:/workspace/models ./models
```

---

## Stap 11: Instance Stoppen

**BELANGRIJK**: Stop je instance als je klaar bent!

1. Ga naar https://cloud.vast.ai/instances/
2. Klik **STOP** of **DESTROY**
   - **STOP**: Data blijft, je betaalt storage
   - **DESTROY**: Alles weg, geen kosten meer

---

## Handige Commands

| Actie | Command |
|-------|---------|
| GPU checken | `nvidia-smi` |
| Disk ruimte | `df -h` |
| Training in background | `nohup python train.py &` |
| Logs bekijken | `tail -f nohup.out` |

---

## Kosten Inschatting

| GPU | Prijs/uur | $5 = uren |
|-----|-----------|-----------|
| RTX 3090 | ~$0.30-0.50 | 10-16 uur |
| RTX 4090 | ~$0.50-0.80 | 6-10 uur |
| A100 | ~$1.50+ | 3 uur |

---

## Troubleshooting

**SSH werkt niet?**
- Check of je SSH key correct is toegevoegd
- Wacht 1-2 minuten na instance start

**Out of memory?**
- Verlaag batch size in training
- Kies machine met meer VRAM

**Instance crashed?**
- Check logs in console
- Mogelijk te weinig disk space
