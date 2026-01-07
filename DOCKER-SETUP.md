### GOAL

Ik wil naar Vast.ai gaan om de ModernBERT training uit te besteden aan virtuele GPUs. Daarvoor moet ik een Docker instance hebben en die connecten met een SSH key. Vast.ai heeft instructies gegeven, lees ### INSTRUCTIES VAST.AI

### EISEN

1. Maak een Docker image voor ModernBERT training. Het doel is om src/train_modernbert.py te runnen.
2. Lees INSTRUCTIES VAST.AI
3. Schrijf stap-voor-stap uitleg van wat je hebt gedaan onder ### VOLTOOID


### INSTRUCTIES VAST.AI

how do I do that step by step

Found results for step by step guide create instance run script tutorial

Found results for getting started beginner workflow search offers rent GPU

Found results for search offers find GPU machine available
Based on my searches, here's a step-by-step guide to run your ML script on Vast.ai:

Step 1: Get an API Key
Create an API key from your Vast.ai account to authenticate requests.

Step 2: Add Your SSH Key
vastai create ssh-key <your_public_ssh_key>
Step 3: Search for GPU Offers
Use the search offers endpoint to find available machines with the specs you need.

Step 4: Create an Instance
vastai create instance <offer_id> <docker_image> [options]
Use a Docker image like pytorch/pytorch or your custom image.

Step 5: SSH Into Your Instance
Connect via SSH using the connection details provided.

Step 6: Run Your Script
Upload your script and run it on the instance.

Step 7: Retrieve Results
Use cloud copy or copy commands to transfer results back:

vastai cloud copy <instance_id> <src> <dst>
Want to know more? These pages may help:

Create instance
Create SSH key
Cloud copy for results

### VOLTOOID

#### Stap 1: Docker Image Geconfigureerd en Geoptimaliseerd

**Dockerfile aangemaakt** met de volgende eigenschappen:
- Base image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (GPU-ready voor Vast.ai)
- Alle dependencies uit `requirements.txt` geïnstalleerd
- ModernBERT model EN tokenizer (`answerdotai/ModernBERT-base`) worden pre-downloaded en gecached
- Flash Attention 2 geïnstalleerd voor GPU optimalisatie (met fallback)
- Source code (`model.py`, `src/`) en processed data splits (`Datasets/processed/`) worden gekopieerd
- Default CMD voert `src/train_modernbert.py` uit
- Environment variabelen voor CUDA en Python optimalisatie

**.dockerignore aangemaakt** om onnodige bestanden uit te sluiten:
- Git directory, Python caches, virtual environments
- Raw data (alleen processed splits worden gekopieerd)
- Notebooks, IDE configuraties, documentatie
- Analysis scripts en test files

**Belangrijke fix toegepast**: Model pre-download nu correct - download zowel tokenizer als model weights

---

### VERIFICATIE: Controleer of alle benodigde bestanden aanwezig zijn

Voordat je de Docker image bouwt, controleer of deze bestanden/directories aanwezig zijn:

```bash
# Ga naar project root
cd E:\Abdelilah\vscode\gitkad\hetyeeproject

# Controleer benodigde bestanden
ls Dockerfile                           # Docker configuratie
ls .dockerignore                        # Docker exclude patterns
ls requirements.txt                     # Python dependencies
ls model.py                             # ModernBERT classifier
ls src/train_modernbert.py              # Training script
ls src/preprocessing.py                 # Data preprocessing
ls Datasets/processed/train.pkl         # Train split
ls Datasets/processed/val.pkl           # Validation split
ls Datasets/processed/test.pkl          # Test split
```

**Verwachte output**: Alle bestanden moeten bestaan. Als er iets ontbreekt, voer eerst de data preprocessing uit.

---

### LOKAAL TESTEN & BOUWEN

**Zorg dat Docker Desktop actief is voordat je begint!**

```bash
# 1. Bouw de Docker image lokaal (kan 10-15 minuten duren)
docker build -t modernbert-trainer:latest .

# 2. Test lokaal (zonder GPU) - verificeer PyTorch installatie
docker run --rm modernbert-trainer:latest python -c "import torch; print(f'PyTorch {torch.__version__} geïnstalleerd')"

# 3. Test met GPU (als je lokaal NVIDIA GPU hebt met CUDA)
docker run --rm --gpus all modernbert-trainer:latest nvidia-smi

# 4. Test of training script kan starten (dry-run, stop na paar seconden met Ctrl+C)
docker run --rm modernbert-trainer:latest python src/train_modernbert.py
```

**Troubleshooting**:
- "Docker daemon not running": Start Docker Desktop
- "no matching manifest": Gebruik de juiste base image voor je platform
- "CUDA error": Controleer of NVIDIA Docker runtime geïnstalleerd is (`nvidia-docker2`)

---

### PUSH NAAR DOCKER HUB (voor Vast.ai)

```bash
# 1. Login bij Docker Hub
docker login

# 2. Tag de image met je Docker Hub username
docker tag modernbert-trainer:latest <jouw-username>/modernbert-trainer:latest

# 3. Push naar Docker Hub
docker push <jouw-username>/modernbert-trainer:latest
```

---

### VAST.AI DEPLOYMENT

#### Stap 1: Installeer en Configureer Vast.ai CLI

```bash
# Installeer vastai CLI (indien nog niet gedaan)
pip install vastai

# Stel API key in (haal API key op van https://cloud.vast.ai/account/)
vastai set api-key <jouw-api-key>

# Voeg SSH key toe voor remote access
# Windows (PowerShell):
vastai create ssh-key "$(Get-Content $env:USERPROFILE\.ssh\id_rsa.pub)"
# Linux/Mac:
vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)"
```

#### Stap 2: Zoek en Huur GPU Instance

```bash
# Zoek beschikbare GPU's (minimaal 24GB VRAM voor 8192 tokens)
# Sorteer op prijs (goedkoopste eerst)
vastai search offers "cuda_vers >= 12.0 gpu_ram >= 24 reliability > 0.95 rentable=true" -o "dph_total"

# Aanbevolen GPU's voor dit project:
# - RTX 4090 (24GB): ~$0.30/uur - batch_size=4, ~5-6 uur training
# - A6000 (48GB):   ~$0.40/uur - batch_size=4, ~4-5 uur training
# - A100 (40GB):    ~$1.00/uur - batch_size=4, ~3-4 uur training
```

#### Stap 3: Start Training Instance

```bash
# Maak instance aan (vervang <offer_id> en <jouw-dockerhub-username>)
vastai create instance <offer_id> \
    --image <jouw-dockerhub-username>/modernbert-trainer:latest \
    --disk 50 \
    --onstart-cmd "python src/train_modernbert.py"

# Voorbeeld met concrete waardes:
# vastai create instance 12345678 \
#     --image johndoe/modernbert-trainer:latest \
#     --disk 50 \
#     --onstart-cmd "python src/train_modernbert.py"
```

#### Stap 4: Monitor Training Progress

```bash
# Bekijk je actieve instances
vastai show instances

# SSH naar instance voor real-time monitoring
vastai ssh-url <instance_id>

# Eenmaal ingelogd via SSH:
# - Bekijk training logs: tail -f /app/training.log (als je logging hebt toegevoegd)
# - Check GPU gebruik: nvidia-smi
# - Bekijk Python processen: ps aux | grep python
# - Monitor disk space: df -h
```

#### Stap 5: Haal Resultaten Op

```bash
# Na voltooiing training (3-6 uur afhankelijk van GPU):
# Kopieer trained model en resultaten terug naar lokaal
vastai cloud copy <instance_id>:/app/models/modernbert ./local_results/

# Verifieer dat je deze bestanden hebt:
# - model weights (pytorch_model.bin of model.safetensors)
# - label_mapping.pkl
# - test_results.pkl
# - training logs
```

#### Stap 6: Stop Instance om Kosten te Besparen

```bash
# Stop instance na ophalen resultaten
vastai destroy instance <instance_id>

# Controleer of instance gestopt is
vastai show instances
```

---

### BESTANDEN AANGEMAAKT / GEVERIFIEERD

| Bestand | Status | Beschrijving |
|---------|--------|--------------|
| `Dockerfile` | ✅ Geoptimaliseerd | Docker configuratie met model pre-caching |
| `.dockerignore` | ✅ Compleet | Exclude patterns voor efficiënte build |
| `requirements.txt` | ✅ Aanwezig | Alle Python dependencies |
| `model.py` | ✅ Aanwezig | ModernBERT classifier implementatie |
| `src/train_modernbert.py` | ✅ Aanwezig | Training script |
| `src/preprocessing.py` | ✅ Aanwezig | Data preprocessing functies |
| `Datasets/processed/*.pkl` | ✅ Aanwezig | Train/val/test splits (3 bestanden) |

---

### TECHNISCHE SPECIFICATIES

**Container eigenschappen**:
- Base: PyTorch 2.1.0 + CUDA 12.1 + cuDNN 8
- Python dependencies: transformers, accelerate, scikit-learn, pandas, etc.
- Pre-cached: ModernBERT-base model + tokenizer (~1.5GB)
- Flash Attention 2: Optioneel geïnstalleerd voor 30% snelheidswinst
- Data: 3 processed splits (~13.7K train, 2.9K val, 2.9K test)

**GPU vereisten voor 8192 token context**:
- **Minimaal**: 24GB VRAM (RTX 4090, A5000)
- **Aanbevolen**: 40GB+ VRAM (A100, A6000)
- **Batch sizes**: 2-4 (afhankelijk van GPU)
- **Training tijd**: 3-6 uur (afhankelijk van GPU)

**Kosten inschatting Vast.ai**:
- RTX 4090: ~$0.30/uur × 5 uur = **$1.50 per run**
- A6000: ~$0.40/uur × 4 uur = **$1.60 per run**
- A100: ~$1.00/uur × 3 uur = **$3.00 per run**

**Output bestanden** (opgeslagen in `/app/models/modernbert/`):
- `pytorch_model.bin` of `model.safetensors` - Trained model weights
- `label_mapping.pkl` - Label ID mapping
- `test_results.pkl` - Final evaluation metrics
- `config.json` - Model configuratie
- `training_args.bin` - Training parameters

---

### VOLGENDE STAPPEN

1. **Start Docker Desktop** (als je lokaal wilt testen)

2. **Bouw en test lokaal** (optioneel):
   ```bash
   docker build -t modernbert-trainer:latest .
   docker run --rm modernbert-trainer:latest python -c "import torch; print(torch.__version__)"
   ```

3. **Push naar Docker Hub**:
   ```bash
   docker login
   docker tag modernbert-trainer:latest <username>/modernbert-trainer:latest
   docker push <username>/modernbert-trainer:latest
   ```

4. **Deploy op Vast.ai**:
   ```bash
   pip install vastai
   vastai set api-key <jouw-api-key>
   vastai search offers "cuda_vers >= 12.0 gpu_ram >= 24" -o "dph_total"
   vastai create instance <offer_id> --image <username>/modernbert-trainer:latest
   ```

5. **Monitor en haal resultaten op**:
   ```bash
   vastai show instances
   vastai cloud copy <instance_id>:/app/models/modernbert ./local_results/
   ```

6. **Analyseer resultaten lokaal** met `analyze_results.py` of `per_class_analysis.py`

---

### TROUBLESHOOTING

**"Docker daemon not running"**:
- Start Docker Desktop en wacht tot het volledig gestart is

**"CUDA error: out of memory"**:
- Verlaag batch_size in `model.py` van 4 naar 2
- Of gebruik GPU met meer VRAM (A100/A6000)

**"Vast.ai instance failed to start"**:
- Controleer of Docker image correct gepusht is naar Docker Hub
- Verifieer dat image publiek toegankelijk is (niet private)

**"Training stopt zonder foutmelding"**:
- Check Vast.ai instance logs: `vastai ssh-url <id>` en bekijk logs
- Mogelijk out-of-memory - verlaag batch_size

**"Model download fails in container"**:
- Huggingface Hub kan geblokkeerd zijn - voeg HF_TOKEN toe als environment variable
- Of gebruik een VPN/proxy in de container

---

### CHECKLIST VOOR DEPLOYMENT

- [ ] Dockerfile en .dockerignore bestaan
- [ ] Alle dependencies in requirements.txt
- [ ] Processed data splits aanwezig (train/val/test.pkl)
- [ ] Docker Desktop gestart (voor lokale build)
- [ ] Docker Hub account aangemaakt
- [ ] Vast.ai account aangemaakt met credits
- [ ] SSH key gegenereerd en toegevoegd aan Vast.ai
- [ ] Image gebouwd en getest lokaal
- [ ] Image gepusht naar Docker Hub
- [ ] GPU instance gevonden en gehuurd
- [ ] Training gestart op Vast.ai
- [ ] Monitoring ingesteld (SSH toegang getest)
- [ ] Resultaten opgehaald na training
- [ ] Instance gestopt om kosten te besparen