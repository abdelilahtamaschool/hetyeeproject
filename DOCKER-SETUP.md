### GOAL

Ik wil naar Vast.ai gaan om de training uit te besteden aan virtuele GPUs. Daarvoor moet ik een Docker instance hebben en die connecten met een SSH key. Vast.ai heeft instructies gegeven, lees ### INSTRUCTIES VAST.AI

Ik wil dat jij dit regelt.

### EISEN

1. Maak een Docker image van de script. Het doel is om train.py te runnen.
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

#### Stap 1: Docker Image Gebouwd

**Dockerfile aangemaakt** met de volgende eigenschappen:
- Base image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (GPU-ready voor Vast.ai)
- Alle dependencies uit `requirements.txt` geinstalleerd
- BERTje model (`GroNLP/bert-base-dutch-cased`) wordt pre-downloaded
- Source code en data worden in de container gekopieerd
- Default CMD voert `train.py` uit

**.dockerignore aangemaakt** om onnodige bestanden uit te sluiten (caches, models, .git, etc.)

---

### LOKAAL TESTEN & BOUWEN

```bash
# 1. Bouw de Docker image lokaal
docker build -t bertje-trainer:latest .

# 2. Test lokaal (zonder GPU)
docker run --rm bertje-trainer:latest python -c "import torch; print(torch.__version__)"

# 3. Test met GPU (als je lokaal NVIDIA GPU hebt)
docker run --rm --gpus all bertje-trainer:latest nvidia-smi
```

---

### PUSH NAAR DOCKER HUB (voor Vast.ai)

```bash
# 1. Login bij Docker Hub
docker login

# 2. Tag de image met je Docker Hub username
docker tag bertje-trainer:latest <jouw-username>/bertje-trainer:latest

# 3. Push naar Docker Hub
docker push <jouw-username>/bertje-trainer:latest
```

---

### VAST.AI STAPPEN

```bash
# 1. Installeer vastai CLI (indien nog niet gedaan)
pip install vastai

# 2. Stel API key in
vastai set api-key <jouw-api-key>

# 3. Voeg SSH key toe
vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)"

# 4. Zoek beschikbare GPU machines
vastai search offers "cuda_vers >= 12.0 gpu_ram >= 16 reliability > 0.95 rentable=true" -o "dph_total"

# 5. Maak instance aan (vervang <offer_id> met ID uit stap 4)
vastai create instance <offer_id> \
    --image <jouw-username>/bertje-trainer:latest \
    --disk 50 \
    --onstart-cmd "python train.py --epochs 15 --batch_size 16"

# 6. Bekijk je instances
vastai show instances

# 7. SSH naar instance (voor monitoring)
vastai ssh-url <instance_id>

# 8. Kopieer resultaten terug na training
vastai cloud copy <instance_id>:/app/models ./results/
```

---

### BESTANDEN AANGEMAAKT

| Bestand | Beschrijving |
|---------|--------------|
| `Dockerfile` | Docker configuratie voor training container |
| `.dockerignore` | Exclude patterns voor Docker build |

---

### OPMERKINGEN

- De image bevat de training data (~20K aktes)
- BERTje model wordt automatisch gedownload in de container
- GPU vereist minimaal 16GB VRAM voor batch_size=16
- Training duurt ~2-4 uur afhankelijk van GPU
- Resultaten komen in `/app/models/` binnen de container