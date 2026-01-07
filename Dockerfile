# Dockerfile for ModernBERT Training on Vast.ai
# Base: PyTorch 2.1.0 with CUDA 12.1 support for A100/RTX GPUs

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Flash Attention 2 for GPU optimization (optional but recommended)
RUN pip install flash-attn --no-build-isolation || echo "Flash Attention install failed, continuing without it"

# Copy source code
COPY model.py .
COPY src/ ./src/

# Copy processed data splits
COPY Datasets/processed/ ./Datasets/processed/

# Pre-download ModernBERT model to cache it in the container
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base'); \
    model = AutoModelForSequenceClassification.from_pretrained('answerdotai/ModernBERT-base', num_labels=2); \
    print('ModernBERT tokenizer and model cached successfully')"

# Create output directory for trained models
RUN mkdir -p models/modernbert

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command: run full ModernBERT training
CMD ["python", "src/train_modernbert.py"]
