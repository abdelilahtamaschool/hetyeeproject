# Dockerfile for BERTje Akte Classification Training
# Optimized for Vast.ai GPU instances

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the BERTje model to avoid download during training
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased'); \
    AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')"

# Copy source code
COPY src/ ./src/
COPY train.py .
COPY predict.py .

# Copy data file
COPY ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl .

# Create output directories
RUN mkdir -p models outputs

# Default command - run training
CMD ["python", "train.py", \
     "--data_path", "ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl", \
     "--output_dir", "models", \
     "--epochs", "15", \
     "--batch_size", "16"]
