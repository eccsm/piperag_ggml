# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-pip \
    python3-dev \
    libcurl4-openssl-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    make LLAMA_CUBLAS=1  # Enable CUDA

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy model and code
COPY models/vicuna-7b-v1.5.Q4_K_M.gguf /app/models/
COPY app.py .

# Expose API port
EXPOSE 8000

# Start the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]