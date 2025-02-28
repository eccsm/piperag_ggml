# --------------------------------------------
# Base Stage: Lightweight foundation image
# --------------------------------------------
FROM python:3.9-slim AS base
WORKDIR /app

# --------------------------------------------
# Builder Stage: Install build tools and dependencies
# --------------------------------------------
FROM base AS builder

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update apt sources to use HTTPS (if available) and configure retries
RUN if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list; \
    fi && \
    echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      rustc \
      cargo \
      cmake && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching.
COPY requirements.txt ./

# Install Python dependencies (with increased timeout to help with network delays).
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Clone and install mlc_llm from source
RUN git clone --recursive https://github.com/mlc-ai/mlc-llm.git && \
    cd mlc-llm && \
    pip install -e .

# Copy the rest of your application code (if needed during build)
COPY . .

# --------------------------------------------
# Final Stage: Create a lightweight runtime image
# --------------------------------------------
FROM base AS final

# Set Python runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime libraries, including libGL (for OpenCV) and libglib2.0-0 (for gthread)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages and binaries from the builder stage.
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/mlc-llm /app/mlc-llm

# Copy the application code from your build context.
COPY . .

# Expose the port that your app listens on (adjust if necessary).
EXPOSE 8000

# Set the default command to run your app (using Uvicorn here for a FastAPI app).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]