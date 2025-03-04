# --------------------------------------------
# Base Stage: Use a Miniconda image as the starting point
# --------------------------------------------
FROM continuumio/miniconda3:latest AS base
WORKDIR /app

# --------------------------------------------
# Builder Stage: Create the conda environment and build any extras
# --------------------------------------------
FROM base AS builder

# Install mamba for faster package management
RUN conda install -y mamba -n base -c conda-forge

# Copy the exported environment file into the image
COPY environment.yml .

# Create the conda environment as defined in environment.yml.
RUN mamba env create -f environment.yml && conda clean -afy

# Switch to the conda environment "mlc-prebuilt" for subsequent commands
SHELL ["conda", "run", "-n", "mlc-prebuilt", "/bin/bash", "-c"]

# Install build dependencies needed for additional packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      cmake \
      curl \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install a specific version of Rust directly
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
# Make Rust available in the current shell session
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    . $HOME/.cargo/env && \
    rustup default 1.70.0 && \
    rustup show

# Rename conda's internal linker to avoid glibc conflicts
RUN mv "$CONDA_PREFIX/compiler_compat/ld" "$CONDA_PREFIX/compiler_compat/ld_bak"

# Install llama-cpp-python using the pre-built CPU wheel index
RUN pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Clone and build MLC-LLM with cargo/rustc environment loaded
RUN . $HOME/.cargo/env && \
    git clone --recursive https://github.com/mlc-ai/mlc-llm.git && \
    cd mlc-llm && \
    mkdir -p build && cd build && \
    echo 'set(CMAKE_BUILD_TYPE Release)' > ../cmake/config.cmake && \
    echo 'set(USE_CUDA OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_METAL OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_OPENCL OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_VULKAN OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_ROCM OFF)' >> ../cmake/config.cmake && \
    echo 'set(USE_OPENMP ON)' >> ../cmake/config.cmake && \
    cmake .. -DUSE_CUDA=OFF -DUSE_METAL=OFF -DUSE_OPENCL=OFF -DUSE_VULKAN=OFF -DUSE_ROCM=OFF -DUSE_OPENMP=ON && \
    cmake --build . --parallel $(nproc)

# Install the MLC-LLM Python package in editable mode
RUN cd mlc-llm/python && pip install -e .

# ---------------------------
# Install TVM Unity (Prebuilt Package)
# ---------------------------
# According to https://llm.mlc.ai/docs/install/tvm.html, install the prebuilt TVM Unity package.
# Here we install the CPU version. For GPU support, choose the appropriate package (e.g., mlc-ai-nightly-cu122).
RUN python -m pip install --pre -U -f https://mlc.ai/wheels mlc-ai-nightly-cpu

# Copy the rest of your application code
COPY . .

# (Optional) Upgrade torchvision to avoid operator errors with torch
RUN pip install --upgrade torchvision

# --------------------------------------------
# Final Stage: Create a lightweight runtime image
# --------------------------------------------
FROM continuumio/miniconda3:latest AS final
WORKDIR /app

# Install runtime libraries (e.g., for OpenCV or similar)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Copy the conda environment from the builder stage.
COPY --from=builder /opt/conda/envs/mlc-prebuilt /opt/conda/envs/mlc-prebuilt

# Set the PATH so that the conda environment is used by default
ENV PATH=/opt/conda/envs/mlc-prebuilt/bin:$PATH
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy your application code from the builder stage
COPY --from=builder /app /app

# Expose the port your app listens on (adjust if needed)
EXPOSE 8000

# Set the default command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
