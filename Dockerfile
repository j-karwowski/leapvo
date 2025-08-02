# Base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS="yes"

# Set the working directory and copy important files
WORKDIR /workspace
COPY environment.yml /workspace/
COPY eigen-3.4.0.zip /workspace/

# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    ffmpeg \
    git \
    curl \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installiere Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/miniconda && \
    rm /miniconda.sh

# Add conda path
ENV PATH=/opt/miniconda/bin:$PATH

# Create conda environment based on environment.yml
RUN conda env create -f environment.yml --debug

# Activate conda environment and set as standard
RUN echo "source activate leapvo-env" > ~/.bashrc
ENV PATH=/opt/miniconda/envs/leapvo-env/bin:$PATH

# Add cuda path
ENV PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}

# Setup Eigen
RUN rm -rf thirdparty && mkdir thirdparty && unzip eigen-3.4.0.zip -d thirdparty

COPY . /workspace/

# Expose API port
# EXPOSE 8000

# Run FastAPI
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Install CUDA extensions on runtime 
# CMD ["bash", "-c", "pip install ."]

# Use bash to run the script
COPY docker/start.sh /workspace/docker/start.sh
RUN chmod +x docker/start.sh

ENTRYPOINT ["/bin/bash", "docker/start.sh"]
