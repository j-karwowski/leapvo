# Base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"

# Set the working directory and copy important files
WORKDIR /workspace
COPY environment.yml /workspace/
COPY eigen-3.4.0.zip /workspace/
# COPY . /workspace/

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
RUN unzip eigen-3.4.0.zip -d thirdparty

CMD ["conda", "run", "-n", "leapvo-env", "python", "run.py"]
# CMD ["conda", "run", "-n", "leapvo-env", "python", "-m", "main.eval", "--config-path=../configs", "--config-name=demo", "data.imagedir=/data/samples/sintel_market_5/frames", "data.calib=data/samples/sintel_market_5/calib.txt", "data.savedir=logs/sintel_market_5", "data.name=sintel_market_5", "save_trajectory=true", "save_video=true", "save_plot=true"]
