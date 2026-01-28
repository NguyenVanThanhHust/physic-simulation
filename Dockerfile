FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        git \
        vim \
        htop \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "jax[cuda12_cudnn9]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip install warp-lang[extras]