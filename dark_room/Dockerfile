FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
WORKDIR /workspace

# python, dependencies for mujoco-py (and in general useful for RL research),
# from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    build-essential \
    patchelf \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/

RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt