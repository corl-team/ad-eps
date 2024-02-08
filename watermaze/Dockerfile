FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
WORKDIR /workspace

# python, dependencies for mujoco-py (and in general useful for RL research),
# from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    git \
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

# Install requirements
COPY ./requirements.txt /workspace/
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

# Install Shimmy requirements
RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    unzip \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    patchelf \
    ffmpeg cmake \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . /usr/local/shimmy/
WORKDIR /usr/local/shimmy/

# Include Shimmy in Python path
ENV PYTHONPATH="$PYTHONPATH:/usr/local/shimmy/"

# Install Shimmy
RUN pip install -U "shimmy[dm-lab, testing] @ git+https://github.com/user402503/Shimmy.git" --no-cache-dir
RUN mkdir -p bin && mv docker_entrypoint bin/docker_entrypoint

RUN pip install pyrallis tqdm
RUN pip install tqdm

# Install DM lab requirements
RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    build-essential curl freeglut3-dev gettext git libffi-dev libglu1-mesa \
    libglu1-mesa-dev libjpeg-dev liblua5.1-0-dev libosmesa6-dev \
    libsdl2-dev lua5.1 pkg-config python3-dev \
    software-properties-common unzip zip zlib1g-dev g++

# Install Bazel
RUN apt-get install -y apt-transport-https curl gnupg  \
    && curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg  \
    && mv bazel.gpg /etc/apt/trusted.gpg.d/  \
    && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
    && apt-get update && apt-get install -y bazel

# Build DM lab
# COPY ~/envs/lab_ad/* /workspace/lab/
# RUN 
RUN git clone https://github.com/suessmann/lab \
    && cd lab \
    && echo 'build --cxxopt=-std=c++17' > .bazelrc \
    && bazel build -c opt //python/pip_package:build_pip_package  \
    && ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg \
    && pip3 install --force-reinstall /tmp/dmlab_pkg/deepmind_lab-*.whl \
    && cd .. \
    && rm -rf lab

ENTRYPOINT ["/usr/local/shimmy/bin/docker_entrypoint"]
