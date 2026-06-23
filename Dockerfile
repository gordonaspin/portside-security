FROM python:3.12-slim AS base

# 1. Combine system updates and clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    bash \
    vim \
    git \
    python3-tk \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. CACHE MOUNT 1: Speed up core pip tool upgrades
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip setuptools build

# 3. Handle Timezone settings natively
ARG TZ="America/New_York"
ENV TZ=${TZ}

# 4. Create the docker user and group immediately
ARG GROUP_NAME=docker
ARG USER_NAME=docker
ARG USER_UID=1000
ARG GROUP_GID=1000

RUN groupadd --gid ${GROUP_GID} ${GROUP_NAME} \
    && useradd --uid ${USER_UID} --gid ${GROUP_GID} --create-home --home-dir /home/${USER_NAME} ${USER_NAME}

# 5. CACHE MOUNT 2: Mount pip cache during dependency installation.
WORKDIR /tmp
COPY dist/*.whl .
COPY requirements.txt .

# FIX: Force pip to look ONLY at the PyTorch CPU mirror for everything.
# 1. We install torch/torchvision strictly from the CPU index.
# 2. We use --index-url (NOT --extra-index-url) for requirements.txt to prevent pip 
#    from checking standard PyPI for torch-adjacent sub-packages.
# FIX: Use download.pytorch.org/whl/cpu to target the exact CPU repository
# FIX: Explicitly target download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --ignore-installed torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple && \
    pip install *.whl && \
    rm -rf /tmp/*

# 6. Set up the application home directory
WORKDIR /home/docker

# 7. Copy assets directly with correct ownership (no layer duplication)
COPY --chown=docker:docker logging-config.json .
COPY --chown=docker:docker nvr.json .
COPY --chown=docker:docker backend/model/yolov8n.pt ./backend/model/yolov8n.pt
COPY --chown=docker:docker backend/frontend_dist ./backend/frontend_dist

USER docker
EXPOSE 7860
ENTRYPOINT [ "pynvr" ]

