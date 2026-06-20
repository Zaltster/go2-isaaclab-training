ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
FROM ${ISAAC_SIM_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV ISAACLAB_ROOT=/workspace/IsaacLab
ENV GO2_HEADLESS=1
ENV TERM=xterm
ENV ACCEPT_EULA=Y
WORKDIR /workspace

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

RUN printf 'setuptools<81\n' > /tmp/pip-constraints.txt
ENV PIP_CONSTRAINT=/tmp/pip-constraints.txt

ARG ISAACLAB_REPO=https://github.com/isaac-sim/IsaacLab.git
ARG ISAACLAB_REF=v2.3.2
RUN git clone --depth 1 --branch "${ISAACLAB_REF}" "${ISAACLAB_REPO}" "${ISAACLAB_ROOT}" \
    && cd "${ISAACLAB_ROOT}" \
    && if [ -d /isaac-sim ] && [ ! -e _isaac_sim ]; then ln -s /isaac-sim _isaac_sim; fi \
    && ./isaaclab.sh --install rsl_rl

ENV OMNI_KIT_ACCEPT_EULA=yes
ENV OMNI_KIT_ALLOW_ROOT=1
ENV HEADLESS=1
ENV LIVESTREAM=0
ENV ENABLE_CAMERAS=0
ENV XR=0
ENV OTEL_SERVICE_NAME=go2-isaaclab-training
ENV GO2_OTEL_CONSOLE=0
ENV GO2_OTEL_METRIC_EXPORT_INTERVAL_MS=30000
ENV GO2_EXPERIMENT_NAME=go2_lidar_walk_flat_upright
ENV GO2_MAX_ITERATIONS=100000
ENV GO2_NUM_ENVS=1024
ENV GO2_RECORD_VIDEO=1
ENV GO2_VIDEO_WALLCLOCK_SECONDS=1800
ENV GO2_VIDEO_SPS=25000
ENV GO2_CHECKPOINT_SECONDS=1800
ENV GO2_CHECKPOINT_SPS=25000
ENV GO2_SAVE_INTERVAL_ITERATIONS=50
ENV GO2_AUTO_RESUME=0
ENV GO2_STALL_WATCHDOG_ENABLED=1
ENV GO2_STALL_GRACE_SECONDS=600
ENV GO2_STALL_TIMEOUT_SECONDS=300

WORKDIR /workspace/go2
COPY train_wallclock.py /workspace/go2/train_wallclock.py
COPY scripts /workspace/go2/scripts
COPY source /workspace/go2/source

ENV PYTHONPATH=/workspace/go2/source/go2_isaaclab_tasks:/workspace/go2/source

ENTRYPOINT ["python3", "train_wallclock.py"]
CMD []
