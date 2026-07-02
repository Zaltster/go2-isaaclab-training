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

ENV PYTHONUNBUFFERED=1
ENV OMNI_KIT_ACCEPT_EULA=yes
ENV OMNI_KIT_ALLOW_ROOT=1
ENV HEADLESS=1
ENV LIVESTREAM=0
ENV ENABLE_CAMERAS=1
ENV XR=0
ENV GO2_TASK=Go2-Lidar-Walk-Flat-v0
ENV GO2_WTW_ARTIFACT_DIR=/workspace/go2/artifacts
ENV GO2_WTW_HIER_NAV_EXPERIMENT=go2_wtw_hier_nav_frozen_walk
ENV GO2_WTW_HIER_NAV_RUN_NAME=smooth_sparkedeb_wtw_frozen_walk_nav
ENV GO2_WTW_HIER_NAV_LOG_ROOT=/logs/wtw_hier_nav
ENV GO2_WTW_HIER_NAV_NUM_ENVS=1
ENV GO2_WTW_HIER_NAV_MAX_UPDATES=20000
ENV GO2_WTW_HIER_NAV_WALLCLOCK_SECONDS=3600
ENV GO2_WTW_HIER_NAV_LEARN_DEVICE=cpu
ENV GO2_WTW_HIER_NAV_WTW_DEVICE=cpu
ENV GO2_WTW_HIER_NAV_TORCH_THREADS=1
ENV GO2_WTW_HIER_NAV_TORCH_INTEROP_THREADS=1
ENV GO2_WTW_HIER_NAV_USE_ENV_REWARD=1
ARG GO2_WTW_HIER_NAV_BUILD_ID=20260702_cpu_proxy_reward_v14
RUN echo "${GO2_WTW_HIER_NAV_BUILD_ID}" > /tmp/go2_wtw_hier_nav_build_id

WORKDIR /workspace/go2
COPY go2_entrypoint.py /workspace/go2/go2_entrypoint.py
COPY artifacts /workspace/go2/artifacts
COPY scripts /workspace/go2/scripts
COPY source /workspace/go2/source

ENV PYTHONPATH=/workspace/go2/source/go2_isaaclab_tasks:/workspace/go2/source

ENTRYPOINT ["python3", "go2_entrypoint.py"]
CMD []
