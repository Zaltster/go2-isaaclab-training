FROM nvcr.io/nvidia/isaac-sim:5.1.0

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=yes
ENV OMNI_KIT_ALLOW_ROOT=1
ENV TERM=xterm
ENV ISAACLAB_ROOT=/workspace/IsaacLab
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

WORKDIR /workspace/probe
COPY artifacts /workspace/probe/artifacts
COPY probe_wtw_policy.py /workspace/probe/probe_wtw_policy.py
ENTRYPOINT ["/workspace/IsaacLab/isaaclab.sh", "-p", "probe_wtw_policy.py"]
