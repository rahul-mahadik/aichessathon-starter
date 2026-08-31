ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace
COPY training/requirements-gpu.txt /tmp/requirements-gpu.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-gpu.txt
COPY training /workspace/training

CMD ["python", "training/device_check.py", "--require-cuda"]
