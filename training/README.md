# GPU training workspace

Training is deliberately separate from the competition runtime. The submitted agent cannot use a
GPU or install packages, but models may be trained elsewhere and exported into `weights/`.

## Local CPU smoke test

```bash
make train-setup
make train-smoke
```

The smoke test trains on synthetic data, writes PT and ONNX files under `/tmp`, and verifies that
ONNX Runtime reproduces the PyTorch output. It tests the pipeline, not chess strength.

## AWS GPU host

AWS is the primary GPU path. Use the current PyTorch 2.13 Deep Learning AMI and follow
[`infra/aws/README.md`](../infra/aws/README.md); it already includes the matching CUDA-enabled
framework, NVIDIA driver, ONNX Runtime, and Systems Manager agent.

## Portable NVIDIA container fallback

The host needs a compatible NVIDIA driver, Docker, and NVIDIA Container Toolkit. The image is
pinned to PyTorch 2.7.0 with CUDA 12.8; override the `PYTORCH_IMAGE` build argument if the chosen
host requires another supported CUDA stack. This path is useful outside the DLAMI but is not the
default AWS workflow.

```bash
make gpu-build
make gpu-smoke
make gpu-train
```

`gpu-train` expects `training/data/positions.npz` with:

- `features`: float32 array shaped `(N, 773)`.
- `targets`: float32 array shaped `(N,)` or `(N, 1)`, normalized to `[-1, 1]`.

The 773 features are reserved for twelve 8x8 piece planes, side to move, and four castling rights.
The exact encoding and label-generation pipeline should be versioned before real training begins.
Engine-labelled training positions are permitted by the competition rules, but the shipped model
must be trained by the team.

Outputs land in `weights/value.onnx`, `weights/value.pt`, and `weights/value.json`. Only ship the
format the eventual agent actually loads, and keep the entire uncompressed submission under 50 MB.
