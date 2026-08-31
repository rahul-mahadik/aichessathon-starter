"""Report accelerator availability and optionally require CUDA."""

from __future__ import annotations

import argparse
import json
import platform

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    arguments = parser.parse_args()
    cuda_available = torch.cuda.is_available()
    details = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    print(json.dumps(details, indent=2))
    if arguments.require_cuda and not cuda_available:
        raise SystemExit("CUDA was required but PyTorch cannot see a CUDA device")


if __name__ == "__main__":
    main()
