"""Train a small value network from NPZ features and export PT and ONNX weights."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

FEATURES = 773  # 12 piece planes * 64 squares + side to move + four castling rights


class PositionValueNet(nn.Module):
    """A deliberately small starting point suitable for CPU inference experiments."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURES, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    return device


def load_dataset(path: Path | None, smoke: bool, seed: int) -> TensorDataset:
    if path is None:
        if not smoke:
            raise SystemExit("pass --data training/data/positions.npz or use --smoke")
        generator = torch.Generator().manual_seed(seed)
        features = torch.randn(4_096, FEATURES, generator=generator)
        targets = torch.tanh(features[:, :16].sum(dim=1, keepdim=True) / 8.0)
        return TensorDataset(features, targets)

    with np.load(path) as archive:
        if "features" not in archive or "targets" not in archive:
            raise SystemExit(f"{path} must contain arrays named features and targets")
        feature_array = np.asarray(archive["features"], dtype=np.float32)
        target_array = np.asarray(archive["targets"], dtype=np.float32).reshape(-1, 1)
    if feature_array.ndim != 2 or feature_array.shape[1] != FEATURES:
        raise SystemExit(f"features must have shape (N, {FEATURES}), got {feature_array.shape}")
    if len(feature_array) != len(target_array):
        raise SystemExit("features and targets have different row counts")
    return TensorDataset(torch.from_numpy(feature_array), torch.from_numpy(target_array))


def export_model(model: PositionValueNet, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    example = torch.zeros(1, FEATURES, dtype=torch.float32)
    checkpoint = output.with_suffix(".pt")
    torch.save({"state_dict": model.state_dict(), "features": FEATURES}, checkpoint)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["features"],
        output_names=["value"],
        dynamic_axes={"features": {0: "batch"}, "value": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_value = session.run(None, {"features": example.numpy()})[0]
    torch_value = model(example).detach().numpy()
    max_error = float(np.max(np.abs(onnx_value - torch_value)))
    metadata = {
        "features": FEATURES,
        "onnx_bytes": output.stat().st_size,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "onnx_validation_max_absolute_error": max_error,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def train(arguments: argparse.Namespace) -> None:
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = select_device(arguments.device)
    dataset = load_dataset(arguments.data, arguments.smoke, arguments.seed)
    validation_size = max(1, len(dataset) // 10)
    train_size = len(dataset) - validation_size
    train_data, validation_data = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(arguments.seed),
    )
    train_loader = DataLoader(
        train_data,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(validation_data, batch_size=arguments.batch_size)
    model = PositionValueNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    loss_function = nn.MSELoss()
    writer = SummaryWriter(arguments.log_dir)
    print(f"training {len(train_data):,} rows on {device} for {arguments.epochs} epoch(s)")

    for epoch in range(arguments.epochs):
        model.train()
        training_loss = 0.0
        for features, targets in tqdm(train_loader, desc=f"epoch {epoch + 1}"):
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(features), targets)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(features)

        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for features, targets in validation_loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                validation_loss += float(loss_function(model(features), targets)) * len(features)
        training_loss /= len(train_data)
        validation_loss /= len(validation_data)
        writer.add_scalars(
            "loss",
            {"train": training_loss, "validation": validation_loss},
            epoch + 1,
        )
        print(
            f"epoch {epoch + 1}: train_loss={training_loss:.6f} "
            f"validation_loss={validation_loss:.6f}"
        )
    writer.close()
    export_model(model, arguments.output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("weights/value.onnx"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-dir", type=Path, default=Path("training/runs/latest"))
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    train(arguments)


if __name__ == "__main__":
    main()
