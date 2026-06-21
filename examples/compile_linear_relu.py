#!/usr/bin/env python3
"""Demo: PyTorch Linear+ReLU model -> NPU .npu binary."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT / "tools"))

from codegen.npu_torch import compile_model  # noqa: E402


class LinearRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def main() -> None:
    out_path = ROOT / "build" / "linear_relu.npu"
    result = compile_model(LinearRelu(), torch.randn(1, 16), out_path)

    print(f"wrote: {result.out_path}")
    print(f"instructions: {result.num_instructions}")
    print(f"descriptors: {result.num_descriptors}")
    print(f"input_shape: {result.input_shape}")


if __name__ == "__main__":
    main()
