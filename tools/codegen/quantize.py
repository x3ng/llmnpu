#!/usr/bin/env python3
"""Per-tensor and per-channel symmetric INT8 quantization for NPU.

Quantization scheme
  symmetric:  scale = absmax / 127, zero_point = 0
  x_int = round(x / scale), clamped to [-128, 127]
"""

from dataclasses import dataclass
import torch


@dataclass
class QuantParams:
    """Quantization parameters for a single tensor or channel."""
    scale: float
    zero_point: int = 0


def quantize_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, QuantParams]:
    """Symmetric per-tensor INT8 quantisation.

    ``x_int = round(x / scale)`` where ``scale = max(|x|) / 127``.

    Args:
        x: float32 tensor of any shape.

    Returns:
        (x_int, params) — x_int is int8, params carries the scale and zero
        point (always 0 for symmetric).
    """
    absmax = float(x.abs().max().item())
    if absmax < 1e-8:
        absmax = 1e-8
    scale = absmax / 127.0
    x_int = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return x_int, QuantParams(scale=scale, zero_point=0)


def quantize_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, list[QuantParams]]:
    """Per-output-channel symmetric INT8 quantisation.

    For a ``Linear`` weight of shape ``[out_features, in_features]`` or a
    ``Conv2d`` weight of shape ``[out_channels, in_channels, ...]``, computes
    a separate scale for each output channel (dimension 0).

    Args:
        w: float32 weight tensor.

    Returns:
        (w_int, params_list) — w_int is int8; params_list has one
        ``QuantParams`` per output channel.
    """
    out_channels = w.shape[0]
    w_int = torch.zeros_like(w, dtype=torch.int8)
    params_list: list[QuantParams] = []

    for c in range(out_channels):
        channel = w[c]
        absmax = float(channel.abs().max().item())
        if absmax < 1e-8:
            absmax = 1e-8
        scale = absmax / 127.0
        params_list.append(QuantParams(scale=scale, zero_point=0))
        w_int[c] = torch.round(channel / scale).clamp(-128, 127).to(torch.int8)

    return w_int, params_list
