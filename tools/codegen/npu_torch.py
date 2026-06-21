#!/usr/bin/env python3
"""Small PyTorch-facing entry point for the NPU codegen pipeline.

This module intentionally stays thin: it traces a ``torch.nn.Module`` with
``torch.fx``, reuses ``NpuCompiler`` for lowering, then writes the resulting
instruction stream and descriptor table with ``serialize_to_binary``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .compiler import NpuCompiler
from .serialize import NpuGraph, serialize_to_binary


@dataclass(frozen=True)
class CompileResult:
    """Artifacts produced by ``compile_model``."""

    graph: NpuGraph
    out_path: Path
    input_shape: tuple[int, ...] | tuple[tuple[int, ...], ...]
    num_instructions: int
    num_descriptors: int


def _shape_of(example_input: Any) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
    if isinstance(example_input, torch.Tensor):
        return tuple(int(dim) for dim in example_input.shape)
    if isinstance(example_input, (tuple, list)):
        return tuple(tuple(int(dim) for dim in item.shape) for item in example_input)
    raise TypeError("example_input must be a torch.Tensor or a tuple/list of tensors")


def compile_model(
    model: torch.nn.Module,
    example_input: torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor],
    out_path: str | Path,
) -> CompileResult:
    """Compile a PyTorch module to a firmware-consumable ``.npu`` file.

    Args:
        model: PyTorch module containing currently supported FX patterns
            such as ``nn.Linear`` followed by ReLU/GELU or simple elementwise
            ops.
        example_input: Representative input tensor(s). The current compiler
            uses FX symbolic tracing, so this is used to validate that the
            module can run and to record shape metadata for demos/reports.
        out_path: Destination path for the serialized ``.npu`` binary.

    Returns:
        ``CompileResult`` containing the lowered graph and artifact metadata.
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    input_shape = _shape_of(example_input)
    model.eval()

    run_args = tuple(example_input) if isinstance(example_input, (tuple, list)) else (example_input,)
    with torch.no_grad():
        model(*run_args)

    traced = torch.fx.symbolic_trace(model)
    graph = NpuCompiler().compile(traced)
    output = Path(out_path)
    serialize_to_binary(graph, output)

    return CompileResult(
        graph=graph,
        out_path=output,
        input_shape=input_shape,
        num_instructions=len(graph.instructions),
        num_descriptors=len(graph.descriptors),
    )
