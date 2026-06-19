#!/usr/bin/env python3
"""NPU compiler — walks a ``torch.fx.GraphModule`` and emits NPU instructions.

Supported FX patterns
  * ``call_module`` → ``nn.Linear``  →  ``OP_GEMM`` + GEMM descriptor
  * ``call_function`` → ``aten.relu`` / ``F.relu``  →  ``OP_ACT_RELU``
  * ``call_function`` → ``aten.gelu`` / ``F.gelu``  →  ``OP_ACT_GELU``
  * ``call_function`` → ``aten.sigmoid``            →  ``OP_ACT_SIGMOID``
  * ``call_function`` → ``aten.tanh``               →  ``OP_ACT_TANH``
  * ``call_function`` → ``aten.add``  / ``operator.add``  →  ``OP_VADD``
  * ``call_function`` → ``aten.mul``  / ``operator.mul``  →  ``OP_VMUL``
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .quantize import quantize_per_channel
from .serialize import (
    NpuGraph,
    NpuInstruction,
    Opcode,
    VOpt,
    build_gemm_descriptor,
)


class NpuCompiler:
    """Compile a traced ``torch.fx.GraphModule`` to an ``NpuGraph``."""

    # ------------------------------------------------------------------
    def compile(self, model: torch.fx.GraphModule) -> NpuGraph:
        """Walk every FX node and emit matching NPU instruction(s).

        Returns an ``NpuGraph`` ready for ``serialize_to_binary``.
        """
        instructions: list[NpuInstruction] = []
        descriptors: list[bytes] = []

        for node in model.graph.nodes:
            if node.op == "placeholder":
                continue

            if node.op == "call_module":
                sub = model.get_submodule(node.target)
                if isinstance(sub, nn.Linear):
                    instr, desc = self._lower_linear(sub)
                    instr.desc_ptr = len(descriptors)
                    descriptors.append(desc)
                    instructions.append(instr)

            elif node.op == "call_function":
                instr = self._lower_call_function(node.target)
                if instr is not None:
                    instructions.append(instr)

            elif node.op == "output":
                pass  # terminal node — nothing to emit

        return NpuGraph(instructions=instructions, descriptors=descriptors)

    # ------------------------------------------------------------------
    #  Linear → GEMM
    # ------------------------------------------------------------------
    @staticmethod
    def _lower_linear(module: nn.Linear) -> tuple[NpuInstruction, bytes]:
        """Quantise weights per-channel and emit a GEMM instruction +
        descriptor for a single-batch inference."""
        weight = module.weight.data                # [out_features, in_features]
        out_features, in_features = weight.shape

        # Per-channel INT8 quantisation (weights only — activations are
        # quantised at runtime by the hardware / DMA path).
        _w_int, _qparams = quantize_per_channel(weight)

        desc = build_gemm_descriptor(
            m=1,                    # single-batch
            n=out_features,
            k=in_features,
            a_sram_bank=0,          # input activations
            b_sram_bank=1,          # weights
            o_sram_bank=0,          # output
            a_zp=0, b_zp=0,         # symmetric → zp=0
            out_scale_shr=0,
            out_scale_mul=1,
            relu=0,                 # fused ReLU off by default
            out_zp=0,
        )

        instr = NpuInstruction(
            opcode=Opcode.GEMM,
            dst=0,                  # output SRAM bank
            src_a=0,                # A operand bank
            src_b=1,                # B operand bank
        )

        return instr, desc

    # ------------------------------------------------------------------
    #  call_function → NPU op
    # ------------------------------------------------------------------
    @staticmethod
    def _lower_call_function(target) -> NpuInstruction | None:
        """Map an FX ``call_function`` target to an NPU instruction.

        FX targets may be:
          * strings like ``"aten.relu"``, ``"aten.add.Tensor"``
          * built-in functions like ``operator.add`` → ``__name__ == "add"``
          * torch functions like ``torch.relu``

        Returns ``None`` for unsupported functions.
        """
        # Build a matchable name from the target.
        #   str(target)   – e.g. "aten.relu", "<built-in function add>"
        #   __name__      – e.g. "add", "mul" (for operator builtins)
        t = str(target).lower()
        fn_name = str(getattr(target, "__name__", "")).lower()

        # -- activations ---------------------------------------------------
        if "relu" in t or fn_name == "relu":
            return NpuInstruction(opcode=Opcode.ACT_RELU, dst=0, src_a=0)
        if "gelu" in t or fn_name == "gelu":
            return NpuInstruction(opcode=Opcode.ACT_GELU, dst=0, src_a=0)
        if "sigmoid" in t or fn_name == "sigmoid":
            return NpuInstruction(opcode=Opcode.ACT_SIGMOID, dst=0, src_a=0)
        if "tanh" in t or fn_name == "tanh":
            return NpuInstruction(opcode=Opcode.ACT_TANH, dst=0, src_a=0)

        # -- element-wise arithmetic (VALU, I-type) ------------------------
        if ".add" in t or fn_name == "add":
            return NpuInstruction(
                opcode=Opcode.VADD,
                dst=0,
                src_a=0,
                src_b=1,
                is_itype=True,
                opt=VOpt.ADD,
            )
        if ".mul" in t or fn_name == "mul":
            return NpuInstruction(
                opcode=Opcode.VMUL,
                dst=0,
                src_a=0,
                src_b=1,
                is_itype=True,
                opt=VOpt.MUL,
            )

        return None
