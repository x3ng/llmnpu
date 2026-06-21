#!/usr/bin/env python3
"""NPU binary serializer — opcodes, instructions, descriptors, .npu writer.

Binary layout of a .npu file:

+──────────+──────────+────────────────────────────────────────+
| offset   | size     | field                                   |
+──────────+──────────+────────────────────────────────────────+
| 0        | 4        | magic       "NPUC"                      |
| 4        | 4        | version      uint32 LE                  |
| 8        | 4        | num_instr    uint32 LE                  |
| 12       | 4        | num_desc     uint32 LE                  |
| 16       | N*4      | instrs[]     NpuInstruction.encode()    |
| 16+N*4   | M*19     | descs[]      gemm_desc_t (19 bytes)     |
+──────────+──────────+────────────────────────────────────────+
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

# ── Magic & version ──────────────────────────────────────────────────────

MAGIC = b"NPUC"
VERSION = 1

# ── Opcodes ──────────────────────────────────────────────────────────────

class Opcode(IntEnum):
    """NPU instruction opcodes (8-bit, bits [31:24] of R-type)."""
    GEMM        = 0x01
    GEMM_SCALE  = 0x02
    VADD        = 0x10   # shared opcode — distinguished by VOpt in I-type
    VMUL        = 0x10
    VMAX        = 0x10
    VMOV        = 0x11
    VCMP        = 0x11
    ACT_RELU    = 0x20
    ACT_GELU    = 0x21
    ACT_SIGMOID = 0x22
    ACT_TANH    = 0x23
    QUANT       = 0x30
    DEQUANT     = 0x31
    DMA_LD      = 0x40
    DMA_ST      = 0x41
    DMA_2D      = 0x42
    SYNC        = 0xF0
    WFI         = 0xF1
    NOP         = 0xFF


class VOpt(IntEnum):
    """VALU sub-opcodes (OPT field, bits [27:20] of I-type)."""
    ADD = 0x00
    SUB = 0x01
    MUL = 0x02
    MIN = 0x03
    MAX = 0x04
    AND = 0x05
    OR  = 0x06
    XOR = 0x07


# ── Instruction ──────────────────────────────────────────────────────────

@dataclass
class NpuInstruction:
    """Single 32-bit NPU instruction.

    **R-type** (``is_itype=False``)::

        [31:24]  OP    8-bit opcode
        [23:16]  DST   destination register / SRAM bank
        [15:8]   SRC_A source A
        [7:0]    SRC_B source B

    **GEMM descriptor-ref** (``opcode`` is GEMM/GEMM_SCALE)::

        [31:24]  OP
        [23:8]   descriptor word offset from CSR_DESC_PTR
        [7:0]    auxiliary / legacy K-count field

    **I-type** (``is_itype=True``)::

        [31:28]  OP[3:0]  lower nibble of opcode
        [27:20]  OPT      sub-opcode / option
        [19:0]   IMM      20-bit immediate
    """
    opcode: Opcode = Opcode.NOP
    desc_ptr: int = 0       # descriptor table index (metadata, **not** encoded
                            # in the 32-bit word — stored in the binary
                            # descriptor section)
    dst: int = 0
    src_a: int = 0
    src_b: int = 0
    imm: int = 0            # immediate (I-type only)
    opt: int = 0            # sub-opcode / option (I-type only)
    is_itype: bool = False

    def encode(self) -> bytes:
        """Pack instruction into 4-byte little-endian word."""
        if self.is_itype:
            word = (
                ((self.opcode & 0xF) << 28)
                | ((self.opt & 0xFF) << 20)
                | (self.imm & 0xFFFFF)
            )
        elif self.opcode in (Opcode.GEMM, Opcode.GEMM_SCALE):
            word = (
                ((self.opcode & 0xFF) << 24)
                | ((self.desc_ptr & 0xFFFF) << 8)
                | (self.src_b & 0xFF)
            )
        else:
            word = (
                ((self.opcode & 0xFF) << 24)
                | ((self.dst & 0xFF) << 16)
                | ((self.src_a & 0xFF) << 8)
                | (self.src_b & 0xFF)
            )
        return struct.pack("<I", word & 0xFFFFFFFF)


# ── GEMM descriptor ──────────────────────────────────────────────────────

GEMM_DESCRIPTOR_SIZE = 19


def build_gemm_descriptor(
    m: int = 1,
    n: int = 1,
    k: int = 1,
    a_sram_bank: int = 0,
    b_sram_bank: int = 1,
    o_sram_bank: int = 0,
    a_zp: int = 0,
    b_zp: int = 0,
    out_scale_shr: int = 0,
    out_scale_mul: int = 1,
    relu: int = 0,
    out_zp: int = 0,
) -> bytes:
    """Build a 19-byte GEMM descriptor (packed little-endian).

    Layout (19 bytes)::

        [ 0] M              uint16  output-row tile count (x16)
        [ 2] N              uint16
        [ 4] K              uint16
        [ 6] a_sram_bank    uint8
        [ 7] b_sram_bank    uint8
        [ 8] o_sram_bank    uint8
        [ 9] a_zp           uint8   INT8 zero point
        [10] b_zp           uint8
        [11] reserved       uint16
        [13] out_scale_shr  uint16  requant right-shift
        [15] out_scale_mul  int16   requant multiplier (signed)
        [17] relu           uint8   fuse ReLU flag
        [18] out_zp         uint8   output zero point
    """
    return struct.pack(
        "<HHHBBBBBHHhBB",
        m & 0xFFFF,
        n & 0xFFFF,
        k & 0xFFFF,
        a_sram_bank & 0xFF,
        b_sram_bank & 0xFF,
        o_sram_bank & 0xFF,
        a_zp & 0xFF,
        b_zp & 0xFF,
        0,                      # reserved
        out_scale_shr & 0xFFFF,
        out_scale_mul,
        relu & 0xFF,
        out_zp & 0xFF,
    )


# ── NpuGraph ─────────────────────────────────────────────────────────────

@dataclass
class NpuGraph:
    """Compiled NPU program.

    Attributes:
        instructions: Ordered list of ``NpuInstruction``.
        descriptors:  Ordered list of 19-byte GEMM descriptor blobs.
    """
    instructions: list[NpuInstruction] = field(default_factory=list)
    descriptors: list[bytes] = field(default_factory=list)


# ── Binary serialization ─────────────────────────────────────────────────

def serialize_to_binary(graph: NpuGraph, path: str | Path) -> None:
    """Write an ``NpuGraph`` to a ``.npu`` binary file.

    The file begins with the ``NPUC`` magic header, followed by a version
    word and counts, then the instruction stream and descriptor table.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as fh:
        # Header
        fh.write(MAGIC)
        fh.write(struct.pack("<I", VERSION))
        fh.write(struct.pack("<I", len(graph.instructions)))
        fh.write(struct.pack("<I", len(graph.descriptors)))

        # Instruction stream
        for instr in graph.instructions:
            fh.write(instr.encode())

        # Descriptor table
        for desc in graph.descriptors:
            fh.write(desc)
