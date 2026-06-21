"""Tests for the NPU codegen toolchain: quantizer, compiler, serializer."""

from __future__ import annotations

import os
import struct
import sys
import tempfile

import pytest

# Make the tools package importable from the test directory.
_srcdir = os.path.join(os.path.dirname(__file__), "..", "tools")
sys.path.insert(0, os.path.abspath(_srcdir))

from codegen.serialize import (          # noqa: E402
    MAGIC,
    NpuGraph,
    NpuInstruction,
    Opcode,
    VOpt,
    GEMM_DESCRIPTOR_SIZE,
    GEMM_DESCRIPTOR_SLOT_SIZE,
    build_gemm_descriptor,
    serialize_to_binary,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Quantizer
# ═══════════════════════════════════════════════════════════════════════════

class TestQuantize:
    """Per-tensor & per-channel symmetric INT8 quantisation."""

    def test_per_tensor_basic(self):
        torch = pytest.importorskip("torch")
        from codegen.quantize import quantize_per_tensor

        x = torch.tensor([-1.0, 0.0, 1.0, 0.5])
        x_int, params = quantize_per_tensor(x)

        assert x_int.dtype == torch.int8
        assert params.scale > 0.0
        assert params.zero_point == 0

        # Round-trip approximate
        x_hat = x_int.float() * params.scale
        assert torch.allclose(x_hat, x, atol=float(params.scale))

    def test_per_tensor_all_zeros(self):
        torch = pytest.importorskip("torch")
        from codegen.quantize import quantize_per_tensor

        x = torch.zeros(4)
        x_int, params = quantize_per_tensor(x)

        assert params.scale > 0.0
        assert x_int.sum().item() == 0

    def test_per_tensor_single_element(self):
        torch = pytest.importorskip("torch")
        from codegen.quantize import quantize_per_tensor

        x = torch.tensor([3.14])
        x_int, params = quantize_per_tensor(x)

        assert x_int.numel() == 1
        assert params.scale > 0.0

    def test_per_channel_basic(self):
        torch = pytest.importorskip("torch")
        from codegen.quantize import quantize_per_channel

        w = torch.randn(8, 16)
        w_int, params_list = quantize_per_channel(w)

        assert w_int.shape == w.shape
        assert w_int.dtype == torch.int8
        assert len(params_list) == 8
        for p in params_list:
            assert p.scale > 0.0
            assert p.zero_point == 0

    def test_per_channel_zeros_channel(self):
        torch = pytest.importorskip("torch")
        from codegen.quantize import quantize_per_channel

        w = torch.tensor([[1.0, -2.0], [0.0, 0.0]])
        w_int, params_list = quantize_per_channel(w)

        assert w_int.shape == w.shape
        assert len(params_list) == 2
        # Second channel is all zeros — scale must still be positive.
        assert params_list[1].scale > 0.0
        assert w_int[1].sum().item() == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Serializer
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialize:
    """Instruction encoding, descriptor packing, binary I/O."""

    # -- opcode values ----------------------------------------------------
    def test_opcode_values(self):
        assert Opcode.GEMM == 0x01
        assert Opcode.GEMM_SCALE == 0x02
        assert Opcode.ACT_RELU == 0x20
        assert Opcode.ACT_GELU == 0x21
        assert Opcode.ACT_SIGMOID == 0x22
        assert Opcode.ACT_TANH == 0x23
        assert Opcode.QUANT == 0x30
        assert Opcode.DEQUANT == 0x31
        assert Opcode.DMA_LD == 0x40
        assert Opcode.DMA_ST == 0x41
        assert Opcode.DMA_2D == 0x42
        assert Opcode.SYNC == 0xF0
        assert Opcode.WFI == 0xF1
        assert Opcode.NOP == 0xFF

    def test_vopt_values(self):
        assert VOpt.ADD == 0x00
        assert VOpt.MUL == 0x02

    # -- Instruction encoding --------------------------------------------
    def test_encode_gemm_descriptor_ref(self):
        instr = NpuInstruction(
            opcode=Opcode.GEMM, desc_ptr=5, src_b=16, is_itype=False,
        )
        data = instr.encode()
        assert len(data) == 4
        word = struct.unpack("<I", data)[0]
        assert (word >> 24) & 0xFF == Opcode.GEMM
        assert (word >> 8) & 0xFFFF == 5
        assert word & 0xFF == 16

    def test_encode_rtype(self):
        instr = NpuInstruction(
            opcode=Opcode.ACT_RELU, dst=1, src_a=2, src_b=3, is_itype=False,
        )
        data = instr.encode()
        assert len(data) == 4
        word = struct.unpack("<I", data)[0]
        assert (word >> 24) & 0xFF == Opcode.ACT_RELU
        assert (word >> 16) & 0xFF == 1
        assert (word >> 8) & 0xFF == 2
        assert word & 0xFF == 3

    def test_encode_rtype_default(self):
        """is_itype defaults to False → R-type."""
        instr = NpuInstruction(opcode=Opcode.NOP)
        data = instr.encode()
        word = struct.unpack("<I", data)[0]
        assert (word >> 24) & 0xFF == Opcode.NOP

    # -- I-type encoding --------------------------------------------------
    def test_encode_itype(self):
        instr = NpuInstruction(
            opcode=Opcode.VADD, opt=VOpt.ADD, imm=42, is_itype=True,
        )
        data = instr.encode()
        assert len(data) == 4
        word = struct.unpack("<I", data)[0]
        assert (word >> 28) & 0xF == (Opcode.VADD & 0xF)
        assert (word >> 20) & 0xFF == VOpt.ADD
        assert word & 0xFFFFF == 42

    def test_encode_itype_imm_mask(self):
        """20-bit immediate is correctly masked."""
        instr = NpuInstruction(
            opcode=Opcode.DMA_LD, opt=0, imm=0xABCDE, is_itype=True,
        )
        word = struct.unpack("<I", instr.encode())[0]
        assert word & 0xFFFFF == 0xABCDE  # lower 20 bits preserved

    # -- GEMM descriptor --------------------------------------------------
    def test_build_gemm_descriptor_size(self):
        desc = build_gemm_descriptor()
        assert len(desc) == GEMM_DESCRIPTOR_SIZE

    def test_build_gemm_descriptor_fields(self):
        desc = build_gemm_descriptor(
            m=2, n=3, k=4,
            a_sram_bank=1, b_sram_bank=2, o_sram_bank=3,
            a_zp=10, b_zp=20,
            out_scale_shr=4, out_scale_mul=100,
            relu=1, out_zp=5,
        )
        m, n, k = struct.unpack_from("<HHH", desc, 0)
        assert m == 2
        assert n == 3
        assert k == 4

        a_bank, b_bank, o_bank = struct.unpack_from("<BBB", desc, 6)
        assert a_bank == 1
        assert b_bank == 2
        assert o_bank == 3

        a_zp, b_zp = struct.unpack_from("<BB", desc, 9)
        assert a_zp == 10
        assert b_zp == 20

        shr, mul = struct.unpack_from("<Hh", desc, 13)
        assert shr == 4
        assert mul == 100

        relu, out_zp = struct.unpack_from("<BB", desc, 17)
        assert relu == 1
        assert out_zp == 5

    # -- binary serialization ---------------------------------------------
    def test_serialize_empty(self):
        graph = NpuGraph()
        with tempfile.NamedTemporaryFile(suffix=".npu", delete=False) as f:
            path = f.name
        try:
            serialize_to_binary(graph, path)
            with open(path, "rb") as fh:
                data = fh.read()
            assert data[:4] == MAGIC
            ver, ni, nd = struct.unpack("<III", data[4:16])
            assert ver == 1
            assert ni == 0
            assert nd == 0
            assert len(data) == 16       # header only
        finally:
            os.unlink(path)

    def test_serialize_with_instructions(self):
        graph = NpuGraph(
            instructions=[
                NpuInstruction(opcode=Opcode.GEMM, dst=0, src_a=0, src_b=1),
                NpuInstruction(opcode=Opcode.ACT_RELU, dst=0, src_a=0),
            ],
            descriptors=[
                build_gemm_descriptor(m=1, n=1, k=1),
            ],
        )
        with tempfile.NamedTemporaryFile(suffix=".npu", delete=False) as f:
            path = f.name
        try:
            serialize_to_binary(graph, path)
            with open(path, "rb") as fh:
                data = fh.read()
            assert data[:4] == MAGIC
            ver, ni, nd = struct.unpack("<III", data[4:16])
            assert ver == 1
            assert ni == 2
            assert nd == 1
            expected_size = 16 + 2 * 4 + 1 * GEMM_DESCRIPTOR_SLOT_SIZE
            assert len(data) == expected_size
        finally:
            os.unlink(path)

    def test_serialize_creates_directory(self):
        graph = NpuGraph()
        tmpdir = tempfile.mkdtemp()
        nested = os.path.join(tmpdir, "a", "b", "test.npu")
        try:
            serialize_to_binary(graph, nested)
            assert os.path.isfile(nested)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Compiler
# ═══════════════════════════════════════════════════════════════════════════

class TestCompiler:
    """FX graph → NpuGraph lowering."""

    def test_compile_linear_relu(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from codegen.compiler import NpuCompiler

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)
            def forward(self, x):
                return torch.relu(self.linear(x))

        gm = torch.fx.symbolic_trace(Model())
        graph = NpuCompiler().compile(gm)

        assert len(graph.instructions) == 4
        assert graph.instructions[0].opcode == Opcode.GEMM
        assert graph.instructions[1].opcode == Opcode.SYNC
        assert graph.instructions[2].opcode == Opcode.ACT_RELU
        assert graph.instructions[3].opcode == Opcode.WFI
        assert len(graph.descriptors) == 1
        assert len(graph.descriptors[0]) == GEMM_DESCRIPTOR_SIZE
        m, n, k = struct.unpack_from("<HHH", graph.descriptors[0], 0)
        assert (m, n, k) == (1, 1, 1)

    def test_compile_linear_gelu(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from codegen.compiler import NpuCompiler

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 8)
            def forward(self, x):
                return nn.functional.gelu(self.linear(x))

        gm = torch.fx.symbolic_trace(Model())
        graph = NpuCompiler().compile(gm)

        assert len(graph.instructions) == 4
        assert graph.instructions[0].opcode == Opcode.GEMM
        assert graph.instructions[1].opcode == Opcode.SYNC
        assert graph.instructions[2].opcode == Opcode.ACT_GELU
        assert graph.instructions[3].opcode == Opcode.WFI
        m, n, k = struct.unpack_from("<HHH", graph.descriptors[0], 0)
        assert (m, n, k) == (1, 1, 1)

    def test_compile_add(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from codegen.compiler import NpuCompiler

        class Model(nn.Module):
            def forward(self, x, y):
                return x + y

        gm = torch.fx.symbolic_trace(Model())
        graph = NpuCompiler().compile(gm)

        assert len(graph.instructions) == 2
        instr = graph.instructions[0]
        assert instr.opcode == Opcode.VADD
        assert instr.is_itype is True
        assert instr.opt == VOpt.ADD
        assert graph.instructions[1].opcode == Opcode.WFI

    def test_compile_mul(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from codegen.compiler import NpuCompiler

        class Model(nn.Module):
            def forward(self, x, y):
                return x * y

        gm = torch.fx.symbolic_trace(Model())
        graph = NpuCompiler().compile(gm)

        assert len(graph.instructions) == 2
        instr = graph.instructions[0]
        assert instr.opcode == Opcode.VMUL
        assert instr.opt == VOpt.MUL
        assert graph.instructions[1].opcode == Opcode.WFI

    def test_compile_sigmoid(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from codegen.compiler import NpuCompiler

        class Model(nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)

        gm = torch.fx.symbolic_trace(Model())
        graph = NpuCompiler().compile(gm)

        assert len(graph.instructions) == 2
        assert graph.instructions[0].opcode == Opcode.ACT_SIGMOID
        assert graph.instructions[1].opcode == Opcode.WFI

    def test_compile_tanh(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from codegen.compiler import NpuCompiler

        class Model(nn.Module):
            def forward(self, x):
                return torch.tanh(x)

        gm = torch.fx.symbolic_trace(Model())
        graph = NpuCompiler().compile(gm)

        assert len(graph.instructions) == 2
        assert graph.instructions[0].opcode == Opcode.ACT_TANH
        assert graph.instructions[1].opcode == Opcode.WFI

    # -- end-to-end binary round-trip -------------------------------------
    def test_end_to_end_binary(self):
        """Compile Linear(16,16)+ReLU → write .npu → verify magic + counts."""
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from codegen.compiler import NpuCompiler

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)
            def forward(self, x):
                return torch.relu(self.linear(x))

        gm = torch.fx.symbolic_trace(Model())
        graph = NpuCompiler().compile(gm)

        with tempfile.NamedTemporaryFile(suffix=".npu", delete=False) as f:
            path = f.name
        try:
            serialize_to_binary(graph, path)
            with open(path, "rb") as fh:
                data = fh.read()
            assert data[:4] == MAGIC
            ver, ni, nd = struct.unpack("<III", data[4:16])
            assert ver == 1
            assert ni == 4       # GEMM + SYNC + ACT_RELU + WFI
            assert nd == 1       # one GEMM descriptor
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
#  PyTorch-like public entry point
# ═══════════════════════════════════════════════════════════════════════════

class TestNpuTorch:
    """Minimal PyTorch-facing compile_model API."""

    def test_compile_model_writes_linear_relu_npu(self):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from codegen.npu_torch import compile_model

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(16, 16)
            def forward(self, x):
                return torch.relu(self.linear(x))

        with tempfile.NamedTemporaryFile(suffix=".npu", delete=False) as f:
            path = f.name
        try:
            result = compile_model(Model(), torch.randn(1, 16), path)

            assert os.fspath(result.out_path) == path
            assert result.input_shape == (1, 16)
            assert result.num_instructions == 4
            assert result.num_descriptors == 1
            assert result.graph.instructions[0].opcode == Opcode.GEMM
            assert result.graph.instructions[1].opcode == Opcode.SYNC
            assert result.graph.instructions[2].opcode == Opcode.ACT_RELU
            assert result.graph.instructions[3].opcode == Opcode.WFI

            with open(path, "rb") as fh:
                data = fh.read()
            assert data[:4] == MAGIC
            ver, ni, nd = struct.unpack("<III", data[4:16])
            assert ver == 1
            assert ni == 4
            assert nd == 1
            assert len(data) == 16 + 4 * 4 + 1 * GEMM_DESCRIPTOR_SLOT_SIZE
        finally:
            os.unlink(path)
