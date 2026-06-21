"""test_sfu.py — Cocotb test for NPU SFU (Special Function Unit).

Tests the three activation functions (GELU, Sigmoid, Tanh) and the
integer quantise/dequantise datapath.

Each activation function is exercised with random INT8 inputs.  The
hardware output is compared against a golden value computed from the
same mathematical function, quantised to INT8 with the same scale
factor that ``generate_luts.py`` uses.  Tolerance: +/-1 LSB.

Pipeline latency (sfu_top): 4 cycles from valid_in to valid_out
(input register -> interpolation -> alignment -> output mux).
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
import math
import random

# ── Opcode constants (must match isa_defines.svh) ──────────────────────

OP_ACT_GELU    = 0x21
OP_ACT_RELU    = 0x20
OP_ACT_SIGMOID = 0x22
OP_ACT_TANH    = 0x23
OP_ACT_RELU6   = 0x24
OP_ACT_CLIP    = 0x25
OP_QUANT       = 0x30
OP_DEQUANT     = 0x31
OP_NOP         = 0xFF

SFU_LATENCY = 4        # cycles from valid_in to valid_out

# ── Helper ─────────────────────────────────────────────────────────────

async def reset_dut(dut):
    """Apply synchronous reset."""
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


def to_signed(v, bits=8):
    """Convert an ``n``-bit unsigned Python int to signed Python int."""
    if v >= (1 << (bits - 1)):
        return v - (1 << bits)
    return v


def _round_shift_product(product, shift):
    if shift == 0:
        return product
    bias = 1 << (shift - 1)
    if product >= 0:
        return (product + bias) >> shift
    return (product + bias - 1) >> shift


def golden_qd(opcode, x, zp, scale_mul, scale_shr):
    if opcode == OP_DEQUANT:
        raw = (x - zp) * scale_mul
        y = raw >> scale_shr
    else:
        raw = x * scale_mul
        y = _round_shift_product(raw, scale_shr) + zp
    return max(-128, min(127, y))

# ── Activation functions ───────────────────────────────────────────────

def _gelu(x):
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def _tanh(x):
    return math.tanh(x)


def _compute_scale(fn, n=32):
    """Compute the INT8 scale factor used by ``generate_luts.py``."""
    xs = [-4.0 + i * 0.25 for i in range(n)]
    ys = [fn(x) for x in xs]
    max_abs = max(abs(y) for y in ys)
    return 127.0 / max_abs if max_abs > 1e-8 else 1.0


# Pre-compute scale factors (one per activation)
_GELU_SCALE    = _compute_scale(_gelu)
_SIGMOID_SCALE = _compute_scale(_sigmoid)
_TANH_SCALE    = _compute_scale(_tanh)


def _golden(x_int8, fn, scale):
    """Compute the ideal INT8 result for *fn* applied to ``x_int8/32``."""
    y = fn(x_int8 / 32.0)
    q = round(y * scale)
    return max(-128, min(127, q))


def golden_gelu(x_int8):
    return _golden(x_int8, _gelu, _GELU_SCALE)

def golden_sigmoid(x_int8):
    return _golden(x_int8, _sigmoid, _SIGMOID_SCALE)

def golden_tanh(x_int8):
    return _golden(x_int8, _tanh, _TANH_SCALE)


# ── Single test runner (pipelined) ────────────────────────────────────

async def _test_activation(dut, opcode, func_name, golden_fn,
                           num_tests=80):
    """Feed *num_tests* random INT8 values, verify each against *golden_fn*.

    Because the SFU is fully pipelined, we feed one value every cycle
    and compare the output against a software emulation of the pipeline.

    Pipeline: input -> interp -> align -> mux.  Result for input[i]
    appears at y_out while input[i+SFU_LATENCY] is being driven.
    """
    inputs    = [random.randint(-128, 127) for _ in range(num_tests)]
    goldens   = [golden_fn(x) for x in inputs]
    expected_fifo = []
    errors        = 0

    for i, x in enumerate(inputs):
        # ── Drive ──
        dut.opcode.value  = opcode
        dut.x_in.value    = x & 0xFF
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)

        expected_fifo.append(goldens[i])

        # ── Check (only after pipeline has filled) ──
        # Result for input[i-SFU_LATENCY] is visible once
        # SFU_LATENCY+1 entries are in the FIFO.
        if len(expected_fifo) >= SFU_LATENCY + 1:
            expected = expected_fifo.pop(0)
            actual   = dut.y_out.value.to_signed()
            diff     = abs(actual - expected)
            if diff > 1:
                errors += 1
                if errors <= 5:
                    dut._log.warning(
                        f"  [{func_name}] x={inputs[i - SFU_LATENCY]}: "
                        f"exp={expected} got={actual}  (diff={diff})")
            assert diff <= 1, (
                f"{func_name}: x={inputs[i - SFU_LATENCY]}: "
                f"exp={expected} got={actual} (diff > 1 LSB)")

    # ── Drain remaining pipeline entries ──
    # SFU_LATENCY results are still in the pipeline.  Read y_out
    # *after* each clock edge so the output register advances.
    dut.valid_in.value = 0
    for _ in range(SFU_LATENCY):
        expected = expected_fifo.pop(0)
        await RisingEdge(dut.clk)
        actual = dut.y_out.value.to_signed()
        diff   = abs(actual - expected)
        assert diff <= 1, (
            f"{func_name} drain: exp={expected} got={actual} (diff > 1)")

    dut._log.info(
        f"  {func_name}: {num_tests} inputs, "
        f"all within +/-1 LSB")


# ── Tests ──────────────────────────────────────────────────────────────

@cocotb.test()
async def test_sfu_relu(dut):
    """SFU ReLU: negative INT8 values clamp to zero, positive values pass."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    dut.opcode.value  = 0
    dut.x_in.value    = 0
    dut.zp.value      = 0
    dut.scale_mul.value = 0
    dut.scale_shr.value = 0
    dut.valid_in.value = 0
    await reset_dut(dut)

    for x in [-128, -17, -1, 0, 1, 42, 127]:
        dut.opcode.value = OP_ACT_RELU
        dut.x_in.value = x & 0xFF
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0

        valid_seen = False
        for _ in range(SFU_LATENCY + 2):
            await RisingEdge(dut.clk)
            if int(dut.valid_out.value):
                valid_seen = True
                break

        assert valid_seen, f"ReLU x={x}: valid_out did not assert"
        expected = max(0, x)
        actual = dut.y_out.value.to_signed()
        assert actual == expected, (
            f"ReLU x={x}: expected {expected}, got {actual}"
        )

    dut._log.info("  ReLU: all vectors PASS")


@cocotb.test()
async def test_sfu_relu6_clip(dut):
    """SFU ReLU6/Clip clamp signed INT8 inputs to non-negative ranges."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    dut.opcode.value = 0
    dut.x_in.value = 0
    dut.zp.value = 0
    dut.scale_mul.value = 0
    dut.scale_shr.value = 0
    dut.valid_in.value = 0
    await reset_dut(dut)

    vectors = [
        (OP_ACT_RELU6, 0, -8, 0),
        (OP_ACT_RELU6, 0, 0, 0),
        (OP_ACT_RELU6, 0, 5, 5),
        (OP_ACT_RELU6, 0, 6, 6),
        (OP_ACT_RELU6, 0, 7, 6),
        (OP_ACT_RELU6, 0, 127, 6),
        (OP_ACT_CLIP, 10, -3, 0),
        (OP_ACT_CLIP, 10, 0, 0),
        (OP_ACT_CLIP, 10, 7, 7),
        (OP_ACT_CLIP, 10, 10, 10),
        (OP_ACT_CLIP, 10, 42, 10),
    ]

    for opcode, clip_max, x, expected in vectors:
        dut.opcode.value = opcode
        dut.zp.value = clip_max
        dut.x_in.value = x & 0xFF
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        dut.valid_in.value = 0

        valid_seen = False
        for _ in range(SFU_LATENCY + 2):
            await RisingEdge(dut.clk)
            if int(dut.valid_out.value):
                valid_seen = True
                break

        assert valid_seen, f"opcode=0x{opcode:02X} x={x}: valid_out did not assert"
        actual = dut.y_out.value.to_signed()
        assert actual == expected, (
            f"opcode=0x{opcode:02X} x={x} clip={clip_max}: "
            f"expected {expected}, got {actual}"
        )

    dut._log.info("  ReLU6/Clip: all vectors PASS")


@cocotb.test()
async def test_sfu_gelu(dut):
    """SFU GELU: random INT8 inputs vs math-domain golden, +/-1 LSB."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    dut.opcode.value  = 0
    dut.x_in.value    = 0
    dut.zp.value      = 0
    dut.scale_mul.value = 0
    dut.scale_shr.value = 0
    dut.valid_in.value = 0
    await reset_dut(dut)

    await _test_activation(dut, OP_ACT_GELU, "GELU", golden_gelu)


@cocotb.test()
async def test_sfu_sigmoid(dut):
    """SFU Sigmoid: random INT8 inputs vs math-domain golden, +/-1 LSB."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    dut.opcode.value  = 0
    dut.x_in.value    = 0
    dut.zp.value      = 0
    dut.scale_mul.value = 0
    dut.scale_shr.value = 0
    dut.valid_in.value = 0
    await reset_dut(dut)

    await _test_activation(dut, OP_ACT_SIGMOID, "Sigmoid", golden_sigmoid)


@cocotb.test()
async def test_sfu_tanh(dut):
    """SFU Tanh: random INT8 inputs vs math-domain golden, +/-1 LSB."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    dut.opcode.value  = 0
    dut.x_in.value    = 0
    dut.zp.value      = 0
    dut.scale_mul.value = 0
    dut.scale_shr.value = 0
    dut.valid_in.value = 0
    await reset_dut(dut)

    await _test_activation(dut, OP_ACT_TANH, "Tanh", golden_tanh)


@cocotb.test()
async def test_sfu_output_uses_issued_opcode_not_current_opcode(dut):
    """In-flight SFU results use the opcode captured with valid_in."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    dut.valid_in.value = 0
    await reset_dut(dut)

    # Issue QUANT, then change opcode to ReLU while the quant result is
    # still in flight.  The result must still come from quant_dequant.
    dut.opcode.value = OP_QUANT
    dut.x_in.value = 31
    dut.zp.value = 7
    dut.scale_mul.value = 3
    dut.scale_shr.value = 1
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    dut.opcode.value = OP_ACT_RELU
    dut.x_in.value = (-5) & 0xFF
    dut.valid_in.value = 0
    expected_quant = golden_qd(OP_QUANT, 31, 7, 3, 1)

    quant_seen = False
    for _ in range(SFU_LATENCY + 4):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.valid_out.value):
            actual = dut.y_out.value.to_signed()
            assert actual == expected_quant, (
                f"in-flight QUANT selected wrong path: expected "
                f"{expected_quant}, got {actual}"
            )
            quant_seen = True
            break
    assert quant_seen, "in-flight QUANT never produced valid_out"

    # Issue ReLU, then change opcode to QUANT with valid_in low.  The result
    # must still come from the ReLU path.
    dut.opcode.value = OP_ACT_RELU
    dut.x_in.value = (-12) & 0xFF
    dut.valid_in.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    dut.opcode.value = OP_QUANT
    dut.x_in.value = 127
    dut.valid_in.value = 0

    relu_seen = False
    for _ in range(SFU_LATENCY + 4):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.valid_out.value):
            actual = dut.y_out.value.to_signed()
            assert actual == 0, (
                f"in-flight ReLU selected wrong path: expected 0, got {actual}"
            )
            relu_seen = True
            break
    assert relu_seen, "in-flight ReLU never produced valid_out"

    dut._log.info("  output opcode alignment: in-flight results PASS")


@cocotb.test()
async def test_sfu_quant_dequant(dut):
    """SFU quant/dequant: verify identity and constant-scale paths."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    dut.valid_in.value = 0
    await reset_dut(dut)

    vectors = [
        # (opcode, x, zp, scale_mul, scale_shr, description, ref_fn)
        (OP_DEQUANT,  42, 0,   1,   0, "dequant identity",  lambda x, zp, sm, ss: x),
        (OP_DEQUANT,  -5, 0,   1,   0, "dequant neg",       lambda x, zp, sm, ss: x),
        (OP_DEQUANT,  42, 10,  1,   0, "dequant sub zp",    lambda x, zp, sm, ss: x - zp),
        (OP_DEQUANT,  42, 0,   2,   1, "dequant mul>>",     lambda x, zp, sm, ss: (x - zp) * sm >> ss),
        (OP_QUANT,    42, 0,   1,   0, "quant identity",    lambda x, zp, sm, ss: max(-128, min(127, x + zp))),
        (OP_QUANT,   -10, 0,   1,   0, "quant neg",         lambda x, zp, sm, ss: max(-128, min(127, x + zp))),
        (OP_QUANT,    42, 20,  1,   0, "quant add zp",      lambda x, zp, sm, ss: max(-128, min(127, x + zp))),
        (OP_QUANT,   100, 0,   2,   1, "quant scale identity", lambda x, zp, sm, ss: max(-128, min(127, round(x * sm / (1 << ss)) + zp))),
    ]

    for opcode, x, zp, sm, ss, desc, ref_fn in vectors:
        # Drive
        dut.opcode.value    = opcode
        dut.x_in.value      = x & 0xFF
        dut.zp.value        = zp
        dut.scale_mul.value = sm
        dut.scale_shr.value = ss
        dut.valid_in.value  = 1
        await RisingEdge(dut.clk)

        # Expected
        expected = ref_fn(x, zp, sm, ss)
        # Saturate to INT8
        expected = max(-128, min(127, expected))

        # Wait for pipeline (SFU_LATENCY cycles from the drive edge)
        # We already advanced 1 edge above; need SFU_LATENCY-1 more.
        # *Then* read y_out (after the output register has clocked).
        dut.valid_in.value = 0
        for _ in range(SFU_LATENCY):
            await RisingEdge(dut.clk)

        actual = dut.y_out.value.to_signed()
        diff   = abs(actual - expected)
        assert diff <= 1, (
            f"{desc}: x={x} zp={zp} sm={sm} ss={ss}: "
            f"exp={expected} got={actual} (diff={diff})")

    dut._log.info("  quant/dequant: all vectors PASS")


@cocotb.test()
async def test_sfu_quant_dequant_pipelined_controls(dut):
    """Back-to-back Q/DQ ops keep mode/zp/shift aligned with each product."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    dut.valid_in.value = 0
    await reset_dut(dut)

    vectors = [
        (OP_QUANT,    31,  7,  3, 1),
        (OP_DEQUANT, -40,  5,  2, 0),
        (OP_QUANT,   -17, 11,  5, 2),
        (OP_DEQUANT, 100, 30,  3, 1),
        (OP_QUANT,   64,  0, -2, 1),
    ]
    expected_fifo = []

    for opcode, x, zp, sm, ss in vectors:
        if int(dut.valid_out.value) and expected_fifo:
            expected = expected_fifo.pop(0)
            actual = dut.y_out.value.to_signed()
            assert actual == expected, (
                f"pipelined Q/DQ expected {expected}, got {actual}"
            )

        dut.opcode.value = opcode
        dut.x_in.value = x & 0xFF
        dut.zp.value = zp & 0xFF
        dut.scale_mul.value = sm & 0xFFFF
        dut.scale_shr.value = ss
        dut.valid_in.value = 1
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        expected_fifo.append(golden_qd(opcode, x, zp, sm, ss))

    dut.valid_in.value = 0
    for _ in range(SFU_LATENCY + 4):
        if int(dut.valid_out.value) and expected_fifo:
            expected = expected_fifo.pop(0)
            actual = dut.y_out.value.to_signed()
            assert actual == expected, (
                f"pipelined Q/DQ expected {expected}, got {actual}"
            )
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")

    assert not expected_fifo, f"missing pipelined Q/DQ outputs: {expected_fifo}"

    dut._log.info("  quant/dequant pipelined controls PASS")
