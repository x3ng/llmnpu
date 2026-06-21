"""test_dma.py — Cocotb test for NPU DMA (1D mode).

Test scenarios:
  1. DMA LOAD: write pattern to simulated ExtMem, LOAD into SRAM, verify
  2. DMA STORE: write pattern to SRAM, STORE to ExtMem, verify
  3. Short 1D LOAD (single 8-byte word)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import random

# Opcodes matching isa_defines.svh
OP_DMA_LD = 0x40
OP_DMA_ST = 0x41
OP_DMA_2D = 0x42


async def reset_dut(dut):
    """Apply synchronous reset."""
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


# ---------------------------------------------------------------------------
# Helper: write a 64-bit word to external memory via the simulator debug port
# ---------------------------------------------------------------------------
async def ext_write(dut, byte_addr, data_64):
    """Write one 64-bit word to external memory via sim debug port."""
    dut.sim_ext_en.value = 1
    dut.sim_ext_we.value = 1
    dut.sim_ext_addr.value = byte_addr
    dut.sim_ext_wdata.value = data_64
    await RisingEdge(dut.clk)
    dut.sim_ext_en.value = 0
    dut.sim_ext_we.value = 0


# ---------------------------------------------------------------------------
# Helper: read a 64-bit word from external memory via the simulator debug port
# ---------------------------------------------------------------------------
async def ext_read(dut, byte_addr):
    """Read one 64-bit word from external memory via sim debug port.
    Returns int for easy comparison.
    Note: takes 2 clock edges for the registered debug read to settle."""
    dut.sim_ext_en.value = 1
    dut.sim_ext_we.value = 0
    dut.sim_ext_addr.value = byte_addr
    await RisingEdge(dut.clk)  # NBA: sim_rdata_reg <= ext_mem[addr]
    await RisingEdge(dut.clk)  # NBA settled
    val = int(dut.sim_ext_rdata.value)
    dut.sim_ext_en.value = 0
    return val


# ---------------------------------------------------------------------------
# Helper: write a 64-bit word to SRAM via the simulator debug port
# ---------------------------------------------------------------------------
async def sram_write(dut, byte_addr, data_64):
    """Write one 64-bit word to SRAM via sim debug port."""
    dut.sim_sram_en.value = 1
    dut.sim_sram_we.value = 1
    dut.sim_sram_addr.value = byte_addr
    dut.sim_sram_wdata.value = data_64
    await RisingEdge(dut.clk)
    dut.sim_sram_en.value = 0
    dut.sim_sram_we.value = 0


# ---------------------------------------------------------------------------
# Helper: read a 64-bit word from SRAM via the simulator debug port
# ---------------------------------------------------------------------------
async def sram_read(dut, byte_addr):
    """Read one 64-bit word from SRAM via sim debug port.
    Returns int for easy comparison.
    Note: takes 2 clock edges for the registered debug read to settle."""
    dut.sim_sram_en.value = 1
    dut.sim_sram_we.value = 0
    dut.sim_sram_addr.value = byte_addr
    await RisingEdge(dut.clk)  # NBA: sram_rdata_reg <= sram[addr]
    await RisingEdge(dut.clk)  # NBA settled
    val = int(dut.sim_sram_rdata.value)
    dut.sim_sram_en.value = 0
    return val


# ---------------------------------------------------------------------------
# Helper: start a DMA operation and wait for completion
# ---------------------------------------------------------------------------
async def start_dma(dut, opcode, ext_addr, sram_addr, length,
                    row_count=1, row_bytes=None, ext_stride=None,
                    sram_stride=None):
    """Pulse start with the given parameters, then wait for done.
    Returns once done is asserted (DMA is in DONE state)."""
    dut.start.value = 1
    dut.opcode.value = opcode
    dut.ext_addr.value = ext_addr
    dut.sram_addr.value = sram_addr
    dut.length.value = length
    dut.row_count.value = row_count
    dut.row_bytes.value = length if row_bytes is None else row_bytes
    dut.ext_stride.value = length if ext_stride is None else ext_stride
    dut.sram_stride.value = length if sram_stride is None else sram_stride
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for DMA to complete
    while not bool(dut.done.value):
        await RisingEdge(dut.clk)


# ===========================================================================
# Test 1: 1D DMA LOAD — ExtMem → SRAM
# ===========================================================================
@cocotb.test()
async def test_dma_1d_load(dut):
    """Write 8 words to ExtMem, DMA LOAD them into SRAM, verify."""
    clock = Clock(dut.clk, 2, unit="ns")  # 500 MHz
    cocotb.start_soon(clock.start())

    # Initialise all control signals
    dut.start.value = 0
    dut.opcode.value = 0
    dut.ext_addr.value = 0
    dut.sram_addr.value = 0
    dut.length.value = 0
    dut.row_count.value = 0
    dut.row_bytes.value = 0
    dut.ext_stride.value = 0
    dut.sram_stride.value = 0
    dut.sim_ext_en.value = 0
    dut.sim_ext_we.value = 0
    dut.sim_ext_addr.value = 0
    dut.sim_ext_wdata.value = 0
    dut.sim_sram_en.value = 0
    dut.sim_sram_we.value = 0
    dut.sim_sram_addr.value = 0
    dut.sim_sram_wdata.value = 0

    await reset_dut(dut)

    # Check post-reset state
    assert bool(dut.busy.value) is False, "busy should be 0 after reset"
    assert bool(dut.done.value) is False, "done should be 0 after reset"

    # ------------------------------------------------------------------
    # Write 8 random 64-bit words to external memory at offset 0x100
    # ------------------------------------------------------------------
    num_words = 8
    pattern = [random.randint(0, 2**64 - 1) for _ in range(num_words)]
    base_ext = 0x100  # arbitrary non-zero offset

    for i, val in enumerate(pattern):
        await ext_write(dut, base_ext + i * 8, val)

    # Verify we can read them back (debug read)
    for i, expected in enumerate(pattern):
        actual = await ext_read(dut, base_ext + i * 8)
        assert actual == expected, \
            f"ExtMem verify before DMA: word {i}: expected 0x{expected:016x}, " \
            f"got 0x{actual:016x}"

    # ------------------------------------------------------------------
    # DMA LOAD: base_ext → sram_base=0x200, length = 64 bytes (8x8)
    # ------------------------------------------------------------------
    sram_base = 0x200
    length = num_words * 8  # 64 bytes
    await start_dma(dut, OP_DMA_LD, base_ext, sram_base, length)

    # DMA is in DONE state; deasserting start lets FSM return to IDLE
    assert bool(dut.done.value) is True, "done should be 1 after DMA completes"
    await RisingEdge(dut.clk)
    assert bool(dut.done.value) is False, "done should clear after start deassert"
    assert bool(dut.busy.value) is False, "busy should be 0 after DMA completes"

    # ------------------------------------------------------------------
    # Read back from SRAM and verify
    # ------------------------------------------------------------------
    for i, expected in enumerate(pattern):
        actual = await sram_read(dut, sram_base + i * 8)
        assert actual == expected, \
            f"SRAM verify: word {i}: expected 0x{expected:016x}, " \
            f"got 0x{actual:016x}"

    dut._log.info(
        f"test_dma_1d_load PASS: {num_words} words, "
        f"ext=0x{base_ext:x} -> sram=0x{sram_base:x}, len={length}"
    )


# ===========================================================================
# Test 2: 1D DMA STORE — SRAM -> ExtMem
# ===========================================================================
@cocotb.test()
async def test_dma_1d_store(dut):
    """Write 8 words to SRAM, DMA STORE them into ExtMem, verify."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    dut.start.value = 0
    dut.opcode.value = 0
    dut.ext_addr.value = 0
    dut.sram_addr.value = 0
    dut.length.value = 0
    dut.row_count.value = 0
    dut.row_bytes.value = 0
    dut.ext_stride.value = 0
    dut.sram_stride.value = 0
    dut.sim_ext_en.value = 0
    dut.sim_ext_we.value = 0
    dut.sim_ext_addr.value = 0
    dut.sim_ext_wdata.value = 0
    dut.sim_sram_en.value = 0
    dut.sim_sram_we.value = 0
    dut.sim_sram_addr.value = 0
    dut.sim_sram_wdata.value = 0

    await reset_dut(dut)

    # ------------------------------------------------------------------
    # Write 8 random 64-bit words to SRAM at offset 0x400
    # ------------------------------------------------------------------
    num_words = 8
    pattern = [random.randint(0, 2**64 - 1) for _ in range(num_words)]
    sram_base = 0x400

    for i, val in enumerate(pattern):
        await sram_write(dut, sram_base + i * 8, val)

    # Verify SRAM write
    for i, expected in enumerate(pattern):
        actual = await sram_read(dut, sram_base + i * 8)
        assert actual == expected, \
            f"SRAM verify before DMA: word {i}: expected 0x{expected:016x}, " \
            f"got 0x{actual:016x}"

    # ------------------------------------------------------------------
    # DMA STORE: sram_base=0x400 -> ext_base=0x600, length = 64 bytes
    # ------------------------------------------------------------------
    ext_base = 0x600
    length = num_words * 8  # 64 bytes
    await start_dma(dut, OP_DMA_ST, ext_base, sram_base, length)

    # ------------------------------------------------------------------
    # Read back from ExtMem and verify
    # ------------------------------------------------------------------
    for i, expected in enumerate(pattern):
        actual = await ext_read(dut, ext_base + i * 8)
        assert actual == expected, \
            f"ExtMem verify: word {i}: expected 0x{expected:016x}, " \
            f"got 0x{actual:016x}"

    dut._log.info(
        f"test_dma_1d_store PASS: {num_words} words, "
        f"sram=0x{sram_base:x} -> ext=0x{ext_base:x}, len={length}"
    )


# ===========================================================================
# Test 3: Short 1D LOAD (single word = 8 bytes)
# ===========================================================================
@cocotb.test()
async def test_dma_1d_short(dut):
    """Single 8-byte DMA LOAD -- minimum transfer unit."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    dut.start.value = 0
    dut.opcode.value = 0
    dut.ext_addr.value = 0
    dut.sram_addr.value = 0
    dut.length.value = 0
    dut.row_count.value = 0
    dut.row_bytes.value = 0
    dut.ext_stride.value = 0
    dut.sram_stride.value = 0
    dut.sim_ext_en.value = 0
    dut.sim_ext_we.value = 0
    dut.sim_ext_addr.value = 0
    dut.sim_ext_wdata.value = 0
    dut.sim_sram_en.value = 0
    dut.sim_sram_we.value = 0
    dut.sim_sram_addr.value = 0
    dut.sim_sram_wdata.value = 0

    await reset_dut(dut)

    # Write one word to ExtMem
    expected = random.randint(0, 2**64 - 1)
    await ext_write(dut, 0x000, expected)

    # DMA LOAD single word
    await start_dma(dut, OP_DMA_LD, 0x000, 0x100, 8)

    # Verify
    actual = await sram_read(dut, 0x100)
    assert actual == expected, \
        f"Short DMA: expected 0x{expected:016x}, got 0x{actual:016x}"

    dut._log.info(f"test_dma_1d_short PASS: single 8-byte transfer")


@cocotb.test()
async def test_dma_2d_load_strided(dut):
    """2D DMA LOAD copies row_bytes per row with independent strides."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    dut.start.value = 0
    dut.opcode.value = 0
    dut.ext_addr.value = 0
    dut.sram_addr.value = 0
    dut.length.value = 0
    dut.row_count.value = 0
    dut.row_bytes.value = 0
    dut.ext_stride.value = 0
    dut.sram_stride.value = 0
    dut.sim_ext_en.value = 0
    dut.sim_ext_we.value = 0
    dut.sim_ext_addr.value = 0
    dut.sim_ext_wdata.value = 0
    dut.sim_sram_en.value = 0
    dut.sim_sram_we.value = 0
    dut.sim_sram_addr.value = 0
    dut.sim_sram_wdata.value = 0

    await reset_dut(dut)

    rows = 3
    row_bytes = 16
    ext_stride = 32
    sram_stride = 24
    ext_base = 0x800
    sram_base = 0x900
    pattern = [
        [0x11111111_00000000, 0x11111111_00000001],
        [0x22222222_00000000, 0x22222222_00000001],
        [0x33333333_00000000, 0x33333333_00000001],
    ]

    for r in range(rows):
        for w, val in enumerate(pattern[r]):
            await ext_write(dut, ext_base + r * ext_stride + w * 8, val)

    gap_sentinel = 0xA5A5A5A5_5A5A5A5A
    await sram_write(dut, sram_base + row_bytes, gap_sentinel)

    await start_dma(
        dut, OP_DMA_2D, ext_base, sram_base, row_bytes,
        row_count=rows,
        row_bytes=row_bytes,
        ext_stride=ext_stride,
        sram_stride=sram_stride,
    )

    for r in range(rows):
        for w, expected in enumerate(pattern[r]):
            actual = await sram_read(dut, sram_base + r * sram_stride + w * 8)
            assert actual == expected, (
                f"2D SRAM row {r} word {w}: expected 0x{expected:016x}, "
                f"got 0x{actual:016x}"
            )

    # Verify stride gaps were not filled by linear copying.
    gap = await sram_read(dut, sram_base + row_bytes)
    assert gap == gap_sentinel, "2D load overwrote the first SRAM stride gap"

    dut._log.info(
        f"test_dma_2d_load_strided PASS: rows={rows}, row_bytes={row_bytes}, "
        f"ext_stride={ext_stride}, sram_stride={sram_stride}"
    )
