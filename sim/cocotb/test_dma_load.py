"""test_dma_load.py — Minimal DMA LOAD test.

Verifies that DMA can read known data from ext_mem and load it into
internal SRAM.  Drives npu_dma directly (standalone TOPLEVEL).

Sequence:
  1. Write a known 64-bit pattern to ext_mem via the wrapper sim port
  2. Trigger DMA LOAD (OP_DMA_LD) to copy ext_mem → SRAM
  3. Read back SRAM via sim debug port
  4. Assert data matches
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

# Opcode matching isa_defines.svh
OP_DMA_LD = 0x40


async def reset_dut(dut):
    """Apply synchronous reset."""
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def ext_write64(dut, byte_addr, data_64):
    """Write one 64-bit word to wrapper ext_mem via sim_ext port."""
    dut.sim_ext_en.value = 1
    dut.sim_ext_we.value = 1
    dut.sim_ext_addr.value = byte_addr
    dut.sim_ext_wdata.value = data_64
    await RisingEdge(dut.clk)
    dut.sim_ext_en.value = 0
    dut.sim_ext_we.value = 0


async def sram_read64(dut, byte_addr):
    """Read one 64-bit word from SRAM via sim debug port. Returns int.
    Note: 2 clock edges needed for registered debug read to settle."""
    dut.sim_sram_en.value = 1
    dut.sim_sram_we.value = 0
    dut.sim_sram_addr.value = byte_addr
    await RisingEdge(dut.clk)  # NBA: sram_rdata_reg <= sram[addr]
    await RisingEdge(dut.clk)  # NBA settled
    val = int(dut.sim_sram_rdata.value)
    dut.sim_sram_en.value = 0
    return val


async def start_dma(dut, opcode, ext_addr, sram_addr, length):
    """Pulse start with parameters, then wait for done."""
    dut.start.value = 1
    dut.opcode.value = opcode
    dut.ext_addr.value = ext_addr
    dut.sram_addr.value = sram_addr
    dut.length.value = length
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # Wait for DMA to reach DONE state
    while not bool(dut.done.value):
        await RisingEdge(dut.clk)


# ===========================================================================
# Test: Single-word DMA LOAD
# ===========================================================================
@cocotb.test()
async def test_dma_load_single(dut):
    """Load a single 8-byte word from ext_mem into SRAM."""

    clock = Clock(dut.clk, 2, unit="ns")  # 500 MHz
    cocotb.start_soon(clock.start())

    # Initialise all control signals
    dut.start.value = 0
    dut.opcode.value = 0
    dut.ext_addr.value = 0
    dut.sram_addr.value = 0
    dut.length.value = 0
    dut.sim_ext_en.value = 0
    dut.sim_ext_we.value = 0
    dut.sim_ext_addr.value = 0
    dut.sim_ext_wdata.value = 0
    dut.sim_sram_en.value = 0
    dut.sim_sram_we.value = 0
    dut.sim_sram_addr.value = 0
    dut.sim_sram_wdata.value = 0

    await reset_dut(dut)

    # Post-reset checks
    assert bool(dut.busy.value) is False, "busy should be 0 after reset"
    assert bool(dut.done.value) is False, "done should be 0 after reset"

    # Write known pattern to ext_mem
    expected = 0xDEADBEEF_CAFEBABE
    ext_offset = 0x100
    sram_offset = 0x200
    await ext_write64(dut, ext_offset, expected)

    # DMA LOAD: ext_offset → sram_offset, length = 8 bytes
    await start_dma(dut, OP_DMA_LD, ext_offset, sram_offset, 8)

    # Verify DMA completion flags
    assert bool(dut.done.value) is True, "done should be 1 after DMA completes"
    await RisingEdge(dut.clk)
    assert bool(dut.done.value) is False, "done should clear after start deassert"
    assert bool(dut.busy.value) is False, "busy should be 0 after DMA completes"

    # Read back from SRAM and verify
    actual = await sram_read64(dut, sram_offset)
    assert actual == expected, \
        f"DMA LOAD mismatch: expected 0x{expected:016X}, got 0x{actual:016X}"

    dut._log.info(f"test_dma_load_single PASS: "
                  f"ext=0x{ext_offset:X} -> sram=0x{sram_offset:X}")


# ===========================================================================
# Test: Multi-word DMA LOAD
# ===========================================================================
@cocotb.test()
async def test_dma_load_multi(dut):
    """Load 4 × 8-byte words (32 bytes) from ext_mem into SRAM."""

    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    dut.start.value = 0
    dut.opcode.value = 0
    dut.ext_addr.value = 0
    dut.sram_addr.value = 0
    dut.length.value = 0
    dut.sim_ext_en.value = 0
    dut.sim_ext_we.value = 0
    dut.sim_ext_addr.value = 0
    dut.sim_ext_wdata.value = 0
    dut.sim_sram_en.value = 0
    dut.sim_sram_we.value = 0
    dut.sim_sram_addr.value = 0
    dut.sim_sram_wdata.value = 0

    await reset_dut(dut)

    # Write 4 known words to ext_mem
    num_words = 4
    pattern = [
        0x00000001_00000000,
        0x00000003_00000002,
        0x00000005_00000004,
        0x00000007_00000006,
    ]
    ext_offset = 0x300
    sram_offset = 0x400

    for i, val in enumerate(pattern):
        await ext_write64(dut, ext_offset + i * 8, val)

    # DMA LOAD: ext_offset → sram_offset, length = 32 bytes
    await start_dma(dut, OP_DMA_LD, ext_offset, sram_offset, num_words * 8)

    # Verify DMA completion
    assert bool(dut.done.value) is True, "done should be 1"
    await RisingEdge(dut.clk)
    assert bool(dut.busy.value) is False, "busy should be 0"

    # Verify all words in SRAM
    for i, expected in enumerate(pattern):
        actual = await sram_read64(dut, sram_offset + i * 8)
        assert actual == expected, \
            f"DMA LOAD mismatch at word {i}: " \
            f"expected 0x{expected:016X}, got 0x{actual:016X}"

    dut._log.info(f"test_dma_load_multi PASS: {num_words} words, "
                  f"ext=0x{ext_offset:X} -> sram=0x{sram_offset:X}")
