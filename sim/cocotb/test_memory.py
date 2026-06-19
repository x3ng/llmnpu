"""test_memory.py — Cocotb test for the NPU Memory System.

Tests:
  1. Single master (DMA) write to A-SRAM, read back verification.
  2. Two masters (DMA + GEMM) request the same slave — verify DMA wins.
  3. Three masters write different slaves concurrently — verify no cross-talk.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge
import random


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

async def reset_dut(dut):
    """Apply synchronous active-low reset."""
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def master_write(dut, master, addr, data):
    """Issue a single-cycle write via *master* (0=DMA, 1=GEMM, 2=VALU)."""
    sigs = [(dut, "m0"), (dut, "m1"), (dut, "m2")]
    prefix = sigs[master]
    getattr(prefix[0], f"{prefix[1]}_req").value   = 1
    getattr(prefix[0], f"{prefix[1]}_wen").value   = 1
    getattr(prefix[0], f"{prefix[1]}_addr").value  = addr
    getattr(prefix[0], f"{prefix[1]}_wdata").value = data
    await RisingEdge(dut.clk)
    getattr(prefix[0], f"{prefix[1]}_req").value   = 0
    getattr(prefix[0], f"{prefix[1]}_wen").value   = 0


async def master_read(dut, master, addr):
    """Issue a read via *master* and return the data.

    Icarus fires VPI callbacks *before* NBA updates, so the SRAM's
    registered rdata output still shows the *previous* value immediately
    after a RisingEdge.  We therefore keep req asserted for two cycles:

      Cycle 1 (RisingEdge A): SRAM captures ren, NBA scheduled.
      Gap: NBA applied — SRAM rdata updates, combinational mux settles.
      Cycle 2 (RisingEdge B): SRAM rdata is already valid; sample here.
    """
    sigs = [(dut, "m0"), (dut, "m1"), (dut, "m2")]
    prefix = sigs[master]
    getattr(prefix[0], f"{prefix[1]}_req").value   = 1
    getattr(prefix[0], f"{prefix[1]}_wen").value   = 0
    getattr(prefix[0], f"{prefix[1]}_addr").value  = addr
    await RisingEdge(dut.clk)   # SRAM captures ren; NBA pending
    await RisingEdge(dut.clk)   # NBA applied; rdata valid through mux
    rdata = getattr(prefix[0], f"{prefix[1]}_rdata").value.to_unsigned()
    getattr(prefix[0], f"{prefix[1]}_req").value   = 0
    return rdata


# ---------------------------------------------------------------------------
#  Test 1:  Single master write / read (A-SRAM via DMA)
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_single_write_read(dut):
    """Write a known pattern to A-SRAM via master 0 (DMA), read back, verify."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    ADDR = 0x0008       # A-SRAM base + offset 8
    DATA = 0xA5A5_A5A5

    await master_write(dut, 0, ADDR, DATA)
    assert dut.m0_grant.value == 1, "M0 should be granted during write"

    rd = await master_read(dut, 0, ADDR)
    assert rd == DATA, f"Test 1: expected 0x{DATA:08X}, read 0x{rd:08X}"

    dut._log.info(f"Test 1 PASS — write/read A-SRAM via DMA (0x{ADDR:04X} = 0x{DATA:08X})")


# ---------------------------------------------------------------------------
#  Test 2:  Two masters, same slave — priority arbitration (DMA wins)
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_priority_arbitration(dut):
    """M0 (DMA) and M1 (GEMM) request the same slave simultaneously.

    DMA should be granted; GEMM blocked.  Read-back validates only DMA's write
    took effect.
    """
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    ADDR = 0x0010       # A-SRAM offset 0x10
    DMA_DATA  = 0x1111_1111
    GEMM_DATA = 0x2222_2222

    # Both masters request the same slave in the same cycle
    dut.m0_req.value   = 1
    dut.m0_wen.value   = 1
    dut.m0_addr.value  = ADDR
    dut.m0_wdata.value = DMA_DATA

    dut.m1_req.value   = 1
    dut.m1_wen.value   = 1
    dut.m1_addr.value  = ADDR
    dut.m1_wdata.value = GEMM_DATA

    await RisingEdge(dut.clk)

    # Verify arbitration result
    assert dut.m0_grant.value == 1, "M0 (DMA) should be granted (higher priority)"
    assert dut.m1_grant.value == 0, "M1 (GEMM) should NOT be granted"

    # Deassert
    dut.m0_req.value = 0
    dut.m0_wen.value = 0
    dut.m1_req.value = 0
    dut.m1_wen.value = 0

    # Read back via DMA — should see DMA's data, NOT GEMM's
    rd = await master_read(dut, 0, ADDR)
    assert rd == DMA_DATA, \
        f"Test 2: expected DMA data 0x{DMA_DATA:08X}, read 0x{rd:08X}"

    dut._log.info("Test 2 PASS — DMA wins arbitration over GEMM")


# ---------------------------------------------------------------------------
#  Test 3:  Three masters, different slaves — no cross-talk
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_no_crosstalk(dut):
    """M0->A-SRAM, M1->W-SRAM, M2->O-SRAM concurrently.

    Each bank should only see its own write.  Read-back via each master checks
    for cross-contamination.
    """
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())
    await reset_dut(dut)

    # Each master targets a DIFFERENT slave (upper nibble selects bank)
    ADDR_A = 0x0020       # A-SRAM  (0x0xxx)
    ADDR_W = 0x1020       # W-SRAM  (0x1xxx)
    ADDR_O = 0x2020       # O-SRAM  (0x2xxx)
    # ADDR_D (0x3xxx) left unused for this test

    DATA_A = 0xAAAA_AAAA
    DATA_W = 0xBBBB_BBBB
    DATA_O = 0xCCCC_CCCC

    # All three masters issue a write simultaneously
    dut.m0_req.value   = 1; dut.m0_wen.value   = 1
    dut.m0_addr.value  = ADDR_A; dut.m0_wdata.value = DATA_A

    dut.m1_req.value   = 1; dut.m1_wen.value   = 1
    dut.m1_addr.value  = ADDR_W; dut.m1_wdata.value = DATA_W

    dut.m2_req.value   = 1; dut.m2_wen.value   = 1
    dut.m2_addr.value  = ADDR_O; dut.m2_wdata.value = DATA_O

    await RisingEdge(dut.clk)

    # Every master should be granted (different slaves → no contention)
    assert dut.m0_grant.value == 1, "M0 should be granted (A-SRAM)"
    assert dut.m1_grant.value == 1, "M1 should be granted (W-SRAM)"
    assert dut.m2_grant.value == 1, "M2 should be granted (O-SRAM)"

    dut.m0_req.value = 0; dut.m0_wen.value = 0
    dut.m1_req.value = 0; dut.m1_wen.value = 0
    dut.m2_req.value = 0; dut.m2_wen.value = 0

    # Read back each bank via its respective master
    rd_a = await master_read(dut, 0, ADDR_A)
    rd_w = await master_read(dut, 1, ADDR_W)
    rd_o = await master_read(dut, 2, ADDR_O)

    assert rd_a == DATA_A, \
        f"Test 3 A-SRAM: expected 0x{DATA_A:08X}, read 0x{rd_a:08X}"
    assert rd_w == DATA_W, \
        f"Test 3 W-SRAM: expected 0x{DATA_W:08X}, read 0x{rd_w:08X}"
    assert rd_o == DATA_O, \
        f"Test 3 O-SRAM: expected 0x{DATA_O:08X}, read 0x{rd_o:08X}"

    dut._log.info("Test 3 PASS — no cross-talk between A/W/O SRAM banks")
