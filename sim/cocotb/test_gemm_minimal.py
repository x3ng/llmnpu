"""Minimal test to isolate cocotb hang."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

@cocotb.test()
async def test_minimal(dut):
    """Just clock a few cycles, check basic outputs."""
    dut._log.info("Starting minimal test...")
    clock = Clock(dut.clk, 2, unit="ns")
    await cocotb.start_soon(clock.start())
    dut._log.info("Clock started")

    dut.rst_n.value = 0
    dut.a_in.value = 0
    dut.a_valid.value = 0
    dut.b_in.value = 0
    dut.start.value = 0
    dut.k_count.value = 0

    dut._log.info("Asserting reset...")
    await ClockCycles(dut.clk, 5)
    dut._log.info("Reset done, 5 cycles elapsed")

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    dut._log.info("Reset deasserted, busy=%s done=%s", dut.busy.value, dut.done.value)

    # Try START
    dut.k_count.value = 1
    dut.start.value = 1
    dut.b_in.value = 0x05050505050505050505050505050505  # all 5s
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    dut._log.info("After LOAD_B edge, busy=%s", dut.busy.value)
    dut.start.value = 0

    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    dut._log.info("After PREFETCH edge, busy=%s", dut.busy.value)

    dut.a_valid.value = 1
    dut.a_in.value = 0x03030303030303030303030303030303  # all 3s
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    dut._log.info("After COMPUTE edge, busy=%s", dut.busy.value)

    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    dut._log.info("After REDUCE edge, busy=%s", dut.busy.value)

    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    dut._log.info("After WRITEBACK edge, busy=%s done=%s psum_valid=%s",
                  dut.busy.value, dut.done.value, dut.psum_valid.value)

    dut._log.info("Minimal test PASSED")
