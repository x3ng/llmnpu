"""minimal test — just verify simulation runs without hanging"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer

@cocotb.test()
async def test_minimal(dut):
    clock = Clock(dut.clk, 2, unit="ns")
    await cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.a_in.value = 0
    dut.a_valid.value = 0
    dut.b_in.value = 0
    dut.start.value = 0
    dut.k_count.value = 0

    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    dut._log.info("Reset done, running 10 cycles...")
    await ClockCycles(dut.clk, 10)
    dut._log.info("Minimal test PASS")
