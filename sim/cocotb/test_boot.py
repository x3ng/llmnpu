"""test_boot.py — Minimal SoC boot diagnostics

Quick test: load minimal firmware that writes 'P' to UART.
Run with:
  make SIM_BUILD=sim_build_boot PYTHONPATH=$(PWD)/cocotb \
    MODULE=test_boot TOPLEVEL=top_soc \
    VERILOG_SOURCES="...all RTL..."
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles


@cocotb.test()
async def test_boot(dut):
    """Minimal boot: wait for UART 'P' or timeout at 50K cycles."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    for cyc in range(50000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        val = int(dut.uart_tx.value)
        if val != 0:
            dut._log.info(f"UART_TX = 0x{val:02X} ('{chr(val)}') at cycle {cyc}")
            assert chr(val) == 'P', f"Expected 'P', got '{chr(val)}'"
            dut._log.info("PASS: SoC boots and UART works")
            return

    dut._log.error("FAIL: No UART output within 50K cycles")
    assert False, "SoC boot timeout"
