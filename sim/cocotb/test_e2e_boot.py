"""test_e2e_boot.py — E2E Stage 1: CPU Boot + UART Test

Firmware is pre-loaded into ext_mem via $readmemh from firmware.hex
(built and placed by the Makefile target).  This test starts the
clock, releases reset, and watches UART TX for 'P' (0x50) within
1000 cycles.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles


@cocotb.test()
async def test_e2e_boot(dut):
    """Stage 1 boot test: check UART outputs 'P'."""

    # ── Clock ─────────────────────────────────────────────────────
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # ── Assert reset for a few cycles ─────────────────────────────
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)

    # ── Release reset — PicoRV32 starts executing at 0x40000000 ──
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    # ── Watch UART_TX for 'P' within 1000 cycles ─────────────────
    for cyc in range(1000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        val = int(dut.uart_tx.value)
        if val != 0:
            dut._log.info(
                f"UART_TX = 0x{val:02X} ('{chr(val)}') "
                f"at cycle {cyc + 1}"
            )
            if val == 0x50:
                dut._log.info("E2E Stage 1 (Boot): PASS")
                return
            else:
                dut._log.error(
                    f"E2E Stage 1 (Boot): FAIL — "
                    f"unexpected char 0x{val:02X} ('{chr(val)}')"
                )
                assert False, (
                    f"Expected 'P' (0x50), got 0x{val:02X}"
                )

    # Timeout
    dut._log.error(
        "E2E Stage 1 (Boot): FAIL — no UART output within 1000 cycles"
    )
    assert False, "Boot timeout: no UART output within 1000 cycles"
