"""test_e2e_csr.py — E2E Stage 2: NPU CSR Read/Write Test

Loads the full SoC (top_soc) with firmware_csr.hex that writes/reads
the NPU CSR registers and outputs 'P' (pass) or 'F' (fail) on UART.

Tests:
  1. test_e2e_csr — correct firmware → expect UART_TX = 'P'
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles


# ── Helpers ──────────────────────────────────────────────────────────────


async def wait_uart_char(dut, timeout_cycles=500000, log=None):
    """Wait for the first non-zero UART character.

    Returns the character ordinal or None on timeout.
    """
    last_val = int(dut.uart_tx.value)
    for cyc in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        val = int(dut.uart_tx.value)
        if val != last_val and val != 0:
            if log:
                log.info(f"UART_TX = 0x{val:02X} ('{chr(val)}') "
                         f"at cycle {cyc}")
            return val
        last_val = val
    return None


# ── Test ─────────────────────────────────────────────────────────────────


@cocotb.test()
async def test_e2e_csr(dut):
    """E2E Stage 2: CSR register read/write.

    Firmware writes CTRL, STATUS, reads PC, writes DESC_PTR —
    expects 'P' on UART if all checks pass.
    """

    # ── Clock ─────────────────────────────────────────────────────────
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # ── Initial reset ─────────────────────────────────────────────────
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)

    # ── Release reset — PicoRV32 starts executing firmware ────────────
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    # ── Wait for UART character ───────────────────────────────────────
    char = await wait_uart_char(dut, timeout_cycles=500000,
                                log=dut._log)

    assert char is not None, (
        "Timeout: no UART character received within 500 kcycles"
    )
    assert chr(char) == 'P', (
        f"UART_TX = 0x{char:02X} ('{chr(char)}'), expected 'P'"
    )
    dut._log.info("PASS: E2E Stage 2 (CSR) — UART_TX = 'P'")
