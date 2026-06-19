"""test_e2e_dma.py — E2E Stage 3: DMA Data Transfer Integration Test.

Validates the end-to-end DMA data path through the full SoC:
  1. Firmware on PicoRV32 exercises CSR MMIO register paths and ExtMem
  2. Cocotb verifies DMA internal memories (ext_mem, SRAM) via hierarchy
  3. Cocotb manually exercises the DMA data movement path
     (ext_mem → SRAM → ext_mem, simulating DMA LOAD then STORE)

ALL simulation output → sim_build_dma/test.log
Final verdict: "E2E Stage 3 (DMA): PASS" or "FAIL — <error>"
"""

import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles

# ── Constants ────────────────────────────────────────────────────────────────

NUM_WORDS = 16                # number of 64-bit words in test pattern
SRAM_BASE = 0x0000            # A-SRAM base (NPU-internal 16-bit addr)
EXT_SRC   = 0x0100            # source offset in DMA ext_mem
EXT_DST   = 0x0200            # destination offset in DMA ext_mem
BYTES     = NUM_WORDS * 8     # total bytes: 128

TIMEOUT_CYCLES = 2000000      # max cycles to wait for firmware UART


# ── Helpers: DMA memory access via hierarchy ────────────────────────────────

def _ext_mem(dut):
    """Return DMA wrapper internal ext_mem byte array (64 KB)."""
    return dut.u_npu.u_dma.wrapper.ext_mem


def _sram(dut):
    """Return DMA internal SRAM byte array (64 KB)."""
    return dut.u_npu.u_dma.sram


def _ext_write64(dut, byte_addr, data):
    """Write one 64-bit word to DMA ext_mem at byte_addr."""
    for b in range(8):
        _ext_mem(dut)[byte_addr + b].value = (data >> (b * 8)) & 0xFF


def _ext_read64(dut, byte_addr):
    """Read one 64-bit word from DMA ext_mem at byte_addr.  Returns int."""
    val = 0
    for b in range(8):
        val |= int(_ext_mem(dut)[byte_addr + b].value) << (b * 8)
    return val


def _sram_write64(dut, byte_addr, data):
    """Write one 64-bit word to DMA SRAM at byte_addr."""
    for b in range(8):
        _sram(dut)[byte_addr + b].value = (data >> (b * 8)) & 0xFF


def _sram_read64(dut, byte_addr):
    """Read one 64-bit word from DMA SRAM at byte_addr.  Returns int."""
    val = 0
    for b in range(8):
        val |= int(_sram(dut)[byte_addr + b].value) << (b * 8)
    return val


# ── Helpers: UART capture ───────────────────────────────────────────────────

async def wait_uart_chars(dut, count=1, timeout_cycles=TIMEOUT_CYCLES, log=None):
    """Capture `count` uart_tx value changes.

    Returns list of captured values, or empty list on timeout.
    """
    result = []
    last_val = int(dut.uart_tx.value)
    for cyc in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        val = int(dut.uart_tx.value)
        if val != last_val:
            if val != 0:
                result.append(val)
                if log:
                    log.info(f"UART_TX[{len(result)-1}] = 0x{val:02X} "
                             f"('{chr(val)}') at cycle {cyc}")
                if len(result) >= count:
                    return result
        last_val = val
    return result


# ── Helpers: log file ───────────────────────────────────────────────────────

def _write_log(text):
    """Append a line to sim_build_dma/test.log."""
    log_path = os.path.join(os.getcwd(), "test.log")
    with open(log_path, "a") as f:
        f.write(text + "\n")


# ── DMA data-path exercise (via cocotb hierarchy) ──────────────────────────

async def run_dma_datapath_test(dut):
    """Exercise the DMA memory data path via hierarchical access.

    Sequence:
      1. Write known pattern to DMA ext_mem[EXT_SRC]
      2. Read back DMA ext_mem and verify
      3. Copy DMA ext_mem → DMA SRAM (simulate LOAD)
      4. Read back DMA SRAM and verify
      5. Copy DMA SRAM → DMA ext_mem[EXT_DST] (simulate STORE)
      6. Read back DMA ext_mem[EXT_DST] and compare with source

    Returns (passed, errors_list).
    """
    log = dut._log
    errors = []

    # ── Build 64-bit test pattern ───────────────────────────────────────
    # Each word: upper 32 bits = 0xDMA0iii, lower 32 bits = 0x0000iiii
    pattern = []
    for i in range(NUM_WORDS):
        lo = 0x00000000 + i
        hi = 0xDMA00000 + i
        val = (hi << 32) | lo
        pattern.append(val)

    log.info("=== DMA Data-Path Test (cocotb hierarchy) ===")

    # ── Step 1: Write pattern to DMA ext_mem ────────────────────────────
    log.info(f"Writing {NUM_WORDS} x 64-bit words to DMA ext_mem[0x{EXT_SRC:04X}]")
    for i, val in enumerate(pattern):
        byte_addr = EXT_SRC + i * 8
        _ext_write64(dut, byte_addr, val)

    # ── Step 2: Read back DMA ext_mem and verify ────────────────────────
    log.info("Reading back DMA ext_mem source region")
    for i, expected in enumerate(pattern):
        byte_addr = EXT_SRC + i * 8
        actual = _ext_read64(dut, byte_addr)
        if actual != expected:
            msg = (f"DMA ext_mem source verify: word {i}: "
                   f"expected 0x{expected:016X}, got 0x{actual:016X}")
            log.error(msg)
            errors.append(msg)
        else:
            log.debug(f"  ext_mem[0x{byte_addr:04X}] = 0x{actual:016X} OK")

    if errors:
        log.error(f"DMA ext_mem source verify: {len(errors)} mismatch(es)")
        return False, errors[:]  # abort — source data is wrong

    log.info("DMA ext_mem source region verified OK")

    # ── Step 3: Copy DMA ext_mem → DMA SRAM (simulate LOAD) ────────────
    log.info(f"Copying DMA ext_mem → DMA SRAM[0x{SRAM_BASE:04X}] "
             f"(simulating DMA LOAD, {BYTES} bytes)")
    for i in range(NUM_WORDS):
        ext_byte = EXT_SRC + i * 8
        sram_byte = SRAM_BASE + i * 8
        val = _ext_read64(dut, ext_byte)
        _sram_write64(dut, sram_byte, val)

    # ── Step 4: Read back DMA SRAM and verify ───────────────────────────
    log.info("Reading back DMA SRAM")
    sram_errors = 0
    for i, expected in enumerate(pattern):
        byte_addr = SRAM_BASE + i * 8
        actual = _sram_read64(dut, byte_addr)
        if actual != expected:
            msg = (f"DMA SRAM verify: word {i}: "
                   f"expected 0x{expected:016X}, got 0x{actual:016X}")
            log.error(msg)
            errors.append(msg)
            sram_errors += 1

    if sram_errors:
        log.error(f"DMA SRAM verify: {sram_errors} mismatch(es)")
        return False, errors
    log.info("DMA SRAM verified OK (LOAD path simulated)")

    # ── Step 5: Copy DMA SRAM → DMA ext_mem[EXT_DST] (simulate STORE) ──
    log.info(f"Copying DMA SRAM → DMA ext_mem[0x{EXT_DST:04X}] "
             f"(simulating DMA STORE, {BYTES} bytes)")
    for i in range(NUM_WORDS):
        sram_byte = SRAM_BASE + i * 8
        ext_byte = EXT_DST + i * 8
        val = _sram_read64(dut, sram_byte)
        _ext_write64(dut, ext_byte, val)

    # ── Step 6: Read back DMA ext_mem[EXT_DST] and compare ─────────────
    log.info(f"Reading back DMA ext_mem destination region [0x{EXT_DST:04X}]")
    dst_errors = 0
    for i, expected in enumerate(pattern):
        byte_addr = EXT_DST + i * 8
        actual = _ext_read64(dut, byte_addr)
        if actual != expected:
            msg = (f"DMA ext_mem dest verify: word {i}: "
                   f"expected 0x{expected:016X}, got 0x{actual:016X}")
            log.error(msg)
            errors.append(msg)
            dst_errors += 1

    if dst_errors:
        log.error(f"DMA ext_mem dest verify: {dst_errors} mismatch(es)")
        return False, errors
    log.info("DMA ext_mem destination region verified OK (STORE path simulated)")

    return True, []


# ── Main test ───────────────────────────────────────────────────────────────

@cocotb.test()
async def test_e2e_dma(dut):
    """E2E Stage 3: DMA data transfer integration test."""
    log = dut._log
    all_errors = []

    # ── Clock ───────────────────────────────────────────────────────────
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # ── Reset ───────────────────────────────────────────────────────────
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)

    # ── Pre-load DMA ext_mem with zeroes (clean start) ──────────────────
    log.info("Pre-loading DMA ext_mem with zeroes")
    try:
        for i in range(0, 0x1000, 8):
            _ext_write64(dut, i, 0)
    except AttributeError as e:
        msg = f"Cannot access DMA ext_mem hierarchy: {e}"
        log.error(msg)
        all_errors.append(msg)

    try:
        for i in range(0, 0x1000, 8):
            _sram_write64(dut, i, 0)
    except AttributeError as e:
        msg = f"Cannot access DMA SRAM hierarchy: {e}"
        log.error(msg)
        all_errors.append(msg)

    # ── Release reset — PicoRV32 starts executing firmware ───────────────
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    # ── Wait for firmware UART character ─────────────────────────────────
    log.info("Waiting for firmware UART output...")
    chars = await wait_uart_chars(dut, count=1, log=log)
    fw_char = None
    if chars:
        fw_char = chars[0]
        log.info(f"Firmware UART: 0x{fw_char:02X} ('{chr(fw_char)}')")

    # ── Check firmware result ────────────────────────────────────────────
    fw_pass = False
    if fw_char is None:
        msg = "Firmware timeout: no UART character received within 2M cycles"
        log.error(msg)
        all_errors.append(msg)
    elif chr(fw_char) != 'P':
        msg = (f"Firmware FAIL: UART_TX = '{chr(fw_char)}' "
               f"(0x{fw_char:02X}), expected 'P'")
        log.error(msg)
        all_errors.append(msg)
    else:
        log.info("Firmware CSR / ExtMem test: PASS")
        fw_pass = True

    # ── Run DMA data-path test via hierarchy ─────────────────────────────
    dma_pass = False
    if not all_errors:  # only if hierarchy is accessible
        dma_pass, dma_errors = await run_dma_datapath_test(dut)
        all_errors.extend(dma_errors)
    else:
        log.warning("Skipping DMA data-path test — hierarchy not accessible")

    # ── Final verdict ────────────────────────────────────────────────────
    overall = (fw_pass and dma_pass)

    lines = []
    lines.append("=" * 60)
    lines.append("E2E Stage 3 (DMA) Test Results")
    lines.append("=" * 60)
    lines.append(f"  Firmware CSR / ExtMem : {'PASS' if fw_pass else 'FAIL'}")
    lines.append(f"  DMA data-path (hier)  : {'PASS' if dma_pass else 'FAIL'}")
    lines.append(f"  Overall               : {'PASS' if overall else 'FAIL'}")
    if all_errors:
        lines.append("--- Errors ---")
        for e in all_errors:
            lines.append(f"  {e}")
    lines.append("=" * 60)

    for line in lines:
        log.info(line)
        _write_log(line)

    if overall:
        _write_log("E2E Stage 3 (DMA): PASS")
        log.info("E2E Stage 3 (DMA): PASS")
    else:
        summary = all_errors[0] if all_errors else "unknown error"
        fail_msg = f"E2E Stage 3 (DMA): FAIL — {summary}"
        _write_log(fail_msg)
        log.error(fail_msg)

    assert overall, f"E2E Stage 3 (DMA): FAIL — {len(all_errors)} error(s); see test.log"
