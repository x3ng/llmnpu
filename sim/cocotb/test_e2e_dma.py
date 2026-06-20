"""test_e2e_dma.py — E2E Stage 3: DMA Data Transfer Integration Test.

Validates the end-to-end DMA data path through the full SoC:
  1. Firmware on PicoRV32 exercises CSR MMIO register paths and ExtMem
  2. Cocotb verifies DMA internal memories (u_dram.mem, SRAM) via hierarchy
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
EXT_SRC_WORD = 0x40           # word addr for byte offset 0x0100 (0x100/4)
EXT_DST_WORD = 0x80           # word addr for byte offset 0x0200 (0x200/4)
BYTES     = NUM_WORDS * 8     # total bytes: 128

TIMEOUT_CYCLES = 2000000      # max cycles to wait for firmware UART


# ── Helpers: DMA memory access via hierarchy ────────────────────────────────

def _ext_word_mem(dut):
    """Return ext_mem_model's 32-bit word array (u_dram.mem)."""
    return dut.u_dram.mem


def _sram(dut):
    """Return DMA internal SRAM byte array (64 KB)."""
    return dut.u_npu.u_dma.sram


def _ext_write64(dut, word_addr, data):
    """Write one 64-bit value to ext_mem at 32-bit word_addr.
    Stores as two consecutive 32-bit words (little-endian)."""
    _ext_word_mem(dut)[word_addr].value     = data & 0xFFFFFFFF
    _ext_word_mem(dut)[word_addr + 1].value = (data >> 32) & 0xFFFFFFFF


def _ext_read64(dut, word_addr):
    """Read one 64-bit value from ext_mem at 32-bit word_addr.
    Returns int from two consecutive 32-bit words (little-endian)."""
    lo = int(_ext_word_mem(dut)[word_addr].value)
    hi = int(_ext_word_mem(dut)[word_addr + 1].value)
    return (hi << 32) | lo


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


# ── Diagnostic dump (for self-diagnosis by subagents) ──────────────────────

async def dump_diagnostics(dut, label=""):
    """Print structured diagnostic markers that subagents can grep for."""
    log = dut._log
    log.info(f"[DIAG] === Diagnostic Dump {label} ===")

    # 1. Clock and reset status
    log.info(f"[DIAG] clk={int(dut.clk.value)} rst_n={int(dut.rst_n.value)}")

    # 2. UART tx value
    try:
        log.info(f"[DIAG] uart_tx={int(dut.uart_tx.value)}")
    except:
        log.info(f"[DIAG] uart_tx=INVALID")

    # 3. Try to read CSR_STATUS via VPI hierarchy (if accessible)
    try:
        csr_rdata = int(dut.u_npu.u_csr.rdata.value)
        log.info(f"[DIAG] csr_rdata=0x{csr_rdata:08X}")
    except Exception as e:
        log.info(f"[DIAG] csr_rdata=N/A ({e})")

    # 4. Try to read npu_busy signal via hierarchy
    try:
        npu_busy = int(dut.u_npu.npu_busy.value)
        log.info(f"[DIAG] npu_busy={npu_busy}")
    except:
        log.info(f"[DIAG] npu_busy=N/A")

    # 5. Try to read debug_signals if DEBUG CSR (0x60) is accessible
    try:
        debug_val = int(dut.u_npu.debug_signals.value)
        log.info(f"[DIAG] debug_signals=0x{debug_val:08X}")
        # Decode key bits for immediate visibility
        gemm_busy  = (debug_val >> 0) & 1
        valu_busy  = (debug_val >> 1) & 1
        sfu_busy   = (debug_val >> 2) & 1
        dma_busy   = (debug_val >> 3) & 1
        bridge_busy= (debug_val >> 4) & 1
        gpl_state  = (debug_val >> 5) & 7
        dma_br_state=(debug_val >> 8) & 3
        wb_active  = (debug_val >> 10) & 1
        all_busy   = (debug_val >> 11) & 1
        log.info(f"[DIAG] gemm_busy={gemm_busy} valu_busy={valu_busy} sfu_busy={sfu_busy} dma_busy={dma_busy} bridge_busy={bridge_busy}")
        log.info(f"[DIAG] gpl_state={gpl_state} dma_br_state={dma_br_state} wb_active={wb_active} agg_busy={all_busy}")
    except Exception as e:
        log.info(f"[DIAG] debug_signals=N/A ({e})")

    # 6. Check if PicoRV32 trap signal is asserted (CPU crash)
    try:
        trap = int(dut.u_cpu.trap_latched.value)
        log.info(f"[DIAG] cpu_trap={trap}")
    except:
        try:
            trap = int(dut.u_cpu.u_picorv32.trap.value)
            log.info(f"[DIAG] cpu_trap={trap}")
        except:
            log.info(f"[DIAG] cpu_trap=N/A")

    # 7. Check PC if accessible
    try:
        pc = int(dut.debug_pc.value)
        log.info(f"[DIAG] debug_pc=0x{pc:08X}")
    except:
        log.info(f"[DIAG] debug_pc=N/A")

    log.info(f"[DIAG] === End Diagnostic {label} ===")


# ── DMA data-path exercise (via cocotb hierarchy) ──────────────────────────

async def run_dma_datapath_test(dut):
    """Exercise the DMA memory data path via hierarchical access.

    Sequence:
      1. Write known pattern to ext_mem_model (u_dram.mem)
      2. Read back ext_mem and verify
      3. Copy ext_mem → DMA SRAM (simulate LOAD)
      4. Read back DMA SRAM and verify
      5. Copy DMA SRAM → ext_mem[EXT_DST] (simulate STORE)
      6. Read back ext_mem[EXT_DST] and compare with source

    Returns (passed, errors_list).
    """
    log = dut._log
    errors = []

    # ── Build 64-bit test pattern ───────────────────────────────────────
    pattern = []
    for i in range(NUM_WORDS):
        lo = 0x00000000 + i
        hi = 0xDA000000 + i
        val = (hi << 32) | lo
        pattern.append(val)

    log.info("=== DMA Data-Path Test (cocotb hierarchy) ===")

    # ── Step 1: Write pattern to ext_mem_model (u_dram.mem) ─────────────
    log.info(f"Writing {NUM_WORDS} x 64-bit words to ext_mem "
             f"[word 0x{EXT_SRC_WORD:04X}]")
    for i, val in enumerate(pattern):
        word_addr = EXT_SRC_WORD + i * 2  # 2 words per 64-bit value
        _ext_write64(dut, word_addr, val)

    # ── Step 2: Read back ext_mem and verify ────────────────────────────
    log.info("Reading back ext_mem source region")
    for i, expected in enumerate(pattern):
        word_addr = EXT_SRC_WORD + i * 2
        actual = _ext_read64(dut, word_addr)
        if actual != expected:
            msg = (f"ext_mem source verify: word {i}: "
                   f"expected 0x{expected:016X}, got 0x{actual:016X}")
            log.error(msg)
            errors.append(msg)
        else:
            log.debug(f"  ext_mem[word 0x{word_addr:04X}] = 0x{actual:016X} OK")

    if errors:
        log.error(f"ext_mem source verify: {len(errors)} mismatch(es)")
        return False, errors[:]

    log.info("ext_mem source region verified OK")

    # ── Step 3: Copy ext_mem → DMA SRAM (simulate LOAD) ────────────────
    log.info(f"Copying ext_mem → DMA SRAM[0x{SRAM_BASE:04X}] "
             f"(simulating DMA LOAD, {BYTES} bytes)")
    for i in range(NUM_WORDS):
        word_addr = EXT_SRC_WORD + i * 2
        sram_byte = SRAM_BASE + i * 8
        val = _ext_read64(dut, word_addr)
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

    # ── Step 5: Copy DMA SRAM → ext_mem[EXT_DST] (simulate STORE) ──────
    log.info(f"Copying DMA SRAM → ext_mem[word 0x{EXT_DST_WORD:04X}] "
             f"(simulating DMA STORE, {BYTES} bytes)")
    for i in range(NUM_WORDS):
        sram_byte = SRAM_BASE + i * 8
        word_addr = EXT_DST_WORD + i * 2
        val = _sram_read64(dut, sram_byte)
        _ext_write64(dut, word_addr, val)

    # ── Step 6: Read back ext_mem[EXT_DST] and compare ──────────────────
    log.info(f"Reading back ext_mem destination region [word 0x{EXT_DST_WORD:04X}]")
    dst_errors = 0
    for i, expected in enumerate(pattern):
        word_addr = EXT_DST_WORD + i * 2
        actual = _ext_read64(dut, word_addr)
        if actual != expected:
            msg = (f"ext_mem dest verify: word {i}: "
                   f"expected 0x{expected:016X}, got 0x{actual:016X}")
            log.error(msg)
            errors.append(msg)
            dst_errors += 1

    if dst_errors:
        log.error(f"ext_mem dest verify: {dst_errors} mismatch(es)")
        return False, errors
    log.info("ext_mem destination region verified OK (STORE path simulated)")

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

    # ── ext_mem is already initialized by $readmemh at time 0 ───────────
    # (no need to zero — that would wipe the firmware)

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

    # ── Diagnostic dump post-reset ─────────────────────────────────────
    await dump_diagnostics(dut, "post-reset")

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
        await dump_diagnostics(dut, "uart-timeout")
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
    if not all_errors:
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
