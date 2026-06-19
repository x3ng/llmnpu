"""test_e2e_gemm.py — E2E Stage 4: Full NPU GEMM Execution Test

Loads the full SoC (top_soc) with firmware_gemm.c which performs
a 16x16x16 INT8 GEMM via the NPU runtime, DMA STOREs the result
from O-SRAM to gemm_result in ext_mem, and compares all 256 output
elements against a precomputed golden matrix embedded in firmware .rodata.

The firmware outputs a diagnostic sequence on UART TX:
  Stage markers: I/i (init), A/a (A-DMA), B/b (B-DMA), G/g (GEMM),
                 S/s (STORE)
  Result:        'P' (all pass) or
                 'F' + 1-byte-count + up to 4 x (idx, hw_le16, gold_le16)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles


# ── Helpers ──────────────────────────────────────────────────────────────


async def wait_uart_chars(dut, count=1, timeout_cycles=2000000, log=None):
    """Capture `count` uart_tx value changes, starting from the first
    non-zero value. Tracks any change of value (including the initial
    transition from 0).

    Returns a list of captured values, or empty list on timeout.
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
                             f"('{chr(val) if 0x20 <= val < 0x7F else '?'}') at cycle {cyc}")
                if len(result) >= count:
                    return result
        last_val = val
    return result


async def wait_uart_all(dut, max_chars=64, timeout_cycles=2000000,
                        idle_cycles=5000, log=None):
    """Capture all UART characters until the line goes idle for `idle_cycles`.
    Returns list of captured byte values.
    """
    result = []
    last_val = int(dut.uart_tx.value)
    idle_count = 0
    for cyc in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        val = int(dut.uart_tx.value)
        if val != last_val:
            if val != 0:
                result.append(val)
                if log:
                    c = chr(val) if 0x20 <= val < 0x7F else '?'
                    log.info(f"UART_TX[{len(result)-1}] = 0x{val:02X} ('{c}') at cycle {cyc}")
                idle_count = 0
                if len(result) >= max_chars:
                    return result
            last_val = val
        else:
            idle_count += 1
            if len(result) > 0 and idle_count >= idle_cycles:
                return result
    return result


# ── ext_mem_model debug helpers (read-only diagnostics) ───────────────


def _ext_mem_word_addr(byte_addr):
    """Convert a 0x4000_xxxx byte address to ext_mem_model word index."""
    return (byte_addr - 0x40000000) // 4


def _read_ext_mem_word(dut, word_idx):
    """Read a 32-bit word from ext_mem_model.
    Returns (is_valid, value).  is_valid=False if read returned 'x'."""
    try:
        val = int(dut.u_dram.mem[word_idx].value)
        return (True, val)
    except (ValueError, Exception):
        return (False, 0)


def ext_mem_read_bytes(dut, byte_addr, length):
    """Read `length` bytes from ext_mem_model starting at byte_addr
    (in 0x4000_xxxx space).

    Returns (is_valid, bytes) tuple.  is_valid=False if any read returned 'x'.
    """
    data = bytearray()
    word_addr = _ext_mem_word_addr(byte_addr)
    remainder = byte_addr & 3
    all_valid = True

    while len(data) < length:
        ok, word = _read_ext_mem_word(dut, word_addr)
        if not ok:
            all_valid = False
            data.extend(b'\x00' * min(4, length - len(data)))
        else:
            word_bytes = word.to_bytes(4, 'little')
            start = remainder if len(data) == 0 else 0
            end = min(4, start + (length - len(data)))
            data.extend(word_bytes[start:end])
        word_addr += 1
    return (all_valid, bytes(data))


# ── Golden reference computation ───────────────────────────────────────


def _gen_golden_C():
    """Compute golden_C = A x B (int16) using Python for diagnostic
    comparison on test failure.  Matches the firmware's embedded golden_C
    and test_A/test_B patterns."""
    import struct
    A = [[0]*16 for _ in range(16)]
    B = [[0]*16 for _ in range(16)]
    for i in range(16):
        for j in range(16):
            A[i][j] = (i + j) % 7 - 3
            B[i][j] = (i * 3 + j) % 7 - 3
    data = bytearray()
    for i in range(16):
        for j in range(16):
            acc = 0
            for k in range(16):
                acc += A[i][k] * B[k][j]
            data.extend(struct.pack('<h', acc))
    return bytes(data)


# ── Tests ────────────────────────────────────────────────────────────────


@cocotb.test()
async def test_e2e_gemm_pass(dut):
    """Run the GEMM firmware and verify the NPU computes the correct
    matrix multiplication result.

    The firmware outputs a diagnostic sequence on UART.  We parse it
    to identify exactly where the pipeline fails.
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

    # ── Read firmware symbol addresses from ELF (for diagnostics) ─────
    import subprocess, os
    elf_path = os.path.join(os.path.dirname(__file__),
                            "..", "..", "build", "firmware_gemm.elf")
    elf_path = os.path.abspath(elf_path)

    def _get_elf_symbols(elf):
        """Extract symbol addresses from ELF. Returns dict name->addr or {}."""
        syms = {}
        try:
            out = subprocess.check_output(
                ["riscv32-none-elf-objdump", "-t", elf],
                stderr=subprocess.STDOUT
            ).decode()
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[-1] not in ('', '*ABS*'):
                    try:
                        addr = int(parts[0], 16)
                        name = parts[-1]
                        syms[name] = addr
                    except ValueError:
                        pass
        except Exception as e:
            dut._log.warning(f"Could not read ELF symbols: {e}")
        return syms

    symbols = _get_elf_symbols(elf_path)
    C_addr = symbols.get('gemm_result', 0x40001024)
    dut._log.info(f"gemm_result @ 0x{C_addr:08X}")

    # ── Capture all UART output ───────────────────────────────────────
    chars = await wait_uart_all(dut, max_chars=64, timeout_cycles=2000000,
                                idle_cycles=100000, log=dut._log)

    dut._log.info(f"Total UART chars received: {len(chars)}")
    dut._log.info(f"Raw UART bytes: {' '.join(f'0x{c:02X}' for c in chars)}")

    if len(chars) == 0:
        dut._log.error("No UART characters received — firmware may be hung")
        assert False, "No UART output received within timeout"

    # ── Parse stage markers (first 5 chars) ───────────────────────────
    stage_names = ["INIT", "A-DMA", "B-DMA", "GEMM", "STORE"]
    stage_expected = ['I', 'A', 'B', 'G', 'S']
    stage_results = []

    if len(chars) < 5:
        dut._log.warning(
            f"Only {len(chars)} UART chars received, expected at least 5 stage markers"
        )
        # Pad with what we have
        for i in range(min(len(chars), 5)):
            c = chr(chars[i]) if 0x20 <= chars[i] < 0x7F else '?'
            expected_ok = stage_expected[i]
            ok = (c == expected_ok)
            stage_results.append((stage_names[i], c, ok))
            dut._log.info(f"  Stage {stage_names[i]}: got '{c}' — "
                          f"{'PASS' if ok else 'FAIL'}")
        # Fill remaining with 'missing'
        for i in range(len(chars), 5):
            stage_results.append((stage_names[i], '<none>', False))
            dut._log.info(f"  Stage {stage_names[i]}: <no output> — MISSING")
    else:
        for i in range(5):
            c = chr(chars[i]) if 0x20 <= chars[i] < 0x7F else '?'
            expected_ok = stage_expected[i]
            ok = (c == expected_ok)
            stage_results.append((stage_names[i], c, ok))
            dut._log.info(f"  Stage {stage_names[i]}: got '{c}' — "
                          f"{'PASS' if ok else 'FAIL'}")

    # Check which stage failed first
    first_fail = None
    for name, c, ok in stage_results:
        if not ok:
            first_fail = name
            break

    if first_fail:
        dut._log.info(f"*** FIRST FAILING STAGE: {first_fail} ***")
    else:
        dut._log.info("All 5 pipeline stages reported OK")

    # ── Parse result (6th char onwards) ──────────────────────────────
    if len(chars) >= 6:
        result_char = chars[5]
        if chr(result_char) == 'P':
            dut._log.info("Comparison result: PASS (0 mismatches)")
        elif chr(result_char) == 'F':
            # Parse mismatch details
            if len(chars) >= 7:
                mismatch_count = chars[6]
                dut._log.info(f"Comparison result: FAIL — "
                              f"{mismatch_count} mismatches (reported, may be capped at 255)")

                # Parse up to 4 mismatch details (each: 1 idx + 2 hw + 2 gold = 5 bytes)
                pos = 7
                n_detail = 0
                while pos + 4 < len(chars) and n_detail < 4:
                    idx = chars[pos]
                    hw_lo = chars[pos + 1]
                    hw_hi = chars[pos + 2]
                    gold_lo = chars[pos + 3]
                    gold_hi = chars[pos + 4]

                    # Interpret as signed int16 LE
                    hw_val = (hw_lo | (hw_hi << 8))
                    if hw_val >= 0x8000:
                        hw_val -= 0x10000
                    gold_val = (gold_lo | (gold_hi << 8))
                    if gold_val >= 0x8000:
                        gold_val -= 0x10000

                    row = idx // 16
                    col = idx % 16
                    dut._log.info(
                        f"  mismatch[{idx}] (row={row},col={col}): "
                        f"hw={hw_val}, golden={gold_val}"
                    )
                    pos += 5
                    n_detail += 1
            else:
                dut._log.info("Comparison result: FAIL — "
                              "(no mismatch details received)")
        else:
            c = chr(result_char) if 0x20 <= result_char < 0x7F else f'0x{result_char:02X}'
            dut._log.info(f"Unexpected result char: '{c}'")
    else:
        dut._log.info("No result character received (only stage markers)")

    # ── Additional: read HW gemm_result for cross-check ───────────────
    golden_bytes = _gen_golden_C()
    rok, result_bytes = ext_mem_read_bytes(dut, C_addr, 512)

    if rok:
        import struct
        hw_mismatches = 0
        for i in range(256):
            hw_val = struct.unpack_from('<h', result_bytes, i * 2)[0]
            gold_val = struct.unpack_from('<h', golden_bytes, i * 2)[0]
            if hw_val != gold_val:
                if hw_mismatches < 4:
                    row = i // 16
                    col = i % 16
                    dut._log.info(
                        f"  [XCHECK] mismatch[{i}] (row={row},col={col}): "
                        f"hw={hw_val}, golden={gold_val}"
                    )
                hw_mismatches += 1
        dut._log.info(f"[XCHECK] Total HW mismatches: {hw_mismatches}/256")

        # Summarize first few HW values vs golden for pattern diagnosis
        dut._log.info("[XCHECK] First 8 gemm_result values vs golden:")
        for i in range(min(8, 256)):
            hw_val = struct.unpack_from('<h', result_bytes, i * 2)[0]
            gold_val = struct.unpack_from('<h', golden_bytes, i * 2)[0]
            match_str = "==" if hw_val == gold_val else "!="
            dut._log.info(f"  [{i:3d}] hw={hw_val:6d} {match_str} golden={gold_val:6d}")
    else:
        dut._log.warning("Could not read gemm_result from ext_mem_model (returned X)")

    # ── Final assertion ───────────────────────────────────────────────
    # The test passes only if all stage markers are uppercase AND
    # the result is 'P'.
    all_stages_ok = all(ok for _, _, ok in stage_results)
    result_ok = (len(chars) >= 6 and chr(chars[5]) == 'P')

    pass_str = (
        f"Stages: {'ALL-OK' if all_stages_ok else 'FAIL at ' + (first_fail or '?')}, "
        f"Result: {'PASS' if result_ok else 'FAIL'}"
    )

    assert all_stages_ok and result_ok, (
        f"E2E GEMM test failed. {pass_str}. "
        f"UART sequence: {' '.join(f'0x{c:02X}' for c in chars)}"
    )

    dut._log.info(f"E2E GEMM: FULL PASS — {pass_str}")
