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
import os
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
    def result_complete(buf):
        if ord('P') in buf:
            return True
        for pos, byte in enumerate(buf):
            if byte == ord('F') and pos + 1 < len(buf):
                detail_count = min(buf[pos + 1], 4)
                return len(buf) >= pos + 2 + detail_count * 5
        return False

    result = []
    last_val = int(dut.uart_tx.value)
    idle_count = 0
    result_seen_idle_cycles = 20000
    have_uart_strobe = all(
        hasattr(dut, name)
        for name in ("periph_req", "periph_write", "periph_addr", "periph_wdata")
    )
    for cyc in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")

        if have_uart_strobe:
            try:
                saw_strobe = False
                if (int(dut.periph_req.value)
                        and int(dut.periph_write.value)
                        and int(dut.periph_addr.value) == 0x00000008):
                    saw_strobe = True
                    val = int(dut.periph_wdata.value) & 0xFF
                    if val != 0:
                        result.append(val)
                        if log:
                            c = chr(val) if 0x20 <= val < 0x7F else '?'
                            log.info(f"UART_WR[{len(result)-1}] = 0x{val:02X} ('{c}') at cycle {cyc}")
                        idle_count = 0
                        if len(result) >= max_chars or result_complete(result):
                            return result
                if not saw_strobe:
                    idle_count += 1
                    if ord('F') in result and idle_count >= result_seen_idle_cycles:
                        return result
                    if len(result) > 0 and idle_count >= idle_cycles:
                        return result
                continue
            except Exception:
                have_uart_strobe = False

        val = int(dut.uart_tx.value)
        if val != last_val:
            if val != 0:
                result.append(val)
                if log:
                    c = chr(val) if 0x20 <= val < 0x7F else '?'
                    log.info(f"UART_TX[{len(result)-1}] = 0x{val:02X} ('{c}') at cycle {cyc}")
                idle_count = 0
                if len(result) >= max_chars or result_complete(result):
                    return result
            last_val = val
        else:
            idle_count += 1
            if ord('F') in result and idle_count >= result_seen_idle_cycles:
                return result
            if len(result) > 0 and idle_count >= idle_cycles:
                return result
    return result


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

    # ═══════════════════════════════════════════════════════════════════
    # DIAGNOSTIC: AXI read path signal monitor (background coroutine)
    # ═══════════════════════════════════════════════════════════════════
    async def axi_diag():
        """Monitor key AXI read path signals and log state changes."""
        log = dut._log
        log.info("=== AXI DIAG: starting background monitor ===")

        # Hierarchy paths (verify exist before monitoring)
        sig_map = {
            # -- DMA wrapper external AXI ports (passthrough from engine) --
            'wrap_arvalid':   'u_npu.u_dma.wrapper.m_axi_arvalid',
            'wrap_araddr':    'u_npu.u_dma.wrapper.m_axi_araddr',
            'wrap_arready':   'u_npu.u_dma.wrapper.m_axi_arready',
            'wrap_rvalid':    'u_npu.u_dma.wrapper.m_axi_rvalid',
            'wrap_rdata':     'u_npu.u_dma.wrapper.m_axi_rdata',
            'wrap_rready':    'u_npu.u_dma.wrapper.m_axi_rready',
            # -- DMA wrapper internal DMA-engine signals --
            'wrap_dma_arvalid':  'u_npu.u_dma.wrapper.dma_arvalid',
            'wrap_dma_arready':  'u_npu.u_dma.wrapper.dma_arready',
            'wrap_dma_rvalid':   'u_npu.u_dma.wrapper.dma_rvalid',
            'wrap_dma_rready':   'u_npu.u_dma.wrapper.dma_rready',
            'wrap_dma_rdata':    'u_npu.u_dma.wrapper.dma_rdata',
            'wrap_dma_rlast':    'u_npu.u_dma.wrapper.dma_rlast',
            # -- DMA wrapper FSM state --
            'wrap_state':     'u_npu.u_dma.wrapper.state',
            'wrap_word_cnt':  'u_npu.u_dma.wrapper.word_cnt',
            # -- DMA wrapper FSM state --
            'wrap_state':     'u_npu.u_dma.wrapper.state',
            'wrap_word_cnt':  'u_npu.u_dma.wrapper.word_cnt',
            # -- DMA wrapper stream from axi_dma_rd --
            'wrap_st_valid':  'u_npu.u_dma.wrapper.rd_stream_tvalid',
            'wrap_st_tlast':  'u_npu.u_dma.wrapper.rd_stream_tlast',
            'wrap_st_tdata':  'u_npu.u_dma.wrapper.rd_stream_tdata',
            'wrap_st_tready': 'u_npu.u_dma.wrapper.rd_stream_tready',
            # -- DMA wrapper descriptor interface --
            'wrap_desc_valid': 'u_npu.u_dma.wrapper.rd_desc_valid',
            'wrap_desc_ready': 'u_npu.u_dma.wrapper.rd_desc_ready',
            # -- npu_dma FSM state --
            'dma_state':      'u_npu.u_dma.state',
            'dma_wr_valid':   'u_npu.u_dma.wrapper_rd_valid',
            'dma_wr_done':    'u_npu.u_dma.wrapper_done',
            'dma_wr_start':   'u_npu.u_dma.wrapper_start',
            'dma_busy':       'u_npu.u_dma.busy',
            'dma_done':       'u_npu.u_dma.done',
            # -- npu_top: DMA bridge and busy signals --
            'npu_busy':       'u_npu.npu_busy',
            'npubr_state':    'u_npu.dma_br_state',
            'npubr_cnt':      'u_npu.dma_br_cnt',
            # -- npu_top: csr_dma_start and related --
            'csr_dma_start':  'u_npu.csr_dma_start',
            'csr_dma_is_store':'u_npu.csr_dma_is_store',
            # -- AXI bridge in top_soc --
            'soc_ar_state':   'ar_state',
            'soc_aw_state':   'aw_state',
            'soc_ar_beat':    'ar_beat',
            'soc_ar_len':     'ar_len',
            'soc_ar_addr':    'ar_addr',
            'soc_dma_arvalid':'dma_axi_arvalid',
            'soc_dma_arready':'dma_axi_arready',
            'soc_dma_araddr': 'dma_axi_araddr',
            'soc_dma_rvalid': 'dma_axi_rvalid',
            'soc_dma_rdata':  'dma_axi_rdata',
            'soc_dma_rlast':  'dma_axi_rlast',
            'soc_dma_rready': 'dma_axi_rready',
            'soc_axi_rd_en':  'axi_rd_en',
            'soc_axi_rd_addr':'axi_rd_addr',
            'soc_axi_rd_rdata':'axi_rd_rdata',
            # -- ext_mem_model AXI port --
            'dram_axi_rd_rdata': 'u_dram.axi_rd_rdata',
        }

        # Cache last values
        prev = {}
        for name, hpath in sig_map.items():
            try:
                sig = eval(f'dut.{hpath}')
                prev[name] = int(sig.value)
            except Exception as e:
                log.warning(f"AXI DIAG: cannot access {hpath}: {e}")
                prev[name] = None

        # Track first transition
        first_seen = set()

        for cyc in range(500000):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ps")

            for name, hpath in sig_map.items():
                if prev[name] is None:
                    continue
                try:
                    sig = eval(f'dut.{hpath}')
                    cur = int(sig.value)
                except Exception:
                    continue
                if cur != prev[name]:
                    # Always log: state machines and critical control signals
                    # For data signals: log first transition + non-zero changes
                    is_state = name.endswith('_state')
                    ctrl_signals = {'npu_busy', 'dma_busy', 'dma_done', 'dma_wr_start',
                                    'dma_wr_valid', 'dma_wr_done', 'csr_dma_start',
                                    'csr_dma_is_store', 'wrap_arvalid', 'wrap_rvalid',
                                    'wrap_dma_arvalid', 'wrap_dma_rvalid',
                                    'wrap_st_valid', 'wrap_st_tlast',
                                    'soc_dma_arvalid', 'soc_dma_rvalid',
                                    'soc_axi_rd_en', 'wrap_desc_valid', 'wrap_desc_ready'}
                    if is_state or name in ctrl_signals or name not in first_seen or cur != 0:
                        ar_state_names = {0:'IDLE',1:'LOW',2:'HIGH',3:'WAIT'}
                        aw_state_names = {0:'IDLE',1:'WAIT_W',2:'WR_LO',3:'WR_HI',4:'NEXT',5:'RESP'}
                        wr_state_names = {0:'IDLE',1:'RD_START',2:'RD_XFER',3:'RD_DONE',
                                          4:'WR_ACCEPT',5:'WR_AW',6:'WR_W',7:'WR_B'}
                        dma_state_names = {0:'IDLE',1:'XFER',2:'DONE'}
                        br_state_names  = {0:'IDLE',1:'COPY',2:'PREFILL'}

                        val_str = str(cur)
                        if name == 'soc_ar_state':
                            val_str = ar_state_names.get(cur, str(cur))
                        elif name == 'soc_aw_state':
                            val_str = aw_state_names.get(cur, str(cur))
                        elif name == 'wrap_state':
                            val_str = wr_state_names.get(cur, str(cur))
                        elif name == 'dma_state':
                            val_str = dma_state_names.get(cur, str(cur))
                        elif name == 'npubr_state':
                            val_str = br_state_names.get(cur, str(cur))

                        log.info(f"[AXI_DIAG c{cyc}] {name}: {prev[name]} -> {val_str}")
                        prev[name] = cur
                        first_seen.add(name)

            # Early termination: if UART raised (means test finished or almost done)
            if cyc > 50000 and all(v == 0 for v in [
                prev.get('wrap_arvalid', 0), prev.get('wrap_rvalid', 0),
                prev.get('dma_busy', 0), prev.get('dma_done', 0)]) and prev.get('soc_ar_state', 0) == 0:
                # nothing happening for a while — might be done
                pass

    if os.getenv("E2E_AXI_DIAG") == "1":
        cocotb.start_soon(axi_diag())

    async def dma_diag():
        prev_dma_start = 0
        prev_dma_done = 0
        wsram_writes = 0
        wsram_hi_writes = 0
        osram_reads = 0
        store_prefills = 0
        axi_writes = 0
        ext_writes = 0

        for _ in range(200000):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ps")
            try:
                dma_start = int(dut.u_npu.dma_start.value)
                dma_done = int(dut.u_npu.dma_done.value)
                m0_req = int(dut.u_npu.m0_req.value)
                m0_wen = int(dut.u_npu.m0_wen.value)
                m0_grant = int(dut.u_npu.xbar_m0_grant.value)
                m0_addr = int(dut.u_npu.m0_addr.value) & 0xFFFF
                m0_wdata = int(dut.u_npu.m0_wdata.value) & 0xFFFFFFFF
                m0_rdata = int(dut.u_npu.xbar_m0_rdata.value) & 0xFFFFFFFF
                br_state = int(dut.u_npu.dma_br_state.value)
                br_cnt = int(dut.u_npu.dma_br_cnt.value) & 0xFFFF
                csr_sram = int(dut.u_npu.csr_dma_sram_addr.value) & 0xFFFF
                csr_len = int(dut.u_npu.csr_dma_length.value) & 0xFFFF
                csr_ext = int(dut.u_npu.csr_dma_ext_addr.value) & 0xFFFFFFFF
                sim_we = int(dut.u_npu.dma_sim_sram_we.value)
                sim_addr = int(dut.u_npu.dma_sim_sram_addr.value) & 0xFFFF
                sim_wdata = int(dut.u_npu.dma_sim_sram_wdata.value) & 0xFFFFFFFFFFFFFFFF
                axi_wvalid = int(dut.dma_axi_wvalid.value)
                axi_wready = int(dut.dma_axi_wready.value)
                axi_wdata = int(dut.dma_axi_wdata.value) & 0xFFFFFFFFFFFFFFFF
                axi_awaddr = int(dut.dma_axi_awaddr.value) & 0xFFFFFFFF
                axi_wr_en = int(dut.axi_wr_en.value)
                axi_wr_addr = int(dut.axi_wr_addr.value) & 0xFFFFFF
                axi_wr_wdata = int(dut.axi_wr_wdata.value) & 0xFFFFFFFF
            except Exception:
                continue

            if dma_start and not prev_dma_start:
                dut._log.info(
                    "[DMADIAG] start ext=0x%08X sram=0x%04X len=%d br_state=%d",
                    csr_ext, csr_sram, csr_len, br_state,
                )
            if dma_done and not prev_dma_done:
                dut._log.info(
                    "[DMADIAG] dma_done sram=0x%04X len=%d br_state=%d",
                    csr_sram, csr_len, br_state,
                )
            if m0_req and m0_wen and m0_grant and ((m0_addr >> 12) == 1):
                wsram_writes += 1
                if m0_addr >= 0x1080:
                    wsram_hi_writes += 1
                if wsram_writes <= 8 or m0_addr >= 0x1080:
                    dut._log.info(
                        "[DMADIAG] WSRAM write addr=0x%04X data=0x%08X br_cnt=%d total=%d hi=%d",
                        m0_addr, m0_wdata, br_cnt, wsram_writes, wsram_hi_writes,
                    )
            if m0_req and not m0_wen and m0_grant and ((m0_addr >> 12) == 2):
                osram_reads += 1
                if osram_reads <= 8:
                    dut._log.info(
                        "[DMADIAG] OSRAM read addr=0x%04X rdata=0x%08X br_cnt=%d total=%d",
                        m0_addr, m0_rdata, br_cnt, osram_reads,
                    )
            if sim_we and 0x2000 <= sim_addr < 0x2200:
                store_prefills += 1
                if store_prefills <= 8:
                    dut._log.info(
                        "[DMADIAG] prefill sim_sram addr=0x%04X data=0x%016X total=%d",
                        sim_addr, sim_wdata, store_prefills,
                    )
            if axi_wvalid and axi_wready and 0x40001100 <= axi_awaddr < 0x40001400:
                axi_writes += 1
                if axi_writes <= 8:
                    dut._log.info(
                        "[DMADIAG] AXI W awaddr=0x%08X wdata=0x%016X total=%d",
                        axi_awaddr, axi_wdata, axi_writes,
                    )
            if axi_wr_en and 0x440 <= axi_wr_addr < 0x500:
                ext_writes += 1
                if ext_writes <= 16:
                    dut._log.info(
                        "[DMADIAG] ext write word=%d data=0x%08X total=%d",
                        axi_wr_addr, axi_wr_wdata, ext_writes,
                    )

            prev_dma_start = dma_start
            prev_dma_done = dma_done

    if os.getenv("E2E_DMA_DIAG") == "1":
        cocotb.start_soon(dma_diag())

    async def gemm_diag():
        logged = 0
        for _ in range(120000):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ps")
            try:
                state = int(dut.u_npu.u_gemm.ctrl_state.value)
                k_cnt = int(dut.u_npu.u_gemm.ctrl.k_cnt.value)
                feed_k = int(dut.u_npu.gpl_feed_k.value)
                feeding = int(dut.u_npu.gpl_feeding.value)
                gemm_busy = int(dut.u_npu.gemm_busy.value)
                a_valid = int(dut.u_npu.u_gemm.a_valid.value)
            except Exception:
                continue

            if state == 3 or (logged and state in (4, 5)):
                if logged < 24:
                    dut._log.info(
                        "[GEMMDIAG] state=%d k_cnt=%d feed_k=%d feeding=%d busy=%d a_valid=%d",
                        state, k_cnt, feed_k, feeding, gemm_busy, a_valid,
                    )
                    logged += 1

    if os.getenv("E2E_GEMM_DIAG") == "1":
        cocotb.start_soon(gemm_diag())

    # ═══════════════════════════════════════════════════════════════════
    # HANG DIAGNOSTIC: probe trap_latched, PC, npu_busy (background)
    # ═══════════════════════════════════════════════════════════════════
    async def hang_diag():
        log = dut._log
        def read_path(path, default="N/A"):
            obj = dut
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return int(obj.value)
            except Exception:
                return default

        for cyc in range(2000000):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ps")
            # Probe 1: trap_latched at cycle 1000
            if cyc == 999:
                try:
                    trap_val = int(dut.trap_latched.value)
                    log.info(f"[HANGDIAG] cycle 1000: trap_latched = {trap_val}")
                except Exception as e:
                    log.warning(f"[HANGDIAG] cycle 1000: trap_latched read failed: {e}")
            # Probe 2+3: PC, NPU busy, and CPU bus state at early checkpoints.
            if cyc in (59999, 79999, 99999, 119999, 149999,
                       199999, 399999, 599999, 799999, 999999,
                       1199999, 1399999, 1599999, 1799999, 1999999):
                pc = read_path("u_cpu.u_picorv32.reg_pc")
                busy = read_path("u_npu.npu_busy")
                mem_valid = read_path("u_cpu.cpu_mem_valid")
                mem_ready = read_path("u_cpu.cpu_mem_ready")
                mem_addr = read_path("u_cpu.cpu_mem_addr")
                mem_wstrb = read_path("u_cpu.cpu_mem_wstrb")
                periph_req = read_path("periph_req")
                periph_write = read_path("periph_write")
                periph_addr = read_path("periph_addr")
                csr_req = read_path("npu_csr_req")
                csr_write = read_path("npu_csr_write")
                csr_addr = read_path("npu_csr_addr")
                csr_rdata = read_path("npu_csr_rdata")
                pc_str = f"0x{pc:08X}" if isinstance(pc, int) else pc
                mem_addr_str = f"0x{mem_addr:08X}" if isinstance(mem_addr, int) else mem_addr
                periph_addr_str = f"0x{periph_addr:08X}" if isinstance(periph_addr, int) else periph_addr
                csr_addr_str = f"0x{csr_addr:03X}" if isinstance(csr_addr, int) else csr_addr
                csr_rdata_str = f"0x{csr_rdata:08X}" if isinstance(csr_rdata, int) else csr_rdata
                log.info(
                    f"[HANGDIAG] cycle {cyc+1}: pc={pc_str} npu_busy={busy} "
                    f"mem_valid={mem_valid} mem_ready={mem_ready} "
                    f"mem_addr={mem_addr_str} mem_wstrb={mem_wstrb} "
                    f"periph_req={periph_req} periph_write={periph_write} "
                    f"periph_addr={periph_addr_str} csr_req={csr_req} "
                    f"csr_write={csr_write} csr_addr={csr_addr_str} csr_rdata={csr_rdata_str}"
                )
    cocotb.start_soon(hang_diag())

    # ── Release reset — PicoRV32 starts executing firmware ────────────
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    # ── Diagnostic dump post-reset ─────────────────────────────────────
    await dump_diagnostics(dut, "post-reset")

    # ═══════════════════════════════════════════════════════════════════
    # ADDRESS TRANSLATION SANITY CHECK (inline — symbols not yet loaded)
    # ═══════════════════════════════════════════════════════════════════
    # We know from the diagnostic that the bridge translates
    # dma_axi_araddr=0x40000090 → axi_rd_addr=868.
    # Expected: (0x40000090 - 0x40000000) >> 2 = 0x24 = 36.
    # Let's verify what ext_mem actually has at words 36 and 868.
    ok36, v36 = _read_ext_mem_word(dut, 36)
    ok868, v868 = _read_ext_mem_word(dut, 868)
    dut._log.info(f"[SANITY] ext_mem word[36]  = 0x{v36:08X} (valid={ok36}) — should be test_A[0..3]: 0x00FFFFFD")
    dut._log.info(f"[SANITY] ext_mem word[868] = 0x{v868:08X} (valid={ok868}) — what bridge actually reads")
    dut._log.info(f"[SANITY] Expected to_word_addr(0x40000090) = 36, Bridge shows 868 "
                  f"(MISMATCH by {(868-36)} words = {(868-36)*4} bytes)")

    # ── Read firmware symbol addresses from ELF (for diagnostics) ─────
    import subprocess
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
    chars = await wait_uart_all(dut, max_chars=64, timeout_cycles=4000000,
                                idle_cycles=100000, log=dut._log)

    dut._log.info(f"Total UART chars received: {len(chars)}")
    dut._log.info(f"Raw UART bytes: {' '.join(f'0x{c:02X}' for c in chars)}")

    if len(chars) == 0:
        await dump_diagnostics(dut, "uart-timeout")
        dut._log.error("No UART characters received — firmware may be hung")
        assert False, "No UART output received within timeout"

    # ── Parse stage markers (search stream for uppercase=success) ─────
    stage_names = ["INIT", "A-DMA", "B-DMA", "GEMM", "STORE"]
    stage_upper  = ['I', 'A', 'B', 'G', 'S']
    stage_lower  = ['i', 'a', 'b', 'g', 's']
    stage_results = []

    for i in range(5):
        has_upper = ord(stage_upper[i]) in chars
        has_lower = ord(stage_lower[i]) in chars
        if has_upper:
            ok = True
            c = stage_upper[i]
        elif has_lower:
            ok = False
            c = stage_lower[i]
        else:
            ok = False
            c = '<none>'
        stage_results.append((stage_names[i], c, ok))
        if c == '<none>':
            dut._log.info(f"  Stage {stage_names[i]}: <no output> — MISSING")
        else:
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

    # ── Parse result (find 'P' or 'F' in stream) ─────────────────────
    result_char = None
    f_pos = None
    for i, c in enumerate(chars):
        if c == ord('P'):
            result_char = c
            break
        if c == ord('F'):
            result_char = c
            f_pos = i
            break

    if result_char is not None:
        if chr(result_char) == 'P':
            dut._log.info("Comparison result: PASS (0 mismatches)")
        elif chr(result_char) == 'F':
            # Parse mismatch details
            if f_pos is not None and f_pos + 1 < len(chars):
                mismatch_count = chars[f_pos + 1]
                dut._log.info(f"Comparison result: FAIL — "
                              f"{mismatch_count} mismatches (reported, may be capped at 255)")

                # Parse up to 4 mismatch details (each: 1 idx + 2 hw + 2 gold = 5 bytes)
                pos = f_pos + 2
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

    if os.getenv("E2E_MEM_DIAG") == "1":
        def read_mem_words(mem_obj, count, start=0):
            vals = []
            for idx in range(count):
                try:
                    vals.append(int(mem_obj[start + idx].value) & 0xFFFFFFFF)
                except Exception as exc:
                    vals.append(f"ERR:{exc}")
            return vals

        def log_words(label, vals):
            formatted = " ".join(
                f"{v:08X}" if isinstance(v, int) else str(v)
                for v in vals
            )
            dut._log.info(f"[MEMDIAG] {label}: {formatted}")

        try:
            log_words("ASRAM words[0:8]",
                      read_mem_words(dut.u_npu.u_crossbar.u_asram.mem, 8))
            log_words("WSRAM words[0:8]",
                      read_mem_words(dut.u_npu.u_crossbar.u_wsram.mem, 8))
            log_words("WSRAM words[32:40]",
                      read_mem_words(dut.u_npu.u_crossbar.u_wsram.mem, 8, 32))
            log_words("OSRAM words[0:8]",
                      read_mem_words(dut.u_npu.u_crossbar.u_osram.mem, 8))
            osram_words = read_mem_words(dut.u_npu.u_crossbar.u_osram.mem, 128)
            if all(isinstance(v, int) for v in osram_words):
                import struct
                osram_bytes = bytearray()
                for word in osram_words:
                    osram_bytes.extend(word.to_bytes(4, 'little'))
                osram_mismatches = 0
                col_mismatches = [0] * 16
                for i in range(256):
                    hw_val = struct.unpack_from('<h', osram_bytes, i * 2)[0]
                    gold_val = struct.unpack_from('<h', golden_bytes, i * 2)[0]
                    if hw_val != gold_val:
                        col_mismatches[i % 16] += 1
                        if osram_mismatches < 8:
                            dut._log.info(
                                "[MEMDIAG] OSRAM mismatch[%d] row=%d col=%d hw=%d golden=%d",
                                i, i // 16, i % 16, hw_val, gold_val,
                            )
                        osram_mismatches += 1
                dut._log.info("[MEMDIAG] OSRAM mismatches: %d/256", osram_mismatches)
                dut._log.info("[MEMDIAG] OSRAM mismatch columns: %s", col_mismatches)
        except Exception as exc:
            dut._log.warning(f"[MEMDIAG] SRAM read failed: {exc}")

        try:
            dut._log.info(
                "[MEMDIAG] gemm_issue_cmd_latched=0x%08X gemm_k_count=%d",
                int(dut.u_npu.gemm_issue_cmd_latched.value) & 0xFFFFFFFF,
                int(dut.u_npu.gemm_k_count.value) & 0xFF,
            )
            gpl_a0 = int(dut.u_npu.gpl_a_row[0].value)
            gpl_b0 = int(dut.u_npu.gpl_b_row[0].value)
            log_words("GPL A row0 words",
                      [(gpl_a0 >> (32 * i)) & 0xFFFFFFFF for i in range(4)])
            log_words("GPL B row0 words",
                      [(gpl_b0 >> (32 * i)) & 0xFFFFFFFF for i in range(4)])
            gpl_a8 = int(dut.u_npu.gpl_a_row[8].value)
            gpl_b8 = int(dut.u_npu.gpl_b_row[8].value)
            log_words("GPL A row8 words",
                      [(gpl_a8 >> (32 * i)) & 0xFFFFFFFF for i in range(4)])
            log_words("GPL B row8 words",
                      [(gpl_b8 >> (32 * i)) & 0xFFFFFFFF for i in range(4)])
            gpl_a15 = int(dut.u_npu.gpl_a_row[15].value)
            gpl_b15 = int(dut.u_npu.gpl_b_row[15].value)
            log_words("GPL A row15 words",
                      [(gpl_a15 >> (32 * i)) & 0xFFFFFFFF for i in range(4)])
            log_words("GPL B row15 words",
                      [(gpl_b15 >> (32 * i)) & 0xFFFFFFFF for i in range(4)])
        except Exception as exc:
            dut._log.warning(f"[MEMDIAG] GPL buffer read failed: {exc}")

    # ── Final assertion ───────────────────────────────────────────────
    # The test passes only if all stage markers are uppercase AND
    # the result is 'P'.
    all_stages_ok = all(ok for _, _, ok in stage_results)
    result_ok = (result_char is not None and chr(result_char) == 'P')

    pass_str = (
        f"Stages: {'ALL-OK' if all_stages_ok else 'FAIL at ' + (first_fail or '?')}, "
        f"Result: {'PASS' if result_ok else 'FAIL'}"
    )

    assert all_stages_ok and result_ok, (
        f"E2E GEMM test failed. {pass_str}. "
        f"UART sequence: {' '.join(f'0x{c:02X}' for c in chars)}"
    )

    dut._log.info(f"E2E GEMM: FULL PASS — {pass_str}")
