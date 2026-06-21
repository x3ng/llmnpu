"""test_npu_top.py — Minimal integration test for NPU top-level.

Tests:
  1. CSR read/write: write CTRL register, read back STATUS and PERF_CYCLE
  2. VALU VADD via IF/ID dispatch: load instruction, execute, verify result
  3. IRQ on done: enable IRQ, verify irq_stat[0] set when NPU goes idle
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles


# ---------- Helpers ----------

def pack_bytes(bytes_list):
    """Pack 64 bytes into a 512-bit integer. Lane 0 -> bits [7:0]."""
    result = 0
    for i, b in enumerate(bytes_list):
        result |= ((b & 0xFF) << (i * 8))
    return result


def unpack_bytes(value, count=64):
    """Unpack an integer into a list of bytes (little-endian per lane)."""
    v = int(value)
    return [(v >> (i * 8)) & 0xFF for i in range(count)]


# CSR word addresses (byte addr >> 2)
CSR_CTRL      = 0x00
CSR_STATUS    = 0x01
CSR_PC        = 0x02
CSR_DESC_PTR  = 0x04
CSR_DMA_CSR0  = 0x08
CSR_DMA_CSR1  = 0x0A
CSR_DMA_CSR2  = 0x0C
CSR_DMA_CSR3  = 0x0E
CSR_IRQ_EN    = 0x10
CSR_IRQ_STAT  = 0x11
CSR_DEBUG     = 0x18
CSR_PERF_CYC  = 0x20
CSR_PERF_BUSY = 0x21
CSR_PERF_GEMM = 0x22
CSR_PERF_VALU = 0x23
CSR_PERF_SFU  = 0x24
CSR_PERF_DMA  = 0x25

OP_GEMM       = 0x01
OP_VADD       = 0x10
OP_ACT_RELU   = 0x20
OP_QUANT      = 0x30
OP_DMA_LD     = 0x40
OP_DMA_ST     = 0x41
OP_SYNC       = 0xF0
OP_WFI        = 0xF1

DMA_CSR3_START   = 1 << 0
DMA_CSR3_DIR_ST  = 1 << 1
DMA_CSR3_MODE_2D = 1 << 2

CSR_CTRL_START = 1 << 0
CSR_CTRL_RESET = 1 << 1
CSR_CTRL_HALT  = 1 << 2

IRQ_DONE     = 1 << 0
IRQ_DMA_ERR  = 1 << 1
IRQ_ILL_INSN = 1 << 2

DEBUG_GPL_STATE_SHIFT = 5
DEBUG_GPL_STATE_MASK = 0x7 << DEBUG_GPL_STATE_SHIFT

DSRAM_BASE = 0x3000
ASRAM_BASE = 0x0000
WSRAM_BASE = 0x1000
OSRAM_BASE = 0x2000
PP_GEMM_A_SIZE = 256
PP_GEMM_B_SIZE = 256
PP_GEMM_P_SIZE = 512


def ctrl_start(opcode):
    return 0x00000001 | ((opcode & 0xFF) << 8)


async def csr_write(dut, addr_word, value):
    """Write a 32-bit value to a CSR register (word address)."""
    dut.csr_addr.value = addr_word << 2
    dut.csr_wdata.value = value
    dut.csr_we.value = 1
    await RisingEdge(dut.clk)
    dut.csr_we.value = 0
    dut.csr_addr.value = 0
    dut.csr_wdata.value = 0


async def csr_read(dut, addr_word):
    """Read a 32-bit value from a CSR register (word address)."""
    dut.csr_addr.value = addr_word << 2
    dut.csr_re.value = 1
    await Timer(1, unit="ns")
    val = int(dut.csr_rdata.value)
    dut.csr_re.value = 0
    dut.csr_addr.value = 0
    return val


async def wait_csr_idle_after_busy(dut, timeout_cycles=500):
    busy_seen = False
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        busy = (await csr_read(dut, CSR_STATUS)) & 1
        if busy:
            busy_seen = True
        if busy_seen and not busy:
            return
    raise AssertionError("CSR busy did not assert and return idle")


async def start_dma_load_and_collect_copy_addrs(dut, ext_addr, sram_addr,
                                                length, timeout_cycles=500):
    await csr_write(dut, CSR_DMA_CSR0, ext_addr)
    await csr_write(dut, CSR_DMA_CSR1, sram_addr)
    await csr_write(dut, CSR_DMA_CSR2, length)
    await csr_write(dut, CSR_DMA_CSR3, DMA_CSR3_START)

    busy_seen = False
    copy_addrs = []
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.m0_wen.value) and int(dut.xbar_m0_grant.value):
            copy_addrs.append(int(dut.m0_addr.value) & 0xFFFF)
        busy = (await csr_read(dut, CSR_STATUS)) & 1
        if busy:
            busy_seen = True
        if busy_seen and not busy:
            return copy_addrs
    raise AssertionError("DMA LOAD did not assert busy and return idle")


async def start_dma_store_and_collect_prefill_addrs(dut, ext_addr, sram_addr,
                                                    length, timeout_cycles=700):
    await csr_write(dut, CSR_DMA_CSR0, ext_addr)
    await csr_write(dut, CSR_DMA_CSR1, sram_addr)
    await csr_write(dut, CSR_DMA_CSR2, length)
    await csr_write(dut, CSR_DMA_CSR3, DMA_CSR3_DIR_ST | DMA_CSR3_START)

    busy_seen = False
    dma_done_seen = False
    prefill_addrs = []
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.m0_req.value) and int(dut.xbar_m0_grant.value) and not int(dut.m0_wen.value):
            prefill_addrs.append(int(dut.m0_addr.value) & 0xFFFF)
        if prefill_addrs and int(dut.dma_done.value):
            dma_done_seen = True
        busy = (await csr_read(dut, CSR_STATUS)) & 1
        if busy:
            busy_seen = True
        if busy_seen and dma_done_seen and not busy:
            return prefill_addrs
    raise AssertionError(
        f"DMA STORE did not complete: busy_seen={busy_seen} "
        f"dma_done_seen={dma_done_seen} prefill_words={len(prefill_addrs)}"
    )


async def wait_gemm_idle_after_busy(dut, timeout_cycles=2000):
    busy_seen = False
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        busy = (await csr_read(dut, CSR_STATUS)) & 1
        if busy:
            busy_seen = True
        if busy_seen and not busy:
            return
    status = await csr_read(dut, CSR_STATUS)
    raise AssertionError(
        f"GEMM did not complete: busy_seen={busy_seen} "
        f"status=0x{status:08X} gpl_state={int(dut.gpl_state.value)} "
        f"gemm_busy={int(dut.gemm_busy.value)} wb_active={int(dut.gemm_wb_active.value)}"
    )


async def if_load(dut, addr, instr):
    """Load a 32-bit instruction into IF/ID imem at given address."""
    dut.dbg_imem_we.value = 1
    dut.dbg_imem_addr.value = addr
    dut.dbg_imem_wdata.value = instr
    await RisingEdge(dut.clk)
    dut.dbg_imem_we.value = 0


async def valu_write_reg(dut, reg_idx, byte_list):
    """Write 64 bytes to a VALU register file entry."""
    dut.dbg_valu_wen.value = 1
    dut.dbg_valu_waddr.value = reg_idx
    dut.dbg_valu_wdata_flat.value = pack_bytes(byte_list)
    await RisingEdge(dut.clk)
    dut.dbg_valu_wen.value = 0


async def valu_read_reg(dut, reg_idx):
    """Read 64 bytes from a VALU register file entry."""
    dut.dbg_valu_raddr.value = reg_idx
    await Timer(1, unit="ns")
    val = int(dut.dbg_valu_rdata_flat.value)
    dut.dbg_valu_raddr.value = 0
    return unpack_bytes(val)


async def respond_axi_read_burst(dut, expected_addr, words):
    """Respond to one 64-bit AXI read burst with little-endian 32-bit words."""
    assert len(words) == 32, "IF refill expects exactly 32 instruction words"

    for _ in range(80):
        if int(dut.dma_axi_arvalid.value):
            break
        await RisingEdge(dut.clk)
    else:
        raise AssertionError("AXI read address was not issued for IF refill")

    araddr = int(dut.dma_axi_araddr.value)
    arlen = int(dut.dma_axi_arlen.value)
    assert araddr == expected_addr, (
        f"IF refill ARADDR mismatch: got 0x{araddr:08X}, "
        f"expected 0x{expected_addr:08X}"
    )
    assert arlen == 15, f"IF refill ARLEN should be 15 for 16 beats, got {arlen}"

    dut.dma_axi_arready.value = 1
    await RisingEdge(dut.clk)
    dut.dma_axi_arready.value = 0

    for beat in range(16):
        await Timer(1, unit="ps")
        while not int(dut.dma_axi_rready.value):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ps")

        lo = words[2 * beat] & 0xFFFFFFFF
        hi = words[2 * beat + 1] & 0xFFFFFFFF
        dut.dma_axi_rdata.value = (hi << 32) | lo
        dut.dma_axi_rresp.value = 0
        dut.dma_axi_rlast.value = int(beat == 15)
        dut.dma_axi_rvalid.value = 1
        await RisingEdge(dut.clk)
        dut.dma_axi_rvalid.value = 0
        dut.dma_axi_rlast.value = 0
        await Timer(1, unit="ps")


def dsram_write_word(dut, byte_addr, value):
    """Write a 32-bit word directly into DSRAM for focused top-level tests."""
    word_idx = (byte_addr - DSRAM_BASE) >> 2
    dut.u_crossbar.u_dsram.mem[word_idx].value = value & 0xFFFFFFFF


def _pack_word_i8(vals):
    word = 0
    for i, val in enumerate(vals):
        word |= ((int(val) & 0xFF) << (8 * i))
    return word


def _sat16(val):
    return max(-32768, min(32767, int(val)))


def asram_write_word(dut, byte_addr, value):
    word_idx = (byte_addr - ASRAM_BASE) >> 2
    dut.u_crossbar.u_asram.mem[word_idx].value = value & 0xFFFFFFFF


def wsram_write_word(dut, byte_addr, value):
    word_idx = (byte_addr - WSRAM_BASE) >> 2
    dut.u_crossbar.u_wsram.mem[word_idx].value = value & 0xFFFFFFFF


def osram_read_word(dut, byte_addr):
    word_idx = (byte_addr - OSRAM_BASE) >> 2
    return int(dut.u_crossbar.u_osram.mem[word_idx].value) & 0xFFFFFFFF


def osram_write_word(dut, byte_addr, value):
    word_idx = (byte_addr - OSRAM_BASE) >> 2
    dut.u_crossbar.u_osram.mem[word_idx].value = value & 0xFFFFFFFF


def dma_ext_write_byte(dut, byte_addr, value):
    dut.u_dma.wrapper.axi_ram[byte_addr].value = value & 0xFF


def dma_ext_read_byte(dut, byte_addr):
    return int(dut.u_dma.wrapper.axi_ram[byte_addr].value) & 0xFF


def dma_ext_write_i8_tile(dut, byte_addr, matrix):
    for r in range(16):
        for c in range(16):
            dma_ext_write_byte(dut, byte_addr + r * 16 + c, matrix[r][c])


def dma_ext_read_word(dut, byte_addr):
    result = 0
    for i in range(4):
        result |= dma_ext_read_byte(dut, byte_addr + i) << (8 * i)
    return result


def check_dma_ext_i16_tile_constant(dut, byte_addr, expected):
    mismatches = []
    for r in range(16):
        for c in range(16):
            word_addr = byte_addr + r * 32 + (c // 2) * 4
            word = dma_ext_read_word(dut, word_addr)
            raw = (word >> (16 * (c & 1))) & 0xFFFF
            got = raw if raw < 0x8000 else raw - 0x10000
            if got != expected:
                mismatches.append((r, c, got))
    return mismatches


def write_gemm_constant_tile(dut, a_base, b_base, a_value, b_value):
    a_word = _pack_word_i8([a_value] * 4)
    b_word = _pack_word_i8([b_value] * 4)
    for r in range(16):
        for w in range(4):
            asram_write_word(dut, a_base + r * 16 + w * 4, a_word)
    for k in range(16):
        for w in range(4):
            wsram_write_word(dut, b_base + k * 16 + w * 4, b_word)


def dma_sram_read_word(dut, byte_addr):
    result = 0
    for i in range(4):
        result |= (int(dut.u_dma.sram[byte_addr + i].value) & 0xFF) << (8 * i)
    return result


async def setup_dut(dut):
    """Start clock and initialize all inputs."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # Drive all inputs low
    dut.rst_n.value = 0
    dut.csr_addr.value = 0
    dut.csr_wdata.value = 0
    dut.csr_we.value = 0
    dut.csr_re.value = 0
    dut.dbg_imem_we.value = 0
    dut.dbg_imem_addr.value = 0
    dut.dbg_imem_wdata.value = 0
    dut.dbg_valu_wen.value = 0
    dut.dbg_valu_waddr.value = 0
    dut.dbg_valu_wdata_flat.value = 0
    dut.dbg_valu_raddr.value = 0
    dut.dma_axi_arready.value = 0
    dut.dma_axi_rdata.value = 0
    dut.dma_axi_rresp.value = 0
    dut.dma_axi_rlast.value = 0
    dut.dma_axi_rvalid.value = 0
    dut.dma_axi_awready.value = 0
    dut.dma_axi_wready.value = 0
    dut.dma_axi_bresp.value = 0
    dut.dma_axi_bvalid.value = 0

    await ClockCycles(dut.clk, 3)


# =========================================================================


@cocotb.test()
async def test_csr_rw(dut):
    """Test 1: CSR register read/write.

    Write CTRL, read back STATUS, verify PERF_CYCLE is incrementing.
    """
    await setup_dut(dut)

    # Release external reset
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Read STATUS at reset — should be 0
    status = await csr_read(dut, CSR_STATUS)
    assert status == 0, f"STATUS after reset: 0x{status:08X} (expected 0)"

    # Read PERF_CYCLE, wait a few cycles, read again
    pc1 = await csr_read(dut, CSR_PERF_CYC)
    await ClockCycles(dut.clk, 5)
    pc2 = await csr_read(dut, CSR_PERF_CYC)
    assert pc2 > pc1, f"PERF_CYCLE not incrementing: {pc1} -> {pc2}"

    dut._log.info(f"PASS: CSR R/W — PERF_CYCLE {pc1} → {pc2}")


@cocotb.test()
async def test_perf_counters_track_valu_busy(dut):
    """Performance counters track aggregate and per-engine busy cycles."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    perf_regs = [
        CSR_PERF_CYC,
        CSR_PERF_BUSY,
        CSR_PERF_GEMM,
        CSR_PERF_VALU,
        CSR_PERF_SFU,
        CSR_PERF_DMA,
    ]
    for reg in perf_regs:
        await csr_write(dut, reg, 0)

    vadd_instr = (OP_VADD << 24) | (0x00 << 20) | (1 << 16) | (2 << 8) | 3
    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    await if_load(dut, 0, vadd_instr)
    await if_load(dut, 1, (OP_WFI << 24))
    await valu_write_reg(dut, 2, [3] * 64)
    await valu_write_reg(dut, 3, [4] * 64)
    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)

    busy_seen = False
    for _ in range(80):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        status = await csr_read(dut, CSR_STATUS)
        if status & 1:
            busy_seen = True
        if busy_seen and not (status & 1):
            break

    assert busy_seen, "VALU command never made NPU busy"

    cycle_count = await csr_read(dut, CSR_PERF_CYC)
    busy_count = await csr_read(dut, CSR_PERF_BUSY)
    gemm_count = await csr_read(dut, CSR_PERF_GEMM)
    valu_count = await csr_read(dut, CSR_PERF_VALU)
    sfu_count = await csr_read(dut, CSR_PERF_SFU)
    dma_count = await csr_read(dut, CSR_PERF_DMA)

    assert cycle_count > 0, "PERF_CYCLE did not increment"
    assert busy_count > 0, "PERF_BUSY did not count VALU activity"
    assert valu_count > 0, "PERF_VALU did not count VALU activity"
    assert gemm_count == 0, f"PERF_GEMM changed during VALU-only op: {gemm_count}"
    assert sfu_count == 0, f"PERF_SFU changed during VALU-only op: {sfu_count}"
    assert dma_count == 0, f"PERF_DMA changed during VALU-only op: {dma_count}"

    await csr_write(dut, CSR_PERF_VALU, 0)
    assert await csr_read(dut, CSR_PERF_VALU) == 0, "PERF_VALU write-clear failed"

    dut._log.info(
        f"PASS: perf counters cycle={cycle_count} busy={busy_count} valu={valu_count}"
    )


@cocotb.test()
async def test_csr_start_opcode_gates_gemm(dut):
    """CSR START only enters GEMM preloader when CTRL[15:8] is OP_GEMM."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    # Hold datapath reset while loading NOPs so IF/ID cannot dispatch X data.
    await csr_write(dut, CSR_CTRL, 0x00000002)
    for i in range(4):
        await if_load(dut, i, 0xFF000000)
    await csr_write(dut, CSR_CTRL, 0x00000000)
    await ClockCycles(dut.clk, 2)

    # Non-GEMM opcode must not be interpreted as implicit GEMM.
    await csr_write(dut, CSR_CTRL, ctrl_start(OP_ACT_RELU))
    await ClockCycles(dut.clk, 8)
    status = await csr_read(dut, CSR_STATUS)
    debug = await csr_read(dut, CSR_DEBUG)
    gpl_state = (debug & DEBUG_GPL_STATE_MASK) >> DEBUG_GPL_STATE_SHIFT
    assert (status & 1) == 0, f"non-GEMM CSR start made NPU busy: status=0x{status:08X}"
    assert gpl_state == 0, f"non-GEMM CSR start entered GPL state {gpl_state}"

    # Re-assert datapath reset, then start with OP_GEMM.  This should still
    # reach the GEMM preload path even though data contents are just zeros.
    await csr_write(dut, CSR_CTRL, 0x00000002)
    await ClockCycles(dut.clk, 2)
    await csr_write(dut, CSR_CTRL, 0x00000000)
    await ClockCycles(dut.clk, 2)
    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))

    busy_seen = False
    for _ in range(50):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        status = await csr_read(dut, CSR_STATUS)
        debug = await csr_read(dut, CSR_DEBUG)
        gpl_state = (debug & DEBUG_GPL_STATE_MASK) >> DEBUG_GPL_STATE_SHIFT
        if (status & 1) or gpl_state != 0:
            busy_seen = True
            break

    assert busy_seen, "OP_GEMM CSR start did not enter GEMM path"
    dut._log.info("PASS: CSR START opcode gates GEMM issue path")


@cocotb.test()
async def test_csr_gemm_fetches_descriptor(dut):
    """CSR GEMM issue reads GemmDesc from DSRAM at CSR_DESC_PTR."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, 0x00000002)
    for i in range(4):
        await if_load(dut, i, 0xFF000000)

    # GemmDesc little-endian words:
    # word0: M=1, N=1
    # word1: K=2 tiles, A bank=0, W bank=1
    # word2: O bank=2, zero-points default 0
    dsram_write_word(dut, DSRAM_BASE + 0, 0x00010001)
    dsram_write_word(dut, DSRAM_BASE + 4, 0x01000002)
    dsram_write_word(dut, DSRAM_BASE + 8, 0x00000002)
    dsram_write_word(dut, DSRAM_BASE + 12, 0x00000000)
    dsram_write_word(dut, DSRAM_BASE + 16, 0x00000000)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_DESC_PTR, DSRAM_BASE)
    await csr_write(dut, CSR_CTRL, 0x00000000)
    await ClockCycles(dut.clk, 2)
    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))

    entered_preload = False
    for _ in range(40):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        state = int(dut.gpl_state.value)
        if state in (1, 2, 3, 4, 5):
            entered_preload = True
            break

    assert entered_preload, "CSR GEMM did not leave descriptor fetch for preload"
    assert int(dut.gpl_k_count.value) == 32, (
        "descriptor K=2 tiles produced "
        f"k_count={int(dut.gpl_k_count.value)} "
        f"dsram[1]=0x{int(dut.u_crossbar.u_dsram.mem[1].value):08X} "
        f"csr_desc=0x{int(dut.csr_desc_ptr.value):04X} "
        f"latched=0x{int(dut.gpl_desc_ptr_latched.value):04X} "
        f"bases=0x{int(dut.gpl_a_base.value):04X}/"
        f"0x{int(dut.gpl_b_base.value):04X}/"
        f"0x{int(dut.gpl_o_base.value):04X}"
    )
    assert int(dut.gpl_a_base.value) == 0x0000
    assert int(dut.gpl_b_base.value) == 0x1000
    assert int(dut.gpl_o_base.value) == 0x2000

    dut._log.info("PASS: CSR GEMM consumes DSRAM descriptor")


@cocotb.test()
async def test_ifid_gemm_uses_descriptor_ref(dut):
    """IF/ID GEMM instruction fetches GemmDesc at CSR_DESC_PTR + desc_ref."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, 0x00000002)

    # Base descriptor is K=1; the instruction references the second
    # descriptor at +5 words, which is K=2.  This catches using the CSR base
    # directly instead of the instruction descriptor reference.
    dsram_write_word(dut, DSRAM_BASE + 0, 0x00010001)
    dsram_write_word(dut, DSRAM_BASE + 4, 0x01000001)
    dsram_write_word(dut, DSRAM_BASE + 8, 0x00000002)
    dsram_write_word(dut, DSRAM_BASE + 12, 0x00000000)
    dsram_write_word(dut, DSRAM_BASE + 16, 0x00000000)

    desc_ref_words = 5
    desc_addr = DSRAM_BASE + desc_ref_words * 4
    dsram_write_word(dut, desc_addr + 0, 0x00010001)
    dsram_write_word(dut, desc_addr + 4, 0x01000002)
    dsram_write_word(dut, desc_addr + 8, 0x00000002)
    dsram_write_word(dut, desc_addr + 12, 0x00000000)
    dsram_write_word(dut, desc_addr + 16, 0x00000000)

    await if_load(dut, 0, (OP_GEMM << 24) | (desc_ref_words << 8))
    await if_load(dut, 1, 0xFF000000)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_DESC_PTR, DSRAM_BASE)
    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)

    entered_preload = False
    for _ in range(60):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        state = int(dut.gpl_state.value)
        if state in (1, 2, 3, 4, 5):
            entered_preload = True
            break

    assert entered_preload, "IF/ID GEMM did not enter descriptor preload"
    assert int(dut.gpl_desc_ptr_latched.value) == desc_addr
    assert int(dut.gpl_k_count.value) == 32, (
        f"IF/ID desc_ref did not select K=2 descriptor: "
        f"k_count={int(dut.gpl_k_count.value)} "
        f"latched=0x{int(dut.gpl_desc_ptr_latched.value):04X}"
    )
    dut._log.info("PASS: IF/ID GEMM descriptor reference selects K=2 descriptor")


@cocotb.test()
async def test_ifid_sync_waits_for_gemm_path(dut):
    """SYNC holds later IF/ID instructions until GEMM preload/compute/WB is idle."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, 0x00000002)

    dsram_write_word(dut, DSRAM_BASE + 0, 0x00010001)  # M=1, N=1
    dsram_write_word(dut, DSRAM_BASE + 4, 0x01000001)  # K=1, A=0, B=1
    dsram_write_word(dut, DSRAM_BASE + 8, 0x00000002)  # O=2
    dsram_write_word(dut, DSRAM_BASE + 12, 0x00000000)
    dsram_write_word(dut, DSRAM_BASE + 16, 0x00000000)

    await valu_write_reg(dut, 1, [1] * 64)
    await valu_write_reg(dut, 2, [2] * 64)

    await if_load(dut, 0, (OP_GEMM << 24))
    await if_load(dut, 1, (OP_SYNC << 24))
    await if_load(dut, 2, (OP_VADD << 24) | (3 << 16) | (1 << 8) | 2)
    await if_load(dut, 3, (OP_WFI << 24))
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_DESC_PTR, DSRAM_BASE)
    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)

    gemm_path_seen = False
    stall_seen = False
    valu_seen = False
    for _ in range(1600):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        gemm_path_busy = int(dut.ifid_gemm_busy.value)
        valu_valid = int(dut.valu_cmd_valid.value)

        if gemm_path_busy:
            gemm_path_seen = True
        if int(dut.if_stall.value):
            stall_seen = True
        assert not (valu_valid and gemm_path_busy), (
            "SYNC allowed VADD dispatch while GEMM path was still busy"
        )
        if valu_valid:
            valu_seen = True
            break

    assert gemm_path_seen, "GEMM path never became busy"
    assert stall_seen, "SYNC never stalled the IF/ID stream"
    assert valu_seen, "VADD after SYNC did not dispatch after GEMM path completed"
    dut._log.info("PASS: IF/ID SYNC waits for full GEMM path before VADD")


@cocotb.test()
async def test_csr_gemm_k2_computes_from_descriptor(dut):
    """CSR GEMM with descriptor K=2 tiles applies zp, scale, and ReLU."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, 0x00000002)
    for i in range(4):
        await if_load(dut, i, 0xFF000000)

    k_count = 32
    a_mat = [[((r * 3 + k * 5) % 9) - 4 for k in range(k_count)]
             for r in range(16)]
    b_mat = [[((k * 7 + c * 2) % 11) - 5 for c in range(16)]
             for k in range(k_count)]

    for r in range(16):
        for w in range(k_count // 4):
            vals = a_mat[r][w * 4:(w + 1) * 4]
            asram_write_word(dut, ASRAM_BASE + r * k_count + w * 4,
                             _pack_word_i8(vals))

    for k in range(k_count):
        for w in range(4):
            vals = b_mat[k][w * 4:(w + 1) * 4]
            wsram_write_word(dut, WSRAM_BASE + k * 16 + w * 4,
                             _pack_word_i8(vals))

    dsram_write_word(dut, DSRAM_BASE + 0, 0x00010001)  # M=1, N=1
    dsram_write_word(dut, DSRAM_BASE + 4, 0x01000002)  # K=2, A=0, B=1
    dsram_write_word(dut, DSRAM_BASE + 8, 0x00FE0102)  # O=2, a_zp=1, b_zp=-2
    dsram_write_word(dut, DSRAM_BASE + 12, 0x02000100)  # shr=1, mul low=2
    dsram_write_word(dut, DSRAM_BASE + 16, 0x00030100)  # mul high=0, relu=1, out_zp=3
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_DESC_PTR, DSRAM_BASE)
    await csr_write(dut, CSR_CTRL, 0x00000000)
    await ClockCycles(dut.clk, 2)
    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))

    busy_seen = False
    idle_seen = False
    for _ in range(1200):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        busy = (await csr_read(dut, CSR_STATUS)) & 1
        if busy:
            busy_seen = True
        if busy_seen and not busy:
            idle_seen = True
            break

    assert busy_seen, "K=2 CSR GEMM never went busy"
    assert idle_seen, "K=2 CSR GEMM did not finish"

    mismatches = []
    for r in range(16):
        for c in range(16):
            word = osram_read_word(dut, OSRAM_BASE + r * 32 + (c // 2) * 4)
            raw = (word >> (16 * (c & 1))) & 0xFFFF
            got = raw if raw < 0x8000 else raw - 0x10000
            acc = sum((a_mat[r][k] - 1) * (b_mat[k][c] + 2) for k in range(k_count))
            exp = _sat16((acc * 2) >> 1)
            if exp < 0:
                exp = 0
            exp = _sat16(exp + 3)
            if got != exp:
                mismatches.append((r, c, got, exp))

    assert not mismatches, (
        f"K=2 CSR GEMM mismatches: {mismatches[:8]} "
        f"total={len(mismatches)}"
    )
    dut._log.info("PASS: CSR GEMM K=2 descriptor applies zp, scale, and ReLU")


@cocotb.test()
async def test_csr_dma_2d_store_prefill_strided(dut):
    """CSR 2D STORE bridge prefill reads NPU SRAM with row stride."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, 0x00000002)
    for i in range(4):
        await if_load(dut, i, 0xFF000000)
    await csr_write(dut, CSR_CTRL, 0x00000000)
    await ClockCycles(dut.clk, 2)

    rows = 3
    row_bytes = 8
    sram_stride = 16
    ext_stride = 24
    sram_base = OSRAM_BASE
    pattern = [
        0x11111111_00000000,
        0x22222222_00000000,
        0x33333333_00000000,
    ]

    for r, value in enumerate(pattern):
        osram_write_word(dut, sram_base + r * sram_stride + 0, value & 0xFFFFFFFF)
        osram_write_word(dut, sram_base + r * sram_stride + 4, value >> 32)

    await csr_write(dut, CSR_DMA_CSR0, 0x00000800)
    await csr_write(dut, CSR_DMA_CSR1,
                    ((sram_stride & 0xFFFF) << 16) | (sram_base & 0xFFFF))
    await csr_write(dut, CSR_DMA_CSR2,
                    ((rows & 0xFFFF) << 16) | (row_bytes & 0xFFFF))
    await csr_write(dut, CSR_DMA_CSR3,
                    ((ext_stride & 0xFFFF) << 16) |
                    DMA_CSR3_MODE_2D | DMA_CSR3_DIR_ST | DMA_CSR3_START)

    busy_seen = False
    idle_seen = False
    for _ in range(400):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        busy = (await csr_read(dut, CSR_STATUS)) & 1
        if busy:
            busy_seen = True
        if busy_seen and not busy:
            idle_seen = True
            break

    assert busy_seen, "2D STORE never entered busy state"
    assert idle_seen, "2D STORE did not finish"

    for r, value in enumerate(pattern):
        got_lo = dma_sram_read_word(dut, sram_base + r * sram_stride + 0)
        got_hi = dma_sram_read_word(dut, sram_base + r * sram_stride + 4)
        got = got_lo | (got_hi << 32)
        assert got == value, (
            f"2D STORE prefill row {r}: expected 0x{value:016X}, "
            f"got 0x{got:016X}"
        )

    dut._log.info("PASS: CSR DMA 2D STORE bridge prefill uses SRAM stride")


@cocotb.test()
async def test_csr_dma_2d_load_copy_strided(dut):
    """CSR 2D LOAD bridge copy writes crossbar SRAM rows with stride."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, 0x00000002)
    for i in range(4):
        await if_load(dut, i, 0xFF000000)
    await csr_write(dut, CSR_CTRL, 0x00000000)
    await ClockCycles(dut.clk, 2)

    rows = 3
    row_bytes = 8
    ext_stride = 24
    sram_stride = 16
    ext_base = 0x00000900
    sram_base = ASRAM_BASE
    await csr_write(dut, CSR_DMA_CSR0, ext_base)
    await csr_write(dut, CSR_DMA_CSR1,
                    ((sram_stride & 0xFFFF) << 16) | (sram_base & 0xFFFF))
    await csr_write(dut, CSR_DMA_CSR2,
                    ((rows & 0xFFFF) << 16) | (row_bytes & 0xFFFF))
    await csr_write(dut, CSR_DMA_CSR3,
                    ((ext_stride & 0xFFFF) << 16) |
                    DMA_CSR3_MODE_2D | DMA_CSR3_START)

    busy_seen = False
    idle_seen = False
    copy_addrs = []
    for _ in range(500):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.m0_wen.value) and int(dut.xbar_m0_grant.value):
            copy_addrs.append(int(dut.m0_addr.value) & 0xFFFF)
        busy = (await csr_read(dut, CSR_STATUS)) & 1
        if busy:
            busy_seen = True
        if busy_seen and not busy:
            idle_seen = True
            break

    assert busy_seen, "2D LOAD never entered busy state"
    assert idle_seen, "2D LOAD did not finish"
    expected_addrs = [
        sram_base + 0,
        sram_base + 4,
        sram_base + sram_stride + 0,
        sram_base + sram_stride + 4,
        sram_base + 2 * sram_stride + 0,
        sram_base + 2 * sram_stride + 4,
    ]
    assert copy_addrs == expected_addrs, (
        f"2D LOAD bridge copy addresses: expected {expected_addrs}, got {copy_addrs}; "
        f"br_state={int(dut.dma_br_state.value)} "
        f"load_inflight={int(dut.dma_load_inflight.value)} "
        f"dma_done={int(dut.dma_done.value)} "
        f"dma_busy={int(dut.dma_busy.value)} "
        f"opcode=0x{int(dut.dma_opcode_latched.value):02X}"
    )
    assert (sram_base + row_bytes) not in copy_addrs, (
        "2D LOAD bridge wrote the first SRAM stride gap"
    )
    dut._log.info("PASS: CSR DMA 2D LOAD bridge copy uses SRAM stride")


@cocotb.test()
async def test_dma_load_marks_gemm_pingpong_inputs_ready(dut):
    """DMA LOAD completion marks GEMM A/B ping-pong input banks ready."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    await csr_write(dut, CSR_CTRL, 0)

    assert int(dut.pp_gemm_a_ready.value) == 0, "GEMM A ping-pong ready after reset"
    assert int(dut.pp_gemm_b_ready.value) == 0, "GEMM B ping-pong ready after reset"

    await csr_write(dut, CSR_DMA_CSR0, 0x00000A00)
    await csr_write(dut, CSR_DMA_CSR1, ASRAM_BASE)
    await csr_write(dut, CSR_DMA_CSR2, 8)
    await csr_write(dut, CSR_DMA_CSR3, DMA_CSR3_START)
    await wait_csr_idle_after_busy(dut)

    assert int(dut.pp_gemm_a_ready.value) == 1, "ASRAM DMA LOAD did not fill GEMM A ping-pong"
    assert int(dut.pp_gemm_a_active_bank.value) == 0, "first GEMM A fill should select bank 0"

    await csr_write(dut, CSR_DMA_CSR0, 0x00000B00)
    await csr_write(dut, CSR_DMA_CSR1, WSRAM_BASE)
    await csr_write(dut, CSR_DMA_CSR2, 8)
    await csr_write(dut, CSR_DMA_CSR3, DMA_CSR3_START)
    await wait_csr_idle_after_busy(dut)

    assert int(dut.pp_gemm_b_ready.value) == 1, "WSRAM DMA LOAD did not fill GEMM B ping-pong"
    assert int(dut.pp_gemm_b_active_bank.value) == 0, "first GEMM B fill should select bank 0"
    dut._log.info("PASS: DMA LOAD marks GEMM A/B ping-pong inputs ready")


@cocotb.test()
async def test_gemm_pingpong_bank_offsets_drive_dma_and_preloader(dut):
    """GEMM A/B DMA fill and compute consume use ping-pong bank windows."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    for i in range(4):
        await if_load(dut, i, (OP_WFI << 24))
    await csr_write(dut, CSR_CTRL, 0)
    await ClockCycles(dut.clk, 2)

    a0_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, 0x00000A00, ASRAM_BASE, 8)
    assert a0_addrs == [ASRAM_BASE, ASRAM_BASE + 4], (
        f"first ASRAM LOAD should fill A bank0, got {a0_addrs}"
    )
    assert int(dut.pp_gemm_a_active_bank.value) == 0
    assert int(dut.pp_gemm_a_fill_bank.value) == 1

    a1_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, 0x00000A40, ASRAM_BASE, 8)
    assert a1_addrs == [ASRAM_BASE + PP_GEMM_A_SIZE,
                        ASRAM_BASE + PP_GEMM_A_SIZE + 4], (
        f"second ASRAM LOAD should fill A bank1, got {a1_addrs}"
    )

    b0_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, 0x00000B00, WSRAM_BASE, 8)
    assert b0_addrs == [WSRAM_BASE, WSRAM_BASE + 4], (
        f"first WSRAM LOAD should fill B bank0, got {b0_addrs}"
    )
    assert int(dut.pp_gemm_b_active_bank.value) == 0
    assert int(dut.pp_gemm_b_fill_bank.value) == 1

    b1_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, 0x00000B40, WSRAM_BASE, 8)
    assert b1_addrs == [WSRAM_BASE + PP_GEMM_B_SIZE,
                        WSRAM_BASE + PP_GEMM_B_SIZE + 4], (
        f"second WSRAM LOAD should fill B bank1, got {b1_addrs}"
    )

    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))
    await wait_gemm_idle_after_busy(dut, timeout_cycles=2200)

    assert int(dut.pp_gemm_a_active_bank.value) == 1, (
        "GEMM consume did not advance A active bank to bank1"
    )
    assert int(dut.pp_gemm_b_active_bank.value) == 1, (
        "GEMM consume did not advance B active bank to bank1"
    )

    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))

    first_b_read = None
    first_a_read = None
    for _ in range(400):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        state = int(dut.gpl_state.value)
        if int(dut.xbar_m1_grant.value):
            addr = int(dut.m1_addr.value) & 0xFFFF
            if state in (1, 2, 3, 4) and first_b_read is None:
                first_b_read = addr
            elif state == 5 and first_a_read is None:
                first_a_read = addr
                break

    assert first_b_read == WSRAM_BASE + PP_GEMM_B_SIZE, (
        f"second GEMM should preload B active bank1 at "
        f"0x{WSRAM_BASE + PP_GEMM_B_SIZE:04X}, got {first_b_read}"
    )
    assert first_a_read == ASRAM_BASE + PP_GEMM_A_SIZE, (
        f"second GEMM should preload A active bank1 at "
        f"0x{ASRAM_BASE + PP_GEMM_A_SIZE:04X}, got {first_a_read}"
    )
    await wait_gemm_idle_after_busy(dut, timeout_cycles=2200)
    dut._log.info("PASS: GEMM ping-pong bank offsets drive DMA fill and preload")


@cocotb.test()
async def test_gemm_p_pingpong_writeback_and_dma_store_use_active_bank(dut):
    """GEMM writes P fill bank and DMA STORE reads P active bank."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    for i in range(4):
        await if_load(dut, i, (OP_WFI << 24))
    await csr_write(dut, CSR_CTRL, 0)
    await ClockCycles(dut.clk, 2)

    await start_dma_load_and_collect_copy_addrs(dut, 0x00000A00, ASRAM_BASE, 8)
    await start_dma_load_and_collect_copy_addrs(dut, 0x00000A40, ASRAM_BASE, 8)
    await start_dma_load_and_collect_copy_addrs(dut, 0x00000B00, WSRAM_BASE, 8)
    await start_dma_load_and_collect_copy_addrs(dut, 0x00000B40, WSRAM_BASE, 8)

    write_gemm_constant_tile(dut, ASRAM_BASE, WSRAM_BASE, 1, 1)
    write_gemm_constant_tile(
        dut,
        ASRAM_BASE + PP_GEMM_A_SIZE,
        WSRAM_BASE + PP_GEMM_B_SIZE,
        2,
        1,
    )
    dsram_write_word(dut, DSRAM_BASE + 0, 0x00010001)
    dsram_write_word(dut, DSRAM_BASE + 4, 0x01000001)
    dsram_write_word(dut, DSRAM_BASE + 8, 0x00000002)
    dsram_write_word(dut, DSRAM_BASE + 12, 0x01000000)
    dsram_write_word(dut, DSRAM_BASE + 16, 0x00000000)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_DESC_PTR, DSRAM_BASE)
    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))
    await wait_gemm_idle_after_busy(dut, timeout_cycles=2200)

    bank0_word = osram_read_word(dut, OSRAM_BASE)
    assert bank0_word == 0x00100010, (
        f"first GEMM P bank0 word expected 0x00100010, got 0x{bank0_word:08X}"
    )
    assert int(dut.pp_gemm_p_ready.value) == 1
    assert int(dut.pp_gemm_p_active_bank.value) == 0
    assert int(dut.pp_gemm_p_fill_bank.value) == 1

    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))
    await wait_gemm_idle_after_busy(dut, timeout_cycles=2200)

    bank1_word = osram_read_word(dut, OSRAM_BASE + PP_GEMM_P_SIZE)
    assert bank1_word == 0x00200020, (
        f"second GEMM P bank1 word expected 0x00200020, got 0x{bank1_word:08X}"
    )
    assert int(dut.pp_gemm_p_active_bank.value) == 0

    store0_addrs = await start_dma_store_and_collect_prefill_addrs(
        dut, 0x00000C00, OSRAM_BASE, 8)
    assert store0_addrs[:2] == [OSRAM_BASE, OSRAM_BASE + 4], (
        f"first STORE should prefill from P active bank0, got {store0_addrs}"
    )
    got0 = dma_sram_read_word(dut, OSRAM_BASE)
    assert got0 == 0x00100010, (
        f"first STORE copied 0x{got0:08X}, expected P bank0 0x00100010"
    )
    assert int(dut.pp_gemm_p_active_bank.value) == 1

    store1_addrs = await start_dma_store_and_collect_prefill_addrs(
        dut, 0x00000C40, OSRAM_BASE, 8)
    assert store1_addrs[:2] == [
        OSRAM_BASE + PP_GEMM_P_SIZE,
        OSRAM_BASE + PP_GEMM_P_SIZE + 4,
    ], f"second STORE should prefill from P active bank1, got {store1_addrs}"
    got1 = dma_sram_read_word(dut, OSRAM_BASE)
    assert got1 == 0x00200020, (
        f"second STORE copied 0x{got1:08X}, expected P bank1 0x00200020"
    )

    dut._log.info("PASS: GEMM P ping-pong writeback and STORE use active bank")


@cocotb.test()
async def test_tile_flow_dma_load_gemm_dma_store_result(dut):
    """NPU tile flow: DMA LOAD A/B, GEMM, DMA STORE P, verify output."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    for i in range(4):
        await if_load(dut, i, (OP_WFI << 24))
    await csr_write(dut, CSR_CTRL, 0)
    await ClockCycles(dut.clk, 2)

    a_ext = 0x00004000
    b_ext = 0x00005000
    c_ext = 0x00006000
    a_mat = [[1 for _ in range(16)] for _ in range(16)]
    b_mat = [[1 for _ in range(16)] for _ in range(16)]
    dma_ext_write_i8_tile(dut, a_ext, a_mat)
    dma_ext_write_i8_tile(dut, b_ext, b_mat)
    await Timer(1, unit="ps")

    a_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, a_ext, ASRAM_BASE, 256, timeout_cycles=2000)
    b_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, b_ext, WSRAM_BASE, 256, timeout_cycles=2000)
    assert a_addrs[0] == ASRAM_BASE and a_addrs[-1] == ASRAM_BASE + 252, (
        f"A LOAD copy window mismatch: first/last={a_addrs[:1]}/{a_addrs[-1:]}"
    )
    assert b_addrs[0] == WSRAM_BASE and b_addrs[-1] == WSRAM_BASE + 252, (
        f"B LOAD copy window mismatch: first/last={b_addrs[:1]}/{b_addrs[-1:]}"
    )

    dsram_write_word(dut, DSRAM_BASE + 0, 0x00010001)
    dsram_write_word(dut, DSRAM_BASE + 4, 0x01000001)
    dsram_write_word(dut, DSRAM_BASE + 8, 0x00000002)
    dsram_write_word(dut, DSRAM_BASE + 12, 0x01000000)
    dsram_write_word(dut, DSRAM_BASE + 16, 0x00000000)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_DESC_PTR, DSRAM_BASE)
    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))
    await wait_gemm_idle_after_busy(dut, timeout_cycles=2200)

    store_addrs = await start_dma_store_and_collect_prefill_addrs(
        dut, c_ext, OSRAM_BASE, 512, timeout_cycles=2200)
    assert store_addrs[0] == OSRAM_BASE and store_addrs[-1] == OSRAM_BASE + 508, (
        f"STORE prefill window mismatch: first/last={store_addrs[:1]}/{store_addrs[-1:]}"
    )
    staging_word = dma_sram_read_word(dut, OSRAM_BASE)
    assert staging_word == 0x00100010, (
        f"STORE staging SRAM word expected 0x00100010, got 0x{staging_word:08X}"
    )
    await ClockCycles(dut.clk, 4)

    mismatches = []
    for r in range(16):
        for c in range(16):
            byte_addr = c_ext + r * 32 + (c // 2) * 4
            word = dma_ext_read_word(dut, byte_addr)
            raw = (word >> (16 * (c & 1))) & 0xFFFF
            got = raw if raw < 0x8000 else raw - 0x10000
            if got != 16:
                mismatches.append((r, c, got))

    assert not mismatches, (
        f"tile flow output mismatches: {mismatches[:8]} total={len(mismatches)}"
    )
    dut._log.info("PASS: tile flow DMA LOAD -> GEMM -> DMA STORE result")


@cocotb.test()
async def test_tile_flow_gemm_sfu_relu_dma_store_result(dut):
    """NPU postprocess flow: DMA LOAD, GEMM, SFU ReLU on P, DMA STORE."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    for i in range(4):
        await if_load(dut, i, (OP_WFI << 24))
    await csr_write(dut, CSR_CTRL, 0)
    await ClockCycles(dut.clk, 2)

    a_ext = 0x00004400
    b_ext = 0x00005400
    c_ext = 0x00006400
    a_mat = [[-1 for _ in range(16)] for _ in range(16)]
    b_mat = [[1 for _ in range(16)] for _ in range(16)]
    dma_ext_write_i8_tile(dut, a_ext, a_mat)
    dma_ext_write_i8_tile(dut, b_ext, b_mat)
    await Timer(1, unit="ps")

    await start_dma_load_and_collect_copy_addrs(
        dut, a_ext, ASRAM_BASE, 256, timeout_cycles=2000)
    await start_dma_load_and_collect_copy_addrs(
        dut, b_ext, WSRAM_BASE, 256, timeout_cycles=2000)

    dsram_write_word(dut, DSRAM_BASE + 0, 0x00010001)
    dsram_write_word(dut, DSRAM_BASE + 4, 0x01000001)
    dsram_write_word(dut, DSRAM_BASE + 8, 0x00000002)
    dsram_write_word(dut, DSRAM_BASE + 12, 0x01000000)
    dsram_write_word(dut, DSRAM_BASE + 16, 0x00000000)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_DESC_PTR, DSRAM_BASE)
    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))
    await wait_gemm_idle_after_busy(dut, timeout_cycles=2200)

    pre_relu = osram_read_word(dut, OSRAM_BASE)
    assert pre_relu == 0xFFF0FFF0, (
        f"GEMM before ReLU expected 0xFFF0FFF0, got 0x{pre_relu:08X}"
    )

    await csr_write(dut, CSR_CTRL, ctrl_start(OP_ACT_RELU))
    sfu_seen = False
    sfu_done_seen = False
    for _ in range(900):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.sfu_busy.value):
            sfu_seen = True
        if int(dut.sfu_mem_relu_done.value):
            sfu_done_seen = True
        if sfu_seen and not int(dut.sfu_busy.value):
            break

    assert sfu_seen, "SFU ReLU postprocess did not start"
    assert sfu_done_seen, "SFU ReLU postprocess did not complete"
    post_relu = osram_read_word(dut, OSRAM_BASE)
    assert post_relu == 0x00000000, (
        f"SFU ReLU should clamp P word to zero, got 0x{post_relu:08X}"
    )

    await start_dma_store_and_collect_prefill_addrs(
        dut, c_ext, OSRAM_BASE, 512, timeout_cycles=2200)

    mismatches = []
    for r in range(16):
        for c in range(16):
            byte_addr = c_ext + r * 32 + (c // 2) * 4
            word = dma_ext_read_word(dut, byte_addr)
            raw = (word >> (16 * (c & 1))) & 0xFFFF
            if raw != 0:
                mismatches.append((r, c, raw))

    assert not mismatches, (
        f"postprocess tile output mismatches: {mismatches[:8]} total={len(mismatches)}"
    )
    dut._log.info("PASS: tile flow GEMM -> SFU ReLU -> DMA STORE result")


@cocotb.test()
async def test_full_pingpong_two_tile_dma_gemm_store_flow(dut):
    """Two-tile NPU flow keeps A/B/P ping-pong banks distinct end-to-end."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    for i in range(4):
        await if_load(dut, i, (OP_WFI << 24))
    await csr_write(dut, CSR_CTRL, 0)
    await ClockCycles(dut.clk, 2)

    a0_ext = 0x00004600
    b0_ext = 0x00005600
    c0_ext = 0x00006600
    a1_ext = 0x00004800
    b1_ext = 0x00005800
    c1_ext = 0x00006800

    dma_ext_write_i8_tile(dut, a0_ext, [[1 for _ in range(16)] for _ in range(16)])
    dma_ext_write_i8_tile(dut, b0_ext, [[1 for _ in range(16)] for _ in range(16)])
    dma_ext_write_i8_tile(dut, a1_ext, [[2 for _ in range(16)] for _ in range(16)])
    dma_ext_write_i8_tile(dut, b1_ext, [[1 for _ in range(16)] for _ in range(16)])
    await Timer(1, unit="ps")

    a0_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, a0_ext, ASRAM_BASE, 256, timeout_cycles=2000)
    a1_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, a1_ext, ASRAM_BASE, 256, timeout_cycles=2000)
    b0_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, b0_ext, WSRAM_BASE, 256, timeout_cycles=2000)
    b1_addrs = await start_dma_load_and_collect_copy_addrs(
        dut, b1_ext, WSRAM_BASE, 256, timeout_cycles=2000)

    assert a0_addrs[0] == ASRAM_BASE and a0_addrs[-1] == ASRAM_BASE + 252
    assert a1_addrs[0] == ASRAM_BASE + PP_GEMM_A_SIZE
    assert a1_addrs[-1] == ASRAM_BASE + PP_GEMM_A_SIZE + 252
    assert b0_addrs[0] == WSRAM_BASE and b0_addrs[-1] == WSRAM_BASE + 252
    assert b1_addrs[0] == WSRAM_BASE + PP_GEMM_B_SIZE
    assert b1_addrs[-1] == WSRAM_BASE + PP_GEMM_B_SIZE + 252

    dsram_write_word(dut, DSRAM_BASE + 0, 0x00010001)
    dsram_write_word(dut, DSRAM_BASE + 4, 0x01000001)
    dsram_write_word(dut, DSRAM_BASE + 8, 0x00000002)
    dsram_write_word(dut, DSRAM_BASE + 12, 0x01000000)
    dsram_write_word(dut, DSRAM_BASE + 16, 0x00000000)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_DESC_PTR, DSRAM_BASE)
    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))
    await wait_gemm_idle_after_busy(dut, timeout_cycles=2200)

    assert int(dut.pp_gemm_a_active_bank.value) == 1
    assert int(dut.pp_gemm_b_active_bank.value) == 1
    assert int(dut.pp_gemm_p_active_bank.value) == 0
    assert osram_read_word(dut, OSRAM_BASE) == 0x00100010

    await csr_write(dut, CSR_CTRL, ctrl_start(OP_GEMM))
    await wait_gemm_idle_after_busy(dut, timeout_cycles=2200)

    assert osram_read_word(dut, OSRAM_BASE + PP_GEMM_P_SIZE) == 0x00200020
    assert int(dut.pp_gemm_p_active_bank.value) == 0

    store0_addrs = await start_dma_store_and_collect_prefill_addrs(
        dut, c0_ext, OSRAM_BASE, 512, timeout_cycles=2200)
    assert store0_addrs[0] == OSRAM_BASE and store0_addrs[-1] == OSRAM_BASE + 508
    assert int(dut.pp_gemm_p_active_bank.value) == 1

    store1_addrs = await start_dma_store_and_collect_prefill_addrs(
        dut, c1_ext, OSRAM_BASE, 512, timeout_cycles=2200)
    assert store1_addrs[0] == OSRAM_BASE + PP_GEMM_P_SIZE
    assert store1_addrs[-1] == OSRAM_BASE + PP_GEMM_P_SIZE + 508

    c0_mismatches = check_dma_ext_i16_tile_constant(dut, c0_ext, 16)
    c1_mismatches = check_dma_ext_i16_tile_constant(dut, c1_ext, 32)
    assert not c0_mismatches, (
        f"tile0 output mismatches: {c0_mismatches[:8]} total={len(c0_mismatches)}"
    )
    assert not c1_mismatches, (
        f"tile1 output mismatches: {c1_mismatches[:8]} total={len(c1_mismatches)}"
    )
    dut._log.info("PASS: full ping-pong two-tile DMA/GEMM/STORE flow")


@cocotb.test()
async def test_ifid_dma_store_prefills_dma_sram(dut):
    """IF/ID DMA_ST uses the crossbar prefill path before starting STORE."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, 0x00000002)

    sram_base = OSRAM_BASE
    length = 8
    expected = 0x55667788_11223344
    osram_write_word(dut, sram_base + 0, expected & 0xFFFFFFFF)
    osram_write_word(dut, sram_base + 4, expected >> 32)

    await csr_write(dut, CSR_DMA_CSR0, 0x00000800)
    await csr_write(dut, CSR_DMA_CSR1, sram_base)
    await csr_write(dut, CSR_DMA_CSR2, length)

    await if_load(dut, 0, (OP_DMA_ST << 24))
    await if_load(dut, 1, (OP_WFI << 24))
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)

    prefill_writes = 0
    for _ in range(120):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.dma_sim_sram_we.value):
            prefill_writes += 1
        if prefill_writes >= 2:
            break

    assert prefill_writes >= 2, "IF/ID DMA_ST did not prefill DMA SRAM"
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    got_lo = dma_sram_read_word(dut, sram_base + 0)
    got_hi = dma_sram_read_word(dut, sram_base + 4)
    got = got_lo | (got_hi << 32)
    assert got == expected, (
        f"IF/ID DMA_ST prefill expected 0x{expected:016X}, got 0x{got:016X}"
    )
    dut._log.info("PASS: IF/ID DMA_ST prefill copies O-SRAM into DMA SRAM")


@cocotb.test()
async def test_dma_invalid_length_sets_fault_irq(dut):
    """Invalid DMA command raises IRQ_STAT[1] and does not leave NPU busy."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    await csr_write(dut, CSR_IRQ_EN, IRQ_DMA_ERR)
    await csr_write(dut, CSR_IRQ_STAT, 0xFFFFFFFF)
    await csr_write(dut, CSR_CTRL, 0)

    await csr_write(dut, CSR_DMA_CSR0, 0x00000100)
    await csr_write(dut, CSR_DMA_CSR1, ASRAM_BASE)
    await csr_write(dut, CSR_DMA_CSR2, 6)
    await csr_write(dut, CSR_DMA_CSR3, DMA_CSR3_START)

    fault_seen = False
    for _ in range(12):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        irq_stat = await csr_read(dut, CSR_IRQ_STAT)
        if irq_stat & IRQ_DMA_ERR:
            fault_seen = True
            assert not (irq_stat & IRQ_ILL_INSN), (
                f"DMA fault also set illegal-instruction IRQ: 0x{irq_stat:08X}"
            )
            break

    assert fault_seen, "invalid DMA length did not set IRQ_STAT[1]"
    assert int(dut.irq.value) == 1, "invalid DMA fault IRQ output did not assert"

    status = await csr_read(dut, CSR_STATUS)
    assert (status & 1) == 0, f"invalid DMA left NPU busy: status=0x{status:08X}"
    dut._log.info("PASS: invalid DMA length raises fault IRQ")


@cocotb.test()
async def test_illegal_instruction_sets_fault_irq(dut):
    """Unknown IF/ID opcode raises IRQ_STAT[2] fault."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    await if_load(dut, 0, 0xEE000000)
    await if_load(dut, 1, (OP_VADD << 24) | (1 << 16) | (2 << 8) | 3)
    await csr_write(dut, CSR_IRQ_EN, IRQ_ILL_INSN)
    await csr_write(dut, CSR_IRQ_STAT, 0xFFFFFFFF)
    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)

    fault_seen = False
    for _ in range(12):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        irq_stat = await csr_read(dut, CSR_IRQ_STAT)
        if irq_stat & IRQ_ILL_INSN:
            fault_seen = True
            assert not (irq_stat & IRQ_DMA_ERR), (
                f"illegal instruction also set DMA error IRQ: 0x{irq_stat:08X}"
            )
            break

    assert fault_seen, "illegal instruction did not set IRQ_STAT[2]"
    assert int(dut.irq.value) == 1, "fault IRQ output did not assert"

    await ClockCycles(dut.clk, 4)
    assert not int(dut.valu_cmd_valid.value), (
        "IF/ID continued dispatching after illegal instruction fault"
    )
    dut._log.info("PASS: illegal instruction raises fault IRQ")


@cocotb.test()
async def test_csr_start_releases_ifid_pc(dut):
    """IF/ID PC stays stopped after reset until CSR CTRL.START is written."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    vadd_instr = (OP_VADD << 24) | (1 << 16) | (2 << 8) | 3
    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    await if_load(dut, 0, vadd_instr)
    await if_load(dut, 1, (OP_WFI << 24))

    await csr_write(dut, CSR_CTRL, 0)
    for cycle in range(8):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        assert not int(dut.valu_cmd_valid.value), (
            f"IF/ID dispatched before CTRL.START at cycle {cycle}"
        )
        assert int(dut.if_stall.value), "IF/ID should remain stalled before START"

    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)
    dispatched = False
    for _ in range(8):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.valu_cmd_valid.value):
            dispatched = True
            assert int(dut.valu_cmd.value) == vadd_instr
            break

    assert dispatched, "IF/ID did not dispatch after CTRL.START"
    dut._log.info("PASS: CSR CTRL.START releases IF/ID PC")


@cocotb.test()
async def test_csr_pc_sets_ifid_start_address(dut):
    """CSR PC write sets the IF/ID fetch start address."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    vadd_instr = (OP_VADD << 24) | (1 << 16) | (2 << 8) | 3
    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    for i in range(4):
        await if_load(dut, i, (OP_WFI << 24))
    await if_load(dut, 4, vadd_instr)
    await if_load(dut, 5, (OP_WFI << 24))

    await csr_write(dut, CSR_CTRL, 0)
    await csr_write(dut, CSR_PC, 4)
    pc = await csr_read(dut, CSR_PC)
    assert pc == 4, f"CSR_PC readback expected 4, got {pc}"

    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)
    dispatched = False
    for _ in range(8):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.valu_cmd_valid.value):
            dispatched = True
            assert int(dut.valu_cmd.value) == vadd_instr
            break

    assert dispatched, "IF/ID did not dispatch instruction at CSR_PC start address"
    dut._log.info("PASS: CSR PC sets IF/ID start address")


@cocotb.test()
async def test_csr_halt_stalls_ifid_only(dut):
    """CSR CTRL.HALT stops IF/ID dispatch without resetting datapath state."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET | CSR_CTRL_HALT)
    await if_load(dut, 0, (OP_VADD << 24) | (1 << 16) | (2 << 8) | 3)
    await if_load(dut, 1, (OP_WFI << 24))

    await csr_write(dut, CSR_CTRL, CSR_CTRL_HALT)
    for cycle in range(8):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        assert not int(dut.valu_cmd_valid.value), (
            f"VALU dispatched while CSR_CTRL.HALT=1 at cycle {cycle}"
        )
        assert int(dut.if_stall.value), "IF/ID stall should assert while HALT is set"

    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)
    dispatched = False
    for _ in range(8):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.valu_cmd_valid.value):
            dispatched = True
            assert int(dut.valu_cmd.value) == ((OP_VADD << 24) | (1 << 16) | (2 << 8) | 3)
            break

    assert dispatched, "IF/ID did not resume dispatch after CSR_CTRL.HALT cleared"
    dut._log.info("PASS: CSR CTRL.HALT stalls IF/ID dispatch and resumes cleanly")


@cocotb.test()
async def test_ifid_refill_fetches_program_from_axi(dut):
    """IF/ID refills a missing instruction block from external AXI memory."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    instr_base = 0x00000400
    start_pc = 224
    refill_addr = instr_base + (start_pc >> 5) * 128
    vadd_instr = (OP_VADD << 24) | (0x00 << 20) | (1 << 16) | (2 << 8) | 3
    words = [0xFF000000] * 32
    words[0] = vadd_instr
    words[1] = (OP_WFI << 24)

    await valu_write_reg(dut, 2, [11] * 64)
    await valu_write_reg(dut, 3, [24] * 64)
    await valu_write_reg(dut, 1, [0] * 64)
    await csr_write(dut, CSR_DESC_PTR, instr_base)
    await csr_write(dut, CSR_PC, start_pc)

    refill_task = cocotb.start_soon(respond_axi_read_burst(dut, refill_addr, words))
    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)
    await refill_task
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    imem_vadd = int(dut.u_if_id.imem[start_pc].value) & 0xFFFFFFFF
    imem_wfi = int(dut.u_if_id.imem[start_pc + 1].value) & 0xFFFFFFFF
    assert imem_vadd == vadd_instr, (
        f"refilled imem[{start_pc}] got 0x{imem_vadd:08X}, "
        f"expected 0x{vadd_instr:08X}"
    )
    assert imem_wfi == (OP_WFI << 24), (
        f"refilled imem[{start_pc + 1}] got 0x{imem_wfi:08X}, "
        f"expected WFI"
    )

    expected = 35
    for _ in range(120):
        result = await valu_read_reg(dut, 1)
        if result[0] == expected:
            break
        await RisingEdge(dut.clk)
    else:
        raise AssertionError(
            f"IF refill program did not execute VADD, lane0={result[0]} "
            f"pc={int(dut.if_current_pc.value)} "
            f"debug_pc={int(dut.debug_pc.value)} "
            f"debug_instr=0x{int(dut.debug_instr.value):08X} "
            f"stall={int(dut.if_stall.value)} "
            f"block_valid=0x{int(dut.u_if_id.block_valid.value):02X} "
            f"refill_active={int(dut.u_if_id.refill_active.value)} "
            f"refill_idx={int(dut.u_if_id.refill_idx.value)} "
            f"refill_block={int(dut.u_if_id.refill_block.value)} "
            f"ifr_state={int(dut.ifr_state.value)} "
            f"running={int(dut.running.value)} "
            f"valu_busy={int(dut.valu_busy.value)}"
        )

    assert int(dut.if_refill_busy.value) == 0, "IF refill stream stayed busy"
    assert int(dut.ifr_active.value) == 0, "IF refill AXI FSM stayed active"
    dut._log.info("PASS: IF/ID refill fetches AXI program block and dispatches VADD")


@cocotb.test()
async def test_ifid_quant_sfu_busy_covers_valid(dut):
    """IF/ID OP_QUANT keeps SFU busy until the aligned quant valid_out."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    await if_load(dut, 0, (OP_QUANT << 24) | 0x2A)
    await if_load(dut, 1, (OP_WFI << 24))
    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)

    busy_seen = False
    valid_seen = False
    idle_seen = False
    for _ in range(50):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        busy = int(dut.sfu_busy.value)
        valid = int(dut.sfu_valid_out.value)
        if busy:
            busy_seen = True
        if valid:
            valid_seen = True
            assert busy, "SFU valid_out asserted after top-level busy dropped"
        if busy_seen and not busy:
            idle_seen = True
            break

    assert busy_seen, "IF/ID QUANT did not make SFU busy"
    assert valid_seen, "IF/ID QUANT did not produce SFU valid_out"
    assert idle_seen, "SFU did not return idle after IF/ID QUANT"
    dut._log.info("PASS: IF/ID QUANT keeps SFU busy through valid_out")


@cocotb.test()
async def test_ifid_relu_sfu_completes(dut):
    """IF/ID OP_ACT_RELU routes through SFU and returns to idle."""
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    await csr_write(dut, CSR_CTRL, CSR_CTRL_RESET)
    await if_load(dut, 0, (OP_ACT_RELU << 24))
    await if_load(dut, 1, (OP_WFI << 24))
    await csr_write(dut, CSR_CTRL, CSR_CTRL_START)

    busy_seen = False
    valid_seen = False
    idle_seen = False
    for _ in range(40):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.sfu_busy.value):
            busy_seen = True
        if int(dut.sfu_valid_out.value):
            valid_seen = True
        if busy_seen and not int(dut.sfu_busy.value):
            idle_seen = True
            break

    assert busy_seen, "IF/ID ReLU did not make SFU busy"
    assert valid_seen, "IF/ID ReLU did not produce SFU valid_out"
    assert idle_seen, "SFU did not return idle after IF/ID ReLU"
    dut._log.info("PASS: IF/ID ReLU routes through SFU and completes")


@cocotb.test()
async def test_valu_integration(dut):
    """Test 2: VALU VADD via IF/ID dispatch pipeline.

    Flow:
      1. CSR reset=1 (hold IF/ID in reset)
      2. Load VADD instruction into IF/ID imem[0]
      3. Write VALU regfile[2]=10, regfile[3]=25
      4. CSR start=1 (release reset + start)
      5. Wait for VALU to complete (busy → idle)
      6. Read VALU regfile[1] = 10+25 = 35
    """
    await setup_dut(dut)

    # Release external reset (CSR reset still holds dp_rst_n low)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    # ---- Step 1: Assert CSR reset ----
    await csr_write(dut, CSR_CTRL, 0x00000002)  # bit1=reset
    await RisingEdge(dut.clk)

    # ---- Step 2: Load instruction: VADD rd=1, rs1=2, rs2=3 ----
    #   opcode=0x10 (VADD/VALU), opt=0x00 (VOPT_ADD)
    #   Instruction format: [31:24]=opcode, [27:20]=opt,
    #                       [23:16]=rd, [15:8]=rs1, [7:0]=rs2
    vadd_instr = (0x10 << 24) | (0x00 << 20) | (1 << 16) | (2 << 8) | 3
    await if_load(dut, 0, vadd_instr)
    # Pad with NOPs so pipeline doesn't read X
    NOP = 0xFF000000
    for i in range(1, 4):
        await if_load(dut, i, NOP)

    # ---- Step 3: Write VALU register file ----
    #   regfile[2] = all lanes = 10 (INT8)
    #   regfile[3] = all lanes = 25 (INT8)
    vec_a = [10] * 64
    vec_b = [25] * 64
    await valu_write_reg(dut, 2, vec_a)
    await valu_write_reg(dut, 3, vec_b)

    # Verify regfile writes
    readback_a = await valu_read_reg(dut, 2)
    assert readback_a[0] == 10, f"Reg[2] lane 0: {readback_a[0]} != 10"
    readback_b = await valu_read_reg(dut, 3)
    assert readback_b[0] == 25, f"Reg[3] lane 0: {readback_b[0]} != 25"
    dut._log.info("VALU regfile preload verified")

    # ---- Step 4: Release CSR reset + start ----
    await csr_write(dut, CSR_CTRL, 0x00000001)  # release reset; IF/ID drives VALU
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    # Check that STATUS shows busy
    status = await csr_read(dut, CSR_STATUS)
    dut._log.info(f"STATUS after start: 0x{status:08X}")

    # ---- Step 5: Poll for NPU busy, then idle ----
    busy_seen = False
    idle_seen = False
    for cycle in range(200):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        status = await csr_read(dut, CSR_STATUS)
        busy = status & 1
        if busy:
            busy_seen = True
        if busy_seen and not busy:
            idle_seen = True
            dut._log.info(f"NPU idle at cycle {cycle} (busy first seen)")
            break

    assert busy_seen, "NPU never went busy"
    assert idle_seen, "NPU did not return to idle within timeout"

    # ---- Step 6: Read VALU result regfile[1] ----
    result = await valu_read_reg(dut, 1)
    expected = [(10 + 25) & 0xFF] * 64

    errors = []
    for i in range(64):
        if result[i] != expected[i]:
            errors.append(f"  Lane {i}: got 0x{result[i]:02x}, expected 0x{expected[i]:02x}")

    if errors:
        for err in errors[:10]:
            dut._log.error(err)
        assert False, f"VADD mismatch in {len(errors)}/64 lanes"

    dut._log.info(f"PASS: VALU VADD via IF/ID — reg[1] lane[0]={result[0]} (expected 35)")


@cocotb.test()
async def test_irq_on_done(dut):
    """Test 3: IRQ asserted on NPU done.

    Enable IRQ, run a VALU instruction, verify irq_stat[0] set when NPU idle.
    """
    await setup_dut(dut)

    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    # Assert CSR reset
    await csr_write(dut, CSR_CTRL, 0x00000002)
    await RisingEdge(dut.clk)

    # Load a simple VADD instruction
    vadd_instr = (0x10 << 24) | (0x00 << 20) | (1 << 16) | (2 << 8) | 3
    await if_load(dut, 0, vadd_instr)
    for i in range(1, 4):
        await if_load(dut, i, 0xFF000000)

    # Preload VALU registers
    await valu_write_reg(dut, 2, [5] * 64)
    await valu_write_reg(dut, 3, [7] * 64)

    # Enable IRQ
    await csr_write(dut, CSR_IRQ_EN, 0x00000001)
    await csr_write(dut, CSR_IRQ_STAT, 0x00000001)  # clear any pending

    # Start NPU
    await csr_write(dut, CSR_CTRL, 0x00000001)
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    # Wait for busy then idle
    busy_seen = False
    for _ in range(200):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        status = await csr_read(dut, CSR_STATUS)
        if status & 1:
            busy_seen = True
        if busy_seen and not (status & 1):
            break

    assert busy_seen, "NPU never went busy in IRQ test"

    # Check IRQ status
    irq_stat = await csr_read(dut, CSR_IRQ_STAT)
    assert irq_stat & 1, f"IRQ_STAT[0] not set: 0x{irq_stat:08X}"

    # Verify irq output
    irq_val = int(dut.irq.value)
    assert irq_val == 1, f"irq output not asserted: got {irq_val}"

    dut._log.info(f"PASS: IRQ on done — irq_stat=0x{irq_stat:08X}, irq_out={irq_val}")

    # Read result
    result = await valu_read_reg(dut, 1)
    assert result[0] == ((5 + 7) & 0xFF), f"Unexpected result: {result[0]}"
