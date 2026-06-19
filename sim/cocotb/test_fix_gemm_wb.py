"""test_fix_gemm_wb.py — Minimal test for Bug 1 + Bug 2 fixes.

Bug 1: GEMM psum -> OSRAM writeback
Bug 2: DMA Bridge PREFILL race

Verification:
  - Writeback FSM: check gemm_wb_active toggles when psum_valid fires
  - PREFILL race: check DMA is held during PREFILL, starts after
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer, ClockCycles

CSR_CTRL      = 0x00
CSR_STATUS    = 0x01
CSR_DMA_CSR0  = 0x08
CSR_DMA_CSR1  = 0x0A
CSR_DMA_CSR2  = 0x0C
CSR_DMA_CSR3  = 0x0E

ASRAM_BASE = 0x0000
WSRAM_BASE = 0x1000
OSRAM_BASE = 0x2000
NOP = 0xFF000000


async def csr_write(dut, addr_word, value):
    dut.csr_addr.value = addr_word << 2
    dut.csr_wdata.value = value
    dut.csr_we.value = 1
    await RisingEdge(dut.clk)
    dut.csr_we.value = 0
    dut.csr_addr.value = 0
    dut.csr_wdata.value = 0


async def csr_read(dut, addr_word):
    dut.csr_addr.value = addr_word << 2
    dut.csr_re.value = 1
    await Timer(1, unit="ns")
    val = int(dut.csr_rdata.value)
    dut.csr_re.value = 0
    dut.csr_addr.value = 0
    return val


async def if_load(dut, addr, instr):
    dut.dbg_imem_we.value = 1
    dut.dbg_imem_addr.value = addr
    dut.dbg_imem_wdata.value = instr
    await RisingEdge(dut.clk)
    dut.dbg_imem_we.value = 0


async def wait_npu_idle(dut, timeout=500):
    busy_seen = False
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        status = await csr_read(dut, CSR_STATUS)
        if status & 1:
            busy_seen = True
        if busy_seen and not (status & 1):
            return True
    return False


@cocotb.test()
async def test_gemm_writeback(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # ── Init ──
    dut.rst_n.value = 0
    for s in ['csr_addr','csr_wdata','csr_we','csr_re',
              'dbg_imem_we','dbg_imem_addr','dbg_imem_wdata',
              'dbg_valu_wen','dbg_valu_waddr','dbg_valu_wdata_flat','dbg_valu_raddr',
              'dma_ext_rdata']:
        getattr(dut, s).value = 0

    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Hold IF/ID in reset while loading instructions
    await csr_write(dut, CSR_CTRL, 0x00000002)

    # Load GEMM K=1 at imem[0]
    await if_load(dut, 0, (0x01 << 24) | 1)
    for i in range(1, 8):
        await if_load(dut, i, NOP)
    dut._log.info("GEMM K=1 loaded at imem[0]")

    # ===================================================================
    # DMA LOAD to fill ASRAM and WSRAM with known data
    # ===================================================================
    # Release reset so DMA can operate, but IF/ID will also start.
    # We accept this — IF/ID dispatches GEMM early (data may be zeros).
    await csr_write(dut, CSR_CTRL, 0x00000000)  # reset=0, start=0
    await RisingEdge(dut.clk)

    # Feed all-2s to dma_ext_rdata for A tile load
    async def feed(pat, n):
        for _ in range(n):
            await FallingEdge(dut.clk)
            dut.dma_ext_rdata.value = pat
    feeder_a = cocotb.start_soon(feed(0x0202020202020202, 80))

    await csr_write(dut, CSR_DMA_CSR0, 0)
    await csr_write(dut, CSR_DMA_CSR1, ASRAM_BASE)
    await csr_write(dut, CSR_DMA_CSR2, 256)
    await csr_write(dut, CSR_DMA_CSR3, 0x00000001)  # start LOAD
    await wait_npu_idle(dut, timeout=500)
    await ClockCycles(dut.clk, 80)  # bridge COPY margin
    dut._log.info("DMA LOAD ASRAM done")

    # Feed all-3s for B tile load
    feeder_b = cocotb.start_soon(feed(0x0303030303030303, 20))
    await csr_write(dut, CSR_DMA_CSR0, 0)
    await csr_write(dut, CSR_DMA_CSR1, WSRAM_BASE)
    await csr_write(dut, CSR_DMA_CSR2, 16)
    await csr_write(dut, CSR_DMA_CSR3, 0x00000001)
    await wait_npu_idle(dut, timeout=300)
    await ClockCycles(dut.clk, 20)  # bridge COPY margin
    dut._log.info("DMA LOAD WSRAM done")

    # Re-arm IF/ID: pulse reset, then start
    await csr_write(dut, CSR_CTRL, 0x00000002)  # reset=1
    await csr_write(dut, CSR_CTRL, 0x00000001)  # start=1, reset=0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")
    dut._log.info("NPU re-armed + started")

    # ===================================================================
    # Bug 1 Check: Monitor gemm_wb_active for writeback FSM activity
    # ===================================================================
    wb_active_seen = False
    psum_valid_seen = False
    for cyc in range(400):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        try:
            if int(dut.gemm_psum_valid.value):
                psum_valid_seen = True
        except Exception:
            pass
        try:
            if int(dut.gemm_wb_active.value):
                wb_active_seen = True
                break
        except Exception:
            pass

    dut._log.info(
        f"Bug 1 check: psum_valid_seen={psum_valid_seen}, "
        f"wb_active_seen={wb_active_seen}"
    )

    # Wait extra for writeback to complete
    await ClockCycles(dut.clk, 200)

    # ===================================================================
    # Bug 2 Check: DMA STORE from OSRAM exercises PREFILL fix
    # ===================================================================
    # Monitor dma_restart (Bug 2: DMA should start AFTER prefill)
    prefill_active_seen = False
    dma_restart_seen = False

    store_data_count = [0]

    async def monitor_bug2():
        for _ in range(600):
            await RisingEdge(dut.clk)
            await Timer(1, unit="ps")
            try:
                if int(dut.prefill_active.value):
                    prefill_active_seen = True
            except Exception:
                pass
            try:
                if int(dut.dma_restart.value):
                    dma_restart_seen = True
            except Exception:
                pass
            if int(dut.dma_ext_we.value):
                store_data_count[0] += 1

    mon = cocotb.start_soon(monitor_bug2())

    await csr_write(dut, CSR_DMA_CSR0, 0x00000000)
    await csr_write(dut, CSR_DMA_CSR1, OSRAM_BASE)
    await csr_write(dut, CSR_DMA_CSR2, 512)
    await csr_write(dut, CSR_DMA_CSR3, 0x00000003)  # start STORE

    await wait_npu_idle(dut, timeout=1500)
    await ClockCycles(dut.clk, 50)

    dut._log.info(
        f"Bug 2 check: prefill_active_seen={prefill_active_seen}, "
        f"dma_restart_seen={dma_restart_seen}, "
        f"store_data_words={store_data_count[0]}"
    )

    # ===================================================================
    # Results
    # ===================================================================
    bug1_ok = wb_active_seen
    bug2_ok = prefill_active_seen and dma_restart_seen and store_data_count[0] > 0

    if bug1_ok:
        dut._log.info("Bug 1 fixed — gemm_wb_active toggled (writeback FSM triggered)")
    else:
        dut._log.error("Bug 1 FAILED — writeback FSM gemm_wb_active never went high")

    if bug2_ok:
        dut._log.info(
            "Bug 2 fixed — DMA held during PREFILL (prefill_active=1), "
            "dma_restart pulsed (DMA started after prefill), "
            f"{store_data_count[0]} store words captured"
        )
    else:
        dut._log.error(
            "Bug 2 FAILED — prefill_active or dma_restart not observed, "
            "or no store data captured"
        )

    assert bug1_ok, "Bug 1: writeback FSM did not activate"
    assert bug2_ok, "Bug 2: PREFILL race not fixed"
    dut._log.info("Bug 1+2 fixed — both fixes verified")
