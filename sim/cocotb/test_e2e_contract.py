"""RISC-V driver-visible NPU contract test."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles


RESULT_BASE = 0x40080000
RESULT_MAGIC = 0x4E505543
RESULT_PASS = 1
TIMEOUT_CYCLES = 1_500_000
RESULT_WORDS = 7


def _dram_word(dut, byte_addr):
    word_addr = (byte_addr - 0x40000000) >> 2
    return int(dut.u_dram.mem[word_addr].value) & 0xFFFFFFFF


@cocotb.test()
async def test_e2e_contract(dut):
    """CPU firmware calls NPU driver APIs for DMA, VALU, and SFU."""
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    last = int(dut.uart_tx.value)
    progress = []
    aw_addrs = []
    ar_addrs = []
    for cyc in range(TIMEOUT_CYCLES):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        if int(dut.dma_axi_arvalid.value) and int(dut.dma_axi_arready.value):
            ar_addrs.append(int(dut.dma_axi_araddr.value))
        if int(dut.dma_axi_awvalid.value) and int(dut.dma_axi_awready.value):
            aw_addrs.append(int(dut.dma_axi_awaddr.value))
        val = int(dut.uart_tx.value)
        if val != 0 and val != last:
            ch = chr(val)
            progress.append(ch)
            dut._log.info(
                f"UART_TX = 0x{val:02X} ('{ch}') at cycle {cyc + 1}"
            )
            if ch == "P":
                break
            if ch == "F":
                raise AssertionError(
                    f"contract firmware failed after progress={''.join(progress)} "
                    f"results={[hex(_dram_word(dut, RESULT_BASE + i * 4)) for i in range(RESULT_WORDS)]} "
                    f"ar_addrs={[hex(a) for a in ar_addrs[-12:]]} "
                    f"aw_addrs={[hex(a) for a in aw_addrs[-12:]]}"
                )
        last = val
    else:
        try:
            dut._log.info(
                "timeout diag: "
                f"progress={''.join(progress)} "
                f"result0=0x{_dram_word(dut, RESULT_BASE + 0):08X} "
                f"pass=0x{_dram_word(dut, RESULT_BASE + 4):08X} "
                f"dma1d=0x{_dram_word(dut, RESULT_BASE + 8):08X} "
                f"dma2d=0x{_dram_word(dut, RESULT_BASE + 12):08X} "
                f"valu_scalar=0x{_dram_word(dut, RESULT_BASE + 16):08X} "
                f"valu_in1=0x{_dram_word(dut, RESULT_BASE + 20):08X} "
                f"sfu=0x{_dram_word(dut, RESULT_BASE + 24):08X} "
                f"csr_status=0x{int(dut.u_npu.u_csr.status_reg.value):08X} "
                f"debug=0x{int(dut.u_npu.debug_signals.value):08X} "
                f"trap={int(dut.trap_latched.value)} "
                f"npu_busy={int(dut.u_npu.npu_busy.value)} "
                f"dma_busy={int(dut.u_npu.dma_busy.value)} "
                f"bridge={int(dut.u_npu.dma_br_state.value)}"
                f"aw_addrs={[hex(a) for a in aw_addrs[-12:]]}"
            )
        except Exception as exc:
            dut._log.info(f"timeout diag unavailable: {exc}")
        raise AssertionError("contract firmware timed out")

    magic = _dram_word(dut, RESULT_BASE + 0)
    passed = _dram_word(dut, RESULT_BASE + 4)
    dma1d_ret = _dram_word(dut, RESULT_BASE + 8)
    dma2d_ret = _dram_word(dut, RESULT_BASE + 12)
    valu_scalar_ret = _dram_word(dut, RESULT_BASE + 16)
    valu_in1_ret = _dram_word(dut, RESULT_BASE + 20)
    sfu_ret = _dram_word(dut, RESULT_BASE + 24)
    assert magic == RESULT_MAGIC, f"bad result magic 0x{magic:08X}"
    assert passed == RESULT_PASS, f"firmware did not set pass flag: 0x{passed:08X}"
    assert dma1d_ret == 0, f"DMA 1D contract returned 0x{dma1d_ret:08X}"
    assert dma2d_ret == 0, f"DMA 2D contract returned 0x{dma2d_ret:08X}"
    assert valu_scalar_ret == 0, f"VALU scalar contract returned 0x{valu_scalar_ret:08X}"
    assert valu_in1_ret == 0, f"VALU in1 contract returned 0x{valu_in1_ret:08X}"
    assert sfu_ret == 0, f"SFU contract returned 0x{sfu_ret:08X}"
    assert "".join(progress[-6:]) == "dDvVsP", f"unexpected progress {progress}"

    dut._log.info("E2E contract test: PASS")
