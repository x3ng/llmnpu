"""E2E test for executing a serialized .npu program through IF/ID."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer, ClockCycles


@cocotb.test()
async def test_e2e_program(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")

    last = int(dut.uart_tx.value)
    expected_progress = "iabrgpsP"
    progress = []
    valid_markers = set(expected_progress[:-1])
    for cyc in range(240_000):
        await RisingEdge(dut.clk)
        await Timer(1, unit="ps")
        val = int(dut.uart_tx.value)
        if val != 0 and val != last:
            ch = chr(val)
            progress.append(ch)
            dut._log.info(
                f"UART_TX = 0x{val:02X} ('{ch}') at cycle {cyc + 1}"
            )
            if ch == "P":
                assert "".join(progress) == expected_progress, (
                    f"unexpected program progress {''.join(progress)}"
                )
                dut._log.info("E2E program-stream test: PASS")
                return
            if ch not in valid_markers:
                assert False, (
                    f"program-stream firmware failed at stage '{ch}', "
                    f"progress={''.join(progress)}"
                )
        last = val

    try:
        dut._log.info(
            "timeout diag: "
            f"progress={''.join(progress)} "
            f"uart=0x{int(dut.uart_tx.value):02X} "
            f"csr_status=0x{int(dut.u_npu.u_csr.status_reg.value):08X} "
            f"csr_ctrl=0x{int(dut.u_npu.u_csr.ctrl_reg.value):08X} "
            f"desc_ptr=0x{int(dut.u_npu.u_csr.desc_ptr_reg.value):08X} "
            f"if_pc={int(dut.u_npu.u_if_id.pc.value)} "
            f"if_instr=0x{int(dut.u_npu.u_if_id.id_instr.value):08X} "
            f"imem0=0x{int(dut.u_npu.u_if_id.imem0.value):08X} "
            f"gpl_state={int(dut.u_npu.gpl_state.value)} "
            f"gpl_row={int(dut.u_npu.gpl_row.value)} "
            f"gpl_word={int(dut.u_npu.gpl_word.value)} "
            f"gpl_last={int(dut.u_npu.gpl_a_word_last.value)} "
            f"gpl_desc=0x{int(dut.u_npu.gpl_desc_ptr_latched.value):04X} "
            f"gpl_k={int(dut.u_npu.gpl_k_count.value)} "
            f"m1_req={int(dut.u_npu.m1_req.value)} "
            f"m1_addr=0x{int(dut.u_npu.m1_addr.value):04X} "
            f"m1_grant={int(dut.u_npu.xbar_m1_grant.value)} "
            f"dma_busy={int(dut.u_npu.dma_busy.value)} "
            f"bridge={int(dut.u_npu.dma_br_state.value)} "
            f"gemm_busy={int(dut.u_npu.gemm_busy.value)} "
            f"npu_busy={int(dut.u_npu.npu_busy.value)}"
        )
    except Exception as exc:
        dut._log.info(f"timeout diag unavailable: {exc}")
    assert False, "program-stream firmware timed out"
