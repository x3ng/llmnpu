import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

@cocotb.test()
async def test_debug(dut):
    clock = Clock(dut.clk, 2, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.mem_we.value = 0
    dut.mem_addr.value = 0
    dut.mem_wdata.value = 0
    dut.gemm_busy.value = 0
    dut.valu_busy.value = 0
    dut.sfu_busy.value = 0
    dut.dma_busy.value = 0
    
    await ClockCycles(dut.clk, 3)
    
    # Load instruction
    dut.mem_we.value = 1
    dut.mem_addr.value = 0
    dut.mem_wdata.value = 0xDEADBEEF
    await RisingEdge(dut.clk)
    dut.mem_we.value = 0
    
    dut._log.info(f"After load: mem_we={int(dut.mem_we.value)}")
    
    # Release reset
    dut.rst_n.value = 1
    
    # Check signals before any more edges
    await RisingEdge(dut.clk)
    dut._log.info(f"After 1st pipeline: debug_instr=0x{int(dut.debug_instr.value):08X} debug_pc={int(dut.debug_pc.value)}")
    dut._log.info(f"  pc={int(dut.u_dispatch.u_ifetch.pc.value)} stall_if={int(dut.stall_if.value)}")
    
    await RisingEdge(dut.clk)
    dut._log.info(f"After 2nd: debug_instr=0x{int(dut.debug_instr.value):08X} gemm_cmd_valid={int(dut.gemm_cmd_valid.value)}")
    dut._log.info(f"  pc={int(dut.u_dispatch.u_ifetch.pc.value)}")
    
    await RisingEdge(dut.clk)
    dut._log.info(f"After 3rd: debug_instr=0x{int(dut.debug_instr.value):08X} gemm_cmd_valid={int(dut.gemm_cmd_valid.value)}")
    dut._log.info(f"  pc={int(dut.u_dispatch.u_ifetch.pc.value)}")

