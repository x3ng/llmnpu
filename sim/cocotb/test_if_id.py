"""test_if_id.py — Cocotb test for the IF/ID/Dispatch pipeline.

Tests:
  1. Decode R-type GEMM instruction (0x01020304)
  2. Decode I-type DMA_LD instruction (0x400AAAAA)
  3. Dispatch GEMM->VADD->RELU sequence, verify routing to correct units
  4. SYNC instruction asserts stall_if when units report busy
  5. WFI halts fetch/dispatch
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer, ReadOnly

NOP = 0xFF000000  # opcode 0xFF = NOP (+ padding)


async def setup_reset(dut):
    """Start clock and put DUT in reset (rst_n=0), drive all inputs low."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.mem_we.value = 0
    dut.mem_addr.value = 0
    dut.mem_wdata.value = 0
    dut.halt.value = 0
    dut.gemm_busy.value = 0
    dut.valu_busy.value = 0
    dut.sfu_busy.value = 0
    dut.dma_busy.value = 0

    await ClockCycles(dut.clk, 3)


async def load_instrs(dut, instrs):
    """Load a list of 32-bit instructions into I-SRAM starting at address 0,
    then fill the next address with NOP to prevent X propagation."""
    dut.mem_we.value = 1
    for i, instr in enumerate(instrs):
        dut.mem_addr.value = i
        dut.mem_wdata.value = instr
        await RisingEdge(dut.clk)
    # Pad with NOP so the pipeline never reads X
    dut.mem_addr.value = len(instrs)
    dut.mem_wdata.value = NOP
    await RisingEdge(dut.clk)
    dut.mem_we.value = 0


# =========================================================================


@cocotb.test()
async def test_decode_rtype(dut):
    """Test 1: Decode R-type GEMM instruction 0x01020304.

    R-type format: [31:24]=OP=0x01, [23:16]=DST=0x02,
                   [15:8]=SRC_A=0x03, [7:0]=SRC_B=0x04
    """
    await setup_reset(dut)
    await load_instrs(dut, [0x01020304])

    # Debug: check if imem[0] was written
    dut._log.info(f"debug_imem0=0x{int(dut.debug_imem0.value):08X}")

    # Release reset + let it propagate
    dut.rst_n.value = 1
    await Timer(1, unit='ns')

    # Posedge A: pipeline captures imem[0] into id_instr, PC 0->1
    await RisingEdge(dut.clk)
    await ReadOnly()

    # Verify the instruction was captured in the debug register.
    # At this point debug_instr reflects the JUST-captured instruction
    # (the pipeline captured it at the posedge we just waited for).
    instr = int(dut.debug_instr.value)
    assert instr == 0x01020304, \
        f"debug_instr=0x{instr:08X} != 0x01020304"
    assert (instr >> 24) & 0xFF == 0x01, "opcode != 0x01"
    assert (instr >> 16) & 0xFF == 0x02, "dst != 0x02"
    assert (instr >> 8) & 0xFF == 0x03,  "src_a != 0x03"
    assert instr & 0xFF == 0x04,          "src_b != 0x04"

    # Posedge B: dispatch GEMM -> gemm_cmd_valid pulses high.
    # Pipeline also advances (id_instr becomes imem[1] = NOP).
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.gemm_cmd_valid.value), \
        "gemm_cmd_valid not asserted after GEMM decode"
    assert int(dut.gemm_cmd.value) == 0x01020304, \
        f"gemm_cmd=0x{int(dut.gemm_cmd.value):08X} != 0x01020304"

    dut._log.info("PASS: Test 1 R-type decode 0x01020304 -> "
                  "opcode=0x01 dst=0x02 src_a=0x03 src_b=0x04")


@cocotb.test()
async def test_itype_decode(dut):
    """Test 2: Decode I-type DMA_LD instruction 0x400AAAAA.

    I-type: [31:28]=OP[3:0]=4 (DMA), [27:20]=OPT=0x00,
            [19:0]=IMM=0xAAAAA.  is_itype=1 for high nibble 4.
    """
    await setup_reset(dut)

    # 0x400AAAAA: opcode=0x40, OPT=0x00, IMM=0xAAAAA (20 bits)
    await load_instrs(dut, [0x400AAAAA])

    dut.rst_n.value = 1
    await Timer(1, unit='ns')

    # Posedge A: pipeline captures instruction
    await RisingEdge(dut.clk)
    await ReadOnly()
    instr = int(dut.debug_instr.value)
    assert instr == 0x400AAAAA, \
        f"debug_instr=0x{instr:08X} != 0x400AAAAA"

    # Check opcode[31:24]
    assert (instr >> 24) & 0xFF == 0x40, "opcode != 0x40"
    # Check is_itype: opcode high nibble == 4
    assert (instr >> 28) & 0xF == 0x4, "is_itype: top nibble != 4"
    # Check imm[19:0]
    imm_val = instr & 0xFFFFF
    assert imm_val == 0xAAAAA, f"imm=0x{imm_val:05X} != 0xAAAAA"

    # Posedge B: dispatch DMA_LD -> dma_cmd_valid pulses high
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.dma_cmd_valid.value), \
        "dma_cmd_valid not asserted after DMA_LD decode"

    dut._log.info("PASS: Test 2 I-type decode 0x400AAAAA -> "
                  "opcode=0x40 imm=0xAAAAA")


@cocotb.test()
async def test_dispatch_routing(dut):
    """Test 3: Dispatch GEMM->VADD->RELU sequence to correct units.

    Verifies each instruction routes to its target unit exclusively.
    """
    await setup_reset(dut)

    await load_instrs(dut, [
        0x01020304,  # GEMM      (opcode 0x01)
        0x10000000,  # VADD      (opcode 0x10)
        0x20000000,  # ACT_RELU  (opcode 0x20)
    ])

    dut.rst_n.value = 1
    await Timer(1, unit='ns')

    # Posedge A: pipeline captures imem[0] (GEMM)
    await RisingEdge(dut.clk)

    # Posedge B: dispatch GEMM -> gemm_cmd_valid=1, pipeline captures imem[1] (VADD)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.gemm_cmd_valid.value), \
        "Test 3 cycle 1: gemm_cmd_valid not asserted"
    assert not int(dut.valu_cmd_valid.value), \
        "Test 3 cycle 1: valu_cmd_valid should be 0"
    assert not int(dut.sfu_cmd_valid.value), \
        "Test 3 cycle 1: sfu_cmd_valid should be 0"
    assert not int(dut.dma_cmd_valid.value), \
        "Test 3 cycle 1: dma_cmd_valid should be 0"

    # Posedge C: dispatch VADD -> valu_cmd_valid=1, pipeline captures imem[2] (RELU)
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.valu_cmd_valid.value), \
        "Test 3 cycle 2: valu_cmd_valid not asserted"
    assert not int(dut.gemm_cmd_valid.value), \
        "Test 3 cycle 2: gemm_cmd_valid should be 0"
    assert not int(dut.sfu_cmd_valid.value), \
        "Test 3 cycle 2: sfu_cmd_valid should be 0"
    assert not int(dut.dma_cmd_valid.value), \
        "Test 3 cycle 2: dma_cmd_valid should be 0"

    # Posedge D: dispatch RELU -> sfu_cmd_valid=1
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.sfu_cmd_valid.value), \
        "Test 3 cycle 3: sfu_cmd_valid not asserted"
    assert not int(dut.gemm_cmd_valid.value), \
        "Test 3 cycle 3: gemm_cmd_valid should be 0"
    assert not int(dut.valu_cmd_valid.value), \
        "Test 3 cycle 3: valu_cmd_valid should be 0"
    assert not int(dut.dma_cmd_valid.value), \
        "Test 3 cycle 3: dma_cmd_valid should be 0"

    dut._log.info("PASS: Test 3 GEMM->VADD->RELU routed to correct units")


@cocotb.test()
async def test_dispatch_past_four_word_boundary(dut):
    """IF/ID fetches and dispatches instructions beyond address 3."""
    await setup_reset(dut)

    await load_instrs(dut, [
        NOP,
        NOP,
        NOP,
        NOP,
        0x01020304,  # GEMM at imem[4]
    ])

    dut.rst_n.value = 1
    await Timer(1, unit='ns')

    for cycle in range(1, 7):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if cycle < 6:
            assert not int(dut.gemm_cmd_valid.value), \
                f"GEMM dispatched too early at cycle {cycle}"
        else:
            assert int(dut.gemm_cmd_valid.value), \
                "GEMM at imem[4] did not dispatch"
            assert int(dut.gemm_cmd.value) == 0x01020304, \
                f"gemm_cmd=0x{int(dut.gemm_cmd.value):08X}"
        await Timer(1, unit='ns')

    dut._log.info("PASS: IF/ID dispatches instruction beyond imem[3]")


@cocotb.test()
async def test_sync_stall(dut):
    """Test 4: SYNC instruction asserts stall_if when units report busy.

    SYNC (opcode 0xF0) stalls the pipeline until ALL units are idle.
    """
    await setup_reset(dut)

    # Load SYNC instruction at address 0
    await load_instrs(dut, [0xF0000000])

    # Set gemm busy before releasing reset
    dut.gemm_busy.value = 1

    # Release reset
    dut.rst_n.value = 1
    await Timer(1, unit='ns')

    # Posedge A: pipeline captures SYNC.
    # stall_internal = 1 (is_sync && gemm_busy)
    await RisingEdge(dut.clk)

    # Posedge B: stall_if <= stall_internal (= 1).
    # Pipeline does NOT advance (stall_internal gates the pipeline register).
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert int(dut.stall_if.value), \
        "Test 4: stall_if should be 1 when SYNC + unit busy"
    assert int(dut.debug_instr.value) == 0xF0000000, \
        "Test 4: debug_instr should still show SYNC when stalled"

    # Exit ReadOnly phase before setting signals
    await Timer(1, unit='ns')

    # Clear the busy source
    dut.gemm_busy.value = 0

    # Posedge C: stall_internal = 0 (all units idle) -> stall_if de-asserts.
    # Pipeline advances past SYNC (to NOP).
    await RisingEdge(dut.clk)
    await ReadOnly()
    assert not int(dut.stall_if.value), \
        "Test 4: stall_if should be 0 when SYNC and no busy sources"

    dut._log.info("PASS: Test 4 SYNC stall_if=1 when busy, de-asserts when idle")


@cocotb.test()
async def test_wfi_halts_dispatch(dut):
    """WFI stops the IF/ID stream; later instructions must not dispatch."""
    await setup_reset(dut)

    await load_instrs(dut, [
        0xF1000000,  # WFI
        0x01020304,  # GEMM must not dispatch after WFI
    ])

    dut.rst_n.value = 1
    await Timer(1, unit='ns')

    for _ in range(8):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert not int(dut.gemm_cmd_valid.value), \
            "GEMM dispatched after WFI halted the IF/ID stream"
        await Timer(1, unit='ns')

    assert int(dut.stall_if.value), "WFI should hold stall_if high after halt"
    dut._log.info("PASS: Test 5 WFI halts fetch/dispatch")


@cocotb.test()
async def test_external_halt_stalls_pc_without_dispatch(dut):
    """External CSR HALT input stops fetch/dispatch until released."""
    await setup_reset(dut)

    await load_instrs(dut, [
        0x01020304,  # GEMM must wait while halted
        NOP,
    ])

    dut.halt.value = 1
    dut.rst_n.value = 1
    await Timer(1, unit='ns')

    for cycle in range(5):
        await RisingEdge(dut.clk)
        await ReadOnly()
        assert not int(dut.gemm_cmd_valid.value), \
            f"GEMM dispatched while halt=1 at cycle {cycle}"
        assert int(dut.debug_pc.value) == 0, \
            f"PC advanced while halt=1: {int(dut.debug_pc.value)}"
        assert int(dut.stall_if.value), "stall_if should assert while halt=1"
        await Timer(1, unit='ns')

    dut.halt.value = 0
    dispatched = False
    for _ in range(4):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if int(dut.gemm_cmd_valid.value):
            dispatched = True
            assert int(dut.gemm_cmd.value) == 0x01020304, \
                f"halt release dispatched wrong cmd 0x{int(dut.gemm_cmd.value):08X}"
            break
        await Timer(1, unit='ns')

    assert dispatched, "GEMM did not dispatch after halt release"
    dut._log.info("PASS: external halt stalls PC/dispatch and resumes cleanly")
