# test_valu.py — Cocotb test for NPU VALU 64-lane SIMD
# Tests VADD of two random 64-element INT8 vectors

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import random


def pack_bytes(bytes_list):
    """Pack 64 bytes into a 512-bit integer. Lane 0 -> bits [7:0]."""
    result = 0
    for i, b in enumerate(bytes_list):
        result |= ((b & 0xFF) << (i * 8))
    return result


def unpack_bytes(value):
    """Unpack a 512-bit integer into 64 bytes."""
    v = int(value)
    return [(v >> (i * 8)) & 0xFF for i in range(64)]


def int8_add(a, b):
    """Signed 8-bit addition with wrapping, returns unsigned 8-bit value."""
    return ((a + b) & 0xFF)


def to_s8(x):
    x &= 0xFF
    return x - 256 if x & 0x80 else x


def expected_op(opt, a, b):
    base_opt = opt & 0x0F
    if base_opt == 0:
        return (to_s8(a) + to_s8(b)) & 0xFF
    if base_opt == 1:
        return (to_s8(a) - to_s8(b)) & 0xFF
    if base_opt == 2:
        return (to_s8(a) * to_s8(b)) & 0xFF
    if base_opt == 3:
        return a if to_s8(a) < to_s8(b) else b
    if base_opt == 4:
        return a if to_s8(a) > to_s8(b) else b
    if base_opt == 5:
        return a & b
    if base_opt == 6:
        return a | b
    if base_opt == 7:
        return a ^ b
    if base_opt == 8:
        return (a << (b & 0x7)) & 0xFF
    if base_opt == 9:
        return (to_s8(a) >> (b & 0x7)) & 0xFF
    if base_opt == 10:
        return b if (a & 0xFF) != 0 else 0
    raise ValueError(opt)


async def reset_valu(dut):
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.test_wen.value = 0
    dut.test_waddr.value = 0
    dut.test_raddr.value = 0
    dut.cmd_valid.value = 0
    dut.opcode.value = 0
    dut.opt.value = 0
    dut.rs1.value = 0
    dut.rs2.value = 0
    dut.rd.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)


async def write_vec(dut, reg_idx, values):
    dut.test_wen.value = 1
    dut.test_waddr.value = reg_idx
    dut.test_wdata.value = pack_bytes(values)
    await RisingEdge(dut.clk)
    dut.test_wen.value = 0
    await RisingEdge(dut.clk)


async def read_vec(dut, reg_idx):
    dut.test_raddr.value = reg_idx
    await Timer(1, unit="ns")
    return unpack_bytes(dut.test_rdata.value)


async def issue_valu(dut, opt, rd=2, rs1=0, rs2=1):
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 1
    dut.opcode.value = 0x10
    dut.opt.value = opt
    dut.rs1.value = rs1
    dut.rs2.value = rs2
    dut.rd.value = rd
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 0

    done_seen = False
    for _ in range(20):
        await RisingEdge(dut.clk)
        if not done_seen and dut.done.value:
            done_seen = True
        if done_seen and not dut.done.value and not dut.busy.value:
            return
    assert False, f"VALU FSM timeout for opt={opt}"


@cocotb.test()
async def test_valu_vadd(dut):
    """VADD: add two random 64-element INT8 vectors, verify each lane."""

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # ---- Reset -----------------------------------------------------------
    dut.rst_n.value = 0
    dut.test_wen.value = 0
    dut.test_waddr.value = 0
    dut.test_raddr.value = 0
    dut.cmd_valid.value = 0
    dut.opcode.value = 0
    dut.opt.value = 0
    dut.rs1.value = 0
    dut.rs2.value = 0
    dut.rd.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    # ---- Generate random test vectors -----------------------------------
    vec_a = [random.randint(0, 255) for _ in range(64)]
    vec_b = [random.randint(0, 255) for _ in range(64)]

    # ---- Load vec_a into register 0 --------------------------------------
    dut.test_wen.value = 1
    dut.test_waddr.value = 0
    dut.test_wdata.value = pack_bytes(vec_a)
    await RisingEdge(dut.clk)
    dut.test_wen.value = 0
    await RisingEdge(dut.clk)

    # ---- Load vec_b into register 1 --------------------------------------
    dut.test_wen.value = 1
    dut.test_waddr.value = 1
    dut.test_wdata.value = pack_bytes(vec_b)
    await RisingEdge(dut.clk)
    dut.test_wen.value = 0
    await RisingEdge(dut.clk)

    # ---- Sanity-check register writes ------------------------------------
    dut.test_raddr.value = 0
    await Timer(1, unit="ns")
    readback = unpack_bytes(dut.test_rdata.value)
    for i in range(64):
        assert readback[i] == vec_a[i], \
            f"Register 0 load failed lane {i}: got 0x{readback[i]:02x}, expected 0x{vec_a[i]:02x}"

    dut.test_raddr.value = 1
    await Timer(1, unit="ns")
    readback = unpack_bytes(dut.test_rdata.value)
    for i in range(64):
        assert readback[i] == vec_b[i], \
            f"Register 1 load failed lane {i}: got 0x{readback[i]:02x}, expected 0x{vec_b[i]:02x}"

    dut._log.info("Register-load sanity check passed")

    # ---- Issue VADD: rs1=0, rs2=1, rd=2, opt=VOPT_ADD=0 -----------------
    await RisingEdge(dut.clk)
    dut.cmd_valid.value = 1
    dut.opcode.value = 0x10   # OP_VADD
    dut.opt.value = 0x00      # VOPT_ADD
    dut.rs1.value = 0
    dut.rs2.value = 1
    dut.rd.value = 2
    await RisingEdge(dut.clk)  # IDLE -> READ
    dut.cmd_valid.value = 0

    # ---- Wait for FSM to complete ---------------------------------------
    # Poll: wait for done to be asserted (one WB cycle), then back to IDLE
    done_seen = False
    for _ in range(20):
        await RisingEdge(dut.clk)
        if not done_seen and dut.done.value:
            done_seen = True
            dut._log.info("done asserted (WRITEBACK state)")
        if done_seen and not dut.done.value and not dut.busy.value:
            dut._log.info("VALU returned to IDLE")
            break
    else:
        assert False, f"VALU FSM timeout: done_seen={done_seen} busy={int(dut.busy.value)} done={int(dut.done.value)}"

    # ---- Read result from register 2 ------------------------------------
    dut.test_raddr.value = 2
    await Timer(1, unit="ns")
    result_bytes = unpack_bytes(dut.test_rdata.value)

    # ---- Verify each lane -----------------------------------------------
    errors = []
    for i in range(64):
        expected = int8_add(vec_a[i], vec_b[i])
        if result_bytes[i] != expected:
            errors.append(f"  Lane {i}: got 0x{result_bytes[i]:02x}, expected 0x{expected:02x}")

    if errors:
        for err in errors[:10]:
            dut._log.error(err)
        assert False, f"VADD mismatch in {len(errors)}/{64} lanes"

    dut._log.info(f"VADD OK — all 64 lanes match expected (sample a[0]=0x{vec_a[0]:02x}, b[0]=0x{vec_b[0]:02x}, r[0]=0x{result_bytes[0]:02x})")
    dut._log.info("TASK 1.3 COMPLETE: VALU 64-lane SIMD verified")


@cocotb.test()
async def test_valu_all_defined_opts(dut):
    """Verify all defined VALU operations for all 64 lanes."""
    await reset_valu(dut)

    vec_a = [((i * 17 + 0x83) & 0xFF) for i in range(64)]
    vec_b = [((i * 29 + 0x35) & 0xFF) for i in range(64)]
    vec_a[0] = 0
    vec_a[1] = 1
    vec_b = [(v & 0x07) if i % 5 == 0 else v for i, v in enumerate(vec_b)]
    await write_vec(dut, 0, vec_a)
    await write_vec(dut, 1, vec_b)

    ops = [
        (0x00, "ADD"),
        (0x01, "SUB"),
        (0x02, "MUL"),
        (0x03, "MIN"),
        (0x04, "MAX"),
        (0x05, "AND"),
        (0x06, "OR"),
        (0x07, "XOR"),
        (0x08, "SLL"),
        (0x09, "SRA"),
        (0x0A, "CMOV"),
    ]
    for opt, name in ops:
        rd = 2 + opt
        await issue_valu(dut, opt, rd=rd)
        result = await read_vec(dut, rd)
        expected = [expected_op(opt, a, b) for a, b in zip(vec_a, vec_b)]
        mismatches = [
            (i, result[i], expected[i])
            for i in range(64)
            if result[i] != expected[i]
        ]
        assert not mismatches, (
            f"VALU {name} mismatches: {mismatches[:8]} total={len(mismatches)}"
        )

    dut._log.info("PASS: VALU all defined VOPT operations match expected values")


@cocotb.test()
async def test_valu_broadcast_rs2_lane0(dut):
    """VALU broadcast mode uses rs2 lane0 for every lane."""
    await reset_valu(dut)

    vec_a = [((i * 11 + 3) & 0xFF) for i in range(64)]
    vec_b = [0x05] + [((i * 7 + 0x40) & 0xFF) for i in range(1, 64)]
    await write_vec(dut, 0, vec_a)
    await write_vec(dut, 1, vec_b)

    opt = 0x80 | 0x00  # broadcast + ADD
    await issue_valu(dut, opt, rd=2)
    result = await read_vec(dut, 2)
    expected = [expected_op(opt, a, vec_b[0]) for a in vec_a]
    mismatches = [
        (i, result[i], expected[i])
        for i in range(64)
        if result[i] != expected[i]
    ]
    assert not mismatches, (
        f"VALU broadcast ADD mismatches: {mismatches[:8]} total={len(mismatches)}"
    )

    dut._log.info("PASS: VALU broadcast uses rs2 lane0 across all lanes")
