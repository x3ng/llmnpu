"""test_pe.py — Cocotb test for PE (Processing Element).

Verifies signed INT8xINT8 multiply-accumulate with INT32 accumulation.

Uses Timer(1, "ps") after each RisingEdge to advance past the delta
where NBA updates are committed, ensuring VPI reads show settled values.
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer
import random


async def posedge(dut):
    """Wait for rising edge then advance to next delta for settled NBA reads."""
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")


async def reset_dut(dut):
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await posedge(dut)


@cocotb.test()
async def test_pe_basic_accumulate(dut):
    """Load weight, feed 16 acts, verify psum at each step and final value."""
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    dut.load_b.value = 0
    dut.clear_acc.value = 0
    dut.valid_in.value = 0
    dut.b_in.value = 0
    dut.a_in.value = 0

    await reset_dut(dut)
    assert dut.psum_out.value.to_signed() == 0
    assert dut.a_out.value.to_signed() == 0
    assert dut.b_out.value.to_signed() == 0

    weight = random.randint(-128, 127)
    acts = [random.randint(-128, 127) for _ in range(16)]

    # === Load weight ===
    dut.load_b.value = 1
    dut.b_in.value = weight & 0x1FF
    await posedge(dut)
    dut.load_b.value = 0
    await posedge(dut)
    await posedge(dut)
    assert dut.b_out.value.to_signed() == weight

    # === Pipeline model ===
    a_reg_m = 0       # what a_reg contains
    acc_m = 0          # what acc contains
    psum_m = 0         # what psum_out should show

    for i in range(16):
        act = acts[i]

        # Check psum_out
        actual = dut.psum_out.value.to_signed()
        assert actual == psum_m, \
            f"Step {i}: psum={actual}, expected={psum_m} " \
            f"(act={act}, weight={weight})"

        if i == 8:
            # Clear — acc resets, a_reg captures stale a_in
            dut.clear_acc.value = 1
            await posedge(dut)
            dut.clear_acc.value = 0
            psum_m = acc_m        # psum got old acc before clear
            acc_m = 0
            # a_reg_m unchanged — a_reg <= a_in (= acts[7])
            continue

        # Feed activation
        dut.valid_in.value = 1
        dut.a_in.value = act & 0x1FF
        await posedge(dut)

        # Model: psum <= old_acc, acc <= acc + a_reg*b_reg, a_reg <= a_in
        psum_m = acc_m
        acc_m += a_reg_m * weight
        a_reg_m = act

    # === Drain ===
    # Flush last a_reg through MAC
    dut.valid_in.value = 1
    dut.a_in.value = 0
    await posedge(dut)
    psum_m = acc_m
    acc_m += a_reg_m * weight
    a_reg_m = 0

    # Two cycles for psum to show final acc
    dut.valid_in.value = 0
    for _ in range(2):
        actual = dut.psum_out.value.to_signed()
        assert actual == psum_m, f"Drain: psum={actual}, expected={psum_m}"
        await posedge(dut)
        psum_m = acc_m

    # Final check
    actual = dut.psum_out.value.to_signed()
    assert actual == psum_m, f"Final: psum={actual}, expected={psum_m}"

    total_expected = (acts[7] + sum(acts[9:])) * weight
    assert actual == total_expected, \
        f"Total: psum={actual}, expected={total_expected}"

    dut._log.info(f"PASS: weight={weight}, total={total_expected}")
