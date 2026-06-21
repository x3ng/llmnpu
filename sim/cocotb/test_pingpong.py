"""Focused contract tests for the ping-pong buffer controller."""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer


async def posedge(dut):
    await RisingEdge(dut.clk)
    await Timer(1, unit="ps")


async def reset_dut(dut):
    dut.fill_done.value = 0
    dut.consume_done.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await posedge(dut)


async def pulse(dut, fill=False, consume=False):
    dut.fill_done.value = int(fill)
    dut.consume_done.value = int(consume)
    await posedge(dut)
    dut.fill_done.value = 0
    dut.consume_done.value = 0


def snapshot(dut):
    return (
        int(dut.fill_bank.value),
        int(dut.active_bank.value),
        int(dut.ready.value),
    )


def assert_state(dut, expected, label):
    actual = snapshot(dut)
    assert actual == expected, (
        f"{label}: expected fill/active/ready={expected}, got {actual}"
    )


@cocotb.test()
async def test_first_fill_is_ready(dut):
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    assert_state(dut, (0, 0, 0), "reset")

    await pulse(dut, fill=True)
    assert_state(dut, (1, 0, 1), "first filled bank is consumable")

    await pulse(dut, consume=True)
    assert_state(dut, (0, 0, 0), "single consumed bank leaves no ready data")


@cocotb.test()
async def test_consume_advances_to_next_valid_bank(dut):
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    await pulse(dut, fill=True)
    assert_state(dut, (1, 0, 1), "bank A ready")

    await pulse(dut, fill=True)
    assert_state(dut, (0, 0, 1), "both banks full keeps bank A active")

    await pulse(dut, consume=True)
    assert_state(dut, (0, 1, 1), "consume bank A advances to valid bank B")

    await pulse(dut, consume=True)
    assert_state(dut, (0, 1, 0), "consume bank B leaves controller empty")


@cocotb.test()
async def test_simultaneous_fill_and_consume_selects_valid_bank(dut):
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    await pulse(dut, fill=True)
    assert_state(dut, (1, 0, 1), "bank A ready before overlap")

    await pulse(dut, fill=True, consume=True)
    assert_state(dut, (0, 1, 1), "consume A while filling B selects valid B")


@cocotb.test()
async def test_refill_consumed_bank_in_same_cycle(dut):
    clock = Clock(dut.clk, 2, unit="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    await pulse(dut, fill=True)
    await pulse(dut, fill=True)
    assert_state(dut, (0, 0, 1), "both banks full before refill")

    await pulse(dut, fill=True, consume=True)
    assert_state(dut, (1, 0, 1), "refilled bank A remains valid and active")
