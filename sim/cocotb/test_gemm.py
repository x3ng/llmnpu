"""test_gemm.py — Cocotb test for 16x16 Systolic Array GEMM.

Verifies INT8 outer product: C = A_col @ B_row clamped to INT16.

Architecture (B broadcast, A broadcast):
  - B: b_in[c] broadcast to ALL PE rows and reloaded for each COMPUTE k.
  - A: a_in[r] broadcast to all PEs in row r during COMPUTE.
  - Pipeline: IDLE → LOAD_B → PREFETCH → COMPUTE → REDUCE → WRITEBACK → IDLE

Lessons applied:
  - Use Timer(1, "ps") after each RisingEdge for settled VPI reads (iverilog quirk).
  - Use logic signed in RTL (not $signed() — broken in iverilog).
  - Reduction uses generate-for with genvar (iverilog rejects for(int i...) in always_ff).
  - PE psum_out is REGISTERED (lags acc by 1 cycle). Reduction fires in WRITEBACK state to compensate.
  - Manual clock toggle: cocotb Clock() hangs iverilog with 256 PE VPI signals.
"""

import cocotb
from cocotb.triggers import RisingEdge, Timer
import numpy as np

ROWS = 16
COLS = 16


# ---------------------------------------------------------------------------
# Manual clock helpers (cocotb Clock() hangs iverilog with 256 PE instances)
# ---------------------------------------------------------------------------
HALF_PERIOD = 2  # ns


def init_clk(dut):
    """Ensure clock starts at 0."""
    dut.clk.value = 0


async def manual_tick(dut):
    """One full clock cycle: low -> half -> high -> half."""
    dut.clk.value = 0
    await Timer(HALF_PERIOD, unit="ns")
    dut.clk.value = 1
    await Timer(HALF_PERIOD, unit="ns")


async def posedge(dut):
    """Manual rising edge + 1 ps delta for settled NBA reads (iverilog VPI quirk)."""
    dut.clk.value = 0
    await Timer(HALF_PERIOD, unit="ns")
    dut.clk.value = 1
    await Timer(HALF_PERIOD, unit="ns")
    await Timer(1, unit="ps")


async def clock_cycles(dut, n):
    """Wait N full clock cycles."""
    for _ in range(n):
        await manual_tick(dut)


# ---------------------------------------------------------------------------
# Pack / unpack helpers
# ---------------------------------------------------------------------------
def pack_a(arr):
    """Pack 16 x INT8 a_in vector into a 128-bit integer.

    SV packed array mapping: a_in[15:0][7:0] -> bits [r*8+7 : r*8] = a_in[r].
    """
    val = 0
    for r in range(ROWS):
        byte = int(arr[r]) & 0xFF
        val |= byte << (r * 8)
    return val


def pack_b(arr):
    """Pack 16 x INT8 b_in vector into a 128-bit integer.

    SV packed array mapping: b_in[15:0][7:0] -> bits [c*8+7 : c*8] = b_in[c].
    """
    val = 0
    for c in range(COLS):
        byte = int(arr[c]) & 0xFF
        val |= byte << (c * 8)
    return val


def unpack_psum(val):
    """Unpack 16x16 INT16 psum_out into a (16,16) numpy int32 array.

    SV packed array mapping: psum_out[15:0][15:0][15:0] ->
      bits [(r*COLS + c)*16 +: 16] = psum_out[r][c].
    """
    out = np.zeros((ROWS, COLS), dtype=np.int32)
    for r in range(ROWS):
        for c in range(COLS):
            shift = (r * COLS + c) * 16
            raw = (val >> shift) & 0xFFFF
            # Convert unsigned 16-bit to signed INT16
            out[r, c] = raw if raw < 32768 else raw - 65536
    return out


# ---------------------------------------------------------------------------
# Pipeline runners
# ---------------------------------------------------------------------------
def gemm_golden(A, B):
    acc = A.astype(np.int32) @ B.astype(np.int32)
    return np.clip(acc, -32768, 32767).astype(np.int32)


def assert_matrix_close(dut, result, golden, context):
    mismatches = 0
    for r in range(ROWS):
        for c in range(COLS):
            got = int(result[r, c])
            exp = int(golden[r, c])
            if got != exp:
                if mismatches < 10:
                    dut._log.error("%s [%d,%d] got=%d expected=%d", context, r, c, got, exp)
                mismatches += 1

    assert mismatches == 0, f"{context}: {mismatches} mismatches out of {ROWS * COLS}"


async def run_outer_product(dut, A_col, B_row):
    """Run the SA pipeline with k_count=1 and return the (16,16) result array.

    A_col: (16,) int32 — single column of A (A[:, k])
    B_row: (16,) int32 — single row of B (B[k, :])

    Timing (PE psum_out is REGISTERED, lags acc by 1 cycle):
      Edge 0: reset deassert → IDLE
      Edge 1: IDLE → LOAD_B     (load_b=1 asserted, B driven — NOT yet latched)
      Edge 2: LOAD_B → PREFETCH  (B latched into b_reg, clear_acc asserted)
      Edge 3: PREFETCH → COMPUTE (acc cleared, A latched into a_reg)
      Edge 4: COMPUTE → REDUCE   (MAC fires: acc <= a_reg * b_reg; pe.psum_out lags)
      Edge 5: REDUCE → WRITEBACK (pe.psum_out catches up to acc)
      Edge 6: WRITEBACK → IDLE   (reduction fires, psum_out<=saturated; psum_valid=1)
    """
    # ---- initialise ----
    dut.a_in.value = 0
    dut.a_valid.value = 0
    dut.b_in.value = 0
    dut.start.value = 0
    dut.k_count.value = 0

    # ---- reset ----
    dut.rst_n.value = 0
    init_clk(dut)
    await clock_cycles(dut, 10)
    dut.rst_n.value = 1
    await posedge(dut)            # Edge 0: reset → IDLE

    # ---- LOAD_B: 1 cycle, B broadcast to all PE rows ----
    dut.k_count.value = 1
    dut.start.value = 1
    dut.b_in.value = pack_b(B_row)
    await posedge(dut)            # Edge 1: IDLE → LOAD_B (B NOT yet latched)
    dut.start.value = 0

    # ---- PREFETCH: 1 cycle, clear accumulators ----
    await posedge(dut)            # Edge 2: LOAD_B → PREFETCH (B latched, clear_acc=1)

    # ---- Drive A, enter COMPUTE ----
    dut.a_valid.value = 1
    dut.a_in.value = pack_a(A_col)
    await posedge(dut)            # Edge 3: PREFETCH → COMPUTE (acc cleared, A latched)

    # ---- COMPUTE: MAC executes (acc <= a_reg * b_reg) ----
    await posedge(dut)            # Edge 4: COMPUTE → REDUCE (MAC fires, pe.psum_out lags)
    dut.a_valid.value = 0
    dut.a_in.value = 0

    # ---- REDUCE → WRITEBACK: pe.psum_out catches up to acc ----
    await posedge(dut)            # Edge 5: REDUCE → WRITEBACK (pe.psum_out now valid)

    # ---- WRITEBACK → IDLE: reduction fires, psum_valid asserted ----
    await posedge(dut)            # Edge 6: WRITEBACK → IDLE (reduction + psum_valid=1)

    # ---- read output ----
    assert dut.psum_valid.value == 1, "psum_valid not asserted after WRITEBACK"
    return unpack_psum(int(dut.psum_out.value))


async def run_gemm_tile(dut, A, B):
    """Run one 16x16xK tile and return the (16,16) INT16 result.

    PE timing matters: the MAC uses registered A/B from the previous edge.
    B is loaded in LOAD_B for k=0, then reloaded during each COMPUTE cycle
    for the next k. A follows the same one-cycle-ahead schedule.
    """
    k_count = int(A.shape[1])
    assert A.shape == (ROWS, k_count)
    assert B.shape == (k_count, COLS)

    dut.a_in.value = 0
    dut.a_valid.value = 0
    dut.b_in.value = 0
    dut.start.value = 0
    dut.k_count.value = 0

    dut.rst_n.value = 0
    init_clk(dut)
    await clock_cycles(dut, 10)
    dut.rst_n.value = 1
    await posedge(dut)

    dut.k_count.value = k_count
    dut.start.value = 1
    dut.b_in.value = pack_b(B[0])
    dut.a_in.value = pack_a(A[:, 0])
    await posedge(dut)            # IDLE -> LOAD_B
    dut.start.value = 0

    await posedge(dut)            # LOAD_B -> PREFETCH, latches B[0]

    dut.a_valid.value = 1
    dut.a_in.value = pack_a(A[:, 0])
    await posedge(dut)            # PREFETCH -> COMPUTE, latches A[:,0]

    for k in range(1, k_count):
        dut.a_in.value = pack_a(A[:, k])
        dut.b_in.value = pack_b(B[k])
        await posedge(dut)        # COMPUTE k-1, latch inputs for k

    await posedge(dut)            # final COMPUTE -> REDUCE
    dut.a_valid.value = 0
    dut.a_in.value = 0
    dut.b_in.value = 0

    await posedge(dut)            # REDUCE -> WRITEBACK
    await posedge(dut)            # WRITEBACK -> IDLE

    assert dut.psum_valid.value == 1, "psum_valid not asserted after WRITEBACK"
    return unpack_psum(int(dut.psum_out.value))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@cocotb.test()
async def test_gemm_basic(dut):
    """k_count=1 outer product with random A, B — compare to numpy matmul golden.

    Tolerance: ±1 LSB (quantisation / rounding).
    """
    init_clk(dut)

    rng = np.random.default_rng(42)
    A_col = rng.integers(-128, 127, size=ROWS, dtype=np.int32)
    B_row = rng.integers(-128, 127, size=COLS, dtype=np.int32)

    # Golden: outer product A_col @ B_row, clamped to INT16
    A_mat = A_col.reshape(ROWS, 1).astype(np.float64)
    B_mat = B_row.reshape(1, COLS).astype(np.float64)
    golden = np.matmul(A_mat, B_mat)
    golden = np.clip(np.round(golden), -32768, 32767).astype(np.int32)

    result = await run_outer_product(dut, A_col, B_row)

    assert_matrix_close(dut, result, golden, "k_count=1")
    dut._log.info("PASS: all %dx%d entries match golden (±1 LSB)", ROWS, COLS)


@cocotb.test()
async def test_gemm_zero_a(dut):
    """All-zero A should produce all-zero output."""
    init_clk(dut)

    A_col = np.zeros(ROWS, dtype=np.int32)
    B_row = np.arange(COLS, dtype=np.int32) - 8

    result = await run_outer_product(dut, A_col, B_row)

    for r in range(ROWS):
        for c in range(COLS):
            assert result[r, c] == 0, f"[{r},{c}] got={result[r,c]} expected=0"

    dut._log.info("PASS: A=0 => all-zero output")


@cocotb.test()
async def test_gemm_zero_b(dut):
    """All-zero B should produce all-zero output."""
    init_clk(dut)

    A_col = np.arange(ROWS, dtype=np.int32) - 8
    B_row = np.zeros(COLS, dtype=np.int32)

    result = await run_outer_product(dut, A_col, B_row)

    for r in range(ROWS):
        for c in range(COLS):
            assert result[r, c] == 0, f"[{r},{c}] got={result[r,c]} expected=0"

    dut._log.info("PASS: B=0 => all-zero output")


@cocotb.test()
async def test_gemm_k2(dut):
    """k_count=4: sum_k A[:,k] * B[k,:] with changing B every COMPUTE."""
    init_clk(dut)

    rng = np.random.default_rng(99)
    k_count = 4
    A = rng.integers(-64, 63, size=(ROWS, k_count), dtype=np.int32)
    B = rng.integers(-64, 63, size=(k_count, COLS), dtype=np.int32)
    golden = gemm_golden(A, B)

    result = await run_gemm_tile(dut, A, B)
    assert_matrix_close(dut, result, golden, "k_count=4")
    dut._log.info("PASS: k_count=%d multi-K values match golden", k_count)


@cocotb.test()
async def test_gemm_idle_after_done(dut):
    """After completing one GEMM, the array returns to IDLE and can start again."""
    init_clk(dut)

    rng = np.random.default_rng(7)

    for run_idx in range(2):
        A_col = rng.integers(-128, 127, size=ROWS, dtype=np.int32)
        B_row = rng.integers(-128, 127, size=COLS, dtype=np.int32)

        A_mat = A_col.reshape(ROWS, 1).astype(np.float64)
        B_mat = B_row.reshape(1, COLS).astype(np.float64)
        golden = np.clip(np.round(np.matmul(A_mat, B_mat)), -32768, 32767).astype(np.int32)

        result = await run_outer_product(dut, A_col, B_row)

        for r in range(ROWS):
            for c in range(COLS):
                got = int(result[r, c])
                exp = int(golden[r, c])
                assert abs(got - exp) <= 1, f"run {run_idx} [{r},{c}] got={got} expected={exp}"

        # After WRITEBACK→IDLE, controller should be back in IDLE
        assert dut.busy.value == 0, f"busy not deasserted after run {run_idx}"
        assert dut.done.value == 0, f"done not deasserted after run {run_idx}"

        # Extra idle cycles between runs
        await clock_cycles(dut, 4)

    dut._log.info("PASS: two back-to-back runs — controller returns to IDLE cleanly")
