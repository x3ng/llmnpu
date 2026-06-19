// ============================================================
// firmware_gemm.c — E2E Stage 4: Full NPU GEMM Execution Test
//
// Exercises the complete GEMM data path:
//   1. A[16][16] and B[16][16] in .rodata (ExtMem 0x40000000+)
//   2. NPU init via npu_init (CSR base 0x10000000)
//   3. npu_rt_gemm with M=N=K=16 (manual inlined flow)
//   4. DMA STORE result from O-SRAM to gemm_result[256] in ext_mem
//   5. Compare gemm_result vs precomputed golden_C element-by-element
//
// Diagnostic UART output sequence (each stage always outputs):
//   'I'=init OK / 'i'=init stuck busy
//   'A'=A-DMA-load OK / 'a'=A-DMA-load fail
//   'B'=B-DMA-load OK / 'b'=B-DMA-load fail
//   'G'=GEMM done OK  / 'g'=GEMM fail/timeout
//   'S'=STORE OK       / 's'=STORE fail
//   Then:
//     'P' if all 256 match
//     'F' + 1-byte mismatch-count + up to 4×(idx, hw_lo, hw_hi, gold_lo, gold_hi)
// ============================================================

#include "../../sw/runtime/npu_runtime.h"

// ------------------------------------------------------------
// Minimal libc stubs (nostdlib)
// ------------------------------------------------------------
void *memset(void *s, int c, unsigned int n)
{
    unsigned char *p = (unsigned char *)s;
    while (n--) *p++ = (unsigned char)c;
    return s;
}

// ------------------------------------------------------------
// UART TX helper
// ------------------------------------------------------------
static void uart_putc(char c)
{
    *(volatile uint32_t *)0x00000008u = (uint32_t)(unsigned char)c;
}

// ------------------------------------------------------------
// Local DMA transfer — polls CSR_STATUS.BUSY (bit0) which
// correctly reflects npu_busy.  The driver's _dma_xfer polls
// CSR_DMA_CSR3 bits 2/3, which the CSR module never drives
// from hardware — those bits always read back as written by
// firmware, so the driver never actually waits for the DMA.
//
// After the DMA FSM returns to idle, the bridge FSM may still
// be copying data between DMA-internal SRAM and crossbar SRAM
// banks.  We add a word-proportional bridge delay.
// ------------------------------------------------------------
static int _dma_xfer_local(uint32_t ext_addr, uint32_t sram_off,
                           uint32_t len, int is_st)
{
    volatile uint32_t *csr_status = (volatile uint32_t *)0x10000004u;
    volatile uint32_t *csr_dma0   = (volatile uint32_t *)0x10000020u;
    volatile uint32_t *csr_dma1   = (volatile uint32_t *)0x10000028u;
    volatile uint32_t *csr_dma2   = (volatile uint32_t *)0x10000030u;
    volatile uint32_t *csr_dma3   = (volatile uint32_t *)0x10000038u;
    volatile uint32_t to;

    // Wait for any previous DMA to finish (BUSY poll CSR_STATUS bit0)
    to = 10000u;
    while (*csr_status & 1u) {
        if (--to == 0) return -1;
    }

    // Program DMA CSRs
    *csr_dma0 = ext_addr;
    *csr_dma1 = sram_off;
    *csr_dma2 = len;

    uint32_t ctrl = 0x1u;        // START bit
    if (is_st) ctrl |= 0x2u;     // DIR_ST = store
    *csr_dma3 = ctrl;

    // Wait for DMA to assert BUSY (may happen immediately after CSR3 write)
    to = 1000u;
    while (!(*csr_status & 1u)) { if (--to == 0) break; }

    // Wait for DMA to de-assert BUSY (FSM returns to S_IDLE)
    to = 10000u;
    while (*csr_status & 1u) {
        if (--to == 0) return -1;
    }

    // Bridge FSM: COPY/PREFILL runs after dma_done and is NOT
    // reflected in CSR_STATUS.BUSY.  Budget ~10 cycles per 4-byte
    // crossbar word + margin.
    to = (len >> 2) * 10u + 40u;
    while (--to) { __asm__ volatile ("" : : : "memory"); }

    return 0;
}

// ------------------------------------------------------------
// Test data — 16x16 INT8 GEMM
//   A[i][j] = (i + j) % 7 - 3   (row-major, in .rodata)
//   B[i][j] = (i * 3 + j) % 7 - 3
//
// Golden computed as C[i][j] = sum_k A[i][k] * B[k][j] (int16)
//   precomputed and embedded as golden_C in .rodata
// ------------------------------------------------------------
static const int8_t test_A[256] __attribute__((aligned(4))) = {
     -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -2,  -1,   0,   1,   2,   3,  -3,  -2,
     -2,  -1,   0,   1,   2,   3,  -3,  -2,
     -1,   0,   1,   2,   3,  -3,  -2,  -1,
     -1,   0,   1,   2,   3,  -3,  -2,  -1,
      0,   1,   2,   3,  -3,  -2,  -1,   0,
      0,   1,   2,   3,  -3,  -2,  -1,   0,
      1,   2,   3,  -3,  -2,  -1,   0,   1,
      1,   2,   3,  -3,  -2,  -1,   0,   1,
      2,   3,  -3,  -2,  -1,   0,   1,   2,
      2,   3,  -3,  -2,  -1,   0,   1,   2,
      3,  -3,  -2,  -1,   0,   1,   2,   3,
      3,  -3,  -2,  -1,   0,   1,   2,   3,
     -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -2,  -1,   0,   1,   2,   3,  -3,  -2
};

static const int8_t test_B[256] __attribute__((aligned(4))) = {
     -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -2,  -1,   0,   1,   2,   3,  -3,  -2,
      0,   1,   2,   3,  -3,  -2,  -1,   0,
      1,   2,   3,  -3,  -2,  -1,   0,   1,
      3,  -3,  -2,  -1,   0,   1,   2,   3,
     -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -1,   0,   1,   2,   3,  -3,  -2,  -1,
      0,   1,   2,   3,  -3,  -2,  -1,   0,
      2,   3,  -3,  -2,  -1,   0,   1,   2,
      3,  -3,  -2,  -1,   0,   1,   2,   3,
     -2,  -1,   0,   1,   2,   3,  -3,  -2,
     -1,   0,   1,   2,   3,  -3,  -2,  -1,
      1,   2,   3,  -3,  -2,  -1,   0,   1,
      2,   3,  -3,  -2,  -1,   0,   1,   2,
     -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -2,  -1,   0,   1,   2,   3,  -3,  -2
};

// Golden: C = A x B  (int16, in .rodata — auto-generated)
// Regenerate with: python3 tools/codegen/generate_golden.py --header > build/golden_gemm.h
#include "../../build/golden_gemm.h"

// ------------------------------------------------------------
// Global variables (.bss — zeroed at boot, then filled by DMA STORE)
// ------------------------------------------------------------
volatile int16_t gemm_result[256] __attribute__((aligned(4)));

// ------------------------------------------------------------
// Bare-metal entry point
// ------------------------------------------------------------
void main(void);

extern int __bss_start, __bss_end;

__attribute__((section(".text._start")))
void _start(void)
{
    __asm__ volatile (
        ".option push\n\t"
        ".option norelax\n\t"
        "la   sp, __stack_top\n\t"
        ".option pop\n\t"
    );

    for (char *p = (char *)&__bss_start; p < (char *)&__bss_end;)
        *p++ = 0;

    main();

    while (1)
        __asm__ volatile ("wfi");
}

// ------------------------------------------------------------
// Main — E2E GEMM test with stage-by-stage UART diagnostics
// ------------------------------------------------------------
void main(void)
{
    npu_dev_t  npu;

    // Stage-success flags: 0=fail/skip, 1=completed successfully
    int a_dma_ok  = 0;
    int b_dma_ok  = 0;
    int gemm_ok   = 0;
    int store_ok  = 0;

    // ================================================================
    // Stage 1: NPU init — check CSR_STATUS.BUSY after init
    // ================================================================
    npu_init(&npu, 0x10000000u);

    {
        volatile uint32_t *csr_status =
            (volatile uint32_t *)(0x10000000u + 0x04u);
        uint32_t st = *csr_status;

        if (st & 1u) {
            uart_putc('i');   // BUSY=1 — stuck busy
        } else {
            uart_putc('I');   // BUSY=0 — OK to proceed
        }
    }

    // ================================================================
    // Stages 2-5: A-DMA, B-DMA, GEMM, STORE (tiled 16x16x16)
    //
    // Each stage uses a local `fail` flag. On failure, we abort the
    // remaining stages (pipeline_abort=1).  On success, we set the
    // corresponding ok flag to 1.
    // ================================================================
    {
        const int M = 16, N = 16, K = 16;
        const int m_tiles = 1, n_tiles = 1, k_tiles = 1;

        int pipeline_abort = 0;

        for (int mt = 0; mt < m_tiles && !pipeline_abort; mt++) {
          int m_off = mt * 16;
          for (int nt = 0; nt < n_tiles && !pipeline_abort; nt++) {
            int n_off = nt * 16;

            for (int kt = 0; kt < k_tiles && !pipeline_abort; kt++) {
              int k_off = kt * 16;

              // --------------------------------------------------
              // A-DMA: LOAD A tile rows into ASRAM
              // --------------------------------------------------
              {
                  int fail = 0;
                  for (int r = 0; r < 16; r++) {
                    if (_dma_xfer_local(
                          (uint32_t)(uintptr_t)(&test_A[(m_off+r)*K + k_off]),
                          0x0000u + (uint32_t)(r * 16), 16, 0) != 0) {
                      fail = 1; break;
                    }
                  }

                  // Verify A-DMA CSRs post-transfer:
                  // CSR0 = last ext addr (must be >= 0x40000000)
                  // CSR1 = last SRAM offset (must be in ASRAM range)
                  if (!fail) {
                    volatile uint32_t *csr_dma0 =
                        (volatile uint32_t *)(0x10000000u + 0x20u);
                    volatile uint32_t *csr_dma1 =
                        (volatile uint32_t *)(0x10000000u + 0x28u);
                    uint32_t dma0 = *csr_dma0;
                    uint32_t dma1 = *csr_dma1;
                    if (dma0 < 0x40000000u || dma1 > 0x1000u) {
                      fail = 1;
                    }
                  }

                  if (fail) { pipeline_abort = 1; break; }
                  a_dma_ok = 1;
              }

              // --------------------------------------------------
              // B-DMA: LOAD B tile rows into WSRAM
              // --------------------------------------------------
              {
                  int fail = 0;
                  for (int r = 0; r < 16; r++) {
                    if (_dma_xfer_local(
                          (uint32_t)(uintptr_t)(&test_B[(k_off+r)*N + n_off]),
                          0x1000u + (uint32_t)(r * 16), 16, 0) != 0) {
                      fail = 1; break;
                    }
                  }
                  if (fail) { pipeline_abort = 1; break; }
                  b_dma_ok = 1;
              }

              // --------------------------------------------------
              // GEMM: load descriptor, issue, wait for done
              // --------------------------------------------------
              {
                  int fail = 0;
                  uint32_t desc[2];
                  desc[0] = 0;
                  desc[1] = 0;
                  // Load descriptor into DSRAM at bump-allocator base
                  // (npu_load_descriptor uses _dma_xfer internally — bypass it)
                  int dp = 0x3000;
                  if (_dma_xfer_local((uint32_t)(uintptr_t)desc, dp, 8, 0) != 0) {
                    fail = 1; dp = -1;
                  }
                  if (dp < 0) { fail = 1; }
                  else if (npu_issue(&npu, 0x01 /* OP_GEMM */, (uint32_t)dp) != 0) {
                      fail = 1;
                  }
                  else if (npu_wait_done(&npu, 1000000u) != 0) {
                      fail = 1;
                  }

                  // Verify GEMM finished: CSR_STATUS.BUSY must be 0
                  if (!fail) {
                      volatile uint32_t *csr_status =
                          (volatile uint32_t *)(0x10000000u + 0x04u);
                      if (*csr_status & 1u) {
                          fail = 1;
                      }
                  }

                  if (fail) { pipeline_abort = 1; break; }
                  gemm_ok = 1;
              }

              // --------------------------------------------------
              // STORE: DMA STORE C tile rows from OSRAM → ext_mem
              // --------------------------------------------------
              {
                  // Prime DMA direction bit so first STORE is correct
                  *(volatile uint32_t *)(0x10000000u + 0x38u) = 0x00000002u;

                  int fail = 0;
                  for (int r = 0; r < 16; r++) {
                    uint32_t ext_c = (uint32_t)(uintptr_t)(
                        &gemm_result[(m_off + r) * N + n_off]);
                    uint32_t sro   = 0x2000u + (uint32_t)(r * 32);
                    if (_dma_xfer_local(ext_c, sro, 32, 1) != 0) {
                      fail = 1; break;
                    }
                  }
                  if (fail) { pipeline_abort = 1; break; }
                  store_ok = 1;
              }

            }  // kt
          }  // nt
        }  // mt
    }

    // ---- Emit stage markers -------------------------------------------
    uart_putc(a_dma_ok ? 'A' : 'a');
    uart_putc(b_dma_ok ? 'B' : 'b');
    uart_putc(gemm_ok  ? 'G' : 'g');
    uart_putc(store_ok ? 'S' : 's');

    // ================================================================
    // Stage 6: Compare hardware result against golden
    // ================================================================
    int mismatches = 0;

    // Track up to 4 mismatches for detailed reporting
    int     mismatch_idx[4];
    int16_t mismatch_hw[4];
    int16_t mismatch_gold[4];

    for (int i = 0; i < 256; i++) {
        if (gemm_result[i] != golden_C[i]) {
            if (mismatches < 4) {
                mismatch_idx[mismatches]  = i;
                mismatch_hw[mismatches]   = gemm_result[i];
                mismatch_gold[mismatches] = golden_C[i];
            }
            mismatches++;
        }
    }

    // ---- Report ------------------------------------------------------
    if (mismatches == 0) {
        uart_putc('P');
    } else {
        uart_putc('F');

        // Mismatch count as one byte (capped at 255)
        unsigned char cnt = (mismatches > 255) ? 255 : (unsigned char)mismatches;
        uart_putc((char)cnt);

        // First up-to-4 mismatches: idx(1) + hw_le(2) + gold_le(2)
        int nreport = (mismatches < 4) ? mismatches : 4;
        for (int i = 0; i < nreport; i++) {
            // index as byte
            uart_putc((char)(mismatch_idx[i] & 0xFF));

            // hw value as int16 LE (2 bytes)
            uint16_t hw_u = (uint16_t)mismatch_hw[i];
            uart_putc((char)(hw_u & 0xFF));
            uart_putc((char)((hw_u >> 8) & 0xFF));

            // golden value as int16 LE (2 bytes)
            uint16_t gold_u = (uint16_t)mismatch_gold[i];
            uart_putc((char)(gold_u & 0xFF));
            uart_putc((char)((gold_u >> 8) & 0xFF));
        }
    }
}
