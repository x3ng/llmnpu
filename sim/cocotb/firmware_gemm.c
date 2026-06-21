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

static unsigned int csr_debug_read(void)
{
    return *(volatile unsigned int *)0x10000060u;
}

static void uart_puthex(unsigned int val)
{
    val &= 0xFu;
    if (val < 10) uart_putc('0' + (char)val);
    else          uart_putc('A' + (char)(val - 10));
}

static void uart_puthex32(unsigned int val)
{
    int i;
    for (i = 7; i >= 0; i--) {
        uart_puthex((val >> (i * 4)) & 0xFu);
    }
}

// Wait for npu_busy with timeout and debug dump on hang
// Returns 0 on success, -1 on timeout (with UART debug output)
static int wait_busy_clear(const char tag)
{
    volatile unsigned int *csr_status = (volatile unsigned int *)0x10000004u;
    volatile unsigned int to = 500000u;  // generous timeout for Verilator
    while (*csr_status & 1u) {
        if (--to == 0) {
            uart_putc('T');        // Timeout
            uart_putc(tag);        // which stage
            uart_puthex32(csr_debug_read());  // 8-char hex debug word
            return -1;
        }
    }
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
     -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,
     -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,
     -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,
      0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,
      1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,
      2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,
      3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,
     -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,
     -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,
      0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,
      1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,
      2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,
      3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,
     -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1
};

static const int8_t test_B[256] __attribute__((aligned(4))) = {
     -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,
      0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,
      3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,
      2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,
     -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,
      1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,
     -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,
      0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,
      3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,
     -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,
      2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,
     -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,
      1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,
     -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,
      0,   1,   2,   3,  -3,  -2,  -1,   0,   1,   2,   3,  -3,  -2,  -1,   0,   1
};

// Golden: C = A x B  (int16, in .rodata — auto-generated)
// Regenerate with: python3 tools/codegen/generate_golden.py --header > build/golden_gemm.h
#include "../../build/golden_gemm.h"

// ------------------------------------------------------------
// Global variables (.bss — zeroed at boot, then filled by DMA STORE)
// ------------------------------------------------------------
volatile int16_t gemm_result[256] __attribute__((aligned(8)));

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
                    if (npu_dma_ld(&npu,
                          (uint32_t)(uintptr_t)(&test_A[(m_off+r)*K + k_off]),
                          0x0000u + (uint32_t)(r * 16), 16) != 0) {
                      fail = 1; break;
                    }
                    uart_putc('a');
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

                  if (fail) { uart_putc('a'); pipeline_abort = 1; break; }
                  a_dma_ok = 1;
                  uart_putc('A');
              }

              // --------------------------------------------------
              // B-DMA: LOAD B tile rows into WSRAM
              // --------------------------------------------------
              {
                  int fail = 0;
                  for (int r = 0; r < 16; r++) {
                    if (npu_dma_ld(&npu,
                          (uint32_t)(uintptr_t)(&test_B[(k_off+r)*N + n_off]),
                          0x1000u + (uint32_t)(r * 16), 16) != 0) {
                      fail = 1; break;
                    }
                  }
                  if (fail) { uart_putc('b'); pipeline_abort = 1; break; }
                  b_dma_ok = 1;
                  uart_putc('B');
              }

              // --------------------------------------------------
              // GEMM: load descriptor, issue, wait for done
              // --------------------------------------------------
              {
                  int fail = 0;
                  gemm_desc_t desc;
                  memset(&desc, 0, sizeof(desc));
                  desc.M = 1;             // one 16-row tile
                  desc.N = 1;             // one 16-col tile
                  desc.K = 1;             // one 16-deep tile
                  desc.a_sram_bank = 0;   // ASRAM
                  desc.b_sram_bank = 1;   // WSRAM
                  desc.o_sram_bank = 2;   // OSRAM
                  desc.out_scale_mul = 1;
                  // Load descriptor into DSRAM via driver DMA path.
                  int dp = npu_load_descriptor(&npu, &desc, sizeof(desc));
                  if (dp < 0) { fail = 1; }
                  else if (npu_issue(&npu, 0x01 /* OP_GEMM */, (uint32_t)dp) != 0) {
                      fail = 1;
                  }
                  else if (npu_wait_done(&npu, 1000000u) != 0) {
                      fail = 1;
                      uart_putc('D');  // Debug dump
                      uart_puthex32(csr_debug_read());
                  }

                  // Verify GEMM finished: CSR_STATUS.BUSY must be 0
                  if (!fail) {
                      volatile uint32_t *csr_status =
                          (volatile uint32_t *)(0x10000000u + 0x04u);
                      if (*csr_status & 1u) {
                          fail = 1;
                      }
                  }

                  // Final busy check with debug on timeout
                  if (!fail) {
                      if (wait_busy_clear('G') != 0) {  // 'G' = GEMM busy clear
                          fail = 1;
                      }
                  }

                  if (fail) { uart_putc('g'); pipeline_abort = 1; break; }
                  gemm_ok = 1;
                  uart_putc('G');
              }

              // --------------------------------------------------
              // STORE: DMA STORE C tile rows from OSRAM → ext_mem
              // --------------------------------------------------
              {
                  int fail = 0;
                  for (int r = 0; r < 16; r++) {
                    uint32_t ext_c = (uint32_t)(uintptr_t)(
                        &gemm_result[(m_off + r) * N + n_off]);
                    uint32_t sro   = 0x2000u + (uint32_t)(r * 32);
                    if (npu_dma_st(&npu, ext_c, sro, 32) != 0) {
                      fail = 1; break;
                    }
                  }
                  if (fail) { uart_putc('s'); pipeline_abort = 1; break; }
                  store_ok = 1;
                  uart_putc('S');
              }

            }  // kt
          }  // nt
        }  // mt
    }


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
