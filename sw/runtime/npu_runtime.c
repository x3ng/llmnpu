// ============================================================
// npu_runtime.c — NPU high-level operator implementation
//
// GEMM   : tiled across M / N in 16x16 output tiles.  For each output
//          tile, up to 16 K-tiles (K<=256) are loaded as one slab and
//          issued with a single GEMM descriptor.
//
// ReLU / GELU : load vector → SRAM, issue activation opcode,
//          wait, DMA store result back.
// ============================================================

#include "npu_runtime.h"
#include "../driver/npu_csr.h"
#include <string.h>

// ---------------------------------------------------------------
// Internal constants
// ---------------------------------------------------------------

#define TILE_DIM        16          // systolic array rows / cols
#define TILE_BYTES      (TILE_DIM * TILE_DIM)           // 256 B  (int8 tile)
#define TILE_OUT_BYTES  (TILE_DIM * TILE_DIM * (int)sizeof(int16_t)) // 512 B

#define OP_GEMM         0x01u       // from isa_defines.svh
#define OP_ACT_RELU     0x20u
#define OP_ACT_GELU     0x21u
#define OP_ACT_RELU6    0x24u
#define OP_ACT_CLIP     0x25u

#define GEMM_TIMEOUT_US 1000000u    // 1 second

// SRAM bank indices (matched to crossbar addr[15:12] decode)
#define BANK_A          0u
#define BANK_W          1u
#define BANK_O          2u

// ---------------------------------------------------------------
// Activation descriptor (internal only — 4 bytes packed)
// ---------------------------------------------------------------

typedef struct {
    uint16_t length;     // number of int8 elements
    uint16_t reserved;   // zero
} __attribute__((packed)) act_desc_t;

// ---------------------------------------------------------------
// npu_rt_gemm  –  tiled GEMM with automatic accumulation
// ---------------------------------------------------------------

int npu_rt_gemm(npu_dev_t *d, const npu_gemm_args_t *args)
{
    const int M = args->M, N = args->N, K = args->K;
    const int8_t *A = args->A, *B = args->B;
    int16_t      *C = args->C;

    // --- dimension alignment check ----------------------------------
    if ((M & (TILE_DIM - 1)) || (N & (TILE_DIM - 1)) || (K & (TILE_DIM - 1)))
        return -10;   // unaligned dimension

    const int m_tiles = M / TILE_DIM;
    const int n_tiles = N / TILE_DIM;
    const int k_tiles = K / TILE_DIM;
    if (k_tiles > 16)
        return -11;  // current RTL descriptor path supports K<=256

    // --- walk output tiles ------------------------------------------
    for (int mt = 0; mt < m_tiles; mt++) {
        const int m_off = mt * TILE_DIM;

        for (int nt = 0; nt < n_tiles; nt++) {
            const int n_off = nt * TILE_DIM;

            // DMA load A slab -> A-SRAM.  Each row stores K bytes
            // contiguously so the GPL can feed K steps from one row buffer.
            for (int r = 0; r < TILE_DIM; r++) {
                uint32_t ext_a = (uint32_t)(uintptr_t)(&A[(m_off + r) * K]);
                uint32_t sra = ASRAM_BASE + (uint32_t)(r * K);
                if (npu_dma_ld(d, ext_a, sra, (uint32_t)K) != 0)
                    return -1;
            }

            // DMA load B slab -> W-SRAM as K rows of 16 columns.
            for (int k = 0; k < K; k++) {
                uint32_t ext_b = (uint32_t)(uintptr_t)(&B[k * N + n_off]);
                uint32_t srb = WSRAM_BASE + (uint32_t)(k * TILE_DIM);
                if (npu_dma_ld(d, ext_b, srb, TILE_DIM) != 0)
                    return -2;
            }

            // Build and issue one GEMM descriptor for the whole K slab.
            gemm_desc_t desc;
            memset(&desc, 0, sizeof(desc));
            // One RTL issue computes one 16x16 output tile.  M/N tiling is
            // handled by these software loops; K tiling is handled by RTL.
            desc.M             = 1;
            desc.N             = 1;
            desc.K             = (uint16_t)k_tiles;
            desc.a_sram_bank   = BANK_A;
            desc.b_sram_bank   = BANK_W;
            desc.o_sram_bank   = BANK_O;
            desc.a_zp          = (uint8_t)args->a_zp;
            desc.b_zp          = (uint8_t)args->b_zp;
            desc.out_scale_shr = args->out_scale_shr;
            desc.out_scale_mul = (int16_t)args->out_scale_mul;

            int desc_ptr = npu_load_descriptor(d, &desc, sizeof(desc));
            if (desc_ptr < 0)
                return -3;

            if (npu_issue(d, OP_GEMM, (uint32_t)desc_ptr) != 0)
                return -4;

            if (npu_wait_done(d, GEMM_TIMEOUT_US) != 0)
                return -5;

            // DMA store C tile from O-SRAM → external (row by row)
            //   O-SRAM[r*32 .. r*32+31]  →  C[m_off+r][n_off .. n_off+15]
            for (int r = 0; r < TILE_DIM; r++) {
                uint32_t ext_c = (uint32_t)(uintptr_t)(
                    &C[(m_off + r) * N + n_off]);
                uint32_t sro   = OSRAM_BASE
                               + (uint32_t)(r * TILE_DIM * (int)sizeof(int16_t));
                uint32_t len   = (uint32_t)(TILE_DIM * (int)sizeof(int16_t));
                if (npu_dma_st(d, ext_c, sro, len) != 0)
                    return -6;
            }
        }
    }

    return 0;
}

// ---------------------------------------------------------------
// npu_rt_relu  –  element-wise ReLU via OP_ACT_RELU
// ---------------------------------------------------------------

int npu_rt_relu(npu_dev_t *d, const int8_t *in, int8_t *out, int len)
{
    act_desc_t desc;
    int desc_ptr;

    // DMA load input vector → A-SRAM
    if (npu_dma_ld(d, (uint32_t)(uintptr_t)in, ASRAM_BASE, (uint32_t)len) != 0)
        return -1;

    // Build activation descriptor
    memset(&desc, 0, sizeof(desc));
    desc.length = (uint16_t)len;

    desc_ptr = npu_load_descriptor(d, &desc, sizeof(desc));
    if (desc_ptr < 0)
        return -2;

    // Issue ReLU
    if (npu_issue(d, OP_ACT_RELU, (uint32_t)desc_ptr) != 0)
        return -3;

    // Wait for completion
    if (npu_wait_done(d, GEMM_TIMEOUT_US) != 0)
        return -4;

    // DMA store result O-SRAM → output
    if (npu_dma_st(d, (uint32_t)(uintptr_t)out, OSRAM_BASE, (uint32_t)len) != 0)
        return -5;

    return 0;
}

// ---------------------------------------------------------------
// npu_rt_gelu  –  element-wise GELU via OP_ACT_GELU
// ---------------------------------------------------------------

int npu_rt_gelu(npu_dev_t *d, const int8_t *in, int8_t *out, int len)
{
    act_desc_t desc;
    int desc_ptr;

    // DMA load input vector → A-SRAM
    if (npu_dma_ld(d, (uint32_t)(uintptr_t)in, ASRAM_BASE, (uint32_t)len) != 0)
        return -1;

    memset(&desc, 0, sizeof(desc));
    desc.length = (uint16_t)len;

    desc_ptr = npu_load_descriptor(d, &desc, sizeof(desc));
    if (desc_ptr < 0)
        return -2;

    // Issue GELU
    if (npu_issue(d, OP_ACT_GELU, (uint32_t)desc_ptr) != 0)
        return -3;

    if (npu_wait_done(d, GEMM_TIMEOUT_US) != 0)
        return -4;

    // DMA store result O-SRAM → output
    if (npu_dma_st(d, (uint32_t)(uintptr_t)out, OSRAM_BASE, (uint32_t)len) != 0)
        return -5;

    return 0;
}
