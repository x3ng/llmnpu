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
#include "../driver/npu_abi.h"

// ---------------------------------------------------------------
// Internal constants
// ---------------------------------------------------------------

#define TILE_DIM        16          // systolic array rows / cols
#define TILE_BYTES      (TILE_DIM * TILE_DIM)           // 256 B  (int8 tile)
#define TILE_OUT_BYTES  (TILE_DIM * TILE_DIM * (int)sizeof(int16_t)) // 512 B

#define GEMM_TIMEOUT_US 1000000u    // 1 second

// SRAM bank indices (matched to crossbar addr[15:12] decode)
#define BANK_A          0u
#define BANK_W          1u
#define BANK_O          2u

static int8_t  a_slab[TILE_DIM * 256] __attribute__((aligned(8)));
static int8_t  b_slab[256 * TILE_DIM] __attribute__((aligned(8)));
static int16_t c_tile[TILE_DIM * TILE_DIM] __attribute__((aligned(8)));

static void rt_memcpy(void *dst, const void *src, unsigned int n)
{
    uint8_t *d = (uint8_t *)dst;
    const uint8_t *s = (const uint8_t *)src;

    for (unsigned int i = 0; i < n; i++)
        d[i] = s[i];
}

static void rt_memzero(void *dst, unsigned int n)
{
    uint8_t *d = (uint8_t *)dst;

    for (unsigned int i = 0; i < n; i++)
        d[i] = 0;
}

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

            // Pack and DMA one complete A slab.  The ping-pong contract is
            // tile-level: one LOAD marks one fill bank ready.
            for (int r = 0; r < TILE_DIM; r++) {
                rt_memcpy(&a_slab[r * K], &A[(m_off + r) * K], (unsigned int)K);
            }
            if (npu_dma_ld(d, (uint32_t)(uintptr_t)a_slab,
                           ASRAM_BASE, (uint32_t)(TILE_DIM * K)) != 0)
                return -1;

            // Pack B slab as K rows of 16 output columns.
            for (int k = 0; k < K; k++) {
                rt_memcpy(&b_slab[k * TILE_DIM],
                          &B[k * N + n_off],
                          TILE_DIM);
            }
            if (npu_dma_ld(d, (uint32_t)(uintptr_t)b_slab,
                           WSRAM_BASE, (uint32_t)(K * TILE_DIM)) != 0)
                return -2;

            // Build and issue one GEMM descriptor for the whole K slab.
            gemm_desc_t desc;
            rt_memzero(&desc, sizeof(desc));
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

            if (npu_issue(d, NPU_OP_GEMM, (uint32_t)desc_ptr) != 0)
                return -4;

            if (npu_wait_done(d, GEMM_TIMEOUT_US) != 0)
                return -5;

            // DMA STORE one complete C tile, then scatter into caller layout.
            if (npu_dma_st(d, (uint32_t)(uintptr_t)c_tile,
                           OSRAM_BASE, TILE_OUT_BYTES) != 0)
                return -6;
            for (int r = 0; r < TILE_DIM; r++) {
                rt_memcpy(&C[(m_off + r) * N + n_off],
                          &c_tile[r * TILE_DIM],
                          TILE_DIM * (int)sizeof(int16_t));
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
    npu_sfu_desc_t desc;
    int desc_ptr;

    if (len <= 0 || len > (int)NPU_PP_SFU_SIZE)
        return -10;

    if (npu_dma_ld(d, (uint32_t)(uintptr_t)in,
                   NPU_PP_SFU_IN_BASE, (uint32_t)len) != 0)
        return -1;

    rt_memzero(&desc, sizeof(desc));
    desc.word0 = NPU_SFU_DESC_WORD0((uint32_t)len, NPU_OP_ACT_RELU, 0u);
    desc.in_addr = NPU_PP_SFU_IN_BASE;
    desc.out_addr = NPU_PP_SFU_OUT_BASE;
    desc.scale = NPU_SFU_DESC_SCALE(1u, 0u);

    desc_ptr = npu_load_descriptor(d, &desc, sizeof(desc));
    if (desc_ptr < 0)
        return -2;

    if (npu_issue(d, NPU_OP_ACT_RELU, (uint32_t)desc_ptr) != 0)
        return -3;

    if (npu_wait_done(d, GEMM_TIMEOUT_US) != 0)
        return -4;

    if (npu_dma_st(d, (uint32_t)(uintptr_t)out,
                   NPU_PP_SFU_OUT_BASE, (uint32_t)len) != 0)
        return -5;

    return 0;
}

// ---------------------------------------------------------------
// npu_rt_gelu  –  element-wise GELU via OP_ACT_GELU
// ---------------------------------------------------------------

int npu_rt_gelu(npu_dev_t *d, const int8_t *in, int8_t *out, int len)
{
    npu_sfu_desc_t desc;
    int desc_ptr;

    if (len <= 0 || len > (int)NPU_PP_SFU_SIZE)
        return -10;

    if (npu_dma_ld(d, (uint32_t)(uintptr_t)in,
                   NPU_PP_SFU_IN_BASE, (uint32_t)len) != 0)
        return -1;

    rt_memzero(&desc, sizeof(desc));
    desc.word0 = NPU_SFU_DESC_WORD0((uint32_t)len, NPU_OP_ACT_GELU, 0u);
    desc.in_addr = NPU_PP_SFU_IN_BASE;
    desc.out_addr = NPU_PP_SFU_OUT_BASE;
    desc.scale = NPU_SFU_DESC_SCALE(1u, 0u);

    desc_ptr = npu_load_descriptor(d, &desc, sizeof(desc));
    if (desc_ptr < 0)
        return -2;

    if (npu_issue(d, NPU_OP_ACT_GELU, (uint32_t)desc_ptr) != 0)
        return -3;

    if (npu_wait_done(d, GEMM_TIMEOUT_US) != 0)
        return -4;

    if (npu_dma_st(d, (uint32_t)(uintptr_t)out,
                   NPU_PP_SFU_OUT_BASE, (uint32_t)len) != 0)
        return -5;

    return 0;
}
