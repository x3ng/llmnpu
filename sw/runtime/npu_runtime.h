// ============================================================
// npu_runtime.h — NPU high-level operator API
// Provides tiled GEMM and activation functions on top of the
// bare-metal driver (npu_driver.h).
// ============================================================

#ifndef NPU_RUNTIME_H
#define NPU_RUNTIME_H

#include <stdint.h>
#include "../driver/npu_driver.h"

// --- GEMM arguments (caller-facing) ---
typedef struct {
    int            M, N, K;        // matrix dimensions
    const int8_t  *A, *B;          // input matrices (int8, row-major)
    int16_t       *C;              // output matrix (int16, row-major)
    int8_t         a_zp, b_zp;     // zero points
    int32_t        out_scale_mul;  // requant multiplier (signed)
    uint8_t        out_scale_shr;  // requant right-shift
} npu_gemm_args_t;

// --- GEMM descriptor — bit-exact match of the SystemVerilog packed struct
//     from rtl/include/isa_defines.svh (152 bits = 19 bytes packed) ---
typedef struct {
    uint16_t M, N, K;               // tile counts (each x16)
    uint8_t  a_sram_bank;           // activation SRAM bank index
    uint8_t  b_sram_bank;           // weight    SRAM bank index
    uint8_t  o_sram_bank;           // output    SRAM bank index
    uint8_t  a_zp, b_zp;            // INT8 zero points
    uint16_t reserved;              // reserved (zero)
    uint16_t out_scale_shr;         // INT16→INT8 requant right-shift
    int16_t  out_scale_mul;         // requant multiplier (signed)
    uint8_t  relu;                  // post-GEMM ReLU flag
    uint8_t  out_zp;                // output zero point
} __attribute__((packed)) gemm_desc_t;

// --- Public API ---

// Tiled GEMM: C[m,n] = sum_k (A[m,k] - a_zp) * (B[k,n] - b_zp)
// Dimensions must be multiples of 16.
// Returns 0 on success, negative on error.
int npu_rt_gemm(npu_dev_t *d, const npu_gemm_args_t *args);

// Element-wise ReLU: out[i] = max(in[i], 0)
int npu_rt_relu(npu_dev_t *d, const int8_t *in, int8_t *out, int len);

// Element-wise GELU approximation (int8 quantised)
int npu_rt_gelu(npu_dev_t *d, const int8_t *in, int8_t *out, int len);

#endif // NPU_RUNTIME_H
