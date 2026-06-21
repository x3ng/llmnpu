// ============================================================
// npu_abi.h -- Software-visible NPU ABI constants and descriptors
// ============================================================

#ifndef NPU_ABI_H
#define NPU_ABI_H

#include <stdint.h>
#include "npu_csr.h"

// --- Opcodes (match rtl/include/isa_defines.svh) ---
#define NPU_OP_GEMM        0x01u
#define NPU_OP_GEMM_SCALE  0x02u
#define NPU_OP_VADD        0x10u
#define NPU_OP_VMOV        0x11u
#define NPU_OP_ACT_RELU    0x20u
#define NPU_OP_ACT_GELU    0x21u
#define NPU_OP_ACT_SIGMOID 0x22u
#define NPU_OP_ACT_TANH    0x23u
#define NPU_OP_ACT_RELU6   0x24u
#define NPU_OP_ACT_CLIP    0x25u
#define NPU_OP_QUANT       0x30u
#define NPU_OP_DEQUANT     0x31u
#define NPU_OP_DMA_LD      0x40u
#define NPU_OP_DMA_ST      0x41u
#define NPU_OP_DMA_2D      0x42u
#define NPU_OP_SYNC        0xF0u
#define NPU_OP_WFI         0xF1u
#define NPU_OP_NOP         0xFFu

// --- VALU sub-opcodes ---
#define NPU_VOPT_ADD   0x00u
#define NPU_VOPT_SUB   0x01u
#define NPU_VOPT_MUL   0x02u
#define NPU_VOPT_MIN   0x03u
#define NPU_VOPT_MAX   0x04u
#define NPU_VOPT_AND   0x05u
#define NPU_VOPT_OR    0x06u
#define NPU_VOPT_XOR   0x07u
#define NPU_VOPT_SLL   0x08u
#define NPU_VOPT_SRA   0x09u
#define NPU_VOPT_CMOV  0x0Au
#define NPU_VOPT_BCAST 0x80u

// --- Ping-pong SRAM windows (match rtl/include/npu_defines.svh) ---
#define NPU_PP_GEMM_A_SIZE  256u
#define NPU_PP_GEMM_B_SIZE  256u
#define NPU_PP_GEMM_P_SIZE  512u
#define NPU_PP_VALU_SIZE    256u
#define NPU_PP_SFU_SIZE     256u

#define NPU_PP_GEMM_P_OFFSET   0u
#define NPU_PP_VALU_IN_OFFSET  (NPU_PP_GEMM_P_SIZE * 2u)
#define NPU_PP_VALU_OUT_OFFSET (NPU_PP_VALU_IN_OFFSET + (NPU_PP_VALU_SIZE * 2u))
#define NPU_PP_SFU_IN_OFFSET   (NPU_PP_VALU_OUT_OFFSET + (NPU_PP_VALU_SIZE * 2u))
#define NPU_PP_SFU_OUT_OFFSET  (NPU_PP_SFU_IN_OFFSET + (NPU_PP_SFU_SIZE * 2u))

#define NPU_PP_GEMM_P_BASE   (OSRAM_BASE + NPU_PP_GEMM_P_OFFSET)
#define NPU_PP_VALU_IN_BASE  (OSRAM_BASE + NPU_PP_VALU_IN_OFFSET)
#define NPU_PP_VALU_OUT_BASE (OSRAM_BASE + NPU_PP_VALU_OUT_OFFSET)
#define NPU_PP_SFU_IN_BASE   (OSRAM_BASE + NPU_PP_SFU_IN_OFFSET)
#define NPU_PP_SFU_OUT_BASE  (OSRAM_BASE + NPU_PP_SFU_OUT_OFFSET)

// --- Descriptor layouts ---
typedef struct {
    uint16_t M;
    uint16_t N;
    uint16_t K;
    uint8_t  a_sram_bank;
    uint8_t  b_sram_bank;
    uint8_t  o_sram_bank;
    uint8_t  a_zp;
    uint8_t  b_zp;
    uint16_t reserved;
    uint16_t out_scale_shr;
    int16_t  out_scale_mul;
    uint8_t  relu;
    uint8_t  out_zp;
} __attribute__((packed)) npu_gemm_desc_t;

typedef struct {
    uint32_t word0;       // [15:0] len, [23:16] opt, [24] scalar_b
    uint32_t in0_addr;    // [15:0] input 0 SRAM byte address
    uint32_t in1_addr;    // [15:0] input 1 SRAM byte address
    uint32_t out_addr;    // [15:0] output SRAM byte address
    uint32_t scalar;      // [7:0] scalar operand
} __attribute__((packed)) npu_valu_desc_t;

typedef struct {
    uint32_t word0;       // [15:0] len, [23:16] opcode, [31:24] zp/clip
    uint32_t in_addr;     // [15:0] input SRAM byte address
    uint32_t out_addr;    // [15:0] output SRAM byte address
    uint32_t scale;       // [15:0] scale_mul, [23:16] scale_shr
} __attribute__((packed)) npu_sfu_desc_t;

#define NPU_VALU_DESC_WORD0(len, opt, scalar_b) \
    ((((uint32_t)(len)) & 0xFFFFu) | \
     ((((uint32_t)(opt)) & 0xFFu) << 16) | \
     ((scalar_b) ? (1u << 24) : 0u))

#define NPU_SFU_DESC_WORD0(len, opcode, zp) \
    ((((uint32_t)(len)) & 0xFFFFu) | \
     ((((uint32_t)(opcode)) & 0xFFu) << 16) | \
     ((((uint32_t)(zp)) & 0xFFu) << 24))

#define NPU_SFU_DESC_SCALE(scale_mul, scale_shr) \
    ((((uint32_t)(scale_mul)) & 0xFFFFu) | \
     ((((uint32_t)(scale_shr)) & 0xFFu) << 16))

typedef char npu_abi_gemm_desc_size_check[(sizeof(npu_gemm_desc_t) == 19u) ? 1 : -1];
typedef char npu_abi_valu_desc_size_check[(sizeof(npu_valu_desc_t) == 20u) ? 1 : -1];
typedef char npu_abi_sfu_desc_size_check[(sizeof(npu_sfu_desc_t) == 16u) ? 1 : -1];

#endif // NPU_ABI_H
