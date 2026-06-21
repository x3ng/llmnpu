// ============================================================
// NPU Instruction Set Architecture Definitions
// 32-bit fixed-length instructions per spec §4
// ============================================================

`ifndef ISA_DEFINES_SVH
`define ISA_DEFINES_SVH

// --- Opcodes (8-bit OP field, bits [31:24]) ---
`define OP_GEMM        8'h01
`define OP_GEMM_SCALE  8'h02
`define OP_VADD        8'h10
`define OP_VMUL        8'h10
`define OP_VMAX        8'h10
`define OP_VMOV        8'h11
`define OP_VCMP        8'h11
`define OP_ACT_RELU    8'h20
`define OP_ACT_GELU    8'h21
`define OP_ACT_SIGMOID 8'h22
`define OP_ACT_TANH    8'h23
`define OP_ACT_RELU6   8'h24
`define OP_ACT_CLIP    8'h25
`define OP_QUANT       8'h30
`define OP_DEQUANT     8'h31
`define OP_DMA_LD      8'h40
`define OP_DMA_ST      8'h41
`define OP_DMA_2D      8'h42
`define OP_SYNC        8'hF0
`define OP_WFI         8'hF1
`define OP_NOP         8'hFF

// --- VALU sub-opcodes (OPT field, bits [27:20]) ---
`define VOPT_ADD  8'h00
`define VOPT_SUB  8'h01
`define VOPT_MUL  8'h02
`define VOPT_MIN  8'h03
`define VOPT_MAX  8'h04
`define VOPT_AND  8'h05
`define VOPT_OR   8'h06
`define VOPT_XOR  8'h07
`define VOPT_SLL  8'h08
`define VOPT_SRA  8'h09
`define VOPT_CMOV 8'h0A
`define VOPT_BCAST 8'h80

// --- GEMM Descriptor (152 bits = 19 bytes, packed struct) ---
// GEMM IF/ID instruction descriptor reference:
//   [31:24] opcode (`OP_GEMM or `OP_GEMM_SCALE)
//   [23:8]  descriptor word offset from CSR_DESC_PTR
//   [7:0]   auxiliary field
typedef struct packed {
    logic [15:0] M, N, K;        // tile counts (each x16)
                                  // RTL issue path consumes one M/N tile;
                                  // runtime loops M/N output tiles.
    logic [7:0]  a_sram_bank;
    logic [7:0]  b_sram_bank;
    logic [7:0]  o_sram_bank;
    logic [7:0]  a_zp, b_zp;     // INT8 zero points
    logic [15:0] reserved;
    logic [15:0] out_scale_shr;  // INT16→INT8 requant right-shift
    logic [15:0] out_scale_mul;  // requant multiplier (signed)
    logic [7:0]  relu;
    logic [7:0]  out_zp;
} gemm_desc_t;

// --- VALU Descriptor (20 bytes, 5 words) ---
// CSR issue path consumes this descriptor from CSR_DESC_PTR:
//   word0[15:0]  len_bytes, 0 means one 256B tile
//   word0[23:16] VALU sub-opcode (`VOPT_*)
//   word0[24]    use scalar_b from word4[7:0] instead of in1_addr
//   word1[15:0]  in0 SRAM byte address
//   word2[15:0]  in1 SRAM byte address
//   word3[15:0]  out SRAM byte address
//   word4[7:0]   scalar_b
typedef struct packed {
    logic [15:0] len_bytes;
    logic [7:0]  opt;
    logic        scalar_b;
    logic [6:0]  reserved0;
    logic [15:0] in0_addr;
    logic [15:0] reserved1;
    logic [15:0] in1_addr;
    logic [15:0] reserved2;
    logic [15:0] out_addr;
    logic [15:0] reserved3;
    logic [7:0]  scalar;
    logic [23:0] reserved4;
} valu_desc_t;

// --- SFU Descriptor (16 bytes, 4 words) ---
// CSR issue path consumes this descriptor from CSR_DESC_PTR when operating on
// the SFU input/output SRAM ping-pong windows:
//   word0[15:0]  len_bytes, 0 means one 256B tile
//   word0[23:16] SFU opcode (`OP_ACT_*, `OP_QUANT, `OP_DEQUANT)
//   word0[31:24] zero point / clip threshold
//   word1[15:0]  input SRAM byte address
//   word2[15:0]  output SRAM byte address
//   word3[15:0]  scale_mul
//   word3[23:16] scale_shr
typedef struct packed {
    logic [15:0] len_bytes;
    logic [7:0]  opcode;
    logic [7:0]  zp;
    logic [15:0] in_addr;
    logic [15:0] reserved0;
    logic [15:0] out_addr;
    logic [15:0] reserved1;
    logic [15:0] scale_mul;
    logic [7:0]  scale_shr;
    logic [7:0]  reserved2;
} sfu_desc_t;

`endif // ISA_DEFINES_SVH
