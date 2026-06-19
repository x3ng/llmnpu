// ============================================================
// NPU Shared Definitions — dimensions, types, SRAM layout
// ============================================================

`ifndef NPU_DEFINES_SVH
`define NPU_DEFINES_SVH

// --- Data types ---
typedef logic [7:0]  int8_t;
typedef logic [15:0] int16_t;
typedef logic [31:0] int32_t;
typedef logic [7:0]  uint8_t;
typedef logic [31:0] uint32_t;

// --- GEMM array dimensions ---
`define GEMM_ROWS 16
`define GEMM_COLS 16
`define GEMM_K_MAX 256

// --- VALU ---
`define VALU_LANES 64
`define VREGFILE_COUNT 32

// --- SRAM bank sizes (bytes) ---
`define ISRAM_SIZE  8192
`define WSRAM_SIZE  65536
`define ASRAM_SIZE  65536
`define OSRAM_SIZE  65536
`define DSRAM_SIZE  57344

// --- SRAM base addresses (NPU-internal, 16-bit) ---
`define ASRAM_BASE  16'h0000
`define WSRAM_BASE  16'h1000
`define OSRAM_BASE  16'h2000
`define DSRAM_BASE  16'h3000

// --- Ping-pong buffer sizes (bytes) ---
`define PP_GEMM_A_SIZE  256
`define PP_GEMM_B_SIZE  256
`define PP_GEMM_P_SIZE  512
`define PP_VALU_SIZE    256
`define PP_SFU_SIZE     256

// --- Crossbar master IDs ---
`define XBAR_M_DMA    2'b00
`define XBAR_M_GEMM   2'b01
`define XBAR_M_VALU   2'b10

// --- AXI configuration ---
`define AXI_DATA_WIDTH  64
`define AXI_ADDR_WIDTH  32
`define AXI_ID_WIDTH    4

`endif // NPU_DEFINES_SVH
