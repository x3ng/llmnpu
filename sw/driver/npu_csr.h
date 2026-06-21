// ============================================================
// npu_csr.h — NPU Control/Status Register map
// Bare-metal: no libc dependency.
// ============================================================

#ifndef NPU_CSR_H
#define NPU_CSR_H

#include <stdint.h>

// --- NPU MMIO base address ---
#define NPU_CSR_BASE  0x10000000u

// --- Register byte offsets (32-bit registers) ---
#define CSR_CTRL         0x00
#define CSR_STATUS       0x04
#define CSR_PC           0x08
#define CSR_DESC_PTR     0x10
#define CSR_DMA_CSR0     0x20
#define CSR_DMA_CSR1     0x28
#define CSR_DMA_CSR2     0x30
#define CSR_DMA_CSR3     0x38
#define CSR_IRQ_EN       0x40
#define CSR_IRQ_STAT     0x44
#define CSR_PERF_CYCLE   0x80

// --- CSR_CTRL bits ---
#define CSR_CTRL_START    (1u << 0)
#define CSR_CTRL_RESET    (1u << 1)
#define CSR_CTRL_HALT     (1u << 2)
#define CSR_CTRL_OPCODE_SHIFT 8
#define CSR_CTRL_OPCODE_MASK  (0xFFu << CSR_CTRL_OPCODE_SHIFT)
#define CSR_CTRL_OPCODE(op)   (((uint32_t)(op) & 0xFFu) << CSR_CTRL_OPCODE_SHIFT)

// --- CSR_STATUS bits ---
#define CSR_STATUS_BUSY       (1u << 0)
#define CSR_STATUS_IRQ_PEND   (1u << 1)

// --- DMA CSR3 control bits ---
// RTL stores CSR3 as a plain RW control register. Hardware completion is
// reported through CSR_STATUS.BUSY/IRQ_STAT, not CSR3 status bits.
#define DMA_CSR3_START    (1u << 0)
#define DMA_CSR3_DIR_ST   (1u << 1)   // 0=load (ext→SRAM), 1=store (SRAM→ext)
#define DMA_CSR3_MODE_2D  (1u << 2)
#define DMA_CSR3_EXT_STRIDE(stride) (((uint32_t)(stride) & 0xFFFFu) << 16)

// --- DMA CSR register roles ---
// CSR0 : external AXI address
// CSR1 : [15:0] SRAM byte offset, [31:16] 2D SRAM stride
// CSR2 : [15:0] transfer length / 2D row bytes, [31:16] 2D rows
// CSR3 : control, [31:16] 2D external stride

// --- IRQ bits (IRQ_EN and IRQ_STAT share layout) ---
#define IRQ_DONE          (1u << 0)
#define IRQ_FAULT         (1u << 1)

// --- Internal SRAM layout (16-bit NPU-internal address space) ---
#define ASRAM_BASE  0x0000u
#define WSRAM_BASE  0x1000u
#define OSRAM_BASE  0x2000u
#define DSRAM_BASE  0x3000u

#define DSRAM_SIZE  57344u

#endif // NPU_CSR_H
