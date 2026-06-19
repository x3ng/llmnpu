// ============================================================
// npu_driver.h — NPU bare-metal C driver API (per spec §11.1)
// ============================================================

#ifndef NPU_DRIVER_H
#define NPU_DRIVER_H

#include <stddef.h>
#include <stdint.h>

// --- Device handle ---
typedef struct {
    uintptr_t mmio_base;
} npu_dev_t;

// --- API ---

// Initialise the NPU device: store base address, assert/deassert reset,
// clear pending IRQs.  Call once at boot.
void npu_init(npu_dev_t *d, uintptr_t mmio_base);

// Copy a descriptor blob of n bytes from host memory into NPU DSRAM
// (using the internal DMA engine).  Returns the 16-bit SRAM byte
// offset where the descriptor was placed, or -1 on failure
// (e.g. DSRAM exhausted or DMA fault).
int npu_load_descriptor(npu_dev_t *d, const void *desc, size_t n);

// DMA load:  copy len bytes from external address ext_addr into
// NPU SRAM at byte offset sram_off.  Returns 0 on success, -1 on error.
int npu_dma_ld(npu_dev_t *d, uint32_t ext_addr, uint32_t sram_off,
               uint32_t len);

// DMA store: copy len bytes from NPU SRAM at byte offset sram_off
// to external address ext_addr.  Returns 0 on success, -1 on error.
int npu_dma_st(npu_dev_t *d, uint32_t ext_addr, uint32_t sram_off,
               uint32_t len);

// Issue a compute operation: write desc_ptr to CSR_DESC_PTR and
// pulse CSR_CTRL.START.  opcode is available for future dispatch
// (currently reserved).  Returns 0 on success, -1 if the NPU is
// already busy.
int npu_issue(npu_dev_t *d, uint8_t opcode, uint32_t desc_ptr);

// Spin-wait for the NPU to de-assert CSR_STATUS.BUSY.
// Returns 0 when done, -1 on timeout (timeout_us microseconds).
int npu_wait_done(npu_dev_t *d, uint32_t timeout_us);

// IRQ handler: read CSR_IRQ_STAT, acknowledge (W1C) all pending
// flags.  Extend with application-specific callbacks as needed.
void npu_irq_handler(npu_dev_t *d);

#endif // NPU_DRIVER_H
