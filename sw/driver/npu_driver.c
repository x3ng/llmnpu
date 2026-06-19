// ============================================================
// npu_driver.c — NPU bare-metal C driver implementation
// ============================================================

#include "npu_driver.h"
#include "npu_csr.h"

// ------------------------------------------------------------------
// MMIO access helpers
// ------------------------------------------------------------------
static inline uint32_t csr_rd(const npu_dev_t *d, uint32_t off)
{
    return *(const volatile uint32_t *)(d->mmio_base + off);
}

static inline void csr_wr(const npu_dev_t *d, uint32_t off, uint32_t val)
{
    *(volatile uint32_t *)(d->mmio_base + off) = val;
}

// ------------------------------------------------------------------
// Descriptor SRAM bump allocator (lives in DSRAM)
// ------------------------------------------------------------------
static uint32_t _dsram_next = DSRAM_BASE;

// ------------------------------------------------------------------
// DMA transfer timeout (busy-loop iterations)
// ------------------------------------------------------------------
#define DMA_TIMEOUT_LOOPS  100000u

static int _dma_xfer(npu_dev_t *d,
                     uint32_t ext_addr, uint32_t sram_off, uint32_t len,
                     int is_st)
{
    volatile uint32_t to;

    // Wait for any previous DMA to finish
    to = DMA_TIMEOUT_LOOPS;
    while (csr_rd(d, CSR_DMA_CSR3) & DMA_CSR3_BUSY) {
        if (--to == 0) return -1;
    }

    // Program the four DMA CSRs
    csr_wr(d, CSR_DMA_CSR0, ext_addr);
    csr_wr(d, CSR_DMA_CSR1, sram_off);
    csr_wr(d, CSR_DMA_CSR2, len);

    uint32_t ctrl = DMA_CSR3_START;
    if (is_st) ctrl |= DMA_CSR3_DIR_ST;
    csr_wr(d, CSR_DMA_CSR3, ctrl);

    // Wait for transfer completion
    to = DMA_TIMEOUT_LOOPS;
    while (csr_rd(d, CSR_DMA_CSR3) & DMA_CSR3_BUSY) {
        if (--to == 0) return -1;
    }

    // Check for DMA fault
    if (csr_rd(d, CSR_DMA_CSR3) & DMA_CSR3_FAULT) return -1;

    return 0;
}

// ------------------------------------------------------------------
// Approximate microsecond busy-wait (tuned for ~100 MHz CPU)
// ------------------------------------------------------------------
static void _busy_wait_us(uint32_t us)
{
    volatile uint32_t i;
    // ~10 loop iterations per microsecond at 100 MHz
    for (i = 0; i < us * 10u; i++) {
        __asm__ volatile ("" : : : "memory");
    }
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------

void npu_init(npu_dev_t *d, uintptr_t mmio_base)
{
    d->mmio_base = mmio_base;

    // Pulse RESET
    csr_wr(d, CSR_CTRL, CSR_CTRL_RESET);
    _busy_wait_us(100);
    csr_wr(d, CSR_CTRL, 0);
    _busy_wait_us(100);

    // Clear any stale IRQs (W1C on IRQ_STAT)
    csr_wr(d, CSR_IRQ_STAT, 0xFFFFFFFFu);

    // Reset DSRAM allocator
    _dsram_next = DSRAM_BASE;
}

int npu_load_descriptor(npu_dev_t *d, const void *desc, size_t n)
{
    if (n == 0) return -1;

    // Word-align the allocation size
    uint32_t size = (uint32_t)n;
    uint32_t aligned = (size + 3u) & ~3u;

    // Check DSRAM space
    if (_dsram_next + aligned > DSRAM_BASE + DSRAM_SIZE) return -1;

    uint32_t offset = _dsram_next;

    // DMA the descriptor from host memory into DSRAM
    int ret = _dma_xfer(d, (uint32_t)(uintptr_t)desc, offset, size, 0);
    if (ret != 0) return -1;

    _dsram_next += aligned;
    return (int)offset;
}

int npu_dma_ld(npu_dev_t *d, uint32_t ext_addr, uint32_t sram_off,
               uint32_t len)
{
    return _dma_xfer(d, ext_addr, sram_off, len, 0);
}

int npu_dma_st(npu_dev_t *d, uint32_t ext_addr, uint32_t sram_off,
               uint32_t len)
{
    return _dma_xfer(d, ext_addr, sram_off, len, 1);
}

int npu_issue(npu_dev_t *d, uint8_t opcode, uint32_t desc_ptr)
{
    // Refuse if the NPU is already executing
    if (csr_rd(d, CSR_STATUS) & CSR_STATUS_BUSY) return -1;

    (void)opcode;  // reserved for future opcode-based dispatch

    csr_wr(d, CSR_DESC_PTR, desc_ptr);
    csr_wr(d, CSR_CTRL, CSR_CTRL_START);
    return 0;
}

int npu_wait_done(npu_dev_t *d, uint32_t timeout_us)
{
    uint32_t elapsed = 0u;
    const uint32_t step_us = 100u;  // poll every ~100 µs

    while (elapsed < timeout_us) {
        if (!(csr_rd(d, CSR_STATUS) & CSR_STATUS_BUSY)) return 0;
        _busy_wait_us(step_us);
        elapsed += step_us;
    }

    // Final check after timeout expires
    return (csr_rd(d, CSR_STATUS) & CSR_STATUS_BUSY) ? -1 : 0;
}

void npu_irq_handler(npu_dev_t *d)
{
    uint32_t stat = csr_rd(d, CSR_IRQ_STAT);

    // Write-1-to-clear all pending IRQ flags
    csr_wr(d, CSR_IRQ_STAT, stat);

    // Application callbacks can be wired here, keyed on stat bits.
    (void)stat;
}
