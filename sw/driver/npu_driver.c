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

#define DESC_BOUNCE_SIZE 256u
static uint8_t _desc_bounce[DESC_BOUNCE_SIZE] __attribute__((aligned(8)));

#define PROGRAM_DESC_BOUNCE_SIZE 512u
static uint8_t _program_desc_bounce[PROGRAM_DESC_BOUNCE_SIZE]
    __attribute__((aligned(8)));

// ------------------------------------------------------------------
// DMA transfer timeout (busy-loop iterations)
// ------------------------------------------------------------------
#define DMA_TIMEOUT_LOOPS  100000u
#define DMA_START_TIMEOUT_LOOPS  10000u

static int _wait_not_busy(npu_dev_t *d, uint32_t loops)
{
    while (csr_rd(d, CSR_STATUS) & CSR_STATUS_BUSY) {
        if (--loops == 0) return -1;
    }
    return 0;
}

static int _wait_started_or_done(npu_dev_t *d, uint32_t loops)
{
    while (1) {
        uint32_t status = csr_rd(d, CSR_STATUS);
        if (status & (CSR_STATUS_BUSY | CSR_STATUS_IRQ_PEND)) return 0;
        if (--loops == 0) return -1;
    }
}

static uint32_t _load_le32(const uint8_t *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void _mem_fence(void)
{
    __asm__ volatile ("fence" ::: "memory");
}

static int _dma_xfer(npu_dev_t *d,
                     uint32_t ext_addr, uint32_t sram_off, uint32_t len,
                     int is_st)
{
    if (len == 0) return -1;

    // Wait for any previous NPU/DMA/bridge work to finish.
    if (_wait_not_busy(d, DMA_TIMEOUT_LOOPS) != 0) return -1;

    // Clear stale completion before issuing this DMA.  Otherwise the
    // start-detection poll can observe the previous transfer's IRQ and
    // return before the new DMA/bridge has asserted BUSY.
    csr_wr(d, CSR_IRQ_STAT, IRQ_DONE);

    // Program the four DMA CSRs
    csr_wr(d, CSR_DMA_CSR0, ext_addr);
    csr_wr(d, CSR_DMA_CSR1, sram_off);
    csr_wr(d, CSR_DMA_CSR2, len);

    uint32_t ctrl = DMA_CSR3_START;
    if (is_st) ctrl |= DMA_CSR3_DIR_ST;
    csr_wr(d, CSR_DMA_CSR3, ctrl);

    // CSR3 is not hardware status. STATUS.BUSY is the RTL aggregate of
    // DMA engine busy plus the DMA<->SRAM bridge FSM, so it is the only
    // completion source that prevents back-to-back DMA hazards.
    if (_wait_started_or_done(d, DMA_START_TIMEOUT_LOOPS) != 0) return -1;
    return _wait_not_busy(d, DMA_TIMEOUT_LOOPS);
}

static int _dma_2d_xfer(npu_dev_t *d,
                        uint32_t ext_addr, uint32_t sram_off,
                        uint32_t rows, uint32_t row_bytes,
                        uint32_t ext_stride, uint32_t sram_stride,
                        int is_st)
{
    if (rows == 0 || row_bytes == 0) return -1;
    if (rows > 0xFFFFu || row_bytes > 0xFFFFu) return -1;
    if (ext_stride > 0xFFFFu || sram_stride > 0xFFFFu) return -1;
    if ((row_bytes & 7u) != 0) return -1;

    if (_wait_not_busy(d, DMA_TIMEOUT_LOOPS) != 0) return -1;
    csr_wr(d, CSR_IRQ_STAT, IRQ_DONE);

    csr_wr(d, CSR_DMA_CSR0, ext_addr);
    csr_wr(d, CSR_DMA_CSR1,
           (sram_off & 0xFFFFu) | ((sram_stride & 0xFFFFu) << 16));
    csr_wr(d, CSR_DMA_CSR2,
           (row_bytes & 0xFFFFu) | ((rows & 0xFFFFu) << 16));
    csr_wr(d, CSR_DMA_CSR3,
           DMA_CSR3_START | DMA_CSR3_MODE_2D |
           (is_st ? DMA_CSR3_DIR_ST : 0u) |
           DMA_CSR3_EXT_STRIDE(ext_stride));

    if (_wait_started_or_done(d, DMA_START_TIMEOUT_LOOPS) != 0) return -1;
    return _wait_not_busy(d, DMA_TIMEOUT_LOOPS);
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
    volatile uint32_t *isram;

    d->mmio_base = mmio_base;
    isram = (volatile uint32_t *)NPU_ISRAM_BASE;

    // Hold the datapath in reset while installing a default WFI program.
    // Normal CSR-driven runtime paths do not use IF/ID, so the local
    // instruction stream must be quiescent after reset.
    csr_wr(d, CSR_CTRL, CSR_CTRL_RESET);
    isram[0] = 0xF1000000u;  // WFI
    isram[1] = 0xFF000000u;  // NOP padding
    isram[2] = 0xFF000000u;
    isram[3] = 0xFF000000u;
    _mem_fence();
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
    if (n > DESC_BOUNCE_SIZE) return -1;

    // The DMA datapath moves 64-bit beats.  Descriptor callers commonly
    // pass packed structs from the stack, so bounce through an aligned,
    // zero-padded buffer to preserve byte layout.
    uint32_t size = (uint32_t)n;
    uint32_t aligned = (size + 7u) & ~7u;

    // Check DSRAM space
    if (_dsram_next + aligned > DSRAM_BASE + DSRAM_SIZE) return -1;

    uint32_t offset = _dsram_next;

    for (uint32_t i = 0; i < aligned; i++) {
        _desc_bounce[i] = 0;
    }
    for (uint32_t i = 0; i < size; i++) {
        _desc_bounce[i] = ((const uint8_t *)desc)[i];
    }

    // DMA the descriptor from host memory into DSRAM
    int ret = _dma_xfer(d, (uint32_t)(uintptr_t)_desc_bounce, offset, aligned, 0);
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

int npu_dma_2d_ld(npu_dev_t *d, uint32_t ext_addr, uint32_t sram_off,
                  uint32_t rows, uint32_t row_bytes,
                  uint32_t ext_stride, uint32_t sram_stride)
{
    return _dma_2d_xfer(d, ext_addr, sram_off, rows, row_bytes,
                        ext_stride, sram_stride, 0);
}

int npu_dma_2d_st(npu_dev_t *d, uint32_t ext_addr, uint32_t sram_off,
                  uint32_t rows, uint32_t row_bytes,
                  uint32_t ext_stride, uint32_t sram_stride)
{
    return _dma_2d_xfer(d, ext_addr, sram_off, rows, row_bytes,
                        ext_stride, sram_stride, 1);
}

int npu_run_program(npu_dev_t *d, const void *image, size_t n,
                    uint32_t timeout_us)
{
    const uint8_t *bytes = (const uint8_t *)image;
    volatile uint32_t *isram = (volatile uint32_t *)NPU_ISRAM_BASE;
    uint32_t num_instr;
    uint32_t num_desc;
    uint32_t instr_bytes;
    uint32_t desc_off;
    uint32_t desc_bytes;
    uint32_t desc_aligned;
    uint32_t desc_base = DSRAM_BASE;

    if (image == 0 || n < 16u) return -1;
    if (_load_le32(bytes + 0) != NPU_PROGRAM_MAGIC) return -2;
    if (_load_le32(bytes + 4) != NPU_PROGRAM_VERSION) return -3;

    num_instr = _load_le32(bytes + 8);
    num_desc  = _load_le32(bytes + 12);
    if (num_instr == 0 || num_instr > NPU_MAX_IFID_INSTR) return -4;

    instr_bytes = num_instr * 4u;
    desc_off = 16u + instr_bytes;
    desc_bytes = num_desc * NPU_DESC_SLOT_SIZE;
    if (desc_off > n || desc_bytes > n - desc_off) return -5;

    desc_aligned = (desc_bytes + 7u) & ~7u;
    if (desc_aligned > PROGRAM_DESC_BOUNCE_SIZE) return -6;
    if (_dsram_next + desc_aligned > DSRAM_BASE + DSRAM_SIZE) return -7;

    if (desc_aligned != 0u) {
        desc_base = _dsram_next;
        for (uint32_t i = 0; i < desc_aligned; i++)
            _program_desc_bounce[i] = 0;
        for (uint32_t i = 0; i < desc_bytes; i++)
            _program_desc_bounce[i] = bytes[desc_off + i];

        if (_dma_xfer(d, (uint32_t)(uintptr_t)_program_desc_bounce,
                      desc_base, desc_aligned, 0) != 0)
            return -8;

        _dsram_next += desc_aligned;
    }

    if (_wait_not_busy(d, DMA_TIMEOUT_LOOPS) != 0) return -9;

    csr_wr(d, CSR_CTRL, CSR_CTRL_RESET);
    _mem_fence();

    for (uint32_t i = 0; i < num_instr; i++)
        isram[i] = _load_le32(bytes + 16u + i * 4u);
    for (uint32_t i = num_instr; i < NPU_MAX_IFID_INSTR; i++)
        isram[i] = 0xFF000000u;
    _mem_fence();

    csr_wr(d, CSR_IRQ_STAT, IRQ_DONE);
    csr_wr(d, CSR_DESC_PTR, desc_base);
    csr_wr(d, CSR_CTRL, 0u);

    return npu_wait_done(d, timeout_us);
}

int npu_issue(npu_dev_t *d, uint8_t opcode, uint32_t desc_ptr)
{
    // Refuse if the NPU is already executing
    if (csr_rd(d, CSR_STATUS) & CSR_STATUS_BUSY) return -1;

    // Clear stale done before issuing so npu_wait_done can use IRQ_PEND
    // as a completion edge if the operation finishes before BUSY is sampled.
    csr_wr(d, CSR_IRQ_STAT, IRQ_DONE);
    csr_wr(d, CSR_DESC_PTR, desc_ptr);
    csr_wr(d, CSR_CTRL, CSR_CTRL_START | CSR_CTRL_OPCODE(opcode));
    return 0;
}

int npu_wait_done(npu_dev_t *d, uint32_t timeout_us)
{
    uint32_t elapsed = 0u;
    const uint32_t step_us = 100u;  // poll every ~100 µs

    if (_wait_started_or_done(d, DMA_START_TIMEOUT_LOOPS) != 0) return -1;

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
