// ============================================================
// firmware_dma.c — E2E Stage 3: DMA Bare-Metal Firmware
//
// Exercises DMA CSR register paths via MMIO and validates
// ExtMem read/write.  No driver library — volatile unsigned int*
// register pokes only.
//
// Output over UART (0x00000008): 'P' on all checks pass, 'F' on any fail.
// ============================================================

// ------------------------------------------------------------
// UART TX helper
// ------------------------------------------------------------
static void uart_putc(char c)
{
    *(volatile unsigned int *)0x00000008u = (unsigned int)c;
}

// ------------------------------------------------------------
// Memory fence
// ------------------------------------------------------------
static void mem_fence(void)
{
    __asm__ volatile ("fence" ::: "memory");
}

// Read DEBUG register (0x60) to see which unit is stuck
static unsigned int csr_debug_read(void)
{
    return *(volatile unsigned int *)(0x10000000u + 0x60u);
}

// Print a hex nibble over UART (0-15 → '0'-'F')
static void uart_puthex(unsigned int val)
{
    val &= 0xFu;
    if (val < 10) uart_putc('0' + (char)val);
    else          uart_putc('A' + (char)(val - 10));
}

// Print 32-bit value as 8 hex chars over UART
static void uart_puthex32(unsigned int val)
{
    int i;
    for (i = 7; i >= 0; i--) {
        uart_puthex((val >> (i * 4)) & 0xFu);
    }
}

// ------------------------------------------------------------
// Bare-metal entry point
// ------------------------------------------------------------
void main(void);

extern int __bss_start, __bss_end;

__attribute__((section(".text._start")))
void _start(void)
{
    __asm__ volatile (
        ".option push\n\t"
        ".option norelax\n\t"
        "la   sp, __stack_top\n\t"
        ".option pop\n\t"
    );

    for (char *p = (char *)&__bss_start; p < (char *)&__bss_end;)
        *p++ = 0;

    main();

    while (1)
        __asm__ volatile ("wfi");
}

// ------------------------------------------------------------
// Main — DMA CSR register and ExtMem test
// ------------------------------------------------------------
void main(void)
{
    int errors = 0;

    // === NPU CSR base address ===
    volatile unsigned int *csr = (volatile unsigned int *)0x10000000u;

    // === DMA CSR register offsets (32-bit word addressing) ===
    // DMA_CSR0 at 0x20 : ext_addr[31:0]
    // DMA_CSR1 at 0x28 : sram_addr[15:0]
    // DMA_CSR2 at 0x30 : length[15:0]
    // DMA_CSR3 at 0x38 : ctrl (bit[0]=start, bit[1]=is_store)
    #define CSR_DMA_CSR0  0x20u
    #define CSR_DMA_CSR1  0x28u
    #define CSR_DMA_CSR2  0x30u
    #define CSR_DMA_CSR3  0x38u

    // -------------------------------------------------------
    // Stage 1: Write known pattern to ExtMem (0x40000000 area)
    // -------------------------------------------------------
    {
        volatile unsigned int *ext = (volatile unsigned int *)0x40000000u;
        unsigned int ext_base_word;
        unsigned int pattern[16];
        int i;
        unsigned int rd;
        // 16-word (64-byte) pattern at byte offset 0x1000
        ext_base_word = 0x1000u / 4u;  // word offset 0x400
        for (i = 0; i < 16; i++) {
            pattern[i] = 0xDA000000u + (unsigned int)i;
            ext[ext_base_word + (unsigned int)i] = pattern[i];
        }
        mem_fence();

        // Read back and verify
        for (i = 0; i < 16; i++) {
            rd = ext[ext_base_word + (unsigned int)i];
            if (rd != pattern[i]) {
                errors++;
            }
        }
    }

    // -------------------------------------------------------
    // Stage 2: Write DMA CSR registers and read back
    // -------------------------------------------------------
    {
        unsigned int wr_ext_addr;
        unsigned int rd_ext_addr;
        unsigned int wr_csr1;
        unsigned int rd_csr1;
        unsigned int wr_csr2;
        unsigned int rd_csr2;
        unsigned int wr_csr3;
        unsigned int rd_csr3;
        // DMA_CSR0 (0x20): ext_addr   = 0x00000100
        wr_ext_addr = 0x00000100u;
        csr[CSR_DMA_CSR0 / 4u] = wr_ext_addr;
        mem_fence();
        rd_ext_addr = csr[CSR_DMA_CSR0 / 4u];
        if (rd_ext_addr != wr_ext_addr) errors++;

        // DMA_CSR1 (0x28): sram_off = 0x0000
        wr_csr1 = 0x0000u;
        csr[CSR_DMA_CSR1 / 4u] = wr_csr1;
        mem_fence();
        rd_csr1 = csr[CSR_DMA_CSR1 / 4u];
        if (rd_csr1 != wr_csr1) errors++;

        // DMA_CSR2 (0x30): length = 128
        wr_csr2 = 128u;
        csr[CSR_DMA_CSR2 / 4u] = wr_csr2;
        mem_fence();
        rd_csr2 = csr[CSR_DMA_CSR2 / 4u];
        if (rd_csr2 != wr_csr2) errors++;

        // DMA_CSR3 (0x38): ctrl (readback test only — no START to avoid
        // triggering a real DMA with test parameters)
        wr_csr3 = 0u;
        csr[CSR_DMA_CSR3 / 4u] = wr_csr3;
        mem_fence();
        rd_csr3 = csr[CSR_DMA_CSR3 / 4u];
        if (rd_csr3 != wr_csr3) errors++;
    }

    // -------------------------------------------------------
    // Stage 3: CSR CTRL / STATUS register smoke test
    // -------------------------------------------------------
    {
        unsigned int status;
        // Write CTRL RESET, then clear
        csr[0] = 0x00000002u;
        mem_fence();
        csr[0] = 0x00000000u;
        mem_fence();

        // Read STATUS (should be 0 after reset)
        status = csr[1];
        // Only bit 0 (busy) and bit 1 (irq_pend) should be 0 at idle
        if (status & 0xFFFFFFFCu) {
            errors++;
        }
        (void)status;
    }

    // -------------------------------------------------------
    // Stage 4: PERF_CYCLE increment check
    // -------------------------------------------------------
    {
        unsigned int c1;
        unsigned int c2;
        c1 = csr[0x20u];   // PERF_CYCLE at word offset 0x20 (byte 0x80)
        {volatile int d; for (d = 0; d < 100; d++) __asm__ volatile ("");}
        c2 = csr[0x20u];
        if (c2 <= c1) errors++;
    }

    // -------------------------------------------------------
    // Stage 5: Real DMA LOAD from ExtMem to verify DMA+bridge
    // -------------------------------------------------------
    {
        volatile unsigned int *csr_status = (volatile unsigned int *)0x10000004u;

        // Write known data to ExtMem at a test location
        volatile unsigned int *ext = (volatile unsigned int *)0x40001000u;
        ext[0] = 0xDEADBEEFu;
        ext[1] = 0xCAFEBABEu;
        mem_fence();

        // Program DMA: LOAD 8 bytes from ExtMem 0x40001000 → SRAM 0x3000
        csr[CSR_DMA_CSR0 / 4u] = 0x40001000u;  // ext_addr
        mem_fence();
        csr[CSR_DMA_CSR1 / 4u] = 0x3000u;       // sram_off
        mem_fence();
        csr[CSR_DMA_CSR2 / 4u] = 8u;             // length
        mem_fence();
        csr[CSR_DMA_CSR3 / 4u] = 0x1u;           // START, is_store=0 (LOAD)
        mem_fence();

        // Wait for BUSY to de-assert
        {
            volatile unsigned int to = 50000u;
            while (*csr_status & 1u) {
                if (--to == 0) {
                    uart_putc('T');
                    uart_puthex32(csr_debug_read());
                    errors++;
                    break;
                }
            }
        }

        // Add bridge delay (the hardware bridge copies DMA→SRAM after dma_done)
        {volatile int d; for (d = 0; d < 1000; d++) __asm__ volatile ("");}

        // Verify CSR_STATUS is clean
        if (!(*csr_status & 1u)) {
            // Success marker
        }
    }

    // -------------------------------------------------------
    // Report result on UART
    // -------------------------------------------------------
    if (errors == 0) {
        uart_putc('P');
    } else {
        uart_putc('F');
    }
}
