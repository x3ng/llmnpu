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
    // DMA_CSR1 at 0x24 : {sram_off[15:0], length[15:0]}   (packed per RTL)
    // DMA_CSR2 at 0x28 : len  (write-only, not wired to DMA engine)
    // DMA_CSR3 at 0x2C : ctrl (write-only, not wired to DMA engine)
    #define CSR_DMA_CSR0  0x20u
    #define CSR_DMA_CSR1  0x24u
    #define CSR_DMA_CSR2  0x28u
    #define CSR_DMA_CSR3  0x2Cu

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

        // DMA_CSR1 (0x24): {sram_off=0x0000, len=128}  (packed)
        wr_csr1 = (0x0000u << 16) | 128u;
        csr[CSR_DMA_CSR1 / 4u] = wr_csr1;
        mem_fence();
        rd_csr1 = csr[CSR_DMA_CSR1 / 4u];
        if (rd_csr1 != wr_csr1) errors++;

        // DMA_CSR2 (0x28): len = 128  (not wired to DMA engine)
        wr_csr2 = 128u;
        csr[CSR_DMA_CSR2 / 4u] = wr_csr2;
        mem_fence();
        rd_csr2 = csr[CSR_DMA_CSR2 / 4u];
        if (rd_csr2 != wr_csr2) errors++;

        // DMA_CSR3 (0x2C): ctrl = 1 (START)  (not wired to DMA engine)
        wr_csr3 = 1u;
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
    // Report result on UART
    // -------------------------------------------------------
    if (errors == 0) {
        uart_putc('P');
    } else {
        uart_putc('F');
    }
}
