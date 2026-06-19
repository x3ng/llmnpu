// ============================================================
// firmware_csr.c — E2E Stage 2: NPU CSR Read/Write Test
//
// Bare-metal RISC-V firmware. Tests:
//   1. Write 1 to CTRL       (word 0, byte offset 0x00), read back
//   2. Write to STATUS       (word 1, byte offset 0x04), verify
//        write ignored — STATUS is read-only
//   3. Read PC               (word 2, byte offset 0x08), verify
//        initial value is 0
//   4. Write to DESC_PTR     (word 4, byte offset 0x10), read back
//
// Output over UART (0x00000008): 'P' on all pass, 'F' on any fail.
// ============================================================

// ------------------------------------------------------------
// UART TX helper
// ------------------------------------------------------------
static void uart_putc(char c)
{
    *(volatile unsigned int *)0x00000008u = (unsigned int)c;
}

// ------------------------------------------------------------
// Forward declaration
// ------------------------------------------------------------
void main(void);

extern int __bss_start, __bss_end;

// ------------------------------------------------------------
// Bare-metal entry point
// ------------------------------------------------------------
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
// Main — CSR register tests
// ------------------------------------------------------------
void main(void)
{
    volatile unsigned int *csr = (volatile unsigned int *)0x10000000u;
    int pass = 1;

    // Test 1: Write 1 to CTRL (word 0, byte offset 0x00), read back
    csr[0] = 1;
    if (csr[0] != 1) pass = 0;

    // Test 2: Write to STATUS (word 1, byte offset 0x04), verify
    // write ignored — STATUS is read-only, write hits default case
    csr[1] = 0xDEADBEEFu;
    if (csr[1] != 0) pass = 0;

    // Test 3: Read PC (word 2, byte offset 0x08), verify initial
    // value is 0 (PC resets to 32'd0 in csr.sv)
    if (csr[2] != 0) pass = 0;

    // Test 4: Write to DESC_PTR (word 4, byte offset 0x10), read back
    csr[4] = 0xCAFEBABEu;
    if (csr[4] != 0xCAFEBABEu) pass = 0;

    uart_putc(pass ? 'P' : 'F');
}
