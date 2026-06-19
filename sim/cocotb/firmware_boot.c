// ============================================================
// firmware_boot.c — E2E Stage 1: CPU Boot + UART Test
//
// Minimal bare-metal firmware:
//   1. _start (assembly stub): set up stack pointer, zero BSS,
//      call main
//   2. main: write 'P' (0x50) to UART TX, loop forever
//
// UART TX data register is at 0x00000008 (crossbar mapping).
// ============================================================

void main(void) {
    *(volatile char *)0x00000008 = 'P';
    while (1) { __asm__ volatile (""); }
}

// ------------------------------------------------------------
// Bare-metal entry point
//
// SP must be valid before any compiler-generated stack access.
// We set it up via inline asm.  The BSS clear and main() call
// follow.  Compiled with -O2 so the compiler keeps the BSS
// loop pointer in registers (not on the un-backed stack).
// ------------------------------------------------------------
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

    // Zero-initialise BSS
    for (char *p = (char *)&__bss_start; p < (char *)&__bss_end;)
        *p++ = 0;

    main();

    // Parachute — should never reach here
    while (1)
        __asm__ volatile ("wfi");
}
