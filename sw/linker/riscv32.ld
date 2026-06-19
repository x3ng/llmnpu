/* ============================================================
 * riscv32.ld — Bare-metal RISC-V RV32IM linker script
 *
 * Memory map:
 *   UART TX:     0x00000000
 *   NPU CSR:     0x10000000
 *   ExtMem DRAM: 0x40000000 (64 MB) — .text .rodata .data .bss
 *
 * Entry point: _start
 * Stack grows down from top of DRAM (0x44000000)
 * ============================================================ */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    DRAM (rwx) : ORIGIN = 0x40000000, LENGTH = 64M
}

SECTIONS
{
    /* ---- code -------------------------------------------------- */
    .text : {
        *(.text._start)          /* keep _start as first word    */
        *(.text*)
        *(.glue_7)               /* ARM-Thumb interwork leftovers */
        *(.glue_7t)
    } > DRAM

    /* ---- read-only data --------------------------------------- */
    .rodata : {
        . = ALIGN(4);
        *(.rodata*)
    } > DRAM

    /* ---- initialised data (loaded directly into RAM) ---------- */
    .data : {
        . = ALIGN(4);
        *(.data*)
        *(.sdata*)
    } > DRAM

    /* ---- zero-initialised data -------------------------------- */
    .bss : {
        . = ALIGN(4);
        __bss_start = .;
        *(.bss*)
        *(.sbss*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end = .;
    } > DRAM

    /* ---- discard unused sections ------------------------------ */
    /DISCARD/ : {
        *(.comment)
        *(.riscv.attributes)
    }

    /* ---- stack pointer (top of DRAM) -------------------------- */
    __stack_top = ORIGIN(DRAM) + LENGTH(DRAM);
}
