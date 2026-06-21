// E2E program-stream test:
//   CPU loads A/B tiles, runs a serialized .npu GEMM program through
//   npu_run_program(), stores O-SRAM back to memory, and checks results.

#include <stdint.h>
#include "../../sw/driver/npu_driver.h"
#include "../../sw/driver/npu_csr.h"

#define TILE 16

static int8_t A[TILE * TILE] __attribute__((aligned(8)));
static int8_t B[TILE * TILE] __attribute__((aligned(8)));
static int16_t C[TILE * TILE] __attribute__((aligned(8)));

static const uint8_t program_image[] __attribute__((aligned(8))) = {
    // Header: magic "NPUC", version=1, num_instr=2, num_desc=1
    0x4e, 0x50, 0x55, 0x43,
    0x01, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,

    // Instruction: OP_GEMM, desc_ref_words=0, aux=0
    0x00, 0x00, 0x00, 0x01,
    // Instruction: OP_WFI, halt IF/ID after issuing GEMM
    0x00, 0x00, 0x00, 0xf1,

    // GEMM descriptor slot, 20 bytes:
    // M=1, N=1, K=1, A bank=0, B bank=1, O bank=2,
    // zp=0, scale mul=1, scale shr=0, relu=0, out_zp=0.
    0x01, 0x00, 0x01, 0x00,
    0x01, 0x00, 0x00, 0x01,
    0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x00,
};

static void uart_putc(char c)
{
    *(volatile char *)0x00000008 = c;
}

void main(void)
{
    npu_dev_t npu;

    for (int i = 0; i < TILE * TILE; i++) {
        A[i] = 1;
        B[i] = 1;
        C[i] = 0;
    }

    npu_init(&npu, NPU_CSR_BASE);
    uart_putc('i');

    for (int r = 0; r < TILE; r++) {
        if (npu_dma_ld(&npu, (uint32_t)(uintptr_t)&A[r * TILE],
                       ASRAM_BASE + (uint32_t)(r * TILE), TILE) != 0) {
            uart_putc('A');
            while (1) __asm__ volatile ("wfi");
        }
    }
    uart_putc('a');

    for (int r = 0; r < TILE; r++) {
        if (npu_dma_ld(&npu, (uint32_t)(uintptr_t)&B[r * TILE],
                       WSRAM_BASE + (uint32_t)(r * TILE), TILE) != 0) {
            uart_putc('B');
            while (1) __asm__ volatile ("wfi");
        }
    }
    uart_putc('b');

    uart_putc('r');
    if (npu_run_program(&npu, program_image, sizeof(program_image),
                        1000000u) != 0) {
        uart_putc('G');
        while (1) __asm__ volatile ("wfi");
    }
    uart_putc('g');

    for (int r = 0; r < TILE; r++) {
        if (npu_dma_st(&npu, (uint32_t)(uintptr_t)&C[r * TILE],
                       OSRAM_BASE + (uint32_t)(r * TILE * 2),
                       TILE * 2) != 0) {
            uart_putc('S');
            while (1) __asm__ volatile ("wfi");
        }
    }
    uart_putc('s');

    for (int i = 0; i < TILE * TILE; i++) {
        if (C[i] != TILE) {
            uart_putc('M');
            while (1) __asm__ volatile ("wfi");
        }
    }

    uart_putc('P');
    while (1) __asm__ volatile ("wfi");
}

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
