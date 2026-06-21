// E2E program/runtime test:
//   CPU loads A/B tiles, runs a serialized .npu GEMM program through
//   npu_run_program(), issues an SFU ReLU postprocess through the driver,
//   stores O-SRAM back to memory, and checks results.

#include <stdint.h>
#include "../../sw/driver/npu_driver.h"
#include "../../sw/driver/npu_csr.h"
#include "../../sw/driver/npu_abi.h"

#define TILE 16

static int8_t A[TILE * TILE] __attribute__((aligned(8)));
static int8_t B[TILE * TILE] __attribute__((aligned(8)));
static int16_t C[TILE * TILE] __attribute__((aligned(8)));

static const uint8_t program_image[] __attribute__((aligned(8))) = {
    // Header: magic "NPUC", version=1, num_instr=6, num_desc=1
    0x4e, 0x50, 0x55, 0x43,
    0x01, 0x00, 0x00, 0x00,
    0x06, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,

    // Four NOPs force IF/ID execution past the legacy 4-word imem limit.
    0x00, 0x00, 0x00, 0xff,
    0x00, 0x00, 0x00, 0xff,
    0x00, 0x00, 0x00, 0xff,
    0x00, 0x00, 0x00, 0xff,
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
        A[i] = -1;
        B[i] = 1;
        C[i] = 0;
    }

    npu_init(&npu, NPU_CSR_BASE);
    uart_putc('i');

    if (npu_dma_ld(&npu, (uint32_t)(uintptr_t)A,
                   ASRAM_BASE, TILE * TILE) != 0) {
        uart_putc('A');
        while (1) __asm__ volatile ("wfi");
    }
    uart_putc('a');

    if (npu_dma_ld(&npu, (uint32_t)(uintptr_t)B,
                   WSRAM_BASE, TILE * TILE) != 0) {
        uart_putc('B');
        while (1) __asm__ volatile ("wfi");
    }
    uart_putc('b');

    uart_putc('r');
    if (npu_run_program(&npu, program_image, sizeof(program_image),
                        1000000u) != 0) {
        uart_putc('G');
        while (1) __asm__ volatile ("wfi");
    }
    uart_putc('g');

    if (npu_issue(&npu, NPU_OP_ACT_RELU, 0u) != 0 ||
        npu_wait_done(&npu, 1000000u) != 0) {
        uart_putc('R');
        while (1) __asm__ volatile ("wfi");
    }
    uart_putc('p');

    if (npu_dma_st(&npu, (uint32_t)(uintptr_t)C,
                   OSRAM_BASE, TILE * TILE * sizeof(C[0])) != 0) {
        uart_putc('S');
        while (1) __asm__ volatile ("wfi");
    }
    uart_putc('s');

    for (int i = 0; i < TILE * TILE; i++) {
        if (C[i] != 0) {
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
