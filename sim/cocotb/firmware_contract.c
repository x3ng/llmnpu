// ============================================================
// firmware_contract.c -- RISC-V software-visible NPU contract test
// ============================================================

#include <stdint.h>
#include "npu_driver.h"
#include "npu_abi.h"

#define UART_TX_ADDR 0x00000008u

#define RESULT_BASE 0x40080000u
#define DMA_SRC     0x40081000u
#define DMA_DST     0x40081200u
#define VALU_IN     0x40082000u
#define VALU_OUT    0x40082200u
#define SFU_IN      0x40083000u
#define SFU_OUT     0x40083200u

#define RESULT_MAGIC 0x4E505543u
#define RESULT_PASS  0x00000001u

extern int __bss_start, __bss_end;

void main(void);

static void uart_putc(char c)
{
    *(volatile uint32_t *)UART_TX_ADDR = (uint32_t)c;
}

static void mem_fence(void)
{
    __asm__ volatile ("fence" ::: "memory");
}

static void write_result(unsigned idx, uint32_t value)
{
    volatile uint32_t *result = (volatile uint32_t *)RESULT_BASE;
    result[idx] = value;
    mem_fence();
}

static void fill_bytes(uint32_t addr, uint32_t len, uint32_t seed)
{
    volatile uint8_t *p = (volatile uint8_t *)addr;
    for (uint32_t i = 0; i < len; i++)
        p[i] = (uint8_t)((i * seed + 0x8Du + (i >> 2)) & 0xFFu);
    mem_fence();
}

static int check_bytes(uint32_t addr, uint32_t len,
                       uint8_t (*golden)(uint8_t, uint32_t))
{
    volatile uint8_t *p = (volatile uint8_t *)addr;
    for (uint32_t i = 0; i < len; i++) {
        uint8_t exp = golden((uint8_t)i, i);
        if (p[i] != exp)
            return (int)i + 1;
    }
    return 0;
}

static uint8_t dma_golden(uint8_t ignored, uint32_t i)
{
    (void)ignored;
    return (uint8_t)((i * 7u + 0x31u) & 0xFFu);
}

static uint8_t valu_input(uint32_t i)
{
    return (uint8_t)((i * 5u + 0x11u) & 0xFFu);
}

static uint8_t valu_golden(uint8_t ignored, uint32_t i)
{
    (void)ignored;
    return (uint8_t)(valu_input(i) + 5u);
}

static uint8_t sfu_input(uint32_t i)
{
    return (uint8_t)((i * 9u + 0x80u + (i >> 1)) & 0xFFu);
}

static int8_t to_i8(uint8_t x)
{
    return (int8_t)x;
}

static uint8_t sfu_relu6_golden(uint8_t ignored, uint32_t i)
{
    int8_t x;
    (void)ignored;
    x = to_i8(sfu_input(i));
    if (x < 0)
        return 0u;
    if (x > 6)
        return 6u;
    return (uint8_t)x;
}

static int run_dma_roundtrip(npu_dev_t *npu)
{
    volatile uint8_t *src = (volatile uint8_t *)DMA_SRC;
    volatile uint8_t *dst = (volatile uint8_t *)DMA_DST;

    for (uint32_t i = 0; i < 256u; i++) {
        src[i] = dma_golden(0, i);
        dst[i] = 0u;
    }
    mem_fence();

    if (npu_dma_ld(npu, DMA_SRC, ASRAM_BASE, 256u) != 0)
        return -1;
    if (npu_dma_st(npu, DMA_DST, ASRAM_BASE, 256u) != 0)
        return -2;
    return check_bytes(DMA_DST, 256u, dma_golden);
}

static int run_valu_add_scalar(npu_dev_t *npu)
{
    volatile uint8_t *in = (volatile uint8_t *)VALU_IN;
    volatile uint8_t *out = (volatile uint8_t *)VALU_OUT;
    npu_valu_desc_t desc;
    int desc_ptr;

    for (uint32_t i = 0; i < 256u; i++) {
        in[i] = valu_input(i);
        out[i] = 0u;
    }
    mem_fence();

    if (npu_dma_ld(npu, VALU_IN, NPU_PP_VALU_IN_BASE, 256u) != 0)
        return -1;

    desc.word0 = NPU_VALU_DESC_WORD0(256u, NPU_VOPT_ADD, 1);
    desc.in0_addr = NPU_PP_VALU_IN_BASE;
    desc.in1_addr = 0u;
    desc.out_addr = NPU_PP_VALU_OUT_BASE;
    desc.scalar = 5u;

    desc_ptr = npu_load_descriptor(npu, &desc, sizeof(desc));
    if (desc_ptr < 0)
        return -2;
    if (npu_issue(npu, NPU_OP_VADD, (uint32_t)desc_ptr) != 0)
        return -3;
    if (npu_wait_done(npu, 1000000u) != 0)
        return -4;
    if (npu_dma_st(npu, VALU_OUT, NPU_PP_VALU_OUT_BASE, 256u) != 0)
        return -5;

    return check_bytes(VALU_OUT, 256u, valu_golden);
}

static int run_sfu_relu6(npu_dev_t *npu)
{
    volatile uint8_t *in = (volatile uint8_t *)SFU_IN;
    volatile uint8_t *out = (volatile uint8_t *)SFU_OUT;
    npu_sfu_desc_t desc;
    int desc_ptr;

    for (uint32_t i = 0; i < 256u; i++) {
        in[i] = sfu_input(i);
        out[i] = 0u;
    }
    mem_fence();

    if (npu_dma_ld(npu, SFU_IN, NPU_PP_SFU_IN_BASE, 256u) != 0)
        return -1;

    desc.word0 = NPU_SFU_DESC_WORD0(256u, NPU_OP_ACT_RELU6, 0u);
    desc.in_addr = NPU_PP_SFU_IN_BASE;
    desc.out_addr = NPU_PP_SFU_OUT_BASE;
    desc.scale = NPU_SFU_DESC_SCALE(1u, 0u);

    desc_ptr = npu_load_descriptor(npu, &desc, sizeof(desc));
    if (desc_ptr < 0)
        return -2;
    if (npu_issue(npu, NPU_OP_ACT_RELU6, (uint32_t)desc_ptr) != 0)
        return -3;
    if (npu_wait_done(npu, 1000000u) != 0)
        return -4;
    if (npu_dma_st(npu, SFU_OUT, NPU_PP_SFU_OUT_BASE, 256u) != 0)
        return -5;

    return check_bytes(SFU_OUT, 256u, sfu_relu6_golden);
}

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

void main(void)
{
    npu_dev_t npu;
    int ret;

    write_result(0, RESULT_MAGIC);
    write_result(1, 0u);
    write_result(2, 0u);
    write_result(3, 0u);
    write_result(4, 0u);

    npu_init(&npu, NPU_CSR_BASE);
    ret = run_dma_roundtrip(&npu);
    write_result(2, (uint32_t)ret);
    if (ret != 0) {
        uart_putc('d');
        uart_putc('F');
        return;
    }
    uart_putc('d');

    npu_init(&npu, NPU_CSR_BASE);
    ret = run_valu_add_scalar(&npu);
    write_result(3, (uint32_t)ret);
    if (ret != 0) {
        uart_putc('v');
        uart_putc('F');
        return;
    }
    uart_putc('v');

    npu_init(&npu, NPU_CSR_BASE);
    ret = run_sfu_relu6(&npu);
    write_result(4, (uint32_t)ret);
    if (ret != 0) {
        uart_putc('s');
        uart_putc('F');
        return;
    }
    uart_putc('s');

    write_result(1, RESULT_PASS);
    uart_putc('P');
}
