# ============================================================
# firmware_mini.S — Minimal SoC boot test
#
# After PicoRV32 boot trampoline jumps to 0x40000000,
# writes 'P' (0x50) to UART TX at 0x00000008 and loops.
# ============================================================

.section .text._start
.globl _start

_start:
    # t0 = 0
    addi t0, x0, 0

    # t1 = 0x50 = 'P'
    addi t1, x0, 0x50

    # *(0x00000008) = 'P'
    sw   t1, 8(t0)

    # Spin forever
hang:
    j hang
