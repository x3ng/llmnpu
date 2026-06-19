# ============================================================
# hello_npu.S — Minimal bare-metal NPU bringup (RV32IM)
#
# Steps:
#   1. Write 1 to CTRL (0x1000_0000) → npu_start pulse
#   2. Poll STATUS.BUSY (0x1000_0004) until NPU goes idle
#   3. Write 'P' (0x50) to UART TX (0x0000_0000)
#   4. Loop forever
#
# Build:
#   riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib \
#       -Wl,-Ttext=0x00000000 -o hello_npu.elf hello_npu.S
# ============================================================

.section .text
.globl _start

_start:
    # ----------------------------------------------------------
    # Step 0: Initialize stack pointer
    #   Use top of I-SRAM (0x1001_0000 + 8 KB = 0x1001_2000)
    # ----------------------------------------------------------
    lui  sp, %hi(0x10012000)
    addi sp, sp, %lo(0x10012000)

    # ----------------------------------------------------------
    # Step 1: Write CTRL.START=1
    #   CTRL register @ word offset 0x00 → byte addr 0x1000_0000
    #   CTRL bit0 = START (rising-edge triggered by CSR)
    # ----------------------------------------------------------
    lui  t0, %hi(0x10000000)    # t0 = 0x10000000 (NPU CSR base)
    addi t1, x0, 1              # t1 = 1
    sw   t1, %lo(0x10000000)(t0)  # *(CTRL) = 1

    # ----------------------------------------------------------
    # Step 2: Poll STATUS.BUSY
    #   STATUS register @ word offset 0x01 → byte addr 0x1000_0004
    #   STATUS bit0 = BUSY: 1 while NPU engines active
    # ----------------------------------------------------------
poll:
    lw   t2, %lo(0x10000004)(t0)  # t2 = *(STATUS)
    andi t2, t2, 1                # mask BUSY bit
    bnez t2, poll                 # spin while BUSY == 1

    # ----------------------------------------------------------
    # Step 3: NPU idle — write 'P' to UART
    #   UART TX data register @ byte addr 0x0000_0000
    #   0x50 = ASCII 'P'
    # ----------------------------------------------------------
    addi t0, x0, 0            # t0 = 0x00000000 (peripheral base)
    addi t1, x0, 0x50         # t1 = 'P'
    sw   t1, 0(t0)            # *(UART_TX) = 'P'

    # ----------------------------------------------------------
    # Step 4: Loop forever
    # ----------------------------------------------------------
done:
    j    done
