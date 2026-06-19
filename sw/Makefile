# ============================================================
# sw/Makefile — NPU Software Build System
#
# RISC-V RV32IM cross-compilation for bare-metal firmware.
# ============================================================

CROSS    ?= riscv32-none-elf-
CC       := $(CROSS)gcc
OBJCOPY  := $(CROSS)objcopy
OBJDUMP  := $(CROSS)objdump

ARCH     := rv32im
ABI      := ilp32

CFLAGS   := -march=$(ARCH) -mabi=$(ABI) -nostdlib -ffreestanding -O2
LDFLAGS  := -march=$(ARCH) -mabi=$(ABI) -nostdlib -Wl,-Ttext=0x00000000

BUILD_DIR := ../build
SRC_DIR   := demo

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------
.PHONY: all clean dump

all: $(BUILD_DIR)/hello_npu.elf

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/hello_npu.elf: $(SRC_DIR)/hello_npu.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# Disassemble for inspection
dump: $(BUILD_DIR)/hello_npu.elf
	$(OBJDUMP) -d $<

clean:
	rm -f $(BUILD_DIR)/hello_npu.elf
