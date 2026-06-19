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
LDFLAGS  := -march=$(ARCH) -mabi=$(ABI) -nostdlib

BUILD_DIR := ../build

# ------------------------------------------------------------
# Include paths  (relative to sw/)
# ------------------------------------------------------------
INC      := -Idriver -Iruntime

# ------------------------------------------------------------
# Object files for demo.elf
# ------------------------------------------------------------
DEMO_OBJS := $(BUILD_DIR)/main.o \
             $(BUILD_DIR)/npu_driver.o \
             $(BUILD_DIR)/npu_runtime.o

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------
.PHONY: all clean dump demo.elf

all: $(BUILD_DIR)/hello_npu.elf $(BUILD_DIR)/demo.elf

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# --- hello_npu (assembly bring-up) ------------------------------
$(BUILD_DIR)/hello_npu.elf: demo/hello_npu.S | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -Wl,-Ttext=0x00000000 -o $@ $<

# --- demo application (C) --------------------------------------
demo.elf: $(BUILD_DIR)/demo.elf

$(BUILD_DIR)/demo.elf: $(DEMO_OBJS) linker/riscv32.ld | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) -T linker/riscv32.ld -o $@ $(DEMO_OBJS)

# --- compile rules ---------------------------------------------
$(BUILD_DIR)/main.o: demo/main.c runtime/npu_runtime.h \
                     driver/npu_driver.h driver/npu_csr.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILD_DIR)/npu_driver.o: driver/npu_driver.c driver/npu_driver.h \
                            driver/npu_csr.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

$(BUILD_DIR)/npu_runtime.o: runtime/npu_runtime.c runtime/npu_runtime.h \
                             driver/npu_driver.h driver/npu_csr.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

# --- disassemble for inspection ---------------------------------
dump: $(BUILD_DIR)/demo.elf
	$(OBJDUMP) -d $<

# --- clean ------------------------------------------------------
clean:
	rm -f $(BUILD_DIR)/main.o $(BUILD_DIR)/npu_driver.o \
	      $(BUILD_DIR)/npu_runtime.o $(BUILD_DIR)/hello_npu.elf \
	      $(BUILD_DIR)/demo.elf
