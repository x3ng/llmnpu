# sim/Makefile — Cocotb simulation targets for NPU
# Each test target uses its own SIM_BUILD directory to avoid conflicts.

SHELL := bash
.SHELLFLAGS := -o pipefail -c

TOPLEVEL_LANG = verilog
SIM ?= icarus

REPO_ROOT = $(abspath $(CURDIR)/..)
SIM_DIR = $(CURDIR)

COMPILE_ARGS = -I$(REPO_ROOT)/rtl/include
SIM_ARGS =
EXTRA_ARGS =

VERILATOR_BUILD_JOBS ?= 4
VERILATOR_BUILD_MODE ?= debug
GEMM_WARN_ARGS = -Wno-fatal -Wno-CASEINCOMPLETE -Wno-PINMISSING -Wno-TIMESCALEMOD -Wno-CASEOVERLAP -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC
GEMM_DEBUG_EXTRA_ARGS = --output-split 5000 --output-split-cfuncs 5000 -CFLAGS "-O0"
GEMM_MODE_EXTRA_ARGS = $(if $(filter release,$(VERILATOR_BUILD_MODE)),,$(GEMM_DEBUG_EXTRA_ARGS))
GEMM_MODE_BUILD_ARGS = $(if $(filter release,$(VERILATOR_BUILD_MODE)),,OPT_FAST=-O0 OPT_SLOW=-O0 OPT_GLOBAL=-O0)
GEMM_EXTRA_ARGS = $(GEMM_WARN_ARGS) $(GEMM_MODE_EXTRA_ARGS)
GEMM_BUILD_ARGS = $(if $(BUILD_ARGS),$(BUILD_ARGS),-j$(VERILATOR_BUILD_JOBS) $(GEMM_MODE_BUILD_ARGS))
E2E_COMPILE_ARGS = -I$(REPO_ROOT)/rtl/include -DHEX_FILE=\"$(SIM_BUILD)/sim/verilog/firmware.hex\"
E2E_VERILATOR_INCLUDE_DIRS = $(REPO_ROOT)/rtl/include
E2E_VERILATOR_EXTRA_ARGS = -f $(SIM_BUILD)/hex_def.f
LOG_FILTER ?= PASS|FAIL|ERROR|UART|timeout|Timeout|wfi|WFI|trap|TRAP|mismatch|MISMATCH|FATAL|DIAG|HANG

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: test_pe test_gemm test_valu test_sfu test_dma test_dma_load test_memory test_if_id test_pingpong test_npu_top
.PHONY: test_e2e test_e2e_csr test_e2e_boot test_e2e_gemm test_e2e_dma test_e2e_program
.PHONY: rebuild_e2e_gemm e2e_gemm_log e2e_dma_log e2e_boot_log e2e_csr_log e2e_program_log e2e_log test_quick test_all clean

# ── Shared variables for all E2E (SoC-level) targets ──

E2E_RTL = \
	$(REPO_ROOT)/rtl/soc/top_soc.sv \
	$(REPO_ROOT)/rtl/soc/ext_mem_model.sv \
	$(REPO_ROOT)/rtl/cpu/picorv32_adapted.v \
	$(REPO_ROOT)/rtl/cpu/picorv32_wrapper.sv \
	$(REPO_ROOT)/rtl/interconnect/axi_crossbar_soc.sv \
	$(REPO_ROOT)/ref/verilog-axi/rtl/axi_dma_rd.v \
		$(REPO_ROOT)/rtl/interconnect/axi_dma_wrapper.sv \
	$(REPO_ROOT)/rtl/npu/top.sv \
	$(REPO_ROOT)/rtl/npu/csr.sv \
	$(REPO_ROOT)/rtl/npu/pingpong.sv \
	$(REPO_ROOT)/rtl/npu/if_id/ifetch.sv \
	$(REPO_ROOT)/rtl/npu/if_id/idecode.sv \
	$(REPO_ROOT)/rtl/npu/if_id/dispatch.sv \
	$(REPO_ROOT)/rtl/npu/if_id/if_id_top.sv \
	$(REPO_ROOT)/rtl/npu/gemm/pe.sv \
	$(REPO_ROOT)/rtl/npu/gemm/systolic_array.sv \
	$(REPO_ROOT)/rtl/npu/gemm/gemm_controller.sv \
	$(REPO_ROOT)/rtl/npu/valu/valu_lane.sv \
	$(REPO_ROOT)/rtl/npu/valu/vregfile.sv \
	$(REPO_ROOT)/rtl/npu/valu/valu_top.sv \
	$(REPO_ROOT)/rtl/npu/sfu/lut_engine.sv \
	$(REPO_ROOT)/rtl/npu/sfu/quant_dequant.sv \
	$(REPO_ROOT)/rtl/npu/sfu/sfu_top.sv \
	$(REPO_ROOT)/rtl/npu/dma/npu_dma.sv \
	$(REPO_ROOT)/rtl/npu/memory/sram_bank.sv \
	$(REPO_ROOT)/rtl/npu/memory/crossbar.sv

# ── Individual unit test targets (each isolated, can run in parallel) ──

test_pe: SIM_BUILD = sim_build_pe
test_pe:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_pe TOPLEVEL=pe \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/gemm/pe.sv" \
	$(MAKE)

test_gemm: SIM_BUILD = sim_build_gemm
test_gemm:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_gemm TOPLEVEL=systolic_array \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/gemm/pe.sv $(REPO_ROOT)/rtl/npu/gemm/systolic_array.sv $(REPO_ROOT)/rtl/npu/gemm/gemm_controller.sv" \
	$(MAKE)

test_valu: SIM_BUILD = sim_build_valu
test_valu:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_valu TOPLEVEL=valu_top \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/valu/valu_lane.sv $(REPO_ROOT)/rtl/npu/valu/vregfile.sv $(REPO_ROOT)/rtl/npu/valu/valu_top.sv" \
	$(MAKE)

test_sfu: SIM_BUILD = sim_build_sfu
test_sfu:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_sfu TOPLEVEL=sfu_top \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/sfu/lut_engine.sv $(REPO_ROOT)/rtl/npu/sfu/quant_dequant.sv $(REPO_ROOT)/rtl/npu/sfu/sfu_top.sv" \
	$(MAKE)

test_dma: SIM_BUILD = sim_build_dma
test_dma:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_dma TOPLEVEL=npu_dma \
	VERILOG_SOURCES="$(REPO_ROOT)/ref/verilog-axi/rtl/axi_dma_rd.v $(REPO_ROOT)/rtl/interconnect/axi_dma_wrapper.sv $(REPO_ROOT)/rtl/npu/dma/npu_dma.sv" \
	$(MAKE)

test_dma_load: SIM_BUILD = sim_build_dma_load
test_dma_load:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_dma_load TOPLEVEL=npu_dma \
	VERILOG_SOURCES="$(REPO_ROOT)/ref/verilog-axi/rtl/axi_dma_rd.v $(REPO_ROOT)/rtl/interconnect/axi_dma_wrapper.sv $(REPO_ROOT)/rtl/npu/dma/npu_dma.sv" \
	$(MAKE)

test_memory: SIM_BUILD = sim_build_memory
test_memory:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_memory TOPLEVEL=crossbar \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/memory/sram_bank.sv $(REPO_ROOT)/rtl/npu/memory/crossbar.sv" \
	$(MAKE)

test_if_id: SIM_BUILD = sim_build_if_id
test_if_id:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_if_id TOPLEVEL=if_id_top \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/if_id/ifetch.sv $(REPO_ROOT)/rtl/npu/if_id/idecode.sv $(REPO_ROOT)/rtl/npu/if_id/dispatch.sv $(REPO_ROOT)/rtl/npu/if_id/if_id_top.sv" \
	$(MAKE)

test_pingpong: SIM_BUILD = sim_build_pingpong
test_pingpong:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_pingpong TOPLEVEL=pingpong \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/pingpong.sv" \
	$(MAKE)

test_npu_top: SIM_BUILD = sim_build_npu_top
test_npu_top:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_npu_top TOPLEVEL=npu_top \
	VERILOG_SOURCES="$(E2E_RTL)" \
	$(MAKE) COMPILE_ARGS='$(COMPILE_ARGS) -P npu_top.DMA_STANDALONE=1'

# ── E2E helper: build RISC-V firmware and convert to hex ──
# Usage: $(call build_fw,<target_name>,<c_sources>,<extra_gcc_flags>)
define build_fw
	@echo "=== Building $(1) firmware ==="
	@mkdir -p $(REPO_ROOT)/build
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 $(3) \
		-T $(REPO_ROOT)/sw/linker/riscv32.ld \
		$(2) -o $(REPO_ROOT)/build/$(1).elf
	riscv32-none-elf-objcopy -O binary $(REPO_ROOT)/build/$(1).elf $(REPO_ROOT)/build/$(1).bin
	@python3 -c "import sys; data=open('$(REPO_ROOT)/build/$(1).bin','rb').read(); data+=b'\x00'*(-len(data)%4); [print(f'{int.from_bytes(data[i:i+4],\"little\"):08x}') for i in range(0,len(data),4)]" > $(REPO_ROOT)/build/$(1).hex
	@mkdir -p $(SIM_BUILD)/sim/verilog
	cp $(REPO_ROOT)/build/$(1).hex $(SIM_BUILD)/sim/verilog/$(1).hex
	cp $(REPO_ROOT)/build/$(1).hex $(SIM_BUILD)/sim/verilog/firmware.hex
	@# Generate iverilog command-line file overriding HEX_FILE to per-build path
	@printf '%s\n' '+define+HEX_FILE=\"$(SIM_BUILD)/sim/verilog/firmware.hex\"' > $(SIM_BUILD)/hex_def.f
	@# Fallback: copy to shared location for standalone/test runs
	@mkdir -p $(REPO_ROOT)/sim/verilog
	cp $(REPO_ROOT)/build/$(1).hex $(REPO_ROOT)/sim/verilog/firmware.hex
endef

define show_log
	@if [ -f "$(1)" ]; then \
		rg -n "$(LOG_FILTER)" "$(1)" || true; \
	else \
		echo "No log found: $(1)"; \
		exit 1; \
	fi
endef

# ── E2E Stage 1: CPU Boot ──

test_e2e_boot: SIM_BUILD = sim_build_boot
test_e2e_boot:
	@rm -rf $(SIM_BUILD)
	$(call build_fw,fw_boot,$(SIM_DIR)/cocotb/firmware_boot.c)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_e2e_boot TOPLEVEL=top_soc \
	VERILOG_SOURCES="$(E2E_RTL)" \
	$(MAKE) COMPILE_ARGS='$(E2E_COMPILE_ARGS)' \
		2>&1 | tee $(SIM_BUILD)/test.log

# ── E2E Stage 2: CSR ──

test_e2e_csr: SIM_BUILD = sim_build_csr
test_e2e_csr:
	@rm -rf $(SIM_BUILD)
	$(call build_fw,firmware_csr,$(SIM_DIR)/cocotb/firmware_csr.c)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_e2e_csr TOPLEVEL=top_soc \
	VERILOG_SOURCES="$(E2E_RTL)" \
	$(MAKE) COMPILE_ARGS='$(E2E_COMPILE_ARGS)' \
		2>&1 | tee $(SIM_BUILD)/test.log

# ── E2E Stage 3: DMA ──

test_e2e_dma: SIM_BUILD = sim_build_e2e_dma
test_e2e_dma:
	@rm -rf $(SIM_BUILD)
	$(call build_fw,firmware_dma,$(SIM_DIR)/cocotb/firmware_dma.c,-std=gnu99)
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_e2e_dma TOPLEVEL=top_soc \
	VERILOG_SOURCES="$(E2E_RTL)" \
	$(MAKE) COMPILE_ARGS='$(E2E_COMPILE_ARGS)' \
		2>&1 | tee $(SIM_BUILD)/test.log

# ── E2E Stage 4: GEMM (Verilator) ──

test_e2e_gemm: SIM_BUILD = sim_build_e2e_gemm
test_e2e_gemm: SIM = verilator
test_e2e_gemm:
	@echo "=== Reusing $(SIM_BUILD) when cocotb/Verilator dependencies are current ==="
	@echo "=== Verilator C++ build jobs: $(VERILATOR_BUILD_JOBS) ==="
	@echo "=== Verilator build mode: $(VERILATOR_BUILD_MODE) ==="
	@# Guard: GEMM requires Verilator; refuse if vvp is running
	@# (temporarily disabled: false positive from parent bash cmdline containing 'vvp')
	@#if pgrep -x vvp > /dev/null; then \
	#	echo "ERROR: vvp process detected but GEMM target expects Verilator."; \
	#	echo "Stop any running Icarus simulations first (kill \`ps aux | grep vvp\`)."; \
	#	exit 1; \
	#fi
	@mkdir -p $(REPO_ROOT)/build
	@python3 $(REPO_ROOT)/tools/codegen/generate_golden.py --header > $(REPO_ROOT)/build/golden_gemm.h
	@echo "=== Building firmware_gemm firmware ==="
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 \
		-I$(REPO_ROOT)/build -I$(REPO_ROOT)/sw/driver -I$(REPO_ROOT)/sw/runtime \
		-c $(SIM_DIR)/cocotb/firmware_gemm.c -o $(REPO_ROOT)/build/firmware_gemm.o
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 \
		-I$(REPO_ROOT)/sw/driver -I$(REPO_ROOT)/sw/runtime \
		-c $(REPO_ROOT)/sw/driver/npu_driver.c -o $(REPO_ROOT)/build/npu_driver_gemm.o
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 \
		-I$(REPO_ROOT)/sw/driver -I$(REPO_ROOT)/sw/runtime \
		-c $(REPO_ROOT)/sw/runtime/npu_runtime.c -o $(REPO_ROOT)/build/npu_runtime_gemm.o
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib \
		-T $(REPO_ROOT)/sw/linker/riscv32.ld \
		-o $(REPO_ROOT)/build/firmware_gemm.elf \
		$(REPO_ROOT)/build/firmware_gemm.o \
		$(REPO_ROOT)/build/npu_driver_gemm.o \
		$(REPO_ROOT)/build/npu_runtime_gemm.o
	riscv32-none-elf-objcopy -O binary $(REPO_ROOT)/build/firmware_gemm.elf $(REPO_ROOT)/build/firmware_gemm.bin
	@python3 -c "import sys; data=open('$(REPO_ROOT)/build/firmware_gemm.bin','rb').read(); data+=b'\x00'*(-len(data)%4); [print(f'{int.from_bytes(data[i:i+4],\"little\"):08x}') for i in range(0,len(data),4)]" > $(REPO_ROOT)/build/firmware_gemm.hex
	@mkdir -p $(SIM_BUILD)/sim/verilog
	cp $(REPO_ROOT)/build/firmware_gemm.hex $(SIM_BUILD)/sim/verilog/firmware_gemm.hex
	cp $(REPO_ROOT)/build/firmware_gemm.hex $(SIM_BUILD)/sim/verilog/firmware.hex
	@printf '%s\n' '+define+HEX_FILE=\"$(SIM_BUILD)/sim/verilog/firmware_gemm.hex\"' > $(SIM_BUILD)/hex_def.f
	@mkdir -p $(REPO_ROOT)/sim/verilog
	cp $(REPO_ROOT)/build/firmware_gemm.hex $(REPO_ROOT)/sim/verilog/firmware.hex
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_e2e_gemm TOPLEVEL=top_soc \
	VERILOG_SOURCES="$(E2E_RTL)" \
	$(MAKE) SIM=verilator \
		VERILOG_INCLUDE_DIRS='$(E2E_VERILATOR_INCLUDE_DIRS)' \
		EXTRA_ARGS='$(GEMM_EXTRA_ARGS) $(E2E_VERILATOR_EXTRA_ARGS)' \
		BUILD_ARGS='$(GEMM_BUILD_ARGS)' \
			CUSTOM_COMPILE_DEPS='$(SIM_DIR)/Makefile $(E2E_RTL)' \
		2>&1 | tee $(SIM_BUILD)/test.log

test_e2e_program: SIM_BUILD = sim_build_e2e_program
test_e2e_program: SIM = verilator
test_e2e_program:
	@echo "=== Reusing $(SIM_BUILD) when cocotb/Verilator dependencies are current ==="
	@echo "=== Verilator C++ build jobs: $(VERILATOR_BUILD_JOBS) ==="
	@mkdir -p $(REPO_ROOT)/build
	@echo "=== Building firmware_program firmware ==="
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 \
		-I$(REPO_ROOT)/sw/driver \
		-c $(SIM_DIR)/cocotb/firmware_program.c -o $(REPO_ROOT)/build/firmware_program.o
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 \
		-I$(REPO_ROOT)/sw/driver \
		-c $(REPO_ROOT)/sw/driver/npu_driver.c -o $(REPO_ROOT)/build/npu_driver_program.o
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib \
		-T $(REPO_ROOT)/sw/linker/riscv32.ld \
		-o $(REPO_ROOT)/build/firmware_program.elf \
		$(REPO_ROOT)/build/firmware_program.o \
		$(REPO_ROOT)/build/npu_driver_program.o
	riscv32-none-elf-objcopy -O binary $(REPO_ROOT)/build/firmware_program.elf $(REPO_ROOT)/build/firmware_program.bin
	@python3 -c "import sys; data=open('$(REPO_ROOT)/build/firmware_program.bin','rb').read(); data+=b'\x00'*(-len(data)%4); [print(f'{int.from_bytes(data[i:i+4],\"little\"):08x}') for i in range(0,len(data),4)]" > $(REPO_ROOT)/build/firmware_program.hex
	@mkdir -p $(SIM_BUILD)/sim/verilog
	cp $(REPO_ROOT)/build/firmware_program.hex $(SIM_BUILD)/sim/verilog/firmware_program.hex
	cp $(REPO_ROOT)/build/firmware_program.hex $(SIM_BUILD)/sim/verilog/firmware.hex
	@printf '%s\n' '+define+HEX_FILE=\"$(SIM_BUILD)/sim/verilog/firmware_program.hex\"' > $(SIM_BUILD)/hex_def.f
	@mkdir -p $(REPO_ROOT)/sim/verilog
	cp $(REPO_ROOT)/build/firmware_program.hex $(REPO_ROOT)/sim/verilog/firmware.hex
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_e2e_program TOPLEVEL=top_soc \
	VERILOG_SOURCES="$(E2E_RTL)" \
	$(MAKE) SIM=verilator \
		VERILOG_INCLUDE_DIRS='$(E2E_VERILATOR_INCLUDE_DIRS)' \
		EXTRA_ARGS='$(GEMM_EXTRA_ARGS) $(E2E_VERILATOR_EXTRA_ARGS)' \
		BUILD_ARGS='$(GEMM_BUILD_ARGS)' \
		CUSTOM_COMPILE_DEPS='$(SIM_DIR)/Makefile $(E2E_RTL)' \
		2>&1 | tee $(SIM_BUILD)/test.log

rebuild_e2e_gemm:
	@echo "=== Removing sim_build_e2e_gemm for an explicit Verilator rebuild ==="
	rm -rf sim_build_e2e_gemm
	$(MAKE) test_e2e_gemm

# ── E2E Full SoC Integration ──

test_e2e: SIM_BUILD = sim_build_e2e
test_e2e:
	@rm -rf $(SIM_BUILD)
	@python3 $(REPO_ROOT)/tools/codegen/generate_golden.py --header > $(REPO_ROOT)/build/golden_gemm.h
	$(call build_fw,firmware_e2e, \
		$(SIM_DIR)/cocotb/firmware_e2e.c, \
		-I$(REPO_ROOT)/build -I$(REPO_ROOT)/sw/driver -I$(REPO_ROOT)/sw/runtime)
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 \
		-I$(REPO_ROOT)/sw/driver -I$(REPO_ROOT)/sw/runtime \
		-c $(REPO_ROOT)/sw/driver/npu_driver.c -o $(REPO_ROOT)/build/npu_driver_e2e.o
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 \
		-I$(REPO_ROOT)/sw/driver -I$(REPO_ROOT)/sw/runtime \
		-c $(REPO_ROOT)/sw/runtime/npu_runtime.c -o $(REPO_ROOT)/build/npu_runtime_e2e.o
	riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib \
		-T $(REPO_ROOT)/sw/linker/riscv32.ld \
		-o $(REPO_ROOT)/build/firmware_e2e.elf \
		$(REPO_ROOT)/build/firmware_e2e.o \
		$(REPO_ROOT)/build/npu_driver_e2e.o \
		$(REPO_ROOT)/build/npu_runtime_e2e.o
	riscv32-none-elf-objcopy -O binary $(REPO_ROOT)/build/firmware_e2e.elf $(REPO_ROOT)/build/firmware_e2e.bin
	@python3 -c "import sys; data=open('$(REPO_ROOT)/build/firmware_e2e.bin','rb').read(); data+=b'\x00'*(-len(data)%4); [print(f'{int.from_bytes(data[i:i+4],\"little\"):08x}') for i in range(0,len(data),4)]" > $(REPO_ROOT)/build/firmware_e2e.hex
	@mkdir -p $(SIM_BUILD)/sim/verilog
	cp $(REPO_ROOT)/build/firmware_e2e.hex $(SIM_BUILD)/sim/verilog/firmware_e2e.hex
	cp $(REPO_ROOT)/build/firmware_e2e.hex $(SIM_BUILD)/sim/verilog/firmware.hex
	@printf '%s\n' '+define+HEX_FILE=\"$(SIM_BUILD)/sim/verilog/firmware.hex\"' > $(SIM_BUILD)/hex_def.f
	@mkdir -p $(REPO_ROOT)/sim/verilog
	cp $(REPO_ROOT)/build/firmware_e2e.hex $(REPO_ROOT)/sim/verilog/firmware.hex
	PYTHONPATH=$(SIM_DIR)/cocotb:$$PYTHONPATH \
	MODULE=test_e2e TOPLEVEL=top_soc \
	VERILOG_SOURCES="$(E2E_RTL)" \
	$(MAKE) COMPILE_ARGS='$(E2E_COMPILE_ARGS)' \
		2>&1 | tee $(SIM_BUILD)/test.log

# ── Batch targets ──

e2e_boot_log:
	$(call show_log,sim_build_boot/test.log)

e2e_csr_log:
	$(call show_log,sim_build_csr/test.log)

e2e_dma_log:
	$(call show_log,sim_build_e2e_dma/test.log)

e2e_gemm_log:
	$(call show_log,sim_build_e2e_gemm/test.log)

e2e_program_log:
	$(call show_log,sim_build_e2e_program/test.log)

e2e_log:
	$(call show_log,sim_build_e2e/test.log)

test_quick:
	$(MAKE) test_gemm
	$(MAKE) test_npu_top

test_all:
	$(MAKE) test_pe
	$(MAKE) test_valu
	$(MAKE) test_sfu
	$(MAKE) test_dma
	$(MAKE) test_dma_load
	$(MAKE) test_memory
	$(MAKE) test_if_id
	$(MAKE) test_gemm
	$(MAKE) test_npu_top

clean::
	rm -rf sim_build sim_build_* __pycache__
