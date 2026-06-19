# sim/Makefile — Cocotb simulation targets for NPU
# Each test target uses its own SIM_BUILD directory to avoid conflicts.

TOPLEVEL_LANG = verilog
SIM ?= icarus

REPO_ROOT = $(abspath $(PWD)/..)

COMPILE_ARGS = -I$(REPO_ROOT)/rtl/include
SIM_ARGS =
EXTRA_ARGS =

include $(shell cocotb-config --makefiles)/Makefile.sim

.PHONY: test_pe test_gemm test_valu test_sfu test_dma test_memory test_if_id
.PHONY: test_all clean

# ── Individual test targets (each isolated, can run in parallel) ──

test_pe: SIM_BUILD = sim_build_pe
test_pe:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(PWD)/cocotb:$$PYTHONPATH \
	MODULE=test_pe TOPLEVEL=pe \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/gemm/pe.sv" \
	$(MAKE)

test_gemm: SIM_BUILD = sim_build_gemm
test_gemm:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(PWD)/cocotb:$$PYTHONPATH \
	MODULE=test_gemm TOPLEVEL=systolic_array \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/gemm/pe.sv $(REPO_ROOT)/rtl/npu/gemm/systolic_array.sv $(REPO_ROOT)/rtl/npu/gemm/gemm_controller.sv" \
	$(MAKE)

test_valu: SIM_BUILD = sim_build_valu
test_valu:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(PWD)/cocotb:$$PYTHONPATH \
	MODULE=test_valu TOPLEVEL=valu_top \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/valu/valu_lane.sv $(REPO_ROOT)/rtl/npu/valu/vregfile.sv $(REPO_ROOT)/rtl/npu/valu/valu_top.sv" \
	$(MAKE)

test_sfu: SIM_BUILD = sim_build_sfu
test_sfu:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(PWD)/cocotb:$$PYTHONPATH \
	MODULE=test_sfu TOPLEVEL=sfu_top \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/sfu/lut_engine.sv $(REPO_ROOT)/rtl/npu/sfu/quant_dequant.sv $(REPO_ROOT)/rtl/npu/sfu/sfu_top.sv" \
	$(MAKE)

test_dma: SIM_BUILD = sim_build_dma
test_dma:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(PWD)/cocotb:$$PYTHONPATH \
	MODULE=test_dma TOPLEVEL=npu_dma \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/interconnect/axi_dma_wrapper.sv $(REPO_ROOT)/rtl/npu/dma/npu_dma.sv" \
	$(MAKE)

test_memory: SIM_BUILD = sim_build_memory
test_memory:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(PWD)/cocotb:$$PYTHONPATH \
	MODULE=test_memory TOPLEVEL=crossbar \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/memory/sram_bank.sv $(REPO_ROOT)/rtl/npu/memory/crossbar.sv" \
	$(MAKE)

test_if_id: SIM_BUILD = sim_build_if_id
test_if_id:
	@rm -rf $(SIM_BUILD)
	PYTHONPATH=$(PWD)/cocotb:$$PYTHONPATH \
	MODULE=test_if_id TOPLEVEL=if_id_top \
	VERILOG_SOURCES="$(REPO_ROOT)/rtl/npu/if_id/ifetch.sv $(REPO_ROOT)/rtl/npu/if_id/idecode.sv $(REPO_ROOT)/rtl/npu/if_id/dispatch.sv $(REPO_ROOT)/rtl/npu/if_id/if_id_top.sv" \
	$(MAKE)

# ── Batch targets ──

test_all:
	$(MAKE) test_pe
	$(MAKE) test_valu
	$(MAKE) test_sfu
	$(MAKE) test_dma
	$(MAKE) test_memory
	$(MAKE) test_if_id
	$(MAKE) test_gemm

clean::
	rm -rf sim_build sim_build_* __pycache__
