# Yosys Synthesis Report — NPU Project

**Date:** 2026-06-26
**Tool:** Yosys 0.62
**Target:** Generic gate library (no specific technology node)
**Methodology:** Individual module synthesis with `synth -flatten`

---

## Overview

This report presents the gate-level synthesis results for the NPU (Neural Processing Unit) design. Each module was synthesized individually using Yosys 0.62 with the `synth -flatten` command targeting the generic built-in cell library. No specific ASIC technology node or FPGA device was targeted; cell counts represent abstract gate equivalents.

The top-level module (`npu_top`) was **not** synthesized as a whole due to the 11 GB RAM limit on the remote build server. Area estimates are therefore the arithmetic sum of individually synthesized submodules.

---

## Results

| Module | Cells | Wires | Notes |
|--------|------:|------:|-------|
| systolic_array | 310,636 | 276,591 | 256 PEs + gemm_controller, weight-stationary |
| valu_top | 170,395 | 150,750 | 64 VALU lanes + register file |
| vregfile | 132,721 | 115,322 | Vector register file (64x32bx8 regs) |
| sram_bank | 67,682 | 35,884 | 1024x32b SRAM (behavioral) |
| ifetch | 16,939 | 8,966 | Instruction fetch + I-SRAM interface |
| csr | 2,719 | 2,072 | Control/status registers |
| quant_dequant | 3,130 | 3,035 | INT8<->FP16 quantization/dequantization |
| sfu_top | 3,944 | 3,813 | Special function unit with LUT engine |
| pe | 1,111 | 993 | Processing element (MAC unit) |
| lut_engine | 632 | 627 | LUT-based activation functions |
| valu_lane | 556 | 552 | Single VALU lane |
| gemm_controller | 102 | 89 | GEMM FSM controller |
| pingpong | 21 | 25 | Ping-pong buffer control |
| idecode | 7 | 13 | Instruction decoder |
| **Total (estimated)** | **721,314** | **602,832** | Sum of individual modules |

---

## Analysis

### Dominant Modules

The three largest modules account for 85% of the total cell count:

1. **systolic_array (43%)** — 310,636 cells. This is the 16x16 weight-stationary systolic array containing 256 processing elements and the GEMM controller. Each PE contains a MAC unit with local weight storage. The cell count scales as expected for a 256-element array with multiply-accumulate datapaths and interconnect steering logic.

2. **valu_top (24%)** — 170,395 cells. The 64-lane SIMD vector arithmetic unit, including the vector register file. This handles element-wise operations (add, multiply, activation functions) across 64 data lanes in parallel.

3. **vregfile (18%)** — 132,721 cells. The vector register file providing 8 registers of 64 lanes x 32 bits each. Register files are area-intensive due to the dense read/write port multiplexing required.

### Memory Subsystem

4. **sram_bank (9%)** — 67,682 cells. Behavioral SRAM modeled at 1024x32b. In a physical ASIC flow this would be replaced by a compiled SRAM macro, which would not consume standard-cell area but would occupy dedicated silicon. The cell count here reflects the flip-flop-based behavioral model.

### Control and Datapath Support

The remaining modules (ifetch, csr, quant_dequant, sfu_top, and smaller units) total 27,431 cells (~4%). These include:

- **ifetch** (16,939 cells) — Instruction fetch unit with I-SRAM interface, the largest of the control-path modules.
- **sfu_top** (3,944 cells) — Special function unit containing the LUT engine for nonlinear activation functions (ReLU, sigmoid, tanh approximations).
- **quant_dequant** (3,130 cells) — INT8/FP16 format conversion logic for mixed-precision inference.
- **csr** (2,719 cells) — Control/status register bank for configuration and status reporting.
- **pe, lut_engine, valu_lane** — Synthesized individually as reference points; their cell counts are subcomponents already counted within the larger parent modules above.

### Area Breakdown by Function

| Function | Cells | Percentage |
|----------|------:|-----------:|
| Matrix multiplication (systolic_array) | 310,636 | 43.1% |
| Vector arithmetic (valu_top) | 170,395 | 23.6% |
| Register file (vregfile) | 132,721 | 18.4% |
| Memory (sram_bank) | 67,682 | 9.4% |
| Control & support logic | 27,431 | 3.8% |
| Duplicate submodules (pe, lut_engine, valu_lane) | 2,399 | 0.3% |
| **Net total (non-duplicate)** | **~718,914** | **~99.7%** |

Note: pe, lut_engine, and valu_lane are subcomponents of systolic_array, sfu_top, and valu_top respectively. Their standalone cell counts are included in the raw sum but are already accounted for in the parent module totals.

---

## Conclusions

1. **Clean synthesis.** All 14 modules synthesize without errors or warnings under Yosys 0.62 generic library. No combinational loops, undriven nets, or other structural issues were reported.

2. **Systolic array dominates.** At 43% of total cells, the 16x16 PE array is the expected area bottleneck. Any area optimization effort should focus on PE microarchitecture (e.g., sharing multiplier resources, reducing weight storage per PE).

3. **Register file is significant.** The vector register file at 18% is a known area cost of wide SIMD architectures. Compiler-driven register allocation and reduced register count could trade area for spill/reload overhead.

4. **SRAM is behavioral.** The 9% attributed to sram_bank would disappear in a real ASIC flow where compiled SRAM macros are used. This inflates the gate-level estimate but does not reflect actual silicon area accurately.

5. **Total estimate: ~721k cells.** This is a rough gate-equivalent figure. Mapping to a specific PDK (e.g., TSMC 28nm, GF 22FDX) would change the cell count due to technology-specific cell libraries and optimization passes. Timing-driven synthesis would also produce different results.

6. **RAM limitation.** The full `npu_top` module could not be synthesized due to the 11 GB RAM constraint. Synthesis of the complete design with hierarchy flattening would likely require 32+ GB and may reveal additional optimization opportunities (cross-module constant propagation, dead logic elimination) not visible in the per-module runs.

---

*Report generated from Yosys 0.62 synthesis runs on the NPU RTL. No specific PDK or timing constraints were applied. Cell counts are gate-equivalent estimates using the generic library.*
