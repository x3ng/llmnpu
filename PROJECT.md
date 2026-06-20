# NPU Project Tracker

> Single source of truth for project progress. Update this file as tasks complete.
> Implementation plan: `docs/superpowers/plans/2026-06-19-npu-full-implementation.md`

## Status

| Item | Status |
|------|--------|
| Phase 0 (Setup) | ✅ complete |
| Phase 1 (NPU Core RTL) | ✅ complete — 7/7 PASS |
| Phase 2 (NPU Integration) | ✅ complete — top.sv verified |
| Phase 3 (RISC-V + SoC) | ✅ complete — PicoRV32 adapted, SoC top-level, Verilator harness |
| Phase 4 (Software / C) | ✅ complete — C driver, runtime, demo, linker script |
| Phase 5 (Python Toolchain) | ✅ complete — quantize, compiler, serialize |
| Phase 6 (Verification) | 🔧 in progress — CSR+Boot PASS, DMA+GEMM hex loading fixed, awaiting re-test |
| Phase 7 (Polish) | ⬜ pending |

## Task Checklist

### Phase 0 — Setup
- [x] 0.1 Project skeleton (Makefile, README)
- [x] 0.2 Clone reference projects into ref/
- [x] 0.3 Shared RTL definitions (npu_defines.svh, isa_defines.svh)

### Phase 1 — NPU Core RTL (parallel)
- [x] 1.1 PE cell + cocotb test
- [x] 1.2 Systolic Array 16×16 + Requant + cocotb test
- [x] 1.3 Vector ALU (64-lane SIMD) + cocotb test
- [x] 1.4 SFU (LUT: GELU/Sigmoid/Tanh, Quant/Dequant) + cocotb test
- [x] 1.5 DMA Engine + cocotb test
- [x] 1.6 Memory System (SRAM banks, Crossbar) + cocotb test
- [x] 1.7 IF/ID/Dispatch Pipeline + cocotb test

### Phase 2 — NPU Integration
- [x] 2.1 NPU Top-Level (csr.sv, pingpong.sv, top.sv) + integration test

### Phase 3 — RISC-V + SoC
- [x] 3.1 PicoRV32 adaptation (wrapper, memory map, hello_npu.S)
- [x] 3.2 SoC Top-Level (axi_crossbar_soc, top_soc, ext_mem_model, Verilator harness)

### Phase 4 — Software (C)
- [x] 4.1 NPU Driver (npu_csr.h, npu_driver.c)
- [x] 4.2 NPU Runtime (tile split, descriptor gen, instruction issue)
- [x] 4.3 Linker script + Demo application + sw/Makefile

### Phase 5 — Python Toolchain
- [x] 5.1 Quantizer (per-tensor + per-channel) + Compiler (FX graph → NPU inst) + Serializer (.npu binary)

### Phase 6 — Verification
- [x] 6.1 E2E CSR (PASS)
- [x] 6.1 E2E Boot (PASS)
- [ ] 6.1 E2E DMA (hex loading fixed, re-test pending)
- [ ] 6.1 E2E GEMM (hex loading fixed, re-test pending)
- [ ] 6.2 Yosys synthesis (area/timing estimates)

### Phase 7 — Polish
- [ ] 7.1 Format all code, update README, final commit

## Quick Reference

```bash
nix develop                    # enter dev environment
make test_all                  # run all unit tests (7 targets)
make test_e2e_csr              # CSR E2E test
make test_e2e_boot             # Boot E2E test
make test_e2e_dma              # DMA E2E test
make test_e2e_gemm             # GEMM E2E test (verilator)
```

## Reference Projects (in ref/, gitignored)

| Project | License | Used For |
|---------|---------|----------|
| [tiny-tpu](https://github.com/RightNow-AI/tiny-tpu) | MIT | ISA design, activation units, systolic array microarchitecture |
| [PicoRV32](https://github.com/YosysHQ/picorv32) | ISC | RISC-V control CPU |
| [verilog-axi](https://github.com/alexforencich/verilog-axi) | MIT | AXI DMA, crossbar components |

## Key Decisions Log

| Date | Decision |
|------|----------|
| 2026-06-19 | RTL language: SystemVerilog |
| 2026-06-19 | RISC-V core: adapt PicoRV32 |
| 2026-06-19 | Verification: cocotb + PyTorch golden model |
| 2026-06-19 | Quantization: per-tensor first, per-channel extension |
| 2026-06-19 | Environment: flake.nix (nixpkgs 26.05) |
| 2026-06-19 | Code formatting: verible-verilog-format + clang-format |
| 2026-06-20 | Hex loading: rely on default path (sim/verilog/firmware.hex) — no -P/-G overrides |
| 2026-06-20 | Debug CSR: added DEBUG register (0x60) for per-unit busy/FSM state exposure |
| 2026-06-20 | Testbench: added [DIAG] self-diagnosis markers to cocotb tests |
| 2026-06-20 | Memory: 6-lesson knowledge base in memory/ for cross-session persistence |

## Known Issues

### Hex Loading Saga

**Problem:** E2E DMA and GEMM tests failed because the Verilator harness could not locate the firmware hex file. The harness was using `-P` and `-G` plusarg overrides to pass hex paths, but these overrides were not connected to the actual `$readmemh` calls in the RTL.

**Root Cause:** The `$readmemh` calls in the RTL used a hardcoded default path (`sim/verilog/firmware.hex`), while the testbench was setting plusargs that never reached the RTL-side file I/O.

**Fix Applied:** Removed the `-P`/`-G` overrides from the cocotb testbench. The harness now relies on the default path baked into the RTL. The Makefile copies the built firmware to `sim/verilog/firmware.hex` before running E2E tests.

**Status:** Fix applied, awaiting re-test of DMA and GEMM E2E tests.
