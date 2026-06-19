# NPU Project Tracker

> Single source of truth for project progress. Update this file as tasks complete.
> Implementation plan: `docs/superpowers/plans/2026-06-19-npu-full-implementation.md`

## Status

| Item | Status |
|------|--------|
| Phase 0 (Setup) | ⬜ pending |
| Phase 1 (NPU Core RTL) | ⬜ pending |
| Phase 2 (NPU Integration) | ⬜ pending |
| Phase 3 (RISC-V + SoC) | ⬜ pending |
| Phase 4 (Software / C) | ⬜ pending |
| Phase 5 (Python Toolchain) | ⬜ pending |
| Phase 6 (Verification) | ⬜ pending |
| Phase 7 (Polish) | ⬜ pending |

## Task Checklist

### Phase 0 — Setup
- [ ] 0.1 Project skeleton (Makefile, README)
- [ ] 0.2 Clone reference projects into ref/
- [ ] 0.3 Shared RTL definitions (npu_defines.svh, isa_defines.svh)

### Phase 1 — NPU Core RTL (parallel)
- [ ] 1.1 PE cell + cocotb test
- [ ] 1.2 Systolic Array 16×16 + Requant + cocotb test
- [ ] 1.3 Vector ALU (64-lane SIMD) + cocotb test
- [ ] 1.4 SFU (LUT: GELU/Sigmoid/Tanh, Quant/Dequant) + cocotb test
- [ ] 1.5 DMA Engine + cocotb test
- [ ] 1.6 Memory System (SRAM banks, Crossbar) + cocotb test
- [ ] 1.7 IF/ID/Dispatch Pipeline + cocotb test

### Phase 2 — NPU Integration
- [ ] 2.1 NPU Top-Level (csr.sv, pingpong.sv, top.sv) + integration test

### Phase 3 — RISC-V + SoC
- [ ] 3.1 PicoRV32 adaptation (wrapper, memory map, hello_npu.S)
- [ ] 3.2 SoC Top-Level (axi_crossbar_soc, top_soc, ext_mem_model, Verilator harness)

### Phase 4 — Software (C)
- [ ] 4.1 NPU Driver (npu_csr.h, npu_driver.c)
- [ ] 4.2 NPU Runtime (tile split, descriptor gen, instruction issue)
- [ ] 4.3 Linker script + Demo application + sw/Makefile

### Phase 5 — Python Toolchain
- [ ] 5.1 Quantizer (per-tensor + per-channel) + Compiler (FX graph → NPU inst) + Serializer (.npu binary)

### Phase 6 — Verification
- [ ] 6.1 End-to-end cocotb test (PyTorch golden vs NPU RTL)
- [ ] 6.2 Yosys synthesis (area/timing estimates)

### Phase 7 — Polish
- [ ] 7.1 Format all code, update README, final commit

## Quick Reference

```bash
nix develop                    # enter dev environment
make format                    # format RTL (verible) + C (clang-format)
make sim-gemm                  # run GEMM cocotb test
make sim-e2e                   # run end-to-end test
make synth                     # synthesize with Yosys
make sw                        # build RISC-V demo
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
