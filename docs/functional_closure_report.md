# NPU Functional Closure Report

Date: 2026-06-21

## Scope

This checkpoint closes the functional path needed for the assignment stack:

- RTL NPU execution of generated instruction streams.
- RISC-V bare-metal driver and runtime APIs.
- PyTorch-like FX frontend that emits firmware-consumable `.npu` binaries.
- RTL simulation of generated instructions, including data correctness checks.

Performance targets from `NPU_design_spec.pdf` are tracked separately. DMA
outstanding depth, compute/DMA overlap, GEMM steady-state throughput, and SFU
throughput are not claimed by this functional checkpoint.

## Implemented Stack

### RTL

- IF/ID executes `.npu` instruction streams from I-SRAM.
- `GEMM -> SYNC -> ACT_RELU -> WFI` is supported as a generated program.
- IF/ID `ACT_RELU` can postprocess the GEMM P-buffer in RTL when the GEMM
  output ping-pong bank is ready.
- Existing CSR-driven SFU descriptor/tile paths remain supported.
- Ping-pong SRAM windows are the software-visible data placement contract for
  GEMM A/B/P, VALU input/output, and SFU input/output tiles.

### Driver

- `npu_run_program()` validates the `.npu` header, loads instructions into
  I-SRAM, loads descriptors into D-SRAM, sets `CSR_DESC_PTR`, starts IF/ID, and
  waits for completion.
- DMA 1D/2D, descriptor load, direct issue, and wait APIs remain available for
  runtime and focused contract tests.

### Runtime

- `npu_rt_gemm()` now uses tile-level slab DMA loads for A/B and one tile DMA
  store for C, matching the RTL ping-pong fill/consume contract.
- `npu_rt_relu()` and `npu_rt_gelu()` use SFU descriptors and the SFU
  input/output ping-pong windows.

### PyTorch-like Frontend

- `tools/codegen/npu_torch.py::compile_model()` traces a PyTorch module with
  `torch.fx`, lowers supported ops, and writes a `.npu` binary.
- `NpuCompiler` emits executable programs by inserting conservative `SYNC`
  barriers between generated ops and appending a final `WFI`.
- Linear + ReLU compiles to:

```text
GEMM, SYNC, ACT_RELU, WFI
```

## Verification

Commands run for this checkpoint:

```bash
nix develop --command bash -lc \
  'riscv32-none-elf-gcc -march=rv32im -mabi=ilp32 -nostdlib -ffreestanding -O2 -Isw/driver -Isw/runtime -c sw/runtime/npu_runtime.c -o build/npu_runtime_check.o && echo RUNTIME_COMPILE_PASS'
```

Result: `RUNTIME_COMPILE_PASS`

```bash
nix develop --command bash -lc \
  'python - <<PY
import struct, tempfile, os
import torch
import torch.nn as nn
from tools.codegen.npu_torch import compile_model
from tools.codegen.serialize import Opcode
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)
    def forward(self, x):
        return torch.relu(self.linear(x))
fd, path = tempfile.mkstemp(suffix=".npu")
os.close(fd)
try:
    result = compile_model(Model(), torch.randn(1, 16), path)
    ops = [int(i.opcode) for i in result.graph.instructions]
    data = open(path, "rb").read()
    ver, ni, nd = struct.unpack("<III", data[4:16])
    assert data[:4] == b"NPUC"
    assert (ver, ni, nd) == (1, 4, 1)
    assert ops == [int(Opcode.GEMM), int(Opcode.SYNC), int(Opcode.ACT_RELU), int(Opcode.WFI)]
    print("CODEGEN_SMOKE_PASS", ops, len(data))
finally:
    os.unlink(path)
PY'
```

Result: `CODEGEN_SMOKE_PASS [1, 240, 32, 241] 52`

```bash
nix develop --command bash -lc \
  'cd sim && make test_e2e_program 2>&1 | rg "PASS|FAIL|ERROR|AssertionError|TESTS=|UART|program|timeout|Timeout|trap|debug|progress|GEMM|SFU|ReLU|DMA"'
```

Result:

```text
UART_TX progress: iabrgpsP
E2E program-stream test: PASS
TESTS=1 PASS=1 FAIL=0 SKIP=0
```

```bash
nix develop --command bash -lc \
  'cd sim && COCOTB_TEST_FILTER="test_ifid_relu_sfu_completes\|test_tile_flow_gemm_sfu_relu_dma_store_result" make test_npu_top 2>&1 | rg "PASS|FAIL|ERROR|AssertionError|TESTS=|relu|SFU|GEMM|DMA|Timeout|timeout"'
```

Result:

```text
test_tile_flow_gemm_sfu_relu_dma_store_result PASS
test_ifid_relu_sfu_completes PASS
TESTS=2 PASS=2 FAIL=0 SKIP=0
```

```bash
nix develop --command bash -lc \
  'cd sim && make test_npu_top 2>&1 | rg "PASS|FAIL|ERROR|AssertionError|TESTS=|Timeout|timeout|GEMM|SFU|ReLU|DMA|ping|bank"'
```

Result:

```text
TESTS=30 PASS=30 FAIL=0 SKIP=0
```

```bash
nix develop --command bash -lc \
  'cd sim && make test_pingpong 2>&1 | rg "PASS|FAIL|ERROR|AssertionError|TESTS=|ping|bank|Timeout|timeout"'
```

Result:

```text
TESTS=4 PASS=4 FAIL=0 SKIP=0
```

The `test_pingpong` target still returned a non-zero Makefile status after
printing the all-pass cocotb summary. No cocotb assertion failed.

`pytest` is not installed in the current nix shell, so
`python -m pytest -q tests/test_codegen.py` could not be run in this
environment. The equivalent codegen smoke above was run with the actual
`torch` import and `.npu` binary generation path.

```bash
nix develop --command bash -lc \
  'cd sim && make test_e2e 2>&1 | rg -e "--prefix|verilator|Vtop|vvp|Makefile.icarus|PASS|FAIL|ERROR|AssertionError|TESTS=|UART|timeout|Timeout|stage|ISRAM|VSRAM|ExtMem|GEMM|ReLU|undefined reference"'
```

Result:

```text
Running on Verilator version 5.048
UART_TX[0] = 0x50 ('P')
PASS: E2E pass - UART_TX = 'P'
UART_TX[0] = 0x45 ('E')
PASS: E2E fail-detect - UART_TX = 'E'
TESTS=2 PASS=2 FAIL=0 SKIP=0
```

## Functional Limitations

- Supported frontend patterns are the existing FX patterns: Linear, basic
  activations, and simple elementwise lowering. The generated instruction
  stream is serialized for correctness.
- Runtime GEMM assumes dimensions are multiples of 16 and currently uses the
  single-tile RTL contract for each M/N tile.
- Performance features are intentionally left for a later phase: outstanding
  DMA depth 8, true compute/DMA overlap, GEMM throughput closure, and SFU
  throughput closure.
