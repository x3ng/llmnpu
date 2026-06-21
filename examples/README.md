# NPU PyTorch Codegen Demo

This host-side demo compiles a small PyTorch model into the `.npu` binary
format consumed by the firmware/runtime path:

```bash
python examples/compile_linear_relu.py
```

The demo uses `tools/codegen/npu_torch.py`:

```python
result = compile_model(model, torch.randn(1, 16), "build/linear_relu.npu")
```

The emitted file layout is the existing codegen serializer format:

- `NPUC` magic
- version and instruction/descriptor counts
- 32-bit instruction words
- 19-byte GEMM descriptors matching `gemm_desc_t`

Supported lowering is intentionally limited to the current `torch.fx`
compiler: `nn.Linear`, ReLU/GELU/Sigmoid/Tanh, and simple add/mul elementwise
ops.
