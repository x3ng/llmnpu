#!/usr/bin/env python3
"""Generate LUT hex files for NPU SFU activation functions.

Each LUT has 32 entries covering the domain [-4, 4] uniformly with a step
of 0.25 (i.e. samples at x = -4.0, -3.75, -3.5, ..., 3.5, 3.75).

Values are quantised to INT8 (-128..127) using full-range normalisation
and written as .hex files for $readmemh consumption.

Functions:
  GELU   — 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))
  Sigmoid — 1/(1+exp(-x))
  Tanh   — hyperbolic tangent

Usage:
  python tools/codegen/generate_luts.py
"""

import math
import os
import sys

import numpy as np


# ── Activation functions (single-element, math-domain) ──────────────────

def gelu(x: float) -> float:
    """GELU: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))."""
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))


def sigmoid(x: float) -> float:
    """Logistic sigmoid."""
    return 1.0 / (1.0 + math.exp(-x))


def tanh(x: float) -> float:
    """Hyperbolic tangent."""
    return math.tanh(x)


# ── LUT generation ─────────────────────────────────────────────────────

def gen_lut(fn, n=32):
    """Generate an n-entry LUT for *fn*, sampled uniformly over [-4, 4].

    Each entry is at the *left* boundary of its INT8 bin so that the
    hardware interpolation ``y = lut[idx] + (lut[idx+1]-lut[idx])*frac/8``
    is exact at the bin boundaries.  The *n*-th (last) entry corresponds
    to x = -4.0 + (n-1)*0.25 = 3.75.

    Returns a list of ``n`` Python ``int`` values in [-128, 127].
    """
    xs = np.linspace(-4.0, 3.75, n)          # left-boundary samples
    ys = np.array([fn(float(x)) for x in xs], dtype=np.float64)

    scale = 127.0 / float(max(abs(ys).max(), 1e-8))
    quantised = np.clip(np.round(ys * scale), -128.0, 127.0).astype(np.int8)
    return quantised.tolist()


def write_hex(values, path):
    """Write *values* as a one-value-per-line hex file for ``$readmemh``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            f.write(f"{v & 0xFF:02x}\n")
    print(f"  wrote {len(values)} entries  ->  {path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    # Resolve output directory relative to the repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(os.path.dirname(script_dir))  # npu/
    out_dir    = os.path.join(repo_root, "synth", "luts")

    print("NPU SFU — LUT generator")
    print("=" * 50)

    lut_specs = [
        ("GELU",    gelu,    "gelu.hex"),
        ("Sigmoid", sigmoid, "sigmoid.hex"),
        ("Tanh",    tanh,    "tanh.hex"),
    ]

    for name, fn, filename in lut_specs:
        vals = gen_lut(fn)
        write_hex(vals, os.path.join(out_dir, filename))

        xs = np.linspace(-4.0, 3.75, 32)
        print(f"    {name:8s}  [{xs[0]:5.2f}..{xs[-1]:5.2f}]  "
              f"min={min(vals):4d}  max={max(vals):4d}")

    print("=" * 50)
    print("Done — LUT hex files ready in", out_dir)


if __name__ == "__main__":
    main()
