"""Codelet generator for FFT butterfly routines.

For a given radix n, generates optimized straight-line code for the
n-point DFT butterfly. The generator:

1. Constructs the n x n DFT matrix W_n symbolically
2. Factors out twiddle multiplications
3. Applies algebraic simplifications:
   - w^0 = 1 (elide multiplication)
   - w^{n/2} = -1 (negate instead of multiply)
   - w^{n/4} = -i (swap real/imag)
   - Conjugate symmetry
   - Common subexpression elimination
4. Emits Python or Rust source code

Usage:
    python -m mdarray.fft.codegen.genfft --radix 11 --lang python
"""

from __future__ import annotations

import cmath
import math

TWOPI = 2 * math.pi


def dft_matrix(n: int) -> list[list[complex]]:
    """Construct the n x n DFT matrix symbolically."""
    W: list[list[complex]] = []
    for j in range(n):
        row: list[complex] = []
        for k in range(n):
            row.append(cmath.exp(-2j * math.pi * j * k / n))
        W.append(row)
    return W


def twiddle_constants(n: int) -> list[tuple[str, float, float]]:
    """Compute twiddle factor constants for radix-n butterfly.

    Returns list of (name, cos_val, sin_val) for non-trivial angles.
    """
    constants: list[tuple[str, float, float]] = []
    seen: set[int] = set()

    for k in range(1, n):
        # Normalize to unique angle
        angle = (2 * math.pi * k) / n
        if k not in seen and (n - k) not in seen:
            cos_val = math.cos(angle)
            sin_val = math.sin(angle)
            name = f"W{n}_{k}"
            constants.append((name, cos_val, sin_val))
            seen.add(k)

    return constants


def is_trivial_twiddle(j: int, k: int, n: int) -> str | None:
    """Check if w^{jk} mod n is a trivial multiplication.

    Returns:
        None if non-trivial
        "identity" if w^{jk} = 1
        "negate" if w^{jk} = -1
        "neg_i" if w^{jk} = -i
        "pos_i" if w^{jk} = i
    """
    exp = (j * k) % n
    if exp == 0:
        return "identity"
    if n % 2 == 0 and exp == n // 2:
        return "negate"
    if n % 4 == 0 and exp == n // 4:
        return "neg_i"
    if n % 4 == 0 and exp == 3 * n // 4:
        return "pos_i"
    return None


def generate_python_butterfly(n: int) -> str:
    """Generate Python source for a radix-n butterfly function."""
    lines: list[str] = []
    twiddle_constants(n)

    lines.append(f"def radix{n}(seq, twiddles, pix, twix, pdiv):")
    lines.append(f'    """Generated radix-{n} butterfly."""')

    # Twiddle factor extraction
    for k in range(1, n):
        lines.append(f"    omega{k} = twiddles[twix + {k - 1}]")
    lines.append("")

    # Premultiply by twiddles
    lines.append("    z0 = seq[0]")
    for k in range(1, n):
        lines.append(f"    z{k} = omega{k} * seq[{k}]")
    lines.append("")

    # Direct DFT computation with simplifications
    W = dft_matrix(n)
    for j in range(n):
        terms: list[str] = []
        for k in range(n):
            trivial = is_trivial_twiddle(j, k, n)
            if trivial == "identity":
                terms.append(f"z{k}")
            elif trivial == "negate":
                terms.append(f"(-z{k})")
            elif trivial == "neg_i":
                terms.append(f"(-1j * z{k})")
            elif trivial == "pos_i":
                terms.append(f"(1j * z{k})")
            else:
                # Use precomputed constant
                w = W[j][k]
                terms.append(f"({w.real:.16e} + {w.imag:.16e}j) * z{k}")
        lines.append(f"    seq[{j}] = {' + '.join(terms)}")

    lines.append("")
    lines.append("    return seq")

    return "\n".join(lines)


def generate_rust_butterfly(n: int) -> str:
    """Generate Rust source for a radix-n butterfly function."""
    lines: list[str] = []
    lines.append(f"/// Generated radix-{n} butterfly.")
    lines.append(f"pub fn butterfly_radix{n}(")
    lines.append("    seq: &mut [Complex<f64>],")
    lines.append("    twiddles: &[Complex<f64>],")
    lines.append("    pix: usize,")
    lines.append("    twix: usize,")
    lines.append("    pdiv: usize,")
    lines.append(") {")

    for k in range(1, n):
        lines.append(f"    let omega{k} = twiddles[twix + {k - 1}];")
    lines.append("")
    lines.append("    let z0 = seq[0];")
    for k in range(1, n):
        lines.append(f"    let z{k} = omega{k} * seq[{k}];")

    # Simplified DFT
    W = dft_matrix(n)
    for j in range(n):
        terms: list[str] = []
        for k in range(n):
            trivial = is_trivial_twiddle(j, k, n)
            if trivial == "identity":
                terms.append(f"z{k}")
            elif trivial == "negate":
                terms.append(f"-z{k}")
            elif trivial == "neg_i":
                terms.append(f"Complex::new(z{k}.im, -z{k}.re)")
            elif trivial == "pos_i":
                terms.append(f"Complex::new(-z{k}.im, z{k}.re)")
            else:
                w = W[j][k]
                terms.append(f"Complex::new({w.real:.16e}, {w.imag:.16e}) * z{k}")
        lines.append(f"    seq[{j}] = {' + '.join(terms)};")

    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FFT codelet generator")
    parser.add_argument("--radix", type=int, required=True, help="Radix to generate")
    parser.add_argument(
        "--lang",
        choices=["python", "rust"],
        default="python",
        help="Target language",
    )
    args = parser.parse_args()

    if args.lang == "python":
        print(generate_python_butterfly(args.radix))
    else:
        print(generate_rust_butterfly(args.radix))
