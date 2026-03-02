"""FFT codelet generator.

Generates optimized butterfly routines for arbitrary radices by symbolically
computing the DFT matrix, applying algebraic simplifications, and emitting
source code.

This mirrors the design of FFTW's genfft (Frigo & Johnson, 2005), but in
Python and simpler: no DAG-based scheduling, just symbolic DFT -> CSE ->
straight-line code emission.
"""
