"""FFT module — pure-Python mixed-radix implementation.

Exposes ``fft`` as a convenience alias for ``cfft`` (complex forward
transform), so callers can write ``mdarray.fft.fft(x)`` in the same
vein as ``numpy.fft.fft``.
"""

from .python.factorize import factorize
from .python.fft import cfft, fftn, ifft, ifftn, rfft

# Convenience alias: fft ≡ cfft
fft = cfft

BACKEND = "python"

__all__ = ["BACKEND", "cfft", "factorize", "fft", "fftn", "ifft", "ifftn", "rfft"]
