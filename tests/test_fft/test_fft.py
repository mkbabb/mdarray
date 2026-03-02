"""Comprehensive FFT correctness tests.

Validates the pure-Python mixed-radix FFT against NumPy as the oracle.
Tests cover all 11 categories from the test plan.
"""

from __future__ import annotations

import math

import pytest

# NumPy is a test-only dependency
np = pytest.importorskip("numpy")


def md_to_list(arr):
    """Extract flat data from a mdarray."""
    return list(arr.data)


def np_allclose(md_result, np_result, atol=1e-10, rtol=1e-10):
    """Compare mdarray FFT result against NumPy."""
    md_list = md_to_list(md_result)
    np_list = list(np_result.flatten())
    assert len(md_list) == len(np_list), f"Length mismatch: {len(md_list)} vs {len(np_list)}"
    for i, (a, b) in enumerate(zip(md_list, np_list, strict=False)):
        a, b = complex(a), complex(b)
        if abs(b) > 1e-15:
            assert abs(a - b) <= atol + rtol * abs(b), (
                f"Mismatch at index {i}: mdarray={a}, numpy={b}, diff={abs(a-b)}"
            )
        else:
            assert abs(a - b) <= atol, (
                f"Mismatch at index {i}: mdarray={a}, numpy={b}, diff={abs(a-b)}"
            )


# --- 1. Exact match against NumPy for various lengths ---

class TestFFTCorrectness:
    """Category 1: FFT output matches NumPy for all supported lengths."""

    @pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64, 128, 256])
    def test_power_of_2(self, N):
        from mdarray import cfft
        from mdarray.array import mdarray
        data = [complex(i, i * 0.5) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))
        result = cfft(arr)
        expected = np.fft.fft(data)
        np_allclose(result, expected)

    @pytest.mark.parametrize("N", [3, 9, 27])
    def test_power_of_3(self, N):
        from mdarray import cfft
        from mdarray.array import mdarray
        data = [complex(i, -i * 0.3) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))
        result = cfft(arr)
        expected = np.fft.fft(data)
        np_allclose(result, expected)

    @pytest.mark.parametrize("N", [5, 25])
    def test_power_of_5(self, N):
        from mdarray import cfft
        from mdarray.array import mdarray
        data = [complex(i * 0.1, i * 0.2) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))
        result = cfft(arr)
        expected = np.fft.fft(data)
        np_allclose(result, expected)

    @pytest.mark.parametrize("N", [7, 49])
    def test_power_of_7(self, N):
        from mdarray import cfft
        from mdarray.array import mdarray
        data = [complex(i) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))
        result = cfft(arr)
        expected = np.fft.fft(data)
        np_allclose(result, expected)

    @pytest.mark.parametrize("N", [6, 12, 24, 30, 60, 120])
    def test_highly_composite(self, N):
        from mdarray import cfft
        from mdarray.array import mdarray
        data = [complex(math.sin(2 * math.pi * i / N)) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))
        result = cfft(arr)
        expected = np.fft.fft(data)
        np_allclose(result, expected)

    @pytest.mark.parametrize("N", [11, 13, 17, 19, 23, 29, 31])
    def test_primes(self, N):
        """Exercises Bluestein's algorithm for small prime lengths."""
        from mdarray import cfft
        from mdarray.array import mdarray
        data = [complex(i, i % 3) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))
        result = cfft(arr)
        expected = np.fft.fft(data)
        np_allclose(result, expected, atol=1e-8, rtol=1e-8)

    @pytest.mark.parametrize("N", [127, 251, 509, 1009])
    def test_large_primes(self, N):
        """Exercises Bluestein's algorithm for large prime lengths."""
        from mdarray import cfft
        from mdarray.array import mdarray
        data = [complex(math.sin(2 * math.pi * i / N), math.cos(2 * math.pi * i / N))
                for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))
        result = cfft(arr)
        expected = np.fft.fft(data)
        np_allclose(result, expected, atol=1e-8, rtol=1e-8)


# --- 3. Parseval's theorem ---

class TestParseval:
    @pytest.mark.parametrize("N", [8, 16, 64, 120])
    def test_parseval(self, N):
        """sum(|x|^2) == sum(|X|^2) / N"""
        from mdarray import cfft
        from mdarray.array import mdarray
        data = [complex(i * 0.1, (i % 5) * 0.2) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))
        result = cfft(arr)

        energy_time = sum(abs(x) ** 2 for x in data)
        energy_freq = sum(abs(complex(x)) ** 2 for x in result.data) / N

        assert abs(energy_time - energy_freq) < 1e-8, (
            f"Parseval failed: time={energy_time}, freq={energy_freq}"
        )


# --- 4. Linearity ---

class TestLinearity:
    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_linearity(self, N):
        """fft(a*x + b*y) == a*fft(x) + b*fft(y)"""
        from mdarray import cfft
        from mdarray.array import mdarray

        a, b = 2.5 + 1j, -1.3 + 0.7j
        x = [complex(i, i * 0.5) for i in range(N)]
        y = [complex(i * 0.3, -i * 0.1) for i in range(N)]

        combined = [a * xi + b * yi for xi, yi in zip(x, y, strict=False)]

        fft_combined = cfft(mdarray(shape=[N], data=list(combined)))
        fft_x = cfft(mdarray(shape=[N], data=list(x)))
        fft_y = cfft(mdarray(shape=[N], data=list(y)))

        for i in range(N):
            expected = a * complex(fft_x.data[i]) + b * complex(fft_y.data[i])
            actual = complex(fft_combined.data[i])
            assert abs(actual - expected) < 1e-8, (
                f"Linearity failed at index {i}: {actual} vs {expected}"
            )


# --- 7. FFT-IFFT roundtrip ---

class TestRoundtrip:
    @pytest.mark.parametrize("N", [2, 4, 8, 16, 32, 64, 128])
    def test_roundtrip_power_of_2(self, N):
        """ifft(fft(x)) == x"""
        from mdarray import cfft, ifft
        from mdarray.array import mdarray

        data = [complex(i, i * 0.5 - 3) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))

        result = ifft(cfft(arr))

        for i in range(N):
            assert abs(complex(result.data[i]) - data[i]) < 1e-10, (
                f"Roundtrip failed at index {i}: got {result.data[i]}, expected {data[i]}"
            )

    @pytest.mark.parametrize("N", [5, 7, 11, 13, 127])
    def test_roundtrip_prime(self, N):
        """ifft(fft(x)) == x for prime lengths (Bluestein path)."""
        from mdarray import cfft, ifft
        from mdarray.array import mdarray

        data = [complex(i * 0.1, -(i * 0.3)) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))

        result = ifft(cfft(arr))

        for i in range(N):
            assert abs(complex(result.data[i]) - data[i]) < 1e-8, (
                f"Roundtrip failed at index {i}: got {result.data[i]}, expected {data[i]}"
            )

    @pytest.mark.parametrize("N", [6, 12, 15, 30])
    def test_roundtrip_composite(self, N):
        from mdarray import cfft, ifft
        from mdarray.array import mdarray

        data = [complex(i * 0.1, -(i * 0.2)) for i in range(N)]
        arr = mdarray(shape=[N], data=list(data))

        result = ifft(cfft(arr))

        for i in range(N):
            assert abs(complex(result.data[i]) - data[i]) < 1e-10


# --- 10. Known analytical transforms ---

class TestAnalytical:
    def test_all_ones(self):
        """fft([1,1,...,1]) == [N, 0, 0, ..., 0]"""
        from mdarray import cfft
        from mdarray.array import mdarray

        N = 8
        data = [complex(1)] * N
        arr = mdarray(shape=[N], data=data)
        result = cfft(arr)

        assert abs(complex(result.data[0]) - N) < 1e-10
        for i in range(1, N):
            assert abs(complex(result.data[i])) < 1e-10

    def test_impulse(self):
        """fft([1,0,...,0]) == [1, 1, ..., 1]"""
        from mdarray import cfft
        from mdarray.array import mdarray

        N = 8
        data = [complex(0)] * N
        data[0] = complex(1)
        arr = mdarray(shape=[N], data=data)
        result = cfft(arr)

        for i in range(N):
            assert abs(complex(result.data[i]) - 1) < 1e-10


# --- 11. Edge cases ---

class TestEdgeCases:
    def test_length_1(self):
        """FFT of length 1 is identity."""
        from mdarray import cfft
        from mdarray.array import mdarray

        arr = mdarray(shape=[1], data=[complex(42, 7)])
        result = cfft(arr)
        assert abs(complex(result.data[0]) - complex(42, 7)) < 1e-10

    def test_length_2(self):
        """Minimal butterfly."""
        from mdarray import cfft
        from mdarray.array import mdarray

        data = [complex(1), complex(2)]
        arr = mdarray(shape=[2], data=list(data))
        result = cfft(arr)
        expected = np.fft.fft(data)
        np_allclose(result, expected)


# --- 8. N-D FFT ---

class TestFFTN:
    def test_2d_small(self):
        from mdarray import fftn
        from mdarray.array import mdarray

        shape = [4, 4]
        data = [complex(i) for i in range(16)]
        arr = mdarray(shape=shape, data=data)
        result = fftn(arr)

        np_data = np.array(data).reshape([4, 4])
        expected = np.fft.fftn(np_data)

        np_allclose(result, expected, atol=1e-8, rtol=1e-8)

    def test_2d_rect(self):
        from mdarray import fftn
        from mdarray.array import mdarray

        shape = [3, 4]
        data = [complex(i, i * 0.5) for i in range(12)]
        arr = mdarray(shape=shape, data=data)
        result = fftn(arr)

        np_data = np.array(data).reshape([4, 3])
        expected = np.fft.fftn(np_data)

        np_allclose(result, expected, atol=1e-8, rtol=1e-8)


# --- 9. Real FFT ---

class TestRFFT:
    def test_rfft_basic(self):
        from mdarray import rfft
        from mdarray.array import mdarray

        N = 8
        data = [float(i) for i in range(N)]
        arr = mdarray(shape=[N], data=data)
        result = rfft(arr)

        expected = np.fft.rfft(data)
        assert len(result.data) == len(expected)
        for i in range(len(expected)):
            assert abs(complex(result.data[i]) - expected[i]) < 1e-10
