//! PyO3 bindings for the Rust FFT backend.
//!
//! This module provides Python-callable functions that are isomorphic to the
//! pure-Python FFT implementation. When available, the Python FFT module
//! dispatches to these functions for performance.

// TODO: Implement Rust FFT backend
// The structure mirrors the Python implementation:
// - fft.rs: cfft_internal, factor loop, ping-pong buffer swap
// - butterflies.rs: radix-2 through radix-7 + generated codelets + radixg
// - twiddle.rs: twiddle_table() precomputation
// - stride.rs: lightweight strided view struct
