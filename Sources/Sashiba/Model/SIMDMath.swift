import Foundation

/// SIMD8 transcendental functions for inference gate computations (avoids scalar `expf` per element).
public enum SIMDMath {
    /// exp(x) via range reduction 2^n · e^r (|r| ≤ ln2/2) and a degree-6 polynomial for e^r.
    /// Relative error is about 2e-7. x is clamped to [-87, 88] to avoid infinities and denormals.
    @inline(__always)
    public static func exp(_ x: SIMD8<Float>) -> SIMD8<Float> {
        let lo = SIMD8<Float>(repeating: -87.0)
        let hi = SIMD8<Float>(repeating: 88.0)
        var v = x.replacing(with: lo, where: x .< lo)
        v = v.replacing(with: hi, where: hi .< v)
        let log2e = SIMD8<Float>(repeating: 1.442695041)
        // Adding 1.5 × 2^23 places round(x) in the low mantissa bits, rounding without leaving SIMD.
        // SIMD8<Int32>(_:rounding:) lowers to a slow per-lane conversion.
        let magic = SIMD8<Float>(repeating: 12582912.0)
        let shifted = v * log2e + magic
        let n = shifted - magic
        // Cody-Waite: subtract ln2 as high and low parts to limit rounding error.
        let r = (v - n * SIMD8<Float>(repeating: 0.693359375)) - n * SIMD8<Float>(repeating: -2.12194440e-4)
        var p = SIMD8<Float>(repeating: 1.0 / 720.0)
        p = p * r + SIMD8<Float>(repeating: 1.0 / 120.0)
        p = p * r + SIMD8<Float>(repeating: 1.0 / 24.0)
        p = p * r + SIMD8<Float>(repeating: 1.0 / 6.0)
        p = p * r + SIMD8<Float>(repeating: 0.5)
        p = p * r + SIMD8<Float>(repeating: 1.0)
        p = p * r + SIMD8<Float>(repeating: 1.0)
        let integer = unsafeBitCast(shifted, to: SIMD8<Int32>.self) &- SIMD8<Int32>(repeating: 0x4B40_0000)
        let bits = (integer &+ SIMD8<Int32>(repeating: 127)) &<< SIMD8<Int32>(repeating: 23)
        return p * unsafeBitCast(bits, to: SIMD8<Float>.self)
    }

    /// 1 / (1 + exp(-x))
    @inline(__always)
    public static func sigmoid(_ x: SIMD8<Float>) -> SIMD8<Float> {
        let one = SIMD8<Float>(repeating: 1.0)
        return one / (one + exp(-x))
    }
}
