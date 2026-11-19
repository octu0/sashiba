import Foundation

/// SIMD vector operations on Float buffers (AVX2/FMA unaligned `SIMD8<Float>`).
public enum VectorOperations {
    @inline(__always)
    private static func load8(_ p: UnsafePointer<Float>) -> SIMD8<Float> {
        UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    private static func store8(_ v: SIMD8<Float>, _ p: UnsafeMutablePointer<Float>) {
        UnsafeMutableRawPointer(p).storeBytes(of: v, as: SIMD8<Float>.self)
    }

    /// Dot product (`SIMD8<Float>`, 2x unrolled).
    @inline(__always)
    public static func dotProduct(
        a: UnsafePointer<Float>,
        b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var acc0 = SIMD8<Float>(repeating: 0.0)
        var acc1 = SIMD8<Float>(repeating: 0.0)
        var i = 0
        while i + 16 <= count {
            acc0.addProduct(load8(a + i), load8(b + i))
            acc1.addProduct(load8(a + i + 8), load8(b + i + 8))
            i += 16
        }
        while i + 8 <= count {
            acc0.addProduct(load8(a + i), load8(b + i))
            i += 8
        }
        var sum = (acc0 + acc1).sum()
        while i < count {
            sum += a[i] * b[i]
            i += 1
        }
        return sum
    }

    /// Sum of squares (`SIMD8<Float>`, 2x unrolled).
    @inline(__always)
    public static func sumOfSquares(
        ptr: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var acc0 = SIMD8<Float>(repeating: 0.0)
        var acc1 = SIMD8<Float>(repeating: 0.0)
        var i = 0
        while i + 16 <= count {
            let v0 = load8(ptr + i)
            let v1 = load8(ptr + i + 8)
            acc0.addProduct(v0, v0)
            acc1.addProduct(v1, v1)
            i += 16
        }
        while i + 8 <= count {
            let v0 = load8(ptr + i)
            acc0.addProduct(v0, v0)
            i += 8
        }
        var sum = (acc0 + acc1).sum()
        while i < count {
            let val = ptr[i]
            sum += val * val
            i += 1
        }
        return sum
    }

    /// Element-wise product (`SIMD8<Float>`).
    @inline(__always)
    public static func multiply(
        srcA: UnsafePointer<Float>,
        srcB: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        var i = 0
        while i + 8 <= count {
            store8(load8(srcA + i) * load8(srcB + i), dst + i)
            i += 8
        }
        while i < count {
            dst[i] = srcA[i] * srcB[i]
            i += 1
        }
    }

    /// Maximum absolute value (`SIMD8<Float>`).
    @inline(__always)
    public static func maxMagnitude(
        ptr: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var maxAbs: Float = 0.0
        var i = 0
        var maxVec = SIMD8<Float>(repeating: 0.0)
        let zeroVec = SIMD8<Float>(repeating: 0.0)
        while i + 8 <= count {
            let v = load8(ptr + i)
            let absV = v.replacing(with: -v, where: v .< zeroVec)
            maxVec = maxVec.replacing(with: absV, where: maxVec .< absV)
            i += 8
        }
        for lane in 0..<8 {
            if maxAbs < maxVec[lane] { maxAbs = maxVec[lane] }
        }
        while i < count {
            let absVal = abs(ptr[i])
            if maxAbs < absVal { maxAbs = absVal }
            i += 1
        }
        return maxAbs
    }

    /// Clips values to `[minVal, maxVal]` (`SIMD8<Float>`).
    @inline(__always)
    public static func clamp(
        src: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>,
        count: Int,
        minVal: Float,
        maxVal: Float
    ) {
        let minVec = SIMD8<Float>(repeating: minVal)
        let maxVec = SIMD8<Float>(repeating: maxVal)
        var i = 0
        while i + 8 <= count {
            let v = load8(src + i)
            var clamped = v.replacing(with: minVec, where: v .< minVec)
            clamped = clamped.replacing(with: maxVec, where: maxVec .< clamped)
            store8(clamped, dst + i)
            i += 8
        }
        while i < count {
            dst[i] = max(minVal, min(maxVal, src[i]))
            i += 1
        }
    }
}
