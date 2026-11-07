import Foundation

/// SIMD vector operations on Float buffers.
public enum VectorOperations {
    /// Dot product (SIMD8).
    @inline(__always)
    public static func dotProduct(
        a: UnsafePointer<Float>,
        b: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        let width = 8
        let limit = count - (count % width)
        var sum: Float = 0.0
        var vecSum = SIMD8<Float>(repeating: 0.0)
        var i = 0

        while i < limit {
            let va = SIMD8<Float>(
                a[i+0], a[i+1], a[i+2], a[i+3],
                a[i+4], a[i+5], a[i+6], a[i+7]
            )
            let vb = SIMD8<Float>(
                b[i+0], b[i+1], b[i+2], b[i+3],
                b[i+4], b[i+5], b[i+6], b[i+7]
            )
            vecSum += va * vb
            i += width
        }
        sum += vecSum.sum()

        while i < count {
            sum += a[i] * b[i]
            i += 1
        }
        return sum
    }

    /// Sum of squares (SIMD8).
    @inline(__always)
    public static func sumOfSquares(
        ptr: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        let width = 8
        let limit = count - (count % width)
        var sum: Float = 0.0
        var vecSum = SIMD8<Float>(repeating: 0.0)
        var i = 0

        while i < limit {
            let v = SIMD8<Float>(
                ptr[i+0], ptr[i+1], ptr[i+2], ptr[i+3],
                ptr[i+4], ptr[i+5], ptr[i+6], ptr[i+7]
            )
            vecSum += v * v
            i += width
        }
        sum += vecSum.sum()

        while i < count {
            let val = ptr[i]
            sum += val * val
            i += 1
        }
        return sum
    }

    /// Element-wise product.
    @inline(__always)
    public static func multiply(
        srcA: UnsafePointer<Float>,
        srcB: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>,
        count: Int
    ) {
        let width = 8
        let limit = count - (count % width)
        var i = 0

        while i < limit {
            let va = SIMD8<Float>(
                srcA[i+0], srcA[i+1], srcA[i+2], srcA[i+3],
                srcA[i+4], srcA[i+5], srcA[i+6], srcA[i+7]
            )
            let vb = SIMD8<Float>(
                srcB[i+0], srcB[i+1], srcB[i+2], srcB[i+3],
                srcB[i+4], srcB[i+5], srcB[i+6], srcB[i+7]
            )
            let vr = va * vb
            dst[i+0] = vr[0]
            dst[i+1] = vr[1]
            dst[i+2] = vr[2]
            dst[i+3] = vr[3]
            dst[i+4] = vr[4]
            dst[i+5] = vr[5]
            dst[i+6] = vr[6]
            dst[i+7] = vr[7]
            i += width
        }

        while i < count {
            dst[i] = srcA[i] * srcB[i]
            i += 1
        }
    }

    /// Maximum absolute value (SIMD8).
    @inline(__always)
    public static func maxMagnitude(
        ptr: UnsafePointer<Float>,
        count: Int
    ) -> Float {
        var maxAbs: Float = 0.0
        let width = 8
        let limit = count - (count % width)
        var i = 0
        var maxVec = SIMD8<Float>(repeating: 0.0)
        let zeroVec = SIMD8<Float>(repeating: 0.0)

        while i < limit {
            let v = SIMD8<Float>(
                ptr[i+0], ptr[i+1], ptr[i+2], ptr[i+3],
                ptr[i+4], ptr[i+5], ptr[i+6], ptr[i+7]
            )
            let absV = v.replacing(with: -v, where: v .< zeroVec)
            maxVec = maxVec.replacing(with: absV, where: maxVec .< absV)
            i += width
        }

        var idx = 0
        while idx < 8 {
            if maxAbs < maxVec[idx] {
                maxAbs = maxVec[idx]
            }
            idx += 1
        }

        while i < count {
            let absVal = abs(ptr[i])
            if maxAbs < absVal {
                maxAbs = absVal
            }
            i += 1
        }
        return maxAbs
    }

    /// Clips values to [minVal, maxVal].
    @inline(__always)
    public static func clamp(
        src: UnsafePointer<Float>,
        dst: UnsafeMutablePointer<Float>,
        count: Int,
        minVal: Float,
        maxVal: Float
    ) {
        let width = 8
        let limit = count - (count % width)
        let minVec = SIMD8<Float>(repeating: minVal)
        let maxVec = SIMD8<Float>(repeating: maxVal)
        var i = 0

        while i < limit {
            let v = SIMD8<Float>(
                src[i+0], src[i+1], src[i+2], src[i+3],
                src[i+4], src[i+5], src[i+6], src[i+7]
            )
            var clamped = v.replacing(with: minVec, where: v .< minVec)
            clamped = clamped.replacing(with: maxVec, where: maxVec .< clamped)

            dst[i+0] = clamped[0]
            dst[i+1] = clamped[1]
            dst[i+2] = clamped[2]
            dst[i+3] = clamped[3]
            dst[i+4] = clamped[4]
            dst[i+5] = clamped[5]
            dst[i+6] = clamped[6]
            dst[i+7] = clamped[7]
            i += width
        }

        while i < count {
            let v = src[i]
            var c = v
            if v < minVal {
                c = minVal
            }
            if maxVal < c {
                c = maxVal
            }
            dst[i] = c
            i += 1
        }
    }
}
