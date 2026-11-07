import Foundation
import Testing
@testable import Sashiba

@Suite("Common Vector & SIMD Math Tests")
struct VectorOperationsTests {
    @Test("SIMD8 dotProduct, sumOfSquares, multiply, maxMagnitude, clamp")
    func vectorOpsBasic() {
        let a: [Float] = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10]
        let b: [Float] = [2, 3, -1, 0.5, 2, -1, 0, 1, -2, 0.5]
        let count = a.count

        let dot = a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                VectorOperations.dotProduct(a: aPtr.baseAddress!, b: bPtr.baseAddress!, count: count)
            }
        }
        let sumSq = a.withUnsafeBufferPointer { aPtr in
            VectorOperations.sumOfSquares(ptr: aPtr.baseAddress!, count: count)
        }
        let maxMag = a.withUnsafeBufferPointer { aPtr in
            VectorOperations.maxMagnitude(ptr: aPtr.baseAddress!, count: count)
        }

        var expectedDot: Float = 0
        var expectedSq: Float = 0
        for i in 0..<count {
            expectedDot += a[i] * b[i]
            expectedSq += a[i] * a[i]
        }
        #expect(abs(dot - expectedDot) < 1e-5)
        #expect(abs(sumSq - expectedSq) < 1e-4)
        #expect(abs(maxMag - 10.0) < 1e-6)

        var c = a
        c.withUnsafeMutableBufferPointer { cPtr in
            VectorOperations.clamp(
                src: cPtr.baseAddress!, dst: cPtr.baseAddress!, count: count, minVal: -3.0, maxVal: 4.0
            )
        }
        for v in c {
            #expect(v >= -3.0 && v <= 4.0)
        }
    }

    @Test("SIMDMath exp and sigmoid accuracy")
    func simdMathAccuracy() {
        let xs = SIMD8<Float>(-8.0, -3.5, -1.0, -0.1, 0.0, 0.25, 2.0, 6.0)
        let exps = SIMDMath.exp(xs)
        let sigs = SIMDMath.sigmoid(xs)
        for lane in 0..<8 {
            let x = xs[lane]
            let refExp = expf(x)
            let refSig = 1.0 / (1.0 + expf(-x))
            #expect(abs(exps[lane] - refExp) / max(1e-5, refExp) < 1e-4)
            #expect(abs(sigs[lane] - refSig) < 1e-4)
        }
    }
}
