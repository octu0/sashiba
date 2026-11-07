import Foundation
import Testing
import MLX
@testable import Sashiba

@Suite("Step 3: SWT-Hawk Encoder, Time-Frequency Fusion & 1024-D Bottleneck Verification")
struct Step3EncoderBottleneckTests {
    @Test("Step 3.1: Arbitrary sequence lengths T (37, 128, 256) always compress to fixed z: (B, 1024)")
    func encoderFixedLatentDimensionAcrossSequenceLengths() {
        let batch = 2
        let vocabSize = 64
        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocabSize,
            dim: 32,
            latentDim: 1024,
            encoderLayers: 2,
            decoderLayers: 1,
            seed: 707
        )
        let model = SWTHawkAutoencoder(weights: weights)

        for seqLen in [37, 128, 256] {
            var tokens = [Int](repeating: 0, count: batch * seqLen)
            for i in 0..<tokens.count {
                tokens[i] = 4 + (i % (vocabSize - 4))
            }
            let z = model.encode(tokens: tokens, batch: batch, time: seqLen)
            #expect(z.count == batch * 1024)
            let sumSq = z.withUnsafeBufferPointer { ptr in
                VectorOperations.sumOfSquares(ptr: ptr.baseAddress!, count: z.count)
            }
            #expect(sumSq > 0.0)
        }
    }

    @Test("Step 3.2: High-frequency single-token negation/exception spike changes z ∈ R^(B × 1024)")
    func encoderSensitivityToLocalExceptionToken() {
        let vocabSize = 32
        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocabSize,
            dim: 32,
            latentDim: 1024,
            seed: 808
        )
        let model = SWTHawkAutoencoder(weights: weights)
        let seqLen = 64
        let seqA = [Int](repeating: 5, count: seqLen)
        var seqB = seqA
        // Flip a single local token in the middle of the clause
        seqB[31] = 19

        let zA = model.encode(tokens: seqA, batch: 1, time: seqLen)
        let zB = model.encode(tokens: seqB, batch: 1, time: seqLen)

        var diffSq: Float = 0.0
        for d in 0..<1024 {
            let diff = zA[d] - zB[d]
            diffSq += diff * diff
        }
        #expect(sqrtf(diffSq) > 1e-2)
    }

    @Test("Step 3.3: CPU SWTHawkAutoencoder.encode matches MLXSWTHawkAutoencoder.encode")
    func cpuAndMLXEncoderConsistency() {
        let batch = 2
        let time = 16
        let vocabSize = 32
        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocabSize,
            dim: 16,
            latentDim: 1024,
            encoderLayers: 1,
            decoderLayers: 1,
            seed: 909
        )
        let cpuModel = SWTHawkAutoencoder(weights: weights)
        let mlxModel = MLXSWTHawkAutoencoder(weights: weights)

        var tokens = [Int](repeating: 0, count: batch * time)
        for i in 0..<tokens.count {
            tokens[i] = 4 + (i % (vocabSize - 4))
        }

        let cpuZ = cpuModel.encode(tokens: tokens, batch: batch, time: time)
        let tokenIds = MLXArray(tokens.map { Int32($0) }, [batch, time])
        let pe = MLXArray(
            HawkConditionalDecoder.positionalEncoding(time: time, dim: weights.dim),
            [1, time, weights.dim]
        )
        let mlxZTensor = mlxModel.encode(tokenIds, pe: pe)
        eval(mlxZTensor)
        let gpuZ = mlxZTensor.asArray(Float.self)

        #expect(cpuZ.count == gpuZ.count)
        var maxDiff: Float = 0.0
        for i in 0..<cpuZ.count {
            maxDiff = max(maxDiff, abs(cpuZ[i] - gpuZ[i]))
        }
        #expect(maxDiff < 1e-3)
    }
}
