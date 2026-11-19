import Foundation
import Testing
@testable import Sashiba

@Suite("Step 8: Pure-CPU SIMD8/AVX2 Micro-Kernels & Active-Length Trimming Numerical Parity")
struct Step8CPUSpeedupParityTests {
    @Test("8.1: Register-blocked 4x16 gemmRaw matches scalar reference across odd dimensions and beta modes")
    func gemmRawParityAcrossShapes() {
        var rng = SplitMix64(seed: 20261119)
        let shapes: [(m: Int, n: Int, k: Int, beta: Float)] = [
            (1, 2, 128, 0.0),
            (3, 7, 19, 0.0),
            (4, 16, 64, 0.0),
            (7, 23, 37, 1.0),
            (19, 128, 256, 0.35)
        ]
        for sh in shapes {
            var a = [Float](repeating: 0.0, count: sh.m * sh.k)
            var b = [Float](repeating: 0.0, count: sh.k * sh.n)
            var cInit = [Float](repeating: 0.0, count: sh.m * sh.n)
            for i in 0..<a.count { a[i] = rng.nextSignedUniform(scale: 1.0) }
            for i in 0..<b.count { b[i] = rng.nextSignedUniform(scale: 1.0) }
            for i in 0..<cInit.count { cInit[i] = rng.nextSignedUniform(scale: 1.0) }

            var cFast = cInit
            SWTHawkAutoencoder.gemm(a, b, &cFast, m: sh.m, n: sh.n, k: sh.k, beta: sh.beta)

            var maxDiff: Float = 0.0
            for r in 0..<sh.m {
                for col in 0..<sh.n {
                    var s: Float = 0.0
                    for p in 0..<sh.k {
                        s += a[r * sh.k + p] * b[p * sh.n + col]
                    }
                    let expected = (sh.beta == 0.0) ? s : (sh.beta * cInit[r * sh.n + col] + s)
                    maxDiff = max(maxDiff, abs(cFast[r * sh.n + col] - expected))
                }
            }
            #expect(maxDiff < 1e-4)
        }
    }

    @Test("8.2: Active-length trimming matches full padded sequence length (T=288 & T=512) to < 1e-5")
    func activeLengthTrimmingParity() {
        let weights = SWTHawkWeights.makeRandom(vocabSize: 256, dim: 64, latentDim: 128, seed: 8080)
        let pairs: [(s: [Int], r: [Int])] = [
            ([], []),
            ([5, 12, 19, 23, 41], [7, 12, 19, 88, 91, 102]),
            (Array(repeating: 14, count: 47), Array(repeating: 29, count: 63)),
            ((0..<135).map { 4 + ($0 % 200) }, (0..<142).map { 6 + (($0 * 3) % 200) })
        ]
        for pair in pairs {
            let zTrim = DualStreamRelationalEngine.encodeRelational(
                senderTokenIDs: pair.s, receiverTokenIDs: pair.r, weights: weights
            )
            let s512 = DualStreamRelationalEngine.padOrTruncate(pair.s, to: 512)
            let r512 = DualStreamRelationalEngine.padOrTruncate(pair.r, to: 512)
            let zFull512 = DualStreamRelationalEngine.forwardBatchCPU(
                senderTokens: s512, receiverTokens: r512, batch: 1, time: 512, weights: weights
            ).zRel

            #expect(zTrim.count == zFull512.count)
            var maxDiff: Float = 0.0
            for (a, b) in zip(zTrim, zFull512) {
                maxDiff = max(maxDiff, abs(a - b))
            }
            #expect(maxDiff < 1e-5)
        }
    }

    @Test("8.3: Length-sorted LPT encodeZRelCPUParallel preserves exact index order and matches single-pair encodeRelational")
    func parallelLengthSortedOrderAndParity() {
        let weights = SWTHawkWeights.makeRandom(vocabSize: 192, dim: 64, latentDim: 128, seed: 9090)
        var rng = SplitMix64(seed: 42)
        var senders: [[Int]] = []
        var receivers: [[Int]] = []
        let lengths = [12, 195, 7, 88, 240, 31, 64, 15, 150, 22, 110, 5, 73, 210, 19, 45, 99, 13, 175, 28]
        for len in lengths {
            senders.append((0..<len).map { _ in 4 + Int(rng.next() % 180) })
            let rLen = max(4, len - 3)
            receivers.append((0..<rLen).map { _ in 4 + Int(rng.next() % 180) })
        }

        let parZ = DualStreamRelationalEngine.encodeZRelCPUParallel(
            senders: senders, receivers: receivers, time: 288, weights: weights, chunkSize: 4
        )
        #expect(parZ.count == lengths.count)
        for i in 0..<lengths.count {
            let refZ = DualStreamRelationalEngine.encodeRelational(
                senderTokenIDs: senders[i], receiverTokenIDs: receivers[i], weights: weights
            )
            var maxDiff: Float = 0.0
            for (a, b) in zip(parZ[i], refZ) {
                maxDiff = max(maxDiff, abs(a - b))
            }
            #expect(maxDiff < 1e-5)
        }
    }

    @Test("8.4: KV-cached HawkConditionalDecoder.decodeGreedy agrees 100% with teacher-forced forward argmax")
    func greedyDecoderKVCacheConsistency() {
        var rng = SplitMix64(seed: 7171)
        let dim = 64
        let latentDim = 128
        let vocabSize = 96
        let length = 48
        let batch = 2
        var tokEmb = [Float](repeating: 0.0, count: vocabSize * dim)
        for i in 0..<tokEmb.count { tokEmb[i] = rng.nextSignedUniform(scale: 0.2) }
        let decW = HawkDecoderWeights.makeRandom(dim: dim, latentDim: latentDim, vocabSize: vocabSize, numLayers: 2, rng: &rng)
        var z = [Float](repeating: 0.0, count: batch * latentDim)
        for i in 0..<z.count { z[i] = rng.nextSignedUniform(scale: 0.5) }

        let greedy = HawkConditionalDecoder.decodeGreedy(
            z: z, tokenEmbedding: tokEmb, weights: decW, batch: batch, length: length
        )
        var prevTokens = [Int](repeating: TextVocabulary.bosId, count: batch * length)
        for b in 0..<batch {
            for t in 1..<length {
                prevTokens[b * length + t] = greedy[b][t - 1]
            }
        }
        let tfLogits = HawkConditionalDecoder.forward(
            z: z, prevTokens: prevTokens, tokenEmbedding: tokEmb, weights: decW, batch: batch, time: length
        )
        for b in 0..<batch {
            for t in 0..<length {
                let base = (b * length + t) * vocabSize
                var bestId = 0
                for v in 1..<vocabSize where tfLogits[base + v] > tfLogits[base + bestId] {
                    bestId = v
                }
                #expect(bestId == greedy[b][t])
            }
        }
    }
}
