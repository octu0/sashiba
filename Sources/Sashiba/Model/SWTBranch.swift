import Foundation

/// Learnable 1D convolution (`kernel_size = 3, padding = 1, stride = 1`) over `[B, T, inDim] -> [B, T, outDim]`.
/// Weight `w` has shape `[3, inDim, outDim]` (taps `{t-1, t, t+1}`), bias `b` has shape `[outDim]`.
public struct Conv1D3Weights: Sendable, Equatable {
    public let inDim: Int
    public let outDim: Int
    public let w: [Float] // [3 * inDim * outDim]
    public let b: [Float] // [outDim]

    public init(inDim: Int? = nil, outDim: Int? = nil, w: [Float], b: [Float]) {
        let inferredOut = outDim ?? b.count
        let inferredIn = inDim ?? (inferredOut > 0 ? w.count / max(1, 3 * inferredOut) : 0)
        self.inDim = inferredIn
        self.outDim = inferredOut
        self.w = w
        self.b = b
    }

    public var parameterCount: Int {
        return w.count + b.count
    }

    public static func makeRandom(dim: Int, rng: inout SplitMix64) -> Conv1D3Weights {
        return makeRandom(inDim: dim, outDim: dim, rng: &rng)
    }

    public static func makeRandom(inDim: Int, outDim: Int, rng: inout SplitMix64) -> Conv1D3Weights {
        let bound = sqrtf(6.0 / Float(3 * inDim + outDim))
        var w = [Float](repeating: 0.0, count: 3 * inDim * outDim)
        for i in 0..<w.count {
            w[i] = rng.nextSignedUniform(scale: bound)
        }
        return Conv1D3Weights(inDim: inDim, outDim: outDim, w: w, b: [Float](repeating: 0.0, count: outDim))
    }
}

/// Weights for Branch A: Energy-Preserving 1D Stationary Wavelet Transform (`db1` / Haar) Frequency Branch.
/// High-frequency subbands `CD1, CD2` concatenate signed difference and rectified magnitude `[CD_j ; |CD_j|]` (`2D -> D`).
public struct SWTBranchWeights: Sendable, Equatable {
    /// Orthonormal Haar (`db1`) scaling coefficient `1 / √2`.
    public static let invSqrt2: Float = 0.7071067811865475

    public let dim: Int
    public let convCD1: Conv1D3Weights // Conv1d(2D, D, kernel_size=3, padding=1) for [CD1; |CD1|]
    public let convCD2: Conv1D3Weights // Conv1d(2D, D, kernel_size=3, padding=1) for [CD2; |CD2|]
    public let convCA2: Conv1D3Weights // Conv1d(D, D, kernel_size=3, padding=1) for Level-2 Approx CA2
    public let wDetailRouter: [Float]  // [2D, 2D] adaptive Softmax routing over (CD1_tilde, CD2_tilde)
    public let bDetailRouter: [Float]  // [2D]
    public let wGate: [Float]          // [2D, D] gate G = σ(Linear([F_approx; F_detail]))
    public let bGate: [Float]          // [D]

    public init(
        dim: Int,
        convCD1: Conv1D3Weights,
        convCD2: Conv1D3Weights,
        convCA2: Conv1D3Weights,
        wDetailRouter: [Float],
        bDetailRouter: [Float],
        wGate: [Float],
        bGate: [Float]
    ) {
        self.dim = dim
        self.convCD1 = convCD1
        self.convCD2 = convCD2
        self.convCA2 = convCA2
        self.wDetailRouter = wDetailRouter
        self.bDetailRouter = bDetailRouter
        self.wGate = wGate
        self.bGate = bGate
    }

    public var parameterCount: Int {
        return convCD1.parameterCount + convCD2.parameterCount + convCA2.parameterCount + wDetailRouter.count + bDetailRouter.count + wGate.count + bGate.count
    }

    public static func makeRandom(dim: Int, rng: inout SplitMix64) -> SWTBranchWeights {
        let routerBound = sqrtf(6.0 / Float(2 * dim + 2 * dim))
        var wRouter = [Float](repeating: 0.0, count: (2 * dim) * (2 * dim))
        for i in 0..<wRouter.count {
            wRouter[i] = rng.nextSignedUniform(scale: routerBound)
        }

        let gateBound = sqrtf(6.0 / Float(2 * dim + dim))
        var wGate = [Float](repeating: 0.0, count: (2 * dim) * dim)
        for i in 0..<wGate.count {
            wGate[i] = rng.nextSignedUniform(scale: gateBound)
        }

        return SWTBranchWeights(
            dim: dim,
            convCD1: Conv1D3Weights.makeRandom(inDim: 2 * dim, outDim: dim, rng: &rng),
            convCD2: Conv1D3Weights.makeRandom(inDim: 2 * dim, outDim: dim, rng: &rng),
            convCA2: Conv1D3Weights.makeRandom(inDim: dim, outDim: dim, rng: &rng),
            wDetailRouter: wRouter,
            bDetailRouter: [Float](repeating: 0.0, count: 2 * dim),
            wGate: wGate,
            bGate: [Float](repeating: 0.0, count: dim)
        )
    }
}

/// Branch A: 1D-SWT (`db1` / Haar) 2-level undecimated wavelet decomposition and energy-preserving frequency aggregation.
public enum SWTFrequencyBranch {
    /// 2-level undecimated 1D Stationary Wavelet Transform (`db1` / Haar) with causal replication padding.
    /// Input `x` has shape `[B, T, D]` and all decomposed subbands `(ca1, cd1, ca2, cd2)` preserve `[B, T, D]`.
    public static func decompose(
        x: [Float],
        batch: Int,
        time: Int,
        dim: Int
    ) -> (ca1: [Float], cd1: [Float], ca2: [Float], cd2: [Float]) {
        let total = batch * time * dim
        precondition(x.count == total)
        let s = SWTBranchWeights.invSqrt2

        var ca1 = [Float](repeating: 0.0, count: total)
        var cd1 = [Float](repeating: 0.0, count: total)
        var ca2 = [Float](repeating: 0.0, count: total)
        var cd2 = [Float](repeating: 0.0, count: total)

        // Level 1 (dilation d = 1): h = [1/√2, 1/√2], g = [-1/√2, 1/√2]
        for b in 0..<batch {
            let bBase = b * time * dim
            for t in 0..<time {
                let tPrev = max(0, t - 1)
                let prevOff = bBase + tPrev * dim
                let curOff = bBase + t * dim
                for c in 0..<dim {
                    let xPrev = x[prevOff + c]
                    let xCur = x[curOff + c]
                    ca1[curOff + c] = s * (xPrev + xCur)
                    cd1[curOff + c] = s * (xCur - xPrev)
                }
            }
        }

        // Level 2 (dilation d = 2) on CA1
        for b in 0..<batch {
            let bBase = b * time * dim
            for t in 0..<time {
                let tPrev = max(0, t - 2)
                let prevOff = bBase + tPrev * dim
                let curOff = bBase + t * dim
                for c in 0..<dim {
                    let aPrev = ca1[prevOff + c]
                    let aCur = ca1[curOff + c]
                    ca2[curOff + c] = s * (aPrev + aCur)
                    cd2[curOff + c] = s * (aCur - aPrev)
                }
            }
        }

        return (ca1, cd1, ca2, cd2)
    }

    /// Exact inverse of 1-level undecimated Haar SWT (`(CA + CD) / √2 == X`), useful for verifying energy conservation.
    public static func reconstructLevel1(ca1: [Float], cd1: [Float]) -> [Float] {
        precondition(ca1.count == cd1.count)
        let s = SWTBranchWeights.invSqrt2
        var out = [Float](repeating: 0.0, count: ca1.count)
        for i in 0..<ca1.count {
            out[i] = (ca1[i] + cd1[i]) * s
        }
        return out
    }

    /// Concatenates signed wavelet detail `cd` and rectified magnitude `|cd|` along the channel axis: `[B, T, D] -> [B, T, 2D]`.
    public static func concatSignedAndMagnitude(_ cd: [Float], rows: Int, dim: Int) -> [Float] {
        var out = [Float](repeating: 0.0, count: rows * 2 * dim)
        for r in 0..<rows {
            let srcOff = r * dim
            let dstOff = r * 2 * dim
            for c in 0..<dim {
                let v = cd[srcOff + c]
                out[dstOff + c] = v
                out[dstOff + dim + c] = abs(v)
            }
        }
        return out
    }

    /// 1D convolution (`kernel_size = 3, padding = 1`) over `[B, T, inDim] -> [B, T, outDim]`.
    public static func conv1d3Forward(
        x: [Float],
        weights: Conv1D3Weights,
        batch: Int,
        time: Int,
        inDim: Int? = nil,
        outDim: Int? = nil
    ) -> [Float] {
        let dIn = inDim ?? weights.inDim
        let dOut = outDim ?? weights.outDim
        let totalOut = batch * time * dOut
        let sliceSize = dIn * dOut
        let w0 = Array(weights.w[0..<sliceSize])
        let w1 = Array(weights.w[sliceSize..<(2 * sliceSize)])
        let w2 = Array(weights.w[(2 * sliceSize)..<(3 * sliceSize)])

        var out = [Float](repeating: 0.0, count: totalOut)
        for row in 0..<(batch * time) {
            let base = row * dOut
            for c in 0..<dOut {
                out[base + c] = weights.b[c]
            }
        }

        // Center tap (k = 1) over all B * T rows
        SWTHawkAutoencoder.gemm(x, w1, &out, m: batch * time, n: dOut, k: dIn, beta: 1.0)

        // Left tap (k = 0, reads t - 1) and right tap (k = 2, reads t + 1)
        for b in 0..<batch {
            let inBOff = b * time * dIn
            let outBOff = b * time * dOut
            if time > 1 {
                x.withUnsafeBufferPointer { xp in
                    out.withUnsafeMutableBufferPointer { op in
                        w0.withUnsafeBufferPointer { w0p in
                            SWTHawkAutoencoder.gemmRaw(
                                xp.baseAddress! + inBOff, dIn,
                                w0p.baseAddress!, dOut,
                                op.baseAddress! + outBOff + dOut, dOut,
                                m: time - 1, n: dOut, k: dIn, beta: 1.0
                            )
                        }
                        w2.withUnsafeBufferPointer { w2p in
                            SWTHawkAutoencoder.gemmRaw(
                                xp.baseAddress! + inBOff + dIn, dIn,
                                w2p.baseAddress!, dOut,
                                op.baseAddress! + outBOff, dOut,
                                m: time - 1, n: dOut, k: dIn, beta: 1.0
                            )
                        }
                    }
                }
            }
        }
        return out
    }

    /// Extracts low-frequency approximation features `F_approx` (`CA2`) and energy-preserving adaptively routed
    /// high-frequency detail features `F_detail` (`[CD1; |CD1|], [CD2; |CD2|]`), each of shape `[B, T, D]`.
    public static func extractSubbandFeatures(
        x: [Float],
        weights: SWTBranchWeights,
        batch: Int,
        time: Int
    ) -> (fApprox: [Float], fDetail: [Float]) {
        let dim = weights.dim
        let rows = batch * time
        let total = rows * dim

        let (_, cd1, ca2, cd2) = decompose(x: x, batch: batch, time: time, dim: dim)

        let cd1In: [Float] = if weights.convCD1.inDim == 2 * dim {
            concatSignedAndMagnitude(cd1, rows: rows, dim: dim)
        } else {
            cd1
        }
        let cd2In: [Float] = if weights.convCD2.inDim == 2 * dim {
            concatSignedAndMagnitude(cd2, rows: rows, dim: dim)
        } else {
            cd2
        }

        let uCD1 = SWTHawkAutoencoder.silu(
            conv1d3Forward(x: cd1In, weights: weights.convCD1, batch: batch, time: time, inDim: weights.convCD1.inDim, outDim: dim)
        )
        let uCD2 = SWTHawkAutoencoder.silu(
            conv1d3Forward(x: cd2In, weights: weights.convCD2, batch: batch, time: time, inDim: weights.convCD2.inDim, outDim: dim)
        )
        let fApprox = conv1d3Forward(x: ca2, weights: weights.convCA2, batch: batch, time: time, inDim: dim, outDim: dim)

        var detailCat = [Float](repeating: 0.0, count: rows * 2 * dim)
        for r in 0..<rows {
            let srcOff = r * dim
            let dstOff = r * 2 * dim
            for c in 0..<dim {
                detailCat[dstOff + c] = uCD1[srcOff + c]
                detailCat[dstOff + dim + c] = uCD2[srcOff + c]
            }
        }

        let routerLogits = SWTHawkAutoencoder.linear(
            x: detailCat, w: weights.wDetailRouter, b: weights.bDetailRouter,
            rows: rows, inDim: 2 * dim, outDim: 2 * dim
        )

        var fDetail = [Float](repeating: 0.0, count: total)
        for r in 0..<rows {
            let rLogOff = r * 2 * dim
            let fOff = r * dim
            for c in 0..<dim {
                let l1 = routerLogits[rLogOff + c]
                let l2 = routerLogits[rLogOff + dim + c]
                let m = max(l1, l2)
                let e1 = expf(l1 - m)
                let e2 = expf(l2 - m)
                let invSum = 1.0 / (e1 + e2)
                let a1 = e1 * invSum
                let a2 = e2 * invSum
                fDetail[fOff + c] = a1 * uCD1[fOff + c] + a2 * uCD2[fOff + c]
            }
        }

        return (fApprox, fDetail)
    }

    /// Combines low-frequency `fApprox` (`[B, T, D]`) and high-frequency `fDetail` (`[B, T, D]`)
    /// via the learnable gate `G = σ(Linear([F_approx; F_detail]))`:
    /// `F_SWT = G ⊙ F_approx + (1 - G) ⊙ F_detail`.
    public static func combineSubbandFeatures(
        fApprox: [Float],
        fDetail: [Float],
        weights: SWTBranchWeights,
        batch: Int,
        time: Int
    ) -> [Float] {
        let dim = weights.dim
        let rows = batch * time
        let total = rows * dim
        precondition(fApprox.count == total && fDetail.count == total)

        var gateCat = [Float](repeating: 0.0, count: rows * 2 * dim)
        for r in 0..<rows {
            let srcOff = r * dim
            let dstOff = r * 2 * dim
            for c in 0..<dim {
                gateCat[dstOff + c] = fApprox[srcOff + c]
                gateCat[dstOff + dim + c] = fDetail[srcOff + c]
            }
        }

        let gatePre = SWTHawkAutoencoder.linear(
            x: gateCat, w: weights.wGate, b: weights.bGate,
            rows: rows, inDim: 2 * dim, outDim: dim
        )

        var fSWT = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            let g = 1.0 / (1.0 + expf(-gatePre[i]))
            fSWT[i] = g * fApprox[i] + (1.0 - g) * fDetail[i]
        }
        return fSWT
    }

    /// Forward pass of Branch A (`1D-SWT Frequency Branch`): `[B, T, D] -> [B, T, D]`.
    public static func forward(
        x: [Float],
        weights: SWTBranchWeights,
        batch: Int,
        time: Int
    ) -> [Float] {
        let (fApprox, fDetail) = extractSubbandFeatures(x: x, weights: weights, batch: batch, time: time)
        return combineSubbandFeatures(fApprox: fApprox, fDetail: fDetail, weights: weights, batch: batch, time: time)
    }
}
