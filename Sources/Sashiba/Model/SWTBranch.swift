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
        guard total > 0 else { return ([], [], [], []) }
        let s = SWTBranchWeights.invSqrt2
        let sVec = SIMD8<Float>(repeating: s)

        var ca1 = [Float](repeating: 0.0, count: total)
        var cd1 = [Float](repeating: 0.0, count: total)
        var ca2 = [Float](repeating: 0.0, count: total)
        var cd2 = [Float](repeating: 0.0, count: total)

        x.withUnsafeBufferPointer { xBuf in
            ca1.withUnsafeMutableBufferPointer { ca1Buf in
                cd1.withUnsafeMutableBufferPointer { cd1Buf in
                    ca2.withUnsafeMutableBufferPointer { ca2Buf in
                        cd2.withUnsafeMutableBufferPointer { cd2Buf in
                            let xp = xBuf.baseAddress!
                            let a1p = ca1Buf.baseAddress!
                            let d1p = cd1Buf.baseAddress!
                            let a2p = ca2Buf.baseAddress!
                            let d2p = cd2Buf.baseAddress!

                            // Level 1 (dilation d = 1)
                            for b in 0..<batch {
                                let bBase = b * time * dim
                                for t in 0..<time {
                                    let prevOff = bBase + max(0, t - 1) * dim
                                    let curOff = bBase + t * dim
                                    var c = 0
                                    while c + 8 <= dim {
                                        let vPrev = SWTHawkAutoencoder.load(xp + prevOff + c)
                                        let vCur = SWTHawkAutoencoder.load(xp + curOff + c)
                                        SWTHawkAutoencoder.store(sVec * (vPrev + vCur), a1p + curOff + c)
                                        SWTHawkAutoencoder.store(sVec * (vCur - vPrev), d1p + curOff + c)
                                        c += 8
                                    }
                                    while c < dim {
                                        let xPrev = xp[prevOff + c]
                                        let xCur = xp[curOff + c]
                                        a1p[curOff + c] = s * (xPrev + xCur)
                                        d1p[curOff + c] = s * (xCur - xPrev)
                                        c += 1
                                    }
                                }
                            }

                            // Level 2 (dilation d = 2) on CA1
                            for b in 0..<batch {
                                let bBase = b * time * dim
                                for t in 0..<time {
                                    let prevOff = bBase + max(0, t - 2) * dim
                                    let curOff = bBase + t * dim
                                    var c = 0
                                    while c + 8 <= dim {
                                        let vPrev = SWTHawkAutoencoder.load(ArrayPointerHelper.const(a1p + prevOff + c))
                                        let vCur = SWTHawkAutoencoder.load(ArrayPointerHelper.const(a1p + curOff + c))
                                        SWTHawkAutoencoder.store(sVec * (vPrev + vCur), a2p + curOff + c)
                                        SWTHawkAutoencoder.store(sVec * (vCur - vPrev), d2p + curOff + c)
                                        c += 8
                                    }
                                    while c < dim {
                                        let aPrev = a1p[prevOff + c]
                                        let aCur = a1p[curOff + c]
                                        a2p[curOff + c] = s * (aPrev + aCur)
                                        d2p[curOff + c] = s * (aCur - aPrev)
                                        c += 1
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return (ca1, cd1, ca2, cd2)
    }

    private enum ArrayPointerHelper {
        @inline(__always)
        static func const(_ p: UnsafeMutablePointer<Float>) -> UnsafePointer<Float> {
            UnsafePointer(p)
        }
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
        let rows = batch * time
        let totalOut = rows * dOut
        guard totalOut > 0 else { return [] }
        let sliceSize = dIn * dOut

        return [Float](unsafeUninitializedCapacity: totalOut) { outBuf, initializedCount in
            let op = outBuf.baseAddress!
            weights.b.withUnsafeBufferPointer { bBuf in
                let bp = bBuf.baseAddress!
                for row in 0..<rows {
                    (op + row * dOut).initialize(from: bp, count: dOut)
                }
            }
            x.withUnsafeBufferPointer { xBuf in
                weights.w.withUnsafeBufferPointer { wBuf in
                    let xp = xBuf.baseAddress!
                    let wp = wBuf.baseAddress!
                    let w0p = wp
                    let w1p = wp + sliceSize
                    let w2p = wp + 2 * sliceSize

                    // Center tap (k = 1) over all B * T rows
                    SWTHawkAutoencoder.gemmRaw(
                        xp, dIn,
                        w1p, dOut,
                        op, dOut,
                        m: rows, n: dOut, k: dIn, beta: 1.0
                    )

                    // Left tap (k = 0, reads t - 1) and right tap (k = 2, reads t + 1)
                    if time > 1 {
                        for b in 0..<batch {
                            let inBOff = b * time * dIn
                            let outBOff = b * time * dOut
                            SWTHawkAutoencoder.gemmRaw(
                                xp + inBOff, dIn,
                                w0p, dOut,
                                op + outBOff + dOut, dOut,
                                m: time - 1, n: dOut, k: dIn, beta: 1.0
                            )
                            SWTHawkAutoencoder.gemmRaw(
                                xp + inBOff + dIn, dIn,
                                w2p, dOut,
                                op + outBOff, dOut,
                                m: time - 1, n: dOut, k: dIn, beta: 1.0
                            )
                        }
                    }
                }
            }
            initializedCount = totalOut
        }
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

        let fDetail = [Float](unsafeUninitializedCapacity: total) { fBuf, initCount in
            let fp = fBuf.baseAddress!
            routerLogits.withUnsafeBufferPointer { rBuf in
                uCD1.withUnsafeBufferPointer { u1Buf in
                    uCD2.withUnsafeBufferPointer { u2Buf in
                        let rp = rBuf.baseAddress!
                        let u1p = u1Buf.baseAddress!
                        let u2p = u2Buf.baseAddress!
                        let oneVec = SIMD8<Float>(repeating: 1.0)
                        for r in 0..<rows {
                            let rLogOff = r * 2 * dim
                            let fOff = r * dim
                            var c = 0
                            while c + 8 <= dim {
                                let l1 = SWTHawkAutoencoder.load(rp + rLogOff + c)
                                let l2 = SWTHawkAutoencoder.load(rp + rLogOff + dim + c)
                                let m = l1.replacing(with: l2, where: l1 .< l2)
                                let e1 = SIMDMath.exp(l1 - m)
                                let e2 = SIMDMath.exp(l2 - m)
                                let invSum = oneVec / (e1 + e2)
                                let v1 = SWTHawkAutoencoder.load(u1p + fOff + c)
                                let v2 = SWTHawkAutoencoder.load(u2p + fOff + c)
                                SWTHawkAutoencoder.store((e1 * invSum) * v1 + (e2 * invSum) * v2, fp + fOff + c)
                                c += 8
                            }
                            while c < dim {
                                let l1 = rp[rLogOff + c]
                                let l2 = rp[rLogOff + dim + c]
                                let m = max(l1, l2)
                                let e1 = expf(l1 - m)
                                let e2 = expf(l2 - m)
                                let invSum = 1.0 / (e1 + e2)
                                fp[fOff + c] = (e1 * invSum) * u1p[fOff + c] + (e2 * invSum) * u2p[fOff + c]
                                c += 1
                            }
                        }
                    }
                }
            }
            initCount = total
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
        guard total > 0 else { return [] }

        let gateCat = [Float](unsafeUninitializedCapacity: rows * 2 * dim) { catBuf, initCount in
            let cp = catBuf.baseAddress!
            fApprox.withUnsafeBufferPointer { aBuf in
                fDetail.withUnsafeBufferPointer { dBuf in
                    let ap = aBuf.baseAddress!
                    let dp = dBuf.baseAddress!
                    for r in 0..<rows {
                        let dst = cp + r * 2 * dim
                        (dst).initialize(from: ap + r * dim, count: dim)
                        (dst + dim).initialize(from: dp + r * dim, count: dim)
                    }
                }
            }
            initCount = rows * 2 * dim
        }

        let gatePre = SWTHawkAutoencoder.linear(
            x: gateCat, w: weights.wGate, b: weights.bGate,
            rows: rows, inDim: 2 * dim, outDim: dim
        )

        return [Float](unsafeUninitializedCapacity: total) { outBuf, initCount in
            let op = outBuf.baseAddress!
            gatePre.withUnsafeBufferPointer { gBuf in
                fApprox.withUnsafeBufferPointer { aBuf in
                    fDetail.withUnsafeBufferPointer { dBuf in
                        let gp = gBuf.baseAddress!
                        let ap = aBuf.baseAddress!
                        let dp = dBuf.baseAddress!
                        let oneVec = SIMD8<Float>(repeating: 1.0)
                        var i = 0
                        while i + 8 <= total {
                            let g = SIMDMath.sigmoid(SWTHawkAutoencoder.load(gp + i))
                            let va = SWTHawkAutoencoder.load(ap + i)
                            let vd = SWTHawkAutoencoder.load(dp + i)
                            SWTHawkAutoencoder.store(g * va + (oneVec - g) * vd, op + i)
                            i += 8
                        }
                        while i < total {
                            let g = 1.0 / (1.0 + expf(-gp[i]))
                            op[i] = g * ap[i] + (1.0 - g) * dp[i]
                            i += 1
                        }
                    }
                }
            }
            initCount = total
        }
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
