import Foundation

/// Weights for the Real-Gated Linear Recurrent Unit (`RG-LRU`, Griffin / Hawk):
///   r_t = σ(W_a x_t + b_a),  i_t = σ(W_x x_t + b_x)
///   a = σ(Λ),  a_t = a^(c · r_t)   (c = 8.0)
///   h_t = a_t ⊙ h_{t-1} + √(1 - a_t^2 + ε) ⊙ (i_t ⊙ x_t)
public struct RGLRUWeights: Sendable, Equatable {
    public static let decayConstant: Float = 8.0
    public static let scaleEps: Float = 1e-6

    public let dim: Int
    public let wA: [Float]     // [D, D] recurrence gate projection
    public let bA: [Float]     // [D]
    public let wX: [Float]     // [D, D] input gate projection
    public let bX: [Float]     // [D]
    public let lambda: [Float] // [D] multi-timescale log-decay parameter Λ

    public init(
        dim: Int,
        wA: [Float],
        bA: [Float],
        wX: [Float],
        bX: [Float],
        lambda: [Float]
    ) {
        self.dim = dim
        self.wA = wA
        self.bA = bA
        self.wX = wX
        self.bX = bX
        self.lambda = lambda
    }

    public var parameterCount: Int {
        return wA.count + bA.count + wX.count + bX.count + lambda.count
    }

    /// Multi-timescale initialization of `Λ`:
    /// Half of channels cover short-term local phrases (`a ≈ 0.5 ... 0.9` at `r = 0.5`),
    /// and half cover long-range legal document memory (`a ≈ 0.9 ... 0.999` at `r = 0.5`).
    public static func initialMultiTimescaleLambda(dim: Int) -> [Float] {
        var values = [Float](repeating: 0.0, count: dim)
        let halfC = decayConstant * 0.5 // 4.0
        for i in 0..<dim {
            let frac = dim > 1 ? Float(i) / Float(dim - 1) : 0.5
            let targetA: Float = if frac < 0.5 {
                0.55 + (frac * 2.0) * (0.90 - 0.55)
            } else {
                expf(-1.0 / expf(logf(10.0) + ((frac - 0.5) * 2.0) * (logf(1000.0) - logf(10.0))))
            }
            let baseA = powf(max(1e-4, min(0.9999, targetA)), 1.0 / halfC)
            values[i] = logf(baseA / max(1e-6, 1.0 - baseA))
        }
        return values
    }

    public static func makeRandom(dim: Int, rng: inout SplitMix64) -> RGLRUWeights {
        let bound = sqrtf(6.0 / Float(dim + dim))
        var wA = [Float](repeating: 0.0, count: dim * dim)
        var wX = [Float](repeating: 0.0, count: dim * dim)
        for i in 0..<(dim * dim) {
            wA[i] = rng.nextSignedUniform(scale: bound)
            wX[i] = rng.nextSignedUniform(scale: bound)
        }
        return RGLRUWeights(
            dim: dim,
            wA: wA,
            bA: [Float](repeating: 0.0, count: dim),
            wX: wX,
            bX: [Float](repeating: 0.0, count: dim),
            lambda: initialMultiTimescaleLambda(dim: dim)
        )
    }
}

/// Weights for Griffin Causal Local Sliding-Window Attention (`W = 32`) with Rotary Position Embedding (`RoPE`).
public struct LocalAttentionWeights: Sendable, Equatable {
    public static let defaultWindowSize: Int = 32
    public static let defaultNumHeads: Int = 4

    public let dim: Int
    public let numHeads: Int
    public let windowSize: Int
    public let normGamma: [Float] // [D]
    public let normBeta: [Float]  // [D]
    public let wQ: [Float]        // [D, D]
    public let bQ: [Float]        // [D]
    public let wK: [Float]        // [D, D]
    public let bK: [Float]        // [D]
    public let wV: [Float]        // [D, D]
    public let bV: [Float]        // [D]
    public let wOut: [Float]      // [D, D]
    public let bOut: [Float]      // [D]

    public init(
        dim: Int,
        numHeads: Int = LocalAttentionWeights.defaultNumHeads,
        windowSize: Int = LocalAttentionWeights.defaultWindowSize,
        normGamma: [Float],
        normBeta: [Float],
        wQ: [Float],
        bQ: [Float],
        wK: [Float],
        bK: [Float],
        wV: [Float],
        bV: [Float],
        wOut: [Float],
        bOut: [Float]
    ) {
        self.dim = dim
        self.numHeads = max(1, min(numHeads, dim))
        self.windowSize = max(1, windowSize)
        self.normGamma = normGamma
        self.normBeta = normBeta
        self.wQ = wQ
        self.bQ = bQ
        self.wK = wK
        self.bK = bK
        self.wV = wV
        self.bV = bV
        self.wOut = wOut
        self.bOut = bOut
    }

    public var parameterCount: Int {
        return normGamma.count + normBeta.count + wQ.count + bQ.count + wK.count + bK.count + wV.count + bV.count + wOut.count + bOut.count
    }

    public static func makeRandom(
        dim: Int,
        numHeads: Int = LocalAttentionWeights.defaultNumHeads,
        windowSize: Int = LocalAttentionWeights.defaultWindowSize,
        residualScale: Float = 1.0,
        rng: inout SplitMix64
    ) -> LocalAttentionWeights {
        let dBound = sqrtf(6.0 / Float(dim + dim))
        let outBound = dBound * residualScale * 0.5
        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                a[i] = rng.nextSignedUniform(scale: bound)
            }
            return a
        }
        let heads = (dim % numHeads == 0) ? numHeads : 1
        return LocalAttentionWeights(
            dim: dim,
            numHeads: heads,
            windowSize: windowSize,
            normGamma: [Float](repeating: 1.0, count: dim),
            normBeta: [Float](repeating: 0.0, count: dim),
            wQ: randMat(dim * dim, dBound),
            bQ: [Float](repeating: 0.0, count: dim),
            wK: randMat(dim * dim, dBound),
            bK: [Float](repeating: 0.0, count: dim),
            wV: randMat(dim * dim, dBound),
            bV: [Float](repeating: 0.0, count: dim),
            wOut: randMat(dim * dim, outBound),
            bOut: [Float](repeating: 0.0, count: dim)
        )
    }
}

/// One stacked Griffin (RG-LRU + Local Sliding-Window Attention `W=32` with RoPE + Gated MLP) Temporal Block:
///   1. Pre-LayerNorm -> (SiLU Gate Branch ⊙ RG-LRU Branch) -> Out Projection + Residual
///   2. Pre-LayerNorm -> Causal Local Sliding-Window Attention (`W=32`, RoPE) -> Out Projection + Residual
///   3. Pre-LayerNorm -> 2-layer SiLU MLP + Residual
public struct HawkBlockWeights: Sendable, Equatable {
    public let dim: Int
    public let mlpDim: Int
    public let norm1Gamma: [Float] // [D]
    public let norm1Beta: [Float]  // [D]
    public let wGate: [Float]      // [D, D] SiLU gate branch
    public let bGate: [Float]      // [D]
    public let rglru: RGLRUWeights
    public let wOut: [Float]       // [D, D] recurrent output projection
    public let bOut: [Float]       // [D]
    public let localAttn: LocalAttentionWeights
    public let norm2Gamma: [Float] // [D]
    public let norm2Beta: [Float]  // [D]
    public let wMLPUp: [Float]     // [D, mlpDim]
    public let bMLPUp: [Float]     // [mlpDim]
    public let wMLPDown: [Float]   // [mlpDim, D]
    public let bMLPDown: [Float]   // [D]

    public init(
        dim: Int,
        mlpDim: Int,
        norm1Gamma: [Float],
        norm1Beta: [Float],
        wGate: [Float],
        bGate: [Float],
        rglru: RGLRUWeights,
        wOut: [Float],
        bOut: [Float],
        localAttn: LocalAttentionWeights? = nil,
        norm2Gamma: [Float],
        norm2Beta: [Float],
        wMLPUp: [Float],
        bMLPUp: [Float],
        wMLPDown: [Float],
        bMLPDown: [Float]
    ) {
        self.dim = dim
        self.mlpDim = mlpDim
        self.norm1Gamma = norm1Gamma
        self.norm1Beta = norm1Beta
        self.wGate = wGate
        self.bGate = bGate
        self.rglru = rglru
        self.wOut = wOut
        self.bOut = bOut
        if let la = localAttn {
            self.localAttn = la
        } else {
            var zeroRng = SplitMix64(seed: 1)
            var la = LocalAttentionWeights.makeRandom(dim: dim, residualScale: 0.0, rng: &zeroRng)
            la = LocalAttentionWeights(
                dim: dim,
                numHeads: la.numHeads,
                windowSize: la.windowSize,
                normGamma: [Float](repeating: 1.0, count: dim),
                normBeta: [Float](repeating: 0.0, count: dim),
                wQ: [Float](repeating: 0.0, count: dim * dim),
                bQ: [Float](repeating: 0.0, count: dim),
                wK: [Float](repeating: 0.0, count: dim * dim),
                bK: [Float](repeating: 0.0, count: dim),
                wV: [Float](repeating: 0.0, count: dim * dim),
                bV: [Float](repeating: 0.0, count: dim),
                wOut: [Float](repeating: 0.0, count: dim * dim),
                bOut: [Float](repeating: 0.0, count: dim)
            )
            self.localAttn = la
        }
        self.norm2Gamma = norm2Gamma
        self.norm2Beta = norm2Beta
        self.wMLPUp = wMLPUp
        self.bMLPUp = bMLPUp
        self.wMLPDown = wMLPDown
        self.bMLPDown = bMLPDown
    }

    public var parameterCount: Int {
        return norm1Gamma.count + norm1Beta.count + wGate.count + bGate.count + rglru.parameterCount + wOut.count + bOut.count + localAttn.parameterCount + norm2Gamma.count + norm2Beta.count + wMLPUp.count + bMLPUp.count + wMLPDown.count + bMLPDown.count
    }

    public static func makeRandom(
        dim: Int,
        mlpDim: Int? = nil,
        residualScale: Float = 1.0,
        rng: inout SplitMix64
    ) -> HawkBlockWeights {
        let m = mlpDim ?? (2 * dim)
        let dBound = sqrtf(6.0 / Float(dim + dim))
        let upBound = sqrtf(6.0 / Float(dim + m))
        let downBound = sqrtf(6.0 / Float(m + dim)) * residualScale
        let outBound = dBound * residualScale

        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                a[i] = rng.nextSignedUniform(scale: bound)
            }
            return a
        }

        return HawkBlockWeights(
            dim: dim,
            mlpDim: m,
            norm1Gamma: [Float](repeating: 1.0, count: dim),
            norm1Beta: [Float](repeating: 0.0, count: dim),
            wGate: randMat(dim * dim, dBound),
            bGate: [Float](repeating: 0.0, count: dim),
            rglru: RGLRUWeights.makeRandom(dim: dim, rng: &rng),
            wOut: randMat(dim * dim, outBound),
            bOut: [Float](repeating: 0.0, count: dim),
            localAttn: LocalAttentionWeights.makeRandom(dim: dim, residualScale: residualScale, rng: &rng),
            norm2Gamma: [Float](repeating: 1.0, count: dim),
            norm2Beta: [Float](repeating: 0.0, count: dim),
            wMLPUp: randMat(dim * m, upBound),
            bMLPUp: [Float](repeating: 0.0, count: m),
            wMLPDown: randMat(m * dim, downBound),
            bMLPDown: [Float](repeating: 0.0, count: dim)
        )
    }
}

/// Real-Gated Linear Recurrent Unit (`RG-LRU`) forward scan over `[B, T, D]`.
public enum RGLRULayer {
    /// Numerically stable `log(σ(λ)) = -log(1 + exp(-λ))`.
    @inline(__always)
    public static func logSigmoid(_ lambda: Float) -> Float {
        if lambda >= 0.0 {
            return -log1pf(expf(-lambda))
        } else {
            return lambda - log1pf(expf(lambda))
        }
    }

    /// Forward scan of `RG-LRU` over `[B, T, D]` with optional initial recurrent state `h0` (`[B, D]`).
    public static func forward(
        x: [Float],
        weights: RGLRUWeights,
        h0: [Float]? = nil,
        batch: Int,
        time: Int
    ) -> (h: [Float], hLast: [Float]) {
        let dim = weights.dim
        let rows = batch * time
        let total = rows * dim
        precondition(x.count == total)

        let rPre = SWTHawkAutoencoder.linear(x: x, w: weights.wA, b: weights.bA, rows: rows, inDim: dim, outDim: dim)
        let iPre = SWTHawkAutoencoder.linear(x: x, w: weights.wX, b: weights.bX, rows: rows, inDim: dim, outDim: dim)

        var logDecayBase = [Float](repeating: 0.0, count: dim)
        for c in 0..<dim {
            logDecayBase[c] = RGLRUWeights.decayConstant * logSigmoid(weights.lambda[c])
        }

        let initialH = h0 ?? [Float](repeating: 0.0, count: batch * dim)
        var h = [Float](repeating: 0.0, count: total)
        var hLast = [Float](repeating: 0.0, count: batch * dim)

        for b in 0..<batch {
            let bOff = b * time * dim
            let h0Off = b * dim
            var prevH = Array(initialH[h0Off..<(h0Off + dim)])

            for t in 0..<time {
                let rowOff = bOff + t * dim
                for c in 0..<dim {
                    let idx = rowOff + c
                    let rVal = 1.0 / (1.0 + expf(-rPre[idx]))
                    let iVal = 1.0 / (1.0 + expf(-iPre[idx]))
                    let logAt = rVal * logDecayBase[c]
                    let aVal = expf(logAt)
                    let oneMinusA2 = max(RGLRUWeights.scaleEps, 1.0 - aVal * aVal + RGLRUWeights.scaleEps)
                    let sVal = sqrtf(oneMinusA2)
                    let hCur = aVal * prevH[c] + sVal * (iVal * x[idx])
                    h[idx] = hCur
                    prevH[c] = hCur
                }
            }
            for c in 0..<dim {
                hLast[h0Off + c] = prevH[c]
            }
        }

        return (h, hLast)
    }
}

/// Causal Local Sliding-Window Attention (`W = 32`) with Rotary Position Embedding (`RoPE`).
public enum LocalSlidingWindowAttention {
    /// Precomputes RoPE cosine and sine tables `[time, headDim / 2]`.
    public static func ropeTables(time: Int, headDim: Int) -> (cos: [Float], sin: [Float]) {
        let half = max(1, headDim / 2)
        var cosTable = [Float](repeating: 1.0, count: time * half)
        var sinTable = [Float](repeating: 0.0, count: time * half)
        for t in 0..<time {
            let pos = Float(t)
            let base = t * half
            for i in 0..<half {
                let exponent = -Float(2 * i) / Float(max(2, headDim))
                let invFreq = powf(10000.0, exponent)
                let angle = pos * invFreq
                cosTable[base + i] = cosf(angle)
                sinTable[base + i] = sinf(angle)
            }
        }
        return (cosTable, sinTable)
    }

    /// Applies half-split RoPE in-place to a `[B, T, D]` tensor partitioned into `numHeads` of width `headDim`.
    public static func applyRoPE(
        _ x: [Float],
        cosTable: [Float],
        sinTable: [Float],
        batch: Int,
        time: Int,
        numHeads: Int,
        headDim: Int
    ) -> [Float] {
        let half = headDim / 2
        guard half > 0 else { return x }
        let dim = numHeads * headDim
        var out = x
        for b in 0..<batch {
            let bOff = b * time * dim
            for t in 0..<time {
                let tOff = bOff + t * dim
                let ropeOff = t * half
                for h in 0..<numHeads {
                    let hOff = tOff + h * headDim
                    for i in 0..<half {
                        let c = cosTable[ropeOff + i]
                        let s = sinTable[ropeOff + i]
                        let x1 = x[hOff + i]
                        let x2 = x[hOff + half + i]
                        out[hOff + i] = x1 * c - x2 * s
                        out[hOff + half + i] = x1 * s + x2 * c
                    }
                }
            }
        }
        return out
    }

    /// Causal Local Sliding-Window Attention (`O(T · W)`) over `[B, T, D]`.
    public static func forward(
        x: [Float],
        weights: LocalAttentionWeights,
        batch: Int,
        time: Int
    ) -> [Float] {
        let dim = weights.dim
        let numHeads = weights.numHeads
        let headDim = dim / numHeads
        let windowSize = weights.windowSize
        let rows = batch * time
        let total = rows * dim

        let normed = SWTHawkAutoencoder.layerNorm(
            x: x, gamma: weights.normGamma, beta: weights.normBeta, rows: rows, dim: dim
        )
        let qRaw = SWTHawkAutoencoder.linear(x: normed, w: weights.wQ, b: weights.bQ, rows: rows, inDim: dim, outDim: dim)
        let kRaw = SWTHawkAutoencoder.linear(x: normed, w: weights.wK, b: weights.bK, rows: rows, inDim: dim, outDim: dim)
        let v = SWTHawkAutoencoder.linear(x: normed, w: weights.wV, b: weights.bV, rows: rows, inDim: dim, outDim: dim)

        let (cosT, sinT) = ropeTables(time: time, headDim: headDim)
        let q = applyRoPE(qRaw, cosTable: cosT, sinTable: sinT, batch: batch, time: time, numHeads: numHeads, headDim: headDim)
        let k = applyRoPE(kRaw, cosTable: cosT, sinTable: sinT, batch: batch, time: time, numHeads: numHeads, headDim: headDim)

        let scale = 1.0 / sqrtf(Float(max(1, headDim)))
        var context = [Float](repeating: 0.0, count: total)

        for b in 0..<batch {
            let bOff = b * time * dim
            for t in 0..<time {
                let sStart = max(0, t - windowSize + 1)
                let winLen = t - sStart + 1
                let qRowOff = bOff + t * dim

                for h in 0..<numHeads {
                    let hOff = h * headDim
                    let qHeadOff = qRowOff + hOff
                    var scores = [Float](repeating: 0.0, count: winLen)
                    var maxScore: Float = -.greatestFiniteMagnitude

                    for idx in 0..<winLen {
                        let s = sStart + idx
                        let kHeadOff = bOff + s * dim + hOff
                        var dot: Float = 0.0
                        for d in 0..<headDim {
                            dot += q[qHeadOff + d] * k[kHeadOff + d]
                        }
                        let sc = dot * scale
                        scores[idx] = sc
                        if sc > maxScore { maxScore = sc }
                    }

                    var sumExp: Float = 0.0
                    for idx in 0..<winLen {
                        let e = expf(scores[idx] - maxScore)
                        scores[idx] = e
                        sumExp += e
                    }
                    let invSum = 1.0 / max(sumExp, 1e-12)

                    let outHeadOff = qRowOff + hOff
                    for idx in 0..<winLen {
                        let p = scores[idx] * invSum
                        let s = sStart + idx
                        let vHeadOff = bOff + s * dim + hOff
                        for d in 0..<headDim {
                            context[outHeadOff + d] += p * v[vHeadOff + d]
                        }
                    }
                }
            }
        }

        let attnOut = SWTHawkAutoencoder.linear(x: context, w: weights.wOut, b: weights.bOut, rows: rows, inDim: dim, outDim: dim)
        var out = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            out[i] = x[i] + attnOut[i]
        }
        return out
    }
}

/// Full Griffin Block (`RG-LRU Sub-block -> Local Sliding-Window Attention (W=32, RoPE) Sub-block -> SiLU MLP Sub-block`).
public enum HawkBlock {
    public static func forward(
        x: [Float],
        weights: HawkBlockWeights,
        h0: [Float]? = nil,
        batch: Int,
        time: Int
    ) -> (out: [Float], hLast: [Float]) {
        let dim = weights.dim
        let mlpDim = weights.mlpDim
        let rows = batch * time
        let total = rows * dim

        let normed1 = SWTHawkAutoencoder.layerNorm(
            x: x, gamma: weights.norm1Gamma, beta: weights.norm1Beta, rows: rows, dim: dim
        )
        let gatePre = SWTHawkAutoencoder.linear(
            x: normed1, w: weights.wGate, b: weights.bGate, rows: rows, inDim: dim, outDim: dim
        )
        let gateAct = SWTHawkAutoencoder.silu(gatePre)

        let (h, hLast) = RGLRULayer.forward(
            x: normed1, weights: weights.rglru, h0: h0, batch: batch, time: time
        )

        var gatedH = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            gatedH[i] = gateAct[i] * h[i]
        }

        let recOut = SWTHawkAutoencoder.linear(
            x: gatedH, w: weights.wOut, b: weights.bOut, rows: rows, inDim: dim, outDim: dim
        )
        var xMid1 = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            xMid1[i] = x[i] + recOut[i]
        }

        // Sub-block 2: Causal Local Sliding-Window Attention (W = 32, RoPE)
        let xMid2 = LocalSlidingWindowAttention.forward(
            x: xMid1, weights: weights.localAttn, batch: batch, time: time
        )

        // Sub-block 3: Pre-LayerNorm -> SiLU MLP + Residual
        let normed2 = SWTHawkAutoencoder.layerNorm(
            x: xMid2, gamma: weights.norm2Gamma, beta: weights.norm2Beta, rows: rows, dim: dim
        )
        let mlpPre = SWTHawkAutoencoder.linear(
            x: normed2, w: weights.wMLPUp, b: weights.bMLPUp, rows: rows, inDim: dim, outDim: mlpDim
        )
        let mlpAct = SWTHawkAutoencoder.silu(mlpPre)
        let mlpDown = SWTHawkAutoencoder.linear(
            x: mlpAct, w: weights.wMLPDown, b: weights.bMLPDown, rows: rows, inDim: mlpDim, outDim: dim
        )

        var out = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            out[i] = xMid2[i] + mlpDown[i]
        }
        return (out, hLast)
    }
}
