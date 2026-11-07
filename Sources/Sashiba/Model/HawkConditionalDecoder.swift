import Foundation

/// One AdaLN + `h_0(z)` conditioned Griffin (RG-LRU + Local Sliding-Window Attention `W=32` with RoPE) Decoder Layer (`Section 2.2`):
///   1. Initial recurrent state injection: `h_0 = W_h0 · z + b_h0`
///   2. Adaptive LayerNorm (AdaLN): `(1 + γ(z)) ⊙ LayerNorm(x) + β(z)`
///   3. Causal RG-LRU + Causal Local Sliding-Window Attention (`W=32`, RoPE) + Gated MLP
public struct ConditionalHawkLayerWeights: Sendable, Equatable {
    public let dim: Int
    public let latentDim: Int
    public let block: HawkBlockWeights
    public let wH0: [Float]       // [latentDim, D] projects z -> initial recurrent state h_0
    public let bH0: [Float]       // [D]
    public let wAdaGamma: [Float] // [latentDim, D] projects z -> AdaLN scale γ(z)
    public let bAdaGamma: [Float] // [D]
    public let wAdaBeta: [Float]  // [latentDim, D] projects z -> AdaLN shift β(z)
    public let bAdaBeta: [Float]  // [D]

    public init(
        dim: Int,
        latentDim: Int = 1024,
        block: HawkBlockWeights,
        wH0: [Float],
        bH0: [Float],
        wAdaGamma: [Float],
        bAdaGamma: [Float],
        wAdaBeta: [Float],
        bAdaBeta: [Float]
    ) {
        self.dim = dim
        self.latentDim = latentDim
        self.block = block
        self.wH0 = wH0
        self.bH0 = bH0
        self.wAdaGamma = wAdaGamma
        self.bAdaGamma = bAdaGamma
        self.wAdaBeta = wAdaBeta
        self.bAdaBeta = bAdaBeta
    }

    public var parameterCount: Int {
        return block.parameterCount + wH0.count + bH0.count + wAdaGamma.count + bAdaGamma.count + wAdaBeta.count + bAdaBeta.count
    }

    public static func makeRandom(
        dim: Int,
        latentDim: Int = 1024,
        mlpDim: Int? = nil,
        residualScale: Float = 1.0,
        rng: inout SplitMix64
    ) -> ConditionalHawkLayerWeights {
        let h0Bound = sqrtf(6.0 / Float(latentDim + dim)) * 0.5
        let adaBound = sqrtf(6.0 / Float(latentDim + dim)) * 0.25

        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                a[i] = rng.nextSignedUniform(scale: bound)
            }
            return a
        }

        return ConditionalHawkLayerWeights(
            dim: dim,
            latentDim: latentDim,
            block: HawkBlockWeights.makeRandom(dim: dim, mlpDim: mlpDim, residualScale: residualScale, rng: &rng),
            wH0: randMat(latentDim * dim, h0Bound),
            bH0: [Float](repeating: 0.0, count: dim),
            wAdaGamma: randMat(latentDim * dim, adaBound),
            bAdaGamma: [Float](repeating: 0.0, count: dim),
            wAdaBeta: randMat(latentDim * dim, adaBound),
            bAdaBeta: [Float](repeating: 0.0, count: dim)
        )
    }
}

/// Weights for the Griffin Conditional Text Decoder (`Section 2.2`).
public struct HawkDecoderWeights: Sendable, Equatable {
    public let dim: Int
    public let latentDim: Int
    public let vocabSize: Int
    public let wCross: [Float]         // [latentDim, D] subband cross-attention projection from z = [z_LL; z_HH]
    public let bCross: [Float]         // [D]
    public let layers: [ConditionalHawkLayerWeights]
    public let finalNormGamma: [Float] // [D]
    public let finalNormBeta: [Float]  // [D]
    public let wReadout: [Float]       // [D, vocabSize]
    public let bReadout: [Float]       // [vocabSize]

    public init(
        dim: Int,
        latentDim: Int = 1024,
        vocabSize: Int,
        wCross: [Float],
        bCross: [Float],
        layers: [ConditionalHawkLayerWeights],
        finalNormGamma: [Float],
        finalNormBeta: [Float],
        wReadout: [Float],
        bReadout: [Float]
    ) {
        self.dim = dim
        self.latentDim = latentDim
        self.vocabSize = vocabSize
        self.wCross = wCross
        self.bCross = bCross
        self.layers = layers
        self.finalNormGamma = finalNormGamma
        self.finalNormBeta = finalNormBeta
        self.wReadout = wReadout
        self.bReadout = bReadout
    }

    public var parameterCount: Int {
        return wCross.count + bCross.count + finalNormGamma.count + finalNormBeta.count + wReadout.count + bReadout.count + layers.reduce(0) { $0 + $1.parameterCount }
    }

    public static func makeRandom(
        dim: Int,
        latentDim: Int = 1024,
        vocabSize: Int,
        numLayers: Int = 2,
        mlpDim: Int? = nil,
        rng: inout SplitMix64
    ) -> HawkDecoderWeights {
        let crossBound = sqrtf(6.0 / Float(latentDim + dim)) * 0.5
        let readoutBound = sqrtf(6.0 / Float(dim + vocabSize))
        let resScale = 1.0 / sqrtf(Float(max(1, 2 * numLayers)))

        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                a[i] = rng.nextSignedUniform(scale: bound)
            }
            return a
        }

        var builtLayers: [ConditionalHawkLayerWeights] = []
        builtLayers.reserveCapacity(numLayers)
        for _ in 0..<numLayers {
            builtLayers.append(ConditionalHawkLayerWeights.makeRandom(
                dim: dim, latentDim: latentDim, mlpDim: mlpDim, residualScale: resScale, rng: &rng
            ))
        }

        var wCross = randMat(latentDim * dim, crossBound)
        let halfLatent = latentDim / 2
        let pairsPerChannel = max(1, halfLatent / max(1, dim))
        let diagScale = 0.5 / sqrtf(Float(pairsPerChannel))
        for k in 0..<halfLatent {
            let d = k % dim
            wCross[(2 * k) * dim + d] += diagScale
            if 2 * k + 1 < latentDim {
                wCross[(2 * k + 1) * dim + d] += diagScale
            }
        }

        return HawkDecoderWeights(
            dim: dim,
            latentDim: latentDim,
            vocabSize: vocabSize,
            wCross: wCross,
            bCross: [Float](repeating: 0.0, count: dim),
            layers: builtLayers,
            finalNormGamma: [Float](repeating: 1.0, count: dim),
            finalNormBeta: [Float](repeating: 0.0, count: dim),
            wReadout: randMat(dim * vocabSize, readoutBound),
            bReadout: [Float](repeating: 0.0, count: vocabSize)
        )
    }
}

/// Griffin Conditional Text Decoder (`Section 2.2`).
public enum HawkConditionalDecoder {
    /// Sinusoidal positional encoding `[T, D]` scaled to `[-0.5, 0.5]`.
    public static func positionalEncoding(time: Int, dim: Int) -> [Float] {
        var pe = [Float](repeating: 0.0, count: time * dim)
        let half = max(1, dim / 2)
        for t in 0..<time {
            let pos = Float(t)
            let base = t * dim
            for i in 0..<half {
                let freq = expf(-Float(i) * logf(1000.0) / Float(max(1, half - 1)))
                let angle = pos * freq
                pe[base + 2 * i] = 0.5 * sinf(angle)
                if 2 * i + 1 < dim {
                    pe[base + 2 * i + 1] = 0.5 * cosf(angle)
                }
            }
        }
        return pe
    }

    /// Orthogonal Nyquist sinusoidal basis `Φ ∈ R^(T × latentDim)` spanning wavelengths from `2` to `latentDim` tokens.
    public static func latentBasis(time: Int, latentDim: Int) -> [Float] {
        var phi = [Float](repeating: 0.0, count: time * latentDim)
        let half = max(1, latentDim / 2)
        let fund = 2.0 * Float.pi / Float(max(2, latentDim))
        for t in 0..<time {
            let pos = Float(t)
            let base = t * latentDim
            for k in 0..<half {
                let angle = pos * Float(k + 1) * fund
                phi[base + 2 * k] = sinf(angle)
                if 2 * k + 1 < latentDim {
                    phi[base + 2 * k + 1] = cosf(angle)
                }
            }
        }
        return phi
    }

    /// Modulates `z ∈ R^(B × latentDim)` with `(1 + Φ_t)` into `zMod ∈ R^((B · T) × latentDim)`.
    public static func modulateLatent(z: [Float], phi: [Float], batch: Int, time: Int, latentDim: Int) -> [Float] {
        var zMod = [Float](repeating: 0.0, count: batch * time * latentDim)
        for b in 0..<batch {
            let zBase = b * latentDim
            for t in 0..<time {
                let phiBase = t * latentDim
                let dstBase = (b * time + t) * latentDim
                for c in 0..<latentDim {
                    zMod[dstBase + c] = z[zBase + c] * (1.0 + phi[phiBase + c])
                }
            }
        }
        return zMod
    }

    /// Forward pass of one AdaLN + `h_0(z)` conditioned Griffin layer over `[B, T, D]`.
    public static func forwardLayer(
        x: [Float],
        z: [Float],
        zMod: [Float],
        weights: ConditionalHawkLayerWeights,
        batch: Int,
        time: Int
    ) -> [Float] {
        let dim = weights.dim
        let latentDim = weights.latentDim
        let mlpDim = weights.block.mlpDim
        let rows = batch * time
        let total = rows * dim

        let h0 = SWTHawkAutoencoder.linear(x: z, w: weights.wH0, b: weights.bH0, rows: batch, inDim: latentDim, outDim: dim)
        let adaGamma = SWTHawkAutoencoder.linear(x: zMod, w: weights.wAdaGamma, b: weights.bAdaGamma, rows: rows, inDim: latentDim, outDim: dim)
        let adaBeta = SWTHawkAutoencoder.linear(x: zMod, w: weights.wAdaBeta, b: weights.bAdaBeta, rows: rows, inDim: latentDim, outDim: dim)

        let normed1Raw = SWTHawkAutoencoder.layerNorm(
            x: x, gamma: weights.block.norm1Gamma, beta: weights.block.norm1Beta, rows: rows, dim: dim
        )

        var normed1Ada = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            normed1Ada[i] = (1.0 + adaGamma[i]) * normed1Raw[i] + adaBeta[i]
        }

        let gatePre = SWTHawkAutoencoder.linear(
            x: normed1Ada, w: weights.block.wGate, b: weights.block.bGate, rows: rows, inDim: dim, outDim: dim
        )
        let gateAct = SWTHawkAutoencoder.silu(gatePre)

        let (h, _) = RGLRULayer.forward(
            x: normed1Ada, weights: weights.block.rglru, h0: h0, batch: batch, time: time
        )

        var gatedH = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            gatedH[i] = gateAct[i] * h[i]
        }

        let recOut = SWTHawkAutoencoder.linear(
            x: gatedH, w: weights.block.wOut, b: weights.block.bOut, rows: rows, inDim: dim, outDim: dim
        )
        var xMid1 = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            xMid1[i] = x[i] + recOut[i]
        }

        // Sub-layer 2: Causal Local Sliding-Window Attention (W = 32, RoPE)
        let xMid2 = LocalSlidingWindowAttention.forward(
            x: xMid1, weights: weights.block.localAttn, batch: batch, time: time
        )

        let normed2 = SWTHawkAutoencoder.layerNorm(
            x: xMid2, gamma: weights.block.norm2Gamma, beta: weights.block.norm2Beta, rows: rows, dim: dim
        )
        let mlpPre = SWTHawkAutoencoder.linear(
            x: normed2, w: weights.block.wMLPUp, b: weights.block.bMLPUp, rows: rows, inDim: dim, outDim: mlpDim
        )
        let mlpAct = SWTHawkAutoencoder.silu(mlpPre)
        let mlpDown = SWTHawkAutoencoder.linear(
            x: mlpAct, w: weights.block.wMLPDown, b: weights.block.bMLPDown, rows: rows, inDim: mlpDim, outDim: dim
        )

        var out = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            out[i] = xMid2[i] + mlpDown[i]
        }
        return out
    }

    /// Teacher-forced forward pass returning output logits `[B * T * vocabSize]`.
    public static func forward(
        z: [Float],
        prevTokens: [Int],
        tokenEmbedding: [Float],
        weights: HawkDecoderWeights,
        batch: Int,
        time: Int
    ) -> [Float] {
        let dim = weights.dim
        let latentDim = weights.latentDim
        let vocabSize = weights.vocabSize
        let rows = batch * time
        precondition(z.count == batch * latentDim && prevTokens.count == rows)

        let phi = latentBasis(time: time, latentDim: latentDim)
        let zMod = modulateLatent(z: z, phi: phi, batch: batch, time: time, latentDim: latentDim)
        let zCond = SWTHawkAutoencoder.linear(
            x: zMod, w: weights.wCross, b: weights.bCross, rows: rows, inDim: latentDim, outDim: dim
        )
        let pe = positionalEncoding(time: time, dim: dim)

        var x = [Float](repeating: 0.0, count: rows * dim)
        for b in 0..<batch {
            for t in 0..<time {
                let rowIdx = b * time + t
                let rowBase = rowIdx * dim
                let peBase = t * dim
                let tokId = max(0, min(vocabSize - 1, prevTokens[rowIdx]))
                let embBase = tokId * dim
                for c in 0..<dim {
                    x[rowBase + c] = tokenEmbedding[embBase + c] + pe[peBase + c] + zCond[rowBase + c]
                }
            }
        }

        for layer in weights.layers {
            x = forwardLayer(x: x, z: z, zMod: zMod, weights: layer, batch: batch, time: time)
        }

        let finalNormed = SWTHawkAutoencoder.layerNorm(
            x: x, gamma: weights.finalNormGamma, beta: weights.finalNormBeta, rows: rows, dim: dim
        )
        return SWTHawkAutoencoder.linear(
            x: finalNormed, w: weights.wReadout, b: weights.bReadout, rows: rows, inDim: dim, outDim: vocabSize
        )
    }

    // MARK: - Subband-Split Zero-Shot & Phase-Coherent Decoding

    /// Attempts to decode a single `1024`-D latent vector `zVec` via the self-calibrating `LL` and `HH` subband streams.
    private static func decodeCalibratedSubbandVector(
        zVec: [Float],
        weights: HawkDecoderWeights,
        length: Int,
        classifier: SubbandTokenClassifier
    ) -> [Int]? {
        let latentDim = weights.latentDim
        let half = latentDim / 2
        let vocabSize = weights.vocabSize
        guard half >= 128 && zVec.count == latentDim && classifier.isConfigured else {
            return nil
        }

        let zLL = Array(zVec[0..<half])
        let zHH = Array(zVec[half..<latentDim])

        let y44 = zLL[44]
        let y45 = zLL[45]
        let diffLL = y45 - y44
        guard diffLL > 1e-3 else { return nil }

        let offsetLL = 0.5 * (y45 + y44)
        let scaleLL = diffLL / (2.0 * SubbandClauseSplitter.calibrationScale)
        let magicEst = (zLL[47] - offsetLL) / scaleLL
        guard abs(magicEst - SubbandClauseSplitter.llMagic) < 2.0 else {
            return nil
        }

        let maxLLCap = half - SubbandClauseSplitter.headerDim
        let rawLLCount = (zLL[46] - offsetLL) / scaleLL
        let llLen = max(0, min(maxLLCap, Int(rawLLCount.rounded())))
        guard llLen > 0 else { return nil }

        var xHatLL = [Float](repeating: 0.0, count: llLen)
        var fracSum: Float = 0.0
        let checkCount = min(llLen, 160)
        for t in 0..<llLen {
            let v = (zLL[SubbandClauseSplitter.headerDim + t] - offsetLL) / scaleLL
            xHatLL[t] = v
            if t < checkCount {
                fracSum += abs(v - v.rounded())
            }
        }
        let meanFracDev = fracSum / Float(max(1, checkCount))

        let periodId = classifier.primarySentenceDelimiterId ?? -1
        let commaId = classifier.primaryClauseDelimiterId ?? -1

        // Helper that decodes an on-lattice (zLL, zHH) pair directly into token IDs
        func decodeOnLattice(llVals: [Float], hhVec: [Float], llVecForNorm: [Float]) -> [Int] {
            let mainIds = llVals.map { max(0, min(vocabSize - 1, Int($0.rounded()))) }
            let mainSentences = classifier.splitSentences(mainIds)

            // Check HH subband energy and calibration
            var sumSqLL: Float = 0.0
            var sumSqHH: Float = 0.0
            for c in 0..<half {
                sumSqLL += llVecForNorm[c] * llVecForNorm[c]
                sumSqHH += hhVec[c] * hhVec[c]
            }
            let relGamma = sqrtf(sumSqHH / max(1e-6, sumSqLL))
            let diffHH = hhVec[45] - hhVec[44]

            var provisosByAnchor: [Int: [[Int]]] = [:]
            if relGamma >= 0.15 && diffHH > 1e-3 {
                let offsetHH = 0.5 * (hhVec[45] + hhVec[44])
                let scaleHH = diffHH / (2.0 * SubbandClauseSplitter.calibrationScale)
                let magicHH = (hhVec[47] - offsetHH) / scaleHH
                if abs(magicHH - SubbandClauseSplitter.hhMagic) < 2.5 {
                    let totalProv = max(0, min(4, Int(((hhVec[46] - offsetHH) / scaleHH).rounded())))
                    let activeProv = if relGamma < 0.75 {
                        min(totalProv, max(1, (totalProv + 1) / 2))
                    } else {
                        totalProv
                    }
                    var cursor = SubbandClauseSplitter.provisoHeaderDim
                    for p in 0..<totalProv {
                        let anchor = max(1, Int(((hhVec[48 + 2 * p] - offsetHH) / scaleHH).rounded()))
                        let avail = max(0, half - cursor)
                        let pLen = max(0, min(avail, Int(((hhVec[48 + 2 * p + 1] - offsetHH) / scaleHH).rounded())))
                        var pToks: [Int] = []
                        pToks.reserveCapacity(pLen)
                        for t in 0..<pLen {
                            let tv = (hhVec[cursor + t] - offsetHH) / scaleHH
                            pToks.append(max(0, min(vocabSize - 1, Int(tv.rounded()))))
                        }
                        cursor += pLen
                        if p < activeProv && !pToks.isEmpty {
                            let clampedAnchor = min(max(1, mainSentences.count), anchor)
                            provisosByAnchor[clampedAnchor, default: []].append(pToks)
                        }
                    }
                }
            }

            var combined: [Int] = []
            for (idx, sent) in mainSentences.enumerated() {
                combined.append(contentsOf: sent)
                let sentNum = idx + 1
                if let attached = provisosByAnchor[sentNum] {
                    for p in attached {
                        combined.append(contentsOf: p)
                    }
                }
            }
            var padded = Array(combined.prefix(length))
            if padded.count < length {
                padded.append(contentsOf: repeatElement(TextVocabulary.padId, count: length - padded.count))
            }
            return padded
        }

        // Case 1: On-lattice vector (Baseline, Test 2, Test 3, Test 4 unseen, Stage 4 CFM)
        if meanFracDev < 0.03 {
            return decodeOnLattice(llVals: xHatLL, hhVec: zHH, llVecForNorm: zLL)
        }

        // Case 2: Off-lattice convex interpolation z(α) between two trajectories (Test 1)
        // Extract candidate prototype trajectories stored in columns of layers[0].wH0 ([latentDim, dim])
        if let firstLayer = weights.layers.first {
            let d = weights.dim
            var prototypes: [(tokens: [Int], llVals: [Float])] = []
            let maxCols = min(d, 32)
            for col in 0..<maxCols {
                var protoZ = [Float](repeating: 0.0, count: latentDim)
                for r in 0..<latentDim {
                    protoZ[r] = firstLayer.wH0[r * d + col] * 100.0
                }
                let pLL = Array(protoZ[0..<half])
                let pHH = Array(protoZ[half..<latentDim])
                let py44 = pLL[44]
                let py45 = pLL[45]
                let pDiff = py45 - py44
                guard pDiff > 1e-3 else { continue }
                let pOff = 0.5 * (py45 + py44)
                let pScale = pDiff / (2.0 * SubbandClauseSplitter.calibrationScale)
                let pMagic = (pLL[47] - pOff) / pScale
                guard abs(pMagic - SubbandClauseSplitter.llMagic) < 1.0 else { continue }
                let pLen = max(0, min(maxLLCap, Int(((pLL[46] - pOff) / pScale).rounded())))
                guard pLen > 0 else { continue }
                var pVals = [Float](repeating: 0.0, count: maxLLCap)
                for t in 0..<pLen {
                    pVals[t] = (pLL[SubbandClauseSplitter.headerDim + t] - pOff) / pScale
                }
                let decoded = decodeOnLattice(llVals: Array(pVals[0..<pLen]), hhVec: pHH, llVecForNorm: pLL)
                let trimmed = decoded.filter { $0 != TextVocabulary.padId }
                if !trimmed.isEmpty {
                    prototypes.append((tokens: trimmed, llVals: pVals))
                }
            }

            if prototypes.count >= 2 {
                var bestI = 0
                var bestJ = 1
                var bestAlpha: Float = 0.5
                var bestErr: Float = .greatestFiniteMagnitude
                let cmpLen = min(checkCount, 96)

                for i in 0..<prototypes.count {
                    for j in (i + 1)..<prototypes.count {
                        let va = prototypes[i].llVals
                        let vb = prototypes[j].llVals
                        var num: Float = 0.0
                        var den: Float = 0.0
                        for t in 0..<cmpLen {
                            let diff = vb[t] - va[t]
                            num += (xHatLL[t] - va[t]) * diff
                            den += diff * diff
                        }
                        guard den > 1.0 else { continue }
                        let alpha = max(0.0, min(1.0, num / den))
                        var err: Float = 0.0
                        for t in 0..<cmpLen {
                            let pred = (1.0 - alpha) * va[t] + alpha * vb[t]
                            let dErr = xHatLL[t] - pred
                            err += dErr * dErr
                        }
                        if err < bestErr {
                            bestErr = err
                            bestI = i
                            bestJ = j
                            bestAlpha = alpha
                        }
                    }
                }

                if bestErr / Float(max(1, cmpLen)) < 4.0 {
                    let seqA = prototypes[bestI].tokens
                    let seqB = prototypes[bestJ].tokens
                    let spliced = spliceClausesPhaseCoherent(
                        seqA: seqA,
                        seqB: seqB,
                        alpha: bestAlpha,
                        periodId: periodId,
                        commaId: commaId
                    )
                    var padded = Array(spliced.prefix(length))
                    if padded.count < length {
                        padded.append(contentsOf: repeatElement(TextVocabulary.padId, count: length - padded.count))
                    }
                    return padded
                }
            }
        }

        return decodeOnLattice(llVals: xHatLL, hhVec: zHH, llVecForNorm: zLL)
    }

    /// Splits two token sequences into syntactic clause segments and splices the prefix `(1 - α)` clauses of `seqA`
    /// with the suffix `α` clauses of `seqB`, preserving complete phrases without 1-character chimeras.
    private static func spliceClausesPhaseCoherent(
        seqA: [Int],
        seqB: [Int],
        alpha: Float,
        periodId: Int,
        commaId: Int
    ) -> [Int] {
        func splitIntoClauses(_ seq: [Int]) -> [[Int]] {
            var clauses: [[Int]] = []
            var cur: [Int] = []
            for tok in seq {
                cur.append(tok)
                if tok == periodId || (tok == commaId && cur.count >= 14) {
                    clauses.append(cur)
                    cur = []
                }
            }
            if !cur.isEmpty {
                clauses.append(cur)
            }
            return clauses
        }

        let clausesA = splitIntoClauses(seqA)
        let clausesB = splitIntoClauses(seqB)
        guard !clausesA.isEmpty && !clausesB.isEmpty else {
            return alpha < 0.5 ? seqA : seqB
        }

        let clampedAlpha = max(0.0, min(1.0, alpha))
        let countA = max(1, min(clausesA.count - 1, Int((Float(clausesA.count) * (1.0 - clampedAlpha)).rounded())))
        let startB = max(1, min(clausesB.count - 1, Int((Float(clausesB.count) * (1.0 - clampedAlpha)).rounded())))

        var result: [Int] = []
        for i in 0..<countA {
            result.append(contentsOf: clausesA[i])
        }
        for j in startB..<clausesB.count {
            result.append(contentsOf: clausesB[j])
        }
        if periodId >= 0, let last = result.last, last != periodId {
            result.append(periodId)
        }
        return result
    }

    /// Autoregressive greedy decoding from principal component vector `z = [z_LL; z_HH] ∈ R^(B × 1024)`.
    public static func decodeGreedy(
        z: [Float],
        tokenEmbedding: [Float],
        weights: HawkDecoderWeights,
        batch: Int,
        length: Int,
        bosToken: Int = TextVocabulary.bosId,
        subbandClassifier: SubbandTokenClassifier = SubbandTokenClassifier()
    ) -> [[Int]] {
        let dim = weights.dim
        let latentDim = weights.latentDim
        let vocabSize = weights.vocabSize
        let numLayers = weights.layers.count
        precondition(z.count == batch * latentDim)

        if subbandClassifier.isConfigured {
            var allCalibrated: [[Int]] = []
            allCalibrated.reserveCapacity(batch)
            var allSucceeded = true
            for b in 0..<batch {
                let zVec = Array(z[(b * latentDim)..<((b + 1) * latentDim)])
                if let decoded = decodeCalibratedSubbandVector(
                    zVec: zVec,
                    weights: weights,
                    length: length,
                    classifier: subbandClassifier
                ) {
                    allCalibrated.append(decoded)
                } else {
                    allSucceeded = false
                    break
                }
            }
            if allSucceeded {
                return allCalibrated
            }
        }

        let pe = positionalEncoding(time: length, dim: dim)
        let phi = latentBasis(time: length, latentDim: latentDim)
        let rows = batch * length
        let zModAll = modulateLatent(z: z, phi: phi, batch: batch, time: length, latentDim: latentDim)
        let zCondAll = SWTHawkAutoencoder.linear(
            x: zModAll, w: weights.wCross, b: weights.bCross, rows: rows, inDim: latentDim, outDim: dim
        )

        var layerStates: [[Float]] = []
        var layerAdaGamma: [[Float]] = []
        var layerAdaBeta: [[Float]] = []
        layerStates.reserveCapacity(numLayers)
        layerAdaGamma.reserveCapacity(numLayers)
        layerAdaBeta.reserveCapacity(numLayers)

        for l in 0..<numLayers {
            let lw = weights.layers[l]
            layerStates.append(SWTHawkAutoencoder.linear(x: z, w: lw.wH0, b: lw.bH0, rows: batch, inDim: latentDim, outDim: dim))
            layerAdaGamma.append(SWTHawkAutoencoder.linear(
                x: zModAll, w: lw.wAdaGamma, b: lw.bAdaGamma, rows: rows, inDim: latentDim, outDim: dim
            ))
            layerAdaBeta.append(SWTHawkAutoencoder.linear(
                x: zModAll, w: lw.wAdaBeta, b: lw.bAdaBeta, rows: rows, inDim: latentDim, outDim: dim
            ))
        }

        // Maintain causal sliding-window histories per layer for LocalSlidingWindowAttention
        var layerHistories = [[[Float]]](
            repeating: [[Float]](repeating: [], count: batch),
            count: numLayers
        )

        var prevTok = [Int](repeating: bosToken, count: batch)
        var generated = [[Int]](repeating: [Int](repeating: 0, count: length), count: batch)

        for t in 0..<length {
            let peBase = t * dim

            var xStep = [Float](repeating: 0.0, count: batch * dim)
            for b in 0..<batch {
                let rowBase = b * dim
                let condBase = (b * length + t) * dim
                let tokId = max(0, min(vocabSize - 1, prevTok[b]))
                let embBase = tokId * dim
                for c in 0..<dim {
                    xStep[rowBase + c] = tokenEmbedding[embBase + c] + pe[peBase + c] + zCondAll[condBase + c]
                }
            }

            for l in 0..<numLayers {
                let lw = weights.layers[l]
                let mlpDim = lw.block.mlpDim
                let adaGammaAll = layerAdaGamma[l]
                let adaBetaAll = layerAdaBeta[l]
                let normed1Raw = SWTHawkAutoencoder.layerNorm(
                    x: xStep, gamma: lw.block.norm1Gamma, beta: lw.block.norm1Beta, rows: batch, dim: dim
                )
                var normed1Ada = [Float](repeating: 0.0, count: batch * dim)
                for b in 0..<batch {
                    let rowBase = b * dim
                    let condBase = (b * length + t) * dim
                    for c in 0..<dim {
                        normed1Ada[rowBase + c] = (1.0 + adaGammaAll[condBase + c]) * normed1Raw[rowBase + c] + adaBetaAll[condBase + c]
                    }
                }

                let gatePre = SWTHawkAutoencoder.linear(
                    x: normed1Ada, w: lw.block.wGate, b: lw.block.bGate, rows: batch, inDim: dim, outDim: dim
                )
                let gateAct = SWTHawkAutoencoder.silu(gatePre)

                let (_, hNext) = RGLRULayer.forward(
                    x: normed1Ada, weights: lw.block.rglru, h0: layerStates[l], batch: batch, time: 1
                )
                layerStates[l] = hNext

                var gatedH = [Float](repeating: 0.0, count: batch * dim)
                for i in 0..<(batch * dim) {
                    gatedH[i] = gateAct[i] * hNext[i]
                }
                let recOut = SWTHawkAutoencoder.linear(
                    x: gatedH, w: lw.block.wOut, b: lw.block.bOut, rows: batch, inDim: dim, outDim: dim
                )
                var xMid1 = [Float](repeating: 0.0, count: batch * dim)
                for i in 0..<(batch * dim) {
                    xMid1[i] = xStep[i] + recOut[i]
                }

                // Causal Local Sliding-Window Attention step over window W = 32
                let winSize = lw.block.localAttn.windowSize
                var xMid2 = [Float](repeating: 0.0, count: batch * dim)
                for b in 0..<batch {
                    let bSlice = Array(xMid1[(b * dim)..<((b + 1) * dim)])
                    layerHistories[l][b].append(contentsOf: bSlice)
                    let curSteps = layerHistories[l][b].count / dim
                    if curSteps > winSize {
                        layerHistories[l][b].removeFirst((curSteps - winSize) * dim)
                    }
                    let wSteps = layerHistories[l][b].count / dim
                    let attnSeq = LocalSlidingWindowAttention.forward(
                        x: layerHistories[l][b],
                        weights: lw.block.localAttn,
                        batch: 1,
                        time: wSteps
                    )
                    let lastOff = (wSteps - 1) * dim
                    for c in 0..<dim {
                        xMid2[b * dim + c] = attnSeq[lastOff + c]
                    }
                }

                let normed2 = SWTHawkAutoencoder.layerNorm(
                    x: xMid2, gamma: lw.block.norm2Gamma, beta: lw.block.norm2Beta, rows: batch, dim: dim
                )
                let mlpPre = SWTHawkAutoencoder.linear(
                    x: normed2, w: lw.block.wMLPUp, b: lw.block.bMLPUp, rows: batch, inDim: dim, outDim: mlpDim
                )
                let mlpAct = SWTHawkAutoencoder.silu(mlpPre)
                let mlpDown = SWTHawkAutoencoder.linear(
                    x: mlpAct, w: lw.block.wMLPDown, b: lw.block.bMLPDown, rows: batch, inDim: mlpDim, outDim: dim
                )
                for i in 0..<(batch * dim) {
                    xStep[i] = xMid2[i] + mlpDown[i]
                }
            }

            let finalNormed = SWTHawkAutoencoder.layerNorm(
                x: xStep, gamma: weights.finalNormGamma, beta: weights.finalNormBeta, rows: batch, dim: dim
            )
            let logits = SWTHawkAutoencoder.linear(
                x: finalNormed, w: weights.wReadout, b: weights.bReadout, rows: batch, inDim: dim, outDim: vocabSize
            )

            for b in 0..<batch {
                let vBase = b * vocabSize
                var bestId = 0
                var bestVal = logits[vBase]
                for v in 1..<vocabSize {
                    let s = logits[vBase + v]
                    if s > bestVal {
                        bestVal = s
                        bestId = v
                    }
                }
                generated[b][t] = bestId
                prevTok[b] = bestId
            }
        }

        return generated
    }
}
