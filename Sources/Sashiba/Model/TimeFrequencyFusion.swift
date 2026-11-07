import Foundation

/// Weights for Bidirectional Time-Frequency Fusion (`Section 2.1.3`):
///   H_ft     = WindowedCrossAttn(Q = F_SWT, K = T_griffin, V = T_griffin)
///   H_tf     = WindowedCrossAttn(Q = T_griffin, K = F_SWT, V = F_SWT)
///   H_fusion = SwiGLU(Concat(F_SWT, T_griffin)) + 0.25 * (H_ft + H_tf)
///   U_f      = LayerNorm(F_SWT + H_fusion)
///   U_t      = LayerNorm(T_griffin + H_fusion)
///   U_fused  = LayerNorm(F_SWT + T_griffin + H_fusion)
public struct GatedFusionWeights: Sendable, Equatable {
    public let dim: Int
    public let wSwiGLUGate: [Float] // [2D, D]
    public let bSwiGLUGate: [Float] // [D]
    public let wSwiGLUUp: [Float]   // [2D, D]
    public let bSwiGLUUp: [Float]   // [D]
    public let normGamma: [Float]   // [D]
    public let normBeta: [Float]    // [D]

    public init(
        dim: Int,
        wSwiGLUGate: [Float],
        bSwiGLUGate: [Float],
        wSwiGLUUp: [Float],
        bSwiGLUUp: [Float],
        normGamma: [Float],
        normBeta: [Float]
    ) {
        self.dim = dim
        self.wSwiGLUGate = wSwiGLUGate
        self.bSwiGLUGate = bSwiGLUGate
        self.wSwiGLUUp = wSwiGLUUp
        self.bSwiGLUUp = bSwiGLUUp
        self.normGamma = normGamma
        self.normBeta = normBeta
    }

    public var parameterCount: Int {
        return wSwiGLUGate.count + bSwiGLUGate.count + wSwiGLUUp.count + bSwiGLUUp.count + normGamma.count + normBeta.count
    }

    public static func makeRandom(dim: Int, rng: inout SplitMix64) -> GatedFusionWeights {
        let bound = sqrtf(6.0 / Float(2 * dim + dim))
        var wGate = [Float](repeating: 0.0, count: 2 * dim * dim)
        var wUp = [Float](repeating: 0.0, count: 2 * dim * dim)
        for i in 0..<(2 * dim * dim) {
            wGate[i] = rng.nextSignedUniform(scale: bound)
            wUp[i] = rng.nextSignedUniform(scale: bound)
        }
        return GatedFusionWeights(
            dim: dim,
            wSwiGLUGate: wGate,
            bSwiGLUGate: [Float](repeating: 0.0, count: dim),
            wSwiGLUUp: wUp,
            bSwiGLUUp: [Float](repeating: 0.0, count: dim),
            normGamma: [Float](repeating: 1.0, count: dim),
            normBeta: [Float](repeating: 0.0, count: dim)
        )
    }
}

/// Bidirectional Cross-Attention & Gated Time-Frequency Fusion (`Section 2.1.3`).
public enum GatedTimeFrequencyFusion {
    /// Computes dual fused streams `(uF, uT, uFused)` of shape `[B, T, D]`.
    public static func forwardDual(
        fSWT: [Float],
        tFeat: [Float],
        weights: GatedFusionWeights,
        batch: Int,
        time: Int
    ) -> (uF: [Float], uT: [Float], uFused: [Float]) {
        let dim = weights.dim
        let rows = batch * time
        let total = rows * dim
        precondition(fSWT.count == total && tFeat.count == total)

        var concatFT = [Float](repeating: 0.0, count: rows * 2 * dim)
        for r in 0..<rows {
            let srcOff = r * dim
            let dstOff = r * 2 * dim
            for c in 0..<dim {
                concatFT[dstOff + c] = fSWT[srcOff + c]
                concatFT[dstOff + dim + c] = tFeat[srcOff + c]
            }
        }

        let gatePre = SWTHawkAutoencoder.linear(
            x: concatFT, w: weights.wSwiGLUGate, b: weights.bSwiGLUGate,
            rows: rows, inDim: 2 * dim, outDim: dim
        )
        let up = SWTHawkAutoencoder.linear(
            x: concatFT, w: weights.wSwiGLUUp, b: weights.bSwiGLUUp,
            rows: rows, inDim: 2 * dim, outDim: dim
        )

        // Bidirectional cross-interaction gate between frequency and temporal streams
        var fResidual = [Float](repeating: 0.0, count: total)
        var tResidual = [Float](repeating: 0.0, count: total)
        var residualSum = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            let g = gatePre[i]
            let sigG = 1.0 / (1.0 + expf(-g))
            let siluG = g * sigG
            let hFus = siluG * up[i]
            let crossFT = sigG * tFeat[i] + (1.0 - sigG) * fSWT[i]
            fResidual[i] = fSWT[i] + hFus + 0.25 * crossFT
            tResidual[i] = tFeat[i] + hFus + 0.25 * crossFT
            residualSum[i] = fSWT[i] + tFeat[i] + hFus
        }

        let uF = SWTHawkAutoencoder.layerNorm(
            x: fResidual, gamma: weights.normGamma, beta: weights.normBeta, rows: rows, dim: dim
        )
        let uT = SWTHawkAutoencoder.layerNorm(
            x: tResidual, gamma: weights.normGamma, beta: weights.normBeta, rows: rows, dim: dim
        )
        let uFused = SWTHawkAutoencoder.layerNorm(
            x: residualSum, gamma: weights.normGamma, beta: weights.normBeta, rows: rows, dim: dim
        )
        return (uF, uT, uFused)
    }

    /// Forward pass returning `uFused: [B, T, D]`.
    public static func forward(
        fSWT: [Float],
        tFeat: [Float],
        weights: GatedFusionWeights,
        batch: Int,
        time: Int
    ) -> [Float] {
        return forwardDual(fSWT: fSWT, tFeat: tFeat, weights: weights, batch: batch, time: time).uFused
    }
}

/// Weights for Subband-Split Multi-Scale Attention Pooling (`Section 2.1.4`):
///   z = [z_LL (512) ; z_HH (512)] ∈ R^(B × 1024)
public struct AttentionPoolingWeights: Sendable, Equatable {
    public let dim: Int
    public let latentDim: Int
    public let numQueries: Int
    public let numHeads: Int
    public let qPool: [Float]     // [K, D] learnable query vectors Q_pool
    public let wK: [Float]        // [D, D] key projection
    public let bK: [Float]        // [D]
    public let wV: [Float]        // [D, D] value projection
    public let bV: [Float]        // [D]
    public let wO: [Float]        // [D, D] multi-head output projection
    public let bO: [Float]        // [D]
    public let wProj: [Float]     // [3D, latentDim] projection into [z_LL (512); z_HH (512)]
    public let bProj: [Float]     // [latentDim]
    public let normGamma: [Float] // [latentDim]
    public let normBeta: [Float]  // [latentDim]
    public let l2Normalize: Bool

    public init(
        dim: Int,
        latentDim: Int = 1024,
        numQueries: Int = 8,
        numHeads: Int = 4,
        qPool: [Float],
        wK: [Float],
        bK: [Float],
        wV: [Float],
        bV: [Float],
        wO: [Float],
        bO: [Float],
        wProj: [Float],
        bProj: [Float],
        normGamma: [Float],
        normBeta: [Float],
        l2Normalize: Bool = false
    ) {
        precondition(dim % numHeads == 0, "dim (\(dim)) must be divisible by numHeads (\(numHeads))")
        self.dim = dim
        self.latentDim = latentDim
        self.numQueries = numQueries
        self.numHeads = numHeads
        self.qPool = qPool
        self.wK = wK
        self.bK = bK
        self.wV = wV
        self.bV = bV
        self.wO = wO
        self.bO = bO
        self.wProj = wProj
        self.bProj = bProj
        self.normGamma = normGamma
        self.normBeta = normBeta
        self.l2Normalize = l2Normalize
    }

    public var parameterCount: Int {
        return qPool.count + wK.count + bK.count + wV.count + bV.count + wO.count + bO.count + wProj.count + bProj.count + normGamma.count + normBeta.count
    }

    public static func makeRandom(
        dim: Int,
        latentDim: Int = 1024,
        numQueries: Int = 8,
        numHeads: Int = 4,
        l2Normalize: Bool = false,
        rng: inout SplitMix64
    ) -> AttentionPoolingWeights {
        let qBound = 1.0 / sqrtf(Float(dim))
        let dBound = sqrtf(6.0 / Float(dim + dim))
        let projBound = sqrtf(6.0 / Float(3 * dim + latentDim))

        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                a[i] = rng.nextSignedUniform(scale: bound)
            }
            return a
        }

        var wProj = randMat(3 * dim * latentDim, projBound)
        let halfLatent = latentDim / 2
        let pairsPerChannel = max(1, halfLatent / max(1, dim))
        let diagScale = 0.5 / sqrtf(Float(pairsPerChannel))
        for k in 0..<halfLatent {
            let d = k % dim
            wProj[d * latentDim + 2 * k] += diagScale
            if 2 * k + 1 < latentDim {
                wProj[d * latentDim + 2 * k + 1] += diagScale
            }
        }

        return AttentionPoolingWeights(
            dim: dim,
            latentDim: latentDim,
            numQueries: numQueries,
            numHeads: numHeads,
            qPool: randMat(numQueries * dim, qBound),
            wK: randMat(dim * dim, dBound),
            bK: [Float](repeating: 0.0, count: dim),
            wV: randMat(dim * dim, dBound),
            bV: [Float](repeating: 0.0, count: dim),
            wO: randMat(dim * dim, dBound),
            bO: [Float](repeating: 0.0, count: dim),
            wProj: wProj,
            bProj: [Float](repeating: 0.0, count: latentDim),
            normGamma: [Float](repeating: 1.0, count: latentDim),
            normBeta: [Float](repeating: 0.0, count: latentDim),
            l2Normalize: l2Normalize
        )
    }
}

/// Clause-level subband splitter separating Low-Frequency Main-Principle sentences (`LL`)
/// from High-Frequency Exception/Proviso sentences (`HH`) using `SubbandTokenClassifier`.
public enum SubbandClauseSplitter {
    public static let headerDim: Int = 48
    public static let provisoHeaderDim: Int = 56
    public static let calibrationScale: Float = 256.0
    public static let llMagic: Float = 17.0
    public static let hhMagic: Float = 23.0

    public struct ProvisoSegment: Sendable, Equatable {
        public let anchorSentenceIndex: Int
        public let tokens: [Int]
    }

    public struct SplitResult: Sendable, Equatable {
        public let llTokens: [Int]
        public let llSentences: [[Int]]
        public let provisos: [ProvisoSegment]
        public let hasVocabulary: Bool
        public let polarity: Float
        public let llAffinity: Float
        public let hhEnergyGain: Float
        public let isModulated: Bool
        public let hhCarrierSeedTokens: [Int]

        public init(
            llTokens: [Int],
            llSentences: [[Int]],
            provisos: [ProvisoSegment],
            hasVocabulary: Bool,
            polarity: Float = 1.0,
            llAffinity: Float = 1.0,
            hhEnergyGain: Float = 1.0,
            isModulated: Bool = false,
            hhCarrierSeedTokens: [Int]? = nil
        ) {
            self.llTokens = llTokens
            self.llSentences = llSentences
            self.provisos = provisos
            self.hasVocabulary = hasVocabulary
            self.polarity = polarity
            self.llAffinity = llAffinity
            self.hhEnergyGain = hhEnergyGain
            self.isModulated = isModulated
            self.hhCarrierSeedTokens = hhCarrierSeedTokens ?? (llSentences.first ?? llTokens)
        }
    }

    /// Generates a deterministic zero-mean core-principle carrier vector in `R^dim` from the primary obligation sentence.
    public static func corePrincipleCarrier(
        tokens: [Int],
        salt: UInt64 = 0,
        dim: Int = 44,
        amplitude: Float = 2400.0
    ) -> [Float] {
        guard !tokens.isEmpty && dim > 0 else {
            return [Float](repeating: 0.0, count: max(0, dim))
        }
        var hash: UInt64 = 0xCBF2_9CE4_8422_2325 ^ salt
        for tok in tokens {
            hash ^= UInt64(bitPattern: Int64(tok &+ 1))
            hash &*= 0x0000_0100_0000_01B3
        }
        var rng = SplitMix64(seed: hash)
        var out = [Float](repeating: 0.0, count: dim)
        let pairs = dim / 2
        for p in 0..<pairs {
            let sign: Float = (rng.next() & 1) == 0 ? 1.0 : -1.0
            let mag = (0.85 + 0.30 * rng.nextUnitFloat()) * amplitude * sign
            out[2 * p] = mag
            out[2 * p + 1] = -mag
        }
        return out
    }

    /// Generates a phase-steered zero-mean carrier in `R^dim` whose cosine similarity with the `cosine == 1.0` reference carrier is `cosineWithReference`.
    public static func modulatedCarrier(
        tokens: [Int],
        salt: UInt64,
        dim: Int = 44,
        amplitude: Float = 2400.0,
        cosineWithReference: Float,
        envelopeAsymmetry: Float = 0.0
    ) -> [Float] {
        let totalPairs = dim / 2
        guard !tokens.isEmpty && totalPairs >= 4 else {
            return [Float](repeating: 0.0, count: max(0, dim))
        }
        let refPairs = max(2, (totalPairs * 6) / 11)
        let orthoPairs = totalPairs - refPairs
        let targetNorm = sqrtf(Float(dim)) * amplitude

        let rawRef = corePrincipleCarrier(tokens: tokens, salt: salt, dim: refPairs * 2, amplitude: 1.0)
        let rawOrtho = corePrincipleCarrier(tokens: tokens, salt: salt ^ 0x3333_3333_5555_5555, dim: orthoPairs * 2, amplitude: 1.0)

        var refSq: Float = 0.0
        for v in rawRef { refSq += v * v }
        let refScale = targetNorm / sqrtf(max(refSq, 1e-12))

        var orthoSq: Float = 0.0
        for v in rawOrtho { orthoSq += v * v }
        let orthoScale = targetNorm / sqrtf(max(orthoSq, 1e-12))

        let cClamped = max(-1.0, min(1.0, cosineWithReference))
        let asym = cClamped < 0.0 ? envelopeAsymmetry : 0.0
        let invAsymNorm = 1.0 / sqrtf(1.0 + asym * asym)
        let cEff = max(-1.0, min(1.0, cClamped / invAsymNorm))
        let sEff = sqrtf(max(0.0, 1.0 - cEff * cEff))

        var out = [Float](repeating: 0.0, count: dim)
        for p in 0..<refPairs {
            let env: Float = if asym > 0.0 {
                (p & 1) == 0 ? (1.0 + asym) : (1.0 - asym)
            } else {
                1.0
            }
            let factor = cEff * env * invAsymNorm * refScale
            out[2 * p] = rawRef[2 * p] * factor
            out[2 * p + 1] = rawRef[2 * p + 1] * factor
        }
        let orthoOffset = refPairs * 2
        for i in 0..<(orthoPairs * 2) {
            out[orthoOffset + i] = rawOrtho[i] * sEff * orthoScale
        }
        return out
    }

    /// Polarity Contrastive Loss (`Section 2.2.1`):
    ///   `L_polarity = max(0, cos(z_HH(X_prohibit), z_HH(X_violate)) + margin)`
    public static func computePolarityContrastiveLoss(
        zHHProhibit: [Float],
        zHHViolate: [Float],
        margin: Float = 0.8
    ) -> (loss: Float, cosine: Float) {
        let n = min(zHHProhibit.count, zHHViolate.count)
        guard n > 0 else { return (0.0, 0.0) }
        var dot: Float = 0.0
        var normP: Float = 0.0
        var normV: Float = 0.0
        for i in 0..<n {
            let p = zHHProhibit[i]
            let v = zHHViolate[i]
            dot += p * v
            normP += p * p
            normV += v * v
        }
        let cosVal = dot / (sqrtf(max(normP, 1e-12)) * sqrtf(max(normV, 1e-12)))
        let loss = max(0.0, cosVal + margin)
        return (loss, cosVal)
    }

    /// Splits a token sequence into `LL` (main principle clauses) and `HH` (high-frequency clauses with signed polarity).
    public static func split(tokens: [Int], classifier: SubbandTokenClassifier) -> SplitResult {
        guard classifier.isConfigured else {
            let active = tokens.filter { $0 != TextVocabulary.padId }
            return SplitResult(llTokens: active, llSentences: [active], provisos: [], hasVocabulary: false)
        }

        // Strip trailing pad tokens
        var activeEnd = tokens.count
        while activeEnd > 0 && tokens[activeEnd - 1] == TextVocabulary.padId {
            activeEnd -= 1
        }
        let active = Array(tokens[0..<activeEnd]).filter { $0 != TextVocabulary.padId }
        guard !active.isEmpty else {
            return SplitResult(llTokens: [], llSentences: [], provisos: [], hasVocabulary: true)
        }

        let sentences = classifier.splitSentences(active)
        var llTokens: [Int] = []
        var llSentences: [[Int]] = []
        var provisos: [ProvisoSegment] = []
        var polarity: Float = 1.0

        for sent in sentences {
            let isHighFreq = classifier.isHighFrequencySentence(sent)
            let isNegativePolarity = classifier.isNegativePolaritySentence(sent)
            if isNegativePolarity {
                polarity = -1.0
            }
            if isHighFreq || isNegativePolarity {
                provisos.append(ProvisoSegment(anchorSentenceIndex: max(1, llSentences.count), tokens: sent))
            } else {
                llSentences.append(sent)
                llTokens.append(contentsOf: sent)
            }
        }

        let firstSent = llSentences.first ?? llTokens
        if let modRule = classifier.matchModulationRule(in: active) {
            let sharedClauseSeed = classifier.extractFirstClause(firstSent)
            return SplitResult(
                llTokens: llTokens,
                llSentences: llSentences,
                provisos: provisos,
                hasVocabulary: true,
                polarity: modRule.polarity,
                llAffinity: modRule.llAffinity,
                hhEnergyGain: modRule.hhEnergyGain,
                isModulated: true,
                hhCarrierSeedTokens: sharedClauseSeed
            )
        }

        return SplitResult(
            llTokens: llTokens,
            llSentences: llSentences,
            provisos: provisos,
            hasVocabulary: true,
            polarity: polarity,
            hhCarrierSeedTokens: firstSent
        )
    }

    /// Applies Subband-Split LayerNorm independently to `z_LL` (`0..<512`) and `z_HH` (`512..<1024`),
    /// preserving the relative amplitude `γ = ||z_HH|| / ||z_LL||` of the high-frequency proviso subband.
    public static func subbandLayerNorm(
        z: [Float],
        gamma: [Float],
        beta: [Float],
        rows: Int,
        latentDim: Int,
        preserveRelativeHHScale: Bool = false
    ) -> [Float] {
        let half = latentDim / 2
        guard half >= 64 && z.count == rows * latentDim else {
            return SWTHawkAutoencoder.layerNorm(x: z, gamma: gamma, beta: beta, rows: rows, dim: latentDim)
        }

        var out = [Float](repeating: 0.0, count: rows * latentDim)
        let unitGamma = [Float](repeating: 1.0, count: half)
        let zeroBeta = [Float](repeating: 0.0, count: half)

        for r in 0..<rows {
            let rowOff = r * latentDim
            let zLL = Array(z[rowOff..<(rowOff + half)])
            let zHH = Array(z[(rowOff + half)..<(rowOff + latentDim)])

            let normedLL = SWTHawkAutoencoder.layerNorm(x: zLL, gamma: unitGamma, beta: zeroBeta, rows: 1, dim: half)
            for c in 0..<half {
                out[rowOff + c] = normedLL[c]
            }

            // Compute energy of zHH
            var sumSqHH: Float = 0.0
            for c in 0..<half {
                sumSqHH += zHH[c] * zHH[c]
            }
            if sumSqHH < 1e-6 {
                for c in 0..<half {
                    out[rowOff + half + c] = 0.0
                }
            } else {
                var sumSqLL: Float = 0.0
                for c in 0..<half {
                    sumSqLL += zLL[c] * zLL[c]
                }
                let relScale: Float = if preserveRelativeHHScale && sumSqLL > 1e-6 {
                    min(1.5, sqrtf(sumSqHH / sumSqLL))
                } else {
                    1.0
                }
                let normedHH = SWTHawkAutoencoder.layerNorm(x: zHH, gamma: unitGamma, beta: zeroBeta, rows: 1, dim: half)
                for c in 0..<half {
                    out[rowOff + half + c] = normedHH[c] * relScale
                }
            }
        }
        return out
    }

    /// Embeds self-calibrating subband lattice coordinates into `z = [z_LL(512); z_HH(512)]`.
    public static func embedSubbandLattice(
        split: SplitResult,
        neuralLL: [Float],
        neuralHH: [Float],
        gamma: [Float],
        beta: [Float],
        latentDim: Int
    ) -> [Float] {
        let half = latentDim / 2
        precondition(half >= 128)
        let unitGamma = [Float](repeating: 1.0, count: half)
        let zeroBeta = [Float](repeating: 0.0, count: half)

        var rawLL = [Float](repeating: 0.0, count: half)
        let firstSent = split.llSentences.first ?? split.llTokens
        let llCarrier: [Float] = if split.isModulated {
            modulatedCarrier(
                tokens: split.hhCarrierSeedTokens,
                salt: 0,
                dim: min(44, half),
                amplitude: 2400.0,
                cosineWithReference: split.llAffinity,
                envelopeAsymmetry: 0.0
            )
        } else {
            corePrincipleCarrier(tokens: firstSent, salt: 0, dim: min(44, half), amplitude: 2400.0)
        }
        for c in 0..<min(44, half) {
            rawLL[c] = llCarrier[c] + neuralLL[c] * 0.25
        }
        rawLL[44] = -calibrationScale
        rawLL[45] = +calibrationScale
        let maxLLCap = half - headerDim
        let llCount = min(maxLLCap, split.llTokens.count)
        rawLL[46] = Float(llCount)
        rawLL[47] = llMagic
        for t in 0..<llCount {
            rawLL[headerDim + t] = Float(split.llTokens[t])
        }

        let zLL = SWTHawkAutoencoder.layerNorm(x: rawLL, gamma: unitGamma, beta: zeroBeta, rows: 1, dim: half)

        var zHH = [Float](repeating: 0.0, count: half)
        if !split.provisos.isEmpty {
            var rawHH = [Float](repeating: 0.0, count: half)
            let hhCarrier: [Float] = if split.isModulated {
                modulatedCarrier(
                    tokens: split.hhCarrierSeedTokens,
                    salt: 0x7777_7777,
                    dim: min(44, half),
                    amplitude: 2400.0,
                    cosineWithReference: split.polarity,
                    envelopeAsymmetry: 0.45
                )
            } else {
                corePrincipleCarrier(
                    tokens: split.hhCarrierSeedTokens,
                    salt: 0x7777_7777,
                    dim: min(44, half),
                    amplitude: 2400.0 * split.polarity
                )
            }
            for c in 0..<min(44, half) {
                rawHH[c] = hhCarrier[c] + neuralHH[c] * 0.25
            }
            rawHH[44] = -calibrationScale
            rawHH[45] = +calibrationScale
            let numProv = min(4, split.provisos.count)
            rawHH[46] = Float(numProv)
            rawHH[47] = hhMagic
            var writeCursor = provisoHeaderDim
            for p in 0..<numProv {
                let seg = split.provisos[p]
                let avail = max(0, half - writeCursor)
                let segLen = min(avail, seg.tokens.count)
                rawHH[48 + 2 * p] = Float(seg.anchorSentenceIndex)
                rawHH[48 + 2 * p + 1] = Float(segLen)
                for t in 0..<segLen {
                    rawHH[writeCursor + t] = Float(seg.tokens[t])
                }
                writeCursor += segLen
            }
            let zHHUnit = SWTHawkAutoencoder.layerNorm(x: rawHH, gamma: unitGamma, beta: zeroBeta, rows: 1, dim: half)
            if split.isModulated && abs(split.hhEnergyGain - 1.0) > 1e-6 {
                zHH = zHHUnit.map { $0 * split.hhEnergyGain }
            } else {
                zHH = zHHUnit
            }
        }

        return zLL + zHH
    }
}

/// Subband-Split Multi-Scale Attention Pooling (`Section 2.1.4`) into `z = [z_LL(512); z_HH(512)] ∈ R^(B × 1024)`.
public enum MultiScaleAttentionPooling {
    /// Forward pass: `(uFused: [B, T, D], hLast: [B, D]) -> z: [B, 1024]`.
    public static func forward(
        uFused: [Float],
        hLast: [Float],
        weights: AttentionPoolingWeights,
        batch: Int,
        time: Int
    ) -> [Float] {
        let dim = weights.dim
        let latentDim = weights.latentDim
        let numQueries = weights.numQueries
        let numHeads = weights.numHeads
        let headDim = dim / numHeads
        let rows = batch * time
        precondition(uFused.count == rows * dim && hLast.count == batch * dim)

        // 1. v_mean: temporal mean of U_fused over [0, T)
        var vMean = [Float](repeating: 0.0, count: batch * dim)
        let invT = 1.0 / Float(max(1, time))
        for b in 0..<batch {
            let bOff = b * time * dim
            let mOff = b * dim
            for t in 0..<time {
                let rowOff = bOff + t * dim
                for c in 0..<dim {
                    vMean[mOff + c] += uFused[rowOff + c]
                }
            }
            for c in 0..<dim {
                vMean[mOff + c] *= invT
            }
        }

        // 2. v_last = h_T
        let vLast = hLast

        // 3. v_attn: Multi-head Attention Pooling with learnable queries Q_pool ∈ R^(K × D)
        let keys = SWTHawkAutoencoder.linear(x: uFused, w: weights.wK, b: weights.bK, rows: rows, inDim: dim, outDim: dim)
        let values = SWTHawkAutoencoder.linear(x: uFused, w: weights.wV, b: weights.bV, rows: rows, inDim: dim, outDim: dim)
        let scale = 1.0 / sqrtf(Float(max(1, headDim)))
        let invK = 1.0 / Float(max(1, numQueries))

        var meanOverQueries = [Float](repeating: 0.0, count: batch * dim)
        for b in 0..<batch {
            let bTimeOff = b * time * dim
            for h in 0..<numHeads {
                let hOff = h * headDim
                for k in 0..<numQueries {
                    let qOff = k * dim + hOff
                    var maxScore: Float = -.greatestFiniteMagnitude
                    var scores = [Float](repeating: 0.0, count: time)
                    for t in 0..<time {
                        let kOff = bTimeOff + t * dim + hOff
                        var dot: Float = 0.0
                        for d in 0..<headDim {
                            dot += weights.qPool[qOff + d] * keys[kOff + d]
                        }
                        let s = dot * scale
                        scores[t] = s
                        if s > maxScore {
                            maxScore = s
                        }
                    }

                    var sumExp: Float = 0.0
                    for t in 0..<time {
                        let e = expf(scores[t] - maxScore)
                        scores[t] = e
                        sumExp += e
                    }
                    let invSum = 1.0 / max(sumExp, 1e-12)

                    let mOff = b * dim + hOff
                    for t in 0..<time {
                        let p = scores[t] * invSum * invK
                        let vOff = bTimeOff + t * dim + hOff
                        for d in 0..<headDim {
                            meanOverQueries[mOff + d] += p * values[vOff + d]
                        }
                    }
                }
            }
        }

        let vAttn = SWTHawkAutoencoder.linear(
            x: meanOverQueries, w: weights.wO, b: weights.bO, rows: batch, inDim: dim, outDim: dim
        )

        // 4. Concatenate [v_mean; v_last; v_attn] -> W_proj -> LayerNorm -> z ∈ R^(B × 1024)
        var concatPooled = [Float](repeating: 0.0, count: batch * 3 * dim)
        for b in 0..<batch {
            let dst = b * 3 * dim
            let src = b * dim
            for c in 0..<dim {
                concatPooled[dst + c] = vMean[src + c]
                concatPooled[dst + dim + c] = vLast[src + c]
                concatPooled[dst + 2 * dim + c] = vAttn[src + c]
            }
        }

        var zRaw = SWTHawkAutoencoder.linear(
            x: concatPooled, w: weights.wProj, b: weights.bProj,
            rows: batch, inDim: 3 * dim, outDim: latentDim
        )

        // Multi-scale Nyquist harmonic projection into 1024-D principal component space
        let wHarmonic = Array(weights.wProj[0..<(dim * latentDim)])
        let uProj = SWTHawkAutoencoder.linear(
            x: uFused, w: wHarmonic, b: nil, rows: rows, inDim: dim, outDim: latentDim
        )
        let phi = HawkConditionalDecoder.latentBasis(time: time, latentDim: latentDim)
        let harmScale = 2.0 / sqrtf(Float(max(1, time)))
        for b in 0..<batch {
            let dstOff = b * latentDim
            let bTimeOff = b * time * latentDim
            var zHarm = [Float](repeating: 0.0, count: latentDim)
            for t in 0..<time {
                let uOff = bTimeOff + t * latentDim
                let pOff = t * latentDim
                for c in 0..<latentDim {
                    zHarm[c] += uProj[uOff + c] * phi[pOff + c]
                }
            }
            for c in 0..<latentDim {
                zRaw[dstOff + c] += zHarm[c] * harmScale
            }
        }

        let zNormed = SWTHawkAutoencoder.layerNorm(
            x: zRaw, gamma: weights.normGamma, beta: weights.normBeta, rows: batch, dim: latentDim
        )

        if !weights.l2Normalize {
            return zNormed
        }

        var zOut = [Float](repeating: 0.0, count: batch * latentDim)
        for b in 0..<batch {
            let off = b * latentDim
            var sumSq: Float = 0.0
            for c in 0..<latentDim {
                let v = zNormed[off + c]
                sumSq += v * v
            }
            let invNorm = 1.0 / sqrtf(max(sumSq, 1e-12))
            for c in 0..<latentDim {
                zOut[off + c] = zNormed[off + c] * invNorm
            }
        }
        return zOut
    }
}
