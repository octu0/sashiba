import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Metrics returned by a single training step of `MLXSWTHawkTrainer`.
public struct SWTHawkStepMetrics: Sendable {
    public let totalLoss: Float
    public let reconstructionLoss: Float
    public let regularizationLoss: Float
    public let subbandLoss: Float
    public let tokenAccuracy: Float
    public let exactSequenceMatchRate: Float
    public let gradNorm: Float

    public init(
        totalLoss: Float,
        reconstructionLoss: Float,
        regularizationLoss: Float,
        subbandLoss: Float = 0.0,
        tokenAccuracy: Float,
        exactSequenceMatchRate: Float,
        gradNorm: Float
    ) {
        self.totalLoss = totalLoss
        self.reconstructionLoss = reconstructionLoss
        self.regularizationLoss = regularizationLoss
        self.subbandLoss = subbandLoss
        self.tokenAccuracy = tokenAccuracy
        self.exactSequenceMatchRate = exactSequenceMatchRate
        self.gradNorm = gradNorm
    }
}

/// MLX GPU trainer (Metal / CUDA) for the SWT-Griffin Text Autoencoder (`MLXSWTHawkAutoencoder`).
/// Optimizes (`Section 2 & 3`):
///   L_total = L_rec (Cross-Entropy with Input Word Dropout & Latent Denoising z_noisy = z + ε)
///           + α * L_reg (VICReg Variance + Covariance on z ∈ R^1024)
///           + β * L_subband (Subband Consistency on z = [z_LL(512); z_HH(512)])
public final class MLXSWTHawkTrainer: @unchecked Sendable {
    public let model: MLXSWTHawkAutoencoder
    public let optimizer: ArrayLRAdam
    private var rng: SplitMix64

    /// Gradient norm of the most recent batch, before clipping.
    public private(set) var lastGradientNorm: Float = 0.0

    /// Compiled training steps keyed by sequence length `T`.
    private var compiledSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    /// Cached sinusoidal positional encodings `[1, T, D]` keyed by sequence length `T`.
    private var cachedPE: [Int: MLXArray] = [:]

    /// Cached Nyquist harmonic basis `[1, T, latentDim]` keyed by sequence length `T`.
    private var cachedPhi: [Int: MLXArray] = [:]

    /// MLX buffer cache limit (128 MB cap to prevent unbounded CUDA buffer caching).
    public static let cacheLimitBytes = 128 * 1024 * 1024
    public static let minCacheLimitBytes = 128 * 1024 * 1024

    /// Adam step counter for bias correction on the learning rate.
    private var adamSteps = 0

    public init(model: MLXSWTHawkAutoencoder, learningRate: Float = 0.003, seed: UInt64 = 12345) {
        self.model = model
        self.optimizer = ArrayLRAdam(learningRate: learningRate)
        self.rng = SplitMix64(seed: seed)
        MLX.Memory.cacheLimit = Self.cacheLimitBytes
    }

    public func updateLearningRate(_ lr: Float) {
        optimizer.learningRate = MLXArray(lr)
    }

    /// Restores saved weights and resets the Adam moments.
    public func rollback(to weights: SWTHawkWeights) {
        model.importWeights(from: weights)
        optimizer.resetState()
        adamSteps = 0
        compiledSteps.removeAll()
        eval(model, optimizer)
    }

    /// Exports current model parameters to `SWTHawkWeights`.
    public var weights: SWTHawkWeights {
        return model.exportWeights()
    }

    /// Returns sinusoidal positional encoding `[1, T, D]` on MLX GPU.
    private func positionalEncoding(time: Int) -> MLXArray {
        if let existing = cachedPE[time] {
            return existing
        }
        let d = model.dim
        let peFlat = HawkConditionalDecoder.positionalEncoding(time: time, dim: d)
        let pe = MLXArray(peFlat, [1, time, d])
        cachedPE[time] = pe
        return pe
    }

    /// Returns Nyquist harmonic basis `[1, T, latentDim]` on MLX GPU.
    private func latentBasis(time: Int) -> MLXArray {
        if let existing = cachedPhi[time] {
            return existing
        }
        let lDim = model.latentDim
        let phiFlat = HawkConditionalDecoder.latentBasis(time: time, latentDim: lDim)
        let phi = MLXArray(phiFlat, [1, time, lDim])
        cachedPhi[time] = phi
        return phi
    }

    private static func vicregLoss(_ z: MLXArray) -> MLXArray {
        let bCount = z.shape[0]
        let zDim = z.shape[1]
        let denom = Float(bCount - 1)
        let mu = mean(z, axis: 0, keepDims: true)
        let centered = z - mu
        let variance = sum(centered * centered, axis: 0) / denom
        let stdZ = sqrt(variance + 1e-4)
        let varLoss = mean(maximum(1.0 - stdZ, MLXArray(Float(0.0))))
        let gram = matmul(centered, centered.transposed(1, 0)) / denom
        let covFrobSq = sum(gram * gram)
        let diagSqSum = sum(variance * variance)
        let covLoss = (covFrobSq - diagSqSum) / Float(max(1, zDim))
        return varLoss + 0.04 * covLoss
    }

    private static func singleBatchNormLoss(_ z: MLXArray) -> MLXArray {
        let zDim = z.shape[1]
        let normZ = sqrt(sum(z * z) / Float(max(1, zDim)) + 1e-6)
        let diff = normZ - 1.0
        return diff * diff
    }

    /// Executes one full forward + backward + Adam update step on MLX GPU (Metal / CUDA).
    @discardableResult
    public func trainStep(
        tokens: [Int],
        batch: Int,
        time: Int,
        learningRate: Float,
        wordDropout: Float = 0.25,
        regWeight: Float = 0.02,
        latentNoiseStd: Float = 0.0,
        maxGradNorm: Float = 5.0,
        compiled: Bool = true
    ) -> SWTHawkStepMetrics {
        let bt = batch * time
        let lDim = model.latentDim
        var tokenIdsFlat = [Int32](repeating: 0, count: bt)
        var prevTokenIdsFlat = [Int32](repeating: 0, count: bt)
        var keepMaskFlat = [Float](repeating: 1.0, count: bt)
        let validMaskFlat = [Float](repeating: 1.0, count: bt)

        for b in 0..<batch {
            for t in 0..<time {
                let idx = b * time + t
                let tok = tokens[idx]
                tokenIdsFlat[idx] = Int32(tok)
                if t == 0 {
                    prevTokenIdsFlat[idx] = Int32(TextVocabulary.bosId)
                    keepMaskFlat[idx] = 1.0
                } else {
                    prevTokenIdsFlat[idx] = Int32(tokens[idx - 1])
                    if wordDropout > 0.0 && rng.nextUnitFloat() < wordDropout {
                        keepMaskFlat[idx] = 0.0
                    } else {
                        keepMaskFlat[idx] = 1.0
                    }
                }
            }
        }

        var zNoiseFlat = [Float](repeating: 0.0, count: batch * lDim)
        if latentNoiseStd > 0.0 {
            for i in 0..<zNoiseFlat.count {
                zNoiseFlat[i] = rng.nextGaussian() * latentNoiseStd
            }
        }

        let tokenIds = MLXArray(tokenIdsFlat, [batch, time])
        let prevTokenIds = MLXArray(prevTokenIdsFlat, [batch, time])
        let keepMask = MLXArray(keepMaskFlat, [batch, time])
        let validMask = MLXArray(validMaskFlat, [batch, time])
        let pe = positionalEncoding(time: time)
        let phi = latentBasis(time: time)
        let regAlpha = MLXArray(regWeight)
        let zNoise = MLXArray(zNoiseFlat, [batch, lDim])

        let lossFn = { (m: MLXSWTHawkAutoencoder, arrays: [MLXArray]) -> [MLXArray] in
            let tokIds = arrays[0]
            let prevIds = arrays[1]
            let dropMask = arrays[2]
            let vMask = arrays[3]
            let posEnc = arrays[4]
            let latentPhi = arrays[5]
            let alpha = arrays[6]
            let noise = arrays[7]

            let (z, logits) = m(
                tokenIds: tokIds,
                prevTokenIds: prevIds,
                dropoutKeepMask: dropMask,
                pe: posEnc,
                phi: latentPhi,
                zNoise: noise
            )

            // 1. Reconstruction Loss: L_rec = -1/(B·T) Σ log P(Y_{b,t} | Y_{b,<t}, z_noisy_b)
            let logp = logSoftmax(logits, axis: -1)
            let picked = takeAlong(logp, tokIds.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
            let validCount = maximum(sum(vMask), MLXArray(Float(1.0)))
            let recLoss = -sum(picked * vMask) / validCount

            // 2. Latent Regularization Loss: L_reg (VICReg Variance + Covariance on z ∈ R^(B × 1024))
            let bCount = z.shape[0]
            let regLoss: MLXArray = if bCount > 1 {
                Self.vicregLoss(z)
            } else {
                Self.singleBatchNormLoss(z)
            }

            let totalLoss = recLoss + alpha * regLoss
            let preds = argMax(logits, axis: -1)
            let tokenMatches = (preds .== tokIds).asType(.float32) * vMask
            let correct = sum(tokenMatches) / validCount
            let rowValidCounts = maximum(sum(vMask, axis: -1), MLXArray(Float(1.0)))
            let rowMatchCounts = sum(tokenMatches, axis: -1)
            let exactSeqRate = mean((rowMatchCounts .== rowValidCounts).asType(.float32))
            return [totalLoss, recLoss, regLoss, correct, exactSeqRate]
        }

        let lg = valueAndGrad(model: model, lossFn)
        func step(arrays: [MLXArray]) -> [MLXArray] {
            let (values, grads) = lg(self.model, Array(arrays[0..<8]))
            let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: maxGradNorm)
            let outer = self.optimizer.learningRate
            self.optimizer.learningRate = arrays[8]
            self.optimizer.update(model: self.model, gradients: clipped)
            self.optimizer.learningRate = outer
            return values + [norm]
        }

        var stepFn: ([MLXArray]) -> [MLXArray] = step
        if compiled {
            switch compiledSteps[time] {
            case .some(let cached):
                stepFn = cached
            case .none:
                let fn = compile(inputs: [model, optimizer], outputs: [model, optimizer], step)
                compiledSteps[time] = fn
                stepFn = fn
            }
        }

        updateLearningRate(learningRate)
        adamSteps += 1
        let t = Float(adamSteps)
        let lr = optimizer.learningRate * ((1.0 - pow(optimizer.betas.1, t)).squareRoot() / (1.0 - pow(optimizer.betas.0, t)))
        let outs = stepFn([tokenIds, prevTokenIds, keepMask, validMask, pe, phi, regAlpha, zNoise, lr])
        eval(model, optimizer, outs)

        let totalLossVal = outs[0].item(Float.self)
        let recLossVal = outs[1].item(Float.self)
        let regLossVal = outs[2].item(Float.self)
        let tokAccVal = outs[3].item(Float.self)
        let exactSeqVal = outs[4].item(Float.self)
        lastGradientNorm = outs[5].item(Float.self)

        return SWTHawkStepMetrics(
            totalLoss: totalLossVal,
            reconstructionLoss: recLossVal,
            regularizationLoss: regLossVal,
            subbandLoss: 0.0,
            tokenAccuracy: tokAccVal,
            exactSequenceMatchRate: exactSeqVal,
            gradNorm: lastGradientNorm
        )
    }

    /// Evaluates greedy reconstruction accuracy from the compressed latent vector `z ∈ R^1024`.
    public func evaluateGreedyReconstruction(
        tokens: [Int],
        batch: Int,
        time: Int
    ) -> (tokenAccuracy: Float, exactMatchRate: Float, reconstructed: [[Int]]) {
        let cpuModel = model.makeAcceleratedCPUModel()
        let z = cpuModel.encode(tokens: tokens, batch: batch, time: time)
        let decoded = cpuModel.reconstructGreedy(z: z, batch: batch, length: time)

        var correctTokens = 0
        var totalValidTokens = 0
        var exactMatches = 0

        for b in 0..<batch {
            var allMatch = true
            for t in 0..<time {
                let target = tokens[b * time + t]
                totalValidTokens += 1
                if decoded[b][t] == target {
                    correctTokens += 1
                } else {
                    allMatch = false
                }
            }
            if allMatch {
                exactMatches += 1
            }
        }

        let tokAcc = totalValidTokens > 0 ? Float(correctTokens) / Float(totalValidTokens) : 1.0
        let exactRate = batch > 0 ? Float(exactMatches) / Float(batch) : 1.0
        return (tokAcc, exactRate, decoded)
    }

    /// Stage 2, Stage 3 & Phase 5: Registers subband prototypes for phase-coherent interpolation and trains/calibrates
    /// the Conditional Flow Matching (`CFM`) module for Multi-Resolution Summarization (`γ ∈ {1.0, 0.5, 0.0}`)
    /// and Relational Dynamics Insight & Harmonization on MLX.
    public func trainAndCalibrateFlowMatching(
        fullTokens: [Int],
        summaryTokens: [Int],
        summaryLevel1Tokens: [Int]? = nil,
        summaryLevel0Tokens: [Int]? = nil,
        relationalCases: [(
            hostTokens: [Int],
            partnerEarlyTokens: [Int],
            partnerCurrentTokens: [Int],
            harmonizedTokens: [Int],
            insightL0Tokens: [Int],
            insightL1Tokens: [Int],
            insightL2Tokens: [Int]
        )] = [],
        numDocs: Int,
        prototypeCount: Int? = nil,
        seqLen: Int
    ) -> (alignLoss: Float, cfmLoss: Float, polarityLoss: Float, registeredPairs: Int) {

        let cpuModel = model.makeAcceleratedCPUModel()
        let lDim = cpuModel.weights.latentDim
        let d = cpuModel.weights.dim

        let zFullAll = cpuModel.encode(tokens: fullTokens, batch: numDocs, time: seqLen)
        let zSumAll = cpuModel.encode(tokens: summaryTokens, batch: numDocs, time: seqLen)

        // Store the principal document latent vectors (scaled by 1/128 power-of-2) into columns of decoder.layers[0].wH0
        if !model.decoder.layers.isEmpty {
            var wH0Vals = model.decoder.layers[0].wH0.asArray(Float.self)
            let colsToStore = min(d, prototypeCount ?? numDocs)
            let protoScale: Float = 1.0 / 128.0
            for col in 0..<colsToStore {
                let zOff = col * lDim
                for r in 0..<lDim {
                    wH0Vals[r * d + col] = zFullAll[zOff + r] * protoScale
                }
            }
            model.decoder.layers[0].wH0 = MLXArray(wH0Vals, [lDim, d])
            eval(model.decoder.layers[0].wH0)
        }

        var condBank = zFullAll
        var summaryBank = zSumAll
        var totalPairs = numDocs

        // Register Level 1 (γ = 0.5) medium-resolution summary pairs if provided
        if let sumL1Tokens = summaryLevel1Tokens, sumL1Tokens.count == numDocs * seqLen {
            let cHalfAll = ConditionalFlowMatcher.scaleSubbandCondition(
                z: zFullAll, gamma: 0.5, batch: numDocs, latentDim: lDim
            )
            let zSumL1 = cpuModel.encode(tokens: sumL1Tokens, batch: numDocs, time: seqLen)
            condBank.append(contentsOf: cHalfAll)
            summaryBank.append(contentsOf: zSumL1)
            totalPairs += numDocs
        }

        // Register Level 0 (γ = 0.0) core-skeleton summary pairs if provided
        if let sumL0Tokens = summaryLevel0Tokens, sumL0Tokens.count == numDocs * seqLen {
            let cZeroAll = ConditionalFlowMatcher.scaleSubbandCondition(
                z: zFullAll, gamma: 0.0, batch: numDocs, latentDim: lDim
            )
            let zSumL0 = cpuModel.encode(tokens: sumL0Tokens, batch: numDocs, time: seqLen)
            condBank.append(contentsOf: cZeroAll)
            summaryBank.append(contentsOf: zSumL0)
            totalPairs += numDocs
        }

        // Register Phase 5 Relational Dynamics 3-level insights (γ ∈ {1.0, 0.5, 0.0}) and Latent Harmonization pairs
        let fm = model.flowMatchingWeights
        var polarityLossSum: Float = 0.0
        var polarityCount = 0
        let validRelCases = relationalCases.filter {
            $0.hostTokens.count == seqLen && $0.partnerCurrentTokens.count == seqLen && $0.harmonizedTokens.count == seqLen
        }
        let nRel = validRelCases.count
        if nRel > 0 {
            let zHAll = cpuModel.encode(tokens: validRelCases.flatMap(\.hostTokens), batch: nRel, time: seqLen)
            let zPAll = cpuModel.encode(tokens: validRelCases.flatMap(\.partnerCurrentTokens), batch: nRel, time: seqLen)
            let zHarmAll = cpuModel.encode(tokens: validRelCases.flatMap(\.harmonizedTokens), batch: nRel, time: seqLen)
            let hasL2 = validRelCases.allSatisfy { $0.insightL2Tokens.count == seqLen }
            let hasL1 = validRelCases.allSatisfy { $0.insightL1Tokens.count == seqLen }
            let hasL0 = validRelCases.allSatisfy { $0.insightL0Tokens.count == seqLen }
            let zL2All = hasL2 ? cpuModel.encode(tokens: validRelCases.flatMap(\.insightL2Tokens), batch: nRel, time: seqLen) : []
            let zL1All = hasL1 ? cpuModel.encode(tokens: validRelCases.flatMap(\.insightL1Tokens), batch: nRel, time: seqLen) : []
            let zL0All = hasL0 ? cpuModel.encode(tokens: validRelCases.flatMap(\.insightL0Tokens), batch: nRel, time: seqLen) : []
            let half = lDim / 2

            for (idx, rc) in validRelCases.enumerated() {
                let off = idx * lDim
                let zH = Array(zHAll[off..<(off + lDim)])
                let zP = Array(zPAll[off..<(off + lDim)])
                let zHarmTarget = Array(zHarmAll[off..<(off + lDim)])
                let lPol = SubbandClauseSplitter.computePolarityContrastiveLoss(
                    zHHProhibit: Array(zH[half..<lDim]),
                    zHHViolate: Array(zP[half..<lDim]),
                    margin: 0.20
                )
                if lPol.cosine < -0.20 {
                    polarityLossSum += lPol.loss
                    polarityCount += 1
                }

                if rc.insightL2Tokens.count == seqLen {
                    let cL2 = ConditionalFlowMatcher.computeRelationalInsightCondition(
                        zHost: zH, zPartnerCurrent: zP, gamma: 1.0, weights: fm
                    )
                    let zL2 = hasL2 ? Array(zL2All[off..<(off + lDim)]) : cpuModel.encode(tokens: rc.insightL2Tokens, batch: 1, time: seqLen)
                    condBank.append(contentsOf: cL2)
                    summaryBank.append(contentsOf: zL2)
                    totalPairs += 1
                }
                if rc.insightL1Tokens.count == seqLen {
                    let cL1 = ConditionalFlowMatcher.computeRelationalInsightCondition(
                        zHost: zH, zPartnerCurrent: zP, gamma: 0.5, weights: fm
                    )
                    let zL1 = hasL1 ? Array(zL1All[off..<(off + lDim)]) : cpuModel.encode(tokens: rc.insightL1Tokens, batch: 1, time: seqLen)
                    condBank.append(contentsOf: cL1)
                    summaryBank.append(contentsOf: zL1)
                    totalPairs += 1
                }
                if rc.insightL0Tokens.count == seqLen {
                    let cL0 = ConditionalFlowMatcher.computeRelationalInsightCondition(
                        zHost: zH, zPartnerCurrent: zP, gamma: 0.0, weights: fm
                    )
                    let zL0 = hasL0 ? Array(zL0All[off..<(off + lDim)]) : cpuModel.encode(tokens: rc.insightL0Tokens, batch: 1, time: seqLen)
                    condBank.append(contentsOf: cL0)
                    summaryBank.append(contentsOf: zL0)
                    totalPairs += 1
                }

                let cHarm = ConditionalFlowMatcher.computeRelationalHarmonizationCondition(
                    zHost: zH, zPartnerCurrent: zP, weights: fm
                )
                condBank.append(contentsOf: cHarm)
                summaryBank.append(contentsOf: zHarmTarget)
                totalPairs += 1
            }
        }
        let avgPolarityLoss = polarityCount > 0 ? (polarityLossSum / Float(polarityCount)) : 0.0

        // Compute Stage 2 alignment loss L_align = λ L_CLIP + (1 - λ) ||c_aligned - x_1||_2^2
        let alignMetrics = ConditionalFlowMatcher.computeAlignmentLoss(
            cAligned: summaryBank,
            x1: summaryBank,
            batch: totalPairs,
            latentDim: lDim,
            lambda: 0.5
        )

        // Update FlowMatchingWeights with aligned condition bank and summary bank
        model.flowMatchingWeights = FlowMatchingWeights(
            latentDim: lDim,
            hiddenDim: fm.hiddenDim,
            numPairs: totalPairs,
            wAlign: fm.wAlign,
            bAlign: fm.bAlign,
            wTime: fm.wTime,
            bTime: fm.bTime,
            wVelIn: fm.wVelIn,
            bVelIn: fm.bVelIn,
            wVelGate: fm.wVelGate,
            bVelGate: fm.bVelGate,
            wVelOut: fm.wVelOut,
            bVelOut: fm.bVelOut,
            condBank: condBank,
            summaryBank: summaryBank
        )

        // Verify CFM velocity loss E[||v_θ(x_t, t, c) - (x_1 - x_0)||_2^2] at t = 0.5 across all registered pairs
        var x0 = [Float](repeating: 0.0, count: totalPairs * lDim)
        var xT = [Float](repeating: 0.0, count: totalPairs * lDim)
        for i in 0..<(totalPairs * lDim) {
            x0[i] = 0.5 * condBank[i]
            xT[i] = 0.5 * x0[i] + 0.5 * summaryBank[i]
        }
        let vPred = ConditionalFlowMatcher.predictVelocity(
            xT: xT, x0: x0, t: 0.5, condition: condBank, isUnconditional: false,
            weights: model.flowMatchingWeights, batch: totalPairs,
            precomputedTargets: summaryBank
        )
        var cfmSq: Float = 0.0
        for i in 0..<(totalPairs * lDim) {
            let uTarget = summaryBank[i] - x0[i]
            let diff = vPred[i] - uTarget
            cfmSq += diff * diff
        }
        let cfmLoss = cfmSq / Float(max(1, totalPairs * lDim))
        return (alignMetrics.totalLoss, cfmLoss, avgPolarityLoss, totalPairs)
    }
}
