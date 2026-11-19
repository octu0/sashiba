import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// One Conditioned Griffin + 8-Token Cross-Attention Layer on MLX GPU.
public final class MLXPsychDecoderLayer: Module, @unchecked Sendable {
    public let dim: Int
    public let numHeads: Int
    public var block: MLXHawkBlock
    public var wH0: MLXArray
    public var bH0: MLXArray
    public var wAdaGamma: MLXArray
    public var bAdaGamma: MLXArray
    public var wAdaBeta: MLXArray
    public var bAdaBeta: MLXArray
    public var crossNormGamma: MLXArray
    public var crossNormBeta: MLXArray
    public var wCrossQ: MLXArray
    public var bCrossQ: MLXArray
    public var wCrossK: MLXArray
    public var bCrossK: MLXArray
    public var wCrossV: MLXArray
    public var bCrossV: MLXArray
    public var wCrossOut: MLXArray
    public var bCrossOut: MLXArray

    public init(weights: PsychDecoderLayerWeights) {
        let d = weights.dim
        self.dim = d
        self.numHeads = weights.numHeads
        self.block = MLXHawkBlock(weights: weights.block)
        self.wH0 = MLXArray(weights.wH0, [d, d])
        self.bH0 = MLXArray(weights.bH0, [d])
        self.wAdaGamma = MLXArray(weights.wAdaGamma, [d, d])
        self.bAdaGamma = MLXArray(weights.bAdaGamma, [d])
        self.wAdaBeta = MLXArray(weights.wAdaBeta, [d, d])
        self.bAdaBeta = MLXArray(weights.bAdaBeta, [d])
        self.crossNormGamma = MLXArray(weights.crossNormGamma, [d])
        self.crossNormBeta = MLXArray(weights.crossNormBeta, [d])
        self.wCrossQ = MLXArray(weights.wCrossQ, [d, d])
        self.bCrossQ = MLXArray(weights.bCrossQ, [d])
        self.wCrossK = MLXArray(weights.wCrossK, [d, d])
        self.bCrossK = MLXArray(weights.bCrossK, [d])
        self.wCrossV = MLXArray(weights.wCrossV, [d, d])
        self.bCrossV = MLXArray(weights.bCrossV, [d])
        self.wCrossOut = MLXArray(weights.wCrossOut, [d, d])
        self.bCrossOut = MLXArray(weights.bCrossOut, [d])
        super.init()
    }

    public func forward(_ x: MLXArray, cGlobal: MLXArray, condMem: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let time = x.shape[1]
        let mTok = condMem.shape[1]
        let hDim = dim / numHeads
        let scale = 1.0 / sqrt(Float(max(1, hDim)))

        // 1 & 2: AdaLN + h0(cGlobal) RG-LRU + Causal Local Self-Attention
        let h0 = matmul(cGlobal, wH0) + bH0
        let adaG = (matmul(cGlobal, wAdaGamma) + bAdaGamma).expandedDimensions(axis: 1) // [B, 1, D]
        let adaB = (matmul(cGlobal, wAdaBeta) + bAdaBeta).expandedDimensions(axis: 1)   // [B, 1, D]

        let norm1Raw = mlxLayerNorm(x, gamma: block.norm1Gamma, beta: block.norm1Beta)
        let u1 = (1.0 + adaG) * norm1Raw + adaB
        let gateAct = mlxSiLU(matmul(u1, block.wGate) + block.bGate)
        let r = sigmoid(matmul(u1, block.wA) + block.bA)
        let i = sigmoid(matmul(u1, block.wX) + block.bX)
        let logA = (RGLRUWeights.decayConstant * r) * logSigmoid(block.lambda)
        let a = exp(logA)
        let sScale = sqrt(maximum(1.0 - exp(2.0 * logA) + RGLRUWeights.scaleEps, MLXArray(RGLRUWeights.scaleEps)))
        let hRec = MLXLinearScan.scan(a, sScale * (i * u1), h0: h0)
        let x1 = x + matmul(gateAct * hRec, block.wOut) + block.bOut
        let x2 = block.localAttn(x1)

        // 3: 8-Token Condition Cross-Attention
        let uCross = mlxLayerNorm(x2, gamma: crossNormGamma, beta: crossNormBeta)
        let q = (matmul(uCross, wCrossQ) + bCrossQ).reshaped([batch, time, numHeads, hDim]).transposed(0, 2, 1, 3) // [B, H, T, hDim]
        let kT = (matmul(condMem, wCrossK) + bCrossK).reshaped([batch, mTok, numHeads, hDim]).transposed(0, 2, 3, 1) // [B, H, hDim, M]
        let v = (matmul(condMem, wCrossV) + bCrossV).reshaped([batch, mTok, numHeads, hDim]).transposed(0, 2, 1, 3)   // [B, H, M, hDim]

        let probs = softmax(matmul(q, kT) * scale, axis: -1)
        let ctx = matmul(probs, v).transposed(0, 2, 1, 3).reshaped([batch, time, dim])
        let x3 = x2 + matmul(ctx, wCrossOut) + bCrossOut

        // 4: Pre-LN SiLU MLP
        let norm2 = mlxLayerNorm(x3, gamma: block.norm2Gamma, beta: block.norm2Beta)
        let mlpAct = mlxSiLU(matmul(norm2, block.wMLPUp) + block.bMLPUp)
        return x3 + matmul(mlpAct, block.wMLPDown) + block.bMLPDown
    }

    public func exportWeights() -> PsychDecoderLayerWeights {
        func arr(_ a: MLXArray) -> [Float] { a.asType(.float32).asArray(Float.self) }
        let la = LocalAttentionWeights(
            dim: dim,
            numHeads: block.localAttn.numHeads,
            windowSize: block.localAttn.windowSize,
            normGamma: arr(block.localAttn.normGamma),
            normBeta: arr(block.localAttn.normBeta),
            wQ: arr(block.localAttn.wQ),
            bQ: arr(block.localAttn.bQ),
            wK: arr(block.localAttn.wK),
            bK: arr(block.localAttn.bK),
            wV: arr(block.localAttn.wV),
            bV: arr(block.localAttn.bV),
            wOut: arr(block.localAttn.wOut),
            bOut: arr(block.localAttn.bOut)
        )
        let rg = RGLRUWeights(
            dim: dim,
            wA: arr(block.wA),
            bA: arr(block.bA),
            wX: arr(block.wX),
            bX: arr(block.bX),
            lambda: arr(block.lambda)
        )
        let hb = HawkBlockWeights(
            dim: dim,
            mlpDim: block.mlpDim,
            norm1Gamma: arr(block.norm1Gamma),
            norm1Beta: arr(block.norm1Beta),
            wGate: arr(block.wGate),
            bGate: arr(block.bGate),
            rglru: rg,
            wOut: arr(block.wOut),
            bOut: arr(block.bOut),
            localAttn: la,
            norm2Gamma: arr(block.norm2Gamma),
            norm2Beta: arr(block.norm2Beta),
            wMLPUp: arr(block.wMLPUp),
            bMLPUp: arr(block.bMLPUp),
            wMLPDown: arr(block.wMLPDown),
            bMLPDown: arr(block.bMLPDown)
        )
        return PsychDecoderLayerWeights(
            dim: dim,
            numHeads: numHeads,
            block: hb,
            wH0: arr(wH0),
            bH0: arr(bH0),
            wAdaGamma: arr(wAdaGamma),
            bAdaGamma: arr(bAdaGamma),
            wAdaBeta: arr(wAdaBeta),
            bAdaBeta: arr(bAdaBeta),
            crossNormGamma: arr(crossNormGamma),
            crossNormBeta: arr(crossNormBeta),
            wCrossQ: arr(wCrossQ),
            bCrossQ: arr(bCrossQ),
            wCrossK: arr(wCrossK),
            bCrossK: arr(bCrossK),
            wCrossV: arr(wCrossV),
            bCrossV: arr(bCrossV),
            wCrossOut: arr(wCrossOut),
            bCrossOut: arr(bCrossOut)
        )
    }
}

/// MLX GPU Module for `PsychologicalStateLM`.
public final class MLXPsychologicalStateLM: Module, @unchecked Sendable {
    public let vocabSize: Int
    public let dim: Int
    public let numHeads: Int
    public let windowSize: Int
    public let numCondTokens: Int
    public let condRawDim: Int
    public let maxSeqLen: Int
    public let serializedVocabulary: String

    @ParameterInfo public var wCondIn: MLXArray
    @ParameterInfo public var bCondIn: MLXArray
    @ParameterInfo public var wCondOut: MLXArray
    @ParameterInfo public var bCondOut: MLXArray
    @ParameterInfo public var condNormGamma: MLXArray
    @ParameterInfo public var condNormBeta: MLXArray
    @ParameterInfo public var tokenEmbedding: MLXArray
    @ModuleInfo public var layers: [MLXPsychDecoderLayer]
    @ParameterInfo public var finalNormGamma: MLXArray
    @ParameterInfo public var finalNormBeta: MLXArray
    @ParameterInfo public var wReadout: MLXArray
    @ParameterInfo public var bReadout: MLXArray

    private var condMeanArr: [Float]
    private var condStdArr: [Float]
    private var condMeanMLX: MLXArray
    private var condStdMLX: MLXArray

    public init(weights: PsychologicalStateLMWeights) {
        let d = weights.dim
        let totalToks = weights.numCondTokens + 1
        self.vocabSize = weights.vocabSize
        self.dim = d
        self.numHeads = weights.numHeads
        self.windowSize = weights.windowSize
        self.numCondTokens = weights.numCondTokens
        self.condRawDim = weights.condRawDim
        self.maxSeqLen = weights.maxSeqLen
        self.serializedVocabulary = weights.serializedVocabulary

        self._wCondIn.wrappedValue = MLXArray(weights.wCondIn, [weights.condRawDim, 2 * d])
        self._bCondIn.wrappedValue = MLXArray(weights.bCondIn, [2 * d])
        self._wCondOut.wrappedValue = MLXArray(weights.wCondOut, [2 * d, totalToks * d])
        self._bCondOut.wrappedValue = MLXArray(weights.bCondOut, [totalToks * d])
        self._condNormGamma.wrappedValue = MLXArray(weights.condNormGamma, [d])
        self._condNormBeta.wrappedValue = MLXArray(weights.condNormBeta, [d])
        self._tokenEmbedding.wrappedValue = MLXArray(weights.tokenEmbedding, [weights.vocabSize, d])
        self._layers.wrappedValue = weights.layers.map { MLXPsychDecoderLayer(weights: $0) }
        self._finalNormGamma.wrappedValue = MLXArray(weights.finalNormGamma, [d])
        self._finalNormBeta.wrappedValue = MLXArray(weights.finalNormBeta, [d])
        self._wReadout.wrappedValue = MLXArray(weights.wReadout, [d, weights.vocabSize])
        self._bReadout.wrappedValue = MLXArray(weights.bReadout, [weights.vocabSize])

        self.condMeanArr = weights.condMean
        self.condStdArr = weights.condStd
        self.condMeanMLX = MLXArray(weights.condMean, [1, weights.condRawDim])
        self.condStdMLX = MLXArray(weights.condStd.map { max($0, 1e-4) }, [1, weights.condRawDim])
        super.init()
    }

    public func setConditionNormalization(mean: [Float], std: [Float]) {
        precondition(mean.count == condRawDim && std.count == condRawDim)
        self.condMeanArr = mean
        self.condStdArr = std.map { max($0, 1e-4) }
        self.condMeanMLX = MLXArray(self.condMeanArr, [1, condRawDim])
        self.condStdMLX = MLXArray(self.condStdArr, [1, condRawDim])
        eval(condMeanMLX, condStdMLX)
    }

    public func callAsFunction(
        condRaw: MLXArray,
        prevTokenIds: MLXArray,
        pe: MLXArray,
        tokenDropMask: MLXArray? = nil,
        condNoise: MLXArray? = nil
    ) -> MLXArray {
        let batch = prevTokenIds.shape[0]
        let totalToks = numCondTokens + 1

        var zNorm = clip((condRaw - condMeanMLX) / condStdMLX, min: -8.0, max: 8.0)
        if let noise = condNoise {
            zNorm = zNorm + noise
        }
        let hAct = mlxSiLU(matmul(zNorm, wCondIn) + bCondIn)
        let proj = (matmul(hAct, wCondOut) + bCondOut).reshaped([batch, totalToks, dim])
        let projNorm = mlxLayerNorm(proj, gamma: condNormGamma, beta: condNormBeta)
        let cGlobal = projNorm[0..., 0, 0...]
        let condMem = projNorm[0..., 1..<totalToks, 0...]

        var tokEmb = tokenEmbedding[prevTokenIds]
        if let keep = tokenDropMask {
            tokEmb = tokEmb * keep.expandedDimensions(axis: -1)
        }
        var x = tokEmb + pe

        for layer in layers {
            x = layer.forward(x, cGlobal: cGlobal, condMem: condMem)
        }
        let finalNorm = mlxLayerNorm(x, gamma: finalNormGamma, beta: finalNormBeta)
        return matmul(finalNorm, wReadout) + bReadout
    }

    public func exportWeights() -> PsychologicalStateLMWeights {
        eval(self)
        func arr(_ a: MLXArray) -> [Float] { a.asType(.float32).asArray(Float.self) }
        return PsychologicalStateLMWeights(
            vocabSize: vocabSize,
            dim: dim,
            numHeads: numHeads,
            windowSize: windowSize,
            numCondTokens: numCondTokens,
            condRawDim: condRawDim,
            maxSeqLen: maxSeqLen,
            serializedVocabulary: serializedVocabulary,
            condMean: condMeanArr,
            condStd: condStdArr,
            wCondIn: arr(wCondIn),
            bCondIn: arr(bCondIn),
            wCondOut: arr(wCondOut),
            bCondOut: arr(bCondOut),
            condNormGamma: arr(condNormGamma),
            condNormBeta: arr(condNormBeta),
            tokenEmbedding: arr(tokenEmbedding),
            layers: layers.map { $0.exportWeights() },
            finalNormGamma: arr(finalNormGamma),
            finalNormBeta: arr(finalNormBeta),
            wReadout: arr(wReadout),
            bReadout: arr(bReadout)
        )
    }
}

/// MLX GPU Trainer for `MLXPsychologicalStateLM`.
public final class MLXPsychologicalStateTrainer: @unchecked Sendable {
    public let model: MLXPsychologicalStateLM
    public let optimizer: ArrayLRAdam
    private var rng: SplitMix64
    private var adamSteps: Int = 0
    private var cachedPE: [Int: MLXArray] = [:]

    public init(model: MLXPsychologicalStateLM, learningRate: Float = 1.5e-3, seed: UInt64 = 2026) {
        self.model = model
        self.optimizer = ArrayLRAdam(learningRate: learningRate)
        self.rng = SplitMix64(seed: seed)
        MLXSWTHawkAutoencoder.configureLowMemoryCache()
    }

    private func positionalEncoding(time: Int) -> MLXArray {
        if let hit = cachedPE[time] { return hit }
        let peFlat = HawkConditionalDecoder.positionalEncoding(time: time, dim: model.dim)
        let pe = MLXArray(peFlat, [1, time, model.dim])
        eval(pe)
        cachedPE[time] = pe
        return pe
    }

    public func trainStep(
        condFlat: [Float],
        targetTokensFlat: [Int],
        batch: Int,
        time: Int,
        learningRate: Float,
        wordDropout: Float = 0.12,
        condNoiseStd: Float = 0.04,
        labelSmoothing: Float = 0.04,
        weightDecay: Float = 1e-4,
        maxGradNorm: Float = 3.0
    ) -> (loss: Float, tokenAccuracy: Float, gradNorm: Float) {
        let bt = batch * time
        let cDim = model.condRawDim
        var prevFlat = [Int32](repeating: 0, count: bt)
        var targFlat = [Int32](repeating: 0, count: bt)
        var validFlat = [Float](repeating: 0.0, count: bt)
        var keepFlat = [Float](repeating: 1.0, count: bt)

        for b in 0..<batch {
            let base = b * time
            for t in 0..<time {
                let idx = base + t
                let tok = targetTokensFlat[idx]
                targFlat[idx] = Int32(tok)
                validFlat[idx] = (tok != TextVocabulary.padId) ? 1.0 : 0.0
                if t == 0 {
                    prevFlat[idx] = Int32(TextVocabulary.bosId)
                    keepFlat[idx] = 1.0
                } else {
                    let pTok = targetTokensFlat[idx - 1]
                    prevFlat[idx] = Int32(pTok == TextVocabulary.eosId ? TextVocabulary.padId : pTok)
                    if wordDropout > 0.0 && validFlat[idx] > 0.0 && rng.nextUnitFloat() < wordDropout {
                        keepFlat[idx] = 0.0
                    }
                }
            }
        }

        var noiseFlat = [Float](repeating: 0.0, count: batch * cDim)
        if condNoiseStd > 0.0 {
            for i in 0..<noiseFlat.count {
                noiseFlat[i] = rng.nextGaussian() * condNoiseStd
            }
        }

        let condArr = MLXArray(condFlat, [batch, cDim])
        let prevArr = MLXArray(prevFlat, [batch, time])
        let targArr = MLXArray(targFlat, [batch, time])
        let validArr = MLXArray(validFlat, [batch, time])
        let keepArr = MLXArray(keepFlat, [batch, time])
        let noiseArr = MLXArray(noiseFlat, [batch, cDim])
        let pe = positionalEncoding(time: time)

        let smooth = labelSmoothing
        let wd = weightDecay
        let lossFn = { (m: MLXPsychologicalStateLM, arrays: [MLXArray]) -> [MLXArray] in
            let logits = m(
                condRaw: arrays[0],
                prevTokenIds: arrays[1],
                pe: arrays[6],
                tokenDropMask: arrays[4],
                condNoise: arrays[5]
            )
            let logp = logSoftmax(logits, axis: -1)
            let picked = takeAlong(logp, arrays[2].expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
            let uniformNLL = -mean(logp, axis: -1)
            let tokenLoss = (1.0 - smooth) * (-picked) + smooth * uniformNLL
            let denom = maximum(sum(arrays[3]), MLXArray(Float(1.0)))
            let ceLoss = sum(tokenLoss * arrays[3]) / denom

            let l2Reg = wd * (mean(square(m.wCondIn)) + mean(square(m.wCondOut)) + mean(square(m.wReadout)))
            let totalLoss = ceLoss + l2Reg

            let preds = argMax(logits, axis: -1)
            let matches = (preds .== arrays[2]).asType(.float32) * arrays[3]
            let acc = sum(matches) / denom
            return [totalLoss, ceLoss, acc]
        }

        let lg = valueAndGrad(model: model, lossFn)
        let (vals, grads) = lg(model, [condArr, prevArr, targArr, validArr, keepArr, noiseArr, pe])
        let (clipped, gNorm) = clipGradNorm(gradients: grads, maxNorm: maxGradNorm)

        adamSteps += 1
        let stepF = Float(adamSteps)
        let bcFactor = sqrtf(1.0 - powf(optimizer.betas.1, stepF)) / (1.0 - powf(optimizer.betas.0, stepF))
        optimizer.learningRate = MLXArray(learningRate * bcFactor)
        optimizer.update(model: model, gradients: clipped)
        eval(model, optimizer, vals[1], vals[2], gNorm)

        return (vals[1].item(Float.self), vals[2].item(Float.self), gNorm.item(Float.self))
    }

    public func evaluateLoss(
        condFlat: [Float],
        targetTokensFlat: [Int],
        batch: Int,
        time: Int
    ) -> (ceLoss: Float, tokenAccuracy: Float) {
        let bt = batch * time
        let cDim = model.condRawDim
        var prevFlat = [Int32](repeating: 0, count: bt)
        var targFlat = [Int32](repeating: 0, count: bt)
        var validFlat = [Float](repeating: 0.0, count: bt)

        for b in 0..<batch {
            let base = b * time
            for t in 0..<time {
                let idx = base + t
                let tok = targetTokensFlat[idx]
                targFlat[idx] = Int32(tok)
                validFlat[idx] = (tok != TextVocabulary.padId) ? 1.0 : 0.0
                if t == 0 {
                    prevFlat[idx] = Int32(TextVocabulary.bosId)
                } else {
                    let pTok = targetTokensFlat[idx - 1]
                    prevFlat[idx] = Int32(pTok == TextVocabulary.eosId ? TextVocabulary.padId : pTok)
                }
            }
        }
        let condArr = MLXArray(condFlat, [batch, cDim])
        let prevArr = MLXArray(prevFlat, [batch, time])
        let targArr = MLXArray(targFlat, [batch, time])
        let validArr = MLXArray(validFlat, [batch, time])
        let pe = positionalEncoding(time: time)

        let logits = model(condRaw: condArr, prevTokenIds: prevArr, pe: pe)
        let logp = logSoftmax(logits, axis: -1)
        let picked = takeAlong(logp, targArr.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
        let denom = maximum(sum(validArr), MLXArray(Float(1.0)))
        let ceLoss = -sum(picked * validArr) / denom
        let preds = argMax(logits, axis: -1)
        let acc = sum((preds .== targArr).asType(.float32) * validArr) / denom
        eval(ceLoss, acc)
        return (ceLoss.item(Float.self), acc.item(Float.self))
    }
}
