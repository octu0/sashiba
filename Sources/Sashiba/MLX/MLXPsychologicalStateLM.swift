import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// One Conditioned Griffin + Multi-Token Cross-Attention Layer on MLX GPU.
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

        let h0 = matmul(cGlobal, wH0) + bH0
        let adaG = (matmul(cGlobal, wAdaGamma) + bAdaGamma).expandedDimensions(axis: 1)
        let adaB = (matmul(cGlobal, wAdaBeta) + bAdaBeta).expandedDimensions(axis: 1)

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

        let uCross = mlxLayerNorm(x2, gamma: crossNormGamma, beta: crossNormBeta)
        let q = (matmul(uCross, wCrossQ) + bCrossQ).reshaped([batch, time, numHeads, hDim]).transposed(0, 2, 1, 3)
        let kT = (matmul(condMem, wCrossK) + bCrossK).reshaped([batch, mTok, numHeads, hDim]).transposed(0, 2, 3, 1)
        let v = (matmul(condMem, wCrossV) + bCrossV).reshaped([batch, mTok, numHeads, hDim]).transposed(0, 2, 1, 3)

        let probs = softmax(matmul(q, kT) * scale, axis: -1)
        let ctx = matmul(probs, v).transposed(0, 2, 1, 3).reshaped([batch, time, dim])
        let x3 = x2 + matmul(ctx, wCrossOut) + bCrossOut

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

/// MLX GPU Module for Subword-Tokenized `PsychologicalStateLM`.
public final class MLXPsychologicalStateLM: Module, @unchecked Sendable {
    public let vocabSize: Int
    public let dim: Int
    public let numHeads: Int
    public let windowSize: Int
    public let numCondTokens: Int
    public let condRawDim: Int
    public let maxSeqLen: Int
    public let serializedSubwordVocabulary: String
    public var auxiliaryHeads: PsychAuxiliaryHeads
    public let useTiedEmbeddings: Bool

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

    public init(weights: PsychologicalStateLMWeights, useTiedEmbeddings: Bool = false) {
        let d = weights.dim
        let totalToks = weights.numCondTokens + 1
        self.vocabSize = weights.vocabSize
        self.dim = d
        self.numHeads = weights.numHeads
        self.windowSize = weights.windowSize
        self.numCondTokens = weights.numCondTokens
        self.condRawDim = weights.condRawDim
        self.maxSeqLen = weights.maxSeqLen
        self.serializedSubwordVocabulary = weights.serializedSubwordVocabulary
        self.auxiliaryHeads = weights.auxiliaryHeads
        self.useTiedEmbeddings = useTiedEmbeddings

        self._wCondIn.wrappedValue = MLXArray(weights.wCondIn, [weights.condRawDim, d])
        self._bCondIn.wrappedValue = MLXArray(weights.bCondIn, [d])
        self._wCondOut.wrappedValue = MLXArray(weights.wCondOut, [d, totalToks * d])
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
        hiddenDropMask: MLXArray? = nil,
        condDropMask: MLXArray? = nil,
        condNoise: MLXArray? = nil
    ) -> MLXArray {
        let batch = prevTokenIds.shape[0]
        let totalToks = numCondTokens + 1

        var zNorm = clip((condRaw - condMeanMLX) / condStdMLX, min: -6.0, max: 6.0)
        if let noise = condNoise {
            zNorm = zNorm + noise
        }
        if let cmask = condDropMask {
            zNorm = zNorm * cmask
        }
        let hAct = mlxSiLU(matmul(zNorm, wCondIn) + bCondIn)
        let proj = (matmul(hAct, wCondOut) + bCondOut).reshaped([batch, totalToks, dim])
        let projNorm = mlxLayerNorm(proj, gamma: condNormGamma, beta: condNormBeta)
        let cGlobal = projNorm[0..., 0, 0...]
        let condMem = projNorm[0..., 1..<totalToks, 0...]

        let tokEmb = tokenEmbedding[prevTokenIds] * Float(2.5)
        var x = tokEmb + pe

        for layer in layers {
            x = layer.forward(x, cGlobal: cGlobal, condMem: condMem)
        }
        var finalNorm = mlxLayerNorm(x, gamma: finalNormGamma, beta: finalNormBeta)
        if let hMask = hiddenDropMask {
            finalNorm = finalNorm * hMask
        }
        let readoutW = useTiedEmbeddings ? tokenEmbedding.transposed() : wReadout
        return matmul(finalNorm, readoutW) + bReadout
    }

    public func exportWeights() -> PsychologicalStateLMWeights {
        eval(self)
        func arr(_ a: MLXArray) -> [Float] { a.asType(.float32).asArray(Float.self) }
        let exportedReadout: [Float]
        if useTiedEmbeddings {
            let tMat = contiguous(tokenEmbedding.transposed())
            eval(tMat)
            exportedReadout = arr(tMat)
        } else {
            exportedReadout = arr(wReadout)
        }
        return PsychologicalStateLMWeights(
            vocabSize: vocabSize,
            dim: dim,
            numHeads: numHeads,
            windowSize: windowSize,
            numCondTokens: numCondTokens,
            condRawDim: condRawDim,
            maxSeqLen: maxSeqLen,
            serializedSubwordVocabulary: serializedSubwordVocabulary,
            auxiliaryHeads: auxiliaryHeads,
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
            wReadout: exportedReadout,
            bReadout: arr(bReadout)
        )
    }
}

/// Compiled MLX CUDA/Metal Trainer for `MLXPsychologicalStateLM`.
public final class MLXPsychologicalStateTrainer: @unchecked Sendable {
    public let model: MLXPsychologicalStateLM
    public let optimizer: ArrayLRAdam
    private var rng: SplitMix64
    private var adamSteps: Int = 0
    private var cachedPE: [Int: MLXArray] = [:]
    private var compiledSteps: [Int: ([MLXArray]) -> [MLXArray]] = [:]

    public init(model: MLXPsychologicalStateLM, learningRate: Float = 2.0e-3, seed: UInt64 = 2026) {
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
        hiddenDropout: Float = 0.14,
        condDropout: Float = 0.04,
        condNoiseStd: Float = 0.03,
        labelSmoothing: Float = 0.04,
        weightDecay: Float = 8.0e-3,
        maxGradNorm: Float = 3.0
    ) -> (loss: Float, tokenAccuracy: Float, gradNorm: Float) {
        let bt = batch * time
        let cDim = model.condRawDim
        let d = model.dim
        var prevFlat = [Int32](repeating: 0, count: bt)
        var targFlat = [Int32](repeating: 0, count: bt)
        var validFlat = [Float](repeating: 0.0, count: bt)

        for b in 0..<batch {
            let base = b * time
            for t in 0..<time {
                let idx = base + t
                let tok = targetTokensFlat[idx]
                targFlat[idx] = Int32(tok)
                validFlat[idx] = (tok != TextVocabulary.padId && tok != TextVocabulary.unkId) ? 1.0 : 0.0
                if t == 0 {
                    prevFlat[idx] = Int32(TextVocabulary.bosId)
                } else {
                    let pTok = targetTokensFlat[idx - 1]
                    prevFlat[idx] = Int32(pTok == TextVocabulary.eosId ? TextVocabulary.padId : pTok)
                }
            }
        }

        var hDropFlat = [Float](repeating: 1.0, count: batch * d)
        if hiddenDropout > 0.0 {
            let hScale = 1.0 / max(0.1, 1.0 - hiddenDropout)
            for i in 0..<hDropFlat.count {
                hDropFlat[i] = (rng.nextUnitFloat() < hiddenDropout) ? 0.0 : hScale
            }
        }

        var condMaskFlat = [Float](repeating: 1.0, count: batch * cDim)
        if condDropout > 0.0 {
            let scale = 1.0 / max(0.1, 1.0 - condDropout)
            for i in 0..<condMaskFlat.count {
                condMaskFlat[i] = (rng.nextUnitFloat() < condDropout) ? 0.0 : scale
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
        let hMaskArr = MLXArray(hDropFlat, [batch, 1, d])
        let cMaskArr = MLXArray(condMaskFlat, [batch, cDim])
        let noiseArr = MLXArray(noiseFlat, [batch, cDim])
        let pe = positionalEncoding(time: time)

        let smooth = labelSmoothing
        let wd = weightDecay
        let shapeKey = (batch << 16) | time
        let stepFn: ([MLXArray]) -> [MLXArray]
        if let existing = compiledSteps[shapeKey] {
            stepFn = existing
        } else {
            let lossFn = { (m: MLXPsychologicalStateLM, arrays: [MLXArray]) -> [MLXArray] in
                let logits = m(
                    condRaw: arrays[0],
                    prevTokenIds: arrays[1],
                    pe: arrays[7],
                    hiddenDropMask: arrays[4],
                    condDropMask: arrays[5],
                    condNoise: arrays[6]
                )
                let logp = logSoftmax(logits, axis: -1)
                let picked = takeAlong(logp, arrays[2].expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
                let uniformNLL = -mean(logp, axis: -1)
                let tokenLoss = (1.0 - smooth) * (-picked) + smooth * uniformNLL
                let denom = maximum(sum(arrays[3]), MLXArray(Float(1.0)))
                let ceLoss = sum(tokenLoss * arrays[3]) / denom

                var l2Sum = mean(square(m.wCondIn)) + mean(square(m.wCondOut)) + mean(square(m.tokenEmbedding))
                if !m.useTiedEmbeddings {
                    l2Sum = l2Sum + mean(square(m.wReadout))
                }
                for l in m.layers {
                    l2Sum = l2Sum + 0.25 * (mean(square(l.block.wMLPUp)) + mean(square(l.block.wMLPDown)) + mean(square(l.wCrossOut)))
                }
                let totalLoss = ceLoss + wd * l2Sum

                let preds = argMax(logits, axis: -1)
                let matches = (preds .== arrays[2]).asType(.float32) * arrays[3]
                let acc = sum(matches) / denom
                return [totalLoss, ceLoss, acc]
            }
            let lg = valueAndGrad(model: model, lossFn)
            let rawStep = { [unowned self] (arrays: [MLXArray]) -> [MLXArray] in
                let (vals, grads) = lg(self.model, Array(arrays[0..<8]))
                let (clipped, norm) = clipGradNorm(gradients: grads, maxNorm: maxGradNorm)
                let prevLR = self.optimizer.learningRate
                self.optimizer.learningRate = arrays[8]
                self.optimizer.update(model: self.model, gradients: clipped)
                self.optimizer.learningRate = prevLR
                return [vals[1], vals[2], norm]
            }
            let compiled = compile(inputs: [model, optimizer], outputs: [model, optimizer], rawStep)
            compiledSteps[shapeKey] = compiled
            stepFn = compiled
        }

        adamSteps += 1
        let stepF = Float(adamSteps)
        let bcFactor = sqrtf(1.0 - powf(optimizer.betas.1, stepF)) / (1.0 - powf(optimizer.betas.0, stepF))
        let effLR = MLXArray(learningRate * bcFactor)

        let out = stepFn([condArr, prevArr, targArr, validArr, hMaskArr, cMaskArr, noiseArr, pe, effLR])
        eval(model, optimizer, out[0], out[1], out[2])
        return (out[0].item(Float.self), out[1].item(Float.self), out[2].item(Float.self))
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
                validFlat[idx] = (tok != TextVocabulary.padId && tok != TextVocabulary.unkId) ? 1.0 : 0.0
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
