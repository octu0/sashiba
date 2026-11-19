import Foundation

/// Output returned by `PsychologicalStateLM.describePsychology(...)`.
public struct PsychologicalStateResult: Sendable {
    public let psychologicalText: String
    public let inference: RelationalInferenceOutput
    public let usedPastLog: Bool
    public let usedTemporalContext: Bool
    public let relationalLatencyMs: Double
    public let generationLatencyMs: Double

    public var totalLatencyMs: Double { relationalLatencyMs + generationLatencyMs }
}

/// Extracts the `586`-dimensional multimodal psychological condition vector from current/past token logs and optional timing metadata.
public enum PsychConditionExtractor {
    public static let hybridDim: Int = 168
    public static let probsDim: Int = 22
    public static let temporalDim: Int = 14
    /// `168 (hybT4) + 22 (probsT4) + 1 (hasT2) + 168 (hybT2) + 22 (probsT2) + 168 (deltaHyb) + 22 (deltaProbs) + 1 (hasTemporal) + 14 (temporal) = 586`
    public static let rawConditionDim: Int = 586

    private static func packProbs(from out: RelationalInferenceOutput, hybridFeatures: [Float], isEarly: Bool, cal: RelationalCalibratedHeads) -> [Float] {
        var v: [Float] = []
        v.reserveCapacity(probsDim)
        let archP = isEarly ? out.hybridArchetypeProbsT2 : out.hybridArchetypeProbsT4
        let topoP = isEarly ? out.hybridTopologyProbsT2 : out.hybridTopologyProbsT4
        for i in 0..<6 { v.append(i < archP.count ? archP[i] : 0.0) }
        for i in 0..<5 { v.append(i < topoP.count ? topoP[i] : 0.0) }

        let affHead = isEarly ? cal.hybridAffectionT2 : cal.hybridAffectionT4
        let fricHead = isEarly ? cal.hybridFrictionT2 : cal.hybridFrictionT4
        let emo = isEarly ? out.hybridPredictedEmotionT2 : out.hybridPredictedEmotionT4

        if let ah = affHead, ah.featureDim == hybridFeatures.count {
            let p = ah.probabilities(hybridFeatures)
            for i in 0..<5 { v.append(i < p.count ? p[i] : 0.0) }
        } else {
            for i in 0..<5 { v.append(i == emo.affection ? 1.0 : 0.0) }
        }

        if let fh = fricHead, fh.featureDim == hybridFeatures.count {
            let p = fh.probabilities(hybridFeatures)
            for i in 0..<5 { v.append(i < p.count ? p[i] : 0.0) }
        } else {
            for i in 0..<5 { v.append(i == emo.friction ? 1.0 : 0.0) }
        }
        v.append(emo.arousal)
        return v
    }

    public static func buildRawCondition(
        infT4: RelationalInferenceOutput,
        infT2: RelationalInferenceOutput?,
        temporalContext: TemporalRelationalContext?,
        calibratedHeads: RelationalCalibratedHeads
    ) -> [Float] {
        var raw = [Float](repeating: 0.0, count: rawConditionDim)
        var cursor = 0

        let hyb4 = infT4.hybridFeatures
        for i in 0..<min(hybridDim, hyb4.count) { raw[cursor + i] = hyb4[i] }
        cursor += hybridDim

        let p4 = packProbs(from: infT4, hybridFeatures: hyb4, isEarly: false, cal: calibratedHeads)
        for i in 0..<probsDim { raw[cursor + i] = p4[i] }
        cursor += probsDim

        if let early = infT2 {
            raw[cursor] = 1.0
            cursor += 1
            let hyb2 = early.hybridFeatures
            for i in 0..<min(hybridDim, hyb2.count) { raw[cursor + i] = hyb2[i] }
            cursor += hybridDim

            let p2 = packProbs(from: early, hybridFeatures: hyb2, isEarly: true, cal: calibratedHeads)
            for i in 0..<probsDim { raw[cursor + i] = p2[i] }
            cursor += probsDim

            for i in 0..<min(hybridDim, min(hyb4.count, hyb2.count)) {
                raw[cursor + i] = hyb4[i] - hyb2[i]
            }
            cursor += hybridDim

            for i in 0..<probsDim {
                raw[cursor + i] = p4[i] - p2[i]
            }
            cursor += probsDim
        } else {
            raw[cursor] = 0.0
            cursor += 1 + hybridDim + probsDim + hybridDim + probsDim
        }

        if let tc = temporalContext {
            raw[cursor] = 1.0
            cursor += 1
            let tf = RelationalHybridFeatureExtractor.extractTemporalContextFeatures(tc)
            for i in 0..<min(temporalDim, tf.count) { raw[cursor + i] = tf[i] }
            cursor += temporalDim
        } else {
            raw[cursor] = 0.0
            cursor += 1 + temporalDim
        }
        precondition(cursor == rawConditionDim)
        return raw
    }

    public static func extractSingle(
        relationalModel: SWTHawkAutoencoder,
        senderEarlyTokens: [Int]? = nil,
        receiverEarlyTokens: [Int]? = nil,
        senderCurrentTokens: [Int],
        receiverCurrentTokens: [Int],
        temporalContext: TemporalRelationalContext? = nil
    ) -> (rawCondition: [Float], inferenceT4: RelationalInferenceOutput) {
        let inf4 = relationalModel.inferRelational(
            senderTokenIDs: senderCurrentTokens,
            receiverTokenIDs: receiverCurrentTokens
        )
        let inf2: RelationalInferenceOutput? = if let s2 = senderEarlyTokens, let r2 = receiverEarlyTokens, (!s2.isEmpty || !r2.isEmpty) {
            relationalModel.inferRelational(senderTokenIDs: s2, receiverTokenIDs: r2)
        } else {
            nil
        }
        let cond = buildRawCondition(
            infT4: inf4,
            infT2: inf2,
            temporalContext: temporalContext,
            calibratedHeads: relationalModel.weights.relational.calibratedHeads
        )
        return (cond, inf4)
    }

    /// Multi-core CPU batch condition extractor using `encodeZRelCPUParallel` (`650+ pairs/s`).
    public static func extractBatchParallel(
        weights: SWTHawkWeights,
        items: [(s2: [Int], r2: [Int], s4: [Int], r4: [Int], temporal: TemporalRelationalContext)]
    ) -> [(fullCond: [Float], textOnlyCond: [Float])] {
        let n = items.count
        guard n > 0 else { return [] }
        let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)
        let hybExt = RelationalHybridFeatureExtractor(vocabulary: vocab)

        var allS: [[Int]] = []
        var allR: [[Int]] = []
        allS.reserveCapacity(2 * n)
        allR.reserveCapacity(2 * n)
        var maxLen = 8
        for it in items {
            allS.append(it.s2); allR.append(it.r2)
            allS.append(it.s4); allR.append(it.r4)
            maxLen = max(maxLen, max(max(it.s2.count, it.r2.count), max(it.s4.count, it.r4.count)))
        }
        let allZ = DualStreamRelationalEngine.encodeZRelCPUParallel(
            senders: allS, receivers: allR, time: maxLen, weights: weights
        )
        let zeroContent = [Float](repeating: 0.0, count: weights.relational.latentDim / 2)
        var results: [(fullCond: [Float], textOnlyCond: [Float])] = []
        results.reserveCapacity(n)

        for i in 0..<n {
            let z2 = allZ[2 * i]
            let z4 = allZ[2 * i + 1]
            let hyb2 = hybExt.extract(senderTokens: items[i].s2, receiverTokens: items[i].r2, zRel: z2, weights: weights)
            let hyb4 = hybExt.extract(senderTokens: items[i].s4, receiverTokens: items[i].r4, zRel: z4, weights: weights)
            let inf2 = DualStreamRelationalEngine.makeInferenceOutput(zRel: z2, hybridFeatures: hyb2, zContent: zeroContent, weights: weights.relational)
            let inf4 = DualStreamRelationalEngine.makeInferenceOutput(zRel: z4, hybridFeatures: hyb4, zContent: zeroContent, weights: weights.relational)

            let full = buildRawCondition(
                infT4: inf4, infT2: inf2, temporalContext: items[i].temporal,
                calibratedHeads: weights.relational.calibratedHeads
            )
            let textOnly = buildRawCondition(
                infT4: inf4, infT2: nil, temporalContext: nil,
                calibratedHeads: weights.relational.calibratedHeads
            )
            results.append((full, textOnly))
        }
        return results
    }
}

/// Weights for one Conditioned Griffin + 8-Token Cross-Attention Decoder Layer.
public struct PsychDecoderLayerWeights: Sendable, Equatable {
    public let dim: Int
    public let numHeads: Int
    public let block: HawkBlockWeights
    public let wH0: [Float]            // [D, D]
    public let bH0: [Float]            // [D]
    public let wAdaGamma: [Float]      // [D, D]
    public let bAdaGamma: [Float]      // [D]
    public let wAdaBeta: [Float]       // [D, D]
    public let bAdaBeta: [Float]       // [D]
    public let crossNormGamma: [Float] // [D]
    public let crossNormBeta: [Float]  // [D]
    public let wCrossQ: [Float]        // [D, D]
    public let bCrossQ: [Float]        // [D]
    public let wCrossK: [Float]        // [D, D]
    public let bCrossK: [Float]        // [D]
    public let wCrossV: [Float]        // [D, D]
    public let bCrossV: [Float]        // [D]
    public let wCrossOut: [Float]      // [D, D]
    public let bCrossOut: [Float]      // [D]

    public var parameterCount: Int {
        return block.parameterCount + wH0.count + bH0.count + wAdaGamma.count + bAdaGamma.count +
               wAdaBeta.count + bAdaBeta.count + crossNormGamma.count + crossNormBeta.count +
               wCrossQ.count + bCrossQ.count + wCrossK.count + bCrossK.count +
               wCrossV.count + bCrossV.count + wCrossOut.count + bCrossOut.count
    }

    public static func makeRandom(
        dim: Int,
        numHeads: Int = 6,
        windowSize: Int = 48,
        mlpDim: Int? = nil,
        residualScale: Float = 1.0,
        rng: inout SplitMix64
    ) -> PsychDecoderLayerWeights {
        let m = mlpDim ?? (2 * dim)
        let dBound = sqrtf(6.0 / Float(dim + dim))
        let outBound = dBound * residualScale
        let condBound = dBound * 0.35

        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count { a[i] = rng.nextSignedUniform(scale: bound) }
            return a
        }

        let la = LocalAttentionWeights(
            dim: dim,
            numHeads: numHeads,
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
        let upBound = sqrtf(6.0 / Float(dim + m))
        let downBound = sqrtf(6.0 / Float(m + dim)) * residualScale
        let hb = HawkBlockWeights(
            dim: dim,
            mlpDim: m,
            norm1Gamma: [Float](repeating: 1.0, count: dim),
            norm1Beta: [Float](repeating: 0.0, count: dim),
            wGate: randMat(dim * dim, dBound),
            bGate: [Float](repeating: 0.0, count: dim),
            rglru: RGLRUWeights.makeRandom(dim: dim, rng: &rng),
            wOut: randMat(dim * dim, outBound),
            bOut: [Float](repeating: 0.0, count: dim),
            localAttn: la,
            norm2Gamma: [Float](repeating: 1.0, count: dim),
            norm2Beta: [Float](repeating: 0.0, count: dim),
            wMLPUp: randMat(dim * m, upBound),
            bMLPUp: [Float](repeating: 0.0, count: m),
            wMLPDown: randMat(m * dim, downBound),
            bMLPDown: [Float](repeating: 0.0, count: dim)
        )

        return PsychDecoderLayerWeights(
            dim: dim,
            numHeads: numHeads,
            block: hb,
            wH0: randMat(dim * dim, condBound),
            bH0: [Float](repeating: 0.0, count: dim),
            wAdaGamma: randMat(dim * dim, condBound * 0.5),
            bAdaGamma: [Float](repeating: 0.0, count: dim),
            wAdaBeta: randMat(dim * dim, condBound * 0.5),
            bAdaBeta: [Float](repeating: 0.0, count: dim),
            crossNormGamma: [Float](repeating: 1.0, count: dim),
            crossNormBeta: [Float](repeating: 0.0, count: dim),
            wCrossQ: randMat(dim * dim, dBound),
            bCrossQ: [Float](repeating: 0.0, count: dim),
            wCrossK: randMat(dim * dim, dBound),
            bCrossK: [Float](repeating: 0.0, count: dim),
            wCrossV: randMat(dim * dim, dBound),
            bCrossV: [Float](repeating: 0.0, count: dim),
            wCrossOut: randMat(dim * dim, outBound),
            bCrossOut: [Float](repeating: 0.0, count: dim)
        )
    }
}

/// Complete weights of the Small Conditional Griffin Psychological State Language Model.
public struct PsychologicalStateLMWeights: Sendable, Equatable {
    public let vocabSize: Int
    public let dim: Int
    public let numHeads: Int
    public let windowSize: Int
    public let numCondTokens: Int
    public let condRawDim: Int
    public let maxSeqLen: Int
    public let serializedVocabulary: String

    public var condMean: [Float]       // [condRawDim]
    public var condStd: [Float]        // [condRawDim]
    public var wCondIn: [Float]        // [condRawDim, 2*D]
    public var bCondIn: [Float]        // [2*D]
    public var wCondOut: [Float]       // [2*D, (numCondTokens + 1) * D]
    public var bCondOut: [Float]       // [(numCondTokens + 1) * D]
    public var condNormGamma: [Float]  // [D]
    public var condNormBeta: [Float]   // [D]

    public var tokenEmbedding: [Float] // [vocabSize, D]
    public var layers: [PsychDecoderLayerWeights]
    public var finalNormGamma: [Float] // [D]
    public var finalNormBeta: [Float]  // [D]
    public var wReadout: [Float]       // [D, vocabSize]
    public var bReadout: [Float]       // [vocabSize]

    public var parameterCount: Int {
        let base = wCondIn.count + bCondIn.count + wCondOut.count + bCondOut.count +
                   condNormGamma.count + condNormBeta.count + tokenEmbedding.count +
                   finalNormGamma.count + finalNormBeta.count + wReadout.count + bReadout.count
        return base + layers.reduce(0) { $0 + $1.parameterCount }
    }

    public static func makeRandom(
        vocabSize: Int,
        dim: Int = 192,
        numLayers: Int = 3,
        numHeads: Int = 6,
        windowSize: Int = 48,
        numCondTokens: Int = 8,
        condRawDim: Int = PsychConditionExtractor.rawConditionDim,
        maxSeqLen: Int = 180,
        serializedVocabulary: String = "",
        seed: UInt64 = 2026
    ) -> PsychologicalStateLMWeights {
        precondition(dim % numHeads == 0)
        var rng = SplitMix64(seed: seed)
        let resScale = 1.0 / sqrtf(Float(max(1, 3 * numLayers)))
        let embScale: Float = 0.04
        let cInBound = sqrtf(6.0 / Float(condRawDim + 2 * dim))
        let cOutBound = sqrtf(6.0 / Float(2 * dim + (numCondTokens + 1) * dim))
        let readBound = sqrtf(6.0 / Float(dim + vocabSize))

        func randVec(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count { a[i] = rng.nextSignedUniform(scale: bound) }
            return a
        }

        var builtLayers: [PsychDecoderLayerWeights] = []
        builtLayers.reserveCapacity(numLayers)
        for _ in 0..<numLayers {
            builtLayers.append(PsychDecoderLayerWeights.makeRandom(
                dim: dim, numHeads: numHeads, windowSize: windowSize,
                mlpDim: 2 * dim, residualScale: resScale, rng: &rng
            ))
        }

        return PsychologicalStateLMWeights(
            vocabSize: vocabSize,
            dim: dim,
            numHeads: numHeads,
            windowSize: windowSize,
            numCondTokens: numCondTokens,
            condRawDim: condRawDim,
            maxSeqLen: maxSeqLen,
            serializedVocabulary: serializedVocabulary,
            condMean: [Float](repeating: 0.0, count: condRawDim),
            condStd: [Float](repeating: 1.0, count: condRawDim),
            wCondIn: randVec(condRawDim * 2 * dim, cInBound),
            bCondIn: [Float](repeating: 0.0, count: 2 * dim),
            wCondOut: randVec(2 * dim * (numCondTokens + 1) * dim, cOutBound),
            bCondOut: [Float](repeating: 0.0, count: (numCondTokens + 1) * dim),
            condNormGamma: [Float](repeating: 1.0, count: dim),
            condNormBeta: [Float](repeating: 0.0, count: dim),
            tokenEmbedding: randVec(vocabSize * dim, embScale),
            layers: builtLayers,
            finalNormGamma: [Float](repeating: 1.0, count: dim),
            finalNormBeta: [Float](repeating: 0.0, count: dim),
            wReadout: randVec(dim * vocabSize, readBound),
            bReadout: [Float](repeating: 0.0, count: vocabSize)
        )
    }

    public func save(to url: URL) throws {
        let d = dim
        let totalCondToks = numCondTokens + 1
        var tensors: [(name: String, tensor: SafeTensors.Tensor)] = [
            ("condMean", .init(shape: [condRawDim], values: condMean)),
            ("condStd", .init(shape: [condRawDim], values: condStd)),
            ("wCondIn", .init(shape: [condRawDim, 2 * d], values: wCondIn)),
            ("bCondIn", .init(shape: [2 * d], values: bCondIn)),
            ("wCondOut", .init(shape: [2 * d, totalCondToks * d], values: wCondOut)),
            ("bCondOut", .init(shape: [totalCondToks * d], values: bCondOut)),
            ("condNormGamma", .init(shape: [d], values: condNormGamma)),
            ("condNormBeta", .init(shape: [d], values: condNormBeta)),
            ("tokenEmbedding", .init(shape: [vocabSize, d], values: tokenEmbedding)),
            ("finalNormGamma", .init(shape: [d], values: finalNormGamma)),
            ("finalNormBeta", .init(shape: [d], values: finalNormBeta)),
            ("wReadout", .init(shape: [d, vocabSize], values: wReadout)),
            ("bReadout", .init(shape: [vocabSize], values: bReadout))
        ]
        for (i, lw) in layers.enumerated() {
            let p = "layers.\(i)"
            let m = lw.block.mlpDim
            tensors.append(contentsOf: [
                ("\(p).norm1Gamma", .init(shape: [d], values: lw.block.norm1Gamma)),
                ("\(p).norm1Beta", .init(shape: [d], values: lw.block.norm1Beta)),
                ("\(p).wGate", .init(shape: [d, d], values: lw.block.wGate)),
                ("\(p).bGate", .init(shape: [d], values: lw.block.bGate)),
                ("\(p).rglru.wA", .init(shape: [d, d], values: lw.block.rglru.wA)),
                ("\(p).rglru.bA", .init(shape: [d], values: lw.block.rglru.bA)),
                ("\(p).rglru.wX", .init(shape: [d, d], values: lw.block.rglru.wX)),
                ("\(p).rglru.bX", .init(shape: [d], values: lw.block.rglru.bX)),
                ("\(p).rglru.lambda", .init(shape: [d], values: lw.block.rglru.lambda)),
                ("\(p).wOut", .init(shape: [d, d], values: lw.block.wOut)),
                ("\(p).bOut", .init(shape: [d], values: lw.block.bOut)),
                ("\(p).la.normGamma", .init(shape: [d], values: lw.block.localAttn.normGamma)),
                ("\(p).la.normBeta", .init(shape: [d], values: lw.block.localAttn.normBeta)),
                ("\(p).la.wQ", .init(shape: [d, d], values: lw.block.localAttn.wQ)),
                ("\(p).la.bQ", .init(shape: [d], values: lw.block.localAttn.bQ)),
                ("\(p).la.wK", .init(shape: [d, d], values: lw.block.localAttn.wK)),
                ("\(p).la.bK", .init(shape: [d], values: lw.block.localAttn.bK)),
                ("\(p).la.wV", .init(shape: [d, d], values: lw.block.localAttn.wV)),
                ("\(p).la.bV", .init(shape: [d], values: lw.block.localAttn.bV)),
                ("\(p).la.wOut", .init(shape: [d, d], values: lw.block.localAttn.wOut)),
                ("\(p).la.bOut", .init(shape: [d], values: lw.block.localAttn.bOut)),
                ("\(p).norm2Gamma", .init(shape: [d], values: lw.block.norm2Gamma)),
                ("\(p).norm2Beta", .init(shape: [d], values: lw.block.norm2Beta)),
                ("\(p).wMLPUp", .init(shape: [d, m], values: lw.block.wMLPUp)),
                ("\(p).bMLPUp", .init(shape: [m], values: lw.block.bMLPUp)),
                ("\(p).wMLPDown", .init(shape: [m, d], values: lw.block.wMLPDown)),
                ("\(p).bMLPDown", .init(shape: [d], values: lw.block.bMLPDown)),
                ("\(p).wH0", .init(shape: [d, d], values: lw.wH0)),
                ("\(p).bH0", .init(shape: [d], values: lw.bH0)),
                ("\(p).wAdaGamma", .init(shape: [d, d], values: lw.wAdaGamma)),
                ("\(p).bAdaGamma", .init(shape: [d], values: lw.bAdaGamma)),
                ("\(p).wAdaBeta", .init(shape: [d, d], values: lw.wAdaBeta)),
                ("\(p).bAdaBeta", .init(shape: [d], values: lw.bAdaBeta)),
                ("\(p).crossNormGamma", .init(shape: [d], values: lw.crossNormGamma)),
                ("\(p).crossNormBeta", .init(shape: [d], values: lw.crossNormBeta)),
                ("\(p).wCrossQ", .init(shape: [d, d], values: lw.wCrossQ)),
                ("\(p).bCrossQ", .init(shape: [d], values: lw.bCrossQ)),
                ("\(p).wCrossK", .init(shape: [d, d], values: lw.wCrossK)),
                ("\(p).bCrossK", .init(shape: [d], values: lw.bCrossK)),
                ("\(p).wCrossV", .init(shape: [d, d], values: lw.wCrossV)),
                ("\(p).bCrossV", .init(shape: [d], values: lw.bCrossV)),
                ("\(p).wCrossOut", .init(shape: [d, d], values: lw.wCrossOut)),
                ("\(p).bCrossOut", .init(shape: [d], values: lw.bCrossOut))
            ])
        }
        let meta: [String: String] = [
            "vocabSize": String(vocabSize),
            "dim": String(dim),
            "numLayers": String(layers.count),
            "numHeads": String(numHeads),
            "windowSize": String(windowSize),
            "numCondTokens": String(numCondTokens),
            "condRawDim": String(condRawDim),
            "maxSeqLen": String(maxSeqLen),
            "serializedVocabulary": serializedVocabulary
        ]
        try SafeTensors.write(tensors, metadata: meta, to: url)
    }

    public static func load(from url: URL) throws -> PsychologicalStateLMWeights {
        let (tensors, meta) = try SafeTensors.read(from: url)
        func req(_ key: String) throws -> [Float] {
            guard let t = tensors[key] else { throw SafeTensors.Failure.badShape(key) }
            return t.values
        }
        let vocabSize = Int(meta["vocabSize"] ?? "") ?? 0
        let dim = Int(meta["dim"] ?? "") ?? 192
        let numLayers = Int(meta["numLayers"] ?? "") ?? 3
        let numHeads = Int(meta["numHeads"] ?? "") ?? 6
        let windowSize = Int(meta["windowSize"] ?? "") ?? 48
        let numCondTokens = Int(meta["numCondTokens"] ?? "") ?? 8
        let condRawDim = Int(meta["condRawDim"] ?? "") ?? PsychConditionExtractor.rawConditionDim
        let maxSeqLen = Int(meta["maxSeqLen"] ?? "") ?? 180
        let vocabStr = meta["serializedVocabulary"] ?? ""

        var builtLayers: [PsychDecoderLayerWeights] = []
        builtLayers.reserveCapacity(numLayers)
        for i in 0..<numLayers {
            let p = "layers.\(i)"
            let bMLPUp = try req("\(p).bMLPUp")
            let la = LocalAttentionWeights(
                dim: dim,
                numHeads: numHeads,
                windowSize: windowSize,
                normGamma: try req("\(p).la.normGamma"),
                normBeta: try req("\(p).la.normBeta"),
                wQ: try req("\(p).la.wQ"),
                bQ: try req("\(p).la.bQ"),
                wK: try req("\(p).la.wK"),
                bK: try req("\(p).la.bK"),
                wV: try req("\(p).la.wV"),
                bV: try req("\(p).la.bV"),
                wOut: try req("\(p).la.wOut"),
                bOut: try req("\(p).la.bOut")
            )
            let rg = RGLRUWeights(
                dim: dim,
                wA: try req("\(p).rglru.wA"),
                bA: try req("\(p).rglru.bA"),
                wX: try req("\(p).rglru.wX"),
                bX: try req("\(p).rglru.bX"),
                lambda: try req("\(p).rglru.lambda")
            )
            let hb = HawkBlockWeights(
                dim: dim,
                mlpDim: bMLPUp.count,
                norm1Gamma: try req("\(p).norm1Gamma"),
                norm1Beta: try req("\(p).norm1Beta"),
                wGate: try req("\(p).wGate"),
                bGate: try req("\(p).bGate"),
                rglru: rg,
                wOut: try req("\(p).wOut"),
                bOut: try req("\(p).bOut"),
                localAttn: la,
                norm2Gamma: try req("\(p).norm2Gamma"),
                norm2Beta: try req("\(p).norm2Beta"),
                wMLPUp: try req("\(p).wMLPUp"),
                bMLPUp: bMLPUp,
                wMLPDown: try req("\(p).wMLPDown"),
                bMLPDown: try req("\(p).bMLPDown")
            )
            builtLayers.append(PsychDecoderLayerWeights(
                dim: dim,
                numHeads: numHeads,
                block: hb,
                wH0: try req("\(p).wH0"),
                bH0: try req("\(p).bH0"),
                wAdaGamma: try req("\(p).wAdaGamma"),
                bAdaGamma: try req("\(p).bAdaGamma"),
                wAdaBeta: try req("\(p).wAdaBeta"),
                bAdaBeta: try req("\(p).bAdaBeta"),
                crossNormGamma: try req("\(p).crossNormGamma"),
                crossNormBeta: try req("\(p).crossNormBeta"),
                wCrossQ: try req("\(p).wCrossQ"),
                bCrossQ: try req("\(p).bCrossQ"),
                wCrossK: try req("\(p).wCrossK"),
                bCrossK: try req("\(p).bCrossK"),
                wCrossV: try req("\(p).wCrossV"),
                bCrossV: try req("\(p).bCrossV"),
                wCrossOut: try req("\(p).wCrossOut"),
                bCrossOut: try req("\(p).bCrossOut")
            ))
        }

        return PsychologicalStateLMWeights(
            vocabSize: vocabSize,
            dim: dim,
            numHeads: numHeads,
            windowSize: windowSize,
            numCondTokens: numCondTokens,
            condRawDim: condRawDim,
            maxSeqLen: maxSeqLen,
            serializedVocabulary: vocabStr,
            condMean: try req("condMean"),
            condStd: try req("condStd"),
            wCondIn: try req("wCondIn"),
            bCondIn: try req("bCondIn"),
            wCondOut: try req("wCondOut"),
            bCondOut: try req("bCondOut"),
            condNormGamma: try req("condNormGamma"),
            condNormBeta: try req("condNormBeta"),
            tokenEmbedding: try req("tokenEmbedding"),
            layers: builtLayers,
            finalNormGamma: try req("finalNormGamma"),
            finalNormBeta: try req("finalNormBeta"),
            wReadout: try req("wReadout"),
            bReadout: try req("bReadout")
        )
    }
}

/// Pure-CPU AVX2/SIMD8 Conditional Griffin Language Model for generating partner inner-psychology descriptions.
public final class PsychologicalStateLM: @unchecked Sendable {
    public let weights: PsychologicalStateLMWeights
    public let vocabulary: TextVocabulary

    public init(weights: PsychologicalStateLMWeights) {
        self.weights = weights
        self.vocabulary = TextVocabulary.deserialize(weights.serializedVocabulary)
    }

    /// Standardizes `condRaw: [batch * condRawDim]` and projects into `(cGlobal: [batch * D], condMem: [batch * M * D])`.
    public func projectCondition(condRaw: [Float], batch: Int) -> (cGlobal: [Float], condMem: [Float]) {
        let cDim = weights.condRawDim
        let d = weights.dim
        let mTok = weights.numCondTokens
        let totalToks = mTok + 1
        precondition(condRaw.count == batch * cDim)

        var normCond = [Float](repeating: 0.0, count: batch * cDim)
        for b in 0..<batch {
            let off = b * cDim
            for j in 0..<cDim {
                let z = (condRaw[off + j] - weights.condMean[j]) / max(weights.condStd[j], 1e-4)
                normCond[off + j] = max(-8.0, min(8.0, z))
            }
        }
        let hPre = SWTHawkAutoencoder.linear(
            x: normCond, w: weights.wCondIn, b: weights.bCondIn, rows: batch, inDim: cDim, outDim: 2 * d
        )
        let hAct = SWTHawkAutoencoder.silu(hPre)
        let projRaw = SWTHawkAutoencoder.linear(
            x: hAct, w: weights.wCondOut, b: weights.bCondOut, rows: batch, inDim: 2 * d, outDim: totalToks * d
        )
        let projNorm = SWTHawkAutoencoder.layerNorm(
            x: projRaw, gamma: weights.condNormGamma, beta: weights.condNormBeta, rows: batch * totalToks, dim: d
        )

        var cGlobal = [Float](repeating: 0.0, count: batch * d)
        var condMem = [Float](repeating: 0.0, count: batch * mTok * d)
        for b in 0..<batch {
            let srcBase = b * totalToks * d
            for c in 0..<d { cGlobal[b * d + c] = projNorm[srcBase + c] }
            let dstMem = b * mTok * d
            for k in 0..<(mTok * d) { condMem[dstMem + k] = projNorm[srcBase + d + k] }
        }
        return (cGlobal, condMem)
    }

    private func crossAttendCondition(
        x: [Float],
        condMem: [Float],
        lw: PsychDecoderLayerWeights,
        batch: Int,
        time: Int
    ) -> [Float] {
        let d = weights.dim
        let mTok = weights.numCondTokens
        let nH = weights.numHeads
        let hDim = d / nH
        let rows = batch * time
        let scale = 1.0 / sqrtf(Float(max(1, hDim)))

        let uCross = SWTHawkAutoencoder.layerNorm(
            x: x, gamma: lw.crossNormGamma, beta: lw.crossNormBeta, rows: rows, dim: d
        )
        let q = SWTHawkAutoencoder.linear(x: uCross, w: lw.wCrossQ, b: lw.bCrossQ, rows: rows, inDim: d, outDim: d)
        let k = SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossK, b: lw.bCrossK, rows: batch * mTok, inDim: d, outDim: d)
        let v = SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossV, b: lw.bCrossV, rows: batch * mTok, inDim: d, outDim: d)

        var ctx = [Float](repeating: 0.0, count: rows * d)
        var sc = [Float](repeating: 0.0, count: mTok)
        for b in 0..<batch {
            let kvBase = b * mTok * d
            for t in 0..<time {
                let qBase = (b * time + t) * d
                for h in 0..<nH {
                    let hOff = h * hDim
                    var maxS: Float = -.greatestFiniteMagnitude
                    for m in 0..<mTok {
                        var dot: Float = 0.0
                        let kOff = kvBase + m * d + hOff
                        for c in 0..<hDim { dot += q[qBase + hOff + c] * k[kOff + c] }
                        let s = dot * scale
                        sc[m] = s
                        if s > maxS { maxS = s }
                    }
                    var sumE: Float = 0.0
                    for m in 0..<mTok {
                        let e = expf(sc[m] - maxS)
                        sc[m] = e
                        sumE += e
                    }
                    let inv = 1.0 / max(sumE, 1e-12)
                    for m in 0..<mTok {
                        let p = sc[m] * inv
                        let vOff = kvBase + m * d + hOff
                        for c in 0..<hDim { ctx[qBase + hOff + c] += p * v[vOff + c] }
                    }
                }
            }
        }
        let crossOut = SWTHawkAutoencoder.linear(x: ctx, w: lw.wCrossOut, b: lw.bCrossOut, rows: rows, inDim: d, outDim: d)
        var out = [Float](repeating: 0.0, count: rows * d)
        for i in 0..<(rows * d) { out[i] = x[i] + crossOut[i] }
        return out
    }

    /// Teacher-forced CPU forward pass returning logits `[batch * time * vocabSize]` (matches `MLXPsychologicalStateLM` to `< 1e-4`).
    public func forwardTeacherForced(
        condRaw: [Float],
        prevTokens: [Int],
        batch: Int,
        time: Int
    ) -> [Float] {
        let d = weights.dim
        let vSize = weights.vocabSize
        let rows = batch * time
        precondition(prevTokens.count == rows)

        let (cGlobal, condMem) = projectCondition(condRaw: condRaw, batch: batch)
        let pe = HawkConditionalDecoder.positionalEncoding(time: time, dim: d)

        var x = [Float](repeating: 0.0, count: rows * d)
        for b in 0..<batch {
            for t in 0..<time {
                let rIdx = b * time + t
                let tok = max(0, min(vSize - 1, prevTokens[rIdx]))
                let eOff = tok * d
                let peOff = t * d
                let dst = rIdx * d
                for c in 0..<d {
                    x[dst + c] = weights.tokenEmbedding[eOff + c] + pe[peOff + c]
                }
            }
        }

        for lw in weights.layers {
            let h0 = SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wH0, b: lw.bH0, rows: batch, inDim: d, outDim: d)
            let adaG = SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaGamma, b: lw.bAdaGamma, rows: batch, inDim: d, outDim: d)
            let adaB = SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaBeta, b: lw.bAdaBeta, rows: batch, inDim: d, outDim: d)

            let norm1 = SWTHawkAutoencoder.layerNorm(x: x, gamma: lw.block.norm1Gamma, beta: lw.block.norm1Beta, rows: rows, dim: d)
            var u1 = [Float](repeating: 0.0, count: rows * d)
            for b in 0..<batch {
                let gOff = b * d
                for t in 0..<time {
                    let rOff = (b * time + t) * d
                    for c in 0..<d {
                        u1[rOff + c] = (1.0 + adaG[gOff + c]) * norm1[rOff + c] + adaB[gOff + c]
                    }
                }
            }
            let gate = SWTHawkAutoencoder.silu(
                SWTHawkAutoencoder.linear(x: u1, w: lw.block.wGate, b: lw.block.bGate, rows: rows, inDim: d, outDim: d)
            )
            let (hRec, _) = RGLRULayer.forward(x: u1, weights: lw.block.rglru, h0: h0, batch: batch, time: time)
            var gatedH = [Float](repeating: 0.0, count: rows * d)
            for i in 0..<(rows * d) { gatedH[i] = gate[i] * hRec[i] }
            let recOut = SWTHawkAutoencoder.linear(x: gatedH, w: lw.block.wOut, b: lw.block.bOut, rows: rows, inDim: d, outDim: d)
            var x1 = [Float](repeating: 0.0, count: rows * d)
            for i in 0..<(rows * d) { x1[i] = x[i] + recOut[i] }

            let x2 = LocalSlidingWindowAttention.forward(x: x1, weights: lw.block.localAttn, batch: batch, time: time)
            let x3 = crossAttendCondition(x: x2, condMem: condMem, lw: lw, batch: batch, time: time)

            let norm2 = SWTHawkAutoencoder.layerNorm(x: x3, gamma: lw.block.norm2Gamma, beta: lw.block.norm2Beta, rows: rows, dim: d)
            let mlpAct = SWTHawkAutoencoder.silu(
                SWTHawkAutoencoder.linear(x: norm2, w: lw.block.wMLPUp, b: lw.block.bMLPUp, rows: rows, inDim: d, outDim: lw.block.mlpDim)
            )
            let mlpDown = SWTHawkAutoencoder.linear(x: mlpAct, w: lw.block.wMLPDown, b: lw.block.bMLPDown, rows: rows, inDim: lw.block.mlpDim, outDim: d)
            for i in 0..<(rows * d) { x[i] = x3[i] + mlpDown[i] }
        }

        let finalNorm = SWTHawkAutoencoder.layerNorm(x: x, gamma: weights.finalNormGamma, beta: weights.finalNormBeta, rows: rows, dim: d)
        return SWTHawkAutoencoder.linear(x: finalNorm, w: weights.wReadout, b: weights.bReadout, rows: rows, inDim: d, outDim: vSize)
    }

    /// Fast KV-cached single-thread pure-CPU autoregressive generator (`~10–20 ms` for ~110 chars).
    public func generateTokens(
        condRaw: [Float],
        maxLength: Int? = nil,
        minLength: Int = 65,
        temperature: Float = 0.0,
        seed: UInt64 = 42
    ) -> [Int] {
        let d = weights.dim
        let vSize = weights.vocabSize
        let nLayers = weights.layers.count
        let nHeads = weights.numHeads
        let hDim = d / nHeads
        let halfH = hDim / 2
        let winSize = weights.windowSize
        let mTok = weights.numCondTokens
        let maxT = max(8, min(maxLength ?? weights.maxSeqLen, 256))
        let attnScale = 1.0 / sqrtf(Float(max(1, hDim)))

        let (cGlobal, condMem) = projectCondition(condRaw: condRaw, batch: 1)
        let pe = HawkConditionalDecoder.positionalEncoding(time: maxT, dim: d)
        let (cosTable, sinTable) = LocalSlidingWindowAttention.ropeTables(time: maxT, headDim: hDim)

        // Precompute static per-layer states & condition K/V once before t = 0
        var hStates: [[Float]] = []
        var adaGammas: [[Float]] = []
        var adaBetas: [[Float]] = []
        var condKs: [[Float]] = []
        var condVs: [[Float]] = []
        hStates.reserveCapacity(nLayers)
        adaGammas.reserveCapacity(nLayers)
        adaBetas.reserveCapacity(nLayers)
        condKs.reserveCapacity(nLayers)
        condVs.reserveCapacity(nLayers)

        for lw in weights.layers {
            hStates.append(SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wH0, b: lw.bH0, rows: 1, inDim: d, outDim: d))
            adaGammas.append(SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaGamma, b: lw.bAdaGamma, rows: 1, inDim: d, outDim: d))
            adaBetas.append(SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaBeta, b: lw.bAdaBeta, rows: 1, inDim: d, outDim: d))
            condKs.append(SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossK, b: lw.bCrossK, rows: mTok, inDim: d, outDim: d))
            condVs.append(SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossV, b: lw.bCrossV, rows: mTok, inDim: d, outDim: d))
        }

        // Sliding KV caches storing [kRot (D) ; v (D)] per step up to winSize
        var kvCaches = [[Float]](repeating: [], count: nLayers)
        for l in 0..<nLayers { kvCaches[l].reserveCapacity(winSize * 2 * d) }

        var rng = SplitMix64(seed: seed)
        var prevTok = TextVocabulary.bosId
        var generated: [Int] = []
        generated.reserveCapacity(maxT)
        var seen4Grams: Set<UInt64> = []

        var xStep = [Float](repeating: 0.0, count: d)
        var u1 = [Float](repeating: 0.0, count: d)
        var gatedH = [Float](repeating: 0.0, count: d)
        var x1 = [Float](repeating: 0.0, count: d)
        var qRot = [Float](repeating: 0.0, count: d)
        var kRot = [Float](repeating: 0.0, count: d)
        var scores = [Float](repeating: 0.0, count: max(winSize, mTok))

        for t in 0..<maxT {
            let embOff = max(0, min(vSize - 1, prevTok)) * d
            let peOff = t * d
            for c in 0..<d {
                xStep[c] = weights.tokenEmbedding[embOff + c] + pe[peOff + c]
            }

            let ropeOff = t * halfH
            for l in 0..<nLayers {
                let lw = weights.layers[l]
                let adaG = adaGammas[l]
                let adaB = adaBetas[l]

                // 1. AdaLN + RG-LRU step
                let norm1 = SWTHawkAutoencoder.layerNorm(x: xStep, gamma: lw.block.norm1Gamma, beta: lw.block.norm1Beta, rows: 1, dim: d)
                for c in 0..<d { u1[c] = (1.0 + adaG[c]) * norm1[c] + adaB[c] }
                let gate = SWTHawkAutoencoder.silu(
                    SWTHawkAutoencoder.linear(x: u1, w: lw.block.wGate, b: lw.block.bGate, rows: 1, inDim: d, outDim: d)
                )
                let (_, hNext) = RGLRULayer.forward(x: u1, weights: lw.block.rglru, h0: hStates[l], batch: 1, time: 1)
                hStates[l] = hNext
                for c in 0..<d { gatedH[c] = gate[c] * hNext[c] }
                let recOut = SWTHawkAutoencoder.linear(x: gatedH, w: lw.block.wOut, b: lw.block.bOut, rows: 1, inDim: d, outDim: d)
                for c in 0..<d { x1[c] = xStep[c] + recOut[c] }

                // 2. KV-cached Causal Local Self-Attention step (RoPE at exact global position t)
                let la = lw.block.localAttn
                let normLA = SWTHawkAutoencoder.layerNorm(x: x1, gamma: la.normGamma, beta: la.normBeta, rows: 1, dim: d)
                let qRaw = SWTHawkAutoencoder.linear(x: normLA, w: la.wQ, b: la.bQ, rows: 1, inDim: d, outDim: d)
                let kRaw = SWTHawkAutoencoder.linear(x: normLA, w: la.wK, b: la.bK, rows: 1, inDim: d, outDim: d)
                let vStep = SWTHawkAutoencoder.linear(x: normLA, w: la.wV, b: la.bV, rows: 1, inDim: d, outDim: d)

                for h in 0..<nHeads {
                    let hOff = h * hDim
                    for i in 0..<halfH {
                        let cs = cosTable[ropeOff + i], sn = sinTable[ropeOff + i]
                        let q1 = qRaw[hOff + i], q2 = qRaw[hOff + halfH + i]
                        qRot[hOff + i] = q1 * cs - q2 * sn
                        qRot[hOff + halfH + i] = q1 * sn + q2 * cs
                        let k1 = kRaw[hOff + i], k2 = kRaw[hOff + halfH + i]
                        kRot[hOff + i] = k1 * cs - k2 * sn
                        kRot[hOff + halfH + i] = k1 * sn + k2 * cs
                    }
                }
                kvCaches[l].append(contentsOf: kRot)
                kvCaches[l].append(contentsOf: vStep)
                let curSteps = kvCaches[l].count / (2 * d)
                if curSteps > winSize {
                    kvCaches[l].removeFirst((curSteps - winSize) * 2 * d)
                }
                let wSteps = kvCaches[l].count / (2 * d)

                var ctxLA = [Float](repeating: 0.0, count: d)
                kvCaches[l].withUnsafeBufferPointer { kvBuf in
                    qRot.withUnsafeBufferPointer { qrBuf in
                        ctxLA.withUnsafeMutableBufferPointer { ctxBuf in
                            let kvp = kvBuf.baseAddress!, qrp = qrBuf.baseAddress!, cp = ctxBuf.baseAddress!
                            for h in 0..<nHeads {
                                let hOff = h * hDim
                                var maxS: Float = -.greatestFiniteMagnitude
                                for idx in 0..<wSteps {
                                    let dot = VectorOperations.dotProduct(a: qrp + hOff, b: kvp + idx * 2 * d + hOff, count: hDim)
                                    let s = dot * attnScale
                                    scores[idx] = s
                                    if s > maxS { maxS = s }
                                }
                                var sumE: Float = 0.0
                                for idx in 0..<wSteps {
                                    let e = expf(scores[idx] - maxS)
                                    scores[idx] = e
                                    sumE += e
                                }
                                let inv = 1.0 / max(sumE, 1e-12)
                                for idx in 0..<wSteps {
                                    let p = scores[idx] * inv
                                    let vp = kvp + idx * 2 * d + d + hOff
                                    for c in 0..<hDim { cp[hOff + c] += p * vp[c] }
                                }
                            }
                        }
                    }
                }
                let laOut = SWTHawkAutoencoder.linear(x: ctxLA, w: la.wOut, b: la.bOut, rows: 1, inDim: d, outDim: d)
                var x2 = [Float](repeating: 0.0, count: d)
                for c in 0..<d { x2[c] = x1[c] + laOut[c] }

                // 3. Precomputed 8-token Condition Cross-Attention step
                let uCross = SWTHawkAutoencoder.layerNorm(x: x2, gamma: lw.crossNormGamma, beta: lw.crossNormBeta, rows: 1, dim: d)
                let qCross = SWTHawkAutoencoder.linear(x: uCross, w: lw.wCrossQ, b: lw.bCrossQ, rows: 1, inDim: d, outDim: d)
                var ctxCross = [Float](repeating: 0.0, count: d)
                condKs[l].withUnsafeBufferPointer { ckBuf in
                    condVs[l].withUnsafeBufferPointer { cvBuf in
                        qCross.withUnsafeBufferPointer { qcBuf in
                            ctxCross.withUnsafeMutableBufferPointer { ccBuf in
                                let ckp = ckBuf.baseAddress!, cvp = cvBuf.baseAddress!
                                let qcp = qcBuf.baseAddress!, ccp = ccBuf.baseAddress!
                                for h in 0..<nHeads {
                                    let hOff = h * hDim
                                    var maxS: Float = -.greatestFiniteMagnitude
                                    for m in 0..<mTok {
                                        let dot = VectorOperations.dotProduct(a: qcp + hOff, b: ckp + m * d + hOff, count: hDim)
                                        let s = dot * attnScale
                                        scores[m] = s
                                        if s > maxS { maxS = s }
                                    }
                                    var sumE: Float = 0.0
                                    for m in 0..<mTok {
                                        let e = expf(scores[m] - maxS)
                                        scores[m] = e
                                        sumE += e
                                    }
                                    let inv = 1.0 / max(sumE, 1e-12)
                                    for m in 0..<mTok {
                                        let p = scores[m] * inv
                                        let vp = cvp + m * d + hOff
                                        for c in 0..<hDim { ccp[hOff + c] += p * vp[c] }
                                    }
                                }
                            }
                        }
                    }
                }
                let crossOut = SWTHawkAutoencoder.linear(x: ctxCross, w: lw.wCrossOut, b: lw.bCrossOut, rows: 1, inDim: d, outDim: d)
                var x3 = [Float](repeating: 0.0, count: d)
                for c in 0..<d { x3[c] = x2[c] + crossOut[c] }

                // 4. Pre-LN SiLU MLP step
                let norm2 = SWTHawkAutoencoder.layerNorm(x: x3, gamma: lw.block.norm2Gamma, beta: lw.block.norm2Beta, rows: 1, dim: d)
                let mlpAct = SWTHawkAutoencoder.silu(
                    SWTHawkAutoencoder.linear(x: norm2, w: lw.block.wMLPUp, b: lw.block.bMLPUp, rows: 1, inDim: d, outDim: lw.block.mlpDim)
                )
                let mlpDown = SWTHawkAutoencoder.linear(x: mlpAct, w: lw.block.wMLPDown, b: lw.block.bMLPDown, rows: 1, inDim: lw.block.mlpDim, outDim: d)
                for c in 0..<d { xStep[c] = x3[c] + mlpDown[c] }
            }

            let finalNorm = SWTHawkAutoencoder.layerNorm(x: xStep, gamma: weights.finalNormGamma, beta: weights.finalNormBeta, rows: 1, dim: d)
            var logits = SWTHawkAutoencoder.linear(x: finalNorm, w: weights.wReadout, b: weights.bReadout, rows: 1, inDim: d, outDim: vSize)

            // Disallow <pad>, <bos>, <unk>, and premature <eos>
            logits[TextVocabulary.padId] = -1e9
            logits[TextVocabulary.bosId] = -1e9
            logits[TextVocabulary.unkId] = -1e9
            if t < minLength {
                logits[TextVocabulary.eosId] = -1e9
            }

            // 4-gram anti-stutter repetition penalty
            if generated.count >= 3 {
                let n = generated.count
                let prefixKey = (UInt64(generated[n - 3] & 0xFFFF) << 48) |
                                (UInt64(generated[n - 2] & 0xFFFF) << 32) |
                                (UInt64(generated[n - 1] & 0xFFFF) << 16)
                for cand in TextVocabulary.specialCount..<vSize {
                    if seen4Grams.contains(prefixKey | UInt64(cand & 0xFFFF)) {
                        logits[cand] -= 12.0
                    }
                }
            }

            let nextId: Int
            if temperature <= 1e-4 {
                var best = 0
                for v in 1..<vSize where logits[v] > logits[best] { best = v }
                nextId = best
            } else {
                let invTemp = 1.0 / temperature
                var maxL: Float = -.greatestFiniteMagnitude
                for v in 0..<vSize {
                    logits[v] *= invTemp
                    if logits[v] > maxL { maxL = logits[v] }
                }
                var sumP: Float = 0.0
                for v in 0..<vSize {
                    let e = expf(logits[v] - maxL)
                    logits[v] = e
                    sumP += e
                }
                var r = rng.nextUnitFloat() * sumP
                var chosen = TextVocabulary.eosId
                for v in 0..<vSize {
                    r -= logits[v]
                    if r <= 0.0 { chosen = v; break }
                }
                nextId = chosen
            }

            if nextId == TextVocabulary.eosId {
                break
            }
            if generated.count >= 3 {
                let n = generated.count
                let key = (UInt64(generated[n - 3] & 0xFFFF) << 48) |
                          (UInt64(generated[n - 2] & 0xFFFF) << 32) |
                          (UInt64(generated[n - 1] & 0xFFFF) << 16) |
                          UInt64(nextId & 0xFFFF)
                seen4Grams.insert(key)
            }
            generated.append(nextId)
            prevTok = nextId
        }
        return generated
    }

    /// End-to-end inference: evaluates relational state (`v6` / `v6_hybrid`) AND generates a natural inner-psychology description on CPU.
    public func describePsychology(
        relationalModel: SWTHawkAutoencoder,
        senderEarlyText: String? = nil,
        receiverEarlyText: String? = nil,
        senderCurrentText: String,
        receiverCurrentText: String,
        temporalContext: TemporalRelationalContext? = nil,
        temperature: Float = 0.0
    ) -> PsychologicalStateResult {
        let relVocab = relationalModel.decodedVocabulary
        let t0 = Date()
        let s4 = relVocab.encode(senderCurrentText, maxLength: 512, addEOS: false)
        let r4 = relVocab.encode(receiverCurrentText, maxLength: 512, addEOS: false)
        let s2 = senderEarlyText.map { relVocab.encode($0, maxLength: 512, addEOS: false) }
        let r2 = receiverEarlyText.map { relVocab.encode($0, maxLength: 512, addEOS: false) }
        let hasPast = (s2 != nil && r2 != nil && (!(s2!.isEmpty) || !(r2!.isEmpty)))

        let (condRaw, inf4) = PsychConditionExtractor.extractSingle(
            relationalModel: relationalModel,
            senderEarlyTokens: s2,
            receiverEarlyTokens: r2,
            senderCurrentTokens: s4,
            receiverCurrentTokens: r4,
            temporalContext: temporalContext
        )
        let relMs = Date().timeIntervalSince(t0) * 1000.0

        let t1 = Date()
        let outIds = generateTokens(condRaw: condRaw, temperature: temperature)
        let text = vocabulary.decode(outIds)
        let genMs = Date().timeIntervalSince(t1) * 1000.0

        return PsychologicalStateResult(
            psychologicalText: text,
            inference: inf4,
            usedPastLog: hasPast,
            usedTemporalContext: temporalContext != nil,
            relationalLatencyMs: relMs,
            generationLatencyMs: genMs
        )
    }
}
