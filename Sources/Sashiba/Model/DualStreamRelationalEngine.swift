import Foundation

/// Per-module gradient norm and Frobenius weight movement telemetry recorded during End-to-End training (`Phase 5.1 Audit 2`).
public struct AuditModuleMetrics: Codable, Sendable, Equatable {
    public let moduleName: String
    public let parameterCount: Int
    public let initialFrobeniusNorm: Float
    public let finalFrobeniusNorm: Float
    public let weightDeltaFrobeniusNorm: Float
    public let meanGradientNorm: Float
    public let maxGradientNorm: Float

    public init(
        moduleName: String,
        parameterCount: Int,
        initialFrobeniusNorm: Float,
        finalFrobeniusNorm: Float,
        weightDeltaFrobeniusNorm: Float,
        meanGradientNorm: Float,
        maxGradientNorm: Float
    ) {
        self.moduleName = moduleName
        self.parameterCount = parameterCount
        self.initialFrobeniusNorm = initialFrobeniusNorm
        self.finalFrobeniusNorm = finalFrobeniusNorm
        self.weightDeltaFrobeniusNorm = weightDeltaFrobeniusNorm
        self.meanGradientNorm = meanGradientNorm
        self.maxGradientNorm = maxGradientNorm
    }
}

/// Complete telemetry container for `Phase 5.1 Audit 2` stored inside `.safetensors` metadata.
public struct RelationalAuditTelemetry: Codable, Sendable, Equatable {
    public var modules: [AuditModuleMetrics]
    public var totalTrainingSteps: Int
    public var finalTotalLoss: Float
    public var finalClsLoss: Float
    public var finalSupConLoss: Float
    public var finalCfLoss: Float
    public var finalMetricLoss: Float

    public init(
        modules: [AuditModuleMetrics] = [],
        totalTrainingSteps: Int = 0,
        finalTotalLoss: Float = 0.0,
        finalClsLoss: Float = 0.0,
        finalSupConLoss: Float = 0.0,
        finalCfLoss: Float = 0.0,
        finalMetricLoss: Float = 0.0
    ) {
        self.modules = modules
        self.totalTrainingSteps = totalTrainingSteps
        self.finalTotalLoss = finalTotalLoss
        self.finalClsLoss = finalClsLoss
        self.finalSupConLoss = finalSupConLoss
        self.finalCfLoss = finalCfLoss
        self.finalMetricLoss = finalMetricLoss
    }

    public func serialize() -> String {
        guard let data = try? JSONEncoder().encode(self),
              let str = String(data: data, encoding: .utf8) else {
            return ""
        }
        return str
    }

    public static func deserialize(_ str: String) -> RelationalAuditTelemetry {
        guard !str.isEmpty,
              let data = str.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(RelationalAuditTelemetry.self, from: data) else {
            return RelationalAuditTelemetry()
        }
        return decoded
    }
}

/// Container for calibrated linear classification heads serialized alongside model weights (`Phase 5.1 Audit 5`).
public struct RelationalCalibratedHeads: Codable, Sendable, Equatable {
    public var archetypeT4: SoftmaxClassifierHead?
    public var topologyT4: SoftmaxClassifierHead?
    public var archetypeT2: SoftmaxClassifierHead?
    public var topologyT2: SoftmaxClassifierHead?

    public init(
        archetypeT4: SoftmaxClassifierHead? = nil,
        topologyT4: SoftmaxClassifierHead? = nil,
        archetypeT2: SoftmaxClassifierHead? = nil,
        topologyT2: SoftmaxClassifierHead? = nil
    ) {
        self.archetypeT4 = archetypeT4
        self.topologyT4 = topologyT4
        self.archetypeT2 = archetypeT2
        self.topologyT2 = topologyT2
    }

    public func serialize() -> String {
        guard let data = try? JSONEncoder().encode(self),
              let str = String(data: data, encoding: .utf8) else {
            return ""
        }
        return str
    }

    public static func deserialize(_ str: String) -> RelationalCalibratedHeads {
        guard !str.isEmpty,
              let data = str.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(RelationalCalibratedHeads.self, from: data) else {
            return RelationalCalibratedHeads()
        }
        return decoded
    }
}

/// Output of pure-neural Dual-Stream Relational Inference taking only `(senderTokenIDs: [Int], receiverTokenIDs: [Int])` (`Phase 5.1 Audit 1`).
public struct RelationalInferenceOutput: Sendable, Equatable {
    public let zRel: [Float]          // [1024] = [z_LL^rel(512) ; z_HH^rel(512)]
    public let zContent: [Float]      // [512] auxiliary topic/content latent vector
    public let archetypeProbsT4: [Float]
    public let topologyProbsT4: [Float]
    public let archetypeProbsT2: [Float]
    public let topologyProbsT2: [Float]
    public let predictedArchetypeT4: RelationalArchetype
    public let predictedTopologyT4: RelationalTopology
    public let predictedArchetypeT2: RelationalArchetype
    public let predictedTopologyT2: RelationalTopology
    public let pentagon: RelationalPentagonScores
    public let sLL: Float
    public let sHH: Float
    public let rHH: Float
    public let dLL: Float
}

/// Weights for the Phase 5.1 Dual-Stream Subband Bidirectional Cross-Attention & Latent Disentanglement Engine (`Section 2 & 3`).
public struct DualStreamRelationalWeights: Sendable, Equatable {
    public static let coherenceDim: Int = 16
    public static let canonicalSeqLen: Int = 288
    public static let projBlocks: Int = 5
    public static let frameStride: Int = 4

    public let vocabSize: Int
    public let dim: Int
    public let latentDim: Int

    // Disentanglement gate & universal topic/echo embeddings
    public var alphaRel: [Float]     // [vocabSize]
    public var eEcho: [Float]        // [dim]
    public var eTopic: [Float]       // [dim]

    // Learned Sequence Subband Filters & Salience Queries (Bidirectional_CrossAttention_Layer)
    public var convRelLL1: Conv1D3Weights // [3, dim, dim]
    public var convRelLL2: Conv1D3Weights // [3, dim, dim]
    public var convRelHH1: Conv1D3Weights // [3, 2 * dim, dim]
    public var convRelHH2: Conv1D3Weights // [3, dim, dim]
    public var wPoolLL: [Float]      // [dim, 2]
    public var wPoolHH: [Float]      // [dim, 2]

    // Bidirectional_CrossAttention_Layer (LL & HH)
    public var wQLL: [Float]         // [dim, dim]
    public var bQLL: [Float]         // [dim]
    public var wKLL: [Float]         // [dim, dim]
    public var bKLL: [Float]         // [dim]
    public var wVLL: [Float]         // [dim, dim]
    public var bVLL: [Float]         // [dim]
    public var wCohLL: [Float]       // [16, dim]
    public var bCohLL: [Float]       // [dim]

    public var wQHH: [Float]         // [dim, dim]
    public var bQHH: [Float]         // [dim]
    public var wKHH: [Float]         // [dim, dim]
    public var bKHH: [Float]         // [dim]
    public var wVHH: [Float]         // [dim, dim]
    public var bVHH: [Float]         // [dim]
    public var wCohHH: [Float]       // [16, dim]
    public var bCohHH: [Float]       // [dim]

    // Latent_Projection_Layer (z_LL^rel ∈ R^512, z_HH^rel ∈ R^512, z_content ∈ R^512)
    public var wProjLL: [Float]      // [5 * dim, 512]
    public var bProjLL: [Float]      // [512]
    public var normLLGamma: [Float]  // [512]
    public var normLLBeta: [Float]   // [512]

    public var wProjHH: [Float]      // [5 * dim, 512]
    public var bProjHH: [Float]      // [512]
    public var normHHGamma: [Float]  // [512]
    public var normHHBeta: [Float]   // [512]

    public var wContent: [Float]     // [2 * dim, 512]
    public var bContent: [Float]     // [512]

    // End-to-End Linear Classification Heads on z_rel ∈ R^1024
    public var wArchT4: [Float]      // [1024, 6]
    public var bArchT4: [Float]      // [6]
    public var wTopoT4: [Float]      // [1024, 5]
    public var bTopoT4: [Float]      // [5]
    public var wArchT2: [Float]      // [1024, 6]
    public var bArchT2: [Float]      // [6]
    public var wTopoT2: [Float]      // [1024, 5]
    public var bTopoT2: [Float]      // [5]

    public var calibratedHeads: RelationalCalibratedHeads
    public var auditTelemetry: RelationalAuditTelemetry

    public static func makeDefaultSequenceFilters(dim: Int, seed: UInt64 = 2026_0501) -> (
        convRelLL1: Conv1D3Weights,
        convRelLL2: Conv1D3Weights,
        convRelHH1: Conv1D3Weights,
        convRelHH2: Conv1D3Weights,
        wPoolLL: [Float],
        wPoolHH: [Float]
    ) {
        var rng = SplitMix64(seed: seed ^ UInt64(dim * 131))
        let cLL1 = Conv1D3Weights.makeRandom(inDim: dim, outDim: dim, rng: &rng)
        let cLL2 = Conv1D3Weights.makeRandom(inDim: dim, outDim: dim, rng: &rng)
        let cHH1 = Conv1D3Weights.makeRandom(inDim: 2 * dim, outDim: dim, rng: &rng)
        let cHH2 = Conv1D3Weights.makeRandom(inDim: dim, outDim: dim, rng: &rng)
        let pBound = sqrtf(6.0 / Float(dim + 2))
        var pLL = [Float](repeating: 0.0, count: dim * 2)
        var pHH = [Float](repeating: 0.0, count: dim * 2)
        for i in 0..<(dim * 2) {
            pLL[i] = rng.nextSignedUniform(scale: pBound)
            pHH[i] = rng.nextSignedUniform(scale: pBound)
        }
        return (cLL1, cLL2, cHH1, cHH2, pLL, pHH)
    }

    public init(
        vocabSize: Int,
        dim: Int,
        latentDim: Int = SWTHawkShape.relationalLatentDim,
        alphaRel: [Float],
        eEcho: [Float],
        eTopic: [Float],
        convRelLL1: Conv1D3Weights? = nil,
        convRelLL2: Conv1D3Weights? = nil,
        convRelHH1: Conv1D3Weights? = nil,
        convRelHH2: Conv1D3Weights? = nil,
        wPoolLL: [Float]? = nil,
        wPoolHH: [Float]? = nil,
        wQLL: [Float],
        bQLL: [Float],
        wKLL: [Float],
        bKLL: [Float],
        wVLL: [Float],
        bVLL: [Float],
        wCohLL: [Float],
        bCohLL: [Float],
        wQHH: [Float],
        bQHH: [Float],
        wKHH: [Float],
        bKHH: [Float],
        wVHH: [Float],
        bVHH: [Float],
        wCohHH: [Float],
        bCohHH: [Float],
        wProjLL: [Float],
        bProjLL: [Float],
        normLLGamma: [Float],
        normLLBeta: [Float],
        wProjHH: [Float],
        bProjHH: [Float],
        normHHGamma: [Float],
        normHHBeta: [Float],
        wContent: [Float],
        bContent: [Float],
        wArchT4: [Float],
        bArchT4: [Float],
        wTopoT4: [Float],
        bTopoT4: [Float],
        wArchT2: [Float],
        bArchT2: [Float],
        wTopoT2: [Float],
        bTopoT2: [Float],
        calibratedHeads: RelationalCalibratedHeads = RelationalCalibratedHeads(),
        auditTelemetry: RelationalAuditTelemetry = RelationalAuditTelemetry()
    ) {
        let defSeq = Self.makeDefaultSequenceFilters(dim: dim)
        self.vocabSize = vocabSize
        self.dim = dim
        self.latentDim = latentDim
        self.alphaRel = alphaRel
        self.eEcho = eEcho
        self.eTopic = eTopic
        self.convRelLL1 = convRelLL1 ?? defSeq.convRelLL1
        self.convRelLL2 = convRelLL2 ?? defSeq.convRelLL2
        self.convRelHH1 = convRelHH1 ?? defSeq.convRelHH1
        self.convRelHH2 = convRelHH2 ?? defSeq.convRelHH2
        self.wPoolLL = (wPoolLL?.count == dim * 2) ? wPoolLL! : defSeq.wPoolLL
        self.wPoolHH = (wPoolHH?.count == dim * 2) ? wPoolHH! : defSeq.wPoolHH
        self.wQLL = wQLL
        self.bQLL = bQLL
        self.wKLL = wKLL
        self.bKLL = bKLL
        self.wVLL = wVLL
        self.bVLL = bVLL
        self.wCohLL = wCohLL
        self.bCohLL = bCohLL
        self.wQHH = wQHH
        self.bQHH = bQHH
        self.wKHH = wKHH
        self.bKHH = bKHH
        self.wVHH = wVHH
        self.bVHH = bVHH
        self.wCohHH = wCohHH
        self.bCohHH = bCohHH
        self.wProjLL = wProjLL
        self.bProjLL = bProjLL
        self.normLLGamma = normLLGamma
        self.normLLBeta = normLLBeta
        self.wProjHH = wProjHH
        self.bProjHH = bProjHH
        self.normHHGamma = normHHGamma
        self.normHHBeta = normHHBeta
        self.wContent = wContent
        self.bContent = bContent
        self.wArchT4 = wArchT4
        self.bArchT4 = bArchT4
        self.wTopoT4 = wTopoT4
        self.bTopoT4 = bTopoT4
        self.wArchT2 = wArchT2
        self.bArchT2 = bArchT2
        self.wTopoT2 = wTopoT2
        self.bTopoT2 = bTopoT2
        self.calibratedHeads = calibratedHeads
        self.auditTelemetry = auditTelemetry
    }

    public var crossAttentionParameterCount: Int {
        let seqCount = convRelLL1.parameterCount + convRelLL2.parameterCount + convRelHH1.parameterCount + convRelHH2.parameterCount + wPoolLL.count + wPoolHH.count
        return seqCount + eEcho.count + eTopic.count + wQLL.count + bQLL.count + wKLL.count + bKLL.count + wVLL.count + bVLL.count + wCohLL.count + bCohLL.count + wQHH.count + bQHH.count + wKHH.count + bKHH.count + wVHH.count + bVHH.count + wCohHH.count + bCohHH.count
    }

    public var latentProjectionParameterCount: Int {
        return wProjLL.count + bProjLL.count + normLLGamma.count + normLLBeta.count + wProjHH.count + bProjHH.count + normHHGamma.count + normHHBeta.count + wContent.count + bContent.count + wArchT4.count + bArchT4.count + wTopoT4.count + bTopoT4.count + wArchT2.count + bArchT2.count + wTopoT2.count + bTopoT2.count
    }

    public var parameterCount: Int {
        return crossAttentionParameterCount + latentProjectionParameterCount
    }

    public static func makeRandom(
        vocabSize: Int,
        dim: Int,
        latentDim: Int = SWTHawkShape.relationalLatentDim,
        alphaRel: [Float]? = nil,
        rng: inout SplitMix64
    ) -> DualStreamRelationalWeights {
        let half = latentDim / 2
        let cohDim = Self.coherenceDim
        let pBlocks = Self.projBlocks
        let dBound = sqrtf(6.0 / Float(dim + dim))
        let cohBound = sqrtf(6.0 / Float(cohDim + dim))
        let projBound = sqrtf(6.0 / Float(pBlocks * dim + half))
        let contBound = sqrtf(6.0 / Float(2 * dim + half))
        let clsBound = sqrtf(6.0 / Float(latentDim + 6))
        let poolBound = sqrtf(6.0 / Float(dim + 2))

        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                a[i] = rng.nextSignedUniform(scale: bound)
            }
            return a
        }

        let defaultAlpha: [Float] = if let a = alphaRel, a.count == vocabSize {
            a
        } else {
            (0..<vocabSize).map { $0 < TextVocabulary.specialCount ? 0.0 : 1.0 }
        }

        let cLL1 = Conv1D3Weights.makeRandom(inDim: dim, outDim: dim, rng: &rng)
        let cLL2 = Conv1D3Weights.makeRandom(inDim: dim, outDim: dim, rng: &rng)
        let cHH1 = Conv1D3Weights.makeRandom(inDim: 2 * dim, outDim: dim, rng: &rng)
        let cHH2 = Conv1D3Weights.makeRandom(inDim: dim, outDim: dim, rng: &rng)
        let wPLL = randMat(dim * 2, poolBound)
        let wPHH = randMat(dim * 2, poolBound)

        return DualStreamRelationalWeights(
            vocabSize: vocabSize,
            dim: dim,
            latentDim: latentDim,
            alphaRel: defaultAlpha,
            eEcho: randMat(dim, 0.5),
            eTopic: randMat(dim, 0.5),
            convRelLL1: cLL1,
            convRelLL2: cLL2,
            convRelHH1: cHH1,
            convRelHH2: cHH2,
            wPoolLL: wPLL,
            wPoolHH: wPHH,
            wQLL: randMat(dim * dim, dBound),
            bQLL: [Float](repeating: 0.0, count: dim),
            wKLL: randMat(dim * dim, dBound),
            bKLL: [Float](repeating: 0.0, count: dim),
            wVLL: randMat(dim * dim, dBound),
            bVLL: [Float](repeating: 0.0, count: dim),
            wCohLL: randMat(cohDim * dim, cohBound),
            bCohLL: [Float](repeating: 0.0, count: dim),
            wQHH: randMat(dim * dim, dBound),
            bQHH: [Float](repeating: 0.0, count: dim),
            wKHH: randMat(dim * dim, dBound),
            bKHH: [Float](repeating: 0.0, count: dim),
            wVHH: randMat(dim * dim, dBound),
            bVHH: [Float](repeating: 0.0, count: dim),
            wCohHH: randMat(cohDim * dim, cohBound),
            bCohHH: [Float](repeating: 0.0, count: dim),
            wProjLL: randMat(pBlocks * dim * half, projBound),
            bProjLL: [Float](repeating: 0.0, count: half),
            normLLGamma: [Float](repeating: 1.0, count: half),
            normLLBeta: [Float](repeating: 0.0, count: half),
            wProjHH: randMat(pBlocks * dim * half, projBound),
            bProjHH: [Float](repeating: 0.0, count: half),
            normHHGamma: [Float](repeating: 1.0, count: half),
            normHHBeta: [Float](repeating: 0.0, count: half),
            wContent: randMat(2 * dim * half, contBound),
            bContent: [Float](repeating: 0.0, count: half),
            wArchT4: randMat(latentDim * 6, clsBound),
            bArchT4: [Float](repeating: 0.0, count: 6),
            wTopoT4: randMat(latentDim * 5, clsBound),
            bTopoT4: [Float](repeating: 0.0, count: 5),
            wArchT2: randMat(latentDim * 6, clsBound),
            bArchT2: [Float](repeating: 0.0, count: 6),
            wTopoT2: randMat(latentDim * 5, clsBound),
            bTopoT2: [Float](repeating: 0.0, count: 5)
        )
    }
}

/// CPU & shared mathematical engine for Dual-Stream Subband Bidirectional Cross-Attention (`Phase 5.1 Section 2 & 3`).
public enum DualStreamRelationalEngine {

    /// Canonical subband parameters `(sLL, sHH, rHH)` for a given `(RelationalArchetype, RelationalTopology)` pair,
    /// used as the target metric geometry in `L_metric` (`Section 3`) so that neural activations in `z_rel`
    /// directly encode the 5-axis Relational Pentagon.
    public static func targetSubbandGeometry(
        archetype: RelationalArchetype,
        topology: RelationalTopology
    ) -> (sLL: Float, sHH: Float, rHH: Float) {
        switch (archetype, topology) {
        case (.trueSupporter, .deepAffection):
            return (sLL: 0.88, sHH: 0.88, rHH: 1.00)
        case (.trueSupporter, .benevolentGuardian):
            return (sLL: 0.85, sHH: 0.82, rHH: 0.96)
        case (.gachikoiSafe, .deepAffection):
            return (sLL: 0.90, sHH: 0.84, rHH: 1.20)
        case (.gachikoiRisk, .closeConfidant), (.gachikoiRisk, _):
            return (sLL: 0.72, sHH: -0.45, rHH: 1.42)
        case (.hiddenAggressor, .acquaintedObserver), (.hiddenAggressor, _):
            return (sLL: 0.12, sHH: -0.82, rHH: 1.28)
        case (.casualHype, .benevolentGuardian):
            return (sLL: 0.22, sHH: 0.78, rHH: 0.95)
        case (.casualHype, .acquaintedObserver), (.casualHype, _):
            return (sLL: 0.05, sHH: 0.42, rHH: 0.78)
        case (.indifferentTroll, .unconnectedStranger), (.indifferentTroll, _):
            return (sLL: -0.68, sHH: 0.02, rHH: 0.22)
        default:
            return (sLL: 0.50, sHH: 0.50, rHH: 1.00)
        }
    }

    /// Generates the deterministic 1024-D target prototype `z_target = [z_LL_S(256); z_LL_C(256); z_HH_S(256); z_HH_C(256)]`
    /// for `(archetype, topology)` such that after LayerNorm on `z_LL(512)` and `z_HH(512)`,
    /// `cos(z_LL_S, z_LL_C) == sLL`, `cos(z_HH_S, z_HH_C) == sHH`, and `||z_HH_C|| / ||z_HH_S|| == rHH`.
    public static func targetPrototype(
        archetype: RelationalArchetype,
        topology: RelationalTopology,
        latentDim: Int = SWTHawkShape.relationalLatentDim,
        isEarlyT2: Bool = false
    ) -> [Float] {
        let half = latentDim / 2
        let quarter = half / 2
        let geom = targetSubbandGeometry(archetype: archetype, topology: topology)
        let archIdx = RelationalArchetype.allCases.firstIndex(of: archetype) ?? 0
        let topoIdx = RelationalTopology.allCases.firstIndex(of: topology) ?? 0
        let phaseSalt: UInt64 = isEarlyT2 ? 0x0055_AA55_0055_AA55 : 0
        let classSalt = UInt64((archIdx + 1) * 131 + (topoIdx + 1) * 17) ^ phaseSalt

        func makeZeroMeanUnitBasis(seed: UInt64, count: Int) -> [Float] {
            var rng = SplitMix64(seed: seed)
            var v = [Float](repeating: 0.0, count: count)
            var sumVal: Float = 0.0
            for i in 0..<count {
                let val = rng.nextSignedUniform(scale: 1.0)
                v[i] = val
                sumVal += val
            }
            let meanVal = sumVal / Float(max(1, count))
            var sq: Float = 0.0
            for i in 0..<count {
                v[i] -= meanVal
                sq += v[i] * v[i]
            }
            let inv = 1.0 / sqrtf(max(sq, 1e-12))
            for i in 0..<count { v[i] *= inv }
            return v
        }

        // Construct orthogonal u1, u2 in R^quarter using Gram-Schmidt
        func makeCorrelatedPair(seed: UInt64, cosine: Float, normS: Float, normC: Float) -> ([Float], [Float]) {
            let u1 = makeZeroMeanUnitBasis(seed: seed, count: quarter)
            var u2 = makeZeroMeanUnitBasis(seed: seed ^ 0x5555_AAAA_5555_AAAA, count: quarter)
            var dot: Float = 0.0
            for i in 0..<quarter { dot += u1[i] * u2[i] }
            var sq: Float = 0.0
            for i in 0..<quarter {
                u2[i] -= dot * u1[i]
                sq += u2[i] * u2[i]
            }
            let inv = 1.0 / sqrtf(max(sq, 1e-12))
            for i in 0..<quarter { u2[i] *= inv }

            let c = max(-0.99, min(0.99, cosine))
            let s = sqrtf(max(0.0, 1.0 - c * c))
            var vecS = [Float](repeating: 0.0, count: quarter)
            var vecC = [Float](repeating: 0.0, count: quarter)
            for i in 0..<quarter {
                vecS[i] = u1[i] * normS
                vecC[i] = (c * u1[i] + s * u2[i]) * normC
            }
            return (vecS, vecC)
        }

        let targetHalfNorm = sqrtf(Float(half))
        let llSideNorm = targetHalfNorm / sqrtf(2.0)
        let (llS, llC) = makeCorrelatedPair(
            seed: 0x1000_0000_0000_0000 ^ classSalt,
            cosine: geom.sLL,
            normS: llSideNorm,
            normC: llSideNorm
        )

        let r = max(0.15, min(2.0, geom.rHH))
        let hhNormS = targetHalfNorm / sqrtf(1.0 + r * r)
        let hhNormC = r * hhNormS
        let (hhS, hhC) = makeCorrelatedPair(
            seed: 0x2000_0000_0000_0000 ^ classSalt,
            cosine: geom.sHH,
            normS: hhNormS,
            normC: hhNormC
        )

        return llS + llC + hhS + hhC
    }

    /// Computes the 5-axis Relational Pentagon and subband geometric quantities directly from `zRel ∈ R^1024`
    /// (and optional `zRelEarly ∈ R^1024` for temporal consistency).
    public static func computePentagonFromZRel(
        zRel: [Float],
        zRelEarly: [Float]? = nil
    ) -> (pentagon: RelationalPentagonScores, dLL: Float, dHH: Float, rHH: Float, sHH: Float, sLL: Float, temporalConsistencyLL: Float) {
        let latentDim = zRel.count
        let half = latentDim / 2
        let quarter = half / 2
        precondition(quarter > 0)

        var normLLSSq: Float = 0.0
        var normLLCSq: Float = 0.0
        var dotLL: Float = 0.0
        for i in 0..<quarter {
            let s = zRel[i]
            let c = zRel[quarter + i]
            normLLSSq += s * s
            normLLCSq += c * c
            dotLL += s * c
        }
        let normLLS = sqrtf(max(normLLSSq, 1e-12))
        let normLLC = sqrtf(max(normLLCSq, 1e-12))
        let sLL = max(-1.0, min(1.0, dotLL / (normLLS * normLLC + 1e-6)))
        let dLL = sqrtf(max(0.0, 2.0 * Float(half) * (1.0 - sLL)))

        func temporalCosine(_ early: [Float]) -> Float {
            var dotT: Float = 0.0
            var sqEarly: Float = 0.0
            var sqCur: Float = 0.0
            for i in 0..<half {
                dotT += early[i] * zRel[i]
                sqEarly += early[i] * early[i]
                sqCur += zRel[i] * zRel[i]
            }
            return max(-1.0, min(1.0, dotT / (sqrtf(max(sqEarly, 1e-12)) * sqrtf(max(sqCur, 1e-12)) + 1e-6)))
        }
        let temporalConsistencyLL: Float = if let early = zRelEarly, early.count == latentDim {
            temporalCosine(early)
        } else {
            sLL
        }

        var normHHSSq: Float = 0.0
        var normHHCSq: Float = 0.0
        var dotHH: Float = 0.0
        for i in 0..<quarter {
            let s = zRel[half + i]
            let c = zRel[half + quarter + i]
            normHHSSq += s * s
            normHHCSq += c * c
            dotHH += s * c
        }
        let normHHS = sqrtf(max(normHHSSq, 1e-12))
        let normHHC = sqrtf(max(normHHCSq, 1e-12))
        let rHH = normHHC / (normHHS + 1e-6)
        let sHH = max(-1.0, min(1.0, dotHH / (normHHS * normHHC + 1e-6)))
        let dHH = sqrtf(max(0.0, normHHSSq + normHHCSq - 2.0 * normHHS * normHHC * sHH))

        var envDiffSq: Float = 0.0
        for i in 0..<quarter {
            let d = abs(zRel[half + i]) - abs(zRel[half + quarter + i])
            envDiffSq += d * d
        }
        let envDiffNorm = sqrtf(envDiffSq)

        let scoreCtx = 100.0 * max(0.0, min(1.0, 0.5 * (sLL + 1.0)))
        let rDiff = rHH - 1.0
        let rawEmp = 0.5 * (sHH + 1.0) * expf(-0.4 * rDiff * rDiff)
        let scoreEmp = 100.0 * max(0.0, min(1.0, rawEmp))
        let rawSync = 1.0 - (envDiffNorm / (normHHS + normHHC + 1e-6))
        let scoreSync = 100.0 * max(0.0, min(1.0, rawSync))
        let negSpikePenalty = max(0.0, -sHH)
        let excessEnergyPenalty = 0.4 * max(0.0, rHH - 1.3)
        let scoreStab = 100.0 * max(0.0, min(1.0, 1.0 - negSpikePenalty - excessEnergyPenalty))
        let consistencyFactor = max(0.45, min(1.0, max(temporalConsistencyLL, sLL)))
        let engagementScale = min(1.0, max(0.25, rHH / 0.40))
        let rawDev = consistencyFactor * engagementScale * (0.6 * (scoreCtx / 100.0) + 0.4 * (scoreStab / 100.0))
        let scoreDev = 100.0 * max(0.0, min(1.0, rawDev))

        let pentagon = RelationalPentagonScores(
            contextDepth: scoreCtx,
            empathicResonance: scoreEmp,
            communicativeHarmony: scoreSync,
            boundaryStability: scoreStab,
            longTermViability: scoreDev
        )
        return (pentagon, dLL, dHH, rHH, sHH, sLL, temporalConsistencyLL)
    }

    /// Predicts `(archetype, topology)` and probability distributions from `zRel ∈ R^1024` using the learned heads.
    public static func classifyFromZRel(
        zRel: [Float],
        isEarlyT2: Bool,
        weights: DualStreamRelationalWeights
    ) -> (archetype: RelationalArchetype, topology: RelationalTopology, archProbs: [Float], topoProbs: [Float]) {
        let archCases = RelationalArchetype.allCases
        let topoCases = RelationalTopology.allCases

        if isEarlyT2,
           let hArch = weights.calibratedHeads.archetypeT2,
           let hTopo = weights.calibratedHeads.topologyT2,
           hArch.featureDim == zRel.count,
           hTopo.featureDim == zRel.count {
            let aProbs = hArch.probabilities(zRel)
            let tProbs = hTopo.probabilities(zRel)
            let aIdx = hArch.predict(zRel)
            let tIdx = hTopo.predict(zRel)
            return (archCases[aIdx], topoCases[tIdx], aProbs, tProbs)
        }
        if !isEarlyT2,
           let hArch = weights.calibratedHeads.archetypeT4,
           let hTopo = weights.calibratedHeads.topologyT4,
           hArch.featureDim == zRel.count,
           hTopo.featureDim == zRel.count {
            let aProbs = hArch.probabilities(zRel)
            let tProbs = hTopo.probabilities(zRel)
            let aIdx = hArch.predict(zRel)
            let tIdx = hTopo.predict(zRel)
            return (archCases[aIdx], topoCases[tIdx], aProbs, tProbs)
        }

        let wArch = isEarlyT2 ? weights.wArchT2 : weights.wArchT4
        let bArch = isEarlyT2 ? weights.bArchT2 : weights.bArchT4
        let wTopo = isEarlyT2 ? weights.wTopoT2 : weights.wTopoT4
        let bTopo = isEarlyT2 ? weights.bTopoT2 : weights.bTopoT4

        let archLogits = SWTHawkAutoencoder.linear(
            x: zRel, w: wArch, b: bArch, rows: 1, inDim: weights.latentDim, outDim: archCases.count
        )
        let topoLogits = SWTHawkAutoencoder.linear(
            x: zRel, w: wTopo, b: bTopo, rows: 1, inDim: weights.latentDim, outDim: topoCases.count
        )
        let aProbs = SoftmaxClassifierHead.softmax(archLogits)
        let tProbs = SoftmaxClassifierHead.softmax(topoLogits)
        var bestA = 0
        for i in 1..<aProbs.count where aProbs[i] > aProbs[bestA] { bestA = i }
        var bestT = 0
        for i in 1..<tProbs.count where tProbs[i] > tProbs[bestT] { bestT = i }
        return (archCases[bestA], topoCases[bestT], aProbs, tProbs)
    }

    /// Alternating sign vector `[+1, -1, +1, -1, ...]` of length `dim` used for directed 2-gram Haar wavelet coherence.
    public static func alternatingSigns(dim: Int) -> [Float] {
        return (0..<dim).map { ($0 & 1) == 0 ? 1.0 : -1.0 }
    }

    /// CPU forward pass of the Dual-Stream SWT-Griffin + Subband Bidirectional Cross-Attention Encoder.
    /// Matches `MLXDualStreamRelationalModule.forward` to floating-point precision.
    public static func forwardBatchCPU(
        senderTokens: [Int],
        receiverTokens: [Int],
        batch: Int,
        time: Int,
        weights: SWTHawkWeights
    ) -> (phiLL: [Float], phiHH: [Float], zRel: [Float], zContent: [Float]) {
        let d = weights.dim
        let lDim = weights.relational.latentDim
        let half = lDim / 2
        let cohDim = DualStreamRelationalWeights.coherenceDim
        let vocabSize = weights.vocabSize
        let rel = weights.relational
        let bt = batch * time
        precondition(senderTokens.count == bt && receiverTokens.count == bt)

        // 1. Valid masks, pair masks, and normalized temporal pooling weights (mean, early, late)
        var maskS = [Float](repeating: 0.0, count: bt)
        var maskC = [Float](repeating: 0.0, count: bt)
        var pairMaskS = [Float](repeating: 0.0, count: bt)
        var pairMaskC = [Float](repeating: 0.0, count: bt)
        var wMeanS = [Float](repeating: 0.0, count: bt)
        var wMeanC = [Float](repeating: 0.0, count: bt)
        var wEarlyC = [Float](repeating: 0.0, count: bt)
        var wLateC = [Float](repeating: 0.0, count: bt)
        var activeLenS = [Float](repeating: 1.0, count: batch)
        var activeLenC = [Float](repeating: 1.0, count: batch)

        for b in 0..<batch {
            let base = b * time
            var lenS: Float = 0.0
            var lenC: Float = 0.0
            for t in 0..<time {
                let sValid: Float = senderTokens[base + t] > 0 ? 1.0 : 0.0
                let cValid: Float = receiverTokens[base + t] > 0 ? 1.0 : 0.0
                maskS[base + t] = sValid
                maskC[base + t] = cValid
                lenS += sValid
                lenC += cValid
                if t > 0 {
                    pairMaskS[base + t] = sValid * maskS[base + t - 1]
                    pairMaskC[base + t] = cValid * maskC[base + t - 1]
                }
            }
            activeLenS[b] = max(1.0, lenS)
            activeLenC[b] = max(1.0, lenC)

            var cumC: Float = 0.0
            var sumPairS: Float = 0.0
            var sumPairC: Float = 0.0
            var sumEarlyC: Float = 0.0
            var sumLateC: Float = 0.0

            for t in 0..<time {
                let idx = base + t
                cumC += maskC[idx]
                let tauC = (cumC / activeLenC[b]) * maskC[idx]
                let pS = pairMaskS[idx]
                let pC = pairMaskC[idx]
                let omC = max(0.0, 1.0 - tauC)
                let eC = pC * omC * omC
                let lC = pC * tauC * tauC * tauC
                wMeanS[idx] = pS
                wMeanC[idx] = pC
                wEarlyC[idx] = eC
                wLateC[idx] = lC
                sumPairS += pS
                sumPairC += pC
                sumEarlyC += eC
                sumLateC += lC
            }
            let invPS = 1.0 / max(sumPairS, 1e-6)
            let invPC = 1.0 / max(sumPairC, 1e-6)
            let invEC = 1.0 / max(sumEarlyC, 1e-6)
            let invLC = 1.0 / max(sumLateC, 1e-6)
            for t in 0..<time {
                let idx = base + t
                wMeanS[idx] *= invPS
                wMeanC[idx] *= invPC
                wEarlyC[idx] *= invEC
                wLateC[idx] *= invLC
            }
        }

        // 2. Raw embeddings & Multi-Scale Haar Wavelet Cross-Coherence (over active tokens and cohProjDim = min(32, D))
        let cohProjDim = min(32, d)
        var eRawS = [Float](repeating: 0.0, count: bt * d)
        var eRawC = [Float](repeating: 0.0, count: bt * d)
        var eCohS = [Float](repeating: 0.0, count: bt * cohProjDim)
        var eCohC = [Float](repeating: 0.0, count: bt * cohProjDim)
        for i in 0..<bt {
            let tokS = max(0, min(vocabSize - 1, senderTokens[i]))
            let tokC = max(0, min(vocabSize - 1, receiverTokens[i]))
            let mS = maskS[i]
            let mC = maskC[i]
            let offS = tokS * d
            let offC = tokC * d
            let dst = i * d
            let cDst = i * cohProjDim
            for c in 0..<d {
                let vS = weights.tokenEmbedding[offS + c] * mS
                let vC = weights.tokenEmbedding[offC + c] * mC
                eRawS[dst + c] = vS
                eRawC[dst + c] = vC
                if c < cohProjDim {
                    eCohS[cDst + c] = vS
                    eCohC[cDst + c] = vC
                }
            }
        }

        let (ca1RawS, cd1RawS, ca2RawS, _) = SWTFrequencyBranch.decompose(x: eCohS, batch: batch, time: time, dim: cohProjDim)
        let (ca1RawC, cd1RawC, ca2RawC, _) = SWTFrequencyBranch.decompose(x: eCohC, batch: batch, time: time, dim: cohProjDim)
        let altSign = alternatingSigns(dim: cohProjDim)

        var w1NormS = [Float](repeating: 0.0, count: bt * cohProjDim)
        var w1NormC = [Float](repeating: 0.0, count: bt * cohProjDim)
        var w2NormS = [Float](repeating: 0.0, count: bt * cohProjDim)
        var w2NormC = [Float](repeating: 0.0, count: bt * cohProjDim)
        var d1NormS = [Float](repeating: 0.0, count: bt * cohProjDim)
        var d1NormC = [Float](repeating: 0.0, count: bt * cohProjDim)

        for i in 0..<bt {
            let pS = pairMaskS[i]
            let pC = pairMaskC[i]
            let off = i * cohProjDim
            var sq1S: Float = 0.0
            var sq1C: Float = 0.0
            var sq2S: Float = 0.0
            var sq2C: Float = 0.0
            var sqDS: Float = 0.0
            var sqDC: Float = 0.0
            for c in 0..<cohProjDim {
                let sgn = altSign[c]
                let v1S = (ca1RawS[off + c] + 0.75 * cd1RawS[off + c] * sgn) * pS
                let v1C = (ca1RawC[off + c] + 0.75 * cd1RawC[off + c] * sgn) * pC
                let v2S = ca2RawS[off + c] * pS
                let v2C = ca2RawC[off + c] * pC
                let vDS = cd1RawS[off + c] * pS
                let vDC = cd1RawC[off + c] * pC
                w1NormS[off + c] = v1S
                w1NormC[off + c] = v1C
                w2NormS[off + c] = v2S
                w2NormC[off + c] = v2C
                d1NormS[off + c] = vDS
                d1NormC[off + c] = vDC
                sq1S += v1S * v1S
                sq1C += v1C * v1C
                sq2S += v2S * v2S
                sq2C += v2C * v2C
                sqDS += vDS * vDS
                sqDC += vDC * vDC
            }
            let inv1S = 1.0 / sqrtf(max(sq1S, 1e-12))
            let inv1C = 1.0 / sqrtf(max(sq1C, 1e-12))
            let inv2S = 1.0 / sqrtf(max(sq2S, 1e-12))
            let inv2C = 1.0 / sqrtf(max(sq2C, 1e-12))
            let invDS = 1.0 / sqrtf(max(sqDS, 1e-12))
            let invDC = 1.0 / sqrtf(max(sqDC, 1e-12))
            for c in 0..<cohProjDim {
                w1NormS[off + c] *= inv1S
                w1NormC[off + c] *= inv1C
                w2NormS[off + c] *= inv2S
                w2NormC[off + c] *= inv2C
                d1NormS[off + c] *= invDS
                d1NormC[off + c] *= invDC
            }
        }

        var gTokenS = [Float](repeating: 0.0, count: bt)
        var gTokenC = [Float](repeating: 0.0, count: bt)
        var psiCoh = [Float](repeating: 0.0, count: batch * cohDim)
        var gBarBatch = [Float](repeating: 0.0, count: batch)

        for b in 0..<batch {
            let bTimeOff = b * time
            let bMatOff = b * time * cohProjDim
            let actS = min(time, max(2, Int(activeLenS[b]) + 2))
            let actC = min(time, max(2, Int(activeLenC[b]) + 2))
            var w1S_T = [Float](repeating: 0.0, count: cohProjDim * actS)
            var w2S_T = [Float](repeating: 0.0, count: cohProjDim * actS)
            var d1S_T = [Float](repeating: 0.0, count: cohProjDim * actS)
            for s in 0..<actS {
                let src = bMatOff + s * cohProjDim
                for c in 0..<cohProjDim {
                    w1S_T[c * actS + s] = w1NormS[src + c]
                    w2S_T[c * actS + s] = w2NormS[src + c]
                    d1S_T[c * actS + s] = d1NormS[src + c]
                }
            }
            let slice1C = Array(w1NormC[bMatOff..<(bMatOff + actC * cohProjDim)])
            let slice2C = Array(w2NormC[bMatOff..<(bMatOff + actC * cohProjDim)])
            let sliceDC = Array(d1NormC[bMatOff..<(bMatOff + actC * cohProjDim)])
            var sim1 = [Float](repeating: 0.0, count: actC * actS)
            var sim2 = [Float](repeating: 0.0, count: actC * actS)
            var simD = [Float](repeating: 0.0, count: actC * actS)
            SWTHawkAutoencoder.gemm(slice1C, w1S_T, &sim1, m: actC, n: actS, k: cohProjDim)
            SWTHawkAutoencoder.gemm(slice2C, w2S_T, &sim2, m: actC, n: actS, k: cohProjDim)
            SWTHawkAutoencoder.gemm(sliceDC, d1S_T, &simD, m: actC, n: actS, k: cohProjDim)

            var c1C = [Float](repeating: 0.0, count: time)
            var c2C = [Float](repeating: 0.0, count: time)
            var cDC = [Float](repeating: 0.0, count: time)
            var c1S = [Float](repeating: 0.0, count: time)
            var c2S = [Float](repeating: 0.0, count: time)
            var cDS = [Float](repeating: 0.0, count: time)

            for t in 0..<actC {
                var m1: Float = 0.0
                var m2: Float = 0.0
                var mD: Float = 0.0
                let rowOff = t * actS
                for s in 0..<actS {
                    let v1 = sim1[rowOff + s]
                    let v2 = sim2[rowOff + s]
                    let vD = simD[rowOff + s]
                    if v1 > m1 { m1 = v1 }
                    if v2 > m2 { m2 = v2 }
                    if vD > mD { mD = vD }
                    if v1 > c1S[s] { c1S[s] = v1 }
                    if v2 > c2S[s] { c2S[s] = v2 }
                    if vD > cDS[s] { cDS[s] = vD }
                }
                c1C[t] = m1 * pairMaskC[bTimeOff + t]
                c2C[t] = m2 * pairMaskC[bTimeOff + t]
                cDC[t] = mD * pairMaskC[bTimeOff + t]
            }
            for s in 0..<actS {
                c1S[s] *= pairMaskS[bTimeOff + s]
                c2S[s] *= pairMaskS[bTimeOff + s]
                cDS[s] *= pairMaskS[bTimeOff + s]
            }

            var gPairC = [Float](repeating: 0.0, count: time)
            var gPairS = [Float](repeating: 0.0, count: time)
            for t in 0..<time {
                gPairC[t] = min(1.0, max(0.0, (c1C[t] - 0.72) / 0.28)) * pairMaskC[bTimeOff + t]
                gPairS[t] = min(1.0, max(0.0, (c1S[t] - 0.72) / 0.28)) * pairMaskS[bTimeOff + t]
            }
            for t in 0..<time {
                let nextC = (t + 1 < time) ? gPairC[t + 1] : 0.0
                let nextS = (t + 1 < time) ? gPairS[t + 1] : 0.0
                gTokenC[bTimeOff + t] = max(gPairC[t], nextC) * maskC[bTimeOff + t]
                gTokenS[bTimeOff + t] = max(gPairS[t], nextS) * maskS[bTimeOff + t]
            }

            // Compute 16 Wavelet Cross-Coherence statistics ψ_coh(S, C)
            var s0: Float = 0.0, s1: Float = 0.0, s2: Float = 0.0, s3: Float = 0.0
            var s4: Float = 0.0, s5: Float = 0.0, s6: Float = 0.0, s7: Float = 0.0
            var s8: Float = 0.0, s9: Float = 0.0, s10: Float = 0.0, s11: Float = 0.0
            var s12: Float = 0.0
            for t in 0..<time {
                let wmC = wMeanC[bTimeOff + t]
                let wmS = wMeanS[bTimeOff + t]
                let weC = wEarlyC[bTimeOff + t]
                let wlC = wLateC[bTimeOff + t]
                s0 += wmC * c1C[t]
                s1 += wmS * c1S[t]
                s2 += wmC * c2C[t]
                s3 += wmS * c2S[t]
                s4 += wmC * cDC[t]
                s5 += wmS * cDS[t]
                s6 += wmC * gTokenC[bTimeOff + t]
                s7 += wmS * gTokenS[bTimeOff + t]
                s8 += weC * gTokenC[bTimeOff + t]
                s9 += wlC * gTokenC[bTimeOff + t]
                s10 += wmC * min(1.0, max(0.0, (c1C[t] - 0.90) / 0.10))
                s11 += wmS * min(1.0, max(0.0, (c1S[t] - 0.90) / 0.10))
                s12 += wmC * min(1.0, max(0.0, (c2C[t] - 0.80) / 0.20))
            }
            let s13 = min(2.0, activeLenC[b] / activeLenS[b])
            let s14 = min(1.0, activeLenC[b] / 160.0)
            let s15 = min(1.0, activeLenS[b] / 160.0)
            gBarBatch[b] = s6

            let pOff = b * cohDim
            let vals: [Float] = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15]
            for k in 0..<cohDim {
                psiCoh[pOff + k] = vals[k]
            }
        }

        // 3. Disentangled Relational Input Streams E_rel^S, E_rel^C & Content Stream Input
        let pe = HawkConditionalDecoder.positionalEncoding(time: time, dim: d)
        var eRelS = [Float](repeating: 0.0, count: bt * d)
        var eRelC = [Float](repeating: 0.0, count: bt * d)
        var uRawContent = [Float](repeating: 0.0, count: batch * 2 * d)
        let rootD = sqrtf(Float(d))
        var echoSq: Float = 0.0
        var topicSq: Float = 0.0
        for c in 0..<d {
            echoSq += rel.eEcho[c] * rel.eEcho[c]
            topicSq += rel.eTopic[c] * rel.eTopic[c]
        }
        let scaleEcho = rootD / sqrtf(max(echoSq, 1e-12))
        let scaleTopic = 0.12 * rootD / sqrtf(max(topicSq, 1e-12))

        for b in 0..<batch {
            let bTimeOff = b * time
            let contOff = b * 2 * d
            let maxAct = min(time, max(Int(activeLenS[b]), Int(activeLenC[b])) + 2)
            for t in 0..<maxAct {
                let idx = bTimeOff + t
                let tokS = max(0, min(vocabSize - 1, senderTokens[idx]))
                let tokC = max(0, min(vocabSize - 1, receiverTokens[idx]))
                let aS = tokS < rel.alphaRel.count ? rel.alphaRel[tokS] : 1.0
                let aC = tokC < rel.alphaRel.count ? rel.alphaRel[tokC] : 1.0
                let gS = gTokenS[idx]
                let gC = gTokenC[idx]
                let mS = maskS[idx]
                let mC = maskC[idx]
                let wmS = wMeanS[idx]
                let wmC = wMeanC[idx]
                let peOff = t * d
                let dst = idx * d
                var sqS: Float = 0.0
                var sqC: Float = 0.0
                for c in 0..<d {
                    let vs = eRawS[dst + c]
                    let vc = eRawC[dst + c]
                    sqS += vs * vs
                    sqC += vc * vc
                }
                let scaleS = (mS > 0.0) ? (rootD / sqrtf(max(sqS, 1e-12))) : 0.0
                let scaleC = (mC > 0.0) ? (rootD / sqrtf(max(sqC, 1e-12))) : 0.0

                for c in 0..<d {
                    let rawValS = eRawS[dst + c]
                    let rawValC = eRawC[dst + c]
                    let normValS = rawValS * scaleS
                    let normValC = rawValC * scaleC
                    let echoVal = rel.eEcho[c] * scaleEcho
                    let topicVal = rel.eTopic[c] * scaleTopic
                    let subS = gS * echoVal + (1.0 - gS) * topicVal
                    let subC = gC * echoVal + (1.0 - gC) * topicVal
                    eRelS[dst + c] = (aS * normValS + (1.0 - aS) * subS + 0.25 * gS * echoVal + 0.05 * pe[peOff + c]) * mS
                    eRelC[dst + c] = (aC * normValC + (1.0 - aC) * subC + 0.25 * gC * echoVal + 0.05 * pe[peOff + c]) * mC
                    uRawContent[contOff + c] += wmS * (1.0 - aS * 0.8) * rawValS
                    uRawContent[contOff + d + c] += wmC * (1.0 - aC * 0.8) * rawValC
                }
            }
        }

        // 4. Sequence-Level Dual-Stream Subband Filtering & Shared SWT-Griffin Backbone
        let fStride = DualStreamRelationalWeights.frameStride
        let frameCount = max(1, (time + fStride - 1) / fStride)
        let btCount = batch * time
        let bf = batch * frameCount

        func buildSequenceStream(_ eRel: [Float], _ pairMask: [Float]) -> (hLL: [Float], hHH: [Float], fMask: [Float]) {
            let (ca1, cd1, ca2, cd2) = SWTFrequencyBranch.decompose(x: eRel, batch: batch, time: time, dim: d)
            var inLL1 = [Float](repeating: 0.0, count: btCount * d)
            var inLL2 = [Float](repeating: 0.0, count: btCount * d)
            var inHH1 = [Float](repeating: 0.0, count: btCount * 2 * d)
            var inHH2 = [Float](repeating: 0.0, count: btCount * d)

            for b in 0..<batch {
                let bTimeOff = b * time
                let bMatOff = b * time * d
                let bMatOff2 = b * time * 2 * d
                for t in 0..<time {
                    let pm = pairMask[bTimeOff + t]
                    guard pm > 0.0 else { continue }
                    let curOff = bMatOff + t * d
                    let prevOff = bMatOff + max(0, t - 1) * d
                    let prev2Off = bMatOff + max(0, t - 2) * d
                    let curOff2 = bMatOff2 + t * 2 * d
                    for c in 0..<d {
                        let c1 = (c + 1) % d
                        let c2 = (c + 2) % d
                        let c3 = (c + 3) % d
                        let c5 = (c + 5) % d
                        let c9 = (c + 9) % d

                        let mixPrev = 0.6 * eRel[prevOff + c1] - 0.4 * eRel[prevOff + c3] + 0.5 * eRel[prevOff + c]
                        let mixPrev2 = 0.6 * eRel[prevOff + c5] - 0.4 * eRel[prevOff + c9] + 0.5 * eRel[prevOff + c2]
                        let mixSkip = 0.6 * eRel[prev2Off + c5] - 0.4 * eRel[prev2Off + c1] + 0.5 * eRel[prev2Off + c3]
                        let mixCA1Prev = 0.6 * ca1[prevOff + c5] - 0.4 * ca1[prevOff + c9] + 0.5 * ca1[prevOff + c2]
                        let mixCA2 = 0.6 * ca2[curOff + c1] - 0.4 * ca2[curOff + c3] + 0.5 * ca2[curOff + c]

                        let eCur = eRel[curOff + c]
                        let cd1Cur = cd1[curOff + c]
                        let cd2Cur = cd2[curOff + c]

                        let bi1 = mixPrev * eCur * pm
                        let biSkip = mixSkip * eCur * pm
                        let triDetail = mixCA1Prev * cd1Cur * pm

                        inLL1[curOff + c] = (bi1 + 0.50 * cd2Cur * mixCA2) * pm
                        inLL2[curOff + c] = (biSkip + 0.50 * triDetail) * pm
                        inHH1[curOff2 + c] = cd1Cur * (0.40 + abs(mixPrev)) * pm
                        inHH1[curOff2 + d + c] = abs(cd1Cur) * abs(bi1) * pm
                        inHH2[curOff + c] = (triDetail + 0.65 * cd2Cur * mixPrev2) * pm
                    }
                }
            }

            let cLL1 = SWTHawkAutoencoder.silu(SWTFrequencyBranch.conv1d3Forward(x: inLL1, weights: rel.convRelLL1, batch: batch, time: time, inDim: d, outDim: d))
            let cLL2 = SWTHawkAutoencoder.silu(SWTFrequencyBranch.conv1d3Forward(x: inLL2, weights: rel.convRelLL2, batch: batch, time: time, inDim: d, outDim: d))
            let cHH1 = SWTHawkAutoencoder.silu(SWTFrequencyBranch.conv1d3Forward(x: inHH1, weights: rel.convRelHH1, batch: batch, time: time, inDim: 2 * d, outDim: d))
            let cHH2 = SWTHawkAutoencoder.silu(SWTFrequencyBranch.conv1d3Forward(x: inHH2, weights: rel.convRelHH2, batch: batch, time: time, inDim: d, outDim: d))

            var frameLL = [Float](repeating: 0.0, count: bf * d)
            var frameHH = [Float](repeating: 0.0, count: bf * d)
            var fMask = [Float](repeating: 0.0, count: bf)

            for b in 0..<batch {
                let bTimeOff = b * time
                let bFrameOff = b * frameCount
                for f in 0..<frameCount {
                    var fCount: Float = 0.0
                    let tStart = f * fStride
                    let tEnd = min(time, tStart + fStride)
                    let dstOff = (bFrameOff + f) * d
                    for t in tStart..<tEnd {
                        let pm = pairMask[bTimeOff + t]
                        guard pm > 0.0 else { continue }
                        fCount += pm
                        let srcOff = (bTimeOff + t) * d
                        for c in 0..<d {
                            frameLL[dstOff + c] += (cLL1[srcOff + c] + cLL2[srcOff + c]) * pm
                            frameHH[dstOff + c] += (cHH1[srcOff + c] + cHH2[srcOff + c]) * pm
                        }
                    }
                    if fCount > 0.0 {
                        fMask[bFrameOff + f] = 1.0
                        let invDenom = 1.0 / max(fCount, 1.0)
                        for c in 0..<d {
                            frameLL[dstOff + c] *= invDenom
                            frameHH[dstOff + c] *= invDenom
                        }
                    }
                }
            }

            func backbone(_ x: [Float]) -> [Float] {
                let fSWT = SWTFrequencyBranch.forward(x: x, weights: weights.swtBranch, batch: batch, time: frameCount)
                var tFeat = x
                for hb in weights.hawkBranch {
                    tFeat = HawkBlock.forward(x: tFeat, weights: hb, h0: nil, batch: batch, time: frameCount).out
                }
                let uFused = GatedTimeFrequencyFusion.forward(fSWT: fSWT, tFeat: tFeat, weights: weights.fusion, batch: batch, time: frameCount)
                var out = [Float](repeating: 0.0, count: bf * d)
                for b in 0..<batch {
                    for f in 0..<frameCount {
                        let fm = fMask[b * frameCount + f]
                        guard fm > 0.0 else { continue }
                        let off = (b * frameCount + f) * d
                        for c in 0..<d {
                            out[off + c] = (x[off + c] + 0.25 * uFused[off + c]) * fm
                        }
                    }
                }
                return out
            }

            return (backbone(frameLL), backbone(frameHH), fMask)
        }

        let (hLLS, hHHS, fMaskS) = buildSequenceStream(eRelS, pairMaskS)
        let (hLLC, hHHC, fMaskC) = buildSequenceStream(eRelC, pairMaskC)

        // 5. Subband Bidirectional Cross-Attention & Split-Horizon Salience Pooling
        let cohProjLL = SWTHawkAutoencoder.linear(x: psiCoh, w: rel.wCohLL, b: rel.bCohLL, rows: batch, inDim: cohDim, outDim: d)
        let cohProjHH = SWTHawkAutoencoder.linear(x: psiCoh, w: rel.wCohHH, b: rel.bCohHH, rows: batch, inDim: cohDim, outDim: d)
        let scale = 1.0 / sqrtf(Float(max(1, d)))

        func buildFrameWeights(_ fMask: [Float]) -> (wMean: [Float], wEarly: [Float], wLate: [Float]) {
            var wMean = [Float](repeating: 0.0, count: bf)
            var wEarly = [Float](repeating: 0.0, count: bf)
            var wLate = [Float](repeating: 0.0, count: bf)
            for b in 0..<batch {
                let off = b * frameCount
                var sumMask: Float = 0.0
                for f in 0..<frameCount { sumMask += fMask[off + f] }
                let lenF = max(sumMask, 1.0)
                var cum: Float = 0.0
                var sumM: Float = 0.0
                var sumE: Float = 0.0
                var sumL: Float = 0.0
                for f in 0..<frameCount {
                    let fm = fMask[off + f]
                    cum += fm
                    let tauF = (cum / lenF) * fm
                    let omF = max(0.0, 1.0 - tauF)
                    let mVal = fm
                    let eVal = fm * omF * omF
                    let lVal = fm * tauF * tauF
                    wMean[off + f] = mVal
                    wEarly[off + f] = eVal
                    wLate[off + f] = lVal
                    sumM += mVal
                    sumE += eVal
                    sumL += lVal
                }
                let invM = 1.0 / max(sumM, 1e-6)
                let invE = 1.0 / max(sumE, 1e-6)
                let invL = 1.0 / max(sumL, 1e-6)
                for f in 0..<frameCount {
                    wMean[off + f] *= invM
                    wEarly[off + f] *= invE
                    wLate[off + f] *= invL
                }
            }
            return (wMean, wEarly, wLate)
        }

        let (wMeanFS, wEarlyFS, wLateFS) = buildFrameWeights(fMaskS)
        let (wMeanFC, wEarlyFC, wLateFC) = buildFrameWeights(fMaskC)

        func maskFrames(_ x: [Float], _ fMask: [Float]) -> [Float] {
            var out = x
            for i in 0..<bf {
                let fm = fMask[i]
                if fm == 0.0 {
                    let off = i * d
                    for c in 0..<d { out[off + c] = 0.0 }
                } else if fm != 1.0 {
                    let off = i * d
                    for c in 0..<d { out[off + c] *= fm }
                }
            }
            return out
        }

        func computeSalienceWeights(_ logits: [Float], _ fMask: [Float], b: Int, col: Int) -> [Float] {
            let fOff = b * frameCount
            var maxLogit: Float = -.greatestFiniteMagnitude
            var raw = [Float](repeating: 0.0, count: frameCount)
            for f in 0..<frameCount {
                let fm = fMask[fOff + f]
                let v = logits[(fOff + f) * 2 + col] + (1.0 - fm) * (-1e4)
                raw[f] = v
                if v > maxLogit { maxLogit = v }
            }
            var sumExp: Float = 0.0
            for f in 0..<frameCount {
                let e = expf(raw[f] - maxLogit)
                raw[f] = e
                sumExp += e
            }
            let invExp = 1.0 / max(sumExp, 1e-12)
            var sumMasked: Float = 0.0
            for f in 0..<frameCount {
                let v = raw[f] * invExp * fMask[fOff + f]
                raw[f] = v
                sumMasked += v
            }
            let invNorm = 1.0 / max(sumMasked, 1e-6)
            for f in 0..<frameCount { raw[f] *= invNorm }
            return raw
        }

        func crossAttend(_ qB: [Float], _ kB: [Float], _ vB: [Float], keyMaskOff: Int, keyMask: [Float], queryMaskOff: Int, queryMask: [Float]) -> [Float] {
            var kT = [Float](repeating: 0.0, count: d * frameCount)
            for s in 0..<frameCount {
                let src = s * d
                for c in 0..<d { kT[c * frameCount + s] = kB[src + c] }
            }
            var scores = [Float](repeating: 0.0, count: frameCount * frameCount)
            SWTHawkAutoencoder.gemm(qB, kT, &scores, m: frameCount, n: frameCount, k: d)
            for t in 0..<frameCount {
                let rOff = t * frameCount
                var maxVal: Float = -.greatestFiniteMagnitude
                for s in 0..<frameCount {
                    let v = scores[rOff + s] * scale + (1.0 - keyMask[keyMaskOff + s]) * (-1e4)
                    scores[rOff + s] = v
                    if v > maxVal { maxVal = v }
                }
                var sumExp: Float = 0.0
                for s in 0..<frameCount {
                    let e = expf(scores[rOff + s] - maxVal)
                    scores[rOff + s] = e
                    sumExp += e
                }
                let inv = 1.0 / max(sumExp, 1e-12)
                for s in 0..<frameCount { scores[rOff + s] *= inv }
            }
            var attn = [Float](repeating: 0.0, count: frameCount * d)
            SWTHawkAutoencoder.gemm(scores, vB, &attn, m: frameCount, n: d, k: frameCount)
            for t in 0..<frameCount {
                let qm = queryMask[queryMaskOff + t]
                if qm != 1.0 {
                    let off = t * d
                    for c in 0..<d { attn[off + c] *= qm }
                }
            }
            return attn
        }

        func computeSubbandPhi(
            hS: [Float],
            hC: [Float],
            wQ: [Float],
            bQ: [Float],
            wK: [Float],
            bK: [Float],
            wV: [Float],
            bV: [Float],
            wPool: [Float],
            cohProj: [Float]
        ) -> [Float] {
            let qC = maskFrames(SWTHawkAutoencoder.linear(x: hC, w: wQ, b: bQ, rows: bf, inDim: d, outDim: d), fMaskC)
            let kS = maskFrames(SWTHawkAutoencoder.linear(x: hS, w: wK, b: bK, rows: bf, inDim: d, outDim: d), fMaskS)
            let vS = maskFrames(SWTHawkAutoencoder.linear(x: hS, w: wV, b: bV, rows: bf, inDim: d, outDim: d), fMaskS)
            let qS = maskFrames(SWTHawkAutoencoder.linear(x: hS, w: wQ, b: bQ, rows: bf, inDim: d, outDim: d), fMaskS)
            let kC = maskFrames(SWTHawkAutoencoder.linear(x: hC, w: wK, b: bK, rows: bf, inDim: d, outDim: d), fMaskC)
            let vC = maskFrames(SWTHawkAutoencoder.linear(x: hC, w: wV, b: bV, rows: bf, inDim: d, outDim: d), fMaskC)

            let zero2: [Float] = [0.0, 0.0]
            let salLogitsS = SWTHawkAutoencoder.linear(x: hS, w: wPool, b: zero2, rows: bf, inDim: d, outDim: 2)
            let salLogitsC = SWTHawkAutoencoder.linear(x: hC, w: wPool, b: zero2, rows: bf, inDim: d, outDim: 2)

            var phi = [Float](repeating: 0.0, count: batch * 5 * d)
            for b in 0..<batch {
                let fOff = b * frameCount
                let matOff = fOff * d
                let matEnd = matOff + frameCount * d

                let attnCS = crossAttend(Array(qC[matOff..<matEnd]), Array(kS[matOff..<matEnd]), Array(vS[matOff..<matEnd]), keyMaskOff: fOff, keyMask: fMaskS, queryMaskOff: fOff, queryMask: fMaskC)
                let attnSC = crossAttend(Array(qS[matOff..<matEnd]), Array(kC[matOff..<matEnd]), Array(vC[matOff..<matEnd]), keyMaskOff: fOff, keyMask: fMaskC, queryMaskOff: fOff, queryMask: fMaskS)

                let wSalS0 = computeSalienceWeights(salLogitsS, fMaskS, b: b, col: 0)
                let wSalS1 = computeSalienceWeights(salLogitsS, fMaskS, b: b, col: 1)
                let wSalC0 = computeSalienceWeights(salLogitsC, fMaskC, b: b, col: 0)
                let wSalC1 = computeSalienceWeights(salLogitsC, fMaskC, b: b, col: 1)

                var sEarlyPool = [Float](repeating: 0.0, count: d)
                var sLatePool = [Float](repeating: 0.0, count: d)
                var sMeanPool = [Float](repeating: 0.0, count: d)
                var sSal0Pool = [Float](repeating: 0.0, count: d)
                var sSal1Pool = [Float](repeating: 0.0, count: d)

                var cEarlyPool = [Float](repeating: 0.0, count: d)
                var cLatePool = [Float](repeating: 0.0, count: d)
                var cMeanPool = [Float](repeating: 0.0, count: d)
                var cSal0Pool = [Float](repeating: 0.0, count: d)
                var cSal1Pool = [Float](repeating: 0.0, count: d)

                var interCEarly = [Float](repeating: 0.0, count: d)
                var interCLate = [Float](repeating: 0.0, count: d)
                var interCMean = [Float](repeating: 0.0, count: d)
                var interSMean = [Float](repeating: 0.0, count: d)

                for f in 0..<frameCount {
                    let weS = wEarlyFS[fOff + f]
                    let wlS = wLateFS[fOff + f]
                    let wmS = wMeanFS[fOff + f]
                    let ws0 = wSalS0[f]
                    let ws1 = wSalS1[f]
                    let fmS = fMaskS[fOff + f]

                    let weC = wEarlyFC[fOff + f]
                    let wlC = wLateFC[fOff + f]
                    let wmC = wMeanFC[fOff + f]
                    let wc0 = wSalC0[f]
                    let wc1 = wSalC1[f]
                    let fmC = fMaskC[fOff + f]

                    let gOff = matOff + f * d
                    let lOff = f * d
                    for c in 0..<d {
                        let hs = hS[gOff + c]
                        let hc = hC[gOff + c]
                        let zs = hs + 0.50 * kS[gOff + c]
                        let zc = hc + 0.50 * qC[gOff + c]

                        sEarlyPool[c] += weS * zs
                        sLatePool[c] += wlS * zs
                        sMeanPool[c] += wmS * zs
                        sSal0Pool[c] += ws0 * zs
                        sSal1Pool[c] += ws1 * zs

                        cEarlyPool[c] += weC * zc
                        cLatePool[c] += wlC * zc
                        cMeanPool[c] += wmC * zc
                        cSal0Pool[c] += wc0 * zc
                        cSal1Pool[c] += wc1 * zc

                        let acs = attnCS[lOff + c]
                        let asc = attnSC[lOff + c]
                        let ic = (acs + 0.50 * (hc * acs) - 0.35 * abs(hc - acs)) * fmC
                        let isVal = (asc + 0.35 * (hs * asc)) * fmS

                        interCEarly[c] += weC * ic
                        interCLate[c] += wlC * ic
                        interCMean[c] += wmC * ic
                        interSMean[c] += wmS * isVal
                    }
                }

                let gBar = 0.25 + 0.75 * gBarBatch[b]
                let lenRatio = activeLenC[b] / activeLenS[b]
                let betaLate = min(1.0, max(0.0, (lenRatio - 0.62) / 0.12))
                let betaEarly = 1.0 - betaLate
                let phiBOff = b * 5 * d
                let cohOff = b * d
                for c in 0..<d {
                    let sT4 = 0.40 * sMeanPool[c] + 0.60 * sLatePool[c] + 0.25 * (sLatePool[c] - sEarlyPool[c]) + 0.25 * sSal1Pool[c]
                    let cT4 = 0.20 * cMeanPool[c] + 0.80 * cLatePool[c] + 0.35 * (cLatePool[c] - cEarlyPool[c]) + 0.25 * cSal1Pool[c]
                    let sT2 = 0.70 * sEarlyPool[c] + 0.30 * sMeanPool[c] + 0.25 * (sEarlyPool[c] - sLatePool[c]) + 0.20 * sSal0Pool[c]
                    let cT2 = 0.50 * cMeanPool[c] + 0.50 * cLatePool[c] + 0.30 * (cLatePool[c] - cEarlyPool[c]) + 0.25 * cSal0Pool[c]

                    let b0 = betaLate * (0.85 * cT4 + 0.50 * sT4 + 0.45 * abs(sT4 - cT4) + 0.35 * (sT4 * cT4))
                    let b1 = betaLate * (0.85 * cT4 - 0.40 * sT4 + 0.50 * (cLatePool[c] - cEarlyPool[c]) + gBar * (0.75 * interCLate[c] + 0.25 * interCMean[c]))
                    let b2 = betaEarly * (0.85 * cT2 + 0.50 * sT2 + 0.45 * abs(sT2 - cT2) + 0.35 * (sT2 * cT2))
                    let b3 = betaEarly * (0.85 * cT2 - 0.40 * sT2 + 0.45 * (cT2 - sEarlyPool[c]) + gBar * (0.60 * interCEarly[c] + 0.40 * interCMean[c]))
                    let uCrossC = gBar * (betaLate * interCLate[c] + betaEarly * interCMean[c]) + 0.35 * interSMean[c] + cohProj[cohOff + c]

                    phi[phiBOff + c] = b0
                    phi[phiBOff + d + c] = b1
                    phi[phiBOff + 2 * d + c] = b2
                    phi[phiBOff + 3 * d + c] = b3
                    phi[phiBOff + 4 * d + c] = uCrossC
                }
            }
            return phi
        }

        let phiLL = computeSubbandPhi(
            hS: hLLS, hC: hLLC,
            wQ: rel.wQLL, bQ: rel.bQLL,
            wK: rel.wKLL, bK: rel.bKLL,
            wV: rel.wVLL, bV: rel.bVLL,
            wPool: rel.wPoolLL, cohProj: cohProjLL
        )
        let phiHH = computeSubbandPhi(
            hS: hHHS, hC: hHHC,
            wQ: rel.wQHH, bQ: rel.bQHH,
            wK: rel.wKHH, bK: rel.bKHH,
            wV: rel.wVHH, bV: rel.bVHH,
            wPool: rel.wPoolHH, cohProj: cohProjHH
        )

        // 6. Latent Projection Layer & LayerNorm
        let zLLRaw = SWTHawkAutoencoder.linear(x: phiLL, w: rel.wProjLL, b: rel.bProjLL, rows: batch, inDim: 5 * d, outDim: half)
        let zLL = SWTHawkAutoencoder.layerNorm(x: zLLRaw, gamma: rel.normLLGamma, beta: rel.normLLBeta, rows: batch, dim: half)

        let zHHRaw = SWTHawkAutoencoder.linear(x: phiHH, w: rel.wProjHH, b: rel.bProjHH, rows: batch, inDim: 5 * d, outDim: half)
        let zHH = SWTHawkAutoencoder.layerNorm(x: zHHRaw, gamma: rel.normHHGamma, beta: rel.normHHBeta, rows: batch, dim: half)

        let zContRaw = SWTHawkAutoencoder.linear(x: uRawContent, w: rel.wContent, b: rel.bContent, rows: batch, inDim: 2 * d, outDim: half)
        let unitGamma = [Float](repeating: 1.0, count: half)
        let zeroBeta = [Float](repeating: 0.0, count: half)
        let zContent = SWTHawkAutoencoder.layerNorm(x: zContRaw, gamma: unitGamma, beta: zeroBeta, rows: batch, dim: half)

        var zRel = [Float](repeating: 0.0, count: batch * lDim)
        for b in 0..<batch {
            let srcOff = b * half
            let dstOff = b * lDim
            for c in 0..<half {
                zRel[dstOff + c] = zLL[srcOff + c]
                zRel[dstOff + half + c] = zHH[srcOff + c]
            }
        }

        return (phiLL, phiHH, zRel, zContent)
    }

    /// Projects pre-extracted subband interaction features `(phiLL: [B, 5D], phiHH: [B, 5D])` into `zRel = [z_LL^rel(512); z_HH^rel(512)] ∈ R^(B × 1024)`.
    public static func projectLatentFromPhi(
        phiLL: [Float],
        phiHH: [Float],
        batch: Int,
        weights: DualStreamRelationalWeights
    ) -> [Float] {
        let d = weights.dim
        let lDim = weights.latentDim
        let half = lDim / 2
        let zLLRaw = SWTHawkAutoencoder.linear(x: phiLL, w: weights.wProjLL, b: weights.bProjLL, rows: batch, inDim: 5 * d, outDim: half)
        let zLL = SWTHawkAutoencoder.layerNorm(x: zLLRaw, gamma: weights.normLLGamma, beta: weights.normLLBeta, rows: batch, dim: half)
        let zHHRaw = SWTHawkAutoencoder.linear(x: phiHH, w: weights.wProjHH, b: weights.bProjHH, rows: batch, inDim: 5 * d, outDim: half)
        let zHH = SWTHawkAutoencoder.layerNorm(x: zHHRaw, gamma: weights.normHHGamma, beta: weights.normHHBeta, rows: batch, dim: half)

        var zRel = [Float](repeating: 0.0, count: batch * lDim)
        for b in 0..<batch {
            let srcOff = b * half
            let dstOff = b * lDim
            for c in 0..<half {
                zRel[dstOff + c] = zLL[srcOff + c]
                zRel[dstOff + half + c] = zHH[srcOff + c]
            }
        }
        return zRel
    }

    /// Solves the L2-regularized Ridge Regression `min_{W, b} (1/N) ||Φ W + b - Z_target||_F^2 + λ ||W||_F^2`
    /// via Cholesky factorization (`G = L L^T`), returning `(w: [inDim, outDim], b: [outDim])`.
    public static func solveRidgeProjection(
        phi: [Float],
        targetZ: [Float],
        rows: Int,
        inDim: Int,
        outDim: Int,
        ridgeLambda: Float = 0.08
    ) -> (w: [Float], b: [Float]) {
        let augDim = inDim + 1
        let invN = 1.0 / Float(max(1, rows))
        var colSq = [Float](repeating: 0.0, count: inDim)
        for r in 0..<rows {
            let srcOff = r * inDim
            for c in 0..<inDim {
                let v = phi[srcOff + c]
                colSq[c] += v * v
            }
        }
        var colScale = [Float](repeating: 1.0, count: inDim)
        for c in 0..<inDim {
            let rms = sqrtf(colSq[c] * invN)
            colScale[c] = (rms > 1e-9) ? (1.0 / rms) : 0.0
        }

        var phiAugT = [Float](repeating: 0.0, count: augDim * rows)
        var phiAug = [Float](repeating: 0.0, count: rows * augDim)
        for r in 0..<rows {
            let srcOff = r * inDim
            let dstOff = r * augDim
            for c in 0..<inDim {
                let v = phi[srcOff + c] * colScale[c]
                phiAug[dstOff + c] = v
                phiAugT[c * rows + r] = v * invN
            }
            phiAug[dstOff + inDim] = 1.0
            phiAugT[inDim * rows + r] = invN
        }

        var gram = [Float](repeating: 0.0, count: augDim * augDim)
        SWTHawkAutoencoder.gemm(phiAugT, phiAug, &gram, m: augDim, n: augDim, k: rows)
        let reg = max(ridgeLambda, 0.25)
        for i in 0..<inDim {
            gram[i * augDim + i] += reg
        }
        gram[inDim * augDim + inDim] += 1e-5

        var rhs = [Float](repeating: 0.0, count: augDim * outDim)
        SWTHawkAutoencoder.gemm(phiAugT, targetZ, &rhs, m: augDim, n: outDim, k: rows)

        // Cholesky factorization G = L * L^T
        var chol = [Float](repeating: 0.0, count: augDim * augDim)
        for i in 0..<augDim {
            let iOff = i * augDim
            for j in 0...i {
                let jOff = j * augDim
                var sum = gram[iOff + j]
                for k in 0..<j {
                    sum -= chol[iOff + k] * chol[jOff + k]
                }
                if i == j {
                    chol[iOff + j] = sqrtf(max(sum, 1e-8))
                } else {
                    chol[iOff + j] = sum / chol[jOff + j]
                }
            }
        }

        // Forward substitution L * Y = RHS
        var yMat = [Float](repeating: 0.0, count: augDim * outDim)
        for i in 0..<augDim {
            let iAug = i * augDim
            let iOut = i * outDim
            let invDiag = 1.0 / chol[iAug + i]
            for c in 0..<outDim {
                yMat[iOut + c] = rhs[iOut + c]
            }
            for k in 0..<i {
                let lIK = chol[iAug + k]
                let kOut = k * outDim
                for c in 0..<outDim {
                    yMat[iOut + c] -= lIK * yMat[kOut + c]
                }
            }
            for c in 0..<outDim {
                yMat[iOut + c] *= invDiag
            }
        }

        // Backward substitution L^T * X = Y
        var xMat = [Float](repeating: 0.0, count: augDim * outDim)
        for i in stride(from: augDim - 1, through: 0, by: -1) {
            let iAug = i * augDim
            let iOut = i * outDim
            let invDiag = 1.0 / chol[iAug + i]
            for c in 0..<outDim {
                xMat[iOut + c] = yMat[iOut + c]
            }
            for k in (i + 1)..<augDim {
                let lKI = chol[k * augDim + i]
                let kOut = k * outDim
                for c in 0..<outDim {
                    xMat[iOut + c] -= lKI * xMat[kOut + c]
                }
            }
            for c in 0..<outDim {
                xMat[iOut + c] *= invDiag
            }
        }

        let quarter = outDim / 2
        if quarter > 0 && inDim >= 5 * quarter {
            let passScale: Float = 0.12
            for j in 0..<quarter {
                xMat[j * outDim + j] += passScale
                xMat[(2 * quarter + j) * outDim + j] += passScale
                xMat[(4 * quarter + j) * outDim + j] += passScale
                xMat[(quarter + j) * outDim + (quarter + j)] += passScale
                xMat[(3 * quarter + j) * outDim + (quarter + j)] += passScale
                xMat[(4 * quarter + j) * outDim + (quarter + j)] += passScale
            }
        }
        var w = Array(xMat[0..<(inDim * outDim)])
        for i in 0..<inDim {
            let s = colScale[i]
            let off = i * outDim
            for c in 0..<outDim {
                w[off + c] *= s
            }
        }
        let b = Array(xMat[(inDim * outDim)..<(augDim * outDim)])
        return (w, b)
    }

    /// Pads or truncates an arbitrary token ID sequence to `length` using `TextVocabulary.padId`.
    public static func padOrTruncate(_ tokens: [Int], to length: Int) -> [Int] {
        let clean = tokens.filter { $0 != TextVocabulary.padId }
        var out = Array(clean.prefix(length))
        if out.count < length {
            out.append(contentsOf: repeatElement(TextVocabulary.padId, count: length - out.count))
        }
        return out
    }

    /// Pure-neural Dual-Stream Relational Encoder taking strictly `(senderTokenIDs: [Int], receiverTokenIDs: [Int])` (`Phase 5.1 Audit 1`).
    public static func encodeRelational(
        senderTokenIDs: [Int],
        receiverTokenIDs: [Int],
        weights: SWTHawkWeights
    ) -> [Float] {
        let time = max(DualStreamRelationalWeights.canonicalSeqLen, max(senderTokenIDs.count, receiverTokenIDs.count))
        let sPad = padOrTruncate(senderTokenIDs, to: time)
        let cPad = padOrTruncate(receiverTokenIDs, to: time)
        return forwardBatchCPU(
            senderTokens: sPad,
            receiverTokens: cPad,
            batch: 1,
            time: time,
            weights: weights
        ).zRel
    }

    /// Full pure-neural Dual-Stream Relational Inference taking strictly `(senderTokenIDs: [Int], receiverTokenIDs: [Int])` (`Phase 5.1 Audit 1`).
    public static func inferRelational(
        senderTokenIDs: [Int],
        receiverTokenIDs: [Int],
        weights: SWTHawkWeights
    ) -> RelationalInferenceOutput {
        let time = max(DualStreamRelationalWeights.canonicalSeqLen, max(senderTokenIDs.count, receiverTokenIDs.count))
        let sPad = padOrTruncate(senderTokenIDs, to: time)
        let cPad = padOrTruncate(receiverTokenIDs, to: time)
        let res = forwardBatchCPU(
            senderTokens: sPad,
            receiverTokens: cPad,
            batch: 1,
            time: time,
            weights: weights
        )
        return makeInferenceOutput(zRel: res.zRel, zContent: res.zContent, weights: weights.relational)
    }

    /// Builds the inference output (pentagon + 4 calibrated head predictions) from a single `zRel ∈ R^1024`.
    public static func makeInferenceOutput(
        zRel: [Float],
        zContent: [Float],
        weights: DualStreamRelationalWeights
    ) -> RelationalInferenceOutput {
        let p = computePentagonFromZRel(zRel: zRel, zRelEarly: nil)
        let c4 = classifyFromZRel(zRel: zRel, isEarlyT2: false, weights: weights)
        let c2 = classifyFromZRel(zRel: zRel, isEarlyT2: true, weights: weights)
        return RelationalInferenceOutput(
            zRel: zRel,
            zContent: zContent,
            archetypeProbsT4: c4.archProbs,
            topologyProbsT4: c4.topoProbs,
            archetypeProbsT2: c2.archProbs,
            topologyProbsT2: c2.topoProbs,
            predictedArchetypeT4: c4.archetype,
            predictedTopologyT4: c4.topology,
            predictedArchetypeT2: c2.archetype,
            predictedTopologyT2: c2.topology,
            pentagon: p.pentagon,
            sLL: p.sLL,
            sHH: p.sHH,
            rHH: p.rHH,
            dLL: p.dLL
        )
    }
}
