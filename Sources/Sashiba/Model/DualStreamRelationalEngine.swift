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

    public let vocabSize: Int
    public let dim: Int
    public let latentDim: Int

    // Disentanglement gate & universal topic/echo embeddings
    public var alphaRel: [Float]     // [vocabSize]
    public var eEcho: [Float]        // [dim]
    public var eTopic: [Float]       // [dim]

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

    public init(
        vocabSize: Int,
        dim: Int,
        latentDim: Int = 1024,
        alphaRel: [Float],
        eEcho: [Float],
        eTopic: [Float],
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
        self.vocabSize = vocabSize
        self.dim = dim
        self.latentDim = latentDim
        self.alphaRel = alphaRel
        self.eEcho = eEcho
        self.eTopic = eTopic
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
        return eEcho.count + eTopic.count + wQLL.count + bQLL.count + wKLL.count + bKLL.count + wVLL.count + bVLL.count + wCohLL.count + bCohLL.count + wQHH.count + bQHH.count + wKHH.count + bKHH.count + wVHH.count + bVHH.count + wCohHH.count + bCohHH.count
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
        latentDim: Int = 1024,
        alphaRel: [Float]? = nil,
        rng: inout SplitMix64
    ) -> DualStreamRelationalWeights {
        let half = latentDim / 2
        let cohDim = Self.coherenceDim
        let dBound = sqrtf(6.0 / Float(dim + dim))
        let cohBound = sqrtf(6.0 / Float(cohDim + dim))
        let projBound = sqrtf(6.0 / Float(5 * dim + half))
        let contBound = sqrtf(6.0 / Float(2 * dim + half))
        let clsBound = sqrtf(6.0 / Float(latentDim + 6))

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

        return DualStreamRelationalWeights(
            vocabSize: vocabSize,
            dim: dim,
            latentDim: latentDim,
            alphaRel: defaultAlpha,
            eEcho: randMat(dim, 0.5),
            eTopic: randMat(dim, 0.5),
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
            wProjLL: randMat(5 * dim * half, projBound),
            bProjLL: [Float](repeating: 0.0, count: half),
            normLLGamma: [Float](repeating: 1.0, count: half),
            normLLBeta: [Float](repeating: 0.0, count: half),
            wProjHH: randMat(5 * dim * half, projBound),
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
        latentDim: Int = 1024,
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
            let pairs = count / 2
            for p in 0..<pairs {
                let val = rng.nextSignedUniform(scale: 1.0)
                v[2 * p] = val
                v[2 * p + 1] = -val
            }
            var sq: Float = 0.0
            for x in v { sq += x * x }
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
        let lDim = weights.latentDim
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
        var wEarlyS = [Float](repeating: 0.0, count: bt)
        var wEarlyC = [Float](repeating: 0.0, count: bt)
        var wLateS = [Float](repeating: 0.0, count: bt)
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

            var cumS: Float = 0.0
            var cumC: Float = 0.0
            var sumPairS: Float = 0.0
            var sumPairC: Float = 0.0
            var sumEarlyS: Float = 0.0
            var sumEarlyC: Float = 0.0
            var sumLateS: Float = 0.0
            var sumLateC: Float = 0.0

            for t in 0..<time {
                let idx = base + t
                cumS += maskS[idx]
                cumC += maskC[idx]
                let tauS = (cumS / activeLenS[b]) * maskS[idx]
                let tauC = (cumC / activeLenC[b]) * maskC[idx]
                let pS = pairMaskS[idx]
                let pC = pairMaskC[idx]
                let omS = max(0.0, 1.0 - tauS)
                let omC = max(0.0, 1.0 - tauC)
                let eS = pS * omS * omS
                let eC = pC * omC * omC
                let lS = pS * tauS * tauS * tauS
                let lC = pC * tauC * tauC * tauC
                wMeanS[idx] = pS
                wMeanC[idx] = pC
                wEarlyS[idx] = eS
                wEarlyC[idx] = eC
                wLateS[idx] = lS
                wLateC[idx] = lC
                sumPairS += pS
                sumPairC += pC
                sumEarlyS += eS
                sumEarlyC += eC
                sumLateS += lS
                sumLateC += lC
            }
            let invPS = 1.0 / max(sumPairS, 1e-6)
            let invPC = 1.0 / max(sumPairC, 1e-6)
            let invES = 1.0 / max(sumEarlyS, 1e-6)
            let invEC = 1.0 / max(sumEarlyC, 1e-6)
            let invLS = 1.0 / max(sumLateS, 1e-6)
            let invLC = 1.0 / max(sumLateC, 1e-6)
            for t in 0..<time {
                let idx = base + t
                wMeanS[idx] *= invPS
                wMeanC[idx] *= invPC
                wEarlyS[idx] *= invES
                wEarlyC[idx] *= invEC
                wLateS[idx] *= invLS
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
                    uRawContent[contOff + c] += wmS * (1.0 - aS * 0.5) * rawValS
                    uRawContent[contOff + d + c] += wmC * (1.0 - aC * 0.5) * rawValC
                }
            }
        }

        // 4. Full-Resolution Wavelet Transition Scan + 12-Window Triple-Bank Multi-Scale SWT-Griffin Backbone
        let winCount = 12
        let bw = batch * winCount
        var k0 = [Float](repeating: 0.0, count: d)
        var k1 = [Float](repeating: 0.0, count: d)
        var k2 = [Float](repeating: 0.0, count: d)
        var rDecay = [Float](repeating: 0.85, count: d)
        var rInputScale = [Float](repeating: 0.52, count: d)
        let sliceSize = d * d
        for c in 0..<d {
            let diagIdx = c * d + c
            let nextIdx = ((c + 1) % d) * d + c
            k0[c] = 1.0 + 0.5 * (weights.swtBranch.convCA2.w[diagIdx] + weights.swtBranch.convCA2.w[nextIdx])
            k1[c] = 1.0 + 0.5 * (weights.swtBranch.convCA2.w[sliceSize + diagIdx] + weights.swtBranch.convCA2.w[sliceSize + nextIdx])
            k2[c] = 1.0 + 0.5 * (weights.swtBranch.convCA2.w[2 * sliceSize + diagIdx] + weights.swtBranch.convCA2.w[2 * sliceSize + nextIdx])
            if let hb0 = weights.hawkBranch.first {
                let lam = hb0.rglru.lambda[c]
                let sig = 1.0 / (1.0 + expf(-lam))
                let a = powf(sig, 0.25)
                rDecay[c] = a
                rInputScale[c] = sqrtf(max(1e-6, 1.0 - a * a))
            }
        }

        func buildMultiScaleStream(_ eRel: [Float], _ pairMask: [Float], _ wMean: [Float], _ wEarly: [Float], _ wLate: [Float], _ activeLen: [Float]) -> (hLL: [Float], hHH: [Float]) {
            let (ca1, cd1, ca2, cd2) = SWTFrequencyBranch.decompose(x: eRel, batch: batch, time: time, dim: d)
            var winInLL = [Float](repeating: 0.0, count: bw * d)
            var winInHH = [Float](repeating: 0.0, count: bw * d)

            for b in 0..<batch {
                let bTimeOff = b * time
                let bMatOff = b * time * d
                let actT = min(time, max(2, Int(activeLen[b]) + 2))
                var hRGLRU = [Float](repeating: 0.0, count: d)
                var prevH = [Float](repeating: 0.0, count: d)
                var sumMidWeight: Float = 0.0
                for t in 1..<actT {
                    let wm = wMean[bTimeOff + t]
                    let we = wEarly[bTimeOff + t]
                    let wl = wLate[bTimeOff + t]
                    let wMid = max(0.0, wm - 0.35 * (we + wl))
                    sumMidWeight += wMid
                }
                let invMid = 1.0 / max(sumMidWeight, 1e-6)

                let wOff0 = (b * winCount + 0) * d // Bank A early
                let wOff1 = (b * winCount + 1) * d // Bank A mid
                let wOff2 = (b * winCount + 2) * d // Bank A late
                let wOff3 = (b * winCount + 3) * d // Bank A mean
                let wOff4 = (b * winCount + 4) * d // Bank B early
                let wOff5 = (b * winCount + 5) * d // Bank B mid
                let wOff6 = (b * winCount + 6) * d // Bank B late
                let wOff7 = (b * winCount + 7) * d // Bank B mean
                let wOff8 = (b * winCount + 8) * d // Bank C early
                let wOff9 = (b * winCount + 9) * d // Bank C mid
                let wOff10 = (b * winCount + 10) * d // Bank C late
                let wOff11 = (b * winCount + 11) * d // Bank C mean

                for t in 0..<actT {
                    let idx = bTimeOff + t
                    let pm = pairMask[idx]
                    let curOff = bMatOff + t * d
                    let prevOff = bMatOff + max(0, t - 1) * d
                    let prev2Off = bMatOff + max(0, t - 2) * d
                    for c in 0..<d {
                        prevH[c] = hRGLRU[c]
                        hRGLRU[c] = rDecay[c] * hRGLRU[c] + rInputScale[c] * eRel[curOff + c]
                    }
                    guard pm > 0.0 else { continue }
                    let wm = wMean[idx]
                    let we = wEarly[idx]
                    let wl = wLate[idx]
                    let wMid = max(0.0, wm - 0.35 * (we + wl)) * invMid

                    for c in 0..<d {
                        let c1 = (c + 1) % d
                        let c2 = (c + 2) % d
                        let c3 = (c + 3) % d
                        let c4 = (c + 4) % d
                        let c5 = (c + 5) % d
                        let c6 = (c + 6) % d
                        let c7 = (c + 7) % d
                        let c8 = (c + 8) % d
                        let c9 = (c + 9) % d
                        let c11 = (c + 11) % d
                        let c13 = (c + 13) % d
                        let shiftPrevE = (0.6 * eRel[prevOff + c1] - 0.4 * eRel[prevOff + c3] + 0.5 * eRel[prevOff + c]) * k0[c]
                        let shiftPrevE2 = (0.6 * eRel[prevOff + c5] - 0.4 * eRel[prevOff + c9] + 0.5 * eRel[prevOff + c2]) * k2[c]
                        let shiftPrevE3 = (0.6 * eRel[prevOff + c7] - 0.4 * eRel[prevOff + c11] + 0.5 * eRel[prevOff + c4]) * k1[c]
                        let shiftPrevE4 = (0.6 * eRel[prevOff + c8] - 0.4 * eRel[prevOff + c13] + 0.5 * eRel[prevOff + c6]) * k0[c]
                        let shiftSkipE = (0.6 * eRel[prev2Off + c5] - 0.4 * eRel[prev2Off + c1] + 0.5 * eRel[prev2Off + c3]) * k1[c]
                        let shiftSkipE2 = (0.6 * eRel[prev2Off + c8] - 0.4 * eRel[prev2Off + c13] + 0.5 * eRel[prev2Off + c6]) * k0[c]
                        let shiftPrevCA1 = (0.6 * ca1[prevOff + c1] - 0.4 * ca1[prevOff + c3] + 0.5 * ca1[prevOff + c]) * k1[c]
                        let shiftPrevCA1_2 = (0.6 * ca1[prevOff + c5] - 0.4 * ca1[prevOff + c9] + 0.5 * ca1[prevOff + c2]) * k0[c]
                        let shiftPrevCA1_3 = (0.6 * ca1[prevOff + c8] - 0.4 * ca1[prevOff + c13] + 0.5 * ca1[prevOff + c6]) * k2[c]
                        let shiftCA2 = (0.6 * ca2[curOff + c1] - 0.4 * ca2[curOff + c3] + 0.5 * ca2[curOff + c]) * k0[c]

                        let pBi = shiftPrevE * (eRel[curOff + c] * k1[c]) * pm
                        let pBi2 = shiftPrevE2 * (eRel[curOff + c] * k0[c]) * pm
                        let pBi3 = shiftPrevE3 * (eRel[curOff + c] * k2[c]) * pm
                        let pBi4 = shiftPrevE4 * (eRel[curOff + c] * k1[c]) * pm
                        let pSkip = shiftSkipE * (eRel[curOff + c] * k2[c]) * pm
                        let pSkip2 = shiftSkipE2 * (eRel[curOff + c] * k1[c]) * pm
                        let pTri = shiftPrevCA1 * (cd1[curOff + c] * k2[c]) * pm
                        let pTri2 = shiftPrevCA1_2 * (cd1[curOff + c] * k1[c]) * pm
                        let pTri3 = shiftPrevCA1_3 * (cd1[curOff + c] * k0[c]) * pm
                        let pCD2 = shiftCA2 * (cd2[curOff + c] * k2[c]) * pm
                        let dT = (hRGLRU[c] - prevH[c]) * eRel[curOff + c] * pm

                        let tau: Float = 0.45
                        let rawVLL_A = pBi + 0.55 * pSkip + 0.5 * pCD2 + 0.35 * dT
                        let rawVLL_B = pBi3 + 0.65 * pBi2 + 0.55 * pTri - 0.45 * pSkip + 0.35 * dT
                        let rawVLL_C = pBi4 + 0.65 * pSkip2 + 0.55 * pTri3 - 0.40 * pBi2 + 0.35 * dT
                        let rawVHH_A = pTri + 0.85 * pBi2 + 0.45 * pSkip + 0.35 * dT
                        let rawVHH_B = pTri2 + 0.75 * pBi3 + 0.55 * pCD2 - 0.45 * pBi + 0.35 * dT
                        let rawVHH_C = pTri3 + 0.75 * pBi4 + 0.55 * pSkip2 - 0.40 * pTri + 0.35 * dT

                        let vLL_A = max(0.0, rawVLL_A - tau) - max(0.0, -rawVLL_A - tau)
                        let vLL_B = max(0.0, rawVLL_B - tau) - max(0.0, -rawVLL_B - tau)
                        let vLL_C = max(0.0, rawVLL_C - tau) - max(0.0, -rawVLL_C - tau)
                        let vHH_A = max(0.0, rawVHH_A - tau) - max(0.0, -rawVHH_A - tau)
                        let vHH_B = max(0.0, rawVHH_B - tau) - max(0.0, -rawVHH_B - tau)
                        let vHH_C = max(0.0, rawVHH_C - tau) - max(0.0, -rawVHH_C - tau)

                        winInLL[wOff0 + c] += we * vLL_A
                        winInLL[wOff1 + c] += wMid * vLL_A
                        winInLL[wOff2 + c] += wl * vLL_A
                        winInLL[wOff3 + c] += wm * vLL_A
                        winInLL[wOff4 + c] += we * vLL_B
                        winInLL[wOff5 + c] += wMid * vLL_B
                        winInLL[wOff6 + c] += wl * vLL_B
                        winInLL[wOff7 + c] += wm * vLL_B
                        winInLL[wOff8 + c] += we * vLL_C
                        winInLL[wOff9 + c] += wMid * vLL_C
                        winInLL[wOff10 + c] += wl * vLL_C
                        winInLL[wOff11 + c] += wm * vLL_C

                        winInHH[wOff0 + c] += we * vHH_A
                        winInHH[wOff1 + c] += wMid * vHH_A
                        winInHH[wOff2 + c] += wl * vHH_A
                        winInHH[wOff3 + c] += wm * vHH_A
                        winInHH[wOff4 + c] += we * vHH_B
                        winInHH[wOff5 + c] += wMid * vHH_B
                        winInHH[wOff6 + c] += wl * vHH_B
                        winInHH[wOff7 + c] += wm * vHH_B
                        winInHH[wOff8 + c] += we * vHH_C
                        winInHH[wOff9 + c] += wMid * vHH_C
                        winInHH[wOff10 + c] += wl * vHH_C
                        winInHH[wOff11 + c] += wm * vHH_C
                    }
                }
            }

            // Run shared-weight SWT-Griffin Backbone on the 4-window subband state sequences [B, 4, D]
            let fSWT_LL = SWTFrequencyBranch.forward(x: winInLL, weights: weights.swtBranch, batch: batch, time: winCount)
            var tFeat_LL = winInLL
            for hb in weights.hawkBranch {
                tFeat_LL = HawkBlock.forward(x: tFeat_LL, weights: hb, h0: nil, batch: batch, time: winCount).out
            }
            let uFused_LL = GatedTimeFrequencyFusion.forward(fSWT: fSWT_LL, tFeat: tFeat_LL, weights: weights.fusion, batch: batch, time: winCount)

            let fSWT_HH = SWTFrequencyBranch.forward(x: winInHH, weights: weights.swtBranch, batch: batch, time: winCount)
            var tFeat_HH = winInHH
            for hb in weights.hawkBranch {
                tFeat_HH = HawkBlock.forward(x: tFeat_HH, weights: hb, h0: nil, batch: batch, time: winCount).out
            }
            let uFused_HH = GatedTimeFrequencyFusion.forward(fSWT: fSWT_HH, tFeat: tFeat_HH, weights: weights.fusion, batch: batch, time: winCount)

            var hLL = [Float](repeating: 0.0, count: bw * d)
            var hHH = [Float](repeating: 0.0, count: bw * d)
            for i in 0..<(bw * d) {
                hLL[i] = winInLL[i] + 0.25 * uFused_LL[i]
                hHH[i] = winInHH[i] + 0.25 * uFused_HH[i]
            }
            return (hLL, hHH)
        }

        let (hLLS, hHHS) = buildMultiScaleStream(eRelS, pairMaskS, wMeanS, wEarlyS, wLateS, activeLenS)
        let (hLLC, hHHC) = buildMultiScaleStream(eRelC, pairMaskC, wMeanC, wEarlyC, wLateC, activeLenC)

        // 5. Subband Bidirectional Cross-Attention & Multi-Scale Interaction Pooling over the 4 temporal windows
        let qLLC = SWTHawkAutoencoder.linear(x: hLLC, w: rel.wQLL, b: rel.bQLL, rows: bw, inDim: d, outDim: d)
        let kLLS = SWTHawkAutoencoder.linear(x: hLLS, w: rel.wKLL, b: rel.bKLL, rows: bw, inDim: d, outDim: d)
        let vLLS = SWTHawkAutoencoder.linear(x: hLLS, w: rel.wVLL, b: rel.bVLL, rows: bw, inDim: d, outDim: d)

        let qHHC = SWTHawkAutoencoder.linear(x: hHHC, w: rel.wQHH, b: rel.bQHH, rows: bw, inDim: d, outDim: d)
        let kHHS = SWTHawkAutoencoder.linear(x: hHHS, w: rel.wKHH, b: rel.bKHH, rows: bw, inDim: d, outDim: d)
        let vHHS = SWTHawkAutoencoder.linear(x: hHHS, w: rel.wVHH, b: rel.bVHH, rows: bw, inDim: d, outDim: d)

        let cohProjLL = SWTHawkAutoencoder.linear(x: psiCoh, w: rel.wCohLL, b: rel.bCohLL, rows: batch, inDim: cohDim, outDim: d)
        let cohProjHH = SWTHawkAutoencoder.linear(x: psiCoh, w: rel.wCohHH, b: rel.bCohHH, rows: batch, inDim: cohDim, outDim: d)

        let scale = 1.0 / sqrtf(Float(max(1, d)))
        var phiLL = [Float](repeating: 0.0, count: batch * 5 * d)
        var phiHH = [Float](repeating: 0.0, count: batch * 5 * d)

        for b in 0..<batch {
            let bMatOff = b * winCount * d
            var kLL_T = [Float](repeating: 0.0, count: d * winCount)
            var kHH_T = [Float](repeating: 0.0, count: d * winCount)
            var vLL_B = [Float](repeating: 0.0, count: winCount * d)
            var vHH_B = [Float](repeating: 0.0, count: winCount * d)
            for s in 0..<winCount {
                let src = bMatOff + s * d
                for c in 0..<d {
                    kLL_T[c * winCount + s] = kLLS[src + c]
                    kHH_T[c * winCount + s] = kHHS[src + c]
                    vLL_B[s * d + c] = vLLS[src + c]
                    vHH_B[s * d + c] = vHHS[src + c]
                }
            }
            let qLL_B = Array(qLLC[bMatOff..<(bMatOff + winCount * d)])
            let qHH_B = Array(qHHC[bMatOff..<(bMatOff + winCount * d)])
            var scoresLL = [Float](repeating: 0.0, count: winCount * winCount)
            var scoresHH = [Float](repeating: 0.0, count: winCount * winCount)
            SWTHawkAutoencoder.gemm(qLL_B, kLL_T, &scoresLL, m: winCount, n: winCount, k: d)
            SWTHawkAutoencoder.gemm(qHH_B, kHH_T, &scoresHH, m: winCount, n: winCount, k: d)

            for t in 0..<winCount {
                let rOff = t * winCount
                var maxLL: Float = -.greatestFiniteMagnitude
                var maxHH: Float = -.greatestFiniteMagnitude
                for s in 0..<winCount {
                    let vL = scoresLL[rOff + s] * scale
                    let vH = scoresHH[rOff + s] * scale
                    scoresLL[rOff + s] = vL
                    scoresHH[rOff + s] = vH
                    if vL > maxLL { maxLL = vL }
                    if vH > maxHH { maxHH = vH }
                }
                var sumLL: Float = 0.0
                var sumHH: Float = 0.0
                for s in 0..<winCount {
                    let eL = expf(scoresLL[rOff + s] - maxLL)
                    let eH = expf(scoresHH[rOff + s] - maxHH)
                    scoresLL[rOff + s] = eL
                    scoresHH[rOff + s] = eH
                    sumLL += eL
                    sumHH += eH
                }
                let invLL = 1.0 / max(sumLL, 1e-12)
                let invHH = 1.0 / max(sumHH, 1e-12)
                for s in 0..<winCount {
                    scoresLL[rOff + s] *= invLL
                    scoresHH[rOff + s] *= invHH
                }
            }

            var attnLL = [Float](repeating: 0.0, count: winCount * d)
            var attnHH = [Float](repeating: 0.0, count: winCount * d)
            SWTHawkAutoencoder.gemm(scoresLL, vLL_B, &attnLL, m: winCount, n: d, k: winCount)
            SWTHawkAutoencoder.gemm(scoresHH, vHH_B, &attnHH, m: winCount, n: d, k: winCount)

            // Windows: 0..3 = Bank A (early, mid, late, mean), 4..7 = Bank B (early, mid, late, mean)
            let offAE = bMatOff + 0 * d
            let offAMid = bMatOff + 1 * d
            let offAL = bMatOff + 2 * d
            let offAM = bMatOff + 3 * d
            let offBE = bMatOff + 4 * d
            let offBMid = bMatOff + 5 * d
            let offBL = bMatOff + 6 * d
            let offBM = bMatOff + 7 * d
            let attnOffAL = 2 * d
            let attnOffAM = 3 * d
            let attnOffBL = 6 * d
            let attnOffBM = 7 * d

            let gBar = 0.2 + 0.8 * gBarBatch[b]
            let betaLate = min(1.0, max(0.0, (activeLenC[b] - 110.0) / 20.0))
            let betaEarly = 1.0 - betaLate
            let phiBOff = b * 5 * d
            let cohOff = b * d
            for c in 0..<d {
                let sLA_mean = hLLS[offAM + c] + kLLS[offAM + c]
                let sLA_early = hLLS[offAE + c] + kLLS[offAE + c]
                let sLA_mid = hLLS[offAMid + c] + kLLS[offAMid + c]
                let sLA_late = hLLS[offAL + c] + kLLS[offAL + c]
                let cLA_mean = hLLC[offAM + c] + qLLC[offAM + c]
                let cLA_early = hLLC[offAE + c] + qLLC[offAE + c]
                let cLA_mid = hLLC[offAMid + c] + qLLC[offAMid + c]
                let cLA_late = hLLC[offAL + c] + qLLC[offAL + c]

                let sLB_mean = hLLS[offBM + c] + kLLS[offBM + c]
                let sLB_early = hLLS[offBE + c] + kLLS[offBE + c]
                let sLB_mid = hLLS[offBMid + c] + kLLS[offBMid + c]
                let sLB_late = hLLS[offBL + c] + kLLS[offBL + c]
                let cLB_mean = hLLC[offBM + c] + qLLC[offBM + c]
                let cLB_early = hLLC[offBE + c] + qLLC[offBE + c]
                let cLB_mid = hLLC[offBMid + c] + qLLC[offBMid + c]
                let cLB_late = hLLC[offBL + c] + qLLC[offBL + c]

                let sLA_t4 = 0.45 * sLA_mean + 0.55 * sLA_late + 0.20 * (sLA_late - sLA_early)
                let cLA_t4 = 0.25 * cLA_mean + 0.75 * cLA_late + 0.35 * (cLA_late - cLA_early)
                let diffLA_t4 = 0.50 * abs(sLA_t4 - cLA_t4) + 0.45 * (cLA_late - cLA_early) + 0.35 * (cLA_mid - sLA_mid)

                let sLB_t4 = 0.45 * sLB_mean + 0.55 * sLB_late + 0.20 * (sLB_late - sLB_early)
                let cLB_t4 = 0.25 * cLB_mean + 0.75 * cLB_late + 0.35 * (cLB_late - cLB_early)
                let diffLB_t4 = 0.50 * abs(sLB_t4 - cLB_t4) + 0.45 * (cLB_late - cLB_early) + 0.35 * (cLB_mid - sLB_mid)

                let sLA_t2 = 0.55 * sLA_early + 0.45 * sLA_mid + 0.25 * (sLA_early - sLA_late)
                let cLA_t2 = 0.50 * cLA_mean + 0.50 * cLA_late + 0.35 * (cLA_late - cLA_early)
                let diffLA_t2 = 0.50 * abs(sLA_t2 - cLA_t2) + 0.45 * (cLA_late - cLA_early) + 0.35 * (cLA_early - sLA_early)

                let sLB_t2 = 0.55 * sLB_early + 0.45 * sLB_mid + 0.25 * (sLB_early - sLB_late)
                let cLB_t2 = 0.50 * cLB_mean + 0.50 * cLB_late + 0.35 * (cLB_late - cLB_early)
                let diffLB_t2 = 0.50 * abs(sLB_t2 - cLB_t2) + 0.45 * (cLB_late - cLB_early) + 0.35 * (cLB_early - sLB_early)

                let b0L = betaLate * (0.85 * cLA_t4 + 0.55 * sLA_t4 + diffLA_t4 + 0.35 * (sLA_t4 * cLA_t4) + 0.35 * cLA_mid)
                let b1L = betaLate * (0.85 * cLB_t4 - 0.35 * sLB_t4 + diffLB_t4 + 0.35 * (sLB_t4 * cLB_t4) + 0.35 * cLB_mid)
                let b2L = betaEarly * (0.85 * cLA_t2 + 0.55 * sLA_t2 + diffLA_t2 + 0.35 * (sLA_t2 * cLA_t2) + 0.40 * cLA_early + 0.35 * cLA_mid)
                let b3L = betaEarly * (0.85 * cLB_t2 - 0.35 * sLB_t2 + diffLB_t2 + 0.35 * (sLB_t2 * cLB_t2) + 0.40 * cLB_early + 0.35 * cLB_mid)
                let crossL = gBar * (attnLL[attnOffAM + c] + attnLL[attnOffBM + c]) + 0.45 * (cLA_late - cLA_early) + cohProjLL[cohOff + c]

                phiLL[phiBOff + c] = b0L
                phiLL[phiBOff + d + c] = b1L
                phiLL[phiBOff + 2 * d + c] = b2L
                phiLL[phiBOff + 3 * d + c] = b3L
                phiLL[phiBOff + 4 * d + c] = crossL

                let sHA_mean = hHHS[offAM + c] + kHHS[offAM + c]
                let sHA_early = hHHS[offAE + c] + kHHS[offAE + c]
                let sHA_mid = hHHS[offAMid + c] + kHHS[offAMid + c]
                let sHA_late = hHHS[offAL + c] + kHHS[offAL + c]
                let cHA_mean = hHHC[offAM + c] + qHHC[offAM + c]
                let cHA_early = hHHC[offAE + c] + qHHC[offAE + c]
                let cHA_mid = hHHC[offAMid + c] + qHHC[offAMid + c]
                let cHA_late = hHHC[offAL + c] + qHHC[offAL + c]

                let sHB_mean = hHHS[offBM + c] + kHHS[offBM + c]
                let sHB_early = hHHS[offBE + c] + kHHS[offBE + c]
                let sHB_mid = hHHS[offBMid + c] + kHHS[offBMid + c]
                let sHB_late = hHHS[offBL + c] + kHHS[offBL + c]
                let cHB_mean = hHHC[offBM + c] + qHHC[offBM + c]
                let cHB_early = hHHC[offBE + c] + qHHC[offBE + c]
                let cHB_mid = hHHC[offBMid + c] + qHHC[offBMid + c]
                let cHB_late = hHHC[offBL + c] + qHHC[offBL + c]

                let sHA_t4 = 0.35 * sHA_mean + 0.65 * sHA_late + 0.20 * (sHA_late - sHA_early)
                let cHA_t4 = 0.85 * cHA_late + 0.15 * cHA_mean + 0.35 * (cHA_late - cHA_early)
                let diffHA_t4 = 0.50 * abs(sHA_t4 - cHA_t4) + 0.45 * (cHA_late - cHA_early) + 0.35 * (cHA_mid - sHA_mid)

                let sHB_t4 = 0.35 * sHB_mean + 0.65 * sHB_late + 0.20 * (sHB_late - sHB_early)
                let cHB_t4 = 0.85 * cHB_late + 0.15 * cHB_mean + 0.35 * (cHB_late - cHB_early)
                let diffHB_t4 = 0.50 * abs(sHB_t4 - cHB_t4) + 0.45 * (cHB_late - cHB_early) + 0.35 * (cHB_mid - sHB_mid)

                let sHA_t2 = 0.55 * sHA_early + 0.45 * sHA_mid + 0.25 * (sHA_early - sHA_late)
                let cHA_t2 = 0.50 * cHA_mean + 0.50 * cHA_late + 0.35 * (cHA_late - cHA_early)
                let diffHA_t2 = 0.50 * abs(sHA_t2 - cHA_t2) + 0.45 * (cHA_late - cHA_early) + 0.35 * (cHA_early - sHA_early)

                let sHB_t2 = 0.55 * sHB_early + 0.45 * sHB_mid + 0.25 * (sHB_early - sHB_late)
                let cHB_t2 = 0.50 * cHB_mean + 0.50 * cHB_late + 0.35 * (cHB_late - cHB_early)
                let diffHB_t2 = 0.50 * abs(sHB_t2 - cHB_t2) + 0.45 * (cHB_late - cHB_early) + 0.35 * (cHB_early - sHB_early)

                let b0H = betaLate * (0.85 * cHA_t4 + 0.55 * sHA_t4 + diffHA_t4 + 0.35 * (sHA_t4 * cHA_t4) + 0.40 * cHA_early + 0.35 * cHA_mid)
                let b1H = betaLate * (0.85 * cHB_t4 - 0.35 * sHB_t4 + diffHB_t4 + 0.35 * (sHB_t4 * cHB_t4) + 0.40 * cHB_early + 0.35 * cHB_mid)
                let b2H = betaEarly * (0.85 * cHA_t2 + 0.55 * sHA_t2 + diffHA_t2 + 0.35 * (sHA_t2 * cHA_t2) + 0.50 * cHA_early + 0.35 * cHA_mid)
                let b3H = betaEarly * (0.85 * cHB_t2 - 0.35 * sHB_t2 + diffHB_t2 + 0.35 * (sHB_t2 * cHB_t2) + 0.50 * cHB_early + 0.35 * cHB_mid)
                let crossH = betaEarly * (attnHH[attnOffAM + c] + attnHH[attnOffBM + c] + 0.50 * cHA_early) + betaLate * (attnHH[attnOffAL + c] + attnHH[attnOffBL + c] + 0.55 * cHA_early) + cohProjHH[cohOff + c]

                phiHH[phiBOff + c] = b0H
                phiHH[phiBOff + d + c] = b1H
                phiHH[phiBOff + 2 * d + c] = b2H
                phiHH[phiBOff + 3 * d + c] = b3H
                phiHH[phiBOff + 4 * d + c] = crossH
            }
        }

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
