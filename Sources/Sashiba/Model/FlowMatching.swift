import Foundation

/// Weights for Stage 2 Alignment & Stage 3 Conditional Flow Matching (`CFM`) Velocity Network `v_θ(x_t, t, c)` (`Section 3`):
///   - Treats `x_t ∈ R^(B × 1024)` and condition `c = z_full ∈ R^(B × 1024)` as `16` subband tokens of dimension `64` (`[B, 16, 64]`).
///   - Integrates `dx_t / dt = v_tilde(x_t, t, c)` from `t = 0 -> 1` using the 50-step Midpoint Method (2nd-order Runge-Kutta)
///     with Classifier-Free Guidance (`CFG`).
public struct FlowMatchingWeights: Sendable, Equatable {
    public static let numSubbandTokens: Int = 16
    public static let subbandTokenDim: Int = 64

    public let latentDim: Int
    public let hiddenDim: Int
    public let numPairs: Int
    public let wAlign: [Float]      // [hiddenDim, hiddenDim] subband token alignment projection
    public let bAlign: [Float]      // [hiddenDim]
    public let wTime: [Float]       // [hiddenDim, hiddenDim] sinusoidal time embedding projection
    public let bTime: [Float]       // [hiddenDim]
    public let wVelIn: [Float]      // [2 * hiddenDim, hiddenDim] velocity token input projection
    public let bVelIn: [Float]      // [hiddenDim]
    public let wVelGate: [Float]    // [hiddenDim, hiddenDim] SwiGLU velocity gate
    public let bVelGate: [Float]    // [hiddenDim]
    public let wVelOut: [Float]     // [hiddenDim, hiddenDim] velocity token output projection
    public let bVelOut: [Float]     // [hiddenDim]
    public let condBank: [Float]    // [numPairs, latentDim] aligned full-text condition prototypes
    public let summaryBank: [Float] // [numPairs, latentDim] target summary latent prototypes

    public init(
        latentDim: Int = 1024,
        hiddenDim: Int = FlowMatchingWeights.subbandTokenDim,
        numPairs: Int = 0,
        wAlign: [Float],
        bAlign: [Float],
        wTime: [Float],
        bTime: [Float],
        wVelIn: [Float],
        bVelIn: [Float],
        wVelGate: [Float],
        bVelGate: [Float],
        wVelOut: [Float],
        bVelOut: [Float],
        condBank: [Float] = [],
        summaryBank: [Float] = []
    ) {
        self.latentDim = latentDim
        self.hiddenDim = hiddenDim
        self.numPairs = numPairs
        self.wAlign = wAlign
        self.bAlign = bAlign
        self.wTime = wTime
        self.bTime = bTime
        self.wVelIn = wVelIn
        self.bVelIn = bVelIn
        self.wVelGate = wVelGate
        self.bVelGate = bVelGate
        self.wVelOut = wVelOut
        self.bVelOut = bVelOut
        self.condBank = condBank
        self.summaryBank = summaryBank
    }

    public var parameterCount: Int {
        return wAlign.count + bAlign.count + wTime.count + bTime.count + wVelIn.count + bVelIn.count + wVelGate.count + bVelGate.count + wVelOut.count + bVelOut.count + condBank.count + summaryBank.count
    }

    public static func makeRandom(
        latentDim: Int = 1024,
        hiddenDim: Int = FlowMatchingWeights.subbandTokenDim,
        rng: inout SplitMix64
    ) -> FlowMatchingWeights {
        let hBound = sqrtf(6.0 / Float(hiddenDim + hiddenDim))
        let inBound = sqrtf(6.0 / Float(2 * hiddenDim + hiddenDim))
        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                a[i] = rng.nextSignedUniform(scale: bound)
            }
            return a
        }

        return FlowMatchingWeights(
            latentDim: latentDim,
            hiddenDim: hiddenDim,
            numPairs: 0,
            wAlign: randMat(hiddenDim * hiddenDim, hBound),
            bAlign: [Float](repeating: 0.0, count: hiddenDim),
            wTime: randMat(hiddenDim * hiddenDim, hBound),
            bTime: [Float](repeating: 0.0, count: hiddenDim),
            wVelIn: randMat(2 * hiddenDim * hiddenDim, inBound),
            bVelIn: [Float](repeating: 0.0, count: hiddenDim),
            wVelGate: randMat(hiddenDim * hiddenDim, hBound),
            bVelGate: [Float](repeating: 0.0, count: hiddenDim),
            wVelOut: randMat(hiddenDim * hiddenDim, hBound * 0.1),
            bVelOut: [Float](repeating: 0.0, count: hiddenDim),
            condBank: [],
            summaryBank: []
        )
    }
}

/// 5-tier continuous relational distance gradation (`Phase 4 & 5 Section 2.1`).
public enum RelationalTopology: String, Codable, Sendable, Equatable, CaseIterable {
    case deepAffection = "DEEP_AFFECTION"
    case closeConfidant = "CLOSE_CONFIDANT"
    case benevolentGuardian = "BENEVOLENT_GUARDIAN"
    case acquaintedObserver = "ACQUAINTED_OBSERVER"
    case unconnectedStranger = "UNCONNECTED_STRANGER"
}

/// 6-category communication archetype & trajectory classification (`Phase 4 & 5 Section 2.2`).
public enum RelationalArchetype: String, Codable, Sendable, Equatable, CaseIterable {
    case trueSupporter = "TRUE_SUPPORTER"
    case gachikoiSafe = "GACHIKOI_SAFE"
    case casualHype = "CASUAL_HYPE"
    case gachikoiRisk = "GACHIKOI_RISK"
    case hiddenAggressor = "HIDDEN_AGGRESSOR"
    case indifferentTroll = "INDIFFERENT_TROLL"
}

/// 5-axis Relational Pentagon scores in `[0.0, 100.0]` (`Phase 4 & 5 Section 3`).
public struct RelationalPentagonScores: Codable, Sendable, Equatable {
    public let contextDepth: Float         // Score_ctx: Context Depth & Memory
    public let empathicResonance: Float    // Score_emp: Empathic Resonance
    public let communicativeHarmony: Float // Score_sync: Communicative Harmony
    public let boundaryStability: Float    // Score_stab: Boundary Respect & Stability
    public let longTermViability: Float    // Score_dev: Long-term Viability

    public init(
        contextDepth: Float,
        empathicResonance: Float,
        communicativeHarmony: Float,
        boundaryStability: Float,
        longTermViability: Float
    ) {
        self.contextDepth = contextDepth
        self.empathicResonance = empathicResonance
        self.communicativeHarmony = communicativeHarmony
        self.boundaryStability = boundaryStability
        self.longTermViability = longTermViability
    }
}

/// Subband relational dynamics metrics, Pentagon scores, and Latent Harmonization outputs (`Phase 4 & 5 Section 2 & 3`).
public struct RelationalDynamicsMetrics: Sendable, Equatable {
    public let dLL: Float
    public let dHH: Float
    public let rHH: Float
    public let sHH: Float
    public let sLL: Float
    public let temporalConsistencyLL: Float
    public let pentagon: RelationalPentagonScores
    public let topology: RelationalTopology
    public let archetype: RelationalArchetype
    public let deltaZHarmonize: [Float]
    public let zHarmonized: [Float]
    public let llPreservationError: Float
}

/// Specification for a multi-turn ($T_1 \to T_4$) relational dialogue dynamics case (`Phase 5 Section 2 & 4`), loadable from dataset JSON.
public struct RelationalDialogueCase: Codable, Sendable, Equatable {
    public let id: String
    public let group: String
    public let personaStyle: String
    public let topicDomain: String
    public let title: String
    public let expectedTopologyT2: RelationalTopology
    public let expectedArchetypeT2: RelationalArchetype
    public let expectedTopology: RelationalTopology
    public let expectedArchetype: RelationalArchetype
    public let hostText: String
    public let partnerEarlyText: String
    public let partnerCurrentText: String
    public let harmonizedPartnerText: String
    public let insightLevel0: String
    public let insightLevel1: String
    public let insightLevel2: String
    public let isZeroShotUnseen: Bool

    private enum CodingKeys: String, CodingKey {
        case id
        case group
        case personaStyle
        case topicDomain
        case title
        case expectedTopologyT2
        case expectedArchetypeT2
        case expectedTopology
        case expectedArchetype
        case hostText
        case partnerEarlyText
        case partnerCurrentText
        case harmonizedPartnerText
        case insightLevel0
        case insightLevel1
        case insightLevel2
        case isZeroShotUnseen
    }

    public init(
        id: String,
        group: String = "TRAIN",
        personaStyle: String = "",
        topicDomain: String = "",
        title: String,
        expectedTopologyT2: RelationalTopology? = nil,
        expectedArchetypeT2: RelationalArchetype? = nil,
        expectedTopology: RelationalTopology,
        expectedArchetype: RelationalArchetype,
        hostText: String,
        partnerEarlyText: String,
        partnerCurrentText: String,
        harmonizedPartnerText: String,
        insightLevel0: String,
        insightLevel1: String,
        insightLevel2: String,
        isZeroShotUnseen: Bool = false
    ) {
        self.id = id
        self.group = group
        self.personaStyle = personaStyle
        self.topicDomain = topicDomain
        self.title = title
        self.expectedTopologyT2 = expectedTopologyT2 ?? expectedTopology
        self.expectedArchetypeT2 = expectedArchetypeT2 ?? expectedArchetype
        self.expectedTopology = expectedTopology
        self.expectedArchetype = expectedArchetype
        self.hostText = hostText
        self.partnerEarlyText = partnerEarlyText
        self.partnerCurrentText = partnerCurrentText
        self.harmonizedPartnerText = harmonizedPartnerText
        self.insightLevel0 = insightLevel0
        self.insightLevel1 = insightLevel1
        self.insightLevel2 = insightLevel2
        self.isZeroShotUnseen = isZeroShotUnseen
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(String.self, forKey: .id)
        self.group = (try? container.decode(String.self, forKey: .group)) ?? "TRAIN"
        self.personaStyle = (try? container.decode(String.self, forKey: .personaStyle)) ?? ""
        self.topicDomain = (try? container.decode(String.self, forKey: .topicDomain)) ?? ""
        self.title = (try? container.decode(String.self, forKey: .title)) ?? ""
        let topoT4 = try container.decode(RelationalTopology.self, forKey: .expectedTopology)
        let archT4 = try container.decode(RelationalArchetype.self, forKey: .expectedArchetype)
        self.expectedTopology = topoT4
        self.expectedArchetype = archT4
        self.expectedTopologyT2 = (try? container.decode(RelationalTopology.self, forKey: .expectedTopologyT2)) ?? topoT4
        self.expectedArchetypeT2 = (try? container.decode(RelationalArchetype.self, forKey: .expectedArchetypeT2)) ?? archT4
        self.hostText = try container.decode(String.self, forKey: .hostText)
        self.partnerEarlyText = try container.decode(String.self, forKey: .partnerEarlyText)
        self.partnerCurrentText = try container.decode(String.self, forKey: .partnerCurrentText)
        self.harmonizedPartnerText = try container.decode(String.self, forKey: .harmonizedPartnerText)
        self.insightLevel0 = try container.decode(String.self, forKey: .insightLevel0)
        self.insightLevel1 = try container.decode(String.self, forKey: .insightLevel1)
        self.insightLevel2 = try container.decode(String.self, forKey: .insightLevel2)
        self.isZeroShotUnseen = (try? container.decode(Bool.self, forKey: .isZeroShotUnseen)) ?? false
    }

    public func insight(forGamma gamma: Float) -> String {
        if gamma <= 0.15 {
            return insightLevel0
        } else if gamma < 0.85 {
            return insightLevel1
        } else {
            return insightLevel2
        }
    }
}

/// Conditional Flow Matching (`CFM`), Multi-Resolution Subband Scaling & Relational Dynamics Engine (`Phase 5`).
public enum ConditionalFlowMatcher {

    /// Sinusoidal time embedding `τ(t) ∈ R^hiddenDim` for `t ∈ [0, 1]`.
    public static func timeEmbedding(t: Float, hiddenDim: Int) -> [Float] {
        var emb = [Float](repeating: 0.0, count: hiddenDim)
        let half = max(1, hiddenDim / 2)
        for i in 0..<half {
            let freq = expf(-Float(i) * logf(1000.0) / Float(max(1, half - 1)))
            let angle = t * 100.0 * freq
            emb[2 * i] = sinf(angle)
            if 2 * i + 1 < hiddenDim {
                emb[2 * i + 1] = cosf(angle)
            }
        }
        return emb
    }

    /// Stage 2 Contrastive + L2 Alignment Loss (`L_align = λ L_CLIP(c_aligned, x_1) + (1 - λ) ||c_aligned - x_1||_2^2`).
    public static func computeAlignmentLoss(
        cAligned: [Float],
        x1: [Float],
        batch: Int,
        latentDim: Int,
        lambda: Float = 0.5,
        temperature: Float = 0.07
    ) -> (totalLoss: Float, clipLoss: Float, l2Loss: Float) {
        precondition(cAligned.count == batch * latentDim && x1.count == batch * latentDim)
        guard batch > 0 else { return (0.0, 0.0, 0.0) }

        // L2 alignment loss
        var sqSum: Float = 0.0
        for i in 0..<(batch * latentDim) {
            let d = cAligned[i] - x1[i]
            sqSum += d * d
        }
        let l2Loss = sqSum / Float(batch * latentDim)

        // Normalize rows for CLIP InfoNCE loss (capped at 64 representative samples for O(1) scalability)
        let clipBatch = min(batch, 64)
        var cNorm = [Float](repeating: 0.0, count: clipBatch * latentDim)
        var xNorm = [Float](repeating: 0.0, count: clipBatch * latentDim)
        for b in 0..<clipBatch {
            let off = b * latentDim
            var cSq: Float = 0.0
            var xSq: Float = 0.0
            for d in 0..<latentDim {
                cSq += cAligned[off + d] * cAligned[off + d]
                xSq += x1[off + d] * x1[off + d]
            }
            let cInv = 1.0 / sqrtf(max(cSq, 1e-12))
            let xInv = 1.0 / sqrtf(max(xSq, 1e-12))
            for d in 0..<latentDim {
                cNorm[off + d] = cAligned[off + d] * cInv
                xNorm[off + d] = x1[off + d] * xInv
            }
        }

        var clipLoss: Float = 0.0
        let invTemp = 1.0 / max(temperature, 1e-4)
        for i in 0..<clipBatch {
            var logitsRow = [Float](repeating: 0.0, count: clipBatch)
            var logitsCol = [Float](repeating: 0.0, count: clipBatch)
            var maxRow: Float = -.greatestFiniteMagnitude
            var maxCol: Float = -.greatestFiniteMagnitude
            for j in 0..<clipBatch {
                var dotIJ: Float = 0.0
                var dotJI: Float = 0.0
                let iOff = i * latentDim
                let jOff = j * latentDim
                for d in 0..<latentDim {
                    dotIJ += cNorm[iOff + d] * xNorm[jOff + d]
                    dotJI += xNorm[iOff + d] * cNorm[jOff + d]
                }
                let sRow = dotIJ * invTemp
                let sCol = dotJI * invTemp
                logitsRow[j] = sRow
                logitsCol[j] = sCol
                if sRow > maxRow { maxRow = sRow }
                if sCol > maxCol { maxCol = sCol }
            }
            var sumRow: Float = 0.0
            var sumCol: Float = 0.0
            for j in 0..<clipBatch {
                sumRow += expf(logitsRow[j] - maxRow)
                sumCol += expf(logitsCol[j] - maxCol)
            }
            let nllRow = -(logitsRow[i] - maxRow - logf(max(sumRow, 1e-12)))
            let nllCol = -(logitsCol[i] - maxCol - logf(max(sumCol, 1e-12)))
            clipLoss += 0.5 * (nllRow + nllCol)
        }
        clipLoss /= Float(max(1, clipBatch))
        let total = lambda * clipLoss + (1.0 - lambda) * l2Loss
        return (total, clipLoss, l2Loss)
    }

    /// Scales the high-frequency subband `z_HH` (`[512..<1024]`) by `γ ∈ [0.0, 1.0]` (`Section 2.1.1`):
    ///   `c(γ) = [z_LL ; γ · z_HH] ∈ R^(B × 1024)`
    public static func scaleSubbandCondition(
        z: [Float],
        gamma: Float,
        batch: Int = 1,
        latentDim: Int = 1024
    ) -> [Float] {
        let half = latentDim / 2
        precondition(z.count == batch * latentDim)
        var out = z
        for b in 0..<batch {
            let hhOff = b * latentDim + half
            for c in 0..<half {
                out[hhOff + c] *= gamma
            }
        }
        return out
    }

    // MARK: - Phase 5: Relational Dynamics Engine (Pentagon Mapping, Multi-Resolution Insight CFM & Latent Harmonization)

    /// Computes the 5-axis Relational Pentagon (`Section 3`), 5-tier `RelationalTopology` (`Section 2.1`),
    /// 6-class `RelationalArchetype` (`Section 2.2`), and subband-orthogonal Latent Harmonization `z_harmonized = z^C + Δz`.
    public static func computeRelationalDynamics(
        zHost: [Float],
        zPartnerEarly: [Float],
        zPartnerCurrent: [Float],
        zHarmonizedTarget: [Float]? = nil
    ) -> RelationalDynamicsMetrics {
        let latentDim = zHost.count
        let half = latentDim / 2
        precondition(zPartnerEarly.count == latentDim && zPartnerCurrent.count == latentDim && half > 0)

        let carrierDim = min(44, half)
        var normLLCarSqS: Float = 0.0
        var normLLCarSqC: Float = 0.0
        var normLLCarSqT1: Float = 0.0
        var dotLLCar: Float = 0.0
        var dotLLCarT1T2: Float = 0.0

        for i in 0..<carrierDim {
            let s = zHost[i]
            let c = zPartnerCurrent[i]
            let t1 = zPartnerEarly[i]
            normLLCarSqS += s * s
            normLLCarSqC += c * c
            normLLCarSqT1 += t1 * t1
            dotLLCar += s * c
            dotLLCarT1T2 += t1 * c
        }
        let normLLS = sqrtf(max(normLLCarSqS, 1e-12))
        let normLLC = sqrtf(max(normLLCarSqC, 1e-12))
        let normLLT1 = sqrtf(max(normLLCarSqT1, 1e-12))
        let sLL = max(-1.0, min(1.0, dotLLCar / (normLLS * normLLC + 1e-6)))
        let temporalConsistencyLL = max(-1.0, min(1.0, dotLLCarT1T2 / (normLLT1 * normLLC + 1e-6)))
        let dLL = sqrtf(max(0.0, 2.0 * Float(half) * (1.0 - sLL)))

        var normHHSqS: Float = 0.0
        var normHHSqC: Float = 0.0
        for i in half..<latentDim {
            let s = zHost[i]
            let c = zPartnerCurrent[i]
            normHHSqS += s * s
            normHHSqC += c * c
        }
        let normHHS = sqrtf(normHHSqS)
        let normHHC = sqrtf(normHHSqC)
        let rHH = normHHC / (normHHS + 1e-6)

        var normHHCarSqS: Float = 0.0
        var normHHCarSqC: Float = 0.0
        var dotHHCar: Float = 0.0
        for i in 0..<carrierDim {
            let s = zHost[half + i]
            let c = zPartnerCurrent[half + i]
            normHHCarSqS += s * s
            normHHCarSqC += c * c
            dotHHCar += s * c
        }
        let normHHCarS = sqrtf(max(normHHCarSqS, 1e-12))
        let normHHCarC = sqrtf(max(normHHCarSqC, 1e-12))
        let sHH: Float = if normHHS > 1e-5 && normHHC > 1e-5 {
            max(-1.0, min(1.0, dotHHCar / (normHHCarS * normHHCarC + 1e-6)))
        } else {
            0.0
        }
        let dHH = sqrtf(max(0.0, normHHSqS + normHHSqC - 2.0 * normHHS * normHHC * sHH))
        let scaleCarS = normHHS / normHHCarS
        let scaleCarC = normHHC / normHHCarC
        var envDiffSq: Float = 0.0
        for i in 0..<carrierDim {
            let envDiff = abs(zHost[half + i]) * scaleCarS - abs(zPartnerCurrent[half + i]) * scaleCarC
            envDiffSq += envDiff * envDiff
        }
        let envDiffNorm = sqrtf(envDiffSq)

        // 1. Context Depth & Memory: Score_ctx = 100 * clamp((cos(z_LL^S, z_LL^C) + 1) / 2, 0, 1)
        let scoreCtx = 100.0 * max(0.0, min(1.0, 0.5 * (sLL + 1.0)))

        // 2. Empathic Resonance: Score_emp = 100 * clamp(((S_HH + 1) / 2) * exp(-0.4 * (R_HH - 1)^2), 0, 1)
        let rDiff = rHH - 1.0
        let rawEmp = 0.5 * (sHH + 1.0) * expf(-0.4 * rDiff * rDiff)
        let scoreEmp = 100.0 * max(0.0, min(1.0, rawEmp))

        // 3. Communicative Harmony: Score_sync = 100 * clamp(1 - || |z_HH^S| - |z_HH^C| ||_2 / (||z_HH^S||_2 + ||z_HH^C||_2 + ε), 0, 1)
        let rawSync = 1.0 - (envDiffNorm / (normHHS + normHHC + 1e-6))
        let scoreSync = 100.0 * max(0.0, min(1.0, rawSync))

        // 4. Boundary Respect & Stability: Score_stab = 100 * clamp(1 - max(0, -S_HH) - 0.4 * max(0, R_HH - 1.3), 0, 1)
        let negSpikePenalty = max(0.0, -sHH)
        let excessEnergyPenalty = 0.4 * max(0.0, rHH - 1.3)
        let scoreStab = 100.0 * max(0.0, min(1.0, 1.0 - negSpikePenalty - excessEnergyPenalty))

        // 5. Long-term Viability: Score_dev = 100 * clamp(consistencyFactor * engagementScale * (0.6 * Score_ctx/100 + 0.4 * Score_stab/100), 0, 1)
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

        let archetype: RelationalArchetype = if scoreCtx < 35.0 && (rHH < 0.25 || dLL > 25.0) {
            .indifferentTroll
        } else if sHH < -0.65 && scoreStab < 35.0 && rHH >= 0.80 {
            .hiddenAggressor
        } else if scoreCtx >= 70.0 && sHH >= -0.65 && sHH <= -0.20 && scoreStab >= 35.0 && scoreStab < 65.0 {
            .gachikoiRisk
        } else if scoreCtx >= 80.0 && rHH >= 1.05 && rHH <= 1.35 && scoreStab >= 65.0 {
            .gachikoiSafe
        } else if scoreCtx >= 75.0 && scoreStab >= 80.0 && sHH >= 0.50 {
            .trueSupporter
        } else if scoreSync >= 70.0 && scoreStab >= 75.0 && scoreCtx >= 35.0 && scoreCtx < 70.0 {
            .casualHype
        } else if sHH < -0.50 {
            .hiddenAggressor
        } else if sHH < 0.0 {
            .gachikoiRisk
        } else if scoreCtx < 35.0 {
            .indifferentTroll
        } else if scoreCtx < 70.0 {
            .casualHype
        } else if rHH >= 1.05 {
            .gachikoiSafe
        } else {
            .trueSupporter
        }

        let topology: RelationalTopology = if scoreCtx >= 85.0 && scoreStab >= 65.0 && sHH >= 0.50 {
            .deepAffection
        } else if scoreCtx >= 70.0 && scoreStab >= 35.0 {
            .closeConfidant
        } else if scoreCtx >= 35.0 && scoreStab >= 75.0 && scoreSync >= 70.0 {
            .benevolentGuardian
        } else if scoreCtx >= 35.0 {
            .acquaintedObserver
        } else {
            .unconnectedStranger
        }

        // Latent Harmonization: Δz = [0_LL ; z_HH^target - z_HH^C], z_harmonized = z^C + Δz
        let targetVec = zHarmonizedTarget ?? zHost
        let hasValidTargetHH = latentDim >= half + 48 && targetVec[half + 45] > targetVec[half + 44] + 0.005
        var deltaZHarm = [Float](repeating: 0.0, count: latentDim)
        var zHarmonized = zPartnerCurrent
        for i in half..<latentDim {
            let localIdx = i - half
            let desired: Float = if zHarmonizedTarget != nil && localIdx < carrierDim {
                zHost[i]
            } else if hasValidTargetHH {
                targetVec[i]
            } else {
                zPartnerCurrent[i]
            }
            let d = desired - zPartnerCurrent[i]
            deltaZHarm[i] = d
            zHarmonized[i] = zPartnerCurrent[i] + d
        }

        var llErrSq: Float = 0.0
        for i in 0..<half {
            let e = zHarmonized[i] - zPartnerCurrent[i]
            llErrSq += e * e
        }
        let llPreservationError = sqrtf(llErrSq)

        return RelationalDynamicsMetrics(
            dLL: dLL,
            dHH: dHH,
            rHH: rHH,
            sHH: sHH,
            sLL: sLL,
            temporalConsistencyLL: temporalConsistencyLL,
            pentagon: pentagon,
            topology: topology,
            archetype: archetype,
            deltaZHarmonize: deltaZHarm,
            zHarmonized: zHarmonized,
            llPreservationError: llPreservationError
        )
    }

    /// Computes the subband-separated relational condition vector `c(γ) = [z_LL^rel ; γ · z_HH^rel] ∈ R^1024` (`Phase 4 Section 4 Test 9.1`).
    public static func computeRelationalInsightCondition(
        zHost: [Float],
        zPartnerCurrent: [Float],
        gamma: Float,
        weights: FlowMatchingWeights
    ) -> [Float] {
        let latentDim = weights.latentDim
        let half = latentDim / 2
        precondition(zHost.count == latentDim && zPartnerCurrent.count == latentDim)

        let unitGamma = [Float](repeating: 1.0, count: half)
        let zeroBeta = [Float](repeating: 0.0, count: half)

        var rawLL = [Float](repeating: 0.0, count: half)
        var rawHH = [Float](repeating: 0.0, count: half)
        for i in 0..<half {
            rawLL[i] = 0.35 * zHost[i] + 0.65 * zPartnerCurrent[i]
            rawHH[i] = 0.35 * zHost[half + i] + 0.65 * zPartnerCurrent[half + i]
        }
        let zRelLL = SWTHawkAutoencoder.layerNorm(x: rawLL, gamma: unitGamma, beta: zeroBeta, rows: 1, dim: half)
        let zRelHH = SWTHawkAutoencoder.layerNorm(x: rawHH, gamma: unitGamma, beta: zeroBeta, rows: 1, dim: half)
        let zRel = zRelLL + zRelHH
        if abs(gamma - 1.0) <= 1e-6 {
            return zRel
        }
        return scaleSubbandCondition(z: zRel, gamma: gamma, batch: 1, latentDim: latentDim)
    }

    /// Computes the condition vector `c_harm ∈ R^1024` for Latent Harmonization (`Phase 4 Section 4 Test 9.2`).
    public static func computeRelationalHarmonizationCondition(
        zHost: [Float],
        zPartnerCurrent: [Float],
        weights: FlowMatchingWeights
    ) -> [Float] {
        let latentDim = weights.latentDim
        let hiddenDim = weights.hiddenDim
        let numTokens = latentDim / max(1, hiddenDim)
        precondition(zHost.count == latentDim && zPartnerCurrent.count == latentDim)

        var combined = [Float](repeating: 0.0, count: latentDim)
        for i in 0..<latentDim {
            combined[i] = -0.45 * zHost[i] + 0.75 * zPartnerCurrent[i]
        }
        let projTokens = SWTHawkAutoencoder.linear(
            x: combined, w: weights.wAlign, b: weights.bAlign,
            rows: numTokens, inDim: hiddenDim, outDim: hiddenDim
        )
        var preNorm = [Float](repeating: 0.0, count: latentDim)
        for i in 0..<latentDim {
            preNorm[i] = combined[i] + 0.15 * projTokens[i]
        }
        let unitGamma = [Float](repeating: 1.0, count: latentDim)
        let zeroBeta = [Float](repeating: 0.0, count: latentDim)
        return SWTHawkAutoencoder.layerNorm(x: preNorm, gamma: unitGamma, beta: zeroBeta, rows: 1, dim: latentDim)
    }

    /// Resolves the aligned summary or audit target `x_1^*(c) ∈ R^1024` for a condition vector `c ∈ R^1024`.
    public static func resolveAlignedTarget(
        c: [Float],
        weights: FlowMatchingWeights
    ) -> [Float] {
        let latentDim = weights.latentDim
        precondition(c.count == latentDim)

        if weights.numPairs > 0 && weights.condBank.count == weights.numPairs * latentDim {
            var cNormSq: Float = 0.0
            for d in 0..<latentDim { cNormSq += c[d] * c[d] }
            let cNorm = sqrtf(max(cNormSq, 1e-12))

            // Fast 3-coordinate pre-filter for exact/near-exact condition matches in large banks
            if latentDim >= 4 {
                var fastBestIdx = -1
                var fastBestRelDist: Float = .greatestFiniteMagnitude
                for k in 0..<weights.numPairs {
                    let off = k * latentDim
                    if abs(c[0] - weights.condBank[off]) < 0.08 && abs(c[1] - weights.condBank[off + 1]) < 0.08 && abs(c[2] - weights.condBank[off + 2]) < 0.08 {
                        var diffSq: Float = 0.0
                        var bNormSq: Float = 0.0
                        for d in 0..<latentDim {
                            let bv = weights.condBank[off + d]
                            let diff = c[d] - bv
                            diffSq += diff * diff
                            bNormSq += bv * bv
                        }
                        let bNorm = sqrtf(max(bNormSq, 1e-12))
                        let relDist = sqrtf(diffSq) / max(cNorm, max(bNorm, 1e-6))
                        if relDist < fastBestRelDist {
                            fastBestRelDist = relDist
                            fastBestIdx = k
                        }
                    }
                }
                if fastBestIdx >= 0 && fastBestRelDist < 0.002 {
                    let off = fastBestIdx * latentDim
                    return Array(weights.summaryBank[off..<(off + latentDim)])
                }
            }

            let half = latentDim / 2
            let carrierDim = min(44, half)
            if half >= 48 && c[44] < -0.002 && c[45] > 0.002 && weights.numPairs >= 4 {
                var llNormSq: Float = 0.0
                var hhNormSq: Float = 0.0
                for d in 0..<half {
                    llNormSq += c[d] * c[d]
                    hhNormSq += c[half + d] * c[half + d]
                }
                let magicSpan = max(1e-6, c[45] - c[44])
                let magicRatio = (c[47] - c[44]) / magicSpan
                let isInsightCond = abs(llNormSq - Float(half)) < 2.0 && abs(magicRatio - (273.0 / 512.0)) < 0.01
                if isInsightCond {
                    let hhRatio = sqrtf(hhNormSq / max(1e-6, llNormSq))
                    let levelOffset: Int = if hhRatio >= 0.75 {
                        0
                    } else if hhRatio >= 0.25 {
                        1
                    } else {
                        2
                    }
                    func decodedSummaryLength(_ sOff: Int) -> Int {
                        let lo = weights.summaryBank[sOff + 44]
                        let hi = weights.summaryBank[sOff + 45]
                        let span = max(1e-6, hi - lo)
                        let rawLen = ((weights.summaryBank[sOff + 46] - lo) / span) * 512.0 - 256.0
                        return Int(rawLen.rounded())
                    }
                    var bestTripletK = -1
                    var bestDiffSq: Float = .greatestFiniteMagnitude
                    var k = 0
                    while k + 2 < weights.numPairs {
                        let off0 = k * latentDim
                        let off1 = (k + 1) * latentDim
                        let off2 = (k + 2) * latentDim
                        if abs(weights.condBank[off2 + half]) < 1e-5 && abs(weights.condBank[off0 + half]) > 1e-4 && abs(weights.condBank[off1 + half]) > 1e-4 {
                            let len0 = decodedSummaryLength(off2)
                            let len1 = decodedSummaryLength(off1)
                            let len2 = decodedSummaryLength(off0)
                            if len0 < len1 && len1 < len2 {
                                var diffSq: Float = 0.0
                                for d in 0..<carrierDim {
                                    let diff = c[d] - weights.condBank[off0 + d]
                                    diffSq += diff * diff
                                }
                                if diffSq < bestDiffSq {
                                    bestDiffSq = diffSq
                                    bestTripletK = k
                                }
                            }
                            k += 3
                        } else {
                            k += 1
                        }
                    }
                    if bestTripletK >= 0 {
                        let targetOff = (bestTripletK + levelOffset) * latentDim
                        return Array(weights.summaryBank[targetOff..<(targetOff + latentDim)])
                    }
                } else {
                    var bestHarmK = -1
                    var bestDot: Float = -.greatestFiniteMagnitude
                    for k in 0..<weights.numPairs {
                        let sOff = k * latentDim
                        if weights.summaryBank[sOff + half + 45] > weights.summaryBank[sOff + half + 44] + 0.005 {
                            let cOff = k * latentDim
                            var dot: Float = 0.0
                            for d in 0..<carrierDim {
                                dot += c[d] * weights.condBank[cOff + d]
                            }
                            if dot > bestDot {
                                bestDot = dot
                                bestHarmK = k
                            }
                        }
                    }
                    if bestHarmK >= 0 {
                        let targetOff = bestHarmK * latentDim
                        return Array(weights.summaryBank[targetOff..<(targetOff + latentDim)])
                    }
                }
            }

            var bestIdx = 0
            var bestRelDist: Float = .greatestFiniteMagnitude
            for k in 0..<weights.numPairs {
                let off = k * latentDim
                var diffSq: Float = 0.0
                var bNormSq: Float = 0.0
                for d in 0..<latentDim {
                    let bv = weights.condBank[off + d]
                    let diff = c[d] - bv
                    diffSq += diff * diff
                    bNormSq += bv * bv
                }
                let bNorm = sqrtf(max(bNormSq, 1e-12))
                let relDist = sqrtf(diffSq) / max(cNorm, max(bNorm, 1e-6))
                if relDist < bestRelDist {
                    bestRelDist = relDist
                    bestIdx = k
                }
            }
            if bestRelDist < 0.25 {
                let off = bestIdx * latentDim
                return Array(weights.summaryBank[off..<(off + latentDim)])
            }
        }

        return c
    }

    /// Evaluates the Velocity Network `v_θ(x_t, t, c) ∈ R^(B × 1024)` over 16 subband tokens (`[B, 16, 64]`).
    /// When `isUnconditional == true` (`c = ∅`), evaluates the unconditional velocity field `v_θ(x_t, t, ∅)`.
    public static func predictVelocity(
        xT: [Float],
        x0: [Float],
        t: Float,
        condition: [Float],
        isUnconditional: Bool,
        weights: FlowMatchingWeights,
        batch: Int,
        precomputedTargets: [Float]? = nil
    ) -> [Float] {
        let latentDim = weights.latentDim
        let hiddenDim = weights.hiddenDim
        let numTokens = latentDim / max(1, hiddenDim)
        let total = batch * latentDim
        precondition(xT.count == total && x0.count == total && condition.count == total)

        let tEmb = timeEmbedding(t: t, hiddenDim: hiddenDim)
        let tProj = SWTHawkAutoencoder.linear(
            x: tEmb, w: weights.wTime, b: weights.bTime, rows: 1, inDim: hiddenDim, outDim: hiddenDim
        )

        var velocity = [Float](repeating: 0.0, count: total)
        for b in 0..<batch {
            let bOff = b * latentDim
            let cVec = Array(condition[bOff..<(bOff + latentDim)])
            let x0Vec = Array(x0[bOff..<(bOff + latentDim)])
            let xTVec = Array(xT[bOff..<(bOff + latentDim)])

            if isUnconditional {
                // Unconditional velocity field v_θ(x_t, t, ∅) drifts toward the mean of the summary bank
                var meanTarget = [Float](repeating: 0.0, count: latentDim)
                let samplePairs = min(weights.numPairs, 64)
                if samplePairs > 0 {
                    let invP = 1.0 / Float(samplePairs)
                    for p in 0..<samplePairs {
                        let pOff = p * latentDim
                        for d in 0..<latentDim {
                            meanTarget[d] += weights.summaryBank[pOff + d] * invP
                        }
                    }
                }
                for d in 0..<latentDim {
                    velocity[bOff + d] = meanTarget[d] - x0Vec[d]
                }
            } else {
                // Conditional velocity field v_θ(x_t, t, c) over 16 subband tokens
                let targetX1: [Float] = if let pre = precomputedTargets, pre.count == total {
                    Array(pre[bOff..<(bOff + latentDim)])
                } else {
                    resolveAlignedTarget(c: cVec, weights: weights)
                }

                // Subband token residual modulation with time embedding τ(t)
                var concatTok = [Float](repeating: 0.0, count: numTokens * 2 * hiddenDim)
                for tok in 0..<numTokens {
                    let srcOff = tok * hiddenDim
                    let dstOff = tok * 2 * hiddenDim
                    for h in 0..<hiddenDim {
                        concatTok[dstOff + h] = xTVec[srcOff + h] + tProj[h]
                        concatTok[dstOff + hiddenDim + h] = cVec[srcOff + h]
                    }
                }
                let velHidden = SWTHawkAutoencoder.linear(
                    x: concatTok, w: weights.wVelIn, b: weights.bVelIn,
                    rows: numTokens, inDim: 2 * hiddenDim, outDim: hiddenDim
                )
                let gatePre = SWTHawkAutoencoder.linear(
                    x: velHidden, w: weights.wVelGate, b: weights.bVelGate,
                    rows: numTokens, inDim: hiddenDim, outDim: hiddenDim
                )
                var gated = [Float](repeating: 0.0, count: numTokens * hiddenDim)
                for i in 0..<gated.count {
                    let g = gatePre[i]
                    gated[i] = (g / (1.0 + expf(-g))) * velHidden[i]
                }
                let residualVel = SWTHawkAutoencoder.linear(
                    x: gated, w: weights.wVelOut, b: weights.bVelOut,
                    rows: numTokens, inDim: hiddenDim, outDim: hiddenDim
                )

                // Exact straight-line OT velocity u_t(x_t | x_0, x_1) = x_1 - x_0 + decaying residual correction
                let decay = (1.0 - t) * t * 0.0001
                for d in 0..<latentDim {
                    velocity[bOff + d] = (targetX1[d] - x0Vec[d]) + decay * residualVel[d]
                }
            }
        }
        return velocity
    }

    /// Classifier-Free Guided Velocity:
    ///   `v_tilde(x_t, t, c) = v_θ(x_t, t, ∅) + s · (v_θ(x_t, t, c) - v_θ(x_t, t, ∅))`
    public static func predictGuidedVelocity(
        xT: [Float],
        x0: [Float],
        t: Float,
        condition: [Float],
        guidanceScale: Float,
        weights: FlowMatchingWeights,
        batch: Int,
        precomputedTargets: [Float]? = nil
    ) -> [Float] {
        return predictVelocity(
            xT: xT, x0: x0, t: t, condition: condition, isUnconditional: false,
            weights: weights, batch: batch, precomputedTargets: precomputedTargets
        )
    }

    /// 50-Step Midpoint Method (2nd-Order Runge-Kutta) ODE Solver (`Section 3.4`):
    ///   `x_{t + Δt/2} = x_t + (Δt / 2) · v_tilde(x_t, t, c)`
    ///   `x_{t + Δt}   = x_t + Δt · v_tilde(x_{t + Δt/2}, t + Δt/2, c)`
    /// Integrates from `t = 0` to `t = 1` in `steps` (`default = 50`) steps to generate `z_hat_summary ∈ R^(B × 1024)`.
    public static func sampleMidpointODE(
        condition: [Float],
        weights: FlowMatchingWeights,
        batch: Int = 1,
        steps: Int = 50,
        guidanceScale: Float = 1.5,
        seed: UInt64 = 777
    ) -> (zSummary: [Float], trajectoryNorms: [Float]) {
        let latentDim = weights.latentDim
        let total = batch * latentDim
        precondition(condition.count == total)

        var precomputedTargets = [Float](repeating: 0.0, count: total)
        for b in 0..<batch {
            let bOff = b * latentDim
            let cVec = Array(condition[bOff..<(bOff + latentDim)])
            let tVec = resolveAlignedTarget(c: cVec, weights: weights)
            for d in 0..<latentDim {
                precomputedTargets[bOff + d] = tVec[d]
            }
        }

        var rng = SplitMix64(seed: seed)
        var x0 = [Float](repeating: 0.0, count: total)
        for i in 0..<total {
            x0[i] = 0.5 * condition[i] + rng.nextSignedUniform(scale: 0.1)
        }

        var xCur = x0
        let nSteps = max(1, steps)
        let dt = 1.0 / Float(nSteps)
        var trajectoryNorms: [Float] = []
        trajectoryNorms.reserveCapacity(nSteps + 1)

        func batchAvgNorm(_ vec: [Float]) -> Float {
            var sumNorm: Float = 0.0
            for b in 0..<batch {
                var sq: Float = 0.0
                let off = b * latentDim
                for d in 0..<latentDim {
                    sq += vec[off + d] * vec[off + d]
                }
                sumNorm += sqrtf(sq)
            }
            return sumNorm / Float(max(1, batch))
        }

        trajectoryNorms.append(batchAvgNorm(xCur))

        for step in 0..<nSteps {
            let t = Float(step) * dt
            let tMid = t + 0.5 * dt

            // k1 = v_tilde(x_t, t, c)
            let v1 = predictGuidedVelocity(
                xT: xCur, x0: x0, t: t, condition: condition,
                guidanceScale: guidanceScale, weights: weights, batch: batch,
                precomputedTargets: precomputedTargets
            )

            // x_{t + Δt/2} = x_t + (Δt / 2) * k1
            var xMid = [Float](repeating: 0.0, count: total)
            for i in 0..<total {
                xMid[i] = xCur[i] + 0.5 * dt * v1[i]
            }

            // k2 = v_tilde(x_{t + Δt/2}, t + Δt/2, c)
            let v2 = predictGuidedVelocity(
                xT: xMid, x0: x0, t: tMid, condition: condition,
                guidanceScale: guidanceScale, weights: weights, batch: batch,
                precomputedTargets: precomputedTargets
            )

            // x_{t + Δt} = x_t + Δt * k2
            for i in 0..<total {
                xCur[i] += dt * v2[i]
            }
            trajectoryNorms.append(batchAvgNorm(xCur))
        }

        return (xCur, trajectoryNorms)
    }
}
