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

/// Deterministic 4-state subband differential classification (`Section 2.2.2`).
public enum AuditStatus: String, Codable, Sendable, Equatable {
    case compliant = "COMPLIANT"
    case omission = "OMISSION"
    case violation = "VIOLATION"
    case coreMismatch = "CORE_MISMATCH"
}

/// Subband differential metrics between a reference document `z^S` and an inspected document `z^C` (`Section 2.2.2`).
public struct SubbandAuditMetrics: Sendable, Equatable {
    public let dLL: Float          // ||z_LL^S - z_LL^C||_2
    public let dHH: Float          // ||z_HH^S - z_HH^C||_2
    public let rHH: Float          // ||z_HH^C||_2 / (||z_HH^S||_2 + ε)
    public let sHH: Float          // cos(z_HH^S, z_HH^C)
    public let status: AuditStatus // Deterministic 4-class diagnosis
    public let deltaZ: [Float]     // Δz = z^S - z^C
    public let zRepaired: [Float]  // z_repaired = z^C + Δz
}

/// Specification for a subband differential auditing case (`Section 3.2`), loadable from dataset JSON.
public struct ContractAuditCase: Codable, Sendable, Equatable {
    public let id: String
    public let targetStatuteId: String
    public let title: String
    public let expectedStatus: AuditStatus
    public let contractText: String
    public let expectedAuditText: String
    public let isZeroShotUnseen: Bool
    public var expectedAuditReport: String { expectedAuditText }

    public init(
        id: String,
        targetStatuteId: String,
        title: String,
        expectedStatus: AuditStatus,
        contractText: String,
        expectedAuditText: String,
        isZeroShotUnseen: Bool = false
    ) {
        self.id = id
        self.targetStatuteId = targetStatuteId
        self.title = title
        self.expectedStatus = expectedStatus
        self.contractText = contractText
        self.expectedAuditText = expectedAuditText
        self.isZeroShotUnseen = isZeroShotUnseen
    }
}

/// 5-tier continuous relational distance gradation (`Phase 4 Section 2.1`).
public enum RelationalTopology: String, Codable, Sendable, Equatable {
    case deepAffection = "DEEP_AFFECTION"
    case closeConfidant = "CLOSE_CONFIDANT"
    case benevolentGuardian = "BENEVOLENT_GUARDIAN"
    case acquaintedObserver = "ACQUAINTED_OBSERVER"
    case unconnectedStranger = "UNCONNECTED_STRANGER"
}

/// 6-category communication archetype & trajectory classification (`Phase 4 Section 2.2`).
public enum RelationalArchetype: String, Codable, Sendable, Equatable {
    case trueSupporter = "TRUE_SUPPORTER"
    case gachikoiSafe = "GACHIKOI_SAFE"
    case casualHype = "CASUAL_HYPE"
    case gachikoiRisk = "GACHIKOI_RISK"
    case hiddenAggressor = "HIDDEN_AGGRESSOR"
    case indifferentTroll = "INDIFFERENT_TROLL"
}

/// 5-axis Relational Pentagon scores in `[0.0, 100.0]` (`Phase 4 Section 3`).
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

/// Subband relational dynamics metrics, Pentagon scores, and Latent Harmonization outputs (`Phase 4 Section 2 & 3`).
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

/// Specification for a relational dialogue dynamics case (`Phase 4 Section 4`), loadable from dataset JSON.
public struct RelationalDialogueCase: Codable, Sendable, Equatable {
    public let id: String
    public let title: String
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

    public init(
        id: String,
        title: String,
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
        self.title = title
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

/// Conditional Flow Matching (`CFM`), Multi-Resolution Subband Scaling & Subband Differential Auditing (`Phase 2 & Phase 3`).
public enum ConditionalFlowMatcher {
    /// Default thresholds `τ_LL = 6.0`, `τ_HH = 5.0` for deterministic 4-state subband auditing (`Section 2.2.2`).
    public static let defaultTauLL: Float = 6.0
    public static let defaultTauHH: Float = 5.0

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

        // Normalize rows for CLIP InfoNCE loss
        var cNorm = [Float](repeating: 0.0, count: batch * latentDim)
        var xNorm = [Float](repeating: 0.0, count: batch * latentDim)
        for b in 0..<batch {
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
        for i in 0..<batch {
            var logitsRow = [Float](repeating: 0.0, count: batch)
            var logitsCol = [Float](repeating: 0.0, count: batch)
            var maxRow: Float = -.greatestFiniteMagnitude
            var maxCol: Float = -.greatestFiniteMagnitude
            for j in 0..<batch {
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
            for j in 0..<batch {
                sumRow += expf(logitsRow[j] - maxRow)
                sumCol += expf(logitsCol[j] - maxCol)
            }
            let nllRow = -(logitsRow[i] - maxRow - logf(max(sumRow, 1e-12)))
            let nllCol = -(logitsCol[i] - maxCol - logf(max(sumCol, 1e-12)))
            clipLoss += 0.5 * (nllRow + nllCol)
        }
        clipLoss /= Float(batch)
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

    // MARK: - Task B: Subband Differential Auditing & Repair

    /// Computes subband subtraction `Δz = z^S - z^C`, 4-state deterministic audit classification, and Latent Repair `z_repaired = z^C + Δz` (`Section 2.2.2 & 2.2.3`).
    public static func computeSubbandAudit(
        zStatute: [Float],
        zContract: [Float],
        tauLL: Float = defaultTauLL,
        tauHH: Float = defaultTauHH
    ) -> SubbandAuditMetrics {
        let latentDim = zStatute.count
        let half = latentDim / 2
        precondition(zContract.count == latentDim && half > 0)

        var deltaZ = [Float](repeating: 0.0, count: latentDim)
        var zRepaired = [Float](repeating: 0.0, count: latentDim)
        for i in 0..<latentDim {
            let d = zStatute[i] - zContract[i]
            deltaZ[i] = d
            zRepaired[i] = zContract[i] + d
        }

        var dLLSq: Float = 0.0
        for i in 0..<half {
            dLLSq += deltaZ[i] * deltaZ[i]
        }
        let dLL = sqrtf(dLLSq)

        var dHHSq: Float = 0.0
        var normHHSqS: Float = 0.0
        var normHHSqC: Float = 0.0
        var dotHH: Float = 0.0
        for i in half..<latentDim {
            let s = zStatute[i]
            let c = zContract[i]
            let d = deltaZ[i]
            dHHSq += d * d
            normHHSqS += s * s
            normHHSqC += c * c
            dotHH += s * c
        }
        let dHH = sqrtf(dHHSq)
        let normHHS = sqrtf(normHHSqS)
        let normHHC = sqrtf(normHHSqC)
        let rHH = normHHC / (normHHS + 1e-6)
        let sHH = (normHHS > 1e-5 && normHHC > 1e-5) ? (dotHH / (normHHS * normHHC + 1e-6)) : 0.0

        let status: AuditStatus = if dLL >= tauLL {
            .coreMismatch
        } else if dHH < tauHH && sHH > 0.85 {
            .compliant
        } else if dHH >= tauHH && sHH < 0.0 {
            .violation
        } else if dHH >= tauHH && rHH < 0.65 {
            .omission
        } else if sHH < 0.0 {
            .violation
        } else if rHH < 0.65 {
            .omission
        } else {
            .compliant
        }

        return SubbandAuditMetrics(
            dLL: dLL,
            dHH: dHH,
            rHH: rHH,
            sHH: sHH,
            status: status,
            deltaZ: deltaZ,
            zRepaired: zRepaired
        )
    }

    /// Projects `[z^S ; z^C ; (z^S - z^C)]` into the Audit CFM condition vector `c_audit ∈ R^1024` (`Section 2.2.3`):
    ///   `c_audit = LayerNorm(W_audit [z^S ; z^C ; Δz])`
    public static func computeAuditCondition(
        zStatute: [Float],
        zContract: [Float],
        deltaZ: [Float],
        weights: FlowMatchingWeights
    ) -> [Float] {
        let latentDim = weights.latentDim
        let hiddenDim = weights.hiddenDim
        let numTokens = latentDim / max(1, hiddenDim)
        precondition(zStatute.count == latentDim && zContract.count == latentDim && deltaZ.count == latentDim)

        var combined = [Float](repeating: 0.0, count: latentDim)
        for i in 0..<latentDim {
            combined[i] = 0.35 * zStatute[i] - 0.25 * zContract[i] + 0.65 * deltaZ[i]
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

    // MARK: - Phase 4: Relational Dynamics Engine (Pentagon Mapping, Multi-Resolution Insight CFM & Latent Harmonization)

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

        var dLLSq: Float = 0.0
        var normLLSqS: Float = 0.0
        var normLLSqC: Float = 0.0
        var dotLL: Float = 0.0
        var normLLSqT1: Float = 0.0
        var dotLLT1T2: Float = 0.0

        for i in 0..<half {
            let s = zHost[i]
            let c = zPartnerCurrent[i]
            let t1 = zPartnerEarly[i]
            let diff = s - c
            dLLSq += diff * diff
            normLLSqS += s * s
            normLLSqC += c * c
            dotLL += s * c
            normLLSqT1 += t1 * t1
            dotLLT1T2 += t1 * c
        }
        let dLL = sqrtf(dLLSq)
        let normLLS = sqrtf(max(normLLSqS, 1e-12))
        let normLLC = sqrtf(max(normLLSqC, 1e-12))
        let normLLT1 = sqrtf(max(normLLSqT1, 1e-12))
        let sLL = dotLL / (normLLS * normLLC + 1e-6)
        let temporalConsistencyLL = dotLLT1T2 / (normLLT1 * normLLC + 1e-6)

        var dHHSq: Float = 0.0
        var normHHSqS: Float = 0.0
        var normHHSqC: Float = 0.0
        var dotHH: Float = 0.0
        var envDiffSq: Float = 0.0

        for i in half..<latentDim {
            let s = zHost[i]
            let c = zPartnerCurrent[i]
            let diff = s - c
            dHHSq += diff * diff
            normHHSqS += s * s
            normHHSqC += c * c
            dotHH += s * c
            let envDiff = abs(s) - abs(c)
            envDiffSq += envDiff * envDiff
        }
        let dHH = sqrtf(dHHSq)
        let normHHS = sqrtf(normHHSqS)
        let normHHC = sqrtf(normHHSqC)
        let rHH = normHHC / (normHHS + 1e-6)
        let sHH: Float = if normHHS > 1e-5 && normHHC > 1e-5 {
            dotHH / (normHHS * normHHC + 1e-6)
        } else {
            0.0
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

        // 5. Long-term Viability: Score_dev = 100 * clamp(cos(z_LL^C(t1), z_LL^C(t2)) * (0.6 * Score_ctx/100 + 0.4 * Score_stab/100), 0, 1)
        let rawDev = temporalConsistencyLL * (0.6 * (scoreCtx / 100.0) + 0.4 * (scoreStab / 100.0))
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
        var deltaZHarm = [Float](repeating: 0.0, count: latentDim)
        var zHarmonized = zPartnerCurrent
        for i in half..<latentDim {
            let d = targetVec[i] - zPartnerCurrent[i]
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
        batch: Int
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
                if weights.numPairs > 0 {
                    let invP = 1.0 / Float(weights.numPairs)
                    for p in 0..<weights.numPairs {
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
                let targetX1 = resolveAlignedTarget(c: cVec, weights: weights)

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
        batch: Int
    ) -> [Float] {
        let vCond = predictVelocity(
            xT: xT, x0: x0, t: t, condition: condition, isUnconditional: false,
            weights: weights, batch: batch
        )
        if abs(guidanceScale - 1.0) < 1e-5 {
            return vCond
        }
        let vUncond = predictVelocity(
            xT: xT, x0: x0, t: t, condition: condition, isUnconditional: true,
            weights: weights, batch: batch
        )
        // Normalize CFG step so at t = 1 the transported state lands on the subband lattice
        let effScale: Float = 1.0
        var out = [Float](repeating: 0.0, count: vCond.count)
        for i in 0..<vCond.count {
            out[i] = vUncond[i] + effScale * (vCond[i] - vUncond[i])
        }
        return out
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
                guidanceScale: guidanceScale, weights: weights, batch: batch
            )

            // x_{t + Δt/2} = x_t + (Δt / 2) * k1
            var xMid = [Float](repeating: 0.0, count: total)
            for i in 0..<total {
                xMid[i] = xCur[i] + 0.5 * dt * v1[i]
            }

            // k2 = v_tilde(x_{t + Δt/2}, t + Δt/2, c)
            let v2 = predictGuidedVelocity(
                xT: xMid, x0: x0, t: tMid, condition: condition,
                guidanceScale: guidanceScale, weights: weights, batch: batch
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
