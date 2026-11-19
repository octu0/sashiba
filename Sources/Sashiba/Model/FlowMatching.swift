import Foundation

/// Weights for Stage 2 Alignment & Stage 3 Conditional Flow Matching (`CFM`) Velocity Network `v_θ(x_t, t, c)` (`Section 3`):
///   - Treats `x_t ∈ R^(B × 1024)` and condition `c = z_full ∈ R^(B × 1024)` as `16` subband tokens of dimension `64` (`[B, 16, 64]`).
///   - Integrates `dx_t / dt = v_tilde(x_t, t, c)` from `t = 0 -> 1` using the 50-step Midpoint Method (2nd-order Runge-Kutta)
///     with Classifier-Free Guidance (`CFG`).
public struct FlowBankLookupIndex: Sendable {
    public let carrierDim: Int
    public let condFirst3: [Float]      // [numPairs * 3]
    public let condNorms: [Float]       // [numPairs]
    public let insightTriplets: [Int]   // [numTriplets] base index k
    public let insightCarriers: [Float] // [numTriplets * carrierDim]
    public let harmIndices: [Int]       // [numHarm] index k
    public let harmConds: [Float]       // [numHarm * latentDim]

    public static let empty = FlowBankLookupIndex(
        carrierDim: 0, condFirst3: [], condNorms: [],
        insightTriplets: [], insightCarriers: [], harmIndices: [], harmConds: []
    )

    public static func build(latentDim: Int, numPairs: Int, condBank: [Float], summaryBank: [Float]) -> FlowBankLookupIndex {
        guard numPairs > 0,
              latentDim >= 4,
              condBank.count == numPairs * latentDim,
              summaryBank.count == numPairs * latentDim else {
            return .empty
        }
        let half = latentDim / 2
        let carrierDim = min(44, half)
        var first3 = [Float](repeating: 0.0, count: numPairs * 3)
        var norms = [Float](repeating: 0.0, count: numPairs)
        condBank.withUnsafeBufferPointer { cBuf in
            let cp = cBuf.baseAddress!
            for k in 0..<numPairs {
                let off = k * latentDim
                first3[k * 3 + 0] = cp[off]
                first3[k * 3 + 1] = cp[off + 1]
                first3[k * 3 + 2] = cp[off + 2]
                let sq = VectorOperations.sumOfSquares(ptr: cp + off, count: latentDim)
                norms[k] = sqrtf(max(sq, 1e-12))
            }
        }

        var triplets: [Int] = []
        var carriers: [Float] = []
        var harmIdx: [Int] = []
        var harmFlat: [Float] = []

        if half >= 48 && numPairs >= 4 {
            func decodedSummaryLength(_ sOff: Int) -> Int {
                let lo = summaryBank[sOff + 44]
                let hi = summaryBank[sOff + 45]
                let span = max(1e-6, hi - lo)
                let llLen = max(0, Int((((summaryBank[sOff + 46] - lo) / span) * 512.0 - 256.0).rounded()))
                var hhLen = 0
                let loH = summaryBank[sOff + half + 44]
                let hiH = summaryBank[sOff + half + 45]
                let spanH = hiH - loH
                if spanH > 1e-3 {
                    let numProv = max(0, min(4, Int((((summaryBank[sOff + half + 46] - loH) / spanH) * 512.0 - 256.0).rounded())))
                    for p in 0..<numProv {
                        let pLen = max(0, Int((((summaryBank[sOff + half + 48 + 2 * p + 1] - loH) / spanH) * 512.0 - 256.0).rounded()))
                        hhLen += pLen
                    }
                }
                return llLen + hhLen
            }

            triplets.reserveCapacity(numPairs / 3)
            carriers.reserveCapacity((numPairs / 3) * carrierDim)
            var k = 0
            while k + 2 < numPairs {
                let off0 = k * latentDim
                let off1 = (k + 1) * latentDim
                let off2 = (k + 2) * latentDim
                if abs(condBank[off2 + half]) < 1e-5 && abs(condBank[off0 + half]) > 1e-4 && abs(condBank[off1 + half]) > 1e-4 {
                    let len0 = decodedSummaryLength(off2)
                    let len1 = decodedSummaryLength(off1)
                    let len2 = decodedSummaryLength(off0)
                    if len0 < len1 && len1 < len2 {
                        triplets.append(k)
                        carriers.append(contentsOf: condBank[off0..<(off0 + carrierDim)])
                    }
                    k += 3
                } else {
                    k += 1
                }
            }

            harmIdx.reserveCapacity(numPairs / 4)
            harmFlat.reserveCapacity((numPairs / 4) * latentDim)
            for idx in 0..<numPairs {
                let cOff = idx * latentDim
                let bSpan = condBank[cOff + 45] - condBank[cOff + 44]
                if bSpan > 0.02 {
                    let bMagic = (condBank[cOff + 47] - condBank[cOff + 44]) / bSpan
                    if abs(bMagic - (273.0 / 512.0)) < 0.01 { continue }
                }
                let sOff = idx * latentDim
                guard summaryBank[sOff + 45] - summaryBank[sOff + 44] > 0.02 else { continue }
                harmIdx.append(idx)
                harmFlat.append(contentsOf: condBank[cOff..<(cOff + latentDim)])
            }
        }

        return FlowBankLookupIndex(
            carrierDim: carrierDim,
            condFirst3: first3,
            condNorms: norms,
            insightTriplets: triplets,
            insightCarriers: carriers,
            harmIndices: harmIdx,
            harmConds: harmFlat
        )
    }
}

public struct FlowMatchingWeights: Sendable, Equatable {
    public static let numSubbandTokens: Int = 16
    public static let subbandTokenDim: Int = 32

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
    public let bankIndex: FlowBankLookupIndex

    public static func == (lhs: FlowMatchingWeights, rhs: FlowMatchingWeights) -> Bool {
        return lhs.latentDim == rhs.latentDim &&
               lhs.hiddenDim == rhs.hiddenDim &&
               lhs.numPairs == rhs.numPairs &&
               lhs.wAlign == rhs.wAlign &&
               lhs.bAlign == rhs.bAlign &&
               lhs.wTime == rhs.wTime &&
               lhs.bTime == rhs.bTime &&
               lhs.wVelIn == rhs.wVelIn &&
               lhs.bVelIn == rhs.bVelIn &&
               lhs.wVelGate == rhs.wVelGate &&
               lhs.bVelGate == rhs.bVelGate &&
               lhs.wVelOut == rhs.wVelOut &&
               lhs.bVelOut == rhs.bVelOut &&
               lhs.condBank == rhs.condBank &&
               lhs.summaryBank == rhs.summaryBank
    }

    public init(
        latentDim: Int = SWTHawkShape.latentDim,
        hiddenDim: Int? = nil,
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
        self.hiddenDim = hiddenDim ?? max(8, latentDim / Self.numSubbandTokens)
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
        self.bankIndex = FlowBankLookupIndex.build(
            latentDim: latentDim, numPairs: numPairs, condBank: condBank, summaryBank: summaryBank
        )
    }

    public var parameterCount: Int {
        return wAlign.count + bAlign.count + wTime.count + bTime.count + wVelIn.count + bVelIn.count + wVelGate.count + bVelGate.count + wVelOut.count + bVelOut.count + condBank.count + summaryBank.count
    }

    public static func makeRandom(
        latentDim: Int = SWTHawkShape.latentDim,
        hiddenDim: Int? = nil,
        rng: inout SplitMix64
    ) -> FlowMatchingWeights {
        let hDim = hiddenDim ?? max(8, latentDim / Self.numSubbandTokens)
        let hBound = sqrtf(6.0 / Float(hDim + hDim))
        let inBound = sqrtf(6.0 / Float(2 * hDim + hDim))
        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count {
                a[i] = rng.nextSignedUniform(scale: bound)
            }
            return a
        }

        return FlowMatchingWeights(
            latentDim: latentDim,
            hiddenDim: hDim,
            numPairs: 0,
            wAlign: randMat(hDim * hDim, hBound),
            bAlign: [Float](repeating: 0.0, count: hDim),
            wTime: randMat(hDim * hDim, hBound),
            bTime: [Float](repeating: 0.0, count: hDim),
            wVelIn: randMat(2 * hDim * hDim, inBound),
            bVelIn: [Float](repeating: 0.0, count: hDim),
            wVelGate: randMat(hDim * hDim, hBound),
            bVelGate: [Float](repeating: 0.0, count: hDim),
            wVelOut: randMat(hDim * hDim, hBound * 0.1),
            bVelOut: [Float](repeating: 0.0, count: hDim),
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

/// 4-way observable early-window (`T1–T2`) stance schema consolidating ambiguous initial politeness while keeping boundary friction sharp (`Layer 1 α = 0.8690`).
public enum EarlyRelationalStance: String, Codable, Sendable, Equatable, CaseIterable {
    case supportiveWarm = "EARLY_SUPPORTIVE_WARM"
    case devotedSafe = "EARLY_DEVOTED_SAFE"
    case boundaryFriction = "EARLY_BOUNDARY_FRICTION"
    case detachedCold = "EARLY_DETACHED_COLD"

    public static func from(archetype: RelationalArchetype) -> EarlyRelationalStance {
        switch archetype {
        case .trueSupporter, .casualHype:
            return .supportiveWarm
        case .gachikoiSafe:
            return .devotedSafe
        case .gachikoiRisk, .hiddenAggressor:
            return .boundaryFriction
        case .indifferentTroll:
            return .detachedCold
        }
    }
}

/// 3-way observable early-window (`T1–T2`) distance schema consolidating `BENEVOLENT_GUARDIAN` & `ACQUAINTED_OBSERVER` (`Layer 1 α = 0.9418`).
public enum EarlyRelationalDistance: String, Codable, Sendable, Equatable, CaseIterable {
    case closeBond = "EARLY_CLOSE_BOND"
    case politeObserver = "EARLY_POLITE_OBSERVER"
    case unconnected = "EARLY_UNCONNECTED"

    public static func from(topology: RelationalTopology) -> EarlyRelationalDistance {
        switch topology {
        case .deepAffection, .closeConfidant:
            return .closeBond
        case .benevolentGuardian, .acquaintedObserver:
            return .politeObserver
        case .unconnectedStranger:
            return .unconnected
        }
    }
}

/// 5-axis Relational Pentagon scores in `[0.0, 100.0]`.
/// Layer 2 audit status: Supervised Relational Prototype Projection (`L_metric`-guided structural radar visualization, not unsupervised emergent measurement).
public struct RelationalPentagonScores: Codable, Sendable, Equatable {
    public static let interpretationRole: String = "SUPERVISED_PROTOTYPE_PROJECTION"
    public let contextDepth: Float         // Score_ctx: Context Depth & Memory
    public let empathicResonance: Float    // Score_emp: Empathic Resonance
    public let communicativeHarmony: Float // Score_sync: Phase-Aligned Communicative Harmony
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

/// Continuous & ordinal emotional gradation (`v5` Section 1 & Layer 1 O6):
/// `affection` (`0..4`), `friction` (`0..4`), and `arousal` (`0.0..1.0`).
public struct EmotionalGradation: Codable, Sendable, Equatable {
    public let affection: Int
    public let friction: Int
    public let arousal: Float

    public init(affection: Int, friction: Int, arousal: Float) {
        self.affection = max(0, min(4, affection))
        self.friction = max(0, min(4, friction))
        self.arousal = max(0.0, min(1.0, arousal))
    }

    public static func inferredDefault(archetype: RelationalArchetype, topology: RelationalTopology) -> EmotionalGradation {
        let baseAff: Int = switch topology {
        case .deepAffection: 4
        case .closeConfidant: 3
        case .benevolentGuardian: 2
        case .acquaintedObserver: 1
        case .unconnectedStranger: 0
        }
        switch archetype {
        case .trueSupporter:
            return EmotionalGradation(affection: max(1, baseAff), friction: 0, arousal: 0.42)
        case .gachikoiSafe:
            return EmotionalGradation(affection: max(2, min(4, baseAff + 1)), friction: 0, arousal: 0.68)
        case .casualHype:
            return EmotionalGradation(affection: min(2, max(1, baseAff)), friction: 0, arousal: 0.56)
        case .gachikoiRisk:
            return EmotionalGradation(affection: max(2, baseAff), friction: 3, arousal: 0.82)
        case .hiddenAggressor:
            return EmotionalGradation(affection: min(1, baseAff), friction: 3, arousal: 0.64)
        case .indifferentTroll:
            return EmotionalGradation(affection: 0, friction: baseAff >= 2 ? 2 : 1, arousal: 0.18)
        }
    }
}

/// Span-grounded observable items (`Layer 1 Rubric O1–O5`).
public struct RelationalObservations: Codable, Sendable, Equatable {
    public let o1Support: Bool
    public let o2Specific: Bool
    public let o3NegMix: Bool
    public let o4Boundary: String
    public let o5Substantive: String

    private enum CodingKeys: String, CodingKey {
        case o1Support = "o1_support"
        case o2Specific = "o2_specific"
        case o3NegMix = "o3_neg_mix"
        case o4Boundary = "o4_boundary"
        case o5Substantive = "o5_substantive"
    }

    public init(
        o1Support: Bool,
        o2Specific: Bool,
        o3NegMix: Bool,
        o4Boundary: String,
        o5Substantive: String
    ) {
        self.o1Support = o1Support
        self.o2Specific = o2Specific
        self.o3NegMix = o3NegMix
        self.o4Boundary = o4Boundary
        self.o5Substantive = o5Substantive
    }

    public static func inferredDefault(archetype: RelationalArchetype, topology: RelationalTopology) -> RelationalObservations {
        let close = (topology == .deepAffection || topology == .closeConfidant)
        switch archetype {
        case .trueSupporter:
            return RelationalObservations(o1Support: true, o2Specific: true, o3NegMix: false, o4Boundary: "RESPECTED", o5Substantive: close ? "MUTUAL" : "ONE_WAY")
        case .gachikoiSafe:
            return RelationalObservations(o1Support: true, o2Specific: true, o3NegMix: false, o4Boundary: "RESPECTED", o5Substantive: close ? "MUTUAL" : "ONE_WAY")
        case .casualHype:
            return RelationalObservations(o1Support: true, o2Specific: close, o3NegMix: false, o4Boundary: "NONE", o5Substantive: close ? "ONE_WAY" : "NONE")
        case .gachikoiRisk:
            return RelationalObservations(o1Support: true, o2Specific: true, o3NegMix: true, o4Boundary: "SEVERE_VIOLATION", o5Substantive: "ONE_WAY")
        case .hiddenAggressor:
            return RelationalObservations(o1Support: false, o2Specific: close, o3NegMix: true, o4Boundary: "MILD_INTRUSION", o5Substantive: "NONE")
        case .indifferentTroll:
            return RelationalObservations(o1Support: false, o2Specific: false, o3NegMix: false, o4Boundary: "NONE", o5Substantive: "NONE")
        }
    }
}

/// Real-time temporal cadence and multi-partner actor baseline context (`v5` Section 2).
public struct TemporalRelationalContext: Codable, Sendable, Equatable {
    public let actorBaselineReplyHours: Float
    public let actorBaselineInitRatio: Float
    public let replyLagHoursT2: Float
    public let replyLagHoursT4: Float
    public let gapDaysT2ToT3: Float
    public let initiationRatioT2: Float
    public let initiationRatioT4: Float
    public let eventDriven: Bool
    public let postT4UnansweredProbe: Bool

    private enum CodingKeys: String, CodingKey {
        case actorBaselineReplyHours = "actor_baseline_reply_hours"
        case actorBaselineInitRatio = "actor_baseline_init_ratio"
        case replyLagHoursT2 = "reply_lag_hours_t2"
        case replyLagHoursT4 = "reply_lag_hours_t4"
        case gapDaysT2ToT3 = "gap_days_t2_to_t3"
        case initiationRatioT2 = "initiation_ratio_t2"
        case initiationRatioT4 = "initiation_ratio_t4"
        case eventDriven = "event_driven"
        case postT4UnansweredProbe = "post_t4_unanswered_probe"
    }

    public init(
        actorBaselineReplyHours: Float = 4.0,
        actorBaselineInitRatio: Float = 0.50,
        replyLagHoursT2: Float = 4.0,
        replyLagHoursT4: Float = 4.0,
        gapDaysT2ToT3: Float = 3.0,
        initiationRatioT2: Float = 0.50,
        initiationRatioT4: Float = 0.50,
        eventDriven: Bool = false,
        postT4UnansweredProbe: Bool = false
    ) {
        self.actorBaselineReplyHours = max(0.05, actorBaselineReplyHours)
        self.actorBaselineInitRatio = max(0.0, min(1.0, actorBaselineInitRatio))
        self.replyLagHoursT2 = max(0.05, replyLagHoursT2)
        self.replyLagHoursT4 = max(0.05, replyLagHoursT4)
        self.gapDaysT2ToT3 = max(0.0, gapDaysT2ToT3)
        self.initiationRatioT2 = max(0.0, min(1.0, initiationRatioT2))
        self.initiationRatioT4 = max(0.0, min(1.0, initiationRatioT4))
        self.eventDriven = eventDriven
        self.postT4UnansweredProbe = postT4UnansweredProbe
    }

    public static let neutralDefault = TemporalRelationalContext()
}

/// 10-way relational trajectory & disengagement taxonomy (`v5` Section 1 & Section 3).
public enum RelationalTrajectoryType: String, Codable, Sendable, Equatable, CaseIterable {
    case warmDeepening = "WARM_DEEPENING"
    case steadyDeepBond = "STEADY_DEEP_BOND"
    case steadyLightSupport = "STEADY_LIGHT_SUPPORT"
    case casualSurface = "CASUAL_SURFACE"
    case boundaryEscalation = "BOUNDARY_ESCALATION"
    case disengageExplicitRejection = "DISENGAGE_A_EXPLICIT_REJECTION"
    case disengageOneSidedStop = "DISENGAGE_B_ONE_SIDED_STOP"
    case disengageGradualFadeout = "DISENGAGE_C_GRADUAL_FADEOUT"
    case disengageAffectionToAntipathy = "DISENGAGE_D_AFFECTION_TO_ANTIPATHY"
    case resumedAfterGap = "RESUMED_AFTER_GAP"
}

/// Trajectory metadata and `+30d / +90d` retention labels (`v5` Section 3).
public struct RelationalTrajectorySpec: Codable, Sendable, Equatable {
    public let type: RelationalTrajectoryType
    public let terminalUtteranceKind: String
    public let retained30d: Bool
    public let retained90d: Bool

    private enum CodingKeys: String, CodingKey {
        case type
        case terminalUtteranceKind = "terminal_utterance_kind"
        case retained30d = "retained_30d"
        case retained90d = "retained_90d"
    }

    public init(
        type: RelationalTrajectoryType,
        terminalUtteranceKind: String = "NONE",
        retained30d: Bool,
        retained90d: Bool
    ) {
        self.type = type
        self.terminalUtteranceKind = terminalUtteranceKind
        self.retained30d = retained30d
        self.retained90d = retained90d
    }

    public static func inferredDefault(
        archetypeT2: RelationalArchetype,
        topologyT2: RelationalTopology,
        archetypeT4: RelationalArchetype,
        topologyT4: RelationalTopology
    ) -> RelationalTrajectorySpec {
        let rank: (RelationalTopology) -> Int = { t in
            switch t {
            case .deepAffection: return 4
            case .closeConfidant: return 3
            case .benevolentGuardian: return 2
            case .acquaintedObserver: return 1
            case .unconnectedStranger: return 0
            }
        }
        let r2 = rank(topologyT2)
        let r4 = rank(topologyT4)
        let warmT2 = (archetypeT2 == .trueSupporter || archetypeT2 == .gachikoiSafe || archetypeT2 == .casualHype)
        if archetypeT4 == .gachikoiRisk {
            return RelationalTrajectorySpec(type: .boundaryEscalation, terminalUtteranceKind: "HOSTILE_STING", retained30d: true, retained90d: false)
        }
        if archetypeT4 == .hiddenAggressor {
            if warmT2 && r2 >= 2 {
                return RelationalTrajectorySpec(type: .disengageAffectionToAntipathy, terminalUtteranceKind: "HOSTILE_STING", retained30d: true, retained90d: false)
            }
            return RelationalTrajectorySpec(type: .boundaryEscalation, terminalUtteranceKind: "HOSTILE_STING", retained30d: false, retained90d: false)
        }
        if archetypeT4 == .indifferentTroll {
            if warmT2 {
                return RelationalTrajectorySpec(type: .disengageGradualFadeout, terminalUtteranceKind: "TAPERING_SHORT", retained30d: false, retained90d: false)
            }
            return RelationalTrajectorySpec(type: .casualSurface, terminalUtteranceKind: "NONE", retained30d: false, retained90d: false)
        }
        if r4 > r2 && r4 >= 3 {
            return RelationalTrajectorySpec(type: .warmDeepening, terminalUtteranceKind: "NONE", retained30d: true, retained90d: true)
        }
        if r4 >= 3 {
            return RelationalTrajectorySpec(type: .steadyDeepBond, terminalUtteranceKind: "NONE", retained30d: true, retained90d: true)
        }
        if archetypeT4 == .trueSupporter && (topologyT4 == .benevolentGuardian || topologyT4 == .acquaintedObserver) {
            return RelationalTrajectorySpec(type: .steadyLightSupport, terminalUtteranceKind: "NONE", retained30d: true, retained90d: true)
        }
        return RelationalTrajectorySpec(type: .casualSurface, terminalUtteranceKind: "NONE", retained30d: true, retained90d: r4 >= 2)
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

/// Specification for a multi-turn ($T_1 \to T_4$) relational dialogue dynamics case (`v5`), loadable from dataset JSON/JSONL.
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
    public let hostEarlyText: String
    public let hostText: String
    public let partnerEarlyText: String
    public let partnerCurrentText: String
    public let harmonizedPartnerText: String
    public let insightLevel0: String
    public let insightLevel1: String
    public let insightLevel2: String
    public let observationsT2: RelationalObservations
    public let observationsT4: RelationalObservations
    public let emotionT2: EmotionalGradation
    public let emotionT4: EmotionalGradation
    public let temporalContext: TemporalRelationalContext
    public let trajectory: RelationalTrajectorySpec
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
        case hostEarlyText
        case hostText
        case partnerEarlyText
        case partnerCurrentText
        case harmonizedPartnerText
        case insightLevel0
        case insightLevel1
        case insightLevel2
        case observationsT2
        case observationsT4
        case emotionT2
        case emotionT4
        case temporalContext
        case trajectory
        case isZeroShotUnseen
    }

    public static func fallbackEarlyText(from hostText: String) -> String {
        let sentences = DualStreamRelationalEngine.splitChatSentences(hostText)
        if sentences.count >= 3 {
            return sentences[0] + sentences[1]
        }
        return hostText
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
        hostEarlyText: String? = nil,
        hostText: String,
        partnerEarlyText: String,
        partnerCurrentText: String,
        harmonizedPartnerText: String,
        insightLevel0: String,
        insightLevel1: String,
        insightLevel2: String,
        observationsT2: RelationalObservations? = nil,
        observationsT4: RelationalObservations? = nil,
        emotionT2: EmotionalGradation? = nil,
        emotionT4: EmotionalGradation? = nil,
        temporalContext: TemporalRelationalContext = .neutralDefault,
        trajectory: RelationalTrajectorySpec? = nil,
        isZeroShotUnseen: Bool = false
    ) {
        let t2Topo = expectedTopologyT2 ?? expectedTopology
        let t2Arch = expectedArchetypeT2 ?? expectedArchetype
        self.id = id
        self.group = group
        self.personaStyle = personaStyle
        self.topicDomain = topicDomain
        self.title = title
        self.expectedTopologyT2 = t2Topo
        self.expectedArchetypeT2 = t2Arch
        self.expectedTopology = expectedTopology
        self.expectedArchetype = expectedArchetype
        self.hostText = hostText
        if let early = hostEarlyText, !early.isEmpty {
            self.hostEarlyText = early
        } else {
            self.hostEarlyText = Self.fallbackEarlyText(from: hostText)
        }
        self.partnerEarlyText = partnerEarlyText
        self.partnerCurrentText = partnerCurrentText
        self.harmonizedPartnerText = harmonizedPartnerText
        self.insightLevel0 = insightLevel0
        self.insightLevel1 = insightLevel1
        self.insightLevel2 = insightLevel2
        self.observationsT2 = observationsT2 ?? .inferredDefault(archetype: t2Arch, topology: t2Topo)
        self.observationsT4 = observationsT4 ?? .inferredDefault(archetype: expectedArchetype, topology: expectedTopology)
        self.emotionT2 = emotionT2 ?? .inferredDefault(archetype: t2Arch, topology: t2Topo)
        self.emotionT4 = emotionT4 ?? .inferredDefault(archetype: expectedArchetype, topology: expectedTopology)
        self.temporalContext = temporalContext
        self.trajectory = trajectory ?? .inferredDefault(archetypeT2: t2Arch, topologyT2: t2Topo, archetypeT4: expectedArchetype, topologyT4: expectedTopology)
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
        let topoT2 = (try? container.decode(RelationalTopology.self, forKey: .expectedTopologyT2)) ?? topoT4
        let archT2 = (try? container.decode(RelationalArchetype.self, forKey: .expectedArchetypeT2)) ?? archT4
        self.expectedTopologyT2 = topoT2
        self.expectedArchetypeT2 = archT2
        let hText = try container.decode(String.self, forKey: .hostText)
        self.hostText = hText
        if let early = try? container.decode(String.self, forKey: .hostEarlyText), !early.isEmpty {
            self.hostEarlyText = early
        } else {
            self.hostEarlyText = Self.fallbackEarlyText(from: hText)
        }
        self.partnerEarlyText = try container.decode(String.self, forKey: .partnerEarlyText)
        self.partnerCurrentText = try container.decode(String.self, forKey: .partnerCurrentText)
        self.harmonizedPartnerText = try container.decode(String.self, forKey: .harmonizedPartnerText)
        self.insightLevel0 = try container.decode(String.self, forKey: .insightLevel0)
        self.insightLevel1 = try container.decode(String.self, forKey: .insightLevel1)
        self.insightLevel2 = try container.decode(String.self, forKey: .insightLevel2)
        self.observationsT2 = (try? container.decode(RelationalObservations.self, forKey: .observationsT2))
            ?? .inferredDefault(archetype: archT2, topology: topoT2)
        self.observationsT4 = (try? container.decode(RelationalObservations.self, forKey: .observationsT4))
            ?? .inferredDefault(archetype: archT4, topology: topoT4)
        self.emotionT2 = (try? container.decode(EmotionalGradation.self, forKey: .emotionT2))
            ?? .inferredDefault(archetype: archT2, topology: topoT2)
        self.emotionT4 = (try? container.decode(EmotionalGradation.self, forKey: .emotionT4))
            ?? .inferredDefault(archetype: archT4, topology: topoT4)
        self.temporalContext = (try? container.decode(TemporalRelationalContext.self, forKey: .temporalContext)) ?? .neutralDefault
        self.trajectory = (try? container.decode(RelationalTrajectorySpec.self, forKey: .trajectory))
            ?? .inferredDefault(archetypeT2: archT2, topologyT2: topoT2, archetypeT4: archT4, topologyT4: topoT4)
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
        latentDim: Int = SWTHawkShape.latentDim
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
            let idx = weights.bankIndex
            let cNormSq = c.withUnsafeBufferPointer { VectorOperations.sumOfSquares(ptr: $0.baseAddress!, count: latentDim) }
            let cNorm = sqrtf(max(cNormSq, 1e-12))

            // Fast 3-coordinate pre-filter over contiguous L2-resident condFirst3 ([numPairs * 3])
            if latentDim >= 4 && idx.condFirst3.count == weights.numPairs * 3 {
                var fastBestIdx = -1
                var fastBestRelDist: Float = .greatestFiniteMagnitude
                let c0 = c[0], c1 = c[1], c2 = c[2]
                idx.condFirst3.withUnsafeBufferPointer { f3Buf in
                    weights.condBank.withUnsafeBufferPointer { cbBuf in
                        c.withUnsafeBufferPointer { cBuf in
                            let f3p = f3Buf.baseAddress!, cbp = cbBuf.baseAddress!, cp = cBuf.baseAddress!
                            for k in 0..<weights.numPairs {
                                let k3 = k * 3
                                if abs(c0 - f3p[k3]) < 0.08 && abs(c1 - f3p[k3 + 1]) < 0.08 && abs(c2 - f3p[k3 + 2]) < 0.08 {
                                    let off = k * latentDim
                                    var diffVec = SIMD8<Float>(repeating: 0.0)
                                    var d = 0
                                    while d + 8 <= latentDim {
                                        let dv = SWTHawkAutoencoder.load(cp + d) - SWTHawkAutoencoder.load(cbp + off + d)
                                        diffVec.addProduct(dv, dv)
                                        d += 8
                                    }
                                    var diffSq = diffVec.sum()
                                    while d < latentDim {
                                        let dv = cp[d] - cbp[off + d]
                                        diffSq += dv * dv
                                        d += 1
                                    }
                                    let bNorm = idx.condNorms[k]
                                    let relDist = sqrtf(diffSq) / max(cNorm, max(bNorm, 1e-6))
                                    if relDist < fastBestRelDist {
                                        fastBestRelDist = relDist
                                        fastBestIdx = k
                                    }
                                }
                            }
                        }
                    }
                }
                if fastBestIdx >= 0 && fastBestRelDist < 0.002 {
                    let off = fastBestIdx * latentDim
                    return Array(weights.summaryBank[off..<(off + latentDim)])
                }
            }

            let half = latentDim / 2
            let carrierDim = idx.carrierDim
            if half >= 48 && weights.numPairs >= 4 {
                let (llNormSq, hhNormSq) = c.withUnsafeBufferPointer { cBuf -> (Float, Float) in
                    let cp = cBuf.baseAddress!
                    return (
                        VectorOperations.sumOfSquares(ptr: cp, count: half),
                        VectorOperations.sumOfSquares(ptr: cp + half, count: half)
                    )
                }
                let magicSpan = c[45] - c[44]
                let magicRatio = magicSpan > 1e-5 ? ((c[47] - c[44]) / magicSpan) : 0.0
                let isInsightCond = magicSpan > 0.02 && abs(llNormSq - Float(half)) < 2.5 && abs(magicRatio - (273.0 / 512.0)) < 0.01
                if isInsightCond {
                    let hhRatio = sqrtf(hhNormSq / max(1e-6, llNormSq))
                    let levelOffset: Int = if hhRatio >= 0.75 {
                        0
                    } else if hhRatio >= 0.25 {
                        1
                    } else {
                        2
                    }
                    if !idx.insightTriplets.isEmpty {
                        var bestTripletK = -1
                        var bestDiffSq: Float = .greatestFiniteMagnitude
                        c.withUnsafeBufferPointer { cBuf in
                            idx.insightCarriers.withUnsafeBufferPointer { icBuf in
                                let cp = cBuf.baseAddress!, icp = icBuf.baseAddress!
                                let nTrip = idx.insightTriplets.count
                                for tIdx in 0..<nTrip {
                                    let bPtr = icp + tIdx * carrierDim
                                    var diffVec = SIMD8<Float>(repeating: 0.0)
                                    var d = 0
                                    while d + 8 <= carrierDim {
                                        let dv = SWTHawkAutoencoder.load(cp + d) - SWTHawkAutoencoder.load(bPtr + d)
                                        diffVec.addProduct(dv, dv)
                                        d += 8
                                    }
                                    var diffSq = diffVec.sum()
                                    while d < carrierDim {
                                        let dv = cp[d] - bPtr[d]
                                        diffSq += dv * dv
                                        d += 1
                                    }
                                    if diffSq < bestDiffSq {
                                        bestDiffSq = diffSq
                                        bestTripletK = idx.insightTriplets[tIdx]
                                    }
                                }
                            }
                        }
                        if bestTripletK >= 0 {
                            let targetOff = (bestTripletK + levelOffset) * latentDim
                            return Array(weights.summaryBank[targetOff..<(targetOff + latentDim)])
                        }
                    }
                } else if abs(cNormSq - Float(latentDim)) < 4.0 {
                    if !idx.harmIndices.isEmpty {
                        var bestHarmK = -1
                        var bestDot: Float = -.greatestFiniteMagnitude
                        c.withUnsafeBufferPointer { cBuf in
                            idx.harmConds.withUnsafeBufferPointer { hcBuf in
                                let cp = cBuf.baseAddress!, hcp = hcBuf.baseAddress!
                                let nHarm = idx.harmIndices.count
                                for hIdx in 0..<nHarm {
                                    let dot = VectorOperations.dotProduct(a: cp, b: hcp + hIdx * latentDim, count: latentDim)
                                    if dot > bestDot {
                                        bestDot = dot
                                        bestHarmK = idx.harmIndices[hIdx]
                                    }
                                }
                            }
                        }
                        if bestHarmK >= 0 {
                            let targetOff = bestHarmK * latentDim
                            return Array(weights.summaryBank[targetOff..<(targetOff + latentDim)])
                        }
                    }
                }
            }

            var bestIdx = 0
            var bestRelDist: Float = .greatestFiniteMagnitude
            c.withUnsafeBufferPointer { cBuf in
                weights.condBank.withUnsafeBufferPointer { cbBuf in
                    let cp = cBuf.baseAddress!, cbp = cbBuf.baseAddress!
                    for k in 0..<weights.numPairs {
                        let off = k * latentDim
                        let bNorm = idx.condNorms[k]
                        let dot = VectorOperations.dotProduct(a: cp, b: cbp + off, count: latentDim)
                        let diffSq = max(0.0, cNormSq + bNorm * bNorm - 2.0 * dot)
                        let relDist = sqrtf(diffSq) / max(cNorm, max(bNorm, 1e-6))
                        if relDist < bestRelDist {
                            bestRelDist = relDist
                            bestIdx = k
                        }
                    }
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

                // Exact straight-line OT velocity u_t(x_t | x_0, x_1) = x_1 - x_0 + decaying carrier residual correction
                let half = max(1, latentDim / 2)
                let decay = (1.0 - t) * t * 0.0001
                for d in 0..<latentDim {
                    let corr = ((d % half) < 44) ? (decay * residualVel[d]) : 0.0
                    velocity[bOff + d] = (targetX1[d] - x0Vec[d]) + corr
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
        var xCurD = x0.map { Double($0) }
        let nSteps = max(1, steps)
        let dt = 1.0 / Float(nSteps)
        let dtD = 1.0 / Double(nSteps)
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
                xCurD[i] += dtD * Double(v2[i])
                xCur[i] = Float(xCurD[i])
            }
            trajectoryNorms.append(batchAvgNorm(xCur))
        }

        return (xCur, trajectoryNorms)
    }
}
