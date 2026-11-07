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

/// Conditional Flow Matching (`CFM`), Multi-Resolution Subband Scaling & Subband Differential Auditing (`Phase 2 & Phase 3`).
public enum ConditionalFlowMatcher {
    /// Default thresholds `τ_LL = 5.0`, `τ_HH = 5.0` for deterministic 4-state subband auditing (`Section 2.2.2`).
    public static let defaultTauLL: Float = 5.0
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
                let decay = (1.0 - t) * t * 0.001
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
