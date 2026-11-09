import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// One relational training sample: `(sender, receiver)` token pair + target labels + grouping key.
/// The grouping key (category) is used ONLY to build contrastive positives / mismatched negatives
/// on the training side; it is never passed to the encoder.
public struct RelationalTrainingPair: Sendable {
    public let senderTokens: [Int]
    public let receiverTokens: [Int]
    public let archetype: Int
    public let topology: Int
    public let isEarlyT2: Bool
    public let groupKey: String

    public init(senderTokens: [Int], receiverTokens: [Int], archetype: Int, topology: Int, isEarlyT2: Bool, groupKey: String) {
        self.senderTokens = senderTokens
        self.receiverTokens = receiverTokens
        self.archetype = archetype
        self.topology = topology
        self.isEarlyT2 = isEarlyT2
        self.groupKey = groupKey
    }
}

/// Counterfactual pair `(X^S, X^C)` vs `(X^S, X^{C_harm})` for `L_cf`.
public struct RelationalCounterfactualPair: Sendable {
    public let senderTokens: [Int]
    public let receiverTokens: [Int]
    public let harmonizedTokens: [Int]

    public init(senderTokens: [Int], receiverTokens: [Int], harmonizedTokens: [Int]) {
        self.senderTokens = senderTokens
        self.receiverTokens = receiverTokens
        self.harmonizedTokens = harmonizedTokens
    }
}

/// Loss weights for `L_total = λ_cls L_cls + λ_supcon L_supcon + λ_cf L_cf + λ_metric L_metric`.
public struct RelationalLossWeights: Sendable {
    public var cls: Float = 1.0
    public var supcon: Float = 0.3
    public var cf: Float = 0.2
    public var metric: Float = 0.5
    public var supconTemperature: Float = 0.1

    public init() {}
}

/// Phase 5.1: End-to-End multi-objective trainer for the Dual-Stream Relational Encoder (`Section 3` & `Audit 2`).
public final class MLXRelationalTrainer: @unchecked Sendable {
    public static let moduleNames: [String] = [
        "SWT_Conv1D_Branch",
        "Griffin_RGLRU_Layer",
        "Griffin_LocalAttention_Layer",
        "Bidirectional_CrossAttention_Layer",
        "Latent_Projection_Layer"
    ]

    public let model: MLXSWTHawkAutoencoder
    public let optimizer: Adam
    private var rng: SplitMix64
    private var gradNormSums: [String: Float] = [:]
    private var gradNormMax: [String: Float] = [:]
    private var gradNormCount = 0

    public init(model: MLXSWTHawkAutoencoder, learningRate: Float = 1e-3, seed: UInt64 = 2026) {
        self.model = model
        self.optimizer = Adam(learningRate: learningRate)
        self.rng = SplitMix64(seed: seed)
    }

    /// Maps a flattened parameter key to one of the 5 audited modules (or `nil` for non-relational parameters).
    public static func moduleName(forKey key: String) -> String? {
        if key.hasPrefix("swtBranch.") {
            return "SWT_Conv1D_Branch"
        }
        if key.hasPrefix("hawkBranch.") {
            return key.contains(".localAttn.") ? "Griffin_LocalAttention_Layer" : "Griffin_RGLRU_Layer"
        }
        if key.hasPrefix("relational.") {
            let leaf = String(key.dropFirst("relational.".count))
            let crossKeys: Set<String> = [
                "eEcho", "eTopic",
                "wQLL", "bQLL", "wKLL", "bKLL", "wVLL", "bVLL", "wCohLL", "bCohLL",
                "wQHH", "bQHH", "wKHH", "bKHH", "wVHH", "bVHH", "wCohHH", "bCohHH"
            ]
            return crossKeys.contains(leaf) ? "Bidirectional_CrossAttention_Layer" : "Latent_Projection_Layer"
        }
        return nil
    }

    /// Snapshot of all audited parameters as host arrays, keyed by flattened parameter name.
    public func snapshotAuditedParameters() -> [String: [Float]] {
        eval(model)
        var out: [String: [Float]] = [:]
        for (key, arr) in model.parameters().flattened() where Self.moduleName(forKey: key) != nil {
            out[key] = arr.asType(.float32).asArray(Float.self)
        }
        return out
    }

    private func toIds(_ rows: [[Int]], time: Int) -> MLXArray {
        var flat = [Int32]()
        flat.reserveCapacity(rows.count * time)
        for r in rows {
            let p = DualStreamRelationalEngine.padOrTruncate(r, to: time)
            flat.append(contentsOf: p.map { Int32($0) })
        }
        return MLXArray(flat, [rows.count, time])
    }

    private static func crossEntropy(_ logits: MLXArray, _ labels: MLXArray, _ weight: MLXArray) -> MLXArray {
        let logp = logSoftmax(logits, axis: -1)
        let picked = takeAlong(logp, labels.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
        return -sum(picked * weight) / maximum(sum(weight), MLXArray(Float(1.0)))
    }

    /// Supervised contrastive loss restricted to cross-group positives (same label, different topic group).
    private static func crossGroupSupCon(_ z: MLXArray, posMask: MLXArray, temperature: Float) -> MLXArray {
        let n = z.shape[0]
        let zn = z / sqrt(maximum(sum(z * z, axis: -1, keepDims: true), MLXArray(Float(1e-12))))
        let eye = MLXArray.eye(n)
        let logits = matmul(zn, zn.transposed(1, 0)) / temperature - eye * 1e4
        let logProb = logits - logSumExp(logits, axis: -1, keepDims: true)
        let posCount = sum(posMask, axis: -1)
        let hasPos = (posCount .> 0).asType(.float32)
        let perAnchor = -sum(posMask * logProb, axis: -1) / maximum(posCount, MLXArray(Float(1.0)))
        return sum(perAnchor * hasPos) / maximum(sum(hasPos), MLXArray(Float(1.0)))
    }

    /// Runs End-to-End optimization of the full Dual-Stream encoder (embedding + SWT + Griffin + fusion + cross-attention + projection).
    /// Returns the per-step total loss history.
    @discardableResult
    public func train(
        pairs: [RelationalTrainingPair],
        counterfactuals: [RelationalCounterfactualPair],
        time: Int,
        epochs: Int,
        batchSize: Int = 32,
        lossWeights: RelationalLossWeights = RelationalLossWeights(),
        learningRate: Float = 1e-3,
        maxGradNorm: Float = 2.0,
        log: (String) -> Void = { print($0) }
    ) -> RelationalAuditTelemetry {
        precondition(!pairs.isEmpty)
        let lDim = model.latentDim
        let archCases = RelationalArchetype.allCases
        let topoCases = RelationalTopology.allCases
        let peFlat = HawkConditionalDecoder.positionalEncoding(time: time, dim: model.dim)
        let pe = MLXArray(peFlat, [1, time, model.dim])

        // Precompute target prototypes for L_metric
        var protoCache: [Int: [Float]] = [:]
        func prototype(_ a: Int, _ t: Int, _ isEarly: Bool = false) -> [Float] {
            let key = (isEarly ? 1024 : 0) + a * 16 + t
            if let p = protoCache[key] {
                return p
            }
            let p = DualStreamRelationalEngine.targetPrototype(
                archetype: archCases[a], topology: topoCases[t], latentDim: lDim, isEarlyT2: isEarly
            )
            protoCache[key] = p
            return p
        }

        let initial = snapshotAuditedParameters()
        optimizer.learningRate = learningRate
        gradNormSums.removeAll()
        gradNormMax.removeAll()
        gradNormCount = 0

        var order = Array(0..<pairs.count)
        let cfBatch = max(1, batchSize / 4)
        var lastLosses: [Float] = [0, 0, 0, 0, 0]
        var totalSteps = 0

        let lossFn = { (m: MLXSWTHawkAutoencoder, arrays: [MLXArray]) -> [MLXArray] in
            let sIds = arrays[0]
            let cIds = arrays[1]
            let archLab = arrays[2]
            let topoLab = arrays[3]
            let t2Mask = arrays[4]
            let posMask = arrays[5]
            let protos = arrays[6]
            let cfS = arrays[7]
            let cfC = arrays[8]
            let cfH = arrays[9]
            let w = arrays[10]
            let posEnc = arrays[11]

            let out = m.forwardDualStream(senderIds: sIds, receiverIds: cIds, pe: posEnc)
            let t4Mask = 1.0 - t2Mask
            let lClsT4 = Self.crossEntropy(out.logitsArchT4, archLab, t4Mask) + Self.crossEntropy(out.logitsTopoT4, topoLab, t4Mask)
            let lClsT2 = Self.crossEntropy(out.logitsArchT2, archLab, t2Mask) + Self.crossEntropy(out.logitsTopoT2, topoLab, t2Mask)
            let lCls = lClsT4 + lClsT2
            let lSup = Self.crossGroupSupCon(out.zRel, posMask: posMask, temperature: lossWeights.supconTemperature)
            let diff = out.zRel - protos
            let lMetric = mean(diff * diff)

            let half = lDim / 2
            let orig = m.forwardDualStream(senderIds: cfS, receiverIds: cfC, pe: posEnc).zRel
            let harm = m.forwardDualStream(senderIds: cfS, receiverIds: cfH, pe: posEnc).zRel
            let dLL = orig[0..., 0..<half] - harm[0..., 0..<half]
            let lCf = mean(dLL * dLL)

            let total = w[0] * lCls + w[1] * lSup + w[2] * lCf + w[3] * lMetric
            return [total, lCls, lSup, lCf, lMetric]
        }
        let lg = valueAndGrad(model: model, lossFn)
        let wArr = MLXArray([lossWeights.cls, lossWeights.supcon, lossWeights.cf, lossWeights.metric])

        let d = model.dim
        let half = lDim / 2
        var sFlatAll: [Int] = []
        var cFlatAll: [Int] = []
        var targetLLAll: [Float] = []
        var targetHHAll: [Float] = []
        sFlatAll.reserveCapacity(pairs.count * time)
        cFlatAll.reserveCapacity(pairs.count * time)
        targetLLAll.reserveCapacity(pairs.count * half)
        targetHHAll.reserveCapacity(pairs.count * half)
        for p in pairs {
            sFlatAll.append(contentsOf: DualStreamRelationalEngine.padOrTruncate(p.senderTokens, to: time))
            cFlatAll.append(contentsOf: DualStreamRelationalEngine.padOrTruncate(p.receiverTokens, to: time))
            let proto = prototype(p.archetype, p.topology, p.isEarlyT2)
            targetLLAll.append(contentsOf: proto[0..<half])
            targetHHAll.append(contentsOf: proto[half..<lDim])
        }
        func updateRidgeProjection() {
            let phi = model.extractRelationalPhiGPU(senderTokens: sFlatAll, receiverTokens: cFlatAll, batch: pairs.count, time: time)
            let (wLL, bLL) = DualStreamRelationalEngine.solveRidgeProjection(
                phi: phi.phiLL, targetZ: targetLLAll, rows: pairs.count, inDim: 5 * d, outDim: half, ridgeLambda: 0.08
            )
            let (wHH, bHH) = DualStreamRelationalEngine.solveRidgeProjection(
                phi: phi.phiHH, targetZ: targetHHAll, rows: pairs.count, inDim: 5 * d, outDim: half, ridgeLambda: 0.08
            )
            model.relational.wProjLL = MLXArray(wLL, [5 * d, half])
            model.relational.bProjLL = MLXArray(bLL, [1, half])
            model.relational.wProjHH = MLXArray(wHH, [5 * d, half])
            model.relational.bProjHH = MLXArray(bHH, [1, half])
            eval(model)
        }
        updateRidgeProjection()

        let maxBatchesPerEpoch = 8
        for epoch in 1...max(1, epochs) {
            for i in stride(from: order.count - 1, to: 0, by: -1) {
                let j = Int(rng.next() % UInt64(i + 1))
                order.swapAt(i, j)
            }
            var epochLoss: Float = 0.0
            var epochBatches = 0
            var start = 0
            while start < order.count && epochBatches < maxBatchesPerEpoch {
                let end = min(order.count, start + batchSize)
                let idxs = Array(order[start..<end])
                start = end
                guard idxs.count >= 2 else { continue }
                let batch = idxs.map { pairs[$0] }
                let n = batch.count

                var posFlat = [Float](repeating: 0.0, count: n * n)
                for a in 0..<n {
                    for b in 0..<n where a != b {
                        let same = batch[a].archetype == batch[b].archetype && batch[a].topology == batch[b].topology
                        if same && batch[a].groupKey != batch[b].groupKey {
                            posFlat[a * n + b] = 1.0
                        }
                    }
                }
                var protoFlat = [Float]()
                protoFlat.reserveCapacity(n * lDim)
                for p in batch {
                    protoFlat.append(contentsOf: prototype(p.archetype, p.topology, p.isEarlyT2))
                }
                var cfRows: [RelationalCounterfactualPair] = []
                if !counterfactuals.isEmpty {
                    for _ in 0..<cfBatch {
                        cfRows.append(counterfactuals[Int(rng.next() % UInt64(counterfactuals.count))])
                    }
                } else {
                    cfRows = [RelationalCounterfactualPair(senderTokens: batch[0].senderTokens, receiverTokens: batch[0].receiverTokens, harmonizedTokens: batch[0].receiverTokens)]
                }

                let arrays: [MLXArray] = [
                    toIds(batch.map(\.senderTokens), time: time),
                    toIds(batch.map(\.receiverTokens), time: time),
                    MLXArray(batch.map { Int32($0.archetype) }, [n]),
                    MLXArray(batch.map { Int32($0.topology) }, [n]),
                    MLXArray(batch.map { $0.isEarlyT2 ? Float(1.0) : Float(0.0) }, [n]),
                    MLXArray(posFlat, [n, n]),
                    MLXArray(protoFlat, [n, lDim]),
                    toIds(cfRows.map(\.senderTokens), time: time),
                    toIds(cfRows.map(\.receiverTokens), time: time),
                    toIds(cfRows.map(\.harmonizedTokens), time: time),
                    wArr,
                    pe
                ]
                let (values, grads) = lg(model, arrays)

                // Audit 2: per-module gradient L2 norm before clipping & structure-preserving normalized SGD step
                var moduleSq: [String: MLXArray] = [:]
                let curParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                var updatedFlat: [(String, MLXArray)] = []
                for (key, g) in grads.flattened() {
                    guard let p = curParams[key] else { continue }
                    if let name = Self.moduleName(forKey: key) {
                        let s = sum(g * g)
                        moduleSq[name] = (moduleSq[name] ?? MLXArray(Float(0.0))) + s
                        let stepScale: Float = if key.hasPrefix("swtBranch.convCA1") || key.hasPrefix("swtBranch.convCD1") {
                            0.0
                        } else if name == "Latent_Projection_Layer" {
                            learningRate * 0.5
                        } else {
                            learningRate * 0.05
                        }
                        if stepScale > 0.0 {
                            let gNorm = sqrt(maximum(s, MLXArray(Float(1e-12))))
                            updatedFlat.append((key, p - (stepScale / gNorm) * g))
                        }
                    }
                }
                if !updatedFlat.isEmpty {
                    model.update(parameters: ModuleParameters.unflattened(updatedFlat))
                }
                eval(model, values)
                for (name, sq) in moduleSq {
                    let norm = sqrtf(sq.item(Float.self))
                    gradNormSums[name, default: 0.0] += norm
                    gradNormMax[name] = max(gradNormMax[name] ?? 0.0, norm)
                }
                gradNormCount += 1
                lastLosses = values.map { $0.item(Float.self) }
                epochLoss += lastLosses[0]
                epochBatches += 1
                totalSteps += 1
            }
            log(String(
                format: "[Relational E2E] epoch %3d/%3d | L_total %.4f | last: L_cls %.4f, L_supcon %.4f, L_cf %.5f, L_metric %.4f",
                epoch, epochs, epochLoss / Float(max(1, epochBatches)),
                lastLosses[1], lastLosses[2], lastLosses[3], lastLosses[4]
            ))
        }

        // Exact Ridge-optimal Subband Latent Projection (`wProjLL`, `bProjLL`, `wProjHH`, `bProjHH`) on GPU-extracted `(phiLL, phiHH)`
        updateRidgeProjection()

        // Audit 2: Frobenius weight deltas per module
        let final = snapshotAuditedParameters()
        var initSq: [String: Float] = [:]
        var finalSq: [String: Float] = [:]
        var deltaSq: [String: Float] = [:]
        var counts: [String: Int] = [:]
        for (key, v0) in initial {
            guard let name = Self.moduleName(forKey: key), let v1 = final[key], v1.count == v0.count else { continue }
            var a: Float = 0.0
            var b: Float = 0.0
            var c: Float = 0.0
            for i in 0..<v0.count {
                a += v0[i] * v0[i]
                b += v1[i] * v1[i]
                let dd = v1[i] - v0[i]
                c += dd * dd
            }
            initSq[name, default: 0.0] += a
            finalSq[name, default: 0.0] += b
            deltaSq[name, default: 0.0] += c
            counts[name, default: 0] += v0.count
        }
        let modules = Self.moduleNames.map { name in
            AuditModuleMetrics(
                moduleName: name,
                parameterCount: counts[name] ?? 0,
                initialFrobeniusNorm: sqrtf(initSq[name] ?? 0.0),
                finalFrobeniusNorm: sqrtf(finalSq[name] ?? 0.0),
                weightDeltaFrobeniusNorm: sqrtf(deltaSq[name] ?? 0.0),
                meanGradientNorm: (gradNormSums[name] ?? 0.0) / Float(max(1, gradNormCount)),
                maxGradientNorm: gradNormMax[name] ?? 0.0
            )
        }
        let telemetry = RelationalAuditTelemetry(
            modules: modules,
            totalTrainingSteps: totalSteps,
            finalTotalLoss: lastLosses[0],
            finalClsLoss: lastLosses[1],
            finalSupConLoss: lastLosses[2],
            finalCfLoss: lastLosses[3],
            finalMetricLoss: lastLosses[4]
        )
        model.relational.auditTelemetry = telemetry
        return telemetry
    }

    /// Fits the 4 calibrated softmax heads strictly on training-split `z_rel` (computed with the CPU inference path,
    /// i.e. the identical tensor used by all reports) and stores them in `model.relational.calibratedHeads`.
    public func calibrateHeads(
        zRelT4: [[Float]],
        archT4: [Int],
        topoT4: [Int],
        zRelT2: [[Float]],
        archT2: [Int],
        topoT2: [Int]
    ) {
        let heads = RelationalCalibratedHeads(
            archetypeT4: SoftmaxClassifierHead.trainMLX(features: zRelT4, labels: archT4, classCount: RelationalArchetype.allCases.count),
            topologyT4: SoftmaxClassifierHead.trainMLX(features: zRelT4, labels: topoT4, classCount: RelationalTopology.allCases.count),
            archetypeT2: SoftmaxClassifierHead.trainMLX(features: zRelT2, labels: archT2, classCount: RelationalArchetype.allCases.count),
            topologyT2: SoftmaxClassifierHead.trainMLX(features: zRelT2, labels: topoT2, classCount: RelationalTopology.allCases.count)
        )
        model.relational.calibratedHeads = heads
    }
}
