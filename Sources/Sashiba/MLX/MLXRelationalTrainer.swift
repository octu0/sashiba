import Foundation
import MLX
import MLXNN

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
    public let uniquenessWeight: Float

    public init(
        senderTokens: [Int],
        receiverTokens: [Int],
        archetype: Int,
        topology: Int,
        isEarlyT2: Bool,
        groupKey: String,
        uniquenessWeight: Float = 1.0
    ) {
        self.senderTokens = senderTokens
        self.receiverTokens = receiverTokens
        self.archetype = archetype
        self.topology = topology
        self.isEarlyT2 = isEarlyT2
        self.groupKey = groupKey
        self.uniquenessWeight = uniquenessWeight
    }
}

/// Counterfactual pair `(X^S, X^C)` vs `(X^S, X^{C_harm})` for `L_cf`.
public struct RelationalCounterfactualPair: Sendable {
    public let senderTokens: [Int]
    public let receiverTokens: [Int]
    public let harmonizedTokens: [Int]
    public let preserveLL: Bool
    public let separateHH: Bool

    public init(
        senderTokens: [Int],
        receiverTokens: [Int],
        harmonizedTokens: [Int],
        preserveLL: Bool = true,
        separateHH: Bool = false
    ) {
        self.senderTokens = senderTokens
        self.receiverTokens = receiverTokens
        self.harmonizedTokens = harmonizedTokens
        self.preserveLL = preserveLL
        self.separateHH = separateHH
    }
}

/// Loss weights for `L_total = λ_cls L_cls + λ_supcon L_supcon + λ_cf L_cf + λ_metric L_metric + λ_orth L_orth`.
public struct RelationalLossWeights: Sendable {
    public var cls: Float
    public var supcon: Float
    public var cf: Float
    public var metric: Float
    public var orth: Float
    public var dropout: Float
    public var supconTemperature: Float

    public init(
        cls: Float = 1.0,
        supcon: Float = 0.35,
        cf: Float = 0.25,
        metric: Float = 0.45,
        orth: Float = 0.14,
        dropout: Float = 0.22,
        supconTemperature: Float = 0.12
    ) {
        self.cls = cls
        self.supcon = supcon
        self.cf = cf
        self.metric = metric
        self.orth = orth
        self.dropout = dropout
        self.supconTemperature = supconTemperature
    }
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
    public var baseLearningRate: Float
    private var rng: SplitMix64
    private var gradNormSums: [String: Float] = [:]
    private var gradNormMax: [String: Float] = [:]
    private var gradNormCount = 0

    public init(model: MLXSWTHawkAutoencoder, learningRate: Float = 1.5e-3, seed: UInt64 = 2026) {
        self.model = model
        self.baseLearningRate = learningRate
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
            let crossPrefixes = [
                "convRel", "wPool", "eEcho", "eTopic",
                "wQ", "bQ", "wK", "bK", "wV", "bV", "wCoh", "bCoh"
            ]
            for prefix in crossPrefixes where leaf.hasPrefix(prefix) {
                return "Bidirectional_CrossAttention_Layer"
            }
            return "Latent_Projection_Layer"
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

    @inline(__always)
    private static func l2NormalizeRows(_ x: MLXArray) -> MLXArray {
        return x * rsqrt(maximum(sum(x * x, axis: -1, keepDims: true), MLXArray(Float(1e-12))))
    }

    private static func crossEntropy(
        _ logits: MLXArray,
        _ labels: MLXArray,
        _ weight: MLXArray,
        smoothing: Float = 0.04
    ) -> MLXArray {
        let logp = logSoftmax(logits, axis: -1)
        let picked = takeAlong(logp, labels.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
        let smoothNLL = -mean(logp, axis: -1)
        let perRow = (1.0 - smoothing) * (-picked) + smoothing * smoothNLL
        return sum(perRow * weight) / maximum(sum(weight), MLXArray(Float(1.0)))
    }

    /// Supervised contrastive loss restricted to cross-group positives (same label & phase, different topic group).
    private static func crossGroupSupCon(_ z: MLXArray, posMask: MLXArray, temperature: Float) -> MLXArray {
        let n = z.shape[0]
        let zn = l2NormalizeRows(z)
        let eye = MLXArray.eye(n)
        let logits = matmul(zn, zn.transposed(1, 0)) / temperature - eye * 1e4
        let logProb = logits - logSumExp(logits, axis: -1, keepDims: true)
        let posCount = sum(posMask, axis: -1)
        let hasPos = (posCount .> 0).asType(.float32)
        let perAnchor = -sum(posMask * logProb, axis: -1) / maximum(posCount, MLXArray(Float(1.0)))
        return sum(perAnchor * hasPos) / maximum(sum(hasPos), MLXArray(Float(1.0)))
    }

    /// Runs End-to-End stateful GPU AdamW optimization of the Dual-Stream encoder
    /// (`swtBranch` + `hawkBranch` + `fusion` + `Bidirectional_CrossAttention_Layer` + `Latent_Projection_Layer`).
    @discardableResult
    public func train(
        pairs: [RelationalTrainingPair],
        counterfactuals: [RelationalCounterfactualPair],
        time: Int,
        epochs: Int,
        batchSize: Int = 32,
        lossWeights: RelationalLossWeights = RelationalLossWeights(),
        learningRate: Float = 1.5e-3,
        maxGradNorm: Float = 2.0,
        log: (String) -> Void = { print($0) }
    ) -> RelationalAuditTelemetry {
        precondition(!pairs.isEmpty)
        MLX.Memory.cacheLimit = 128 * 1024 * 1024
        MLX.Memory.clearCache()
        try? model.freeze(keys: ["tokenEmbedding", "pooling", "decoder", "cfm"], strict: false)
        model.isTokenEmbeddingFrozen = true
        defer {
            model.isTokenEmbeddingFrozen = false
            try? model.unfreeze(strict: false)
            MLX.Memory.clearCache()
        }
        let d = model.dim
        let lDim = model.relational.latentDim
        let half = lDim / 2
        let phiDim = 5 * d
        let archCases = RelationalArchetype.allCases
        let topoCases = RelationalTopology.allCases
        let peFlat = HawkConditionalDecoder.positionalEncoding(time: time, dim: d)
        let pe = MLXArray(peFlat, [1, time, d])
        let ones2D = MLXArray.ones([2 * d])
        let zeros2D = MLXArray.zeros([2 * d])

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
        baseLearningRate = learningRate
        gradNormSums.removeAll()
        gradNormMax.removeAll()
        gradNormCount = 0

        // Snapshot initial MLX tensors for shared Autoencoder backbone trust-region anchor regularization
        var pInitMap: [String: MLXArray] = [:]
        for (key, arr) in model.parameters().flattened() {
            if key.hasPrefix("swtBranch.") || key.hasPrefix("hawkBranch.") || key.hasPrefix("fusion.") {
                pInitMap[key] = stopGradient(arr)
            }
        }
        eval(Array(pInitMap.values))

        var mState: [String: MLXArray] = [:]
        var vState: [String: MLXArray] = [:]
        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let eps: Float = 1e-8
        let relWeightDecay: Float = 2.5e-3

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
            let cfPreserveLL = arrays[10]
            let cfSeparateHH = arrays[11]
            let dropLL = arrays[12]
            let dropHH = arrays[13]
            let w = arrays[14]
            let posEnc = arrays[15]
            let tokKeepS = arrays[16]
            let tokKeepC = arrays[17]
            let t4Mask = arrays[18]

            let n = sIds.shape[0]
            let cfN = cfS.shape[0]
            let bTime = sIds.shape[1]
            let allS = concatenated([sIds, cfS, cfS], axis: 0)
            let allC = concatenated([cIds, cfC, cfH], axis: 0)
            let cfOnesDrop = MLXArray.ones([2 * cfN, phiDim])
            let allDropLL = concatenated([dropLL, cfOnesDrop], axis: 0)
            let allDropHH = concatenated([dropHH, cfOnesDrop], axis: 0)
            let cfOnesKeep = MLXArray.ones([2 * cfN, bTime, 1])
            let allKeepS = concatenated([tokKeepS, cfOnesKeep], axis: 0)
            let allKeepC = concatenated([tokKeepC, cfOnesKeep], axis: 0)

            let comb = m.forwardDualStream(
                senderIds: allS,
                receiverIds: allC,
                pe: posEnc,
                dropoutMaskLL: allDropLL,
                dropoutMaskHH: allDropHH,
                tokenKeepMaskS: allKeepS,
                tokenKeepMaskC: allKeepC
            )
            let mainZRel = comb.zRel[0..<n]
            let mainZContent = comb.zContent[0..<n]
            let mainUContent = comb.uContent[0..<n]

            // 1. L_cls: Label-smoothed class-balanced cross-entropy on T4 and T2 classification heads
            let lClsT4 = Self.crossEntropy(comb.logitsArchT4[0..<n], archLab, t4Mask, smoothing: 0.06)
                + Self.crossEntropy(comb.logitsTopoT4[0..<n], topoLab, t4Mask, smoothing: 0.06)
            let lClsT2 = Self.crossEntropy(comb.logitsArchT2[0..<n], archLab, t2Mask, smoothing: 0.06)
                + Self.crossEntropy(comb.logitsTopoT2[0..<n], topoLab, t2Mask, smoothing: 0.06)
            let lCls = lClsT4 + lClsT2

            // 2. L_supcon: Cross-group supervised contrastive loss within the same temporal phase
            let lSup = Self.crossGroupSupCon(mainZRel, posMask: posMask, temperature: lossWeights.supconTemperature)

            // 3. L_metric: Subband geometric prototype regression (balanced across groups/classes)
            let diff = mainZRel - protos
            let rowSq = mean(diff * diff, axis: -1)
            let rowW = t4Mask + t2Mask
            let lMetric = sum(rowSq * rowW) / maximum(sum(rowW), MLXArray(Float(1.0)))

            // 4. L_cf: Counterfactual LL preservation & HH high-tension separation (zero hardcoded prototypes)
            let orig = comb.zRel[n..<(n + cfN)]
            let harm = comb.zRel[(n + cfN)..<(n + 2 * cfN)]
            let dLL = (orig[0..., 0..<half] - harm[0..., 0..<half]) * cfPreserveLL
            let origHHNorm = Self.l2NormalizeRows(orig[0..., half..<lDim])
            let harmHHNorm = Self.l2NormalizeRows(harm[0..., half..<lDim])
            let cosHH = sum(origHHNorm * harmHHNorm, axis: -1, keepDims: true)
            let sepHH = maximum(MLXArray(Float(0.0)), cosHH - 0.25) * cfSeparateHH
            let lCf = mean(dLL * dLL) + 0.5 * mean(sepHH * sepHH)

            // 5. L_orth: Content reconstruction + Relational/Content subspace orthogonality (Section 2.3)
            let uContTarget = mlxLayerNorm(stopGradient(mainUContent), gamma: ones2D, beta: zeros2D)
            let reconCont = matmul(mainZContent, m.relational.wContent.transposed(1, 0))
            let diffCont = reconCont - uContTarget
            let lContRecon = mean(diffCont * diffCont)

            let zRelLLNorm = Self.l2NormalizeRows(mainZRel[0..., 0..<half])
            let zRelHHNorm = Self.l2NormalizeRows(mainZRel[0..., half..<lDim])
            let zRelNorm = Self.l2NormalizeRows(mainZRel)
            let zContNorm = Self.l2NormalizeRows(mainZContent)
            let cosLL = sum(zRelLLNorm * zContNorm, axis: -1)
            let cosHHCont = sum(zRelHHNorm * zContNorm, axis: -1)
            let crossMat = matmul(zRelNorm.transposed(1, 0), zContNorm)
            let lOrthCos = mean(cosLL * cosLL) + mean(cosHHCont * cosHHCont) + mean(crossMat * crossMat) * Float(half)
            let lOrth = 0.5 * lContRecon + lOrthCos

            let total = w[0] * lCls + w[1] * lSup + w[2] * lCf + w[3] * lMetric + w[4] * lOrth
            return [total, lCls, lSup, lCf, lMetric, lOrth]
        }
        let lg = valueAndGrad(model: model, lossFn)
        let wArr = MLXArray([lossWeights.cls, lossWeights.supcon, lossWeights.cf, lossWeights.metric, lossWeights.orth])

        let effEpochs = max(1, epochs)
        let effBatch = max(2, batchSize)
        var batchesPerEpoch = 0
        do {
            var probeStart = 0
            while probeStart < pairs.count {
                let rawEnd = min(pairs.count, probeStart + effBatch)
                let end = (pairs.count - rawEnd == 1) ? pairs.count : rawEnd
                if end - probeStart >= 2 {
                    batchesPerEpoch += 1
                }
                probeStart = end
            }
        }
        let plannedTotalSteps = max(1, effEpochs * max(1, batchesPerEpoch))
        let warmupSteps = max(1, plannedTotalSteps / 10)
        let dropP: Float = (pairs.count > 8) ? min(0.35, max(lossWeights.dropout, 0.22)) : 0.0
        let useTokDrop = pairs.count > 8
        let alphaHost = model.relational.alphaRel

        var t4ArchCounts = [Int](repeating: 0, count: archCases.count)
        var t2ArchCounts = [Int](repeating: 0, count: archCases.count)
        var groupPairCounts: [String: Int] = [:]
        var nT4 = 0
        var nT2 = 0
        for p in pairs {
            if !p.groupKey.hasPrefix("MISMATCH:") {
                groupPairCounts[p.groupKey, default: 0] += 1
            }
            if p.isEarlyT2 {
                if p.archetype >= 0 && p.archetype < archCases.count { t2ArchCounts[p.archetype] += 1 }
                nT2 += 1
            } else {
                if p.archetype >= 0 && p.archetype < archCases.count { t4ArchCounts[p.archetype] += 1 }
                nT4 += 1
            }
        }
        let maxGroupPairs = Float(max(1, groupPairCounts.values.max() ?? 1))
        let meanT4 = Float(max(1, nT4)) / Float(max(1, t4ArchCounts.filter { $0 > 0 }.count))
        let meanT2 = Float(max(1, nT2)) / Float(max(1, t2ArchCounts.filter { $0 > 0 }.count))
        func sampleClsWeight(_ p: RelationalTrainingPair) -> Float {
            if p.groupKey.hasPrefix("MISMATCH:") {
                let c = max(1, t4ArchCounts[p.archetype])
                return 0.72 * min(1.75, max(0.75, powf(meanT4 / Float(c), 0.30)))
            }
            let gMul = min(1.50, max(0.85, powf(maxGroupPairs / Float(max(1, groupPairCounts[p.groupKey] ?? 1)), 0.35))) * p.uniquenessWeight
            if p.isEarlyT2 {
                let c = max(1, t2ArchCounts[p.archetype])
                return gMul * min(1.75, max(0.75, powf(meanT2 / Float(c), 0.30)))
            } else {
                let c = max(1, t4ArchCounts[p.archetype])
                return gMul * min(1.75, max(0.75, powf(meanT4 / Float(c), 0.30)))
            }
        }

        var order = Array(0..<pairs.count)
        let cfBatch = max(1, effBatch / 4)
        var lastLosses: [Float] = [0, 0, 0, 0, 0, 0]
        var totalSteps = 0

        for epoch in 1...effEpochs {
            for i in stride(from: order.count - 1, to: 0, by: -1) {
                let j = Int(rng.next() % UInt64(i + 1))
                order.swapAt(i, j)
            }
            var epochLoss: Float = 0.0
            var epochBatches = 0
            var start = 0
            while start < order.count {
                let rawEnd = min(order.count, start + effBatch)
                let end = (order.count - rawEnd == 1) ? order.count : rawEnd
                let idxs = Array(order[start..<end])
                start = end
                guard idxs.count >= 2 else { continue }
                let batch = idxs.map { pairs[$0] }
                let n = batch.count

                var posFlat = [Float](repeating: 0.0, count: n * n)
                for a in 0..<n {
                    for b in 0..<n where a != b {
                        let same = batch[a].isEarlyT2 == batch[b].isEarlyT2
                            && batch[a].archetype == batch[b].archetype
                            && batch[a].topology == batch[b].topology
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
                cfRows.reserveCapacity(cfBatch)
                if !counterfactuals.isEmpty {
                    for _ in 0..<cfBatch {
                        cfRows.append(counterfactuals[Int(rng.next() % UInt64(counterfactuals.count))])
                    }
                } else {
                    cfRows = [
                        RelationalCounterfactualPair(
                            senderTokens: batch[0].senderTokens,
                            receiverTokens: batch[0].receiverTokens,
                            harmonizedTokens: batch[0].receiverTokens
                        )
                    ]
                }
                let cfN = cfRows.count

                let batchTime = time
                let peSlice = pe

                let dropLLArr: MLXArray
                let dropHHArr: MLXArray
                if dropP > 0.0 {
                    let keepScale: Float = 1.0 / (1.0 - dropP)
                    var dropLLFlat = [Float](repeating: keepScale, count: n * phiDim)
                    var dropHHFlat = [Float](repeating: keepScale, count: n * phiDim)
                    for idx in 0..<(n * phiDim) {
                        if rng.nextUnitFloat() < dropP { dropLLFlat[idx] = 0.0 }
                        if rng.nextUnitFloat() < dropP { dropHHFlat[idx] = 0.0 }
                    }
                    dropLLArr = MLXArray(dropLLFlat, [n, phiDim])
                    dropHHArr = MLXArray(dropHHFlat, [n, phiDim])
                } else {
                    dropLLArr = MLXArray.ones([n, phiDim])
                    dropHHArr = MLXArray.ones([n, phiDim])
                }

                let tokKeepSArr: MLXArray
                let tokKeepCArr: MLXArray
                if useTokDrop {
                    var keepSFlat = [Float](repeating: 1.0, count: n * batchTime)
                    var keepCFlat = [Float](repeating: 1.0, count: n * batchTime)
                    for r in 0..<n {
                        let sToks = batch[r].senderTokens
                        let cToks = batch[r].receiverTokens
                        let rowOff = r * batchTime
                        for t in 0..<batchTime {
                            let sid = (t < sToks.count) ? sToks[t] : 0
                            if sid > 0 && sid < alphaHost.count {
                                let pDrop: Float = (alphaHost[sid] < 0.90) ? 0.20 : 0.08
                                if rng.nextUnitFloat() < pDrop { keepSFlat[rowOff + t] = 0.0 }
                            }
                            let cid = (t < cToks.count) ? cToks[t] : 0
                            if cid > 0 && cid < alphaHost.count {
                                let pDrop: Float = (alphaHost[cid] < 0.90) ? 0.20 : 0.08
                                if rng.nextUnitFloat() < pDrop { keepCFlat[rowOff + t] = 0.0 }
                            }
                        }
                    }
                    tokKeepSArr = MLXArray(keepSFlat, [n, batchTime, 1])
                    tokKeepCArr = MLXArray(keepCFlat, [n, batchTime, 1])
                } else {
                    tokKeepSArr = MLXArray.ones([n, batchTime, 1])
                    tokKeepCArr = MLXArray.ones([n, batchTime, 1])
                }

                let arrays: [MLXArray] = [
                    toIds(batch.map(\.senderTokens), time: batchTime),
                    toIds(batch.map(\.receiverTokens), time: batchTime),
                    MLXArray(batch.map { Int32($0.archetype) }, [n]),
                    MLXArray(batch.map { Int32($0.topology) }, [n]),
                    MLXArray(batch.map { $0.isEarlyT2 ? sampleClsWeight($0) : Float(0.0) }, [n]),
                    MLXArray(posFlat, [n, n]),
                    MLXArray(protoFlat, [n, lDim]),
                    toIds(cfRows.map(\.senderTokens), time: batchTime),
                    toIds(cfRows.map(\.receiverTokens), time: batchTime),
                    toIds(cfRows.map(\.harmonizedTokens), time: batchTime),
                    MLXArray(cfRows.map { $0.preserveLL ? Float(1.0) : Float(0.0) }, [cfN, 1]),
                    MLXArray(cfRows.map { $0.separateHH ? Float(1.0) : Float(0.0) }, [cfN, 1]),
                    dropLLArr,
                    dropHHArr,
                    wArr,
                    peSlice,
                    tokKeepSArr,
                    tokKeepCArr,
                    MLXArray(batch.map { $0.isEarlyT2 ? Float(0.0) : sampleClsWeight($0) }, [n])
                ]
                let (values, grads) = lg(model, arrays)

                totalSteps += 1
                let curLR: Float
                if totalSteps <= warmupSteps && warmupSteps > 1 {
                    let wFrac = Float(totalSteps - 1) / Float(max(1, warmupSteps - 1))
                    curLR = learningRate * (0.20 + 0.80 * wFrac)
                } else {
                    let progress = min(1.0, Float(totalSteps - warmupSteps) / Float(max(1, plannedTotalSteps - warmupSteps)))
                    let cosine = 0.5 * (1.0 + cosf(Float.pi * progress))
                    curLR = learningRate * (0.15 + 0.85 * cosine)
                }

                // Audit 2 raw gradient norms + global L2 norm across trainable relational & shared backbone modules
                var moduleSq: [String: MLXArray] = [:]
                var totalGradSq = MLXArray(Float(0.0))
                let curParams = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                var trainableGrads: [(key: String, g: MLXArray, isRelational: Bool)] = []
                for (key, g) in grads.flattened() {
                    let isRelational = key.hasPrefix("relational.")
                    let isBackbone = key.hasPrefix("swtBranch.") || key.hasPrefix("hawkBranch.") || key.hasPrefix("fusion.")
                    guard isRelational || isBackbone else { continue }
                    guard curParams[key] != nil else { continue }
                    let sq = sum(g * g)
                    totalGradSq = totalGradSq + sq
                    if let name = Self.moduleName(forKey: key) {
                        moduleSq[name] = (moduleSq[name] ?? MLXArray(Float(0.0))) + sq
                    }
                    trainableGrads.append((key: key, g: g, isRelational: isRelational))
                }

                let globalGradNorm = sqrt(maximum(totalGradSq, MLXArray(Float(1e-12))))
                let clipCoef = minimum(MLXArray(Float(1.0)), MLXArray(maxGradNorm) / globalGradNorm)
                let bc1 = 1.0 - powf(beta1, Float(totalSteps))
                let bc2 = 1.0 - powf(beta2, Float(totalSteps))
                let invBC1: Float = 1.0 / bc1
                let invBC2: Float = 1.0 / bc2
                let backboneLR: Float = curLR * 0.015

                var updatedFlat: [(String, MLXArray)] = []
                updatedFlat.reserveCapacity(trainableGrads.count)
                for item in trainableGrads {
                    guard let p = curParams[item.key] else { continue }
                    let gClipped = item.g * clipCoef
                    let mPrev = mState[item.key] ?? MLXArray.zeros(like: p)
                    let vPrev = vState[item.key] ?? MLXArray.zeros(like: p)
                    let mNew = beta1 * mPrev + (1.0 - beta1) * gClipped
                    let vNew = beta2 * vPrev + (1.0 - beta2) * (gClipped * gClipped)
                    mState[item.key] = mNew
                    vState[item.key] = vNew

                    let mHat = mNew * invBC1
                    let vHat = vNew * invBC2
                    let adamDir = mHat / (sqrt(vHat) + eps)

                    let pNew: MLXArray
                    if item.isRelational {
                        let pDecayed = (p.ndim >= 2) ? (p * (1.0 - curLR * relWeightDecay)) : p
                        pNew = pDecayed - curLR * adamDir
                    } else {
                        let p0 = pInitMap[item.key] ?? p
                        pNew = p - backboneLR * adamDir - 0.05 * (p - p0)
                    }
                    updatedFlat.append((item.key, pNew))
                }

                if !updatedFlat.isEmpty {
                    model.update(parameters: ModuleParameters.unflattened(updatedFlat))
                }

                let moduleNormArrays = Self.moduleNames.map { name in
                    sqrt(moduleSq[name] ?? MLXArray(Float(0.0)))
                }
                eval(model, values + moduleNormArrays + Array(mState.values) + Array(vState.values))

                for (idx, name) in Self.moduleNames.enumerated() {
                    let norm = moduleNormArrays[idx].item(Float.self)
                    gradNormSums[name, default: 0.0] += norm
                    gradNormMax[name] = max(gradNormMax[name] ?? 0.0, norm)
                }
                gradNormCount += 1
                lastLosses = values.map { $0.item(Float.self) }
                epochLoss += lastLosses[0]
                epochBatches += 1
            }
            log(String(
                format: "[Relational E2E] epoch %3d/%3d | L_total %.4f | last: L_cls %.4f, L_supcon %.4f, L_cf %.5f, L_metric %.4f, L_orth %.4f",
                epoch, effEpochs, epochLoss / Float(max(1, epochBatches)),
                lastLosses[1], lastLosses[2], lastLosses[3], lastLosses[4], lastLosses[5]
            ))
            MLX.Memory.clearCache()
        }

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
        MLX.Memory.clearCache()
        return telemetry
    }

    /// Fits the 4 calibrated softmax heads strictly on training-split `z_rel` (computed with the GPU/CPU unified forward pass,
    /// i.e. the identical tensor used by all reports) and stores them in `model.relational.calibratedHeads`.
    public func calibrateHeads(
        zRelT4: [[Float]],
        archT4: [Int],
        topoT4: [Int],
        zRelT2: [[Float]],
        archT2: [Int],
        topoT2: [Int],
        groups: [String]? = nil,
        uniquenessWeights: [Float]? = nil,
        steps: Int = 350,
        l2: Float = 2e-3
    ) {
        func fitWithGroupCV(features: [[Float]], labels: [Int], classCount: Int) -> SoftmaxClassifierHead {
            guard let gList = groups, gList.count == features.count else {
                return SoftmaxClassifierHead.trainMLX(features: features, labels: labels, classCount: classCount, steps: steps, l2: l2, sampleWeights: uniquenessWeights)
            }
            var gCounts: [String: Int] = [:]
            for g in gList { gCounts[g, default: 0] += 1 }
            let maxGC = Float(max(1, gCounts.values.max() ?? 1))
            let hasU = (uniquenessWeights?.count == features.count)
            let allGroupWeights = (0..<features.count).map { idx -> Float in
                let gw = powf(maxGC / Float(max(1, gCounts[gList[idx]] ?? 1)), 0.30)
                return hasU ? (gw * uniquenessWeights![idx]) : gw
            }
            let uniqueGroups = Array(Set(gList)).sorted()
            guard uniqueGroups.count >= 3 else {
                return SoftmaxClassifierHead.trainMLX(features: features, labels: labels, classCount: classCount, steps: steps, l2: l2, sampleWeights: allGroupWeights)
            }
            var groupFold: [String: Int] = [:]
            for (idx, g) in uniqueGroups.enumerated() {
                groupFold[g] = idx % 3
            }
            let foldOfRow = gList.map { groupFold[$0] ?? 0 }
            let candidates: [Float] = [0.005, 0.02, 0.08, 0.25, 0.80, 2.0]
            let balCandidates: [Float] = [0.0, 0.25]
            var bestL2 = candidates[2]
            var bestBal: Float = 0.0
            var bestScore: Float = -1.0

            for candBal in balCandidates {
                for candL2 in candidates {
                    var wCorrect: Float = 0.0
                    var wTotal: Float = 0.0
                    var logProbSum: Float = 0.0
                    for fold in 0..<3 {
                        var trX: [[Float]] = []
                        var trY: [Int] = []
                        var trW: [Float] = []
                        var vaX: [[Float]] = []
                        var vaY: [Int] = []
                        var vaW: [Float] = []
                        for i in 0..<features.count {
                            if foldOfRow[i] == fold {
                                vaX.append(features[i])
                                vaY.append(labels[i])
                                vaW.append(allGroupWeights[i])
                            } else {
                                trX.append(features[i])
                                trY.append(labels[i])
                                trW.append(allGroupWeights[i])
                            }
                        }
                        guard !trX.isEmpty && !vaX.isEmpty else { continue }
                        let h = SoftmaxClassifierHead.trainMLX(features: trX, labels: trY, classCount: classCount, steps: 200, l2: candL2, classBalancePower: candBal, sampleWeights: trW)
                        for i in 0..<vaX.count {
                            let probs = h.probabilities(vaX[i])
                            var pred = 0
                            for c in 1..<probs.count where probs[c] > probs[pred] { pred = c }
                            let rw = vaW[i]
                            if pred == vaY[i] { wCorrect += rw }
                            let pTrue = (vaY[i] >= 0 && vaY[i] < probs.count) ? max(1e-7, probs[vaY[i]]) : 1e-7
                            logProbSum += rw * logf(pTrue)
                            wTotal += rw
                        }
                    }
                    guard wTotal > 0 else { continue }
                    let acc = wCorrect / wTotal
                    let tieBreak = 1e-4 * (logProbSum / wTotal)
                    let score = acc + tieBreak
                    if score > bestScore {
                        bestScore = score
                        bestL2 = candL2
                        bestBal = candBal
                    }
                }
            }
            return SoftmaxClassifierHead.trainMLX(features: features, labels: labels, classCount: classCount, steps: steps, l2: bestL2, classBalancePower: bestBal, sampleWeights: allGroupWeights)
        }

        let heads = RelationalCalibratedHeads(
            archetypeT4: fitWithGroupCV(features: zRelT4, labels: archT4, classCount: RelationalArchetype.allCases.count),
            topologyT4: fitWithGroupCV(features: zRelT4, labels: topoT4, classCount: RelationalTopology.allCases.count),
            archetypeT2: fitWithGroupCV(features: zRelT2, labels: archT2, classCount: RelationalArchetype.allCases.count),
            topologyT2: fitWithGroupCV(features: zRelT2, labels: topoT2, classCount: RelationalTopology.allCases.count)
        )
        model.relational.calibratedHeads = heads
    }
}
