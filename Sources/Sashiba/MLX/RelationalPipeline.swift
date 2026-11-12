import Foundation
import MLX

/// Phase 5.1: Shared relational pipeline used by `train` (full train split) and `relational_head` (per-fold retraining).
/// Category / label metadata is used ONLY to construct supervised training pairs; the encoder itself receives
/// nothing but `(senderTokenIDs, receiverTokenIDs)`.
public enum RelationalPipeline {
    public static func archetypeIndex(_ a: RelationalArchetype) -> Int {
        return RelationalArchetype.allCases.firstIndex(of: a) ?? 0
    }

    public static func topologyIndex(_ t: RelationalTopology) -> Int {
        return RelationalTopology.allCases.firstIndex(of: t) ?? 0
    }

    public static func encodeTokens(_ text: String, vocab: TextVocabulary, time: Int) -> [Int] {
        return DualStreamRelationalEngine.padOrTruncate(
            vocab.encode(text, maxLength: time, addEOS: false, padToLength: nil), to: time
        )
    }

    /// Builds supervised pairs and counterfactuals:
    ///   - `(X^S, X^C(T_2))` with `T_2` labels (`isEarlyT2: true`),
    ///   - `(X^S, X^C(T_4))` with `T_4` labels (`isEarlyT2: false`),
    ///   - counterfactual `(X^S, X^C)` vs `(X^S, X^{C_harm})` trained strictly via `L_cf` (zero unpermuted label injection),
    ///   - cross-group mismatched `(X_i^S, X_j^C)` pairs using low-coherence label targets without any category string checks.
    /// Computes unsupervised inverse-crowding weights over training dialogues using relational 4-gram TF-IDF similarity
    /// strictly on `cases` token sequences (zero label, category, or unseen access) so repetitive template paraphrases
    /// do not drown out lexically diverse training exemplars.
    public static func computeUniquenessWeights(
        cases: [RelationalDialogueCase],
        vocab: TextVocabulary,
        alphaRel: [Float]
    ) -> [Float] {
        let n = cases.count
        guard n > 8 else { return [Float](repeating: 1.0, count: n) }
        var docGrams: [Set<UInt64>] = []
        docGrams.reserveCapacity(n)
        var df: [UInt64: Int] = [:]
        let sep: UInt64 = 0x7FFF
        for rc in cases {
            var filtered: [UInt64] = []
            for seg in [rc.hostText, rc.partnerCurrentText] {
                let ids = vocab.encode(seg, addEOS: false)
                for id in ids where id >= TextVocabulary.specialCount && id < alphaRel.count && alphaRel[id] >= 0.35 {
                    filtered.append(UInt64(id))
                }
                filtered.append(sep)
            }
            var grams: Set<UInt64> = []
            if filtered.count >= 4 {
                for i in 0...(filtered.count - 4) {
                    let g = (filtered[i] << 48) | (filtered[i + 1] << 32) | (filtered[i + 2] << 16) | filtered[i + 3]
                    grams.insert(g)
                }
            }
            docGrams.append(grams)
            for g in grams { df[g, default: 0] += 1 }
        }
        let maxDF = max(2, n / 3)
        var idf: [UInt64: Float] = [:]
        for (g, c) in df where c >= 2 && c <= maxDF {
            idf[g] = logf(Float(n + 1) / Float(c + 1))
        }
        var vecs: [[(UInt64, Float)]] = []
        vecs.reserveCapacity(n)
        var inv: [UInt64: [(Int, Float)]] = [:]
        for (i, grams) in docGrams.enumerated() {
            var sq: Float = 0.0
            var raw: [(UInt64, Float)] = []
            for g in grams {
                if let w = idf[g] {
                    raw.append((g, w))
                    sq += w * w
                }
            }
            let invNorm = 1.0 / sqrtf(max(sq, 1e-12))
            let normed = raw.map { ($0.0, $0.1 * invNorm) }
            vecs.append(normed)
            for (g, v) in normed {
                inv[g, default: []].append((i, v))
            }
        }
        var rawW = [Float](repeating: 1.0, count: n)
        for i in 0..<n {
            var sims = [Float](repeating: 0.0, count: n)
            for (g, v) in vecs[i] {
                if let postings = inv[g] {
                    for (j, v2) in postings where j != i {
                        sims[j] += v * v2
                    }
                }
            }
            var crowd: Float = 0.0
            for s in sims where s > 0.075 {
                crowd += s
            }
            rawW[i] = 1.0 / (0.18 + crowd)
        }
        let meanW = rawW.reduce(0.0, +) / Float(n)
        return rawW.map { min(2.2, max(0.75, $0 / max(meanW, 1e-6))) }
    }

    public static func buildPairs(
        cases: [RelationalDialogueCase],
        vocab: TextVocabulary,
        time: Int,
        uniquenessWeights: [Float]? = nil,
        includeHarmonized: Bool = true,
        mismatchPerCase: Int = 1,
        seed: UInt64 = 77
    ) -> (pairs: [RelationalTrainingPair], counterfactuals: [RelationalCounterfactualPair]) {
        var rng = SplitMix64(seed: seed)
        var pairs: [RelationalTrainingPair] = []
        var cfs: [RelationalCounterfactualPair] = []
        let host = cases.map { encodeTokens($0.hostText, vocab: vocab, time: time) }
        let early = cases.map { encodeTokens($0.partnerEarlyText, vocab: vocab, time: time) }
        let current = cases.map { encodeTokens($0.partnerCurrentText, vocab: vocab, time: time) }
        let harm = cases.map { encodeTokens($0.harmonizedPartnerText, vocab: vocab, time: time) }
        let hasU = (uniquenessWeights?.count == cases.count)
        let lowCohPool = cases.filter {
            $0.expectedArchetype == .indifferentTroll && $0.expectedTopology == .unconnectedStranger
        }
        let fallbackMismatchArch = archetypeIndex(.indifferentTroll)
        let fallbackMismatchTopo = topologyIndex(.unconnectedStranger)

        for (i, rc) in cases.enumerated() {
            let uw = hasU ? uniquenessWeights![i] : 1.0
            pairs.append(RelationalTrainingPair(
                senderTokens: host[i],
                receiverTokens: early[i],
                archetype: archetypeIndex(rc.expectedArchetypeT2),
                topology: topologyIndex(rc.expectedTopologyT2),
                isEarlyT2: true,
                groupKey: rc.group,
                uniquenessWeight: uw
            ))
            pairs.append(RelationalTrainingPair(
                senderTokens: host[i],
                receiverTokens: current[i],
                archetype: archetypeIndex(rc.expectedArchetype),
                topology: topologyIndex(rc.expectedTopology),
                isEarlyT2: false,
                groupKey: rc.group,
                uniquenessWeight: uw
            ))
            if includeHarmonized {
                cfs.append(RelationalCounterfactualPair(
                    senderTokens: host[i],
                    receiverTokens: current[i],
                    harmonizedTokens: harm[i],
                    preserveLL: rc.expectedArchetype != .indifferentTroll,
                    separateHH: rc.expectedArchetype == .gachikoiRisk || rc.expectedArchetype == .hiddenAggressor
                ))
            }
            if cases.count > 1 && mismatchPerCase > 0 && i % 4 == 0 {
                for k in 0..<mismatchPerCase {
                    var j = Int(rng.next() % UInt64(cases.count))
                    var guardCount = 0
                    while cases[j].group == rc.group && guardCount < 32 {
                        j = Int(rng.next() % UInt64(cases.count))
                        guardCount += 1
                    }
                    guard cases[j].group != rc.group else { continue }
                    let mArch: Int
                    let mTopo: Int
                    if !lowCohPool.isEmpty {
                        let ref = lowCohPool[(i + k) % lowCohPool.count]
                        mArch = archetypeIndex(ref.expectedArchetype)
                        mTopo = topologyIndex(ref.expectedTopology)
                    } else {
                        mArch = fallbackMismatchArch
                        mTopo = fallbackMismatchTopo
                    }
                    pairs.append(RelationalTrainingPair(
                        senderTokens: host[i],
                        receiverTokens: current[j],
                        archetype: mArch,
                        topology: mTopo,
                        isEarlyT2: false,
                        groupKey: "MISMATCH:" + rc.group + ":" + cases[j].group
                    ))
                }
            }
        }
        return (pairs, cfs)
    }

    /// Computes `z_rel` with the MLX GPU Dual-Stream encoder (the single tensor shared by every report).
    public static func encodeZRel(
        model: MLXSWTHawkAutoencoder,
        senders: [[Int]],
        receivers: [[Int]],
        time: Int,
        chunk: Int = 64
    ) -> [[Float]] {
        guard !senders.isEmpty else { return [] }
        MLX.Memory.cacheLimit = 128 * 1024 * 1024
        let lDim = model.relational.latentDim
        var sFlat: [Int] = []
        var cFlat: [Int] = []
        sFlat.reserveCapacity(senders.count * time)
        cFlat.reserveCapacity(senders.count * time)
        for i in 0..<senders.count {
            sFlat.append(contentsOf: DualStreamRelationalEngine.padOrTruncate(senders[i], to: time))
            cFlat.append(contentsOf: DualStreamRelationalEngine.padOrTruncate(receivers[i], to: time))
        }
        let z = model.encodeRelationalGPU(senderTokens: sFlat, receiverTokens: cFlat, batch: senders.count, time: time, chunk: chunk).zRel
        MLX.Memory.clearCache()
        return (0..<senders.count).map { Array(z[($0 * lDim)..<(($0 + 1) * lDim)]) }
    }

    /// Full relational stage: corpus-statistics gate from training texts -> End-to-End training -> head calibration
    /// on GPU `z_rel` of the training cases. Mutates `model` in place and returns the Audit 2 telemetry.
    @discardableResult
    public static func trainRelationalStage(
        model: MLXSWTHawkAutoencoder,
        trainCases: [RelationalDialogueCase],
        vocab: TextVocabulary,
        time: Int,
        epochs: Int,
        batchSize: Int = 32,
        learningRate: Float = 1.5e-3,
        lossWeights: RelationalLossWeights = RelationalLossWeights(),
        seed: UInt64 = 2026,
        log: @escaping (String) -> Void = { print($0) }
    ) -> RelationalAuditTelemetry {
        MLX.Memory.cacheLimit = 128 * 1024 * 1024
        MLX.Memory.clearCache()
        let gateTexts = trainCases.flatMap { [$0.hostText, $0.partnerEarlyText, $0.partnerCurrentText, $0.harmonizedPartnerText] }
        model.relational.alphaRel = vocab.computeRelationalTokenGate(trainTexts: gateTexts)
        let uWeights = computeUniquenessWeights(cases: trainCases, vocab: vocab, alphaRel: model.relational.alphaRel)

        let (pairs, cfs) = buildPairs(cases: trainCases, vocab: vocab, time: time, uniquenessWeights: uWeights, seed: seed)
        let trainer = MLXRelationalTrainer(model: model, learningRate: learningRate, seed: seed)
        let telemetry = trainer.train(
            pairs: pairs,
            counterfactuals: cfs,
            time: time,
            epochs: epochs,
            batchSize: batchSize,
            lossWeights: lossWeights,
            learningRate: learningRate,
            log: log
        )

        // Calibrate heads strictly on the GPU-encoded z_rel of training scenario pairs (identical tensor used by all reports).
        let calHost = trainCases.map { encodeTokens($0.hostText, vocab: vocab, time: time) }
        let calEarly = trainCases.map { encodeTokens($0.partnerEarlyText, vocab: vocab, time: time) }
        let calCurrent = trainCases.map { encodeTokens($0.partnerCurrentText, vocab: vocab, time: time) }
        let z2 = encodeZRel(model: model, senders: calHost, receivers: calEarly, time: time)
        let z4 = encodeZRel(model: model, senders: calHost, receivers: calCurrent, time: time)
        trainer.calibrateHeads(
            zRelT4: z4,
            archT4: trainCases.map { archetypeIndex($0.expectedArchetype) },
            topoT4: trainCases.map { topologyIndex($0.expectedTopology) },
            zRelT2: z2,
            archT2: trainCases.map { archetypeIndex($0.expectedArchetypeT2) },
            topoT2: trainCases.map { topologyIndex($0.expectedTopologyT2) },
            groups: trainCases.map(\.group),
            uniquenessWeights: uWeights,
            steps: 350,
            l2: 2e-3
        )
        MLX.Memory.clearCache()
        return telemetry
    }
}
