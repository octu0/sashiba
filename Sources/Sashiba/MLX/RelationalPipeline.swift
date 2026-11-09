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

    /// Builds supervised pairs:
    ///   - `(X^S, X^C(T_2))` with `T_2` labels, `(X^S, X^C(T_4))` with `T_4` labels,
    ///   - counterfactual `(X^S, X^{C_harm})` with the harmonized target (`TRUE_SUPPORTER` / `DEEP_AFFECTION`, `Section 3`),
    ///   - mismatched `(X_i^S, X_j^C)` across different groups with `INDIFFERENT_TROLL` / `UNCONNECTED_STRANGER`.
    public static func buildPairs(
        cases: [RelationalDialogueCase],
        vocab: TextVocabulary,
        time: Int,
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
        let supportPool = cases.filter { $0.group.contains("cat01") || $0.group.contains("cat13") || $0.group.contains("cat33") }
        let lowCohPool = cases.filter { $0.group.contains("cat04") || $0.group.contains("cat09") || $0.group.contains("cat17") }

        for (i, rc) in cases.enumerated() {
            pairs.append(RelationalTrainingPair(
                senderTokens: host[i], receiverTokens: early[i],
                archetype: archetypeIndex(rc.expectedArchetypeT2), topology: topologyIndex(rc.expectedTopologyT2),
                isEarlyT2: true, groupKey: rc.group
            ))
            pairs.append(RelationalTrainingPair(
                senderTokens: host[i], receiverTokens: current[i],
                archetype: archetypeIndex(rc.expectedArchetype), topology: topologyIndex(rc.expectedTopology),
                isEarlyT2: false, groupKey: rc.group
            ))
            if includeHarmonized {
                if !supportPool.isEmpty && i % 8 == 0 {
                    let ref = supportPool[i % supportPool.count]
                    pairs.append(RelationalTrainingPair(
                        senderTokens: host[i], receiverTokens: harm[i],
                        archetype: archetypeIndex(ref.expectedArchetype), topology: topologyIndex(ref.expectedTopology),
                        isEarlyT2: false, groupKey: "HARM:" + rc.group
                    ))
                }
                cfs.append(RelationalCounterfactualPair(senderTokens: host[i], receiverTokens: current[i], harmonizedTokens: harm[i]))
            }
            if cases.count > 1 && i % 6 == 0 {
                for k in 0..<min(1, mismatchPerCase) {
                    var j = Int(rng.next() % UInt64(cases.count))
                    var guardCount = 0
                    while cases[j].group == rc.group && guardCount < 16 {
                        j = Int(rng.next() % UInt64(cases.count))
                        guardCount += 1
                    }
                    guard cases[j].group != rc.group else { continue }
                    let ref = lowCohPool.isEmpty ? cases[j] : lowCohPool[(i + k) % lowCohPool.count]
                    let mArch = archetypeIndex(ref.expectedArchetype)
                    let mTopo = topologyIndex(ref.expectedTopology)
                    pairs.append(RelationalTrainingPair(
                        senderTokens: host[i], receiverTokens: current[j],
                        archetype: mArch, topology: mTopo,
                        isEarlyT2: false, groupKey: "MISMATCH:" + rc.group + ":" + cases[j].group
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
        let lDim = model.latentDim
        var sFlat: [Int] = []
        var cFlat: [Int] = []
        sFlat.reserveCapacity(senders.count * time)
        cFlat.reserveCapacity(senders.count * time)
        for i in 0..<senders.count {
            sFlat.append(contentsOf: DualStreamRelationalEngine.padOrTruncate(senders[i], to: time))
            cFlat.append(contentsOf: DualStreamRelationalEngine.padOrTruncate(receivers[i], to: time))
        }
        let z = model.encodeRelationalGPU(senderTokens: sFlat, receiverTokens: cFlat, batch: senders.count, time: time, chunk: chunk).zRel
        return (0..<senders.count).map { Array(z[($0 * lDim)..<(($0 + 1) * lDim)]) }
    }

    /// Full relational stage: corpus-statistics gate from training texts -> End-to-End training -> head calibration
    /// on CPU `z_rel` of the training cases. Mutates `model` in place and returns the Audit 2 telemetry.
    @discardableResult
    public static func trainRelationalStage(
        model: MLXSWTHawkAutoencoder,
        trainCases: [RelationalDialogueCase],
        vocab: TextVocabulary,
        time: Int,
        epochs: Int,
        batchSize: Int = 64,
        learningRate: Float = 1e-3,
        lossWeights: RelationalLossWeights = RelationalLossWeights(),
        seed: UInt64 = 2026,
        log: @escaping (String) -> Void = { print($0) }
    ) -> RelationalAuditTelemetry {
        let gateTexts = trainCases.flatMap { [$0.hostText, $0.partnerEarlyText, $0.partnerCurrentText, $0.harmonizedPartnerText] }
        model.relational.alphaRel = vocab.computeRelationalTokenGate(trainTexts: gateTexts)

        let (pairs, cfs) = buildPairs(cases: trainCases, vocab: vocab, time: time, seed: seed)
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

        // Calibrate heads on the GPU-encoded z_rel (identical tensor used by all reports).
        let t2Pairs = pairs.filter { $0.isEarlyT2 && !$0.groupKey.hasPrefix("HARM:") }
        let t4Pairs = pairs.filter { !$0.isEarlyT2 && !$0.groupKey.hasPrefix("HARM:") }
        let z2 = encodeZRel(model: model, senders: t2Pairs.map(\.senderTokens), receivers: t2Pairs.map(\.receiverTokens), time: time)
        let z4 = encodeZRel(model: model, senders: t4Pairs.map(\.senderTokens), receivers: t4Pairs.map(\.receiverTokens), time: time)
        trainer.calibrateHeads(
            zRelT4: z4, archT4: t4Pairs.map(\.archetype), topoT4: t4Pairs.map(\.topology),
            zRelT2: z2, archT2: t2Pairs.map(\.archetype), topoT2: t2Pairs.map(\.topology)
        )
        return telemetry
    }
}

