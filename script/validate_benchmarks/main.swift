import Foundation
import MLX
import Sashiba

struct ValidateBenchmarksCLI: Sendable {
    var v2DataPath: String = ".tmp"
    var v1BackupDir: String = ".tmp/phase5_1_backup"
    var v1WeightsPath: String = ".tmp/phase5_1_backup/weights.safetensors"
    var v2NoMetricWeightsPath: String = ".tmp/train_v2/weights_nometric.safetensors"
    var v2WeightsPath: String = ".tmp/weights.safetensors"
    var layer1ConsensusPath: String = ".tmp/validation/layer1_blind_consensus_60.json"
    var layer1T2FullPath: String = ".tmp/validation/layer1_t2_consensus_60.json"
    var layer1T4FullPath: String = ".tmp/validation/layer1_t4_consensus_60.json"
    var layer3aDir: String = ".tmp/validation/benchmarks/layer3a_style_domain_holdout"
    var layer3bDir: String = ".tmp/validation/benchmarks/layer3b_independent_paraphrases"
    var layer3cDir: String = ".tmp/validation/benchmarks/layer3c_independent_inversions"
    var layer4Dir: String = ".tmp/validation/benchmarks/layer4_gold_outcome_benchmark"
    var cacheDir: String = ".tmp/validation/zrel_cache"
    var outputPath: String = ".tmp/validation/benchmark_metrics.json"
    var maxSeqLen: Int = 512
    var useGPU: Bool = false
    var disableCache: Bool = false

    static func parse(_ args: [String]) -> ValidateBenchmarksCLI {
        var cli = ValidateBenchmarksCLI()
        var i = 1
        while i < args.count {
            switch args[i] {
            case "-d", "--data":
                i += 1
                if i < args.count { cli.v2DataPath = args[i] }
            case "--v1-dir":
                i += 1
                if i < args.count { cli.v1BackupDir = args[i] }
            case "--v1-weights":
                i += 1
                if i < args.count { cli.v1WeightsPath = args[i] }
            case "--v2-nometric-weights":
                i += 1
                if i < args.count { cli.v2NoMetricWeightsPath = args[i] }
            case "--v2-weights":
                i += 1
                if i < args.count { cli.v2WeightsPath = args[i] }
            case "--gpu":
                cli.useGPU = true
            case "--cpu":
                cli.useGPU = false
            case "--no-cache":
                cli.disableCache = true
            case "-o", "--output":
                i += 1
                if i < args.count { cli.outputPath = args[i] }
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

// MARK: - External Metadata Schemas

struct Layer1T2OnlyItem: Codable, Sendable {
    let scenario_id: String
    let pentagon_t2_mean: [Float]
}

struct Layer1T4OnlyItem: Codable, Sendable {
    let scenario_id: String
    let pentagon_t4_mean: [Float]
}

struct Layer1ConsensusItem: Codable, Sendable {
    let scenario_id: String
    let split: String
    let consensus_arch_t2: RelationalArchetype
    let consensus_topo_t2: RelationalTopology
    let consensus_arch_t4: RelationalArchetype
    let consensus_topo_t4: RelationalTopology
    let pentagon_t2_mean: [Float]
    let pentagon_t4_mean: [Float]
}

struct Layer4OutcomeRecord: Codable, Sendable {
    let scenario_id: String
    let retained_30d: Bool
    let retained_90d: Bool
    let moderation_or_block: Bool
    let response_interval_drift: Float
}

struct Layer4MetadataItem: Codable, Sendable {
    let scenario_id: String
    let consensus_archetype_t2: RelationalArchetype
    let consensus_topology_t2: RelationalTopology
    let consensus_archetype_t4: RelationalArchetype
    let consensus_topology_t4: RelationalTopology
    let pentagon_t2_mean: [Float]
    let pentagon_t4_mean: [Float]
    let outcomes: Layer4OutcomeRecord
}

struct Layer4MetadataFile: Codable {
    let items: [Layer4MetadataItem]
}

// MARK: - Output Metric Structs

struct FourTaskMetrics: Codable {
    let archetype_t4: Float
    let topology_t4: Float
    let archetype_t2: Float
    let topology_t2: Float
    let mean: Float

    init(archT4: Float, topoT4: Float, archT2: Float, topoT2: Float) {
        self.archetype_t4 = archT4
        self.topology_t4 = topoT4
        self.archetype_t2 = archT2
        self.topology_t2 = topoT2
        self.mean = 0.25 * (archT4 + topoT4 + archT2 + topoT2)
    }
}

struct UnkStats: Codable {
    let unk_tokens: Int
    let total_tokens: Int
    let unk_token_rate: Float
    let cases_with_unk: Int
    let total_cases: Int
}

struct SuiteModelEval: Codable {
    let accuracy: FourTaskMetrics
    let unk_stats: UnkStats?
    let drop_pt_from_unseen_test: FourTaskMetrics?
    let paired_inversion_accuracy: PairedInversionMetrics?
}

struct PairedInversionMetrics: Codable {
    let total_pairs: Int
    let archetype_t4: Float
    let topology_t4: Float
    let archetype_t2: Float
    let topology_t2: Float
    let mean_paired_accuracy: Float
    let joint_t4_both_correct: Float
}

struct CausalLeakageAuditOutput: Codable {
    let dataset_cases: Int
    let non_causal_t2: FourTaskMetrics
    let strict_causal_t2: FourTaskMetrics
    let archetype_t2_leakage_delta_pt: Float
    let topology_t2_leakage_delta_pt: Float
}

struct SubbandGeometryModelEval: Codable {
    let linear_probe_3d_train_accuracy: FourTaskMetrics
    let linear_probe_3d_unseen_accuracy: FourTaskMetrics
    let mutual_information_bits: FourTaskMetrics
    let label_entropy_bits: FourTaskMetrics
}

struct PentagonAxisCorrelation: Codable {
    let n_cases: Int
    let contextDepth: Float
    let empathicResonance: Float
    let communicativeHarmony: Float
    let boundaryStability: Float
    let longTermViability: Float
    let mean_pearson_r: Float
}

struct PentagonModelValidity: Codable {
    let layer1_blind_t2: PentagonAxisCorrelation
    let layer1_blind_t4: PentagonAxisCorrelation
    let layer4_gold_t2: PentagonAxisCorrelation
    let layer4_gold_t4: PentagonAxisCorrelation
    let combined_t2: PentagonAxisCorrelation
    let combined_t4: PentagonAxisCorrelation
}

struct Layer2ValidationOutput: Codable {
    let subband_geometry_ablation: [String: SubbandGeometryModelEval]
    let pentagon_external_validity: [String: PentagonModelValidity]
}

struct Layer3SuiteOutput: Codable {
    let n_cases: Int
    let models: [String: SuiteModelEval]
}

struct BinaryOutcomeMetrics: Codable {
    let roc_auc: Float
    let accuracy: Float
}

struct ContinuousOutcomeMetrics: Codable {
    let pearson_r: Float
    let mae: Float
}

struct Layer4Output: Codable {
    let n_cases: Int
    let zero_shot_consensus_classification: [String: SuiteModelEval]
    let retained_30d_5fold: [String: BinaryOutcomeMetrics]
    let retained_90d_5fold: [String: BinaryOutcomeMetrics]
    let moderation_or_block_5fold: [String: BinaryOutcomeMetrics]
    let response_interval_drift_5fold: [String: ContinuousOutcomeMetrics]
}

struct CoarseEarlyT2Metrics: Codable {
    let early_stance_4way: Float
    let early_distance_3way: Float
    let mean: Float
}

struct BenchmarkMetricsReport: Codable {
    let generated_at: String
    let execution_mode: String
    let wall_clock_seconds: Double
    let v1_causal_t2_leakage_audit: CausalLeakageAuditOutput
    let in_distribution_unseen_test: Layer3SuiteOutput
    let p0_coarse_t2_benchmarks: [String: [String: CoarseEarlyT2Metrics]]
    let layer2_validation: Layer2ValidationOutput
    let layer3_ood_benchmarks: [String: Layer3SuiteOutput]
    let layer4_gold_outcome_benchmark: Layer4Output
}

// MARK: - Encoding & Evaluation Helpers

struct EncodedSuite {
    let zT2: [[Float]]
    let zT4: [[Float]]
    let hybridT2: [[Float]]
    let hybridT4: [[Float]]
    let predArchT4: [Int]
    let predTopoT4: [Int]
    let predArchT2: [Int]
    let predTopoT2: [Int]
    let predStanceT2: [Int]
    let predDistT2: [Int]
    let subbandT2: [(sLL: Float, sHH: Float, rHH: Float)]
    let subbandT4: [(sLL: Float, sHH: Float, rHH: Float)]
    let pentagonT2: [[Float]]
    let pentagonT4: [[Float]]
    let unk: UnkStats
}

func saveZRelCache(path: String, z2: [[Float]], z4: [[Float]]) {
    guard !z2.isEmpty, z2.count == z4.count else { return }
    let n = Int32(z2.count)
    let d = Int32(z2[0].count)
    var data = Data()
    withUnsafeBytes(of: n) { data.append(contentsOf: $0) }
    withUnsafeBytes(of: d) { data.append(contentsOf: $0) }
    let flat2 = z2.flatMap { $0 }
    let flat4 = z4.flatMap { $0 }
    flat2.withUnsafeBufferPointer { ptr in
        data.append(UnsafeBufferPointer(start: UnsafeRawPointer(ptr.baseAddress!)?.assumingMemoryBound(to: UInt8.self), count: flat2.count * 4))
    }
    flat4.withUnsafeBufferPointer { ptr in
        data.append(UnsafeBufferPointer(start: UnsafeRawPointer(ptr.baseAddress!)?.assumingMemoryBound(to: UInt8.self), count: flat4.count * 4))
    }
    try? FileManager.default.createDirectory(at: URL(fileURLWithPath: path).deletingLastPathComponent(), withIntermediateDirectories: true)
    try? data.write(to: URL(fileURLWithPath: path))
}

func loadZRelCache(path: String, expectedCount: Int, expectedDim: Int) -> ([[Float]], [[Float]])? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
          data.count == 8 + expectedCount * expectedDim * 8 else { return nil }
    let n = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: Int32.self) }
    let d = data.withUnsafeBytes { $0.load(fromByteOffset: 4, as: Int32.self) }
    guard Int(n) == expectedCount && Int(d) == expectedDim else { return nil }
    let totalFloats = expectedCount * expectedDim
    var flat2 = [Float](repeating: 0.0, count: totalFloats)
    var flat4 = [Float](repeating: 0.0, count: totalFloats)
    _ = flat2.withUnsafeMutableBytes { dst in
        data.copyBytes(to: dst, from: 8..<(8 + totalFloats * 4))
    }
    _ = flat4.withUnsafeMutableBytes { dst in
        data.copyBytes(to: dst, from: (8 + totalFloats * 4)..<(8 + totalFloats * 8))
    }
    let z2 = (0..<expectedCount).map { Array(flat2[($0 * expectedDim)..<(($0 + 1) * expectedDim)]) }
    let z4 = (0..<expectedCount).map { Array(flat4[($0 * expectedDim)..<(($0 + 1) * expectedDim)]) }
    return (z2, z4)
}

func computeUnkStats(cases: [RelationalDialogueCase], vocab: TextVocabulary) -> UnkStats {
    var unkTok = 0
    var totTok = 0
    var casesUnk = 0
    for rc in cases {
        var hasUnk = false
        for seg in [rc.hostEarlyText, rc.hostText, rc.partnerEarlyText, rc.partnerCurrentText] {
            let ids = vocab.encode(seg, addEOS: false)
            totTok += ids.count
            let u = ids.filter { $0 == TextVocabulary.unkId }.count
            if u > 0 { hasUnk = true }
            unkTok += u
        }
        if hasUnk { casesUnk += 1 }
    }
    return UnkStats(
        unk_tokens: unkTok,
        total_tokens: totTok,
        unk_token_rate: Float(unkTok) / Float(max(1, totTok)),
        cases_with_unk: casesUnk,
        total_cases: cases.count
    )
}

func pentagonArray(_ p: RelationalPentagonScores) -> [Float] {
    return [p.contextDepth, p.empathicResonance, p.communicativeHarmony, p.boundaryStability, p.longTermViability]
}

func predictCoarseFromProbs(archProbs: [Float], topoProbs: [Float]) -> (stance: Int, dist: Int) {
    let archCases = RelationalArchetype.allCases
    let topoCases = RelationalTopology.allCases
    let stanceCases = EarlyRelationalStance.allCases
    let distCases = EarlyRelationalDistance.allCases
    var sp = [Float](repeating: 0.0, count: stanceCases.count)
    for (i, a) in archCases.enumerated() where i < archProbs.count {
        if let idx = stanceCases.firstIndex(of: EarlyRelationalStance.from(archetype: a)) {
            sp[idx] += archProbs[i]
        }
    }
    var dp = [Float](repeating: 0.0, count: distCases.count)
    for (i, t) in topoCases.enumerated() where i < topoProbs.count {
        if let idx = distCases.firstIndex(of: EarlyRelationalDistance.from(topology: t)) {
            dp[idx] += topoProbs[i]
        }
    }
    var bestS = 0
    for i in 1..<sp.count where sp[i] > sp[bestS] { bestS = i }
    var bestD = 0
    for i in 1..<dp.count where dp[i] > dp[bestD] { bestD = i }
    return (bestS, bestD)
}

func computeCoarseEarlyT2(cases: [RelationalDialogueCase], predStance: [Int], predDist: [Int]) -> CoarseEarlyT2Metrics {
    let stanceCases = EarlyRelationalStance.allCases
    let distCases = EarlyRelationalDistance.allCases
    let yS = cases.map { stanceCases.firstIndex(of: EarlyRelationalStance.from(archetype: $0.expectedArchetypeT2)) ?? 0 }
    let yD = cases.map { distCases.firstIndex(of: EarlyRelationalDistance.from(topology: $0.expectedTopologyT2)) ?? 0 }
    let sAcc = accuracyRate(predStance, yS)
    let dAcc = accuracyRate(predDist, yD)
    return CoarseEarlyT2Metrics(early_stance_4way: sAcc, early_distance_3way: dAcc, mean: 0.5 * (sAcc + dAcc))
}

func encodeSuiteWithModel(
    suiteName: String,
    modelName: String,
    weightsPath: String,
    cases: [RelationalDialogueCase],
    model: MLXSWTHawkAutoencoder?,
    weights: SWTHawkWeights,
    vocab: TextVocabulary,
    extractor: RelationalHybridFeatureExtractor,
    cacheDir: String,
    disableCache: Bool,
    didMissCache: inout Bool,
    maxSeqLen: Int,
    useCausalT2: Bool = true
) -> EncodedSuite {
    let seqLen = cases
        .flatMap { [$0.hostEarlyText, $0.hostText, $0.partnerEarlyText, $0.partnerCurrentText] }
        .map { min(maxSeqLen, vocab.encode($0, maxLength: maxSeqLen, addEOS: false).count) }
        .reduce(max(4, DualStreamRelationalWeights.canonicalSeqLen), max)

    let sT2 = cases.map {
        RelationalPipeline.encodeTokens(useCausalT2 ? $0.hostEarlyText : $0.hostText, vocab: vocab, time: seqLen)
    }
    let sT4 = cases.map { RelationalPipeline.encodeTokens($0.hostText, vocab: vocab, time: seqLen) }
    let rT2 = cases.map { RelationalPipeline.encodeTokens($0.partnerEarlyText, vocab: vocab, time: seqLen) }
    let rT4 = cases.map { RelationalPipeline.encodeTokens($0.partnerCurrentText, vocab: vocab, time: seqLen) }

    let attrs = try? FileManager.default.attributesOfItem(atPath: weightsPath)
    let wSize = (attrs?[.size] as? NSNumber)?.int64Value ?? 0
    let wMtime = Int64(((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0).rounded())
    let cachePath = (cacheDir as NSString).appendingPathComponent(
        "\(modelName)_\(suiteName)_\(useCausalT2 ? "causal" : "noncausal")_n\(cases.count)_t\(seqLen)_s\(wSize)_m\(wMtime).bin"
    )

    let z2: [[Float]]
    let z4: [[Float]]
    if !disableCache, let cached = loadZRelCache(path: cachePath, expectedCount: cases.count, expectedDim: weights.relational.latentDim) {
        z2 = cached.0
        z4 = cached.1
    } else {
        didMissCache = true
        if let m = model {
            z2 = RelationalPipeline.encodeZRel(model: m, senders: sT2, receivers: rT2, time: seqLen, chunk: 64)
            z4 = RelationalPipeline.encodeZRel(model: m, senders: sT4, receivers: rT4, time: seqLen, chunk: 64)
        } else {
            z2 = DualStreamRelationalEngine.encodeZRelCPUParallel(senders: sT2, receivers: rT2, time: seqLen, weights: weights, chunkSize: 16)
            z4 = DualStreamRelationalEngine.encodeZRelCPUParallel(senders: sT4, receivers: rT4, time: seqLen, weights: weights, chunkSize: 16)
        }
        if !disableCache {
            saveZRelCache(path: cachePath, z2: z2, z4: z4)
        }
    }

    let stanceCases = EarlyRelationalStance.allCases
    let distCases = EarlyRelationalDistance.allCases
    var hyb2: [[Float]] = []
    var hyb4: [[Float]] = []
    var pArchT4: [Int] = []
    var pTopoT4: [Int] = []
    var pArchT2: [Int] = []
    var pTopoT2: [Int] = []
    var pStanceT2: [Int] = []
    var pDistT2: [Int] = []
    var sub2: [(sLL: Float, sHH: Float, rHH: Float)] = []
    var sub4: [(sLL: Float, sHH: Float, rHH: Float)] = []
    var pent2: [[Float]] = []
    var pent4: [[Float]] = []

    for i in 0..<cases.count {
        hyb2.append(extractor.extract(senderTokens: sT2[i], receiverTokens: rT2[i], zRel: z2[i], weights: weights))
        hyb4.append(extractor.extract(senderTokens: sT4[i], receiverTokens: rT4[i], zRel: z4[i], weights: weights))

        let c2 = DualStreamRelationalEngine.classifyFromZRel(zRel: z2[i], isEarlyT2: true, weights: weights.relational)
        let c4 = DualStreamRelationalEngine.classifyFromZRel(zRel: z4[i], isEarlyT2: false, weights: weights.relational)
        let cc2 = DualStreamRelationalEngine.classifyEarlyCoarseFromZRel(zRel: z2[i], weights: weights.relational)
        pArchT2.append(RelationalPipeline.archetypeIndex(c2.archetype))
        pTopoT2.append(RelationalPipeline.topologyIndex(c2.topology))
        pArchT4.append(RelationalPipeline.archetypeIndex(c4.archetype))
        pTopoT4.append(RelationalPipeline.topologyIndex(c4.topology))
        pStanceT2.append(stanceCases.firstIndex(of: cc2.stance) ?? 0)
        pDistT2.append(distCases.firstIndex(of: cc2.distance) ?? 0)

        let g2 = DualStreamRelationalEngine.computePentagonFromZRel(zRel: z2[i], zRelEarly: nil)
        let g4 = DualStreamRelationalEngine.computePentagonFromZRel(zRel: z4[i], zRelEarly: z2[i])
        sub2.append((g2.sLL, g2.sHH, g2.rHH))
        sub4.append((g4.sLL, g4.sHH, g4.rHH))
        pent2.append(pentagonArray(g2.pentagon))
        pent4.append(pentagonArray(g4.pentagon))
    }

    return EncodedSuite(
        zT2: z2,
        zT4: z4,
        hybridT2: hyb2,
        hybridT4: hyb4,
        predArchT4: pArchT4,
        predTopoT4: pTopoT4,
        predArchT2: pArchT2,
        predTopoT2: pTopoT2,
        predStanceT2: pStanceT2,
        predDistT2: pDistT2,
        subbandT2: sub2,
        subbandT4: sub4,
        pentagonT2: pent2,
        pentagonT4: pent4,
        unk: computeUnkStats(cases: cases, vocab: vocab)
    )
}

func accuracyRate(_ preds: [Int], _ labels: [Int]) -> Float {
    let n = min(preds.count, labels.count)
    guard n > 0 else { return 0.0 }
    let ok = zip(preds, labels).filter { $0 == $1 }.count
    return Float(ok) / Float(n)
}

func computeFourTaskAccuracy(
    cases: [RelationalDialogueCase],
    predArchT4: [Int],
    predTopoT4: [Int],
    predArchT2: [Int],
    predTopoT2: [Int]
) -> FourTaskMetrics {
    let yArchT4 = cases.map { RelationalPipeline.archetypeIndex($0.expectedArchetype) }
    let yTopoT4 = cases.map { RelationalPipeline.topologyIndex($0.expectedTopology) }
    let yArchT2 = cases.map { RelationalPipeline.archetypeIndex($0.expectedArchetypeT2) }
    let yTopoT2 = cases.map { RelationalPipeline.topologyIndex($0.expectedTopologyT2) }
    return FourTaskMetrics(
        archT4: accuracyRate(predArchT4, yArchT4),
        topoT4: accuracyRate(predTopoT4, yTopoT4),
        archT2: accuracyRate(predArchT2, yArchT2),
        topoT2: accuracyRate(predTopoT2, yTopoT2)
    )
}

func computeDropPt(unseen: FourTaskMetrics, target: FourTaskMetrics) -> FourTaskMetrics {
    return FourTaskMetrics(
        archT4: (unseen.archetype_t4 - target.archetype_t4) * 100.0,
        topoT4: (unseen.topology_t4 - target.topology_t4) * 100.0,
        archT2: (unseen.archetype_t2 - target.archetype_t2) * 100.0,
        topoT2: (unseen.topology_t2 - target.topology_t2) * 100.0
    )
}

func computePairedInversion(
    cases: [RelationalDialogueCase],
    predArchT4: [Int],
    predTopoT4: [Int],
    predArchT2: [Int],
    predTopoT2: [Int]
) -> PairedInversionMetrics {
    var byPairKey: [String: [Int]] = [:]
    for (idx, rc) in cases.enumerated() {
        let key: String
        if rc.id.hasSuffix("_A") || rc.id.hasSuffix("_B") {
            key = String(rc.id.dropLast(2))
        } else {
            key = rc.id
        }
        byPairKey[key, default: []].append(idx)
    }
    let validPairs = byPairKey.keys.sorted().compactMap { k -> (Int, Int)? in
        guard let arr = byPairKey[k], arr.count == 2 else { return nil }
        return (arr[0], arr[1])
    }
    let pCount = max(1, validPairs.count)
    var okAT4 = 0
    var okTT4 = 0
    var okAT2 = 0
    var okTT2 = 0
    var okJointT4 = 0
    for (i, j) in validPairs {
        let a4i = predArchT4[i] == RelationalPipeline.archetypeIndex(cases[i].expectedArchetype)
        let a4j = predArchT4[j] == RelationalPipeline.archetypeIndex(cases[j].expectedArchetype)
        let t4i = predTopoT4[i] == RelationalPipeline.topologyIndex(cases[i].expectedTopology)
        let t4j = predTopoT4[j] == RelationalPipeline.topologyIndex(cases[j].expectedTopology)
        let a2i = predArchT2[i] == RelationalPipeline.archetypeIndex(cases[i].expectedArchetypeT2)
        let a2j = predArchT2[j] == RelationalPipeline.archetypeIndex(cases[j].expectedArchetypeT2)
        let t2i = predTopoT2[i] == RelationalPipeline.topologyIndex(cases[i].expectedTopologyT2)
        let t2j = predTopoT2[j] == RelationalPipeline.topologyIndex(cases[j].expectedTopologyT2)

        if a4i && a4j { okAT4 += 1 }
        if t4i && t4j { okTT4 += 1 }
        if a2i && a2j { okAT2 += 1 }
        if t2i && t2j { okTT2 += 1 }
        if a4i && a4j && t4i && t4j { okJointT4 += 1 }
    }
    let invP = 1.0 / Float(pCount)
    let rAT4 = Float(okAT4) * invP
    let rTT4 = Float(okTT4) * invP
    let rAT2 = Float(okAT2) * invP
    let rTT2 = Float(okTT2) * invP
    return PairedInversionMetrics(
        total_pairs: validPairs.count,
        archetype_t4: rAT4,
        topology_t4: rTT4,
        archetype_t2: rAT2,
        topology_t2: rTT2,
        mean_paired_accuracy: 0.25 * (rAT4 + rTT4 + rAT2 + rTT2),
        joint_t4_both_correct: Float(okJointT4) * invP
    )
}

func correlatePentagon(modelPreds: [[Float]], humanMeans: [[Float]]) -> PentagonAxisCorrelation {
    let n = min(modelPreds.count, humanMeans.count)
    guard n >= 2 else {
        return PentagonAxisCorrelation(
            n_cases: n, contextDepth: 0, empathicResonance: 0, communicativeHarmony: 0, boundaryStability: 0, longTermViability: 0, mean_pearson_r: 0
        )
    }
    var rs = [Float](repeating: 0.0, count: 5)
    for ax in 0..<5 {
        let xs = (0..<n).map { modelPreds[$0][ax] }
        let ys = (0..<n).map { humanMeans[$0][ax] }
        rs[ax] = BenchmarkMath.pearsonR(xs, ys)
    }
    let meanR = rs.reduce(0.0, +) / 5.0
    return PentagonAxisCorrelation(
        n_cases: n,
        contextDepth: rs[0],
        empathicResonance: rs[1],
        communicativeHarmony: rs[2],
        boundaryStability: rs[3],
        longTermViability: rs[4],
        mean_pearson_r: meanR
    )
}

// MARK: - Main Execution

let startWallTime = Date()
let cli = ValidateBenchmarksCLI.parse(CommandLine.arguments)
if cli.useGPU {
    MLX.Memory.cacheLimit = 128 * 1024 * 1024
    MLX.Memory.clearCache()
}
var didMissAnyCache = false
let pct = { (v: Float) in String(format: "%6.2f%%", v * 100.0) }

print("==================================================================")
print("Sashiba Comprehensive Multi-Layer Benchmark & External Validation")
print("Execution mode: \(cli.useGPU ? "GPU (MLX)" : "Pure Multi-Core CPU (0 GPU VRAM)") | Cache: \(cli.disableCache ? "OFF" : cli.cacheDir)")
print("==================================================================")

// Load datasets
let v1Config = DatasetConfiguration.load(from: cli.v1BackupDir)
let v1UnseenCases = v1Config.relationalDialogueCases.filter { $0.isZeroShotUnseen }

let v2Config = DatasetConfiguration.load(from: cli.v2DataPath)
let v2TrainCases = v2Config.relationalDialogueCases.filter { !$0.isZeroShotUnseen }
let v2UnseenCases = v2Config.relationalDialogueCases.filter { $0.isZeroShotUnseen }
let v2CaseByID = Dictionary(uniqueKeysWithValues: v2Config.relationalDialogueCases.map { ($0.id, $0) })

let l3aCases = DatasetConfiguration.load(from: cli.layer3aDir).relationalDialogueCases
let l3bCases = DatasetConfiguration.load(from: cli.layer3bDir).relationalDialogueCases
let l3cCases = DatasetConfiguration.load(from: cli.layer3cDir).relationalDialogueCases
let l4CasesRaw = DatasetConfiguration.load(from: cli.layer4Dir).relationalDialogueCases

let l1Items = try JSONDecoder().decode(
    [Layer1ConsensusItem].self,
    from: Data(contentsOf: URL(fileURLWithPath: cli.layer1ConsensusPath))
)
let l1T2MatchedCases: [(pent: [Float], rc: RelationalDialogueCase)]
if FileManager.default.fileExists(atPath: cli.layer1T2FullPath),
   let items = try? JSONDecoder().decode([Layer1T2OnlyItem].self, from: Data(contentsOf: URL(fileURLWithPath: cli.layer1T2FullPath))) {
    l1T2MatchedCases = items.compactMap { item in
        guard let rc = v2CaseByID[item.scenario_id] else { return nil }
        return (item.pentagon_t2_mean, rc)
    }
} else {
    l1T2MatchedCases = l1Items.compactMap { item in
        guard let rc = v2CaseByID[item.scenario_id] else { return nil }
        return (item.pentagon_t2_mean, rc)
    }
}

let l1T4MatchedCases: [(pent: [Float], rc: RelationalDialogueCase)]
if FileManager.default.fileExists(atPath: cli.layer1T4FullPath),
   let items = try? JSONDecoder().decode([Layer1T4OnlyItem].self, from: Data(contentsOf: URL(fileURLWithPath: cli.layer1T4FullPath))) {
    l1T4MatchedCases = items.compactMap { item in
        guard let rc = v2CaseByID[item.scenario_id] else { return nil }
        return (item.pentagon_t4_mean, rc)
    }
} else {
    l1T4MatchedCases = l1Items.compactMap { item in
        guard let rc = v2CaseByID[item.scenario_id] else { return nil }
        return (item.pentagon_t4_mean, rc)
    }
}

let l4MetaPath = (cli.layer4Dir as NSString).appendingPathComponent("metadata.json")
let l4Meta = try JSONDecoder().decode(
    Layer4MetadataFile.self,
    from: Data(contentsOf: URL(fileURLWithPath: l4MetaPath))
)
let l4RawByID = Dictionary(uniqueKeysWithValues: l4CasesRaw.map { ($0.id, $0) })
let l4Aligned: [(rc: RelationalDialogueCase, meta: Layer4MetadataItem)] = l4Meta.items.compactMap { m in
    guard let raw = l4RawByID[m.scenario_id] else { return nil }
    let rc = RelationalDialogueCase(
        id: raw.id,
        group: raw.group,
        personaStyle: raw.personaStyle,
        topicDomain: raw.topicDomain,
        title: raw.title,
        expectedTopologyT2: m.consensus_topology_t2,
        expectedArchetypeT2: m.consensus_archetype_t2,
        expectedTopology: m.consensus_topology_t4,
        expectedArchetype: m.consensus_archetype_t4,
        hostEarlyText: raw.hostEarlyText,
        hostText: raw.hostText,
        partnerEarlyText: raw.partnerEarlyText,
        partnerCurrentText: raw.partnerCurrentText,
        harmonizedPartnerText: raw.harmonizedPartnerText,
        insightLevel0: raw.insightLevel0,
        insightLevel1: raw.insightLevel1,
        insightLevel2: raw.insightLevel2,
        isZeroShotUnseen: true
    )
    return (rc, m)
}
let l4Cases = l4Aligned.map(\.rc)
let l4MetaOrdered = l4Aligned.map(\.meta)

print("Loaded splits: v2 train=\(v2TrainCases.count), unseen=\(v2UnseenCases.count), L1_T2=\(l1T2MatchedCases.count), L1_T4=\(l1T4MatchedCases.count), L3a=\(l3aCases.count), L3b=\(l3bCases.count), L3c=\(l3cCases.count), L4=\(l4Cases.count)")

struct AllSuitesForModel {
    let train: EncodedSuite
    let unseen: EncodedSuite
    let l1T2: EncodedSuite
    let l1T4: EncodedSuite
    let l3a: EncodedSuite
    let l3b: EncodedSuite
    let l3c: EncodedSuite
    let l4: EncodedSuite
}

func evaluateNeuralModel(name: String, path: String) throws -> (suites: AllSuitesForModel, v1NonCausalAudit: EncodedSuite?, v1CausalAudit: EncodedSuite?, cacheMiss: Bool) {
    let t0 = Date()
    print("[Model: \(name)] Loading \(path)...")
    let w = try SWTHawkWeights.load(from: URL(fileURLWithPath: path))
    let vocab = TextVocabulary.deserialize(w.serializedVocabulary)
    let extractor = RelationalHybridFeatureExtractor(vocabulary: vocab)
    let model: MLXSWTHawkAutoencoder? = cli.useGPU ? MLXSWTHawkAutoencoder(weights: w) : nil
    var localMiss = false

    func enc(_ sName: String, _ cs: [RelationalDialogueCase], causal: Bool = true) -> EncodedSuite {
        return encodeSuiteWithModel(
            suiteName: sName,
            modelName: name,
            weightsPath: path,
            cases: cs,
            model: model,
            weights: w,
            vocab: vocab,
            extractor: extractor,
            cacheDir: cli.cacheDir,
            disableCache: cli.disableCache,
            didMissCache: &localMiss,
            maxSeqLen: cli.maxSeqLen,
            useCausalT2: causal
        )
    }

    var v1NonCausal: EncodedSuite? = nil
    var v1Causal: EncodedSuite? = nil
    if name == "v1" && !v1UnseenCases.isEmpty {
        v1NonCausal = enc("v1_unseen", v1UnseenCases, causal: false)
        v1Causal = enc("v1_unseen", v1UnseenCases, causal: true)
    }

    let tr = enc("train", v2TrainCases)
    let un = enc("unseen", v2UnseenCases)
    let l1T2 = enc("l1_t2", l1T2MatchedCases.map(\.rc))
    let l1T4 = enc("l1_t4", l1T4MatchedCases.map(\.rc))
    let s3a = enc("l3a", l3aCases)
    let s3b = enc("l3b", l3bCases)
    let s3c = enc("l3c", l3cCases)
    let s4 = enc("l4", l4Cases)
    if cli.useGPU {
        MLX.Memory.clearCache()
    }
    print(String(format: "[Model: %@] Encoded all suites in %.2fs", name as NSString, Date().timeIntervalSince(t0)))
    return (AllSuitesForModel(train: tr, unseen: un, l1T2: l1T2, l1T4: l1T4, l3a: s3a, l3b: s3b, l3c: s3c, l4: s4), v1NonCausal, v1Causal, localMiss)
}

let (v1Eval, v1NonCausalOpt, v1CausalOpt, m1) = try evaluateNeuralModel(name: "v1", path: cli.v1WeightsPath)
let (v2NoMetricEval, _, _, m2) = try evaluateNeuralModel(name: "v2_nometric", path: cli.v2NoMetricWeightsPath)
let (v2Eval, _, _, m3) = try evaluateNeuralModel(name: "v2", path: cli.v2WeightsPath)
didMissAnyCache = m1 || m2 || m3

// 1. Causal T2 leakage audit on v1
let v1NonCausalAcc = computeFourTaskAccuracy(
    cases: v1UnseenCases,
    predArchT4: v1NonCausalOpt!.predArchT4,
    predTopoT4: v1NonCausalOpt!.predTopoT4,
    predArchT2: v1NonCausalOpt!.predArchT2,
    predTopoT2: v1NonCausalOpt!.predTopoT2
)
let v1CausalAcc = computeFourTaskAccuracy(
    cases: v1UnseenCases,
    predArchT4: v1CausalOpt!.predArchT4,
    predTopoT4: v1CausalOpt!.predTopoT4,
    predArchT2: v1CausalOpt!.predArchT2,
    predTopoT2: v1CausalOpt!.predTopoT2
)
let causalAudit = CausalLeakageAuditOutput(
    dataset_cases: v1UnseenCases.count,
    non_causal_t2: v1NonCausalAcc,
    strict_causal_t2: v1CausalAcc,
    archetype_t2_leakage_delta_pt: (v1NonCausalAcc.archetype_t2 - v1CausalAcc.archetype_t2) * 100.0,
    topology_t2_leakage_delta_pt: (v1NonCausalAcc.topology_t2 - v1CausalAcc.topology_t2) * 100.0
)
print(String(
    format: "[Step 4.1 Causal Audit v1] arch_T2: non-causal %.2f%% -> causal %.2f%% (Δ %+.2f pt) | topo_T2: non-causal %.2f%% -> causal %.2f%% (Δ %+.2f pt)",
    v1NonCausalAcc.archetype_t2 * 100, v1CausalAcc.archetype_t2 * 100, causalAudit.archetype_t2_leakage_delta_pt,
    v1NonCausalAcc.topology_t2 * 100, v1CausalAcc.topology_t2 * 100, causalAudit.topology_t2_leakage_delta_pt
))

// Train non-neural baselines (char_ngram 2048x2 & surface_stats 12D) strictly on v2TrainCases (strict causal T2!)
let ngram = CharNGramFeaturizer(dim: 2048)
func ngramFeatures(_ cases: [RelationalDialogueCase], early: Bool) -> [[Float]] {
    return cases.map { rc in
        let s = early ? rc.hostEarlyText : rc.hostText
        let r = early ? rc.partnerEarlyText : rc.partnerCurrentText
        return ngram.features(s) + ngram.features(r)
    }
}
func surfaceFeatures(_ cases: [RelationalDialogueCase], early: Bool) -> [[Float]] {
    return cases.map { rc in
        let s = early ? rc.hostEarlyText : rc.hostText
        let r = early ? rc.partnerEarlyText : rc.partnerCurrentText
        return SurfaceStatsFeaturizer.features(sender: s, receiver: r)
    }
}

struct LinearBaselineHeads {
    let archT4: SoftmaxClassifierHead
    let topoT4: SoftmaxClassifierHead
    let archT2: SoftmaxClassifierHead
    let topoT2: SoftmaxClassifierHead

    func predictSuite(_ cases: [RelationalDialogueCase], featFn: ([RelationalDialogueCase], Bool) -> [[Float]]) -> (a4: [Int], t4: [Int], a2: [Int], t2: [Int], s2: [Int], d2: [Int]) {
        let x2 = featFn(cases, true)
        let x4 = featFn(cases, false)
        let coarse = x2.map { predictCoarseFromProbs(archProbs: archT2.probabilities($0), topoProbs: topoT2.probabilities($0)) }
        return (
            x4.map { archT4.predict($0) },
            x4.map { topoT4.predict($0) },
            x2.map { archT2.predict($0) },
            x2.map { topoT2.predict($0) },
            coarse.map(\.stance),
            coarse.map(\.dist)
        )
    }

    func projectHybridSuite(_ base: EncodedSuite) -> EncodedSuite {
        let coarse = base.hybridT2.map { predictCoarseFromProbs(archProbs: archT2.probabilities($0), topoProbs: topoT2.probabilities($0)) }
        return EncodedSuite(
            zT2: base.zT2,
            zT4: base.zT4,
            hybridT2: base.hybridT2,
            hybridT4: base.hybridT4,
            predArchT4: base.hybridT4.map { archT4.predict($0) },
            predTopoT4: base.hybridT4.map { topoT4.predict($0) },
            predArchT2: base.hybridT2.map { archT2.predict($0) },
            predTopoT2: base.hybridT2.map { topoT2.predict($0) },
            predStanceT2: coarse.map(\.stance),
            predDistT2: coarse.map(\.dist),
            subbandT2: base.subbandT2,
            subbandT4: base.subbandT4,
            pentagonT2: base.pentagonT2,
            pentagonT4: base.pentagonT4,
            unk: base.unk
        )
    }
}

func fitBaselineHeads(name: String, featFn: ([RelationalDialogueCase], Bool) -> [[Float]], steps: Int = 320, l2: Float = 2e-3) -> LinearBaselineHeads {
    print("[Baseline: \(name)] Fitting linear heads on \(v2TrainCases.count) train cases (pure CPU)...")
    let x2 = featFn(v2TrainCases, true)
    let x4 = featFn(v2TrainCases, false)
    let yA4 = v2TrainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetype) }
    let yT4 = v2TrainCases.map { RelationalPipeline.topologyIndex($0.expectedTopology) }
    let yA2 = v2TrainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetypeT2) }
    let yT2 = v2TrainCases.map { RelationalPipeline.topologyIndex($0.expectedTopologyT2) }
    let hA4 = SoftmaxClassifierHead.train(features: x4, labels: yA4, classCount: 6, steps: steps, l2: l2)
    let hT4 = SoftmaxClassifierHead.train(features: x4, labels: yT4, classCount: 5, steps: steps, l2: l2)
    let hA2 = SoftmaxClassifierHead.train(features: x2, labels: yA2, classCount: 6, steps: steps, l2: l2)
    let hT2 = SoftmaxClassifierHead.train(features: x2, labels: yT2, classCount: 5, steps: steps, l2: l2)
    return LinearBaselineHeads(archT4: hA4, topoT4: hT4, archT2: hA2, topoT2: hT2)
}

func fitHybridHeads() -> LinearBaselineHeads {
    let t0 = Date()
    let groups = v2TrainCases.map(\.group)
    let yA4 = v2TrainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetype) }
    let yT4 = v2TrainCases.map { RelationalPipeline.topologyIndex($0.expectedTopology) }
    let yA2 = v2TrainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetypeT2) }
    let yT2 = v2TrainCases.map { RelationalPipeline.topologyIndex($0.expectedTopologyT2) }
    let hA4 = SoftmaxClassifierHead.trainGroupCV(features: v2Eval.train.hybridT4, labels: yA4, groups: groups, classCount: 6, steps: 350)
    let hT4 = SoftmaxClassifierHead.trainGroupCV(features: v2Eval.train.hybridT4, labels: yT4, groups: groups, classCount: 5, steps: 350)
    let hA2 = SoftmaxClassifierHead.trainGroupCV(features: v2Eval.train.hybridT2, labels: yA2, groups: groups, classCount: 6, steps: 350)
    let hT2 = SoftmaxClassifierHead.trainGroupCV(features: v2Eval.train.hybridT2, labels: yT2, groups: groups, classCount: 5, steps: 350)
    print(String(format: "[Model: v2_hybrid] Calibrated 160D hybrid heads (pure CPU Group-CV, 0 GPU) in %.2fs", Date().timeIntervalSince(t0)))
    return LinearBaselineHeads(archT4: hA4, topoT4: hT4, archT2: hA2, topoT2: hT2)
}

let v2HybridHeads = fitHybridHeads()
let v2HybridUnseen = v2HybridHeads.projectHybridSuite(v2Eval.unseen)
let v2HybridL3a = v2HybridHeads.projectHybridSuite(v2Eval.l3a)
let v2HybridL3b = v2HybridHeads.projectHybridSuite(v2Eval.l3b)
let v2HybridL3c = v2HybridHeads.projectHybridSuite(v2Eval.l3c)
let v2HybridL4 = v2HybridHeads.projectHybridSuite(v2Eval.l4)

let ngramHeads = fitBaselineHeads(name: "char_ngram", featFn: ngramFeatures, steps: 320, l2: 2e-3)
let surfaceHeads = fitBaselineHeads(name: "surface_stats", featFn: surfaceFeatures, steps: 320, l2: 2e-3)

// Evaluate all models on unseen_test (to serve as reference for Layer 3b drop_pt)
func buildAllModelEvalForSuite(
    cases: [RelationalDialogueCase],
    neuralMap: [String: EncodedSuite],
    unseenRef: [String: FourTaskMetrics]?,
    isInversionSuite: Bool
) -> [String: SuiteModelEval] {
    var out: [String: SuiteModelEval] = [:]
    for (mName, enc) in neuralMap {
        let acc = computeFourTaskAccuracy(
            cases: cases,
            predArchT4: enc.predArchT4,
            predTopoT4: enc.predTopoT4,
            predArchT2: enc.predArchT2,
            predTopoT2: enc.predTopoT2
        )
        let drop = unseenRef?[mName].map { computeDropPt(unseen: $0, target: acc) }
        let paired = isInversionSuite ? computePairedInversion(
            cases: cases,
            predArchT4: enc.predArchT4,
            predTopoT4: enc.predTopoT4,
            predArchT2: enc.predArchT2,
            predTopoT2: enc.predTopoT2
        ) : nil
        out[mName] = SuiteModelEval(accuracy: acc, unk_stats: enc.unk, drop_pt_from_unseen_test: drop, paired_inversion_accuracy: paired)
    }

    for (bName, heads, fn) in [("char_ngram", ngramHeads, ngramFeatures), ("surface_stats", surfaceHeads, surfaceFeatures)] {
        let preds = heads.predictSuite(cases, featFn: fn)
        let acc = computeFourTaskAccuracy(
            cases: cases,
            predArchT4: preds.a4,
            predTopoT4: preds.t4,
            predArchT2: preds.a2,
            predTopoT2: preds.t2
        )
        let drop = unseenRef?[bName].map { computeDropPt(unseen: $0, target: acc) }
        let paired = isInversionSuite ? computePairedInversion(
            cases: cases,
            predArchT4: preds.a4,
            predTopoT4: preds.t4,
            predArchT2: preds.a2,
            predTopoT2: preds.t2
        ) : nil
        out[bName] = SuiteModelEval(accuracy: acc, unk_stats: nil, drop_pt_from_unseen_test: drop, paired_inversion_accuracy: paired)
    }
    return out
}

func buildCoarseT2EvalForSuite(
    cases: [RelationalDialogueCase],
    neuralMap: [String: EncodedSuite]
) -> [String: CoarseEarlyT2Metrics] {
    var out: [String: CoarseEarlyT2Metrics] = [:]
    for (mName, enc) in neuralMap {
        out[mName] = computeCoarseEarlyT2(cases: cases, predStance: enc.predStanceT2, predDist: enc.predDistT2)
    }
    for (bName, heads, fn) in [("char_ngram", ngramHeads, ngramFeatures), ("surface_stats", surfaceHeads, surfaceFeatures)] {
        let preds = heads.predictSuite(cases, featFn: fn)
        out[bName] = computeCoarseEarlyT2(cases: cases, predStance: preds.s2, predDist: preds.d2)
    }
    return out
}

let unseenModelMap = buildAllModelEvalForSuite(
    cases: v2UnseenCases,
    neuralMap: ["v1": v1Eval.unseen, "v2_nometric": v2NoMetricEval.unseen, "v2": v2Eval.unseen, "v2_hybrid": v2HybridUnseen],
    unseenRef: nil,
    isInversionSuite: false
)
let unseenAccRef = unseenModelMap.mapValues(\.accuracy)

for k in ["v1", "v2_nometric", "v2", "v2_hybrid", "char_ngram", "surface_stats"] {
    if let r = unseenModelMap[k]?.accuracy {
        print("[Unseen Test] \(k.padding(toLength: 14, withPad: " ", startingAt: 0)): archT4 \(pct(r.archetype_t4)) | topoT4 \(pct(r.topology_t4)) | archT2 \(pct(r.archetype_t2)) | topoT2 \(pct(r.topology_t2)) | mean \(pct(r.mean))")
    }
}

// 2. Layer 2: (sLL, sHH, rHH) 3D Linear Probe + MI & Pentagon External Validity
func evaluateSubbandGeometry(_ suites: AllSuitesForModel) -> SubbandGeometryModelEval {
    let trX2 = suites.train.subbandT2.map { [$0.sLL, $0.sHH, $0.rHH] }
    let trX4 = suites.train.subbandT4.map { [$0.sLL, $0.sHH, $0.rHH] }
    let unX2 = suites.unseen.subbandT2.map { [$0.sLL, $0.sHH, $0.rHH] }
    let unX4 = suites.unseen.subbandT4.map { [$0.sLL, $0.sHH, $0.rHH] }

    let trYA4 = v2TrainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetype) }
    let trYT4 = v2TrainCases.map { RelationalPipeline.topologyIndex($0.expectedTopology) }
    let trYA2 = v2TrainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetypeT2) }
    let trYT2 = v2TrainCases.map { RelationalPipeline.topologyIndex($0.expectedTopologyT2) }

    let unYA4 = v2UnseenCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetype) }
    let unYT4 = v2UnseenCases.map { RelationalPipeline.topologyIndex($0.expectedTopology) }
    let unYA2 = v2UnseenCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetypeT2) }
    let unYT2 = v2UnseenCases.map { RelationalPipeline.topologyIndex($0.expectedTopologyT2) }

    let hA4 = SoftmaxClassifierHead.train(features: trX4, labels: trYA4, classCount: 6, steps: 350, l2: 1e-3)
    let hT4 = SoftmaxClassifierHead.train(features: trX4, labels: trYT4, classCount: 5, steps: 350, l2: 1e-3)
    let hA2 = SoftmaxClassifierHead.train(features: trX2, labels: trYA2, classCount: 6, steps: 350, l2: 1e-3)
    let hT2 = SoftmaxClassifierHead.train(features: trX2, labels: trYT2, classCount: 5, steps: 350, l2: 1e-3)

    let trAcc = FourTaskMetrics(
        archT4: accuracyRate(trX4.map { hA4.predict($0) }, trYA4),
        topoT4: accuracyRate(trX4.map { hT4.predict($0) }, trYT4),
        archT2: accuracyRate(trX2.map { hA2.predict($0) }, trYA2),
        topoT2: accuracyRate(trX2.map { hT2.predict($0) }, trYT2)
    )
    let unAcc = FourTaskMetrics(
        archT4: accuracyRate(unX4.map { hA4.predict($0) }, unYA4),
        topoT4: accuracyRate(unX4.map { hT4.predict($0) }, unYT4),
        archT2: accuracyRate(unX2.map { hA2.predict($0) }, unYA2),
        topoT2: accuracyRate(unX2.map { hT2.predict($0) }, unYT2)
    )

    let miA4 = BenchmarkMath.discretizedSubbandMI(triples: suites.unseen.subbandT4, labels: unYA4, classCount: 6)
    let miT4 = BenchmarkMath.discretizedSubbandMI(triples: suites.unseen.subbandT4, labels: unYT4, classCount: 5)
    let miA2 = BenchmarkMath.discretizedSubbandMI(triples: suites.unseen.subbandT2, labels: unYA2, classCount: 6)
    let miT2 = BenchmarkMath.discretizedSubbandMI(triples: suites.unseen.subbandT2, labels: unYT2, classCount: 5)

    return SubbandGeometryModelEval(
        linear_probe_3d_train_accuracy: trAcc,
        linear_probe_3d_unseen_accuracy: unAcc,
        mutual_information_bits: FourTaskMetrics(archT4: miA4.miBits, topoT4: miT4.miBits, archT2: miA2.miBits, topoT2: miT2.miBits),
        label_entropy_bits: FourTaskMetrics(archT4: miA4.labelEntropyBits, topoT4: miT4.labelEntropyBits, archT2: miA2.labelEntropyBits, topoT2: miT2.labelEntropyBits)
    )
}

func evaluatePentagonValidity(_ suites: AllSuitesForModel) -> PentagonModelValidity {
    let l1HumT2 = l1T2MatchedCases.map(\.pent)
    let l1HumT4 = l1T4MatchedCases.map(\.pent)
    let l4HumT2 = l4MetaOrdered.map(\.pentagon_t2_mean)
    let l4HumT4 = l4MetaOrdered.map(\.pentagon_t4_mean)

    let c1T2 = correlatePentagon(modelPreds: suites.l1T2.pentagonT2, humanMeans: l1HumT2)
    let c1T4 = correlatePentagon(modelPreds: suites.l1T4.pentagonT4, humanMeans: l1HumT4)
    let c4T2 = correlatePentagon(modelPreds: suites.l4.pentagonT2, humanMeans: l4HumT2)
    let c4T4 = correlatePentagon(modelPreds: suites.l4.pentagonT4, humanMeans: l4HumT4)
    let combT2 = correlatePentagon(modelPreds: suites.l1T2.pentagonT2 + suites.l4.pentagonT2, humanMeans: l1HumT2 + l4HumT2)
    let combT4 = correlatePentagon(modelPreds: suites.l1T4.pentagonT4 + suites.l4.pentagonT4, humanMeans: l1HumT4 + l4HumT4)

    return PentagonModelValidity(
        layer1_blind_t2: c1T2,
        layer1_blind_t4: c1T4,
        layer4_gold_t2: c4T2,
        layer4_gold_t4: c4T4,
        combined_t2: combT2,
        combined_t4: combT4
    )
}

let geomAblation: [String: SubbandGeometryModelEval] = [
    "v1": evaluateSubbandGeometry(v1Eval),
    "v2_nometric": evaluateSubbandGeometry(v2NoMetricEval),
    "v2": evaluateSubbandGeometry(v2Eval)
]
let pentValidity: [String: PentagonModelValidity] = [
    "v1": evaluatePentagonValidity(v1Eval),
    "v2_nometric": evaluatePentagonValidity(v2NoMetricEval),
    "v2": evaluatePentagonValidity(v2Eval)
]

for m in ["v1", "v2_nometric", "v2"] {
    let g = geomAblation[m]!
    let p = pentValidity[m]!
    print(String(
        format: "[Layer 2] %-12@ | 3D Probe mean: %.2f%% | MI mean: %.3f bits | Pentagon Pearson r (comb T2/T4): %.3f / %.3f",
        m as NSString,
        g.linear_probe_3d_unseen_accuracy.mean * 100.0,
        g.mutual_information_bits.mean,
        p.combined_t2.mean_pearson_r,
        p.combined_t4.mean_pearson_r
    ))
}

// 3. Layer 3 OOD Benchmarks (3a, 3b, 3c)
let l3aEval = buildAllModelEvalForSuite(
    cases: l3aCases,
    neuralMap: ["v1": v1Eval.l3a, "v2_nometric": v2NoMetricEval.l3a, "v2": v2Eval.l3a, "v2_hybrid": v2HybridL3a],
    unseenRef: nil,
    isInversionSuite: false
)
let l3bEval = buildAllModelEvalForSuite(
    cases: l3bCases,
    neuralMap: ["v1": v1Eval.l3b, "v2_nometric": v2NoMetricEval.l3b, "v2": v2Eval.l3b, "v2_hybrid": v2HybridL3b],
    unseenRef: unseenAccRef,
    isInversionSuite: false
)
let l3cEval = buildAllModelEvalForSuite(
    cases: l3cCases,
    neuralMap: ["v1": v1Eval.l3c, "v2_nometric": v2NoMetricEval.l3c, "v2": v2Eval.l3c, "v2_hybrid": v2HybridL3c],
    unseenRef: nil,
    isInversionSuite: true
)

for (suiteName, evalMap) in [("layer3a", l3aEval), ("layer3b", l3bEval), ("layer3c", l3cEval)] {
    for k in ["v1", "v2", "v2_hybrid", "char_ngram", "surface_stats"] {
        if let e = evalMap[k] {
            let extra = if let p = e.paired_inversion_accuracy {
                String(format: " | paired_mean %.2f%%", p.mean_paired_accuracy * 100.0)
            } else if let d = e.drop_pt_from_unseen_test {
                String(format: " | drop %+.2f pt", d.mean)
            } else {
                ""
            }
            print("[\(suiteName)] \(k.padding(toLength: 14, withPad: " ", startingAt: 0)): mean \(pct(e.accuracy.mean)) (archT4 \(pct(e.accuracy.archetype_t4)), topoT4 \(pct(e.accuracy.topology_t4)), archT2 \(pct(e.accuracy.archetype_t2)), topoT2 \(pct(e.accuracy.topology_t2)))\(extra)")
        }
    }
}

// 4. Layer 4 External Anchor Gold Benchmark & Future Behavioral Event Prediction (Stratified 5-Fold CV)
let l4ZeroShot = buildAllModelEvalForSuite(
    cases: l4Cases,
    neuralMap: ["v1": v1Eval.l4, "v2_nometric": v2NoMetricEval.l4, "v2": v2Eval.l4, "v2_hybrid": v2HybridL4],
    unseenRef: nil,
    isInversionSuite: false
)

let p0CoarseBenchmarks: [String: [String: CoarseEarlyT2Metrics]] = [
    "unseen_test": buildCoarseT2EvalForSuite(cases: v2UnseenCases, neuralMap: ["v1": v1Eval.unseen, "v2_nometric": v2NoMetricEval.unseen, "v2": v2Eval.unseen, "v2_hybrid": v2HybridUnseen]),
    "layer3a": buildCoarseT2EvalForSuite(cases: l3aCases, neuralMap: ["v1": v1Eval.l3a, "v2_nometric": v2NoMetricEval.l3a, "v2": v2Eval.l3a, "v2_hybrid": v2HybridL3a]),
    "layer3b": buildCoarseT2EvalForSuite(cases: l3bCases, neuralMap: ["v1": v1Eval.l3b, "v2_nometric": v2NoMetricEval.l3b, "v2": v2Eval.l3b, "v2_hybrid": v2HybridL3b]),
    "layer3c": buildCoarseT2EvalForSuite(cases: l3cCases, neuralMap: ["v1": v1Eval.l3c, "v2_nometric": v2NoMetricEval.l3c, "v2": v2Eval.l3c, "v2_hybrid": v2HybridL3c]),
    "layer4_gold": buildCoarseT2EvalForSuite(cases: l4Cases, neuralMap: ["v1": v1Eval.l4, "v2_nometric": v2NoMetricEval.l4, "v2": v2Eval.l4, "v2_hybrid": v2HybridL4])
]
for sKey in ["unseen_test", "layer3a", "layer3b", "layer3c", "layer4_gold"] {
    if let m = p0CoarseBenchmarks[sKey]?["v2"], let mh = p0CoarseBenchmarks[sKey]?["v2_hybrid"] {
        print("[P0 Coarse T2 | \(sKey.padding(toLength: 11, withPad: " ", startingAt: 0))] v2 mean \(pct(m.mean)) (stance \(pct(m.early_stance_4way)), dist \(pct(m.early_distance_3way))) | v2_hybrid mean \(pct(mh.mean)) (stance \(pct(mh.early_stance_4way)), dist \(pct(mh.early_distance_3way)))")
    }
}

let yRet30 = l4MetaOrdered.map { $0.outcomes.retained_30d }
let yRet90 = l4MetaOrdered.map { $0.outcomes.retained_90d }
let yBlock = l4MetaOrdered.map { $0.outcomes.moderation_or_block }
let yDrift = l4MetaOrdered.map { $0.outcomes.response_interval_drift }

let foldsRet30 = BenchmarkMath.stratifiedBinaryFolds(labels: yRet30, k: 5, seed: 2026)
let foldsRet90 = BenchmarkMath.stratifiedBinaryFolds(labels: yRet90, k: 5, seed: 2027)
let foldsBlock = BenchmarkMath.stratifiedBinaryFolds(labels: yBlock, k: 5, seed: 2028)
let foldsDrift = BenchmarkMath.stratifiedContinuousFolds(targets: yDrift, k: 5, seed: 2029)

let featureBanks: [(String, [[Float]])] = [
    ("v2_hybrid_compact_t2", v2Eval.l4.hybridT2.map { Array($0[128..<160]) }),
    ("v2_hybrid_compact_t4", v2Eval.l4.hybridT4.map { Array($0[128..<160]) }),
    ("v2_hybrid_t2", v2Eval.l4.hybridT2),
    ("v2_hybrid_t4", v2Eval.l4.hybridT4),
    ("v2_z_rel_t2", v2Eval.l4.zT2),
    ("v2_z_rel_t4", v2Eval.l4.zT4),
    ("v1_z_rel_t2", v1Eval.l4.zT2),
    ("v1_z_rel_t4", v1Eval.l4.zT4),
    ("v2_nometric_z_rel_t2", v2NoMetricEval.l4.zT2),
    ("v2_nometric_z_rel_t4", v2NoMetricEval.l4.zT4),
    ("char_ngram_t2", ngramFeatures(l4Cases, early: true)),
    ("char_ngram_t4", ngramFeatures(l4Cases, early: false)),
    ("surface_stats_t2", surfaceFeatures(l4Cases, early: true)),
    ("surface_stats_t4", surfaceFeatures(l4Cases, early: false))
]

var outRet30: [String: BinaryOutcomeMetrics] = [:]
var outRet90: [String: BinaryOutcomeMetrics] = [:]
var outBlock: [String: BinaryOutcomeMetrics] = [:]
var outDrift: [String: ContinuousOutcomeMetrics] = [:]

for (fName, feats) in featureBanks {
    let r30 = BenchmarkMath.evaluateBinaryLogisticCV(features: feats, labels: yRet30, folds: foldsRet30)
    let r90 = BenchmarkMath.evaluateBinaryLogisticCV(features: feats, labels: yRet90, folds: foldsRet90)
    let blk = BenchmarkMath.evaluateBinaryLogisticCV(features: feats, labels: yBlock, folds: foldsBlock)
    let drf = BenchmarkMath.evaluateRidgeRegressionCV(features: feats, targets: yDrift, folds: foldsDrift)
    outRet30[fName] = BinaryOutcomeMetrics(roc_auc: r30.rocAUC, accuracy: r30.accuracy)
    outRet90[fName] = BinaryOutcomeMetrics(roc_auc: r90.rocAUC, accuracy: r90.accuracy)
    outBlock[fName] = BinaryOutcomeMetrics(roc_auc: blk.rocAUC, accuracy: blk.accuracy)
    outDrift[fName] = ContinuousOutcomeMetrics(pearson_r: drf.pearsonR, mae: drf.mae)
    print(String(
        format: "[Layer 4 Outcomes] %-22@ | ret30 AUC %.3f | ret90 AUC %.3f | block AUC %.3f | drift r %.3f (MAE %.3f)",
        fName as NSString, r30.rocAUC, r90.rocAUC, blk.rocAUC, drf.pearsonR, drf.mae
    ))
}

let elapsedWallSec = Date().timeIntervalSince(startWallTime)
let execModeLabel = cli.useGPU ? "gpu_mlx" : (didMissAnyCache ? "cpu_multicore_cold" : "cpu_multicore_cached")

let report = BenchmarkMetricsReport(
    generated_at: ISO8601DateFormatter().string(from: Date()),
    execution_mode: execModeLabel,
    wall_clock_seconds: elapsedWallSec,
    v1_causal_t2_leakage_audit: causalAudit,
    in_distribution_unseen_test: Layer3SuiteOutput(n_cases: v2UnseenCases.count, models: unseenModelMap),
    p0_coarse_t2_benchmarks: p0CoarseBenchmarks,
    layer2_validation: Layer2ValidationOutput(
        subband_geometry_ablation: geomAblation,
        pentagon_external_validity: pentValidity
    ),
    layer3_ood_benchmarks: [
        "layer3a_style_domain_holdout": Layer3SuiteOutput(n_cases: l3aCases.count, models: l3aEval),
        "layer3b_independent_paraphrases": Layer3SuiteOutput(n_cases: l3bCases.count, models: l3bEval),
        "layer3c_independent_inversions": Layer3SuiteOutput(n_cases: l3cCases.count, models: l3cEval)
    ],
    layer4_gold_outcome_benchmark: Layer4Output(
        n_cases: l4Cases.count,
        zero_shot_consensus_classification: l4ZeroShot,
        retained_30d_5fold: outRet30,
        retained_90d_5fold: outRet90,
        moderation_or_block_5fold: outBlock,
        response_interval_drift_5fold: outDrift
    )
)

let outURL = URL(fileURLWithPath: cli.outputPath)
try FileManager.default.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let jsonData = try encoder.encode(report)
try jsonData.write(to: outURL)
print("==================================================================")
print(String(format: "Completed in %.2fs (%@) -> Saved to %@", elapsedWallSec, execModeLabel as NSString, cli.outputPath as NSString))
print("==================================================================")
