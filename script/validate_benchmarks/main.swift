import Foundation
import MLX
import Sashiba

struct ValidateBenchmarksCLI: Sendable {
    var dataPath: String = ".tmp"
    var weightsPath: String = ".tmp/weights.safetensors"
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
                if i < args.count { cli.dataPath = args[i] }
            case "-w", "--weights":
                i += 1
                if i < args.count { cli.weightsPath = args[i] }
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

struct PairedInversionMetrics: Codable {
    let total_pairs: Int
    let archetype_t4: Float
    let topology_t4: Float
    let archetype_t2: Float
    let topology_t2: Float
    let mean_paired_accuracy: Float
    let joint_t4_both_correct: Float
}

struct SuiteModelEval: Codable {
    let accuracy: FourTaskMetrics
    let unk_stats: UnkStats?
    let drop_pt_from_unseen_test: FourTaskMetrics?
    let paired_inversion_accuracy: PairedInversionMetrics?
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
    let subband_geometry: SubbandGeometryModelEval
    let pentagon_external_validity: PentagonModelValidity
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

func accuracyRate(_ preds: [Int], _ labels: [Int]) -> Float {
    let n = min(preds.count, labels.count)
    guard n > 0 else { return 0.0 }
    let ok = zip(preds, labels).filter { $0 == $1 }.count
    return Float(ok) / Float(n)
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

func encodeSuite(
    suiteName: String,
    weightsPath: String,
    cases: [RelationalDialogueCase],
    model: MLXSWTHawkAutoencoder?,
    weights: SWTHawkWeights,
    vocab: TextVocabulary,
    extractor: RelationalHybridFeatureExtractor,
    cacheDir: String,
    disableCache: Bool,
    didMissCache: inout Bool,
    maxSeqLen: Int
) -> EncodedSuite {
    let seqLen = cases
        .flatMap { [$0.hostEarlyText, $0.hostText, $0.partnerEarlyText, $0.partnerCurrentText] }
        .map { min(maxSeqLen, vocab.encode($0, maxLength: maxSeqLen, addEOS: false).count) }
        .reduce(max(4, DualStreamRelationalWeights.canonicalSeqLen), max)

    let sT2 = cases.map { RelationalPipeline.encodeTokens($0.hostEarlyText, vocab: vocab, time: seqLen) }
    let sT4 = cases.map { RelationalPipeline.encodeTokens($0.hostText, vocab: vocab, time: seqLen) }
    let rT2 = cases.map { RelationalPipeline.encodeTokens($0.partnerEarlyText, vocab: vocab, time: seqLen) }
    let rT4 = cases.map { RelationalPipeline.encodeTokens($0.partnerCurrentText, vocab: vocab, time: seqLen) }

    let attrs = try? FileManager.default.attributesOfItem(atPath: weightsPath)
    let wSize = (attrs?[.size] as? NSNumber)?.int64Value ?? 0
    let wMtime = Int64(((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0).rounded())
    let cachePath = (cacheDir as NSString).appendingPathComponent(
        "v4_\(suiteName)_causal_n\(cases.count)_t\(seqLen)_s\(wSize)_m\(wMtime).bin"
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
        let key = (rc.id.hasSuffix("_A") || rc.id.hasSuffix("_B")) ? String(rc.id.dropLast(2)) : rc.id
        byPairKey[key, default: []].append(idx)
    }
    let validPairs = byPairKey.keys.sorted().compactMap { k -> (Int, Int)? in
        guard let arr = byPairKey[k], arr.count == 2 else { return nil }
        return (arr[0], arr[1])
    }
    let pCount = max(1, validPairs.count)
    var okAT4 = 0, okTT4 = 0, okAT2 = 0, okTT2 = 0, okJointT4 = 0
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
    return PentagonAxisCorrelation(
        n_cases: n,
        contextDepth: rs[0],
        empathicResonance: rs[1],
        communicativeHarmony: rs[2],
        boundaryStability: rs[3],
        longTermViability: rs[4],
        mean_pearson_r: rs.reduce(0.0, +) / 5.0
    )
}

// MARK: - Main Execution (v4 & v4_hybrid Only)

let startWallTime = Date()
let cli = ValidateBenchmarksCLI.parse(CommandLine.arguments)
if cli.useGPU {
    MLX.Memory.cacheLimit = 128 * 1024 * 1024
    MLX.Memory.clearCache()
}
let pct = { (v: Float) in String(format: "%6.2f%%", v * 100.0) }

print("==================================================================")
print("Sashiba External Benchmark Validation [v4 & v4_hybrid]")
print("Execution mode: \(cli.useGPU ? "GPU (MLX)" : "Pure Multi-Core CPU (0 GPU VRAM)") | Cache: \(cli.disableCache ? "OFF" : cli.cacheDir)")
print("==================================================================")

let mainConfig = DatasetConfiguration.load(from: cli.dataPath)
let trainCases = mainConfig.relationalDialogueCases.filter { !$0.isZeroShotUnseen }
let unseenCases = mainConfig.relationalDialogueCases.filter { $0.isZeroShotUnseen }
let caseByID = Dictionary(uniqueKeysWithValues: mainConfig.relationalDialogueCases.map { ($0.id, $0) })

let l3aCases = DatasetConfiguration.load(from: cli.layer3aDir).relationalDialogueCases
let l3bCases = DatasetConfiguration.load(from: cli.layer3bDir).relationalDialogueCases
let l3cCases = DatasetConfiguration.load(from: cli.layer3cDir).relationalDialogueCases
let l4CasesRaw = DatasetConfiguration.load(from: cli.layer4Dir).relationalDialogueCases

let l1Items = try JSONDecoder().decode(
    [Layer1ConsensusItem].self,
    from: Data(contentsOf: URL(fileURLWithPath: cli.layer1ConsensusPath))
)
let l1T2MatchedCases: [(pent: [Float], rc: RelationalDialogueCase)] = if FileManager.default.fileExists(atPath: cli.layer1T2FullPath),
    let items = try? JSONDecoder().decode([Layer1T2OnlyItem].self, from: Data(contentsOf: URL(fileURLWithPath: cli.layer1T2FullPath))) {
    items.compactMap { item in caseByID[item.scenario_id].map { (item.pentagon_t2_mean, $0) } }
} else {
    l1Items.compactMap { item in caseByID[item.scenario_id].map { (item.pentagon_t2_mean, $0) } }
}

let l1T4MatchedCases: [(pent: [Float], rc: RelationalDialogueCase)] = if FileManager.default.fileExists(atPath: cli.layer1T4FullPath),
    let items = try? JSONDecoder().decode([Layer1T4OnlyItem].self, from: Data(contentsOf: URL(fileURLWithPath: cli.layer1T4FullPath))) {
    items.compactMap { item in caseByID[item.scenario_id].map { (item.pentagon_t4_mean, $0) } }
} else {
    l1Items.compactMap { item in caseByID[item.scenario_id].map { (item.pentagon_t4_mean, $0) } }
}

let l4MetaPath = (cli.layer4Dir as NSString).appendingPathComponent("metadata.json")
let l4Meta = try JSONDecoder().decode(Layer4MetadataFile.self, from: Data(contentsOf: URL(fileURLWithPath: l4MetaPath)))
let l4RawByID = Dictionary(uniqueKeysWithValues: l4CasesRaw.map { ($0.id, $0) })
let l4Aligned: [(rc: RelationalDialogueCase, meta: Layer4MetadataItem)] = l4Meta.items.compactMap { m in
    guard let raw = l4RawByID[m.scenario_id] else { return nil }
    let rc = RelationalDialogueCase(
        id: raw.id, group: raw.group, personaStyle: raw.personaStyle, topicDomain: raw.topicDomain, title: raw.title,
        expectedTopologyT2: m.consensus_topology_t2, expectedArchetypeT2: m.consensus_archetype_t2,
        expectedTopology: m.consensus_topology_t4, expectedArchetype: m.consensus_archetype_t4,
        hostEarlyText: raw.hostEarlyText, hostText: raw.hostText,
        partnerEarlyText: raw.partnerEarlyText, partnerCurrentText: raw.partnerCurrentText,
        harmonizedPartnerText: raw.harmonizedPartnerText,
        insightLevel0: raw.insightLevel0, insightLevel1: raw.insightLevel1, insightLevel2: raw.insightLevel2,
        isZeroShotUnseen: true
    )
    return (rc, m)
}
let l4Cases = l4Aligned.map(\.rc)
let l4MetaOrdered = l4Aligned.map(\.meta)

print("Loaded splits: train=\(trainCases.count), unseen=\(unseenCases.count), L1_T2=\(l1T2MatchedCases.count), L1_T4=\(l1T4MatchedCases.count), L3a=\(l3aCases.count), L3b=\(l3bCases.count), L3c=\(l3cCases.count), L4=\(l4Cases.count)")

let w = try SWTHawkWeights.load(from: URL(fileURLWithPath: cli.weightsPath))
let vocab = TextVocabulary.deserialize(w.serializedVocabulary)
let extractor = RelationalHybridFeatureExtractor(vocabulary: vocab)
let gpuModel: MLXSWTHawkAutoencoder? = cli.useGPU ? MLXSWTHawkAutoencoder(weights: w) : nil
var didMissAnyCache = false

@MainActor
func enc(_ sName: String, _ cs: [RelationalDialogueCase]) -> EncodedSuite {
    var localMiss = false
    let out = encodeSuite(
        suiteName: sName, weightsPath: cli.weightsPath, cases: cs, model: gpuModel, weights: w,
        vocab: vocab, extractor: extractor, cacheDir: cli.cacheDir, disableCache: cli.disableCache,
        didMissCache: &localMiss, maxSeqLen: cli.maxSeqLen
    )
    if localMiss { didMissAnyCache = true }
    return out
}

let tEncode = Date()
let trSuite = enc("train", trainCases)
let unSuite = enc("unseen", unseenCases)
let l1T2Suite = enc("l1_t2", l1T2MatchedCases.map(\.rc))
let l1T4Suite = enc("l1_t4", l1T4MatchedCases.map(\.rc))
let l3aSuite = enc("l3a", l3aCases)
let l3bSuite = enc("l3b", l3bCases)
let l3cSuite = enc("l3c", l3cCases)
let l4Suite = enc("l4", l4Cases)
if cli.useGPU { MLX.Memory.clearCache() }
print(String(format: "[v4] Encoded all suites in %.2fs", Date().timeIntervalSince(tEncode)))

// Resolve v4_hybrid heads (prefer serialized heads inside weights.safetensors; otherwise calibrate on trainCases)
let hybridArchT4: SoftmaxClassifierHead
let hybridTopoT4: SoftmaxClassifierHead
let hybridArchT2: SoftmaxClassifierHead
let hybridTopoT2: SoftmaxClassifierHead

let cal = w.relational.calibratedHeads
if let ha4 = cal.hybridArchetypeT4, let ht4 = cal.hybridTopologyT4,
   let ha2 = cal.hybridArchetypeT2, let ht2 = cal.hybridTopologyT2,
   ha4.featureDim == trSuite.hybridT4[0].count {
    hybridArchT4 = ha4
    hybridTopoT4 = ht4
    hybridArchT2 = ha2
    hybridTopoT2 = ht2
    print("[v4_hybrid] Loaded serialized 160D hybrid heads directly from \(cli.weightsPath)")
} else {
    let groups = trainCases.map(\.group)
    let yA4 = trainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetype) }
    let yT4 = trainCases.map { RelationalPipeline.topologyIndex($0.expectedTopology) }
    let yA2 = trainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetypeT2) }
    let yT2 = trainCases.map { RelationalPipeline.topologyIndex($0.expectedTopologyT2) }
    hybridArchT4 = SoftmaxClassifierHead.trainGroupCV(features: trSuite.hybridT4, labels: yA4, groups: groups, classCount: 6, steps: 350)
    hybridTopoT4 = SoftmaxClassifierHead.trainGroupCV(features: trSuite.hybridT4, labels: yT4, groups: groups, classCount: 5, steps: 350)
    hybridArchT2 = SoftmaxClassifierHead.trainGroupCV(features: trSuite.hybridT2, labels: yA2, groups: groups, classCount: 6, steps: 350)
    hybridTopoT2 = SoftmaxClassifierHead.trainGroupCV(features: trSuite.hybridT2, labels: yT2, groups: groups, classCount: 5, steps: 350)
    print("[v4_hybrid] Calibrated 160D hybrid heads on CPU Group-CV")
}

func projectHybrid(_ base: EncodedSuite) -> EncodedSuite {
    let coarse = base.hybridT2.map { predictCoarseFromProbs(archProbs: hybridArchT2.probabilities($0), topoProbs: hybridTopoT2.probabilities($0)) }
    return EncodedSuite(
        zT2: base.zT2, zT4: base.zT4, hybridT2: base.hybridT2, hybridT4: base.hybridT4,
        predArchT4: base.hybridT4.map { hybridArchT4.predict($0) },
        predTopoT4: base.hybridT4.map { hybridTopoT4.predict($0) },
        predArchT2: base.hybridT2.map { hybridArchT2.predict($0) },
        predTopoT2: base.hybridT2.map { hybridTopoT2.predict($0) },
        predStanceT2: coarse.map(\.stance),
        predDistT2: coarse.map(\.dist),
        subbandT2: base.subbandT2, subbandT4: base.subbandT4,
        pentagonT2: base.pentagonT2, pentagonT4: base.pentagonT4, unk: base.unk
    )
}

let unHybrid = projectHybrid(unSuite)
let l3aHybrid = projectHybrid(l3aSuite)
let l3bHybrid = projectHybrid(l3bSuite)
let l3cHybrid = projectHybrid(l3cSuite)
let l4Hybrid = projectHybrid(l4Suite)

func evalPair(
    cases: [RelationalDialogueCase],
    v4Suite: EncodedSuite,
    hybSuite: EncodedSuite,
    unseenRef: [String: FourTaskMetrics]?,
    isInversionSuite: Bool
) -> [String: SuiteModelEval] {
    var out: [String: SuiteModelEval] = [:]
    for (mName, enc) in [("v4", v4Suite), ("v4_hybrid", hybSuite)] {
        let acc = computeFourTaskAccuracy(
            cases: cases, predArchT4: enc.predArchT4, predTopoT4: enc.predTopoT4,
            predArchT2: enc.predArchT2, predTopoT2: enc.predTopoT2
        )
        let drop = unseenRef?[mName].map { computeDropPt(unseen: $0, target: acc) }
        let paired = isInversionSuite ? computePairedInversion(
            cases: cases, predArchT4: enc.predArchT4, predTopoT4: enc.predTopoT4,
            predArchT2: enc.predArchT2, predTopoT2: enc.predTopoT2
        ) : nil
        out[mName] = SuiteModelEval(accuracy: acc, unk_stats: enc.unk, drop_pt_from_unseen_test: drop, paired_inversion_accuracy: paired)
    }
    return out
}

// 1. In-Distribution unseen_test
let unseenEval = evalPair(cases: unseenCases, v4Suite: unSuite, hybSuite: unHybrid, unseenRef: nil, isInversionSuite: false)
let unseenAccRef = unseenEval.mapValues(\.accuracy)
for k in ["v4", "v4_hybrid"] {
    if let r = unseenEval[k]?.accuracy {
        print("[Unseen Test] \(k.padding(toLength: 10, withPad: " ", startingAt: 0)): archT4 \(pct(r.archetype_t4)) | topoT4 \(pct(r.topology_t4)) | archT2 \(pct(r.archetype_t2)) | topoT2 \(pct(r.topology_t2)) | mean \(pct(r.mean))")
    }
}

// 2. Layer 2: Subband Geometry & Pentagon Validity
let trX2 = trSuite.subbandT2.map { [$0.sLL, $0.sHH, $0.rHH] }
let trX4 = trSuite.subbandT4.map { [$0.sLL, $0.sHH, $0.rHH] }
let unX2 = unSuite.subbandT2.map { [$0.sLL, $0.sHH, $0.rHH] }
let unX4 = unSuite.subbandT4.map { [$0.sLL, $0.sHH, $0.rHH] }

let trYA4 = trainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetype) }
let trYT4 = trainCases.map { RelationalPipeline.topologyIndex($0.expectedTopology) }
let trYA2 = trainCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetypeT2) }
let trYT2 = trainCases.map { RelationalPipeline.topologyIndex($0.expectedTopologyT2) }

let unYA4 = unseenCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetype) }
let unYT4 = unseenCases.map { RelationalPipeline.topologyIndex($0.expectedTopology) }
let unYA2 = unseenCases.map { RelationalPipeline.archetypeIndex($0.expectedArchetypeT2) }
let unYT2 = unseenCases.map { RelationalPipeline.topologyIndex($0.expectedTopologyT2) }

let hA4 = SoftmaxClassifierHead.train(features: trX4, labels: trYA4, classCount: 6, steps: 350, l2: 1e-3)
let hT4 = SoftmaxClassifierHead.train(features: trX4, labels: trYT4, classCount: 5, steps: 350, l2: 1e-3)
let hA2 = SoftmaxClassifierHead.train(features: trX2, labels: trYA2, classCount: 6, steps: 350, l2: 1e-3)
let hT2 = SoftmaxClassifierHead.train(features: trX2, labels: trYT2, classCount: 5, steps: 350, l2: 1e-3)

let miA4 = BenchmarkMath.discretizedSubbandMI(triples: unSuite.subbandT4, labels: unYA4, classCount: 6)
let miT4 = BenchmarkMath.discretizedSubbandMI(triples: unSuite.subbandT4, labels: unYT4, classCount: 5)
let miA2 = BenchmarkMath.discretizedSubbandMI(triples: unSuite.subbandT2, labels: unYA2, classCount: 6)
let miT2 = BenchmarkMath.discretizedSubbandMI(triples: unSuite.subbandT2, labels: unYT2, classCount: 5)

let geomEval = SubbandGeometryModelEval(
    linear_probe_3d_train_accuracy: FourTaskMetrics(
        archT4: accuracyRate(trX4.map { hA4.predict($0) }, trYA4),
        topoT4: accuracyRate(trX4.map { hT4.predict($0) }, trYT4),
        archT2: accuracyRate(trX2.map { hA2.predict($0) }, trYA2),
        topoT2: accuracyRate(trX2.map { hT2.predict($0) }, trYT2)
    ),
    linear_probe_3d_unseen_accuracy: FourTaskMetrics(
        archT4: accuracyRate(unX4.map { hA4.predict($0) }, unYA4),
        topoT4: accuracyRate(unX4.map { hT4.predict($0) }, unYT4),
        archT2: accuracyRate(unX2.map { hA2.predict($0) }, unYA2),
        topoT2: accuracyRate(unX2.map { hT2.predict($0) }, unYT2)
    ),
    mutual_information_bits: FourTaskMetrics(archT4: miA4.miBits, topoT4: miT4.miBits, archT2: miA2.miBits, topoT2: miT2.miBits),
    label_entropy_bits: FourTaskMetrics(archT4: miA4.labelEntropyBits, topoT4: miT4.labelEntropyBits, archT2: miA2.labelEntropyBits, topoT2: miT2.labelEntropyBits)
)

let l1HumT2 = l1T2MatchedCases.map(\.pent)
let l1HumT4 = l1T4MatchedCases.map(\.pent)
let l4HumT2 = l4MetaOrdered.map(\.pentagon_t2_mean)
let l4HumT4 = l4MetaOrdered.map(\.pentagon_t4_mean)

let pentEval = PentagonModelValidity(
    layer1_blind_t2: correlatePentagon(modelPreds: l1T2Suite.pentagonT2, humanMeans: l1HumT2),
    layer1_blind_t4: correlatePentagon(modelPreds: l1T4Suite.pentagonT4, humanMeans: l1HumT4),
    layer4_gold_t2: correlatePentagon(modelPreds: l4Suite.pentagonT2, humanMeans: l4HumT2),
    layer4_gold_t4: correlatePentagon(modelPreds: l4Suite.pentagonT4, humanMeans: l4HumT4),
    combined_t2: correlatePentagon(modelPreds: l1T2Suite.pentagonT2 + l4Suite.pentagonT2, humanMeans: l1HumT2 + l4HumT2),
    combined_t4: correlatePentagon(modelPreds: l1T4Suite.pentagonT4 + l4Suite.pentagonT4, humanMeans: l1HumT4 + l4HumT4)
)
print(String(
    format: "[Layer 2] v4 | 3D Probe mean: %.2f%% | MI mean: %.3f bits | Pentagon Pearson r (comb T2/T4): %.3f / %.3f",
    geomEval.linear_probe_3d_unseen_accuracy.mean * 100.0,
    geomEval.mutual_information_bits.mean,
    pentEval.combined_t2.mean_pearson_r,
    pentEval.combined_t4.mean_pearson_r
))

// 3. Layer 3 OOD Benchmarks (3a, 3b, 3c)
let l3aEval = evalPair(cases: l3aCases, v4Suite: l3aSuite, hybSuite: l3aHybrid, unseenRef: nil, isInversionSuite: false)
let l3bEval = evalPair(cases: l3bCases, v4Suite: l3bSuite, hybSuite: l3bHybrid, unseenRef: unseenAccRef, isInversionSuite: false)
let l3cEval = evalPair(cases: l3cCases, v4Suite: l3cSuite, hybSuite: l3cHybrid, unseenRef: nil, isInversionSuite: true)

for (suiteName, evalMap) in [("layer3a", l3aEval), ("layer3b", l3bEval), ("layer3c", l3cEval)] {
    for k in ["v4", "v4_hybrid"] {
        if let e = evalMap[k] {
            let extra = if let p = e.paired_inversion_accuracy {
                String(format: " | paired_mean %.2f%%", p.mean_paired_accuracy * 100.0)
            } else if let d = e.drop_pt_from_unseen_test {
                String(format: " | drop %+.2f pt", d.mean)
            } else { "" }
            print("[\(suiteName)] \(k.padding(toLength: 10, withPad: " ", startingAt: 0)): mean \(pct(e.accuracy.mean)) (archT4 \(pct(e.accuracy.archetype_t4)), topoT4 \(pct(e.accuracy.topology_t4)), archT2 \(pct(e.accuracy.archetype_t2)), topoT2 \(pct(e.accuracy.topology_t2)))\(extra)")
        }
    }
}

// 4. Layer 4 External Gold Benchmark & Coarse T2
let l4ZeroShot = evalPair(cases: l4Cases, v4Suite: l4Suite, hybSuite: l4Hybrid, unseenRef: nil, isInversionSuite: false)
for k in ["v4", "v4_hybrid"] {
    if let e = l4ZeroShot[k] {
        print("[layer4 ] \(k.padding(toLength: 10, withPad: " ", startingAt: 0)): mean \(pct(e.accuracy.mean)) (archT4 \(pct(e.accuracy.archetype_t4)), topoT4 \(pct(e.accuracy.topology_t4)), archT2 \(pct(e.accuracy.archetype_t2)), topoT2 \(pct(e.accuracy.topology_t2)))")
    }
}

func coarsePair(_ cs: [RelationalDialogueCase], _ s4: EncodedSuite, _ sh: EncodedSuite) -> [String: CoarseEarlyT2Metrics] {
    return [
        "v4": computeCoarseEarlyT2(cases: cs, predStance: s4.predStanceT2, predDist: s4.predDistT2),
        "v4_hybrid": computeCoarseEarlyT2(cases: cs, predStance: sh.predStanceT2, predDist: sh.predDistT2)
    ]
}
let p0CoarseBenchmarks: [String: [String: CoarseEarlyT2Metrics]] = [
    "unseen_test": coarsePair(unseenCases, unSuite, unHybrid),
    "layer3a": coarsePair(l3aCases, l3aSuite, l3aHybrid),
    "layer3b": coarsePair(l3bCases, l3bSuite, l3bHybrid),
    "layer3c": coarsePair(l3cCases, l3cSuite, l3cHybrid),
    "layer4_gold": coarsePair(l4Cases, l4Suite, l4Hybrid)
]
for sKey in ["unseen_test", "layer3a", "layer3b", "layer3c", "layer4_gold"] {
    if let m = p0CoarseBenchmarks[sKey]?["v4"], let mh = p0CoarseBenchmarks[sKey]?["v4_hybrid"] {
        print("[P0 Coarse T2 | \(sKey.padding(toLength: 11, withPad: " ", startingAt: 0))] v4 mean \(pct(m.mean)) (stance \(pct(m.early_stance_4way)), dist \(pct(m.early_distance_3way))) | v4_hybrid mean \(pct(mh.mean)) (stance \(pct(mh.early_stance_4way)), dist \(pct(mh.early_distance_3way)))")
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
    ("v4_z_rel_t2", l4Suite.zT2),
    ("v4_hybrid_t2", l4Suite.hybridT2),
    ("v4_z_rel_t4", l4Suite.zT4),
    ("v4_hybrid_t4", l4Suite.hybridT4)
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
        format: "[Layer 4 Outcomes] %-14@ | ret30 AUC %.3f | ret90 AUC %.3f | block AUC %.3f | drift r %.3f (MAE %.3f)",
        fName as NSString, r30.rocAUC, r90.rocAUC, blk.rocAUC, drf.pearsonR, drf.mae
    ))
}

let elapsedWallSec = Date().timeIntervalSince(startWallTime)
let execModeLabel = cli.useGPU ? "gpu_mlx" : (didMissAnyCache ? "cpu_multicore_cold" : "cpu_multicore_cached")

let report = BenchmarkMetricsReport(
    generated_at: ISO8601DateFormatter().string(from: Date()),
    execution_mode: execModeLabel,
    wall_clock_seconds: elapsedWallSec,
    in_distribution_unseen_test: Layer3SuiteOutput(n_cases: unseenCases.count, models: unseenEval),
    p0_coarse_t2_benchmarks: p0CoarseBenchmarks,
    layer2_validation: Layer2ValidationOutput(
        subband_geometry: geomEval,
        pentagon_external_validity: pentEval
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
try encoder.encode(report).write(to: outURL)
print("==================================================================")
print(String(format: "Completed in %.2fs (%@) -> Saved to %@", elapsedWallSec, execModeLabel as NSString, cli.outputPath as NSString))
print("==================================================================")
