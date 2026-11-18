import Foundation
import MLX
import Sashiba

struct RelationalHeadCLI {
    var dataPath: String = ".tmp"
    var weightsPath: String = ".tmp/weights.safetensors"
    var aeWeightsPath: String = ".tmp/weights_ae.safetensors"
    var headsPath: String = ".tmp/relational_heads.json"
    var reportPath: String = ".tmp/relational_head_report.md"
    var predictionsPath: String = ".tmp/relational_head_predictions_A.json"
    var maxSeqLen: Int = 512
    var steps: Int = 350
    var foldEpochs: Int = 15
    var relBatch: Int = 32
    var relLR: Float = 1.4e-3
    var skipFolds: Bool = true
    var useGPU: Bool = false
    var recalibrate: Bool = false

    static func parse(_ args: [String]) -> RelationalHeadCLI {
        var cli = RelationalHeadCLI()
        var i = 1
        while i < args.count {
            switch args[i] {
            case "-d", "--data":
                i += 1
                if i < args.count { cli.dataPath = args[i] }
            case "-w", "--weights":
                i += 1
                if i < args.count { cli.weightsPath = args[i] }
            case "--ae-weights":
                i += 1
                if i < args.count { cli.aeWeightsPath = args[i] }
            case "--heads":
                i += 1
                if i < args.count { cli.headsPath = args[i] }
            case "-o", "--output":
                i += 1
                if i < args.count { cli.reportPath = args[i] }
            case "--predictions":
                i += 1
                if i < args.count { cli.predictionsPath = args[i] }
            case "--steps":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.steps = max(10, v) }
            case "--fold-epochs":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.foldEpochs = max(1, v) }
            case "--skip-folds":
                cli.skipFolds = true
            case "--retrain-folds":
                cli.skipFolds = false
                cli.useGPU = true
            case "--gpu":
                cli.useGPU = true
            case "--cpu":
                cli.useGPU = false
            case "--recalibrate":
                cli.recalibrate = true
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

struct HeadTask {
    let name: String
    let useEarlyText: Bool
    let classNames: [String]
    let label: (RelationalDialogueCase) -> Int
}

let archetypeNames = RelationalArchetype.allCases.map(\.rawValue)
let topologyNames = RelationalTopology.allCases.map(\.rawValue)

let tasks: [HeadTask] = [
    HeadTask(name: "archetype_t4", useEarlyText: false, classNames: archetypeNames) { RelationalPipeline.archetypeIndex($0.expectedArchetype) },
    HeadTask(name: "topology_t4", useEarlyText: false, classNames: topologyNames) { RelationalPipeline.topologyIndex($0.expectedTopology) },
    HeadTask(name: "archetype_t2", useEarlyText: true, classNames: archetypeNames) { RelationalPipeline.archetypeIndex($0.expectedArchetypeT2) },
    HeadTask(name: "topology_t2", useEarlyText: true, classNames: topologyNames) { RelationalPipeline.topologyIndex($0.expectedTopologyT2) }
]

let cli = RelationalHeadCLI.parse(CommandLine.arguments)
if cli.useGPU {
    MLX.Memory.cacheLimit = 128 * 1024 * 1024
    MLX.Memory.clearCache()
}
let config = DatasetConfiguration.load(from: cli.dataPath)
let trainCasesAll = config.relationalDialogueCases.filter { !$0.isZeroShotUnseen }
let unseenCasesAll = config.relationalDialogueCases.filter { $0.isZeroShotUnseen }
guard !trainCasesAll.isEmpty && !unseenCasesAll.isEmpty else {
    print("No train / unseen relational cases found in \(cli.dataPath)")
    exit(1)
}

var weights = try SWTHawkWeights.load(from: URL(fileURLWithPath: cli.weightsPath))
let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)
let hybridExtractor = RelationalHybridFeatureExtractor(vocabulary: vocab)
let mainModel: MLXSWTHawkAutoencoder? = cli.useGPU ? MLXSWTHawkAutoencoder(weights: weights) : nil

let allCases = trainCasesAll + unseenCasesAll
let seqLen = allCases
    .flatMap { [$0.hostText, $0.partnerEarlyText, $0.partnerCurrentText] }
    .map { min(cli.maxSeqLen, vocab.encode($0).count) }
    .reduce(4, max)

print("==================================================================")
print("Relational Head Evaluation [v5 & v5_hybrid] (Mode: \(cli.useGPU ? "GPU" : "Pure Multi-Core CPU (0 GPU VRAM)"))")
print("==================================================================")
print("Train cases: \(trainCasesAll.count) | Unseen cases: \(unseenCasesAll.count) | seqLen: \(seqLen)")

let hostEarlyTok = allCases.map { RelationalPipeline.encodeTokens($0.hostEarlyText, vocab: vocab, time: seqLen) }
let hostTok = allCases.map { RelationalPipeline.encodeTokens($0.hostText, vocab: vocab, time: seqLen) }
let earlyTok = allCases.map { RelationalPipeline.encodeTokens($0.partnerEarlyText, vocab: vocab, time: seqLen) }
let currentTok = allCases.map { RelationalPipeline.encodeTokens($0.partnerCurrentText, vocab: vocab, time: seqLen) }

struct EvalResult {
    let task: String
    let featurizer: String
    let trainAcc: Float
    let testAcc: Float
    let majorityAcc: Float
    let correct: Int
    let total: Int
    let perClass: [(name: String, correct: Int, total: Int)]
}

func summarize(task: HeadTask, featurizer: String, testIdx: [Int], preds: [Int], majorityPreds: [Int], trainCorrect: Int, trainTotal: Int) -> EvalResult {
    let ys = testIdx.map { task.label(allCases[$0]) }
    let correct = zip(preds, ys).filter { $0 == $1 }.count
    let majorityCorrect = zip(majorityPreds, ys).filter { $0 == $1 }.count
    var perClass: [(name: String, correct: Int, total: Int)] = []
    for c in 0..<task.classNames.count {
        let idx = ys.indices.filter { ys[$0] == c }
        guard !idx.isEmpty else { continue }
        perClass.append((task.classNames[c], idx.filter { preds[$0] == c }.count, idx.count))
    }
    return EvalResult(
        task: task.name,
        featurizer: featurizer,
        trainAcc: Float(trainCorrect) / Float(max(1, trainTotal)),
        testAcc: Float(correct) / Float(max(1, ys.count)),
        majorityAcc: Float(majorityCorrect) / Float(max(1, ys.count)),
        correct: correct,
        total: ys.count,
        perClass: perClass
    )
}

func majorityClass(_ labels: [Int], classCount: Int) -> Int {
    var counts = [Int](repeating: 0, count: classCount)
    for y in labels { counts[y] += 1 }
    return counts.firstIndex(of: counts.max() ?? 0) ?? 0
}

func zHead(_ heads: RelationalCalibratedHeads, _ task: HeadTask) -> SoftmaxClassifierHead? {
    switch task.name {
    case "archetype_t4": return heads.archetypeT4
    case "topology_t4": return heads.topologyT4
    case "archetype_t2": return heads.archetypeT2
    default: return heads.topologyT2
    }
}

func hybridHead(_ heads: RelationalCalibratedHeads, _ task: HeadTask) -> SoftmaxClassifierHead? {
    switch task.name {
    case "archetype_t4": return heads.hybridArchetypeT4
    case "topology_t4": return heads.hybridTopologyT4
    case "archetype_t2": return heads.hybridArchetypeT2
    default: return heads.hybridTopologyT2
    }
}

let trainIndices = allCases.indices.filter { !allCases[$0].isZeroShotUnseen }
let unseenIndices = allCases.indices.filter { allCases[$0].isZeroShotUnseen }
let pct = { (v: Float) in String(format: "%6.2f%%", v * 100.0) }

// ------------------------------------------------------------------
// (A) Scenario split: v4 (z_rel 128D) & v4_hybrid (160D)
// ------------------------------------------------------------------
let zT2All: [[Float]]
let zT4All: [[Float]]
let wAttr = try? FileManager.default.attributesOfItem(atPath: cli.weightsPath)
let wSize = (wAttr?[.size] as? NSNumber)?.int64Value ?? 0
let wMtime = Int64((wAttr?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
let zCacheDir = (cli.dataPath as NSString).appendingPathComponent("validation/zrel_cache")
try? FileManager.default.createDirectory(atPath: zCacheDir, withIntermediateDirectories: true)
let zCacheT2 = (zCacheDir as NSString).appendingPathComponent("relhead_t2_\(allCases.count)_\(seqLen)_\(wSize)_\(wMtime).bin")
let zCacheT4 = (zCacheDir as NSString).appendingPathComponent("relhead_t4_\(allCases.count)_\(seqLen)_\(wSize)_\(wMtime).bin")

func loadZCache(_ path: String, rows: Int, dim: Int) -> [[Float]]? {
    let expectedBytes = rows * dim * MemoryLayout<Float>.size
    guard let d = try? Data(contentsOf: URL(fileURLWithPath: path)), d.count == expectedBytes else { return nil }
    var out: [[Float]] = []
    out.reserveCapacity(rows)
    d.withUnsafeBytes { raw in
        let base = raw.bindMemory(to: Float.self).baseAddress!
        for r in 0..<rows {
            out.append(Array(UnsafeBufferPointer(start: base + r * dim, count: dim)))
        }
    }
    return out
}
func saveZCache(_ path: String, matrix: [[Float]]) {
    let flat = matrix.flatMap { $0 }
    let d = flat.withUnsafeBufferPointer { Data(buffer: $0) }
    try? d.write(to: URL(fileURLWithPath: path), options: .atomic)
}

let relDim = weights.relational.latentDim
if !cli.useGPU, let c2 = loadZCache(zCacheT2, rows: allCases.count, dim: relDim), let c4 = loadZCache(zCacheT4, rows: allCases.count, dim: relDim) {
    zT2All = c2
    zT4All = c4
} else if let m = mainModel {
    zT2All = RelationalPipeline.encodeZRel(model: m, senders: hostEarlyTok, receivers: earlyTok, time: seqLen)
    zT4All = RelationalPipeline.encodeZRel(model: m, senders: hostTok, receivers: currentTok, time: seqLen)
} else {
    zT2All = DualStreamRelationalEngine.encodeZRelCPUParallel(senders: hostEarlyTok, receivers: earlyTok, time: seqLen, weights: weights, chunkSize: 16)
    zT4All = DualStreamRelationalEngine.encodeZRelCPUParallel(senders: hostTok, receivers: currentTok, time: seqLen, weights: weights, chunkSize: 16)
    saveZCache(zCacheT2, matrix: zT2All)
    saveZCache(zCacheT4, matrix: zT4All)
}

let hybridT2All = allCases.indices.map { i in
    hybridExtractor.extract(senderTokens: hostEarlyTok[i], receiverTokens: earlyTok[i], zRel: zT2All[i], weights: weights)
}
let hybridT4All = allCases.indices.map { i in
    hybridExtractor.extract(senderTokens: hostTok[i], receiverTokens: currentTok[i], zRel: zT4All[i], weights: weights)
}

let trainGroups = trainIndices.map { allCases[$0].group }
var didCalibrateHybrid = false
var splitResults: [EvalResult] = []
var predictionsA: [String: [String: Int]] = [:]
for task in tasks {
    let z = task.useEarlyText ? zT2All : zT4All
    let h: SoftmaxClassifierHead
    if !cli.recalibrate, let existing = zHead(weights.relational.calibratedHeads, task) {
        h = existing
    } else if cli.recalibrate {
        let trX = trainIndices.map { z[$0] }
        let trY = trainIndices.map { task.label(allCases[$0]) }
        h = SoftmaxClassifierHead.trainGroupCV(features: trX, labels: trY, groups: trainGroups, classCount: task.classNames.count, steps: cli.steps)
        switch task.name {
        case "archetype_t4": weights.relational.calibratedHeads.archetypeT4 = h
        case "topology_t4": weights.relational.calibratedHeads.topologyT4 = h
        case "archetype_t2": weights.relational.calibratedHeads.archetypeT2 = h
        default: weights.relational.calibratedHeads.topologyT2 = h
        }
        didCalibrateHybrid = true
    } else {
        print("Calibrated head missing for \(task.name); run `make train` with --rel-epochs > 0")
        exit(1)
    }
    let trainCorrect = trainIndices.filter { h.predict(z[$0]) == task.label(allCases[$0]) }.count
    let preds = unseenIndices.map { h.predict(z[$0]) }
    var perCase: [String: Int] = [:]
    for (k, i) in unseenIndices.enumerated() {
        perCase[allCases[i].id] = preds[k]
    }
    predictionsA[task.name] = perCase
    let maj = majorityClass(trainIndices.map { task.label(allCases[$0]) }, classCount: task.classNames.count)
    let r = summarize(
        task: task, featurizer: "v5", testIdx: unseenIndices, preds: preds,
        majorityPreds: unseenIndices.map { _ in maj }, trainCorrect: trainCorrect, trainTotal: trainIndices.count
    )
    splitResults.append(r)
    print("[split   ][v5       ] \(task.name): train \(pct(r.trainAcc)) | unseen \(pct(r.testAcc)) | majority \(pct(r.majorityAcc))")
}

for task in tasks {
    let hyb = task.useEarlyText ? hybridT2All : hybridT4All
    let trainX = trainIndices.map { hyb[$0] }
    let trainY = trainIndices.map { task.label(allCases[$0]) }
    let h: SoftmaxClassifierHead
    if !cli.recalibrate, let existing = hybridHead(weights.relational.calibratedHeads, task), existing.featureDim == trainX[0].count {
        h = existing
    } else {
        h = SoftmaxClassifierHead.trainGroupCV(features: trainX, labels: trainY, groups: trainGroups, classCount: task.classNames.count, steps: cli.steps)
        switch task.name {
        case "archetype_t4": weights.relational.calibratedHeads.hybridArchetypeT4 = h
        case "topology_t4": weights.relational.calibratedHeads.hybridTopologyT4 = h
        case "archetype_t2": weights.relational.calibratedHeads.hybridArchetypeT2 = h
        default: weights.relational.calibratedHeads.hybridTopologyT2 = h
        }
        didCalibrateHybrid = true
    }
    let trainCorrect = zip(trainX, trainY).filter { h.predict($0) == $1 }.count
    let preds = unseenIndices.map { h.predict(hyb[$0]) }
    let maj = majorityClass(trainY, classCount: task.classNames.count)
    let r = summarize(
        task: task, featurizer: "v5_hybrid", testIdx: unseenIndices, preds: preds,
        majorityPreds: unseenIndices.map { _ in maj }, trainCorrect: trainCorrect, trainTotal: trainIndices.count
    )
    splitResults.append(r)
    print("[split   ][v5_hybrid] \(task.name): train \(pct(r.trainAcc)) | unseen \(pct(r.testAcc)) | majority \(pct(r.majorityAcc))")
}

// ------------------------------------------------------------------
// (A2) v5 Emotional Gradation, 10-way Trajectory & +30d/+90d Retention
// ------------------------------------------------------------------
let trajCases = RelationalTrajectoryType.allCases
let trajTextAll = allCases.indices.map { i in
    RelationalHybridFeatureExtractor.extractTrajectoryTextFeatures(hybridT2: hybridT2All[i], hybridT4: hybridT4All[i], zT2: zT2All[i], zT4: zT4All[i])
}
let trajFullAll = allCases.indices.map { i in
    RelationalHybridFeatureExtractor.extractTrajectoryFullFeatures(
        hybridT2: hybridT2All[i], hybridT4: hybridT4All[i], zT2: zT2All[i], zT4: zT4All[i], temporal: allCases[i].temporalContext
    )
}

@MainActor
func resolveOrTrainHead(_ existing: SoftmaxClassifierHead?, _ bank: [[Float]], _ yOf: (RelationalDialogueCase) -> Int, _ classes: Int) -> SoftmaxClassifierHead {
    let trX = trainIndices.map { bank[$0] }
    let trY = trainIndices.map { yOf(allCases[$0]) }
    if !cli.recalibrate, let ex = existing, ex.featureDim == trX[0].count && ex.classCount == classes { return ex }
    didCalibrateHybrid = true
    return SoftmaxClassifierHead.trainGroupCV(features: trX, labels: trY, groups: trainGroups, classCount: classes, steps: cli.steps)
}

weights.relational.calibratedHeads.hybridAffectionT4 = resolveOrTrainHead(weights.relational.calibratedHeads.hybridAffectionT4, hybridT4All, { $0.emotionT4.affection }, 5)
weights.relational.calibratedHeads.hybridFrictionT4 = resolveOrTrainHead(weights.relational.calibratedHeads.hybridFrictionT4, hybridT4All, { $0.emotionT4.friction }, 5)
weights.relational.calibratedHeads.hybridArousalT4 = resolveOrTrainHead(weights.relational.calibratedHeads.hybridArousalT4, hybridT4All, { SoftmaxClassifierHead.arousalBin($0.emotionT4.arousal) }, 5)
weights.relational.calibratedHeads.hybridAffectionT2 = resolveOrTrainHead(weights.relational.calibratedHeads.hybridAffectionT2, hybridT2All, { $0.emotionT2.affection }, 5)
weights.relational.calibratedHeads.hybridFrictionT2 = resolveOrTrainHead(weights.relational.calibratedHeads.hybridFrictionT2, hybridT2All, { $0.emotionT2.friction }, 5)
weights.relational.calibratedHeads.hybridArousalT2 = resolveOrTrainHead(weights.relational.calibratedHeads.hybridArousalT2, hybridT2All, { SoftmaxClassifierHead.arousalBin($0.emotionT2.arousal) }, 5)
weights.relational.calibratedHeads.trajectoryTextHead = resolveOrTrainHead(weights.relational.calibratedHeads.trajectoryTextHead, trajTextAll, { trajCases.firstIndex(of: $0.trajectory.type) ?? 0 }, trajCases.count)
weights.relational.calibratedHeads.trajectoryFullHead = resolveOrTrainHead(weights.relational.calibratedHeads.trajectoryFullHead, trajFullAll, { trajCases.firstIndex(of: $0.trajectory.type) ?? 0 }, trajCases.count)
weights.relational.calibratedHeads.retained30dHead = resolveOrTrainHead(weights.relational.calibratedHeads.retained30dHead, trajFullAll, { $0.trajectory.retained30d ? 1 : 0 }, 2)
weights.relational.calibratedHeads.retained90dHead = resolveOrTrainHead(weights.relational.calibratedHeads.retained90dHead, trajFullAll, { $0.trajectory.retained90d ? 1 : 0 }, 2)

@MainActor
func evalOrdinal5(_ name: String, _ head: SoftmaxClassifierHead, _ bank: [[Float]], _ yOf: (RelationalDialogueCase) -> Int) -> String {
    let ys = unseenIndices.map { yOf(allCases[$0]) }
    let ps = unseenIndices.map { head.predict(bank[$0]) }
    let evs = unseenIndices.map { head.expectedValue(bank[$0]) }
    let exact = Float(zip(ps, ys).filter { $0 == $1 }.count) / Float(max(1, ys.count))
    let within1 = Float(zip(ps, ys).filter { abs($0 - $1) <= 1 }.count) / Float(max(1, ys.count))
    let mae = zip(evs, ys).reduce(Float(0.0)) { $0 + abs($1.0 - Float($1.1)) } / Float(max(1, ys.count))
    print(String(format: "[emotion ][v5_hybrid] %-13@: exact %@ | ±1-lvl %@ | MAE %.3f", name as NSString, pct(exact) as NSString, pct(within1) as NSString, mae))
    return "| `\(name)` | **\(String(format: "%.2f%%", exact * 100.0))** | **\(String(format: "%.2f%%", within1 * 100.0))** | `\(String(format: "%.3f", mae))` |\n"
}

var v5ExtraSection = "\n## (C) 感情強度グラデーション・離脱軌跡・継続予測（unseen_test）\n\n"
v5ExtraSection += "### 感情強度グラデーション（`affection 0..4` × `friction 0..4` × `arousal 0..1`）\n\n"
v5ExtraSection += "| 軸 | 完全一致率 | ±1段階以内一致率 | 期待値 MAE |\n| :--- | ---: | ---: | ---: |\n"
v5ExtraSection += evalOrdinal5("affection_t2", weights.relational.calibratedHeads.hybridAffectionT2!, hybridT2All) { $0.emotionT2.affection }
v5ExtraSection += evalOrdinal5("friction_t2", weights.relational.calibratedHeads.hybridFrictionT2!, hybridT2All) { $0.emotionT2.friction }
v5ExtraSection += evalOrdinal5("affection_t4", weights.relational.calibratedHeads.hybridAffectionT4!, hybridT4All) { $0.emotionT4.affection }
v5ExtraSection += evalOrdinal5("friction_t4", weights.relational.calibratedHeads.hybridFrictionT4!, hybridT4All) { $0.emotionT4.friction }
do {
    let ar2Err = unseenIndices.reduce(Float(0.0)) { acc, i in
        let p = weights.relational.calibratedHeads.hybridArousalT2!.expectedValue(hybridT2All[i], binValues: SoftmaxClassifierHead.arousalBinCenters)
        return acc + abs(p - allCases[i].emotionT2.arousal)
    } / Float(max(1, unseenIndices.count))
    let ar4Err = unseenIndices.reduce(Float(0.0)) { acc, i in
        let p = weights.relational.calibratedHeads.hybridArousalT4!.expectedValue(hybridT4All[i], binValues: SoftmaxClassifierHead.arousalBinCenters)
        return acc + abs(p - allCases[i].emotionT4.arousal)
    } / Float(max(1, unseenIndices.count))
    print(String(format: "[emotion ][v5_hybrid] arousal MAE   : T2 %.3f | T4 %.3f", ar2Err, ar4Err))
    v5ExtraSection += "| `arousal_t2 (0..1)` | - | - | `\(String(format: "%.3f", ar2Err))` |\n"
    v5ExtraSection += "| `arousal_t4 (0..1)` | - | - | `\(String(format: "%.3f", ar4Err))` |\n"
}

do {
    let yTraj = unseenIndices.map { trajCases.firstIndex(of: allCases[$0].trajectory.type) ?? 0 }
    let pText = unseenIndices.map { weights.relational.calibratedHeads.trajectoryTextHead!.predict(trajTextAll[$0]) }
    let pFull = unseenIndices.map { weights.relational.calibratedHeads.trajectoryFullHead!.predict(trajFullAll[$0]) }
    let accText = Float(zip(pText, yTraj).filter { $0 == $1 }.count) / Float(max(1, yTraj.count))
    let accFull = Float(zip(pFull, yTraj).filter { $0 == $1 }.count) / Float(max(1, yTraj.count))
    let accR30 = Float(unseenIndices.filter { (weights.relational.calibratedHeads.retained30dHead!.predict(trajFullAll[$0]) == 1) == allCases[$0].trajectory.retained30d }.count) / Float(max(1, unseenIndices.count))
    let accR90 = Float(unseenIndices.filter { (weights.relational.calibratedHeads.retained90dHead!.predict(trajFullAll[$0]) == 1) == allCases[$0].trajectory.retained90d }.count) / Float(max(1, unseenIndices.count))
    print("[traj 10c][text_only ] unseen \(pct(accText)) | [text+temporal] unseen \(pct(accFull)) (Δ \(String(format: "%+.2f pt", (accFull - accText) * 100.0)))")
    print("[retain  ][v5_hybrid ] retained_30d \(pct(accR30)) | retained_90d \(pct(accR90))")
    v5ExtraSection += "\n### 10分類・距離感＆離脱軌跡（テキスト単独 vs 相対基準線つき時間文脈併用）\n\n"
    v5ExtraSection += "- `v5_text_only (336D)`: **\(String(format: "%.2f%%", accText * 100.0))**\n"
    v5ExtraSection += "- `v5_hybrid_full (350D = text + relative temporal)`: **\(String(format: "%.2f%%", accFull * 100.0))**（時間文脈併用効果 `\(String(format: "%+.2f pt", (accFull - accText) * 100.0))`）\n"
    v5ExtraSection += "- `+30d` 継続予測精度: **\(String(format: "%.2f%%", accR30 * 100.0))** / `+90d` 継続予測精度: **\(String(format: "%.2f%%", accR90 * 100.0))**\n"
}

if didCalibrateHybrid {
    try weights.save(to: URL(fileURLWithPath: cli.weightsPath))
    saveZCache(
        (zCacheDir as NSString).appendingPathComponent("relhead_t2_\(allCases.count)_\(seqLen)_" +
            "\(((try? FileManager.default.attributesOfItem(atPath: cli.weightsPath)[.size] as? NSNumber)?.int64Value ?? 0))_" +
            "\(Int64(((try? FileManager.default.attributesOfItem(atPath: cli.weightsPath)[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0))).bin"),
        matrix: zT2All
    )
    saveZCache(
        (zCacheDir as NSString).appendingPathComponent("relhead_t4_\(allCases.count)_\(seqLen)_" +
            "\(((try? FileManager.default.attributesOfItem(atPath: cli.weightsPath)[.size] as? NSNumber)?.int64Value ?? 0))_" +
            "\(Int64(((try? FileManager.default.attributesOfItem(atPath: cli.weightsPath)[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0))).bin"),
        matrix: zT4All
    )
    print("Persisted calibrated v5_hybrid heads into \(cli.weightsPath)")
}

let predData = try JSONEncoder().encode(predictionsA)
try predData.write(to: URL(fileURLWithPath: cli.predictionsPath))

// ------------------------------------------------------------------
// (B) Stratified Group 6-Fold CV: v4 & v4_hybrid
// ------------------------------------------------------------------
let foldCount = 6
let categories = Array(Set(allCases.map(\.group))).sorted()
var signatureOfCategory: [String: String] = [:]
for c in categories {
    if let rc = allCases.first(where: { $0.group == c }) {
        signatureOfCategory[c] = "\(rc.expectedArchetypeT2.rawValue)|\(rc.expectedTopologyT2.rawValue)|\(rc.expectedArchetype.rawValue)|\(rc.expectedTopology.rawValue)"
    }
}
let signatureGroups = Dictionary(grouping: categories) { signatureOfCategory[$0] ?? "" }
var foldOfCategory: [String: Int] = [:]
var rr = 0
for sig in signatureGroups.keys.sorted() {
    for c in (signatureGroups[sig] ?? []).sorted() {
        foldOfCategory[c] = rr % foldCount
        rr += 1
    }
}

var missingByTask: [String: Int] = [:]
for task in tasks {
    var missing = 0
    for fold in 0..<foldCount {
        let foldTrain = trainIndices.filter { foldOfCategory[allCases[$0].group] != fold }
        let foldTest = allCases.indices.filter { foldOfCategory[allCases[$0].group] == fold }
        let labels = Set(foldTrain.map { task.label(allCases[$0]) })
        missing += foldTest.filter { !labels.contains(task.label(allCases[$0])) }.count
    }
    missingByTask[task.name] = missing
    print("[folds] \(task.name): label unseen in training = \(missing)")
}

var holdoutResults: [EvalResult] = []
var foldLog: [String] = []
if !cli.skipFolds {
    let aeWeights = try SWTHawkWeights.load(from: URL(fileURLWithPath: cli.aeWeightsPath))
    var v4Preds: [String: [Int: Int]] = [:]
    var hybPreds: [String: [Int: Int]] = [:]
    var v4TrainCorrect: [String: Int] = [:]
    var hybTrainCorrect: [String: Int] = [:]
    var trainTotal: [String: Int] = [:]
    var majorityPredsByTask: [String: [Int: Int]] = [:]

    for fold in 0..<foldCount {
        let foldTrain = trainIndices.filter { foldOfCategory[allCases[$0].group] != fold }
        let foldTest = allCases.indices.filter { foldOfCategory[allCases[$0].group] == fold }
        let foldCats = categories.filter { foldOfCategory[$0] == fold }
        print("------------------------------------------------------------------")
        print("[fold \(fold)] held-out categories: \(foldCats.count) | train scenarios: \(foldTrain.count) | test scenarios: \(foldTest.count)")
        let start = Date()

        let model = MLXSWTHawkAutoencoder(weights: aeWeights)
        RelationalPipeline.trainRelationalStage(
            model: model,
            trainCases: foldTrain.map { allCases[$0] },
            vocab: vocab,
            time: seqLen,
            epochs: cli.foldEpochs,
            batchSize: cli.relBatch,
            learningRate: cli.relLR,
            seed: UInt64(9000 + fold),
            log: { print("  [fold \(fold)] " + $0) }
        )
        let fw = model.exportWeights()
        let evalIdx = foldTrain + foldTest
        let z2 = RelationalPipeline.encodeZRel(model: model, senders: evalIdx.map { hostEarlyTok[$0] }, receivers: evalIdx.map { earlyTok[$0] }, time: seqLen)
        let z4 = RelationalPipeline.encodeZRel(model: model, senders: evalIdx.map { hostTok[$0] }, receivers: evalIdx.map { currentTok[$0] }, time: seqLen)
        let hyb2 = evalIdx.enumerated().map { (k, idx) in
            hybridExtractor.extract(senderTokens: hostEarlyTok[idx], receiverTokens: earlyTok[idx], zRel: z2[k], weights: fw)
        }
        let hyb4 = evalIdx.enumerated().map { (k, idx) in
            hybridExtractor.extract(senderTokens: hostTok[idx], receiverTokens: currentTok[idx], zRel: z4[k], weights: fw)
        }
        var row: [Int: Int] = [:]
        for (k, i) in evalIdx.enumerated() { row[i] = k }

        for task in tasks {
            guard let hz = zHead(fw.relational.calibratedHeads, task),
                  let hh = hybridHead(fw.relational.calibratedHeads, task) else { continue }
            let z = task.useEarlyText ? z2 : z4
            let hyb = task.useEarlyText ? hyb2 : hyb4
            v4TrainCorrect[task.name, default: 0] += foldTrain.filter { hz.predict(z[row[$0]!]) == task.label(allCases[$0]) }.count
            hybTrainCorrect[task.name, default: 0] += foldTrain.filter { hh.predict(hyb[row[$0]!]) == task.label(allCases[$0]) }.count
            trainTotal[task.name, default: 0] += foldTrain.count
            let maj = majorityClass(foldTrain.map { task.label(allCases[$0]) }, classCount: task.classNames.count)
            for i in foldTest {
                v4Preds[task.name, default: [:]][i] = hz.predict(z[row[i]!])
                hybPreds[task.name, default: [:]][i] = hh.predict(hyb[row[i]!])
                majorityPredsByTask[task.name, default: [:]][i] = maj
            }
            let foldAcc = Float(foldTest.filter { hybPreds[task.name]![$0] == task.label(allCases[$0]) }.count) / Float(max(1, foldTest.count))
            print("  [fold \(fold)] \(task.name): v5_hybrid held-out \(pct(foldAcc))")
        }
        let line = String(format: "fold %d: %d categories, train %d, test %d, %.1fs", fold, foldCats.count, foldTrain.count, foldTest.count, Date().timeIntervalSince(start))
        foldLog.append(line)
        print("[fold \(fold)] done (\(line))")
        MLX.Memory.clearCache()
    }

    let allTest = allCases.indices.filter { foldOfCategory[allCases[$0].group] != nil }
    for (fName, pMap, tcMap) in [("v5", v4Preds, v4TrainCorrect), ("v5_hybrid", hybPreds, hybTrainCorrect)] {
        for task in tasks {
            let r = summarize(
                task: task, featurizer: fName, testIdx: allTest,
                preds: allTest.map { pMap[task.name]?[$0] ?? 0 },
                majorityPreds: allTest.map { majorityPredsByTask[task.name]?[$0] ?? 0 },
                trainCorrect: tcMap[task.name] ?? 0, trainTotal: trainTotal[task.name] ?? 1
            )
            holdoutResults.append(r)
            print("[holdout ][\(fName)] \(task.name): held-out categories \(pct(r.testAcc)) | majority \(pct(r.majorityAcc))")
        }
    }
} else {
    let startFast = Date()
    var zPreds: [String: [Int: Int]] = [:]
    var hPreds: [String: [Int: Int]] = [:]
    var majPreds: [String: [Int: Int]] = [:]
    var zTrC: [String: Int] = [:], hTrC: [String: Int] = [:], trTot: [String: Int] = [:]

    for fold in 0..<foldCount {
        let foldTrain = trainIndices.filter { foldOfCategory[allCases[$0].group] != fold }
        let foldTest = unseenIndices.filter { foldOfCategory[allCases[$0].group] == fold }
        guard !foldTrain.isEmpty && !foldTest.isEmpty else { continue }
        let fGroups = foldTrain.map { allCases[$0].group }
        for task in tasks {
            let zBank = task.useEarlyText ? zT2All : zT4All
            let hBank = task.useEarlyText ? hybridT2All : hybridT4All
            let yTr = foldTrain.map { task.label(allCases[$0]) }
            let maj = majorityClass(yTr, classCount: task.classNames.count)
            trTot[task.name, default: 0] += foldTrain.count

            let hz = SoftmaxClassifierHead.trainGroupCV(features: foldTrain.map { zBank[$0] }, labels: yTr, groups: fGroups, classCount: task.classNames.count, steps: 200)
            let hh = SoftmaxClassifierHead.trainGroupCV(features: foldTrain.map { hBank[$0] }, labels: yTr, groups: fGroups, classCount: task.classNames.count, steps: 200)

            zTrC[task.name, default: 0] += foldTrain.filter { hz.predict(zBank[$0]) == task.label(allCases[$0]) }.count
            hTrC[task.name, default: 0] += foldTrain.filter { hh.predict(hBank[$0]) == task.label(allCases[$0]) }.count

            for i in foldTest {
                zPreds[task.name, default: [:]][i] = hz.predict(zBank[i])
                hPreds[task.name, default: [:]][i] = hh.predict(hBank[i])
                majPreds[task.name, default: [:]][i] = maj
            }
        }
    }
    let unseenHoldoutIdx = unseenIndices.filter { foldOfCategory[allCases[$0].group] != nil }
    for (featName, pMap, tcMap) in [("v5_frozen", zPreds, zTrC), ("v5_hybrid_frozen", hPreds, hTrC)] {
        for task in tasks {
            let r = summarize(
                task: task, featurizer: featName, testIdx: unseenHoldoutIdx,
                preds: unseenHoldoutIdx.map { pMap[task.name]?[$0] ?? 0 },
                majorityPreds: unseenHoldoutIdx.map { majPreds[task.name]?[$0] ?? 0 },
                trainCorrect: tcMap[task.name] ?? 0, trainTotal: trTot[task.name] ?? 1
            )
            holdoutResults.append(r)
            print("[fast-cv ][\(featName)] \(task.name): unseen-cat \(pct(r.testAcc)) | majority \(pct(r.majorityAcc))")
        }
    }
    foldLog.append(String(format: "Fast pure-CPU frozen-feature Group 6-Fold CV completed in %.2fs (0 GPU retraining; pass --retrain-folds for heavy GPU E2E fold retraining).", Date().timeIntervalSince(startFast)))
}

func resultTable(_ rs: [EvalResult], testLabel: String) -> String {
    var t = "| モデル構成 | タスク | 学習精度 | \(testLabel) | 多数派クラス基準 |\n"
    t += "| :--- | :--- | ---: | ---: | ---: |\n"
    for r in rs {
        t += "| `\(r.featurizer)` | `\(r.task)` | \(String(format: "%.2f", r.trainAcc * 100.0))% | **\(String(format: "%.2f", r.testAcc * 100.0))%** (\(r.correct)/\(r.total)) | \(String(format: "%.2f", r.majorityAcc * 100.0))% |\n"
    }
    return t
}

func perClassList(_ rs: [EvalResult]) -> String {
    var s = ""
    for r in rs {
        let parts = r.perClass.map { "`\($0.name)` \($0.correct)/\($0.total)" }.joined(separator: ", ")
        s += "- `\(r.featurizer)` / `\(r.task)`: \(parts)\n"
    }
    return s
}

var report = "# 関係性分類・感情グラデーション・離脱軌跡の汎化評価（`v5` & `v5_hybrid` 標準構成）\n\n"
report += "- 入力: `(senderTokenIDs, receiverTokenIDs)` の2配列のみ（連結しない Dual-Stream。$T_4$ は受信側全文、$T_2$ は受信側前半）\n"
report += "- `v5`: `z_rel = [z_LL^rel(\(weights.relational.latentDim / 2)); z_HH^rel(\(weights.relational.latentDim / 2))] ∈ R^\(weights.relational.latentDim)`（`verify_latent_space` と同一テンソル・同一校正ヘッド）\n"
report += "- `v5_hybrid`: `z_rel` + 非線形サブバンド幾何(8) + トピック置換不変エコー・Pentagon(8) + 非語彙対話リズム(16) = `\(weights.relational.latentDim + 32)` 次元、および軌跡用 `336D / 350D` 特徴\n\n"
report += "## (A) シナリオ分割: train → unseen_test\n\n"
report += "学習 `\(trainIndices.count)` 件、評価 `\(unseenIndices.count)` 件。\n\n"
report += resultTable(splitResults, testLabel: "未学習精度")
report += "\n### クラス別正解数\n\n"
report += perClassList(splitResults)
report += "\n## (B) Stratified Group \(foldCount)-Fold（カテゴリ単位の取り分け評価）\n\n"
report += "- モード: `\(cli.skipFolds ? "高速 CPU 固定特徴ヘッド Group-CV（評価対象は取り分けカテゴリの unseen_test のみ、GPU 再学習 0 回）" : "End-to-End エンコーダ再学習（--retrain-folds）")`\n"
report += "- 学習側に同じラベルが1件も無かった評価ケース数: "
report += tasks.map { "`\($0.name)` \(missingByTask[$0.name] ?? -1)" }.joined(separator: ", ")
report += "\n\n"
report += resultTable(holdoutResults, testLabel: "未見カテゴリ精度")
report += "\n### フォールド実行ログ\n\n"
report += foldLog.map { "- \($0)" }.joined(separator: "\n")
report += "\n\n### クラス別正解数\n\n"
report += perClassList(holdoutResults)
report += v5ExtraSection

try report.write(to: URL(fileURLWithPath: cli.reportPath), atomically: true, encoding: .utf8)
let headsData = try JSONEncoder().encode(weights.relational.calibratedHeads)
try headsData.write(to: URL(fileURLWithPath: cli.headsPath))
print("Saved report to \(cli.reportPath) and heads to \(cli.headsPath)")
