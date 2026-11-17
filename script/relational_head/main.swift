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
print("Relational Head Evaluation [v4 & v4_hybrid] (Mode: \(cli.useGPU ? "GPU" : "Pure Multi-Core CPU (0 GPU VRAM)"))")
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

var splitResults: [EvalResult] = []
var predictionsA: [String: [String: Int]] = [:]
for task in tasks {
    guard let h = zHead(weights.relational.calibratedHeads, task) else {
        print("Calibrated head missing for \(task.name); run `make train` with --rel-epochs > 0")
        exit(1)
    }
    let z = task.useEarlyText ? zT2All : zT4All
    let trainCorrect = trainIndices.filter { h.predict(z[$0]) == task.label(allCases[$0]) }.count
    let preds = unseenIndices.map { h.predict(z[$0]) }
    var perCase: [String: Int] = [:]
    for (k, i) in unseenIndices.enumerated() {
        perCase[allCases[i].id] = preds[k]
    }
    predictionsA[task.name] = perCase
    let maj = majorityClass(trainIndices.map { task.label(allCases[$0]) }, classCount: task.classNames.count)
    let r = summarize(
        task: task, featurizer: "v4", testIdx: unseenIndices, preds: preds,
        majorityPreds: unseenIndices.map { _ in maj }, trainCorrect: trainCorrect, trainTotal: trainIndices.count
    )
    splitResults.append(r)
    print("[split   ][v4       ] \(task.name): train \(pct(r.trainAcc)) | unseen \(pct(r.testAcc)) | majority \(pct(r.majorityAcc))")
}

let trainGroups = trainIndices.map { allCases[$0].group }
var didCalibrateHybrid = false
for task in tasks {
    let hyb = task.useEarlyText ? hybridT2All : hybridT4All
    let trainX = trainIndices.map { hyb[$0] }
    let trainY = trainIndices.map { task.label(allCases[$0]) }
    let h: SoftmaxClassifierHead
    if let existing = hybridHead(weights.relational.calibratedHeads, task), existing.featureDim == trainX[0].count {
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
        task: task, featurizer: "v4_hybrid", testIdx: unseenIndices, preds: preds,
        majorityPreds: unseenIndices.map { _ in maj }, trainCorrect: trainCorrect, trainTotal: trainIndices.count
    )
    splitResults.append(r)
    print("[split   ][v4_hybrid] \(task.name): train \(pct(r.trainAcc)) | unseen \(pct(r.testAcc)) | majority \(pct(r.majorityAcc))")
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
    print("Persisted calibrated v4_hybrid (160D) heads into \(cli.weightsPath)")
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
            print("  [fold \(fold)] \(task.name): v4_hybrid held-out \(pct(foldAcc))")
        }
        let line = String(format: "fold %d: %d categories, train %d, test %d, %.1fs", fold, foldCats.count, foldTrain.count, foldTest.count, Date().timeIntervalSince(start))
        foldLog.append(line)
        print("[fold \(fold)] done (\(line))")
        MLX.Memory.clearCache()
    }

    let allTest = allCases.indices.filter { foldOfCategory[allCases[$0].group] != nil }
    for (fName, pMap, tcMap) in [("v4", v4Preds, v4TrainCorrect), ("v4_hybrid", hybPreds, hybTrainCorrect)] {
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
    for (featName, pMap, tcMap) in [("v4_frozen", zPreds, zTrC), ("v4_hybrid_frozen", hPreds, hTrC)] {
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

var report = "# 関係性分類の汎化評価（`v4` & `v4_hybrid` 標準構成）\n\n"
report += "- 入力: `(senderTokenIDs, receiverTokenIDs)` の2配列のみ（連結しない Dual-Stream。$T_4$ は受信側全文、$T_2$ は受信側前半）\n"
report += "- `v4`: `z_rel = [z_LL^rel(\(weights.relational.latentDim / 2)); z_HH^rel(\(weights.relational.latentDim / 2))] ∈ R^\(weights.relational.latentDim)`（`verify_latent_space` と同一テンソル・同一校正ヘッド）\n"
report += "- `v4_hybrid`: `z_rel` + 非線形サブバンド幾何(8) + トピック置換不変エコー・Pentagon(8) + 非語彙対話リズム(16) = `\(weights.relational.latentDim + 32)` 次元（`weights.safetensors` 内蔵 Group-CV 校正ヘッド）\n\n"
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

try report.write(to: URL(fileURLWithPath: cli.reportPath), atomically: true, encoding: .utf8)
let headsData = try JSONEncoder().encode(weights.relational.calibratedHeads)
try headsData.write(to: URL(fileURLWithPath: cli.headsPath))
print("Saved report to \(cli.reportPath) and heads to \(cli.headsPath)")
