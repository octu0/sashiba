import Foundation
import MLX
import Sashiba

struct RelationalHeadCLI {
    var dataPath: String = ".tmp"
    var weightsPath: String = ".tmp/weights.safetensors"
    var aeWeightsPath: String = ".tmp/weights_ae.safetensors"
    var reportPath: String = ".tmp/relational_head_report.md"
    var headsPath: String = ".tmp/relational_heads.json"
    var predictionsPath: String = ".tmp/relational_head_predictions_A.json"
    var steps: Int = 300
    var maxSeqLen: Int = 512
    var foldEpochs: Int = 15
    var relBatch: Int = 32
    var relLR: Float = 1.4e-3
    var skipFolds: Bool = false

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
            case "-o", "--report":
                i += 1
                if i < args.count { cli.reportPath = args[i] }
            case "--heads":
                i += 1
                if i < args.count { cli.headsPath = args[i] }
            case "--steps":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.steps = max(1, v) }
            case "--fold-epochs":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.foldEpochs = max(1, v) }
            case "--rel-batch":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.relBatch = max(2, v) }
            case "--rel-lr":
                i += 1
                if i < args.count, let v = Float(args[i]) { cli.relLR = v }
            case "--skip-folds":
                cli.skipFolds = true
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

/// One prediction target: which receiver text is used as input and which label is predicted.
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

MLX.Memory.cacheLimit = 128 * 1024 * 1024
MLX.Memory.clearCache()
let cli = RelationalHeadCLI.parse(CommandLine.arguments)
let config = DatasetConfiguration.load(from: cli.dataPath)
let trainCasesAll = config.relationalDialogueCases.filter { !$0.isZeroShotUnseen }
let unseenCasesAll = config.relationalDialogueCases.filter { $0.isZeroShotUnseen }
guard !trainCasesAll.isEmpty && !unseenCasesAll.isEmpty else {
    print("No train / unseen relational cases found in \(cli.dataPath)")
    exit(1)
}

let weights = try SWTHawkWeights.load(from: URL(fileURLWithPath: cli.weightsPath))
let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)
let mainModel = MLXSWTHawkAutoencoder(weights: weights)

let allCases = trainCasesAll + unseenCasesAll
let seqLen = allCases
    .flatMap { [$0.hostText, $0.partnerEarlyText, $0.partnerCurrentText] }
    .map { min(cli.maxSeqLen, vocab.encode($0).count) }
    .reduce(4, max)

print("==================================================================")
print("Phase 5.1 Relational Head Evaluation (pure-neural z_rel, unified heads)")
print("==================================================================")
print("Train cases: \(trainCasesAll.count) | Unseen cases: \(unseenCasesAll.count) | seqLen: \(seqLen)")

let hostTok = allCases.map { RelationalPipeline.encodeTokens($0.hostText, vocab: vocab, time: seqLen) }
let earlyTok = allCases.map { RelationalPipeline.encodeTokens($0.partnerEarlyText, vocab: vocab, time: seqLen) }
let currentTok = allCases.map { RelationalPipeline.encodeTokens($0.partnerCurrentText, vocab: vocab, time: seqLen) }

let ngram = CharNGramFeaturizer(dim: 2048)
let ngramHost = allCases.map { ngram.features($0.hostText) }
let ngramEarly = allCases.map { ngram.features($0.partnerEarlyText) }
let ngramCurrent = allCases.map { ngram.features($0.partnerCurrentText) }
func ngramInput(_ i: Int, _ early: Bool) -> [Float] {
    return ngramHost[i] + (early ? ngramEarly[i] : ngramCurrent[i])
}

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
    for y in labels {
        counts[y] += 1
    }
    return counts.firstIndex(of: counts.max() ?? 0) ?? 0
}

func head(_ heads: RelationalCalibratedHeads, _ task: HeadTask) -> SoftmaxClassifierHead? {
    switch task.name {
    case "archetype_t4": return heads.archetypeT4
    case "topology_t4": return heads.topologyT4
    case "archetype_t2": return heads.archetypeT2
    default: return heads.topologyT2
    }
}

let trainIndices = allCases.indices.filter { !allCases[$0].isZeroShotUnseen }
let unseenIndices = allCases.indices.filter { allCases[$0].isZeroShotUnseen }
let pct = { (v: Float) in String(format: "%6.2f%%", v * 100.0) }

// ------------------------------------------------------------------
// (A) Scenario split: the SAME calibrated heads and the SAME z_rel tensor used by verify_latent_space (Audit 5).
// ------------------------------------------------------------------
let zT2All = RelationalPipeline.encodeZRel(model: mainModel, senders: hostTok, receivers: earlyTok, time: seqLen)
let zT4All = RelationalPipeline.encodeZRel(model: mainModel, senders: hostTok, receivers: currentTok, time: seqLen)

var splitResults: [EvalResult] = []
var predictionsA: [String: [String: Int]] = [:]
for task in tasks {
    guard let h = head(weights.relational.calibratedHeads, task) else {
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
        task: task, featurizer: "neural", testIdx: unseenIndices, preds: preds,
        majorityPreds: unseenIndices.map { _ in maj }, trainCorrect: trainCorrect, trainTotal: trainIndices.count
    )
    splitResults.append(r)
    print("[split   ][neural] \(task.name): train \(pct(r.trainAcc)) | unseen \(pct(r.testAcc)) | majority \(pct(r.majorityAcc))")
}
for task in tasks {
    let trainX = trainIndices.map { ngramInput($0, task.useEarlyText) }
    let trainY = trainIndices.map { task.label(allCases[$0]) }
    let h = SoftmaxClassifierHead.trainMLX(features: trainX, labels: trainY, classCount: task.classNames.count, steps: cli.steps)
    let trainCorrect = zip(trainX, trainY).filter { h.predict($0) == $1 }.count
    let preds = unseenIndices.map { h.predict(ngramInput($0, task.useEarlyText)) }
    let maj = majorityClass(trainY, classCount: task.classNames.count)
    let r = summarize(
        task: task, featurizer: "char_ngram", testIdx: unseenIndices, preds: preds,
        majorityPreds: unseenIndices.map { _ in maj }, trainCorrect: trainCorrect, trainTotal: trainIndices.count
    )
    splitResults.append(r)
    print("[split   ][char_ngram] \(task.name): train \(pct(r.trainAcc)) | unseen \(pct(r.testAcc)) | majority \(pct(r.majorityAcc))")
}
let predData = try JSONEncoder().encode(predictionsA)
try predData.write(to: URL(fileURLWithPath: cli.predictionsPath))

// ------------------------------------------------------------------
// (B) Stratified Group 6-Fold: categories grouped by (arch_T2, topo_T2, arch_T4, topo_T4) signature and assigned
// round-robin, so every label of a held-out fold also appears on the training side. The encoder is RE-TRAINED
// per fold from the autoencoder-stage weights on the fold's training categories only.
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
    var neuralPreds: [String: [Int: Int]] = [:]
    var neuralTrainCorrect: [String: Int] = [:]
    var neuralTrainTotal: [String: Int] = [:]
    var majorityPredsByTask: [String: [Int: Int]] = [:]
    var ngramPreds: [String: [Int: Int]] = [:]
    var ngramTrainCorrect: [String: Int] = [:]

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
        let z2 = RelationalPipeline.encodeZRel(model: model, senders: evalIdx.map { hostTok[$0] }, receivers: evalIdx.map { earlyTok[$0] }, time: seqLen)
        let z4 = RelationalPipeline.encodeZRel(model: model, senders: evalIdx.map { hostTok[$0] }, receivers: evalIdx.map { currentTok[$0] }, time: seqLen)
        var row: [Int: Int] = [:]
        for (k, i) in evalIdx.enumerated() {
            row[i] = k
        }

        for task in tasks {
            guard let h = head(fw.relational.calibratedHeads, task) else { continue }
            let z = task.useEarlyText ? z2 : z4
            let tc = foldTrain.filter { h.predict(z[row[$0]!]) == task.label(allCases[$0]) }.count
            neuralTrainCorrect[task.name, default: 0] += tc
            neuralTrainTotal[task.name, default: 0] += foldTrain.count
            let maj = majorityClass(foldTrain.map { task.label(allCases[$0]) }, classCount: task.classNames.count)
            for i in foldTest {
                neuralPreds[task.name, default: [:]][i] = h.predict(z[row[i]!])
                majorityPredsByTask[task.name, default: [:]][i] = maj
            }

            let trainX = foldTrain.map { ngramInput($0, task.useEarlyText) }
            let trainY = foldTrain.map { task.label(allCases[$0]) }
            let nh = SoftmaxClassifierHead.trainMLX(features: trainX, labels: trainY, classCount: task.classNames.count, steps: cli.steps)
            ngramTrainCorrect[task.name, default: 0] += zip(trainX, trainY).filter { nh.predict($0) == $1 }.count
            for i in foldTest {
                ngramPreds[task.name, default: [:]][i] = nh.predict(ngramInput(i, task.useEarlyText))
            }
            let foldAcc = Float(foldTest.filter { neuralPreds[task.name]![$0] == task.label(allCases[$0]) }.count) / Float(max(1, foldTest.count))
            print("  [fold \(fold)] \(task.name): neural held-out \(pct(foldAcc))")
        }
        let line = String(format: "fold %d: %d categories, train %d, test %d, %.1fs", fold, foldCats.count, foldTrain.count, foldTest.count, Date().timeIntervalSince(start))
        foldLog.append(line)
        print("[fold \(fold)] done (\(line))")
        MLX.Memory.clearCache()
    }

    let allTest = allCases.indices.filter { foldOfCategory[allCases[$0].group] != nil }
    for task in tasks {
        let rn = summarize(
            task: task, featurizer: "neural", testIdx: allTest,
            preds: allTest.map { neuralPreds[task.name]?[$0] ?? 0 },
            majorityPreds: allTest.map { majorityPredsByTask[task.name]?[$0] ?? 0 },
            trainCorrect: neuralTrainCorrect[task.name] ?? 0, trainTotal: neuralTrainTotal[task.name] ?? 1
        )
        holdoutResults.append(rn)
        print("[holdout ][neural] \(task.name): held-out categories \(pct(rn.testAcc)) | majority \(pct(rn.majorityAcc)) | label unseen in training: \(missingByTask[task.name] ?? -1)")
    }
    for task in tasks {
        let rg = summarize(
            task: task, featurizer: "char_ngram", testIdx: allTest,
            preds: allTest.map { ngramPreds[task.name]?[$0] ?? 0 },
            majorityPreds: allTest.map { majorityPredsByTask[task.name]?[$0] ?? 0 },
            trainCorrect: ngramTrainCorrect[task.name] ?? 0, trainTotal: neuralTrainTotal[task.name] ?? 1
        )
        holdoutResults.append(rg)
        print("[holdout ][char_ngram] \(task.name): held-out categories \(pct(rg.testAcc)) | majority \(pct(rg.majorityAcc))")
    }
}

func resultTable(_ rs: [EvalResult], testLabel: String) -> String {
    var t = "| 特徴 | タスク | 学習精度 | \(testLabel) | 多数派クラス基準 |\n"
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

var report = "# 関係性分類の汎化評価（Phase 5.1: Dual-Stream 純ニューラル `z_rel`）\n\n"
report += "- 入力: `(senderTokenIDs, receiverTokenIDs)` の2配列のみ（連結しない Dual-Stream。$T_4$ は受信側全文、$T_2$ は受信側前半）\n"
report += "- `neural`: `z_rel = [z_LL^rel(\(weights.relational.latentDim / 2)); z_HH^rel(\(weights.relational.latentDim / 2))] ∈ R^\(weights.relational.latentDim)`（`verify_latent_space` と同一テンソル・同一校正ヘッド）\n"
report += "- `char_ngram`: 文字 1-gram / 2-gram のハッシュ特徴（`[送信側 ; 受信側]` 連結、比較用ベースライン）\n\n"
report += "## (A) シナリオ分割: train → unseen_test\n\n"
report += "学習 `\(trainIndices.count)` 件、評価 `\(unseenIndices.count)` 件。`neural` は学習時に保存された校正ヘッドをそのまま適用（再学習なし）。\n\n"
report += resultTable(splitResults, testLabel: "未学習精度")
report += "\n### クラス別正解数\n\n"
report += perClassList(splitResults)
report += "\n## (B) Stratified Group \(foldCount)-Fold（カテゴリ単位の取り分け・フォールドごとにエンコーダ再学習）\n\n"
report += "全 `\(categories.count)` カテゴリをラベル署名 $(\\text{arch}_{T_2}, \\text{topo}_{T_2}, \\text{arch}_{T_4}, \\text{topo}_{T_4})$ で層化し、ラウンドロビンで \(foldCount) フォールドに割り当てた。"
report += "各フォールドでは、オートエンコーダ段階の重み（ラベル不使用）から出発し、**他フォールドのカテゴリの train 分割のみ**で Dual-Stream エンコーダ・校正ヘッドを End-to-End 再学習し、取り分けたカテゴリの全シナリオ（train + unseen）で評価する。\n\n"
report += "- 学習側に同じラベルが1件も無かった評価ケース数: "
report += tasks.map { "`\($0.name)` \(missingByTask[$0.name] ?? -1)" }.joined(separator: ", ")
report += "\n- 注記: オートエンコーダ段階（文章再構成、ラベル不使用）は全カテゴリの train 分割テキストで事前学習されている。\n\n"
if holdoutResults.isEmpty {
    report += "_(--skip-folds により未実行)_\n"
} else {
    report += resultTable(holdoutResults, testLabel: "未見カテゴリ精度")
    report += "\n### フォールド実行ログ\n\n"
    report += foldLog.map { "- \($0)" }.joined(separator: "\n")
    report += "\n\n### クラス別正解数\n\n"
    report += perClassList(holdoutResults)
}

try report.write(to: URL(fileURLWithPath: cli.reportPath), atomically: true, encoding: .utf8)
let headsData = try JSONEncoder().encode(weights.relational.calibratedHeads)
try headsData.write(to: URL(fileURLWithPath: cli.headsPath))
print("Saved report to \(cli.reportPath) and heads to \(cli.headsPath)")
