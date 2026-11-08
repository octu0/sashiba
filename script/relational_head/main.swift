import Foundation
import MLX
import Sashiba

struct RelationalHeadCLI {
    var dataPath: String = ".tmp"
    var weightsPath: String = ".tmp/weights.safetensors"
    var reportPath: String = ".tmp/relational_head_report.md"
    var headsPath: String = ".tmp/relational_heads.json"
    var steps: Int = 300
    var maxSeqLen: Int = 512

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
            case "-o", "--report":
                i += 1
                if i < args.count { cli.reportPath = args[i] }
            case "--heads":
                i += 1
                if i < args.count { cli.headsPath = args[i] }
            case "--steps":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.steps = max(1, v) }
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

/// One prediction target: which text is used as input and which label is predicted.
struct HeadTask {
    let name: String
    let useEarlyText: Bool
    let classNames: [String]
    let label: (RelationalDialogueCase) -> Int
}

let archetypeNames = RelationalArchetype.allCases.map(\.rawValue)
let topologyNames = RelationalTopology.allCases.map(\.rawValue)
func archetypeIndex(_ a: RelationalArchetype) -> Int {
    return RelationalArchetype.allCases.firstIndex(of: a) ?? 0
}
func topologyIndex(_ t: RelationalTopology) -> Int {
    return RelationalTopology.allCases.firstIndex(of: t) ?? 0
}

let tasks: [HeadTask] = [
    HeadTask(name: "archetype_t4", useEarlyText: false, classNames: archetypeNames) { archetypeIndex($0.expectedArchetype) },
    HeadTask(name: "topology_t4", useEarlyText: false, classNames: topologyNames) { topologyIndex($0.expectedTopology) },
    HeadTask(name: "archetype_t2", useEarlyText: true, classNames: archetypeNames) { archetypeIndex($0.expectedArchetypeT2) },
    HeadTask(name: "topology_t2", useEarlyText: true, classNames: topologyNames) { topologyIndex($0.expectedTopologyT2) }
]

let cli = RelationalHeadCLI.parse(CommandLine.arguments)
let config = DatasetConfiguration.load(from: cli.dataPath)
let trainCases = config.relationalDialogueCases.filter { !$0.isZeroShotUnseen }
let unseenCases = config.relationalDialogueCases.filter { $0.isZeroShotUnseen }
guard !trainCases.isEmpty && !unseenCases.isEmpty else {
    print("No train / unseen relational cases found in \(cli.dataPath)")
    exit(1)
}

let weights = try SWTHawkWeights.load(from: URL(fileURLWithPath: cli.weightsPath))
let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)
let mlxModel = MLXSWTHawkAutoencoder(weights: weights)

let allCases = trainCases + unseenCases
let seqLen = allCases
    .flatMap { [$0.hostText, $0.partnerEarlyText, $0.partnerCurrentText] }
    .map { min(cli.maxSeqLen, vocab.encode($0).count) }
    .reduce(4, max)

print("==================================================================")
print("Learned Relational Classification Heads (train on train split only)")
print("==================================================================")
print("Train cases: \(trainCases.count) | Unseen cases: \(unseenCases.count) | seqLen: \(seqLen)")

/// Neural features for a list of texts via the trained encoder (no lattice, no rules).
func neuralFeatures(_ texts: [String]) -> [[Float]] {
    var flat: [Int] = []
    flat.reserveCapacity(texts.count * seqLen)
    for t in texts {
        flat.append(contentsOf: vocab.encode(t, maxLength: seqLen, addEOS: false, padToLength: seqLen))
    }
    let z = mlxModel.encodeNeural(tokens: flat, batch: texts.count, time: seqLen)
    let d = mlxModel.latentDim
    return (0..<texts.count).map { Array(z[($0 * d)..<(($0 + 1) * d)]) }
}

let ngram = CharNGramFeaturizer(dim: 2048)

/// Per-case features for host / early partner / current partner texts, computed once per featurizer.
struct CaseFeatures {
    let host: [[Float]]
    let early: [[Float]]
    let current: [[Float]]

    func input(_ i: Int, useEarlyText: Bool) -> [Float] {
        return host[i] + (useEarlyText ? early[i] : current[i])
    }
}

func computeFeatures(neural: Bool) -> CaseFeatures {
    let f: ([String]) -> [[Float]] = if neural { neuralFeatures } else { { $0.map { ngram.features($0) } } }
    return CaseFeatures(
        host: f(allCases.map(\.hostText)),
        early: f(allCases.map(\.partnerEarlyText)),
        current: f(allCases.map(\.partnerCurrentText))
    )
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

/// Trains a head on `trainIdx` and evaluates on `testIdx` (indices into `allCases`).
func trainAndEvaluate(
    task: HeadTask,
    features: CaseFeatures,
    trainIdx: [Int],
    testIdx: [Int],
    steps: Int
) -> (head: SoftmaxClassifierHead, trainCorrect: Int, testPred: [Int], majorityClass: Int) {
    let trainX = trainIdx.map { features.input($0, useEarlyText: task.useEarlyText) }
    let trainY = trainIdx.map { task.label(allCases[$0]) }
    let head = SoftmaxClassifierHead.train(features: trainX, labels: trainY, classCount: task.classNames.count, steps: steps)
    let trainCorrect = zip(trainX, trainY).filter { head.predict($0) == $1 }.count
    let testPred = testIdx.map { head.predict(features.input($0, useEarlyText: task.useEarlyText)) }
    var counts = [Int](repeating: 0, count: task.classNames.count)
    for y in trainY {
        counts[y] += 1
    }
    let majorityClass = counts.firstIndex(of: counts.max() ?? 0) ?? 0
    return (head, trainCorrect, testPred, majorityClass)
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

let trainIndices = allCases.indices.filter { !allCases[$0].isZeroShotUnseen }
let unseenIndices = allCases.indices.filter { allCases[$0].isZeroShotUnseen }

// Category-holdout folds: every category is held out exactly once, with all of its scenarios (train + unseen).
let foldCount = 6
let categories = Array(Set(allCases.map(\.group))).sorted()
let foldOfCategory = Dictionary(uniqueKeysWithValues: categories.enumerated().map { ($1, $0 % foldCount) })

var splitResults: [EvalResult] = []
var holdoutResults: [EvalResult] = []
var holdoutMissingLabelCases: [String: Int] = [:]
var savedHeads: [String: SoftmaxClassifierHead] = [:]
let pct = { (v: Float) in String(format: "%6.2f%%", v * 100.0) }

for featurizer in ["neural", "char_ngram"] {
    let features = computeFeatures(neural: featurizer == "neural")
    for task in tasks {
        // (A) Scenario split: train split -> unseen_test split (same categories on both sides).
        let a = trainAndEvaluate(task: task, features: features, trainIdx: trainIndices, testIdx: unseenIndices, steps: cli.steps)
        savedHeads["\(featurizer).\(task.name)"] = a.head
        let ra = summarize(
            task: task,
            featurizer: featurizer,
            testIdx: unseenIndices,
            preds: a.testPred,
            majorityPreds: unseenIndices.map { _ in a.majorityClass },
            trainCorrect: a.trainCorrect,
            trainTotal: trainIndices.count
        )
        splitResults.append(ra)
        print("[split   ][\(featurizer)] \(task.name): train \(pct(ra.trainAcc)) | unseen \(pct(ra.testAcc)) | majority \(pct(ra.majorityAcc))")

        // (B) Category holdout: train on train-split scenarios of other categories, test on all scenarios of held-out categories.
        var allTestIdx: [Int] = []
        var allPreds: [Int] = []
        var allMajority: [Int] = []
        var trainCorrectSum = 0
        var trainTotalSum = 0
        var missing = 0
        for fold in 0..<foldCount {
            let foldTrain = trainIndices.filter { foldOfCategory[allCases[$0].group] != fold }
            let foldTest = allCases.indices.filter { foldOfCategory[allCases[$0].group] == fold }
            let b = trainAndEvaluate(task: task, features: features, trainIdx: foldTrain, testIdx: foldTest, steps: cli.steps)
            let trainLabels = Set(foldTrain.map { task.label(allCases[$0]) })
            missing += foldTest.filter { !trainLabels.contains(task.label(allCases[$0])) }.count
            allTestIdx += foldTest
            allPreds += b.testPred
            allMajority += foldTest.map { _ in b.majorityClass }
            trainCorrectSum += b.trainCorrect
            trainTotalSum += foldTrain.count
        }
        holdoutMissingLabelCases["\(featurizer).\(task.name)"] = missing
        let rb = summarize(
            task: task,
            featurizer: featurizer,
            testIdx: allTestIdx,
            preds: allPreds,
            majorityPreds: allMajority,
            trainCorrect: trainCorrectSum,
            trainTotal: trainTotalSum
        )
        holdoutResults.append(rb)
        print("[holdout ][\(featurizer)] \(task.name): train \(pct(rb.trainAcc)) | held-out categories \(pct(rb.testAcc)) | majority \(pct(rb.majorityAcc)) | label unseen in training: \(missing)")
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

var report = "# 学習される分類ヘッドによる汎化評価\n\n"
report += "- 入力特徴: `[送信側テキスト ; 受信側テキスト]` の連結（$T_4$ は全文、$T_2$ は前半のみ）\n"
report += "- `neural`: 学習済みエンコーダのニューラル出力 `z` のみ（文字ID格納やルールを経由しない）\n"
report += "- `char_ngram`: 文字 1-gram / 2-gram のハッシュ特徴（比較用ベースライン）\n\n"
report += "## (A) シナリオ分割: 同じカテゴリ内で train → unseen_test\n\n"
report += "学習 `\(trainIndices.count)` 件、評価 `\(unseenIndices.count)` 件。学習と評価に同じカテゴリが含まれるため、カテゴリ固有の語彙を覚えるだけでも高得点になり得る。\n\n"
report += resultTable(splitResults, testLabel: "未学習精度")
report += "\n### クラス別正解数\n\n"
report += perClassList(splitResults)
report += "\n## (B) カテゴリ単位の取り分け（\(foldCount)分割）\n\n"
report += "全 `\(categories.count)` カテゴリを \(foldCount) グループに分け、各グループのカテゴリを丸ごと評価に回す（学習には他グループの train 分割のみを使用）。見たことのないカテゴリでも属性・距離感を推定できるかを測る。\n\n"
report += resultTable(holdoutResults, testLabel: "未見カテゴリ精度")
report += "\n- 学習側に同じラベルが1件も無かった評価ケース数（原理的に正解不能）: "
report += holdoutMissingLabelCases.sorted { $0.key < $1.key }.map { "`\($0.key)` \($0.value)" }.joined(separator: ", ")
report += "\n\n### クラス別正解数\n\n"
report += perClassList(holdoutResults)

try report.write(to: URL(fileURLWithPath: cli.reportPath), atomically: true, encoding: .utf8)
let headsData = try JSONEncoder().encode(savedHeads)
try headsData.write(to: URL(fileURLWithPath: cli.headsPath))
print("Saved report to \(cli.reportPath) and heads to \(cli.headsPath)")
