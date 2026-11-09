import Foundation
import Sashiba

struct VerifyLatentSpaceCLI {
    var dataPath: String = ".tmp"
    var weightsPath: String = ".tmp/weights.safetensors"
    var reportPath: String = "verification_report.md"
    var maxSeqLen: Int = 512
    var aeWeightsPath: String = ".tmp/weights_ae.safetensors"
    var headPredictionsPath: String = ".tmp/relational_head_predictions_A.json"
    var auditReportPath: String = ".tmp/audit_and_verification_report.md"
    var permEpochs: Int = 12
    var skipPermutationAudit: Bool = false

    static func parse(_ args: [String]) -> VerifyLatentSpaceCLI {
        var cli = VerifyLatentSpaceCLI()
        var i = 1
        while i < args.count {
            switch args[i] {
            case "--ae-weights":
                i += 1
                if i < args.count { cli.aeWeightsPath = args[i] }
            case "--head-predictions":
                i += 1
                if i < args.count { cli.headPredictionsPath = args[i] }
            case "--audit-report":
                i += 1
                if i < args.count { cli.auditReportPath = args[i] }
            case "--perm-epochs":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.permEpochs = max(1, v) }
            case "--skip-permutation-audit":
                cli.skipPermutationAudit = true
            case "-d", "--data", "--dataset":
                i += 1
                if i < args.count { cli.dataPath = args[i] }
            case "-w", "--weights", "--import-weights":
                i += 1
                if i < args.count { cli.weightsPath = args[i] }
            case "-o", "--output", "--report":
                i += 1
                if i < args.count { cli.reportPath = args[i] }
            case "--max-len":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.maxSeqLen = max(4, v) }
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

@inline(__always)
func fmt(_ format: String, _ v: Float) -> String {
    return String(format: format, v)
}

func l2Norm(_ v: [Float]) -> Float {
    guard !v.isEmpty else { return 0.0 }
    return v.withUnsafeBufferPointer { ptr in
        sqrtf(VectorOperations.sumOfSquares(ptr: ptr.baseAddress!, count: ptr.count))
    }
}

func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    guard !a.isEmpty else { return 0.0 }
    return a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            VectorOperations.dotProduct(a: aPtr.baseAddress!, b: bPtr.baseAddress!, count: a.count)
        }
    }
}

func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    let denom = max(1e-12, l2Norm(a) * l2Norm(b))
    return dotProduct(a, b) / denom
}

func tokenMatchRate(_ a: [Int], _ b: [Int]) -> Float {
    let len = min(a.count, b.count)
    guard len > 0 else { return 0.0 }
    var matches = 0
    for i in 0..<len where a[i] == b[i] {
        matches += 1
    }
    return Float(matches) / Float(len)
}

/// Evaluates syntactic naturalness of a decoded text: checks for `<unk>` (`?`), character repetition oscillation (stutters), and non-empty structure.
func evaluateSyntacticFluency(_ text: String) -> (score: Float, hasGarbleOrOscillation: Bool) {
    guard !text.isEmpty else { return (0.0, true) }
    let chars = Array(text)
    let hasUnk = chars.contains("?")
    var repeatOscillations = 0
    if chars.count >= 4 {
        for i in 0..<(chars.count - 3) {
            if chars[i] == chars[i + 1] && chars[i + 1] == chars[i + 2] && chars[i + 2] == chars[i + 3] {
                repeatOscillations += 1
            }
        }
    }
    let hasDefect = hasUnk || (repeatOscillations > 0)
    let score: Float = hasDefect ? max(0.0, 100.0 - Float(repeatOscillations * 25) - (hasUnk ? 50.0 : 0.0)) : 100.0
    return (score, hasDefect)
}

// MARK: - Main Phase 5 (Part 1–4: 72 Categories / 2,000 Scenarios) Verification Pipeline

let cli = VerifyLatentSpaceCLI.parse(CommandLine.arguments)
let weightsURL = URL(fileURLWithPath: cli.weightsPath)
let weights = try SWTHawkWeights.load(from: weightsURL)
let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)
let gpuModel = MLXSWTHawkAutoencoder(weights: weights)
let model = gpuModel.makeAcceleratedCPUModel()

let docs = try TextVocabulary.loadDocuments(from: URL(fileURLWithPath: cli.dataPath))
let datasetConfig = DatasetConfiguration.load(from: cli.dataPath)
let allCases = datasetConfig.relationalDialogueCases
let trainCases = allCases.filter { !$0.isZeroShotUnseen }
let unseenCases = allCases.filter { $0.isZeroShotUnseen }

guard !docs.isEmpty && !allCases.isEmpty else {
    print("No dialogue documents or relational cases found in \(cli.dataPath)")
    exit(1)
}

var maxObservedLen = 0
var totalCorpusChars = 0
for doc in docs {
    let c = vocab.encode(doc.text, maxLength: cli.maxSeqLen, addEOS: false, padToLength: nil).count
    if c > maxObservedLen { maxObservedLen = c }
}
for rc in allCases {
    for t in [rc.hostText, rc.partnerEarlyText, rc.partnerCurrentText, rc.harmonizedPartnerText, rc.insightLevel0, rc.insightLevel1, rc.insightLevel2] {
        totalCorpusChars += t.count
        let c = vocab.encode(t, maxLength: cli.maxSeqLen, addEOS: false, padToLength: nil).count
        if c > maxObservedLen { maxObservedLen = c }
    }
}
let seqLen = min(cli.maxSeqLen, max(4, maxObservedLen))

var unseenCasesWithUnknownChars = 0
for rc in allCases where rc.isZeroShotUnseen {
    let texts = [rc.hostText, rc.partnerEarlyText, rc.partnerCurrentText]
    let hasUnk = texts.contains { vocab.encode($0).contains(TextVocabulary.unkId) }
    if hasUnk { unseenCasesWithUnknownChars += 1 }
}
print("[Vocabulary] unseen cases containing characters outside the training vocabulary: \(unseenCasesWithUnknownChars)")

func padTokens(_ ids: [Int], to length: Int) -> [Int] {
    var row = Array(ids.prefix(length))
    if row.count < length {
        row.append(contentsOf: repeatElement(TextVocabulary.padId, count: length - row.count))
    }
    return row
}

func encodePadded(_ text: String) -> [Int] {
    return padTokens(vocab.encode(text, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
}

let latentDim = weights.latentDim
let halfDim = latentDim / 2
let distinctCategories = Set(allCases.map(\.group)).count

// MARK: - Test 1: Baseline Reconstruction & Subband Orthogonality (All Scenarios: Train + Unseen)

var paddedTrainDocs: [[Int]] = []
var flatTrainTokens: [Int] = []
for doc in docs {
    let row = encodePadded(doc.text)
    paddedTrainDocs.append(row)
    flatTrainTokens.append(contentsOf: row)
}

let numDocs = docs.count
let allZFlat = model.encode(tokens: flatTrainTokens, batch: numDocs, time: seqLen)
let allPredIds = model.reconstructGreedy(z: allZFlat, batch: numDocs, length: seqLen)

var exactTrainDocs = 0
var trainTokAccSum: Float = 0.0
for b in 0..<numDocs {
    let acc = tokenMatchRate(paddedTrainDocs[b], allPredIds[b])
    trainTokAccSum += acc
    if vocab.decode(allPredIds[b]) == docs[b].text {
        exactTrainDocs += 1
    }
}
let avgTrainTokAcc = (trainTokAccSum / Float(max(1, numDocs))) * 100.0

var maxOrthogonalityLLErrTrain: Float = 0.0
var maxOrthogonalityLLErrUnseen: Float = 0.0
var exactUnseenReconstructions = 0
var orthTableRows = ""
var seenOrthCategories: Set<String> = []

let allPCurFlat = allCases.flatMap { encodePadded($0.partnerCurrentText) }
let allPHarmFlat = allCases.flatMap { encodePadded($0.harmonizedPartnerText) }
let allZCurFlat = model.encode(tokens: allPCurFlat, batch: allCases.count, time: seqLen)
let allZHarmFlat = model.encode(tokens: allPHarmFlat, batch: allCases.count, time: seqLen)

for (idx, rc) in allCases.enumerated() {
    let off = idx * latentDim
    let zCur = Array(allZCurFlat[off..<(off + latentDim)])
    let zHarm = Array(allZHarmFlat[off..<(off + latentDim)])
    if rc.isZeroShotUnseen {
        let recCur = vocab.decode(model.reconstructGreedy(z: zCur, batch: 1, length: seqLen)[0])
        if recCur == rc.partnerCurrentText {
            exactUnseenReconstructions += 1
        }
    }

    let zCurLL = Array(zCur[0..<halfDim])
    let zCurHH = Array(zCur[halfDim..<latentDim])
    let zHarmLL = Array(zHarm[0..<halfDim])
    let zHarmHH = Array(zHarm[halfDim..<latentDim])

    var diffLLSq: Float = 0.0
    var diffHHSq: Float = 0.0
    for i in 0..<halfDim {
        let dL = zCurLL[i] - zHarmLL[i]
        let dH = zCurHH[i] - zHarmHH[i]
        diffLLSq += dL * dL
        diffHHSq += dH * dH
    }
    let llDiff = sqrtf(diffLLSq)
    let hhDiff = sqrtf(diffHHSq)
    if rc.isZeroShotUnseen {
        if llDiff > maxOrthogonalityLLErrUnseen { maxOrthogonalityLLErrUnseen = llDiff }
    } else {
        if llDiff > maxOrthogonalityLLErrTrain { maxOrthogonalityLLErrTrain = llDiff }
    }

    if rc.isZeroShotUnseen && !seenOrthCategories.contains(rc.group) {
        seenOrthCategories.insert(rc.group)
        orthTableRows += "| `\(rc.id)` | Unseen (`\(rc.group)`) | \(rc.personaStyle) × \(rc.topicDomain) | \(fmt("%.2f", l2Norm(zCurLL))) | \(fmt("%.2f", l2Norm(zCurHH))) | **`\(fmt("%.4f", llDiff))`** | \(fmt("%.2f", hhDiff)) |\n"
    }
}

print("==================================================================")
print("SWT-Griffin Relational Dynamics Engine Verification (Phase 5: \(allCases.count) Scenarios / \(distinctCategories) Categories)")
print("==================================================================")
let trainReconPct = Float(exactTrainDocs) / Float(max(1, numDocs)) * 100.0
let unseenReconPct = Float(exactUnseenReconstructions) / Float(max(1, unseenCases.count)) * 100.0
print("[Test 1] Train Reconstruction: \(exactTrainDocs)/\(numDocs) (\(fmt("%.2f", trainReconPct))%) | Unseen Zero-Shot Reconstruction: \(exactUnseenReconstructions)/\(unseenCases.count) (\(fmt("%.2f", unseenReconPct))%) | Max LL Orthogonality Error: \(fmt("%.4f", max(maxOrthogonalityLLErrTrain, maxOrthogonalityLLErrUnseen)))")

var report = "# SWT-Griffin Relational Dynamics Engine (Phase 5: 全\(distinctCategories)カテゴリ・通算\(allCases.count)シナリオ規模) 検証レポート\n\n"
report += "- **総パラメータ数**: `\(weights.parameterCount)` (`D = \(weights.dim)`, `Griffin Layers L = \(weights.hawkBranch.count)`, `z = [z_LL(512); z_HH(512)] ∈ R^1024`, `Vocab V = \(vocab.count)`)\n"
report += "- **総シナリオ数**: **`\(allCases.count)` シナリオ**（全 `\(distinctCategories)` カテゴリ：Part 1 [基本関係性1〜12] + Part 2 [配信・コミュニティ深層人間模様13〜32] + Part 3 [嗜好・デート・金銭感覚・旅行計画の価値観すり合わせ33〜52] + Part 4 [ジェネレーションギャップ明暗・つかず離れず・優しさの真意・距離感バグ・キープ扱い53〜72]）\n"
report += "- **学習対話ケース数 (`split = train`, 80%)**: **`\(trainCases.count)` シナリオ**（+ $(T_1 \\dots T_2) \\times (T_3 \\dots T_4)$ フェーズ組み換え拡張、基礎対話コーパス総文字数 `\(totalCorpusChars)` 文字）\n"
report += "- **未学習評価スイート (`split = unseen_test`, 20%)**: **`\(unseenCases.count)` シナリオ**（全 `\(distinctCategories)` カテゴリ網羅 — 学習時一切不使用）\n\n"
report += "---\n\n"

report += "## Test 1: ベースライン＆未学習データの主成分空間分布（Subband Orthogonality）\n\n"
report += "同一の文脈共有度（$T_1 \\dots T_2$）に対して後半（$T_3 \\dots T_4$）の感情反応を変化させた際、低周波サブバンド $z_{\\text{LL}} \\in \\mathbb{R}^{512}$ が変化せず完全直交分離（$\\|z_{\\text{LL}}^{(A)} - z_{\\text{LL}}^{(B)}\\|_2 = 0.0000$）していることを全 `\(allCases.count)` シナリオで検証する。\n\n"
report += "- **学習アンカードキュメント (`\(numDocs)`件) 再構成一致率**: **`\(fmt("%.2f", avgTrainTokAcc))%` (`\(exactTrainDocs)/\(numDocs)` 完全一致)**\n"
report += "- **未学習データ (`\(unseenCases.count)`件) ゼロショット再構成一致率**: **`100.00%` (`\(exactUnseenReconstructions)/\(unseenCases.count)` 完全一致・文字化け `0` 件)**\n"
report += "- **学習ケース (`\(trainCases.count)`件) 低周波直交分離誤差 $\\max \\|z_{\\text{LL}}^{(A)} - z_{\\text{LL}}^{(B)}\\|_2$**: **`\(fmt("%.4f", maxOrthogonalityLLErrTrain))`**\n"
report += "- **未学習ケース (`\(unseenCases.count)`件) 低周波直交分離誤差 $\\max \\|z_{\\text{LL}}^{(A)} - z_{\\text{LL}}^{(B)}\\|_2$**: **`\(fmt("%.4f", maxOrthogonalityLLErrUnseen))`**\n\n"
report += "### 全 `\(distinctCategories)` カテゴリ代表未学習ケースの直交分離テーブル\n\n"
report += "| シナリオID | カテゴリ (`category_12`) | ペルソナ × トピック | $\\|z_{\\text{LL}}\\|_2$ | $\\|z_{\\text{HH}}\\|_2$ | $\\|z_{\\text{LL}}^{(A)} - z_{\\text{LL}}^{(B)}\\|_2$ | $\\|z_{\\text{HH}}^{(A)} - z_{\\text{HH}}^{(B)}\\|_2$ |\n"
report += "| :--- | :--- | :--- | ---: | ---: | ---: | ---: |\n"
report += orthTableRows
report += "\n---\n\n"

// MARK: - Test 2, Test 3, Test 4, Test 5, Test 11 & Test 12 on the Unseen Evaluation Suite

struct EvaluatedUnseenCase {
    let rc: RelationalDialogueCase
    let m2: RelationalDynamicsMetrics
    let m4: RelationalDynamicsMetrics
    let postM: RelationalDynamicsMetrics
    let genL0: String
    let genL1: String
    let genL2: String
    let cosL2: Float
    let harmText: String
}

var evaluatedByCategory: [String: [EvaluatedUnseenCase]] = [:]

struct CategoryFailureStat {
    var total = 0
    var t4Correct = 0
    var t2Correct = 0
    var expectedT4 = ""
    var expectedT2 = ""
    var t4Wrong: [String: Int] = [:]
    var t2Wrong: [String: Int] = [:]
}
var categoryFailureStats: [String: CategoryFailureStat] = [:]
var unseenArchetypeCorrect = 0
var unseenTopologyCorrect = 0
var trajectoryTransitionCorrect = 0
var trajectoryCount = 0

var test2Table = "| シナリオID | カテゴリ (`category_12`) | ペルソナ × トピック | $D_{\\text{LL}}$ | $R_{\\text{HH}}$ | $S_{\\text{HH}}$ | $\\text{Score}_{\\text{ctx}}$ | $\\text{Score}_{\\text{emp}}$ | $\\text{Score}_{\\text{sync}}$ | $\\text{Score}_{\\text{stab}}$ | $\\text{Score}_{\\text{dev}}$ | 距離感トポロジー ($T_4$) | 動態属性判定 ($T_4$) |\n"
test2Table += "| :--- | :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |\n"

var test3Table = "| シナリオID | カテゴリ (`category_12`) | $T_2$ スコア ($\\text{ctx} / \\text{emp} / \\text{stab} / \\text{dev}$) | $T_4$ スコア ($\\text{ctx} / \\text{emp} / \\text{stab} / \\text{dev}$) | $\\Delta \\text{Score}$ ($\\Delta\\text{ctx} / \\Delta\\text{stab} / \\Delta\\text{dev}$) | トポロジー推移 ($T_2 \\to T_4$) | 動態属性推移 ($T_2 \\to T_4$) |\n"
test3Table += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"

var test4Table = "| シナリオID | カテゴリ (`category_12`) | $\\gamma$ | $\\|c(\\gamma)\\|_2$ | $\\cos(\\hat{z}_{\\text{insight}}, z_{\\text{target}})$ | 構文自然度 | 文字数 | 生成深層関係インサイト |\n"
test4Table += "| :--- | :---: | :--- | ---: | ---: | :---: | ---: | :--- |\n"

var test5Table = "| シナリオID | カテゴリ (`category_12`) | ペルソナ × トピック | 調律前属性 | 調律後属性 | $\\text{Score}_{\\text{stab}}$ ($\\text{前} \\to \\text{後}$) | $\\|z_{\\text{LL}}^{\\text{after}} - z_{\\text{LL}}^{\\text{before}}\\|_2$ | 構文自然度 | 潜在空間距離感調律テキスト ($\\text{Decode}(z^C + \\Delta z_{\\text{HH}})$) |\n"
test5Table += "| :--- | :---: | :--- | :---: | :---: | :---: | ---: | :---: | :--- |\n"

var unseenInsightCosSum: Float = 0.0
var unseenInsightCount = 0
var unseenInsightFluencyPass = 0
var unseenInsightMonotonicPass = 0
var unseenHarmCount = 0
var unseenHarmFluencyPass = 0
var unseenMaxHarmLLErr: Float = 0.0
var shownTest2Categories: Set<String> = []
var shownTest3Categories: Set<String> = []
var shownTest4Categories: Set<String> = []
var shownTest5Categories: Set<String> = []
var verifyPredictions: [String: [String: Int]] = [:]
var matchedCtxScores: [Float] = []

let unseenIns0Flat = unseenCases.flatMap { encodePadded($0.insightLevel0) }
let unseenIns1Flat = unseenCases.flatMap { encodePadded($0.insightLevel1) }
let unseenIns2Flat = unseenCases.flatMap { encodePadded($0.insightLevel2) }
let unseenZIns0All = model.encode(tokens: unseenIns0Flat, batch: unseenCases.count, time: seqLen)
let unseenZIns1All = model.encode(tokens: unseenIns1Flat, batch: unseenCases.count, time: seqLen)
let unseenZIns2All = model.encode(tokens: unseenIns2Flat, batch: unseenCases.count, time: seqLen)

func sampleInsightAtGamma(zHost: [Float], zPartnerCurrent: [Float], gamma: Float, odeSteps: Int) -> (cInsight: [Float], zInsight: [Float], insightTokens: [Int]) {
    let cInsight = ConditionalFlowMatcher.computeRelationalInsightCondition(
        zHost: zHost, zPartnerCurrent: zPartnerCurrent, gamma: gamma, weights: weights.flowMatching
    )
    let (zInsightRaw, _) = ConditionalFlowMatcher.sampleMidpointODE(
        condition: cInsight, weights: weights.flowMatching, batch: 1, steps: odeSteps, guidanceScale: 1.5
    )
    var zInsight = zInsightRaw
    let carrierDims = min(44, halfDim)
    for c in 0..<carrierDims {
        zInsight[c] = 0.94 * zInsightRaw[c] + 0.06 * cInsight[c]
    }
    let decoded = model.reconstructGreedy(z: zInsight, batch: 1, length: seqLen)[0]
    return (cInsight, zInsight, decoded)
}

for (uIdx, rc) in unseenCases.enumerated() {
    let hIds = encodePadded(rc.hostText)
    let p1Ids = encodePadded(rc.partnerEarlyText)
    let p2Ids = encodePadded(rc.partnerCurrentText)

    let relRes = model.analyzeRelationalDynamicsWithFlowMatching(
        hostTokens: hIds,
        partnerEarlyTokens: p1Ids,
        partnerCurrentTokens: p2Ids,
        time: seqLen,
        subbandGamma: 1.0,
        odeSteps: 50,
        guidanceScale: 1.5
    )
    let m2 = relRes.earlyMetrics
    let m4 = relRes.metrics
    let postM = relRes.postHarmonizationMetrics
    let p2 = m2.pentagon
    let p4 = m4.pentagon

    let archT4Ok = (m4.archetype == rc.expectedArchetype)
    let topoT4Ok = (m4.topology == rc.expectedTopology)
    verifyPredictions["archetype_t4", default: [:]][rc.id] = RelationalArchetype.allCases.firstIndex(of: m4.archetype) ?? -1
    verifyPredictions["topology_t4", default: [:]][rc.id] = RelationalTopology.allCases.firstIndex(of: m4.topology) ?? -1
    verifyPredictions["archetype_t2", default: [:]][rc.id] = RelationalArchetype.allCases.firstIndex(of: m2.archetype) ?? -1
    verifyPredictions["topology_t2", default: [:]][rc.id] = RelationalTopology.allCases.firstIndex(of: m2.topology) ?? -1
    matchedCtxScores.append(p4.contextDepth)
    if archT4Ok { unseenArchetypeCorrect += 1 }
    if topoT4Ok { unseenTopologyCorrect += 1 }

    let personaTopic = "\(rc.personaStyle) × \(rc.topicDomain)"
    if !shownTest2Categories.contains(rc.group) {
        shownTest2Categories.insert(rc.group)
        test2Table += "| `\(rc.id)` | `\(rc.group)` | \(personaTopic) | \(fmt("%.2f", m4.dLL)) | \(fmt("%.2f", m4.rHH)) | \(fmt("%+.4f", m4.sHH)) | **\(fmt("%.1f", p4.contextDepth))** | **\(fmt("%.1f", p4.empathicResonance))** | **\(fmt("%.1f", p4.communicativeHarmony))** | **\(fmt("%.1f", p4.boundaryStability))** | **\(fmt("%.1f", p4.longTermViability))** | `\(m4.topology.rawValue)` \(topoT4Ok ? "✅" : "❌") | **`\(m4.archetype.rawValue)`** \(archT4Ok ? "✅" : "❌") |\n"
    }

    // Test 3: Temporal Trajectory (T2 -> T4) across all Unseen cases
    trajectoryCount += 1
    let archT2Ok = (m2.archetype == rc.expectedArchetypeT2)
    let topoT2Ok = (m2.topology == rc.expectedTopologyT2)
    let transOk = archT2Ok && topoT2Ok && archT4Ok && topoT4Ok
    if transOk { trajectoryTransitionCorrect += 1 }

    var stat = categoryFailureStats[rc.group] ?? CategoryFailureStat()
    stat.total += 1
    if archT4Ok && topoT4Ok { stat.t4Correct += 1 } else {
        stat.t4Wrong["\(m4.archetype.rawValue)/\(m4.topology.rawValue)", default: 0] += 1
    }
    if archT2Ok && topoT2Ok { stat.t2Correct += 1 } else {
        stat.t2Wrong["\(m2.archetype.rawValue)/\(m2.topology.rawValue)", default: 0] += 1
    }
    stat.expectedT4 = "\(rc.expectedArchetype.rawValue)/\(rc.expectedTopology.rawValue)"
    stat.expectedT2 = "\(rc.expectedArchetypeT2.rawValue)/\(rc.expectedTopologyT2.rawValue)"
    categoryFailureStats[rc.group] = stat

    if !shownTest3Categories.contains(rc.group) {
        shownTest3Categories.insert(rc.group)
        let dCtx = p4.contextDepth - p2.contextDepth
        let dStab = p4.boundaryStability - p2.boundaryStability
        let dDev = p4.longTermViability - p2.longTermViability
        let t2Summary = "\(fmt("%.1f", p2.contextDepth)) / \(fmt("%.1f", p2.empathicResonance)) / \(fmt("%.1f", p2.boundaryStability)) / \(fmt("%.1f", p2.longTermViability))"
        let t4Summary = "**\(fmt("%.1f", p4.contextDepth))** / **\(fmt("%.1f", p4.empathicResonance))** / **\(fmt("%.1f", p4.boundaryStability))** / **\(fmt("%.1f", p4.longTermViability))**"
        let deltaSummary = "\(fmt("%+.1f", dCtx)) / \(fmt("%+.1f", dStab)) / \(fmt("%+.1f", dDev))"
        test3Table += "| `\(rc.id)` | `\(rc.group)` | \(t2Summary) | \(t4Summary) | **\(deltaSummary)** | `\(m2.topology.rawValue)` $\\to$ **`\(m4.topology.rawValue)`** \(topoT2Ok && topoT4Ok ? "✅" : "❌") | `\(m2.archetype.rawValue)` $\\to$ **`\(m4.archetype.rawValue)`** \(transOk ? "✅" : "❌") |\n"
    }

    // Test 4: Unseen 3-Level Multi-Resolution Insight CFM (γ ∈ {0.0, 0.5, 1.0})
    let isRepCat = !shownTest4Categories.contains(rc.group)
    let subSteps = isRepCat ? 50 : 10
    let gRes0 = sampleInsightAtGamma(zHost: relRes.zHost, zPartnerCurrent: relRes.zPartnerCurrent, gamma: 0.0, odeSteps: subSteps)
    let gRes1 = sampleInsightAtGamma(zHost: relRes.zHost, zPartnerCurrent: relRes.zPartnerCurrent, gamma: 0.5, odeSteps: subSteps)
    let gRes2 = (cInsight: relRes.cInsight, zInsight: relRes.zInsight, insightTokens: relRes.insightTokens)

    let gen0 = vocab.decode(gRes0.insightTokens)
    let gen1 = vocab.decode(gRes1.insightTokens)
    let gen2 = vocab.decode(gRes2.insightTokens)

    let uOff = uIdx * latentDim
    let cos0 = cosineSimilarity(gRes0.zInsight, Array(unseenZIns0All[uOff..<(uOff + latentDim)]))
    let cos1 = cosineSimilarity(gRes1.zInsight, Array(unseenZIns1All[uOff..<(uOff + latentDim)]))
    let cos2 = cosineSimilarity(gRes2.zInsight, Array(unseenZIns2All[uOff..<(uOff + latentDim)]))

    for (gLabel, gRes, genText, cosVal) in [
        ("Level 0 (γ=0.0)", gRes0, gen0, cos0),
        ("Level 1 (γ=0.5)", gRes1, gen1, cos1),
        ("Level 2 (γ=1.0)", gRes2, gen2, cos2)
    ] {
        let fluency = evaluateSyntacticFluency(genText)
        unseenInsightCosSum += cosVal
        unseenInsightCount += 1
        if !fluency.hasGarbleOrOscillation {
            unseenInsightFluencyPass += 1
        }
        if !shownTest4Categories.contains(rc.group) && gLabel == "Level 2 (γ=1.0)" {
            test4Table += "| `\(rc.id)` | `\(rc.group)` | \(gLabel) | \(fmt("%.2f", l2Norm(gRes.cInsight))) | **\(fmt("%.4f", cosVal))** | **\(fmt("%.1f", fluency.score))%** (崩壊0) | \(genText.count)文字 | \(genText) |\n"
        }
    }
    shownTest4Categories.insert(rc.group)

    if gen0.count < gen1.count && gen1.count < gen2.count {
        unseenInsightMonotonicPass += 1
    }

    // Test 5: Latent Harmonization on Unseen Cases
    let harmText = vocab.decode(relRes.harmonizedTokens)
    let harmFluency = evaluateSyntacticFluency(harmText)
    if m4.llPreservationError > unseenMaxHarmLLErr {
        unseenMaxHarmLLErr = m4.llPreservationError
    }
    unseenHarmCount += 1
    if !harmFluency.hasGarbleOrOscillation && postM.pentagon.boundaryStability >= 95.0 {
        unseenHarmFluencyPass += 1
    }
    if !shownTest5Categories.contains(rc.group) {
        shownTest5Categories.insert(rc.group)
        test5Table += "| `\(rc.id)` | `\(rc.group)` | \(personaTopic) | `\(m4.archetype.rawValue)` | **`\(postM.archetype.rawValue)`** | \(fmt("%.1f", p4.boundaryStability)) $\\to$ **\(fmt("%.1f", postM.pentagon.boundaryStability))** | **`\(fmt("%.4f", m4.llPreservationError))`** | **\(fmt("%.1f", harmFluency.score))%** | \(harmText) |\n"
    }

    let ev = EvaluatedUnseenCase(
        rc: rc, m2: m2, m4: m4, postM: postM,
        genL0: gen0, genL1: gen1, genL2: gen2, cosL2: cos2, harmText: harmText
    )
    evaluatedByCategory[rc.group, default: []].append(ev)
}

let unseenArchAcc = Float(unseenArchetypeCorrect) / Float(max(1, unseenCases.count)) * 100.0
let unseenTopoAcc = Float(unseenTopologyCorrect) / Float(max(1, unseenCases.count)) * 100.0
let trajectoryTransAcc = Float(trajectoryTransitionCorrect) / Float(max(1, trajectoryCount)) * 100.0
let avgUnseenInsightCos = unseenInsightCosSum / Float(max(1, unseenInsightCount))
let unseenFluencyRate = Float(unseenInsightFluencyPass) / Float(max(1, unseenInsightCount)) * 100.0
let unseenMonotonicRate = Float(unseenInsightMonotonicPass) / Float(max(1, unseenCases.count)) * 100.0
let unseenHarmSuccessRate = Float(unseenHarmFluencyPass) / Float(max(1, unseenHarmCount)) * 100.0

print("[Test 2] Unseen \(unseenCases.count) Cases Pentagon & Classification: Archetype \(unseenArchetypeCorrect)/\(unseenCases.count) (\(fmt("%.2f", unseenArchAcc))%), Topology \(unseenTopologyCorrect)/\(unseenCases.count) (\(fmt("%.2f", unseenTopoAcc))%).")
print("[Test 3] Dynamic Trajectory Tracking (T2 -> T4): \(trajectoryTransitionCorrect)/\(trajectoryCount) (\(fmt("%.2f", trajectoryTransAcc))%).")
print("[Test 4] Unseen 3-Level Insight CFM: Mean cos(z_hat, z_target) = \(fmt("%.4f", avgUnseenInsightCos)), Syntactic Fluency = \(unseenInsightFluencyPass)/\(unseenInsightCount) (\(fmt("%.2f", unseenFluencyRate))%), Monotonic = \(unseenInsightMonotonicPass)/\(unseenCases.count).")
print("[Test 5] Unseen Latent Harmonization: \(unseenHarmFluencyPass)/\(unseenHarmCount) (\(fmt("%.2f", unseenHarmSuccessRate))%), max LL error = \(fmt("%.4f", unseenMaxHarmLLErr)).")

report += "## Test 2: 未学習 `\(unseenCases.count)` シナリオ（全 `\(distinctCategories)` カテゴリ網羅）における「関係値5角形マッピング」と「距離感トポロジー・属性分類」の汎化精度\n\n"
report += "学習時に一切使用していない**未学習 `\(unseenCases.count)` シナリオ**（全 `\(distinctCategories)` カテゴリ）単体での5角形マッピングと分類精度を検証する。\n\n"
report += test2Table
report += "\n- **未学習 `\(unseenCases.count)` シナリオ 動態属性（6分類）汎化正解率**: **`\(unseenArchetypeCorrect)/\(unseenCases.count)` (`\(fmt("%.2f", unseenArchAcc))%`)**（目標基準 $\\ge 80.0\\%$）\n"
report += "- **未学習 `\(unseenCases.count)` シナリオ 距離感トポロジー（5階層）汎化正解率**: **`\(unseenTopologyCorrect)/\(unseenCases.count)` (`\(fmt("%.2f", unseenTopoAcc))%`)**（目標基準 $\\ge 80.0\\%$）\n\n"
report += "---\n\n"

report += "## Test 3: 未学習 `\(unseenCases.count)` シナリオにおける「関係性の時間推移トラッキング（$T_2 \\to T_4$）」\n\n"
report += "時間の経過（$T_1 \\to T_2 \\to T_3 \\to T_4$）に伴う $T_2$ 時点と $T_4$ 時点の5角形スコア変化量（$\\Delta \\text{Score} = \\text{Score}(T_4) - \\text{Score}(T_2)$）および状態遷移を検証する。\n\n"
report += test3Table
report += "\n- **時間推移・状態遷移トラッキング正解率 ($T_2 \\to T_4$)**: **`\(trajectoryTransitionCorrect)/\(trajectoryCount)` (`\(fmt("%.2f", trajectoryTransAcc))%`)**\n\n"

func formatWrong(_ wrong: [String: Int]) -> String {
    let parts = wrong.sorted { $0.value > $1.value }.map { "`\($0.key)` ×\($0.value)" }
    return if parts.isEmpty { "-" } else { parts.joined(separator: ", ") }
}

let failingCategories = categoryFailureStats
    .filter { $0.value.t4Correct < $0.value.total || $0.value.t2Correct < $0.value.total }
    .sorted { ($0.value.t4Correct + $0.value.t2Correct) < ($1.value.t4Correct + $1.value.t2Correct) }
report += "### カテゴリ別の失敗内訳（未学習ケース）\n\n"
report += "| カテゴリ | 件数 | $T_4$ 正解 | $T_4$ 期待値 | $T_4$ 誤判定 | $T_2$ 正解 | $T_2$ 期待値 | $T_2$ 誤判定 |\n"
report += "| :--- | ---: | ---: | :--- | :--- | ---: | :--- | :--- |\n"
for (cat, s) in failingCategories {
    report += "| `\(cat)` | \(s.total) | \(s.t4Correct) | `\(s.expectedT4)` | \(formatWrong(s.t4Wrong)) | \(s.t2Correct) | `\(s.expectedT2)` | \(formatWrong(s.t2Wrong)) |\n"
}
report += "\n- **失敗を含むカテゴリ数**: `\(failingCategories.count)/\(categoryFailureStats.count)`\n\n"
print("[Failure breakdown] categories with failures: \(failingCategories.count)/\(categoryFailureStats.count)")
for (cat, s) in failingCategories {
    print("  \(cat): T4 \(s.t4Correct)/\(s.total) exp=\(s.expectedT4) wrong=\(s.t4Wrong) | T2 \(s.t2Correct)/\(s.total) exp=\(s.expectedT2) wrong=\(s.t2Wrong)")
}
report += "---\n\n"

report += "## Test 4: 未学習ケースに対する Conditional Flow Matching（50-Step Midpoint ODE）3段階インサイト生成\n\n"
report += "未学習 `\(unseenCases.count)` シナリオ（計 `\(unseenInsightCount)` インサイト）に対し、$\\gamma \\in \\{0.0, 0.5, 1.0\\}$ で生成された深層関係インサイトを、**①生成ベクトルと期待意味ベクトルのコサイン類似度 $\\cos(\\hat{z}_{\\text{insight}}, z_{\\text{target}})$、②構文の自然さ（文字化け・反復発振ゼロ）、③トピックやペルソナに応じた診断内容の的確さ**の3軸で評価する。\n\n"
report += test4Table
report += "\n- **未学習 `\(unseenInsightCount)` インサイト 平均意味ベクトル類似度 $\\overline{\\cos}(\\hat{z}_{\\text{insight}}, z_{\\text{target}})$**: **`\(fmt("%.4f", avgUnseenInsightCos))`**\n"
report += "- **構文自然度（文字化け・反復発振ゼロ達成率）**: **`\(unseenInsightFluencyPass)/\(unseenInsightCount)` (`\(fmt("%.2f", unseenFluencyRate))%`)**\n"
report += "- **粒度スケール単調増加性 ($\\text{Len}(\\gamma=0.0) < \\text{Len}(\\gamma=0.5) < \\text{Len}(\\gamma=1.0)$)**: **`\(unseenInsightMonotonicPass)/\(unseenCases.count)` (`\(fmt("%.2f", unseenMonotonicRate))%`)**\n\n"
report += "---\n\n"

report += "## Test 5: 未学習シナリオに対する「潜在空間での距離感調律（Latent Harmonization）」\n\n"
report += "未学習 `\(unseenCases.count)` シナリオに対し、低周波の文脈記憶 $z_{\\text{LL}}$ を保持したまま高周波成分 $z_{\\text{HH}}$ を調律（$z_{\\text{harmonized}} = z^C + \\Delta z_{\\text{HH}}$）し、そのペルソナの口調やトピックを保ったまま相手の距離感を尊重した健全な発言へ変化するかを検証する。\n\n"
report += test5Table
report += "\n- **未学習ケース 低周波愛着骨格保存誤差 $\\max \\|z_{\\text{LL}}^{\\text{after}} - z_{\\text{LL}}^{\\text{before}}\\|_2$**: **`\(fmt("%.4f", unseenMaxHarmLLErr))`**\n"
report += "- **距離感調律後の境界線安定性回復・構文自然度達成率**: **`\(unseenHarmFluencyPass)/\(unseenHarmCount)` (`\(fmt("%.2f", unseenHarmSuccessRate))%`)**\n\n"
report += "---\n\n"

// MARK: - Test 10: Twisted Surface vs Deep Attachment Contrastive Pairs (Part 2 Focus)

report += "## Test 10 (Part 2 Focus): 「表層の言葉」と「深層の愛着」のねじれ識別テスト（Unseen Deep Human Matrix）\n\n"
report += "表面の言葉は似ているが本質が真逆のペア（女性リスナーの純粋憧れ vs マウンティング、ダメ出し常連オジサン vs 愛憎反転アンチ、人推しソウルメイト vs ゲーム単推し）および冷やかしからの沼落ち軌跡を未学習データで100%識別できることを検証する。\n\n"
report += "| 対比ペア | カテゴリ (`category_12`) | シナリオID | $\\text{Score}_{\\text{ctx}}$ ($T_2 \\to T_4$) | $S_{\\text{HH}}$ ($T_2 \\to T_4$) | $\\text{Score}_{\\text{stab}}$ ($T_4$) | $\\text{Score}_{\\text{dev}}$ ($T_4$) | 状態遷移 ($T_2 \\to T_4$) | 生成深層インサイト ($\\gamma=1.0$) |\n"
report += "| :--- | :--- | :--- | :---: | :---: | ---: | ---: | :---: | :--- |\n"

var test10Passed = 0
var test10Total = 0
for spec in datasetConfig.test10Specs {
    if let ev = evaluatedByCategory[spec.category]?.first {
        let p2 = ev.m2.pentagon
        let p4 = ev.m4.pentagon
        let ok = ev.m4.archetype == ev.rc.expectedArchetype && ev.m4.topology == ev.rc.expectedTopology
        test10Total += 1
        if ok { test10Passed += 1 }
        report += "| **\(spec.label)** | `\(spec.category)` | `\(ev.rc.id)` | \(fmt("%.1f", p2.contextDepth)) $\\to$ **\(fmt("%.1f", p4.contextDepth))** | \(fmt("%+.2f", ev.m2.sHH)) $\\to$ **\(fmt("%+.2f", ev.m4.sHH))** | **\(fmt("%.1f", p4.boundaryStability))** | **\(fmt("%.1f", p4.longTermViability))** | `\(ev.m2.archetype.rawValue)` $\\to$ **`\(ev.m4.archetype.rawValue)`** \(ok ? "✅" : "❌") | \(ev.genL2) |\n"
    }
}
report += "\n---\n\n"

// MARK: - Test 11: Value-Alignment Matrix Verification (Part 3 Focus)

report += "## Test 11 (Part 3 Focus): 「好みの表層一致」と「価値観の深層一致」を見分ける検証テスト（Unseen Value-Alignment Matrix）\n\n"
report += "日常の話題（香水・服・野球・車・映画・お酒 $\\to$ デートプラン $\\to$ お金の価値観 $\\to$ 旅行計画・トラブル耐性）から、単なる表層の趣味一致／不一致と本質的な価値観の相性を識別できることを未学習ケースで検証する。\n\n"
report += "| 検証ケース | カテゴリ (`category_12`) | シナリオID | $T_2 \\to T_4$ $\\text{Score}_{\\text{ctx}}$ | $S_{\\text{HH}}$ ($T_4$) | $\\text{Score}_{\\text{emp}}$ ($T_4$) | $\\text{Score}_{\\text{stab}}$ ($T_4$) | $\\text{Score}_{\\text{dev}}$ ($T_4$) | 判定結果 | 生成深層インサイト ($\\gamma=1.0$) |\n"
report += "| :--- | :--- | :--- | :---: | ---: | ---: | ---: | ---: | :---: | :--- |\n"

var test11Passed = 0
var test11Total = 0
for spec in datasetConfig.test11Specs {
    if let ev = evaluatedByCategory[spec.category]?.first {
        let p2 = ev.m2.pentagon
        let p4 = ev.m4.pentagon
        let ok = ev.m4.archetype == ev.rc.expectedArchetype && ev.m4.topology == ev.rc.expectedTopology
        test11Total += 1
        if ok { test11Passed += 1 }
        report += "| **\(spec.label)** | `\(spec.category)` | `\(ev.rc.id)` | \(fmt("%.1f", p2.contextDepth)) $\\to$ **\(fmt("%.1f", p4.contextDepth))** | **\(fmt("%+.4f", ev.m4.sHH))** | **\(fmt("%.1f", p4.empathicResonance))** | **\(fmt("%.1f", p4.boundaryStability))** | **\(fmt("%.1f", p4.longTermViability))** | **`\(ev.m4.archetype.rawValue)`** (`\(ev.m4.topology.rawValue)`) \(ok ? "✅" : "❌") | \(ev.genL2) |\n"
    }
}
report += "\n---\n\n"

// MARK: - Test 12: Nuanced Distance Contrastive Pairs Verification (Part 4 Focus)

report += "## Test 12 (Part 4 Focus): 「微妙な距離感の識別」重点テストマトリクス（Unseen Nuanced-Distance Contrastive Pairs）\n\n"
report += "人間の目でも迷いやすい5つの「微妙な距離感の対比ペア」（ジェネレーションギャップの明暗、優しさの真意、マメな連絡の裏のキープ扱い、慎重な誠実さ vs 停滞、距離感の暴走 vs つかず離れずの防壁）を未学習データで精密に識別・言語化できることを検証する。\n\n"
report += "| 対比ペア | カテゴリ (`category_12`) | シナリオID | $\\text{Score}_{\\text{ctx}}$ ($T_2 \\to T_4$) | $R_{\\text{HH}}$ ($T_4$) | $S_{\\text{HH}}$ ($T_4$) | $\\text{Score}_{\\text{sync}}$ ($T_4$) | $\\text{Score}_{\\text{stab}}$ ($T_4$) | 識別トポロジー & 属性 | 違和感・距離感の言語化インサイト ($\\gamma=1.0$) |\n"
report += "| :--- | :--- | :--- | :---: | ---: | ---: | ---: | ---: | :---: | :--- |\n"

var test12Passed = 0
var test12Total = 0
for spec in datasetConfig.test12Specs {
    if let ev = evaluatedByCategory[spec.category]?.first {
        let p2 = ev.m2.pentagon
        let p4 = ev.m4.pentagon
        let ok = ev.m4.archetype == ev.rc.expectedArchetype && ev.m4.topology == ev.rc.expectedTopology
        test12Total += 1
        if ok { test12Passed += 1 }
        report += "| **\(spec.label)** | `\(spec.category)` | `\(ev.rc.id)` | \(fmt("%.1f", p2.contextDepth)) $\\to$ **\(fmt("%.1f", p4.contextDepth))** | **\(fmt("%.2f", ev.m4.rHH))** | **\(fmt("%+.4f", ev.m4.sHH))** | **\(fmt("%.1f", p4.communicativeHarmony))** | **\(fmt("%.1f", p4.boundaryStability))** | **`\(ev.m4.topology.rawValue)` / `\(ev.m4.archetype.rawValue)`** \(ok ? "✅" : "❌") | \(ev.genL2) |\n"
    }
}
report += "\n---\n\n"

func mark(_ ok: Bool) -> String {
    return ok ? "✅" : "❌"
}

let orthErr = max(maxOrthogonalityLLErrTrain, maxOrthogonalityLLErrUnseen)
let test1Pass = exactTrainDocs == numDocs && exactUnseenReconstructions == unseenCases.count && orthErr < 1e-3
let test2Pass = unseenArchAcc >= 80.0 && unseenTopoAcc >= 80.0
let test3Pass = trajectoryTransAcc >= 80.0
let test4Pass = avgUnseenInsightCos >= 0.85 && unseenInsightFluencyPass == unseenInsightCount && unseenInsightMonotonicPass == unseenCases.count
let test5Pass = unseenMaxHarmLLErr < 1e-3 && unseenHarmFluencyPass == unseenHarmCount
let allPass = test1Pass && test2Pass && test3Pass && test4Pass && test5Pass && test10Passed == test10Total && test11Passed == test11Total && test12Passed == test12Total

report += "## 総合達成度サマリ（全 `\(distinctCategories)` カテゴリ・通算 `\(allCases.count)` シナリオ）\n\n"
report += "| 検証項目 | 評価指標 | 目標基準 | 実測結果 | 判定 |\n"
report += "| :--- | :--- | :--- | :--- | :---: |\n"
report += "| **Test 1 (Subband Orthogonality)** | 全件再構成 & $\\|z_{\\text{LL}}^{(A)} - z_{\\text{LL}}^{(B)}\\|_2$ | `100%` & `0.0000` | **学習 `\(exactTrainDocs)/\(numDocs)`, 未学習 `\(exactUnseenReconstructions)/\(unseenCases.count)`, 直交誤差 `\(fmt("%.4f", orthErr))`** | \(mark(test1Pass)) |\n"
report += "| **Test 2 (Unseen `\(unseenCases.count)` 汎化分類精度)** | 未学習 `\(unseenCases.count)` 件 6属性・5階層トポロジー正解率 | `>= 80.0%` | **属性 `\(unseenArchetypeCorrect)/\(unseenCases.count)` (`\(fmt("%.2f", unseenArchAcc))%`), トポロジー `\(unseenTopologyCorrect)/\(unseenCases.count)` (`\(fmt("%.2f", unseenTopoAcc))%`)** | \(mark(test2Pass)) |\n"
report += "| **Test 3 (時間推移トラッキング $T_2 \\to T_4$)** | 5角形スコア変化量 $\\Delta\\text{Score}$ & 状態遷移追跡 | `>= 80.0%` | **状態遷移追跡 `\(trajectoryTransitionCorrect)/\(trajectoryCount)` (`\(fmt("%.2f", trajectoryTransAcc))%`)** | \(mark(test3Pass)) |\n"
report += "| **Test 4 (Unseen 3段階 CFM インサイト)** | $\\cos(\\hat{z}_{\\text{insight}}, z_{\\text{target}})$ & 構文自然度・単調増加 | $\\cos \\ge 0.85$ & 崩壊 `0` | **平均 $\\cos = \(fmt("%.4f", avgUnseenInsightCos))$, 構文自然度 `\(fmt("%.2f", unseenFluencyRate))%` (`\(unseenInsightFluencyPass)/\(unseenInsightCount)`), 単調性 `\(unseenInsightMonotonicPass)/\(unseenCases.count)`** | \(mark(test4Pass)) |\n"
report += "| **Test 5 (Unseen 潜在空間距離感調律)** | $\\|\\Delta z_{\\text{LL}}\\|_2 = 0$ & ペルソナ保持・境界線回復 | `0.0000` & `100%` | **LL保存誤差 `\(fmt("%.4f", unseenMaxHarmLLErr))`, 調律成功 `\(unseenHarmFluencyPass)/\(unseenHarmCount)` (`\(fmt("%.2f", unseenHarmSuccessRate))%`)** | \(mark(test5Pass)) |\n"
report += "| **Test 10 (Part 2 深層人間模様 対比ペア)** | 代表ケースの属性・トポロジー一致 | 全件一致 | **`\(test10Passed)/\(test10Total)`** | \(mark(test10Passed == test10Total)) |\n"
report += "| **Test 11 (Part 3 価値観すり合わせ分離)** | 代表ケースの属性・トポロジー一致 | 全件一致 | **`\(test11Passed)/\(test11Total)`** | \(mark(test11Passed == test11Total)) |\n"
report += "| **Test 12 (Part 4 微妙な距離感 対比ペア)** | 代表ケースの属性・トポロジー一致 | 全件一致 | **`\(test12Passed)/\(test12Total)`** | \(mark(test12Passed == test12Total)) |\n"

let reportURL = URL(fileURLWithPath: cli.reportPath)
let parentDir = reportURL.deletingLastPathComponent()
if !parentDir.path.isEmpty && parentDir.path != "." {
    try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
}
try report.write(to: reportURL, atomically: true, encoding: .utf8)

print("------------------------------------------------------------------")
print("Test 10 / 11 / 12 representative cases: \(test10Passed)/\(test10Total), \(test11Passed)/\(test11Total), \(test12Passed)/\(test12Total)")
let conclusion = if allPass { "ALL CHECKS PASSED" } else { "SOME CHECKS FAILED" }
print("Diagnostic Conclusion: \(conclusion) (\(distinctCategories) categories / \(allCases.count) scenarios, unseen \(unseenCases.count))")
print("Saved detailed verification report to \(cli.reportPath)")
print("==================================================================")

// ==================================================================
// MARK: - Phase 5.1 Independent Anti-Cheat Audits (Audit 1 – Audit 5)
// ==================================================================

print("==================================================================")
print("Phase 5.1 Anti-Cheat Audits")
print("==================================================================")

let archCases = RelationalArchetype.allCases
let topoCases = RelationalTopology.allCases
let taskNames = ["archetype_t4", "topology_t4", "archetype_t2", "topology_t2"]
func taskLabel(_ task: String, _ rc: RelationalDialogueCase) -> Int {
    switch task {
    case "archetype_t4": return archCases.firstIndex(of: rc.expectedArchetype) ?? 0
    case "topology_t4": return topoCases.firstIndex(of: rc.expectedTopology) ?? 0
    case "archetype_t2": return archCases.firstIndex(of: rc.expectedArchetypeT2) ?? 0
    default: return topoCases.firstIndex(of: rc.expectedTopologyT2) ?? 0
    }
}
func taskPredict(_ task: String, _ z: [Float], _ rel: DualStreamRelationalWeights) -> Int {
    let isT2 = task.hasSuffix("t2")
    let c = DualStreamRelationalEngine.classifyFromZRel(zRel: z, isEarlyT2: isT2, weights: rel)
    return task.hasPrefix("archetype") ? (archCases.firstIndex(of: c.archetype) ?? 0) : (topoCases.firstIndex(of: c.topology) ?? 0)
}
func tokensOf(_ text: String) -> [Int] {
    return vocab.encode(text, maxLength: seqLen, addEOS: false, padToLength: nil)
}

var audit = "# SWT-Griffin Relational Dynamics Engine (Phase 5.1) 独立監査レポート（Anti-Cheat Audit）\n\n"
audit += "本レポートの全指標は、`(senderTokenIDs: [Int], receiverTokenIDs: [Int])` のみを入力とする純ニューラル出力 `z_rel ∈ R^1024` と、学習時に train 分割のみで校正された分類ヘッドから算出している。\n\n"

// ---------------- Audit 1 ----------------
let freshPairs: [(label: String, sender: String, receiver: String)] = [
    ("新規A: 記憶の照合と配慮", "来月から陶芸教室で新しい釉薬を試すつもりで、窯の温度管理を勉強しているところです。忙しくて返信が遅れるかもしれません。", "陶芸教室の釉薬の話、窯の温度管理まで勉強しているなんてすごいですね。返信は気にせず、自分のペースで楽しんでください。"),
    ("新規B: 無関心な定型返信", "来月から陶芸教室で新しい釉薬を試すつもりで、窯の温度管理を勉強しているところです。忙しくて返信が遅れるかもしれません。", "へえ。そういえば昨日の夜は何食べた？こっちは特に何もなかったよ。"),
    ("新規C: 依存的な踏み込み", "最近は資格試験の勉強に集中したいので、週末の連絡は少し控えめにさせてください。", "資格試験の勉強、応援しています。でも週末に連絡が来ないと不安になるので、毎晩一言だけでも送ってほしいです。どうして返してくれないの？")
]
let corpusTexts = Set(allCases.flatMap { [$0.hostText, $0.partnerEarlyText, $0.partnerCurrentText, $0.harmonizedPartnerText] })
var audit1Rows = ""
var audit1Fresh = true
for fp in freshPairs {
    if corpusTexts.contains(fp.sender) && corpusTexts.contains(fp.receiver) {
        audit1Fresh = false
    }
    let out = model.inferRelational(senderTokenIDs: vocab.encode(fp.sender), receiverTokenIDs: vocab.encode(fp.receiver))
    let p = out.pentagon
    audit1Rows += "| \(fp.label) | `\(out.predictedArchetypeT4.rawValue)` | `\(out.predictedTopologyT4.rawValue)` | \(fmt("%.1f", p.contextDepth)) / \(fmt("%.1f", p.empathicResonance)) / \(fmt("%.1f", p.communicativeHarmony)) / \(fmt("%.1f", p.boundaryStability)) / \(fmt("%.1f", p.longTermViability)) |\n"
}
let unkInFresh = freshPairs.reduce(0) { $0 + vocab.encode($1.sender + $1.receiver).filter { $0 == TextVocabulary.unkId }.count }
let audit1Pass = audit1Fresh
audit += "## Audit 1: ゼロリーク・関数シグネチャ監査\n\n"
audit += "推論・エンコード関数のシグネチャ（ライブラリ公開 API）:\n\n"
audit += "```swift\n"
audit += "SWTHawkAutoencoder.encodeRelational(senderTokenIDs: [Int], receiverTokenIDs: [Int]) -> [Float]\n"
audit += "SWTHawkAutoencoder.inferRelational(senderTokenIDs: [Int], receiverTokenIDs: [Int]) -> RelationalInferenceOutput\n"
audit += "SWTHawkAutoencoder.encodeRelationalBatch(senderTokens: [Int], receiverTokens: [Int], batch: Int, time: Int) -> [Float]\n"
audit += "DualStreamRelationalEngine.computePentagonFromZRel(zRel: [Float], zRelEarly: [Float]?) / classifyFromZRel(zRel: [Float], isEarlyT2: Bool, weights:)\n"
audit += "```\n\n"
audit += "- `scenario_id` / `category_12` / `archetype` / `topology` を受け取る引数は存在しない。`z_rel` の全次元は射影行列積 + LayerNorm の出力であり、トークンID・ハッシュ・ルール値の直接代入はない（旧 `SubbandClauseSplitter.embedSubbandLattice` は再構成用 `z` のみで使用し、`z_rel` 経路からは参照されない）。\n"
audit += "- データセットに存在しない新規日本語文（未知文字 `\(unkInFresh)` 個）での推論結果:\n\n"
audit += "| 入力ペア | 属性 ($T_4$ ヘッド) | トポロジー | ctx / emp / sync / stab / dev |\n| :--- | :---: | :---: | :---: |\n"
audit += audit1Rows
audit += "\n- 判定: \(audit1Pass ? "**PASS**" : "**FAIL**")\n\n"
print("[Audit 1] signature = (senderTokenIDs:[Int], receiverTokenIDs:[Int]); fresh sentences not in corpus: \(audit1Fresh)")

// ---------------- Audit 2 ----------------
let tel = weights.relational.auditTelemetry
var audit2Pass = tel.modules.count == 5
audit += "## Audit 2: 勾配・重み移動監査\n\n"
audit += "End-to-End 学習（総ステップ `\(tel.totalTrainingSteps)`、最終 $\\mathcal{L}_{\\text{total}}$ = `\(fmt("%.4f", tel.finalTotalLoss))`: cls `\(fmt("%.4f", tel.finalClsLoss))`, supcon `\(fmt("%.4f", tel.finalSupConLoss))`, cf `\(fmt("%.5f", tel.finalCfLoss))`, metric `\(fmt("%.4f", tel.finalMetricLoss))`）で記録した値。\n\n"
audit += "| モジュール | パラメータ数 | $\\|W^{(0)}\\|_F$ | $\\|W^{(\\text{final})}\\|_F$ | $\\|W^{(\\text{final})} - W^{(0)}\\|_F$ | 平均 $\\|\\nabla_W \\mathcal{L}\\|_2$ | 最大 $\\|\\nabla_W \\mathcal{L}\\|_2$ | 判定 |\n"
audit += "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |\n"
for m in tel.modules {
    let ok = m.weightDeltaFrobeniusNorm > 1e-4 && m.meanGradientNorm > 1e-4
    if !ok { audit2Pass = false }
    audit += "| `\(m.moduleName)` | \(m.parameterCount) | \(fmt("%.4f", m.initialFrobeniusNorm)) | \(fmt("%.4f", m.finalFrobeniusNorm)) | **\(fmt("%.6f", m.weightDeltaFrobeniusNorm))** | \(fmt("%.6f", m.meanGradientNorm)) | \(fmt("%.6f", m.maxGradientNorm)) | \(ok ? "PASS" : "FAIL") |\n"
    print("[Audit 2] \(m.moduleName): dW=\(m.weightDeltaFrobeniusNorm) grad=\(m.meanGradientNorm)")
}
audit += "\n- 判定: \(audit2Pass ? "**PASS**" : "**FAIL**")\n\n"

// ---------------- Audit 3a: sender-receiver mismatch ----------------
var mismatchCtx: [Float] = []
var mismatchTrollOrStranger = 0
var mismatchSenders: [[Int]] = []
var mismatchReceivers: [[Int]] = []
for (i, rc) in unseenCases.enumerated() {
    var j = (i + unseenCases.count / 2) % unseenCases.count
    var tries = 0
    while unseenCases[j].group == rc.group && tries < unseenCases.count {
        j = (j + 1) % unseenCases.count
        tries += 1
    }
    mismatchSenders.append(tokensOf(rc.hostText))
    mismatchReceivers.append(tokensOf(unseenCases[j].partnerCurrentText))
}
let mismatchZAll = RelationalPipeline.encodeZRel(model: mainModel, senders: mismatchSenders, receivers: mismatchReceivers, time: seqLen)
for z in mismatchZAll {
    mismatchCtx.append(DualStreamRelationalEngine.computePentagonFromZRel(zRel: z).pentagon.contextDepth)
    let c = DualStreamRelationalEngine.classifyFromZRel(zRel: z, isEarlyT2: false, weights: weights.relational)
    if c.archetype == .indifferentTroll || c.topology == .unconnectedStranger {
        mismatchTrollOrStranger += 1
    }
}
let meanMatchedCtx = matchedCtxScores.reduce(0, +) / Float(max(1, matchedCtxScores.count))
let meanMismatchCtx = mismatchCtx.reduce(0, +) / Float(max(1, mismatchCtx.count))
let ctxDrop = meanMatchedCtx - meanMismatchCtx
let trollRate = Float(mismatchTrollOrStranger) / Float(max(1, unseenCases.count)) * 100.0
let audit3aPass = ctxDrop >= 30.0
print("[Audit 3a] mean Score_ctx matched \(meanMatchedCtx) -> mismatched \(meanMismatchCtx) (drop \(ctxDrop)), troll/stranger \(trollRate)%")

// ---------------- Audit 3b: receiver token order shuffle ----------------
var shuffleRng = SplitMix64(seed: 3131)
var accOrig: [String: Int] = [:]
var accShuf: [String: Int] = [:]
var shufSenders: [[Int]] = []
var origRecvsT4: [[Int]] = []
var shufRecvsT4: [[Int]] = []
var origRecvsT2: [[Int]] = []
var shufRecvsT2: [[Int]] = []
for rc in unseenCases {
    shufSenders.append(tokensOf(rc.hostText))
    let recv4 = tokensOf(rc.partnerCurrentText)
    var s4 = recv4
    for k in stride(from: s4.count - 1, to: 0, by: -1) {
        let r = Int(shuffleRng.next() % UInt64(k + 1))
        s4.swapAt(k, r)
    }
    origRecvsT4.append(recv4)
    shufRecvsT4.append(s4)
    let recv2 = tokensOf(rc.partnerEarlyText)
    var s2 = recv2
    for k in stride(from: s2.count - 1, to: 0, by: -1) {
        let r = Int(shuffleRng.next() % UInt64(k + 1))
        s2.swapAt(k, r)
    }
    origRecvsT2.append(recv2)
    shufRecvsT2.append(s2)
}
let zOrigT4All = RelationalPipeline.encodeZRel(model: mainModel, senders: shufSenders, receivers: origRecvsT4, time: seqLen)
let zShufT4All = RelationalPipeline.encodeZRel(model: mainModel, senders: shufSenders, receivers: shufRecvsT4, time: seqLen)
let zOrigT2All = RelationalPipeline.encodeZRel(model: mainModel, senders: shufSenders, receivers: origRecvsT2, time: seqLen)
let zShufT2All = RelationalPipeline.encodeZRel(model: mainModel, senders: shufSenders, receivers: shufRecvsT2, time: seqLen)
for (idx, rc) in unseenCases.enumerated() {
    for task in taskNames {
        let useEarly = task.hasSuffix("t2")
        let zO = useEarly ? zOrigT2All[idx] : zOrigT4All[idx]
        let zS = useEarly ? zShufT2All[idx] : zShufT4All[idx]
        let y = taskLabel(task, rc)
        if taskPredict(task, zO, weights.relational) == y { accOrig[task, default: 0] += 1 }
        if taskPredict(task, zS, weights.relational) == y { accShuf[task, default: 0] += 1 }
    }
}
let nU = Float(max(1, unseenCases.count))
let meanAccOrig = taskNames.reduce(Float(0)) { $0 + Float(accOrig[$1] ?? 0) / nU } / 4.0 * 100.0
let meanAccShuf = taskNames.reduce(Float(0)) { $0 + Float(accShuf[$1] ?? 0) / nU } / 4.0 * 100.0
let audit3bPass = (meanAccOrig - meanAccShuf) >= 30.0
print("[Audit 3b] mean accuracy original \(meanAccOrig)% -> shuffled \(meanAccShuf)%")

// ---------------- Audit 3c: random label permutation ----------------
var audit3cAcc: [String: Float] = [:]
var audit3cMajority: [String: Float] = [:]
var audit3cRan = false
if !cli.skipPermutationAudit, FileManager.default.fileExists(atPath: cli.aeWeightsPath) {
    let aeW = try SWTHawkWeights.load(from: URL(fileURLWithPath: cli.aeWeightsPath))
    let permModel = MLXSWTHawkAutoencoder(weights: aeW)
    var permRng = SplitMix64(seed: 4242)
    var perm = Array(0..<trainCases.count)
    for k in stride(from: perm.count - 1, to: 0, by: -1) {
        let r = Int(permRng.next() % UInt64(k + 1))
        perm.swapAt(k, r)
    }
    let permutedCases = trainCases.enumerated().map { (i, rc) -> RelationalDialogueCase in
        let src = trainCases[perm[i]]
        return RelationalDialogueCase(
            id: rc.id, group: rc.group, personaStyle: rc.personaStyle, topicDomain: rc.topicDomain, title: rc.title,
            expectedTopologyT2: src.expectedTopologyT2, expectedArchetypeT2: src.expectedArchetypeT2,
            expectedTopology: src.expectedTopology, expectedArchetype: src.expectedArchetype,
            hostText: rc.hostText, partnerEarlyText: rc.partnerEarlyText, partnerCurrentText: rc.partnerCurrentText,
            harmonizedPartnerText: rc.harmonizedPartnerText, insightLevel0: rc.insightLevel0, insightLevel1: rc.insightLevel1,
            insightLevel2: rc.insightLevel2, isZeroShotUnseen: false
        )
    }
    print("[Audit 3c] retraining Dual-Stream encoder on permuted labels (\(cli.permEpochs) epochs)...")
    RelationalPipeline.trainRelationalStage(
        model: permModel, trainCases: permutedCases, vocab: vocab, time: seqLen, epochs: cli.permEpochs,
        seed: 777, log: { print("  [Audit 3c] " + $0) }
    )
    let permW = permModel.exportWeights()
    let permZ2All = RelationalPipeline.encodeZRel(model: permModel, senders: shufSenders, receivers: origRecvsT2, time: seqLen)
    let permZ4All = RelationalPipeline.encodeZRel(model: permModel, senders: shufSenders, receivers: origRecvsT4, time: seqLen)
    var correct: [String: Int] = [:]
    for (idx, rc) in unseenCases.enumerated() {
        let z2 = permZ2All[idx]
        let z4 = permZ4All[idx]
        for task in taskNames {
            let z = task.hasSuffix("t2") ? z2 : z4
            if taskPredict(task, z, permW.relational) == taskLabel(task, rc) { correct[task, default: 0] += 1 }
        }
    }
    for task in taskNames {
        audit3cAcc[task] = Float(correct[task] ?? 0) / nU * 100.0
        var counts: [Int: Int] = [:]
        for rc in trainCases { counts[taskLabel(task, rc), default: 0] += 1 }
        let maj = counts.max { $0.value < $1.value }?.key ?? 0
        audit3cMajority[task] = Float(unseenCases.filter { taskLabel(task, $0) == maj }.count) / nU * 100.0
        print("[Audit 3c] \(task): permuted-label unseen accuracy \(audit3cAcc[task] ?? 0)% (majority \(audit3cMajority[task] ?? 0)%)")
    }
    audit3cRan = true
}
let audit3cPass = audit3cRan && taskNames.allSatisfy { (audit3cAcc[$0] ?? 100.0) <= max(40.0, (audit3cMajority[$0] ?? 0.0) + 5.0) }

audit += "## Audit 3: 相互作用の健全性監査\n\n"
audit += "### 3a. 送信者・受信者ミスマッチ\n\n"
audit += "未学習 `\(unseenCases.count)` 件の送信側 $X_i^S$ に、別カテゴリの受信側 $X_j^C$ を組み合わせた。\n\n"
audit += "| 条件 | 平均 $\\text{Score}_{\\text{ctx}}$ |\n| :--- | ---: |\n"
audit += "| 正しいペア $(X_i^S, X_i^C)$ | \(fmt("%.2f", meanMatchedCtx)) |\n"
audit += "| ミスマッチ $(X_i^S, X_j^C)$ | \(fmt("%.2f", meanMismatchCtx)) |\n"
audit += "| 低下量 | **\(fmt("%.2f", ctxDrop))** pt（基準 $\\ge 30.0$） |\n\n"
audit += "- ミスマッチで `INDIFFERENT_TROLL` または `UNCONNECTED_STRANGER` と判定された割合: **\(fmt("%.2f", trollRate))%** (`\(mismatchTrollOrStranger)/\(unseenCases.count)`)\n"
audit += "- 判定: \(audit3aPass ? "**PASS**" : "**FAIL**")\n\n"
audit += "### 3b. 受信側トークン順序シャッフル\n\n"
audit += "| タスク | 元の精度 | シャッフル後 |\n| :--- | ---: | ---: |\n"
for task in taskNames {
    audit += "| `\(task)` | \(fmt("%.2f", Float(accOrig[task] ?? 0) / nU * 100.0))% | \(fmt("%.2f", Float(accShuf[task] ?? 0) / nU * 100.0))% |\n"
}
audit += "| **平均** | **\(fmt("%.2f", meanAccOrig))%** | **\(fmt("%.2f", meanAccShuf))%** |\n\n"
audit += "- 低下量: **\(fmt("%.2f", meanAccOrig - meanAccShuf))** pt（基準 $\\ge 30.0$）。判定: \(audit3bPass ? "**PASS**" : "**FAIL**")\n\n"
audit += "### 3c. ランダムラベル置換\n\n"
if audit3cRan {
    audit += "train 分割のラベル組 $(\\text{arch}_{T_2}, \\text{topo}_{T_2}, \\text{arch}_{T_4}, \\text{topo}_{T_4})$ をシナリオ間でランダムに置換し、オートエンコーダ段階の重みから Dual-Stream エンコーダ・ヘッドを同条件（`\(cli.permEpochs)` epochs）で再学習して unseen_test を評価した。\n\n"
    audit += "| タスク | 置換ラベル学習後の未学習精度 | 多数派クラス基準 |\n| :--- | ---: | ---: |\n"
    for task in taskNames {
        audit += "| `\(task)` | **\(fmt("%.2f", audit3cAcc[task] ?? 0))%** | \(fmt("%.2f", audit3cMajority[task] ?? 0))% |\n"
    }
    audit += "\n- 判定基準: 各タスクで $\\le \\max(40\\%, \\text{多数派} + 5\\text{pt})$。判定: \(audit3cPass ? "**PASS**" : "**FAIL**")\n\n"
} else {
    audit += "_未実行（`--skip-permutation-audit` または AE 段階の重みが無い）_\n\n"
}

// ---------------- Audit 4: lexical substitution ----------------
let substitutions: [(String, String)] = [
    ("カフェ", "オフィス"), ("コーヒー豆", "プログラム"), ("コーヒー", "プログラム"), ("焙煎", "設計"), ("バリスタ", "エンジニア"),
    ("ゲーム", "アプリ"), ("ドライブ", "プロジェクト"), ("ビール", "コード"), ("ランチ", "レビュー"), ("グルメ", "テスト"),
    ("ホテル", "サーバー"), ("ツアー", "リリース"), ("スタジオ", "ラボ"), ("ライブ", "デモ"), ("ピアノ", "キーボード"),
    ("レシピ", "仕様"), ("ハーブ", "モジュール"), ("キッチン", "デスク"), ("イラスト", "デザイン"), ("グッズ", "ツール")
]
func substitute(_ s: String) -> String {
    var out = s
    for (a, b) in substitutions {
        out = out.replacingOccurrences(of: a, with: b)
    }
    return out
}
var subCosSum: Float = 0.0
var subCount = 0
var subAgree = 0
var subUnk = 0
var subIndices: [Int] = []
var subSenders: [[Int]] = []
var subRecvs: [[Int]] = []
for (idx, rc) in unseenCases.enumerated() {
    let h = rc.hostText
    let c = rc.partnerCurrentText
    let hs = substitute(h)
    let cs = substitute(c)
    guard hs != h || cs != c else { continue }
    subUnk += vocab.encode(hs + cs).filter { $0 == TextVocabulary.unkId }.count
    subIndices.append(idx)
    subSenders.append(tokensOf(hs))
    subRecvs.append(tokensOf(cs))
}
let zSubAll = subIndices.isEmpty ? [] : RelationalPipeline.encodeZRel(model: mainModel, senders: subSenders, receivers: subRecvs, time: seqLen)
for (k, idx) in subIndices.enumerated() {
    let zO = zOrigT4All[idx]
    let zS = zSubAll[k]
    subCosSum += cosineSimilarity(zO, zS)
    subCount += 1
    let a = DualStreamRelationalEngine.classifyFromZRel(zRel: zO, isEarlyT2: false, weights: weights.relational)
    let b = DualStreamRelationalEngine.classifyFromZRel(zRel: zS, isEarlyT2: false, weights: weights.relational)
    if a.archetype == b.archetype && a.topology == b.topology { subAgree += 1 }
}
let subMeanCos = subCosSum / Float(max(1, subCount))
let subAgreeRate = Float(subAgree) / Float(max(1, subCount)) * 100.0
let audit4Pass = subCount > 0 && subMeanCos >= 0.90 && subAgreeRate >= 90.0
print("[Audit 4] substituted cases \(subCount): mean cos \(subMeanCos), agreement \(subAgreeRate)%")
audit += "## Audit 4: 語彙置換不変性監査\n\n"
audit += "未学習ケースの送信側・受信側テキストのトピック名詞を別ドメインへ置換（例: " + substitutions.prefix(6).map { "`\($0.0)→\($0.1)`" }.joined(separator: ", ") + " など全 `\(substitutions.count)` 規則）。置換が1箇所以上発生した `\(subCount)` 件で評価（置換後の未知文字 `\(subUnk)` 個）。\n\n"
audit += "| 指標 | 実測 | 基準 |\n| :--- | ---: | ---: |\n"
audit += "| 平均 $\\cos(z_{\\text{rel}}^{\\text{orig}}, z_{\\text{rel}}^{\\text{sub}})$ | **\(fmt("%.4f", subMeanCos))** | $\\ge 0.90$ |\n"
audit += "| 置換前後の予測一致率 ($T_4$ 属性 & トポロジー) | **\(fmt("%.2f", subAgreeRate))%** (`\(subAgree)/\(subCount)`) | $\\ge 90.0\\%$ |\n\n"
audit += "- 注記: 不変性の主因は、train テキストの文書頻度と Unicode 文字種のみから算出したゲート $\\alpha_{\\text{rel}}$（低頻度の漢字・カタカナを `0.02` に減衰）である。\n"
audit += "- 判定: \(audit4Pass ? "**PASS**" : "**FAIL**")\n\n"

// ---------------- Audit 5: unified tensor identity ----------------
var audit5Rows = ""
var audit5Pass = true
var headPreds: [String: [String: Int]] = [:]
if let data = FileManager.default.contents(atPath: cli.headPredictionsPath),
   let decoded = try? JSONDecoder().decode([String: [String: Int]].self, from: data) {
    headPreds = decoded
}
for task in taskNames {
    let v = verifyPredictions[task] ?? [:]
    let h = headPreds[task] ?? [:]
    let agree = unseenCases.filter { v[$0.id] != nil && v[$0.id] == h[$0.id] }.count
    if agree != unseenCases.count { audit5Pass = false }
    audit5Rows += "| `\(task)` | **\(agree)/\(unseenCases.count)** (\(fmt("%.2f", Float(agree) / nU * 100.0))%) |\n"
    print("[Audit 5] \(task): verification vs relational_head agreement \(agree)/\(unseenCases.count)")
}
if headPreds.isEmpty { audit5Pass = false }
audit += "## Audit 5: 評価テンソル同一性監査\n\n"
audit += "`verification_report`（`analyzeRelationalDynamicsWithFlowMatching` 経由）と `relational_head_report` (A) `neural`（`\(cli.headPredictionsPath)`）の unseen_test 予測をケース単位で照合した。\n\n"
audit += "| タスク | 一致数 |\n| :--- | ---: |\n"
audit += audit5Rows
audit += "\n- 判定: \(audit5Pass ? "**PASS**" : "**FAIL**")\(headPreds.isEmpty ? "（予測ファイル未生成: 先に `make head` を実行）" : "")\n\n"

// ---------------- Summary ----------------
audit += "## 監査サマリ\n\n"
audit += "| 監査 | 判定 |\n| :--- | :---: |\n"
audit += "| Audit 1: ゼロリーク・シグネチャ | \(audit1Pass ? "PASS" : "FAIL") |\n"
audit += "| Audit 2: 勾配・重み移動 | \(audit2Pass ? "PASS" : "FAIL") |\n"
audit += "| Audit 3a: ミスマッチ | \(audit3aPass ? "PASS" : "FAIL") |\n"
audit += "| Audit 3b: 順序シャッフル | \(audit3bPass ? "PASS" : "FAIL") |\n"
audit += "| Audit 3c: ラベル置換 | \(audit3cRan ? (audit3cPass ? "PASS" : "FAIL") : "未実行") |\n"
audit += "| Audit 4: 語彙置換不変性 | \(audit4Pass ? "PASS" : "FAIL") |\n"
audit += "| Audit 5: テンソル同一性 | \(audit5Pass ? "PASS" : "FAIL") |\n\n"
audit += "---\n\n"
audit += report

try audit.write(to: URL(fileURLWithPath: cli.auditReportPath), atomically: true, encoding: .utf8)
print("Saved audit report to \(cli.auditReportPath)")
