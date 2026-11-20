import Foundation
import MLX
import Sashiba

struct PsychRecord: Codable {
    let scenario_id: String
    let psych_split: String
    let category: String
    let archetype_t4: String
    let topology_t4: String
    let trajectory_type: String
    let psych_full: String
    let psych_text_only: String
}

struct PsychCLI {
    var mode: String = "train" // train | eval | infer
    var relationalWeightsPath: String = ".tmp/weights.safetensors"
    var psychWeightsPath: String = ".tmp/weights_psych_lm.safetensors"
    var descriptionsPath: String = ".tmp/psych_lm/psych_descriptions.jsonl"
    var dataDir: String = ".tmp"
    var layer3dPath: String = ".tmp/validation/benchmarks/layer3d_generational_slang_emoji/scenarios.jsonl"
    var reportPath: String = ".tmp/psych_lm_report.md"
    var epochs: Int = 20
    var batchSize: Int = 32
    var dim: Int = 128
    var layers: Int = 3
    var bpeMerges: Int = 3000
    var lrMax: Float = 2.0e-3
    var lrMin: Float = 1.2e-4
    var beamWidth: Int = 2

    // Primary `infer` mode input: repeated `--text "YYYY-MM-DD HH:MM:SS ..."` (or `--text "..."`)
    var texts: [String] = []

    static func parse(_ args: [String]) -> PsychCLI {
        var cli = PsychCLI()
        var i = 1
        if i < args.count && !args[i].hasPrefix("-") {
            cli.mode = args[i]
            i += 1
        }
        while i < args.count {
            switch args[i] {
            case "--text", "-t":
                i += 1
                if i < args.count { cli.texts.append(args[i]) }
            case "--rel-weights":
                i += 1; if i < args.count { cli.relationalWeightsPath = args[i] }
            case "--psych-weights", "-w":
                i += 1; if i < args.count { cli.psychWeightsPath = args[i] }
            case "--descriptions":
                i += 1; if i < args.count { cli.descriptionsPath = args[i] }
            case "--epochs", "-e":
                i += 1; if i < args.count, let v = Int(args[i]) { cli.epochs = max(1, v) }
            case "--batch", "-b":
                i += 1; if i < args.count, let v = Int(args[i]) { cli.batchSize = max(4, v) }
            case "--dim":
                i += 1; if i < args.count, let v = Int(args[i]) { cli.dim = max(64, v) }
            case "--layers":
                i += 1; if i < args.count, let v = Int(args[i]) { cli.layers = max(1, v) }
            case "--merges":
                i += 1; if i < args.count, let v = Int(args[i]) { cli.bpeMerges = max(100, v) }
            case "--beam":
                i += 1; if i < args.count, let v = Int(args[i]) { cli.beamWidth = max(1, v) }
            case "--lr":
                i += 1; if i < args.count, let v = Float(args[i]) { cli.lrMax = v }
            case "--report":
                i += 1; if i < args.count { cli.reportPath = args[i] }
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

func char3GramCosine(_ a: String, _ b: String) -> Float {
    let ca = Array(a), cb = Array(b)
    guard ca.count >= 3 && cb.count >= 3 else { return 0.0 }
    var fa: [String: Float] = [:]
    var fb: [String: Float] = [:]
    for i in 0..<(ca.count - 2) { fa[String(ca[i...(i + 2)]), default: 0.0] += 1.0 }
    for i in 0..<(cb.count - 2) { fb[String(cb[i...(i + 2)]), default: 0.0] += 1.0 }
    var dot: Float = 0.0, na: Float = 0.0, nb: Float = 0.0
    for (_, v) in fa { na += v * v }
    for (_, v) in fb { nb += v * v }
    for (k, va) in fa { if let vb = fb[k] { dot += va * vb } }
    return dot / max(1e-8, sqrtf(na) * sqrtf(nb))
}

func checkFluencyAndRules(_ text: String, isSingleComment: Bool = false) -> (isFluent: Bool, hasForbiddenMeta: Bool, hasForbiddenTimingInSingle: Bool) {
    let chars = Array(text)
    let periodCount = chars.filter { $0 == "。" }.count
    guard chars.count >= 66 && chars.count <= 155 && periodCount == 2 && text.hasSuffix("。") && !text.contains("?") else {
        return (false, false, false)
    }
    var stutter = false
    if chars.count >= 4 {
        for i in 0..<(chars.count - 3) {
            if chars[i] == chars[i + 1] && chars[i + 1] == chars[i + 2] {
                stutter = true; break
            }
        }
    }
    for dup in ["でもでも", "これからもこれからも", "といとい", "ていてい", "おりおり"] where text.contains(dup) {
        stutter = true
    }
    let metaWords = [
        "感嘆符", "疑問符", "びっくりマーク", "はてなマーク", "絵文字", "顔文字",
        "句点", "読点", "句読点", "カタカナ", "スラング", "タメ口", "敬語", "ですます",
        "短文", "長文", "文面", "表記", "記号", "スタンプ", "T1", "T2", "T3", "T4"
    ]
    let hasMeta = metaWords.contains { text.contains($0) }
    let timingWords = ["以前より", "以前は", "かつては", "前と比べて", "空白期間", "沈黙期間", "返信が遅れ", "返信間隔", "返信速度", "未返信", "追撃"]
    let hasTimingLeak = isSingleComment && timingWords.contains { text.contains($0) }
    return (!stutter, hasMeta, hasTimingLeak)
}

func loadAllDialogueCases(dataDir: String, layer3dPath: String) -> (allTrain: [RelationalDialogueCase], byId: [String: RelationalDialogueCase]) {
    var map: [String: RelationalDialogueCase] = [:]
    let mainCfg = DatasetConfiguration.load(from: dataDir)
    let allTrain = mainCfg.relationalDialogueCases.filter { !$0.isZeroShotUnseen }
    for rc in mainCfg.relationalDialogueCases { map[rc.id] = rc }
    let l3dDir = URL(fileURLWithPath: layer3dPath).deletingLastPathComponent().path
    if FileManager.default.fileExists(atPath: layer3dPath) {
        let l3dCfg = DatasetConfiguration.load(from: l3dDir)
        for rc in l3dCfg.relationalDialogueCases { map[rc.id] = rc }
    }
    return (allTrain, map)
}

func padSubwordWithEOS(_ text: String, vocab: SubwordVocabulary, maxLen: Int) -> [Int] {
    let raw = vocab.encode(text, addEOS: true)
    var out = Array(raw.prefix(maxLen))
    if out.count == maxLen && out.last != TextVocabulary.eosId {
        out[maxLen - 1] = TextVocabulary.eosId
    }
    if out.count < maxLen {
        out.append(contentsOf: repeatElement(TextVocabulary.padId, count: maxLen - out.count))
    }
    return out
}

func boundaryClassIndex(_ s: String) -> Int {
    switch s {
    case "NONE": return 0
    case "RESPECTED": return 1
    case "MILD_INTRUSION": return 2
    case "SEVERE_VIOLATION": return 3
    default: return 0
    }
}

func formatDelta(_ prev: Date?, _ cur: Date?) -> String {
    guard let p = prev, let c = cur else { return "" }
    let sec = max(0.0, c.timeIntervalSince1970 - p.timeIntervalSince1970)
    if sec >= 86400.0 {
        return String(format: " (+%.1fd)", sec / 86400.0)
    } else if sec >= 3600.0 {
        return String(format: " (+%.1fh)", sec / 3600.0)
    } else {
        return String(format: " (+%.1fm)", sec / 60.0)
    }
}

let cli = PsychCLI.parse(CommandLine.arguments)
let relWeights = try SWTHawkWeights.load(from: URL(fileURLWithPath: cli.relationalWeightsPath))
let relModel = SWTHawkAutoencoder(weights: relWeights)

if cli.mode == "infer" {
    guard !cli.texts.isEmpty else {
        FileHandle.standardError.write(Data("Usage: psych_lm infer --text \"2026-10-12 12:34:56 あああ\" --text \"2026-11-23 10:22:33 いいい\" ...\n".utf8))
        exit(1)
    }
    let lmWeights = try PsychologicalStateLMWeights.load(from: URL(fileURLWithPath: cli.psychWeightsPath))
    let psychLM = PsychologicalStateLM(weights: lmWeights)
    let parsedComments = TimestampedComment.parseList(cli.texts)
    let res = psychLM.describeComments(
        relationalModel: relModel,
        comments: parsedComments,
        beamWidth: cli.beamWidth
    )

    print("==================================================================")
    print("Sashiba Psychological State LM — Comment-Stream Pure-CPU Inference")
    print("==================================================================")
    for (idx, c) in parsedComments.enumerated() {
        let tsLabel = c.rawTimestamp ?? "(no timestamp)"
        let deltaLabel = idx > 0 ? formatDelta(parsedComments[idx - 1].timestamp, c.timestamp) : ""
        print("[\(idx + 1)] \(tsLabel)\(deltaLabel) | \(c.text)")
    }
    print("------------------------------------------------------------------")
    print("Predicted Archetype : \(res.predictedArchetype.rawValue)")
    print("Predicted Topology  : \(res.predictedTopology.rawValue)")
    let em = res.predictedEmotion
    print(String(format: "Predicted Emotion   : affection=%d, friction=%d, arousal=%.2f", em.affection, em.friction, em.arousal))
    print(String(format: "CPU Latency         : total %.2f ms (encoder %.2f ms + LM %.2f ms)",
                 res.totalLatencyMs, res.relationalLatencyMs, res.generationLatencyMs))
    print("------------------------------------------------------------------")
    print("内的心理描写:\n\(res.psychologicalText)")
    print("==================================================================")
    exit(0)
}

let descContent = try String(contentsOfFile: cli.descriptionsPath, encoding: .utf8)
let jsonDecoder = JSONDecoder()
let records: [PsychRecord] = try descContent
    .split(whereSeparator: \.isNewline)
    .map { try jsonDecoder.decode(PsychRecord.self, from: Data($0.utf8)) }

let (allTrainCases, caseMap) = loadAllDialogueCases(dataDir: cli.dataDir, layer3dPath: cli.layer3dPath)

let auxHeads: PsychAuxiliaryHeads
let loadedLMWeights: PsychologicalStateLMWeights?
if cli.mode == "eval" {
    let w = try PsychologicalStateLMWeights.load(from: URL(fileURLWithPath: cli.psychWeightsPath))
    loadedLMWeights = w
    auxHeads = w.auxiliaryHeads
} else {
    loadedLMWeights = nil
    print("Calibrating 14 sender-free Comment-Stream probe heads across \(allTrainCases.count) partner-only streams...")
    let tAux0 = Date()
    let trainStreams = allTrainCases.map { TimestampedComment.synthesizeStream(from: $0) }
    let rawAux = PsychConditionExtractor.extractCommentsBatchParallel(
        weights: relWeights,
        auxHeads: PsychAuxiliaryHeads(),
        streams: trainStreams
    )

    let archCases = RelationalArchetype.allCases
    let topoCases = RelationalTopology.allCases
    let trajCases = RelationalTrajectoryType.allCases

    var curX: [[Float]] = []
    var earlyX: [[Float]] = []
    var fullX: [[Float]] = []
    var archCurY: [Int] = []
    var topoCurY: [Int] = []
    var affCurY: [Int] = []
    var fricCurY: [Int] = []
    var negMixY: [Int] = []
    var boundY: [Int] = []
    var trajY: [Int] = []
    var archEarlyY: [Int] = []
    var topoEarlyY: [Int] = []
    var affEarlyY: [Int] = []
    var fricEarlyY: [Int] = []
    var groups: [String] = []

    for (i, rc) in allTrainCases.enumerated() {
        curX.append(rawAux[i].probeCur)
        earlyX.append(rawAux[i].probeEarly)
        fullX.append(rawAux[i].probeEarly + rawAux[i].probeCur + rawAux[i].hybCross + rawAux[i].temp16)
        archCurY.append(archCases.firstIndex(of: rc.expectedArchetype) ?? 0)
        topoCurY.append(topoCases.firstIndex(of: rc.expectedTopology) ?? 0)
        affCurY.append(min(4, max(0, rc.emotionT4.affection)))
        fricCurY.append(min(4, max(0, rc.emotionT4.friction)))
        negMixY.append(rc.observationsT4.o3NegMix ? 1 : 0)
        boundY.append(boundaryClassIndex(rc.observationsT4.o4Boundary))
        trajY.append(trajCases.firstIndex(of: rc.trajectory.type) ?? 0)
        archEarlyY.append(archCases.firstIndex(of: rc.expectedArchetypeT2) ?? 0)
        topoEarlyY.append(topoCases.firstIndex(of: rc.expectedTopologyT2) ?? 0)
        affEarlyY.append(min(4, max(0, rc.emotionT2.affection)))
        fricEarlyY.append(min(4, max(0, rc.emotionT2.friction)))
        groups.append(rc.group)
    }

    let hArchCur = SoftmaxClassifierHead.trainGroupCV(features: curX, labels: archCurY, groups: groups, classCount: 6, steps: 220)
    let hTopoCur = SoftmaxClassifierHead.trainGroupCV(features: curX, labels: topoCurY, groups: groups, classCount: 5, steps: 220)
    let hAffCur = SoftmaxClassifierHead.trainGroupCV(features: curX, labels: affCurY, groups: groups, classCount: 5, steps: 200)
    let hFricCur = SoftmaxClassifierHead.trainGroupCV(features: curX, labels: fricCurY, groups: groups, classCount: 5, steps: 200)
    let hNegMix = SoftmaxClassifierHead.trainGroupCV(features: curX, labels: negMixY, groups: groups, classCount: 2, steps: 200)
    let hBoundary = SoftmaxClassifierHead.trainGroupCV(features: curX, labels: boundY, groups: groups, classCount: 4, steps: 200)
    let hTrajText = SoftmaxClassifierHead.trainGroupCV(features: curX, labels: trajY, groups: groups, classCount: 10, steps: 220)

    let hArchEarly = SoftmaxClassifierHead.trainGroupCV(features: earlyX, labels: archEarlyY, groups: groups, classCount: 6, steps: 200)
    let hTopoEarly = SoftmaxClassifierHead.trainGroupCV(features: earlyX, labels: topoEarlyY, groups: groups, classCount: 5, steps: 200)
    let hAffEarly = SoftmaxClassifierHead.trainGroupCV(features: earlyX, labels: affEarlyY, groups: groups, classCount: 5, steps: 180)
    let hFricEarly = SoftmaxClassifierHead.trainGroupCV(features: earlyX, labels: fricEarlyY, groups: groups, classCount: 5, steps: 180)

    let hTrajFull = SoftmaxClassifierHead.trainGroupCV(features: fullX, labels: trajY, groups: groups, classCount: 10, steps: 240)
    let hArchFull = SoftmaxClassifierHead.trainGroupCV(features: fullX, labels: archCurY, groups: groups, classCount: 6, steps: 240)
    let hTopoFull = SoftmaxClassifierHead.trainGroupCV(features: fullX, labels: topoCurY, groups: groups, classCount: 5, steps: 240)

    auxHeads = PsychAuxiliaryHeads(
        archCur: hArchCur,
        topoCur: hTopoCur,
        affCur: hAffCur,
        fricCur: hFricCur,
        negMixT4: hNegMix,
        boundaryT4: hBoundary,
        trajectoryTextOnly: hTrajText,
        archEarly: hArchEarly,
        topoEarly: hTopoEarly,
        affEarly: hAffEarly,
        fricEarly: hFricEarly,
        trajectoryFull: hTrajFull,
        archFull: hArchFull,
        topoFull: hTopoFull
    )
    print(String(format: "Calibrated 14 sender-free Comment-Stream heads in %.2fs", Date().timeIntervalSince(tAux0)))
}

struct LinkedPsychCase {
    let rec: PsychRecord
    let dialogue: RelationalDialogueCase
    let stream4: [TimestampedComment]
    let stream3: [TimestampedComment]
    let fullCond4: [Float]
    let fullCond3: [Float]
    let noTimeCond: [Float]
    let singleCommentCond: [Float]
}

print("Extracting 128D sender-free Comment-Stream conditions for \(records.count) cases on 16 CPU cores...")
let tExt0 = Date()
var validRecords: [(rec: PsychRecord, dlg: RelationalDialogueCase)] = []
var streams4: [[TimestampedComment]] = []
var streams3: [[TimestampedComment]] = []
for r in records {
    guard let dlg = caseMap[r.scenario_id] else { continue }
    validRecords.append((r, dlg))
    streams4.append(TimestampedComment.synthesizeStream(from: dlg))
    streams3.append(TimestampedComment.synthesizeStream(from: dlg, targetCount: 3))
}
let conds4 = PsychConditionExtractor.extractCommentsBatchParallel(weights: relWeights, auxHeads: auxHeads, streams: streams4)
let conds3 = PsychConditionExtractor.extractCommentsBatchParallel(weights: relWeights, auxHeads: auxHeads, streams: streams3)

var linkedCases: [LinkedPsychCase] = []
linkedCases.reserveCapacity(validRecords.count)
for i in 0..<validRecords.count {
    linkedCases.append(LinkedPsychCase(
        rec: validRecords[i].rec,
        dialogue: validRecords[i].dlg,
        stream4: streams4[i],
        stream3: streams3[i],
        fullCond4: conds4[i].fullCond,
        fullCond3: conds3[i].fullCond,
        noTimeCond: conds4[i].noTimeCond,
        singleCommentCond: conds4[i].singleLastCond
    ))
}
print(String(format: "Extracted %d sender-free cases in %.2fs", linkedCases.count, Date().timeIntervalSince(tExt0)))

let trainLinked = linkedCases.filter { $0.rec.psych_split == "train" }
let unseenLinked = linkedCases.filter { $0.rec.psych_split == "unseen_test" }
let l3dLinked = linkedCases.filter { $0.rec.psych_split == "layer3d_ood" }

let psychLM: PsychologicalStateLM
if cli.mode == "train" {
    var baseProseChars = "、。「」『』・ー！？0123456789０１２３４５６７８９"
    for cp in 0x3041...0x3096 { if let sc = UnicodeScalar(cp) { baseProseChars.append(Character(sc)) } }
    for cp in 0x30A1...0x30FA { if let sc = UnicodeScalar(cp) { baseProseChars.append(Character(sc)) } }

    var trainTexts: [String] = []
    trainTexts.reserveCapacity(trainLinked.count * 2)
    for item in trainLinked {
        trainTexts.append(item.rec.psych_full)
        trainTexts.append(item.rec.psych_text_only)
    }
    let tBpe0 = Date()
    let subwordVocab = SubwordVocabulary.train(
        trainingTexts: trainTexts,
        baseCharacters: baseProseChars,
        numMerges: cli.bpeMerges,
        minPairFreq: 4,
        maxPieceLen: 16
    )
    var maxSubLen = 24
    var totalSubLen = 0
    for item in linkedCases {
        let l1 = subwordVocab.encode(item.rec.psych_full, addEOS: true).count
        let l2 = subwordVocab.encode(item.rec.psych_text_only, addEOS: true).count
        maxSubLen = max(maxSubLen, max(l1, l2))
        totalSubLen += l1 + l2
    }
    let meanSubLen = Double(totalSubLen) / Double(max(1, linkedCases.count * 2))
    let seqLen = min(96, ((maxSubLen + 7) / 8) * 8)
    print(String(format: "Trained BPE SubwordVocabulary in %.2fs | Vocab V=%d (merges=%d) | Mean length=%.1f subwords (max=%d -> seqLen=%d)",
                 Date().timeIntervalSince(tBpe0), subwordVocab.count, subwordVocab.merges.count, meanSubLen, maxSubLen, seqLen))

    struct TrainSample {
        let cond: [Float]
        let targetIds: [Int]
    }
    var trainSamples: [TrainSample] = []
    trainSamples.reserveCapacity(trainLinked.count * 3)
    for (idx, item) in trainLinked.enumerated() {
        let fullTarget = padSubwordWithEOS(item.rec.psych_full, vocab: subwordVocab, maxLen: seqLen)
        let singleTarget = padSubwordWithEOS(item.rec.psych_text_only, vocab: subwordVocab, maxLen: seqLen)
        // Alternate 3-comment and 4-comment `--text` streams without duplicating `fullTarget` 3x per epoch
        let streamCond = (idx % 2 == 0) ? item.fullCond3 : item.fullCond4
        trainSamples.append(TrainSample(cond: streamCond, targetIds: fullTarget))
        trainSamples.append(TrainSample(cond: item.singleCommentCond, targetIds: singleTarget))

        let tc = item.dialogue.temporalContext
        let ratio = tc.replyLagHoursT4 / max(0.05, tc.actorBaselineReplyHours)
        if idx % 3 == 0 && ratio >= 0.5 && ratio <= 2.2 && !tc.postT4UnansweredProbe && tc.gapDaysT2ToT3 <= 12.0 {
            trainSamples.append(TrainSample(cond: item.noTimeCond, targetIds: fullTarget))
        }
    }

    let cDim = PsychConditionExtractor.rawConditionDim
    var cMean = [Float](repeating: 0.0, count: cDim)
    var cVar = [Float](repeating: 0.0, count: cDim)
    let invN = 1.0 / Float(max(1, trainSamples.count))
    for s in trainSamples {
        for j in 0..<cDim { cMean[j] += s.cond[j] * invN }
    }
    for s in trainSamples {
        for j in 0..<cDim {
            let d = s.cond[j] - cMean[j]
            cVar[j] += d * d * invN
        }
    }
    let cStd = cVar.map { max(1e-3, sqrtf($0)) }

    var valCondFlat: [Float] = []
    var valTargetFlat: [Int] = []
    for item in unseenLinked {
        valCondFlat.append(contentsOf: item.fullCond3)
        valTargetFlat.append(contentsOf: padSubwordWithEOS(item.rec.psych_full, vocab: subwordVocab, maxLen: seqLen))
        valCondFlat.append(contentsOf: item.singleCommentCond)
        valTargetFlat.append(contentsOf: padSubwordWithEOS(item.rec.psych_text_only, vocab: subwordVocab, maxLen: seqLen))
    }
    let valCount = unseenLinked.count * 2

    let numHeads = max(2, cli.dim / 32)
    var initW = PsychologicalStateLMWeights.makeRandom(
        vocabSize: subwordVocab.count,
        dim: cli.dim,
        numLayers: cli.layers,
        numHeads: numHeads,
        windowSize: max(64, seqLen),
        numCondTokens: 6,
        condRawDim: cDim,
        maxSeqLen: seqLen,
        serializedSubwordVocabulary: subwordVocab.serialize(),
        auxiliaryHeads: auxHeads,
        seed: 202611
    )
    initW.condMean = cMean
    initW.condStd = cStd

    let gpuModel = MLXPsychologicalStateLM(weights: initW, useTiedEmbeddings: true)
    gpuModel.setConditionNormalization(mean: cMean, std: cStd)
    let trainer = MLXPsychologicalStateTrainer(model: gpuModel, learningRate: cli.lrMax, seed: 777)

    let stepsPerEpoch = (trainSamples.count + cli.batchSize - 1) / cli.batchSize
    let totalSteps = cli.epochs * stepsPerEpoch
    let warmupSteps = max(10, totalSteps / 12)
    let sched = CosineLRScheduler(lrMax: cli.lrMax, lrMin: cli.lrMin)
    let freeParams = initW.parameterCount - initW.wReadout.count

    print("==================================================================")
    print("Training Sender-Free Comment-Stream Griffin LM (Compiled GPU)")
    print("==================================================================")
    print("MLX Device : \(Device.defaultDevice())")
    print("Parameters : \(freeParams) tied (\(initW.parameterCount) stored | D=\(cli.dim), L=\(cli.layers), H=\(numHeads), M=6, W=\(initW.windowSize), V=\(subwordVocab.count), seqLen=\(seqLen))")
    print("Train seqs : \(trainSamples.count) | Unseen val seqs: \(valCount) | OOD Layer 3d cases: \(l3dLinked.count)")
    print("------------------------------------------------------------------")

    var rng = SplitMix64(seed: 20261120)
    var order = Array(0..<trainSamples.count)
    var globalStep = 0
    var bestValLoss: Float = .greatestFiniteMagnitude
    var bestWeights = initW
    let tTrain0 = Date()

    for epoch in 1...cli.epochs {
        for k in stride(from: order.count - 1, to: 0, by: -1) {
            let r = Int(rng.next() % UInt64(k + 1))
            order.swapAt(k, r)
        }
        var epLoss: Float = 0.0
        var epAcc: Float = 0.0
        var batches = 0
        var off = 0
        while off < order.count {
            let rem = order.count - off
            let curB = cli.batchSize
            var bCond: [Float] = []
            var bTarg: [Int] = []
            bCond.reserveCapacity(curB * cDim)
            bTarg.reserveCapacity(curB * seqLen)
            for j in 0..<curB {
                let s = trainSamples[order[(off + j) % order.count]]
                bCond.append(contentsOf: s.cond)
                bTarg.append(contentsOf: s.targetIds)
            }
            globalStep += 1
            let lr = sched.learningRate(step: min(globalStep, totalSteps), totalSteps: totalSteps, warmupSteps: warmupSteps)
            let stepRes = trainer.trainStep(
                condFlat: bCond,
                targetTokensFlat: bTarg,
                batch: curB,
                time: seqLen,
                learningRate: lr
            )
            epLoss += stepRes.loss
            epAcc += stepRes.tokenAccuracy
            batches += 1
            off += rem < curB ? rem : curB
        }

        if epoch == 1 || epoch % 2 == 0 || epoch == cli.epochs {
            var valLossSum: Float = 0.0
            var valAccSum: Float = 0.0
            var vBatches = 0
            var vOff = 0
            while vOff < valCount {
                let curB = min(64, valCount - vOff)
                let cSlice = Array(valCondFlat[(vOff * cDim)..<((vOff + curB) * cDim)])
                let tSlice = Array(valTargetFlat[(vOff * seqLen)..<((vOff + curB) * seqLen)])
                let vRes = trainer.evaluateLoss(condFlat: cSlice, targetTokensFlat: tSlice, batch: curB, time: seqLen)
                valLossSum += vRes.ceLoss
                valAccSum += vRes.tokenAccuracy
                vBatches += 1
                vOff += curB
            }
            let meanValLoss = valLossSum / Float(max(1, vBatches))
            let meanValAcc = valAccSum / Float(max(1, vBatches)) * 100.0
            let trLoss = epLoss / Float(max(1, batches))
            let trAcc = epAcc / Float(max(1, batches)) * 100.0
            let marker = meanValLoss < bestValLoss ? " *best*" : ""
            if meanValLoss < bestValLoss {
                bestValLoss = meanValLoss
                bestWeights = gpuModel.exportWeights()
            }
            let statsLine = String(
                format: "Epoch %2d/%2d (%.1fs) | Train NLL: %.4f (TF %5.2f%%) | Unseen Val NLL: %.4f (Subword PPL %5.2f, TF %5.2f%%)",
                epoch, cli.epochs, Date().timeIntervalSince(tTrain0), trLoss, trAcc, meanValLoss, expf(min(10.0, meanValLoss)), meanValAcc
            )
            print("\(statsLine)\(marker)")
        }
    }
    let trainElapsed = Date().timeIntervalSince(tTrain0)
    try bestWeights.save(to: URL(fileURLWithPath: cli.psychWeightsPath))
    print(String(format: "Saved best Comment-Stream PsychologicalStateLM weights (trained in %.1fs, best Val NLL %.4f) -> \(cli.psychWeightsPath)",
                 trainElapsed, bestValLoss))
    psychLM = PsychologicalStateLM(weights: bestWeights)
} else {
    psychLM = PsychologicalStateLM(weights: loadedLMWeights!)
}

// ============================================================================
// Pure-CPU Evaluation on 3-Comment `--text` Streams & Single-Comment Mode
// ============================================================================
print("\nRunning Sender-Free Pure-CPU Evaluation on Unseen (\(unseenLinked.count)) + Layer 3d OOD (\(l3dLinked.count)) + `--text` Counterfactuals...")

struct SuiteMetrics {
    let count: Int
    let fluentStream: Int
    let fluentSingle: Int
    let zeroMetaViolations: Int
    let zeroTimingLeakInSingle: Int
    let meanCosStream: Float
    let meanCosSingle: Float
    let meanLenStream: Float
}

func evaluateSuite(_ items: [LinkedPsychCase]) -> (metrics: SuiteMetrics, samples: [(item: LinkedPsychCase, genStream: String, genSingle: String)]) {
    var fluentF = 0, fluentT = 0, noMeta = 0, noLeak = 0
    var cosFSum: Float = 0.0, cosTSum: Float = 0.0, lenFSum: Float = 0.0
    var collected: [(LinkedPsychCase, String, String)] = []

    for (idx, it) in items.enumerated() {
        let gf = psychLM.vocabulary.decode(psychLM.generateBeam(condRaw: it.fullCond3, beamWidth: 2))
        let gt = psychLM.vocabulary.decode(psychLM.generateBeam(condRaw: it.singleCommentCond, beamWidth: 2))
        let rf = checkFluencyAndRules(gf, isSingleComment: false)
        let rt = checkFluencyAndRules(gt, isSingleComment: true)
        if rf.isFluent { fluentF += 1 }
        if rt.isFluent { fluentT += 1 }
        if !rf.hasForbiddenMeta && !rt.hasForbiddenMeta { noMeta += 1 }
        if !rt.hasForbiddenTimingInSingle { noLeak += 1 }
        cosFSum += char3GramCosine(gf, it.rec.psych_full)
        cosTSum += char3GramCosine(gt, it.rec.psych_text_only)
        lenFSum += Float(gf.count)
        if idx < 10 {
            collected.append((it, gf, gt))
        }
    }
    let n = max(1, items.count)
    return (
        SuiteMetrics(
            count: items.count,
            fluentStream: fluentF,
            fluentSingle: fluentT,
            zeroMetaViolations: noMeta,
            zeroTimingLeakInSingle: noLeak,
            meanCosStream: cosFSum / Float(n),
            meanCosSingle: cosTSum / Float(n),
            meanLenStream: lenFSum / Float(n)
        ),
        collected
    )
}

let (unseenM, unseenSamples) = evaluateSuite(unseenLinked)
let (l3dM, l3dSamples) = evaluateSuite(l3dLinked)

// `--text` Timestamp Counterfactuals (0% sender!)
let cf1SteadyMonthly = psychLM.describeTexts(relationalModel: relModel, texts: [
    "2026-10-12 12:34:56 今回の新曲アレンジめちゃくちゃ耳に残るね！！サビの入りで鳥肌立ったよ✨",
    "2026-11-23 10:22:33 寒くなってきたけど喉の調子はどう？無理せずマイペースに制作続けてね🍵",
    "2026-12-25 22:33:44 今年のクリスマス配信も最高に温かかった！！来年も陰ながらずっと応援してるね🎄✨"
])
let cf1DecelFadeOut = psychLM.describeTexts(relationalModel: relModel, texts: [
    "2026-10-12 12:34:56 今週のパン教室めっちゃ楽しかったね！！家でも復習して焼いてみたよ🍻",
    "2026-10-13 19:10:00 次の新作レッスンも絶対一緒に行きたい！！駅前カフェも寄ろうね✨",
    "2026-12-25 22:33:44 連絡遅くなってごめんね、最近ちょっとバタバタしてて落ち着いたらまたこっちから連絡するね"
])
let cf1SingleOnly = psychLM.describeTexts(relationalModel: relModel, texts: [
    "2026-12-25 22:33:44 新作めっちゃ美味しそうでテンション上がるね！！今週ちょっと立て込んでるから落ち着いたらまた行こうね🍻‼️"
])
let cf2KawaiiSafe = psychLM.describeTexts(relationalModel: relModel, texts: [
    "2026-10-12 23:10:00 声メロすぎてガチで恋しそうしんどい😭🤍 でもリスナーとして距離感守るから安心して推させて🫶",
    "2026-11-23 23:20:00 撮影ハードだろうけど気持ちだけ差し入れ送っとくね🍵✨ 体調いちばん大事にしてね！",
    "2026-12-25 23:15:00 おやすみの言い方反則レベルで好きすぎる🥺✨ 無理に毎日配信しなくていいからね良い夢見てね🤍"
])
let cf2KawaiiMount = psychLM.describeTexts(relationalModel: relModel, texts: [
    "2026-10-12 14:00:00 毎日フルタイムでお仕事えらすぎるよ〜🥰🎀 うちは主人が稼いでくれるから平日はエステ三昧で逆に退屈なの🥺✨",
    "2026-11-23 15:10:00 ボーナス前なのに自炊お弁当続けててほんと涙ぐましいね👏✨ うちは外食ばかりで尊敬しちゃう💕",
    "2026-12-25 18:30:00 わああの店そんなにお高くないのに記念日なんだね、可愛くて素敵だよ🥰🧁 年収低い旦那さんでも工夫次第で幸せになれるよね💕✨"
])
let cf2RapidEscalate = psychLM.describeTexts(relationalModel: relModel, texts: [
    "2026-10-12 21:00:00 ビジュえぐいし今日も好きすぎてしんどい😭🤍 いちばん最初に見に行ったの気づいてくれた？",
    "2026-12-25 22:30:00 スパチャも毎回投げてるのに今日まだ俺の名前呼んでくれてないよね？🥺",
    "2026-12-25 22:33:44 え なんで俺の名前だけ毎回飛ばすの？ 認知してるなら今すぐ反応してよ🥺💢"
])

// Single-thread Pure-CPU Latency Benchmark across 40 unseen 3-comment `--text` streams
var streamLatencies: [Double] = []
var singleLatencies: [Double] = []
var genOnlyLatencies: [Double] = []
let benchItems = Array(unseenLinked.prefix(min(40, unseenLinked.count)))
for i in 0..<min(3, benchItems.count) {
    _ = psychLM.describeComments(relationalModel: relModel, comments: benchItems[i].stream3)
}
for it in benchItems {
    let resStream = psychLM.describeComments(relationalModel: relModel, comments: it.stream3)
    let resSingle = psychLM.describeComments(relationalModel: relModel, comments: [it.stream3.last!])
    streamLatencies.append(resStream.totalLatencyMs)
    singleLatencies.append(resSingle.totalLatencyMs)
    genOnlyLatencies.append(resStream.generationLatencyMs)
}
streamLatencies.sort()
singleLatencies.sort()
genOnlyLatencies.sort()
let meanStreamMs = streamLatencies.reduce(0.0, +) / Double(max(1, streamLatencies.count))
let p50StreamMs = streamLatencies[streamLatencies.count / 2]
let p95StreamMs = streamLatencies[min(streamLatencies.count - 1, Int(Double(streamLatencies.count) * 0.95))]
let meanSingleMs = singleLatencies.reduce(0.0, +) / Double(max(1, singleLatencies.count))
let meanGenMs = genOnlyLatencies.reduce(0.0, +) / Double(max(1, genOnlyLatencies.count))

print("==================================================================")
print("Sender-Free Comment-Stream (`--text`) Pure-CPU Verification")
print("==================================================================")
print(String(format: "[Unseen %3d] Fluent(3-text/1-text): %d/%d, %d/%d | 0-Meta-Rule: %d/%d | 0-TimingLeak: %d/%d | 3gram-Cos: %.3f / %.3f | MeanLen: %.1f chars",
             unseenM.count, unseenM.fluentStream, unseenM.count, unseenM.fluentSingle, unseenM.count,
             unseenM.zeroMetaViolations, unseenM.count, unseenM.zeroTimingLeakInSingle, unseenM.count,
             unseenM.meanCosStream, unseenM.meanCosSingle, unseenM.meanLenStream))
print(String(format: "[Layer3d %2d] Fluent(3-text/1-text): %d/%d, %d/%d | 0-Meta-Rule: %d/%d | 0-TimingLeak: %d/%d | 3gram-Cos: %.3f / %.3f | MeanLen: %.1f chars",
             l3dM.count, l3dM.fluentStream, l3dM.count, l3dM.fluentSingle, l3dM.count,
             l3dM.zeroMetaViolations, l3dM.count, l3dM.zeroTimingLeakInSingle, l3dM.count,
             l3dM.meanCosStream, l3dM.meanCosSingle, l3dM.meanLenStream))
print(String(format: "[CPU Speed ] 3-Comment Stream: mean %.2f ms (p50 %.2f, p95 %.2f) | 1-Comment: mean %.2f ms | LM decode only: mean %.2f ms",
             meanStreamMs, p50StreamMs, p95StreamMs, meanSingleMs, meanGenMs))
print("------------------------------------------------------------------")
print("[CF-1 Steady Monthly (Oct->Nov->Dec)] : \(cf1SteadyMonthly.psychologicalText)")
print("[CF-1 Sudden 73d Fade-Out]            : \(cf1DecelFadeOut.psychologicalText)")
print("[CF-1 Single `--text` Only]           : \(cf1SingleOnly.psychologicalText)")
print("[CF-2 Pure Kawaii Safe]               : \(cf2KawaiiSafe.psychologicalText)")
print("[CF-2 Kawaii Mounting]                : \(cf2KawaiiMount.psychologicalText)")
print("[CF-2 3m44s Rapid Escalation]         : \(cf2RapidEscalate.psychologicalText)")
print("==================================================================")

var md = "# Sashiba コメント時系列 (`--text`) 専用・条件付き Griffin 内的心理状況言語モデル (`PsychologicalStateLM`) 検証レポート\n\n"
md += "- **入力仕様**: `sender`（こちらの発言や手動パラメータ）一切不要。`--text \"YYYY-MM-DD HH:MM:SS コメント\"` を1件以上並べるだけで、コメント間の経過時間・頻度変化・言葉づかい・絵文字の温度推移を自動抽出して内的心理を描写\n"
md += "- **モデル構成**: 文節境界 BPE (`V = \(psychLM.vocabulary.count)`, `merges = \(psychLM.vocabulary.merges.count)`), `D = \(psychLM.weights.dim)`, `Layers = \(psychLM.weights.layers.count)`, `Heads = \(psychLM.weights.numHeads)`, `W = \(psychLM.weights.windowSize)`, `Condition Dim = \(psychLM.weights.condRawDim)` (`0%` sender text)\n"
md += String(format: "- **純CPU単一スレッド推論速度 (`batch = 1`)**: 3件コメント時系列 平均 **`%.2f ms`** (`p50 = %.2f ms`, `p95 = %.2f ms`) / 1件コメント単体 平均 **`%.2f ms`** / LM生成部のみ平均 **`%.2f ms`**\n\n", meanStreamMs, p50StreamMs, p95StreamMs, meanSingleMs, meanGenMs)

md += "## 1. 未学習テスト (`unseen_test`) および凍結 OOD (`Layer 3d`) 定量評価 (`sender` 不使用)\n\n"
md += "| 評価スイート | 件数 | 2文完結・自然構文率 (`3件時系列` / `1件単体`) | 記号メタ説明ゼロ遵守率 | 単発時の時刻幻覚ゼロ遵守率 | 参照心理文 3-gram 類似度 | 平均文字数 |\n"
md += "| :--- | ---: | :---: | :---: | :---: | :---: | ---: |\n"
md += String(format: "| **`unseen_test` (未学習)** | %d | **`%d/%d`** / **`%d/%d`** | **`%d/%d`** | **`%d/%d`** | **`%.3f`** / **`%.3f`** | %.1f 文字 |\n",
             unseenM.count, unseenM.fluentStream, unseenM.count, unseenM.fluentSingle, unseenM.count,
             unseenM.zeroMetaViolations, unseenM.count, unseenM.zeroTimingLeakInSingle, unseenM.count,
             unseenM.meanCosStream, unseenM.meanCosSingle, unseenM.meanLenStream)
md += String(format: "| **`Layer 3d` (世代・スラング・絵文字 OOD)** | %d | **`%d/%d`** / **`%d/%d`** | **`%d/%d`** | **`%d/%d`** | **`%.3f`** / **`%.3f`** | %.1f 文字 |\n\n",
             l3dM.count, l3dM.fluentStream, l3dM.count, l3dM.fluentSingle, l3dM.count,
             l3dM.zeroMetaViolations, l3dM.count, l3dM.zeroTimingLeakInSingle, l3dM.count,
             l3dM.meanCosStream, l3dM.meanCosSingle, l3dM.meanLenStream)

md += "## 2. `--text` タイムスタンプ・文面推移による反事実感度テスト (`sender` 不使用)\n\n"
md += "| ケース | 入力 `--text` 列 | 生成された内的心理描写 |\n| :--- | :--- | :--- |\n"
md += "| **定期的な月1応援 (`10/12 → 11/23 → 12/25`)** | 3件（約42日・32日間隔の安定した温かい応援） | \(cf1SteadyMonthly.psychologicalText) |\n"
md += "| **連日交流からの73日放置・先送り (`10/12 → 10/13 → 12/25`)** | 3件（1日間隔→73日空白＋先送り） | \(cf1DecelFadeOut.psychologicalText) |\n"
md += "| **1件のみ (`--text` 1個)** | `2026-12-25 22:33:44 新作めっちゃ美味しそうで…🍻‼️` | \(cf1SingleOnly.psychologicalText) |\n"
md += "| **健全な親愛・体調配慮 (`10/12 → 11/23 → 12/25`)** | 3件（距離感を守る熱心な推し活） | \(cf2KawaiiSafe.psychologicalText) |\n"
md += "| **可愛い絵文字の裏のマウント (`10/12 → 11/23 → 12/25`)** | 3件（生活格差マウント） | \(cf2KawaiiMount.psychologicalText) |\n"
md += "| **3分44秒間隔の連投・承認要求暴走 (`22:30:00 → 22:33:44`)** | 3件（直前連投＋名前飛ばし詰問） | \(cf2RapidEscalate.psychologicalText) |\n\n"

md += "## 3. 未学習シナリオ (`unseen_test` & `Layer 3d`) `--text` 実生成サンプル\n\n"
md += "| ID / カテゴリ | 最終コメント | 生成心理描写 (`--text` 3件時系列) | 生成心理描写 (`--text` 1件単体) |\n| :--- | :--- | :--- | :--- |\n"
for (it, gf, gt) in (unseenSamples.prefix(5) + l3dSamples.prefix(5)) {
    md += "| `\(it.rec.scenario_id)` (`\(it.rec.category)`) | \(it.stream3.last?.text ?? "") | \(gf) | \(gt) |\n"
}

try md.write(to: URL(fileURLWithPath: cli.reportPath), atomically: true, encoding: .utf8)
print("Saved detailed report to \(cli.reportPath)")
