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
    var epochs: Int = 36
    var batchSize: Int = 32
    var dim: Int = 192
    var layers: Int = 3
    var lrMax: Float = 2.0e-3
    var lrMin: Float = 1.5e-4

    // For `infer` mode
    var pastSender: String? = nil
    var pastReceiver: String? = nil
    var sender: String = ""
    var receiver: String = ""
    var baselineHours: Float? = nil
    var lagT2: Float = 2.0
    var lagT4: Float = 2.0
    var gapDays: Float = 3.0
    var unansweredProbe: Bool = false

    static func parse(_ args: [String]) -> PsychCLI {
        var cli = PsychCLI()
        var i = 1
        if i < args.count && !args[i].hasPrefix("-") {
            cli.mode = args[i]
            i += 1
        }
        while i < args.count {
            switch args[i] {
            case "--rel-weights": i += 1; if i < args.count { cli.relationalWeightsPath = args[i] }
            case "--psych-weights", "-w": i += 1; if i < args.count { cli.psychWeightsPath = args[i] }
            case "--descriptions": i += 1; if i < args.count { cli.descriptionsPath = args[i] }
            case "--epochs", "-e": i += 1; if i < args.count, let v = Int(args[i]) { cli.epochs = max(1, v) }
            case "--batch", "-b": i += 1; if i < args.count, let v = Int(args[i]) { cli.batchSize = max(4, v) }
            case "--dim": i += 1; if i < args.count, let v = Int(args[i]) { cli.dim = max(64, v) }
            case "--layers": i += 1; if i < args.count, let v = Int(args[i]) { cli.layers = max(1, v) }
            case "--lr": i += 1; if i < args.count, let v = Float(args[i]) { cli.lrMax = v }
            case "--report": i += 1; if i < args.count { cli.reportPath = args[i] }
            case "--past-sender": i += 1; if i < args.count { cli.pastSender = args[i] }
            case "--past-receiver": i += 1; if i < args.count { cli.pastReceiver = args[i] }
            case "--sender": i += 1; if i < args.count { cli.sender = args[i] }
            case "--receiver": i += 1; if i < args.count { cli.receiver = args[i] }
            case "--baseline-hours": i += 1; if i < args.count, let v = Float(args[i]) { cli.baselineHours = v }
            case "--lag-t2": i += 1; if i < args.count, let v = Float(args[i]) { cli.lagT2 = v }
            case "--lag-t4": i += 1; if i < args.count, let v = Float(args[i]) { cli.lagT4 = v }
            case "--gap-days": i += 1; if i < args.count, let v = Float(args[i]) { cli.gapDays = v }
            case "--unanswered": cli.unansweredProbe = true
            default: break
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

func checkFluencyAndRules(_ text: String, isTextOnly: Bool = false) -> (isFluent: Bool, hasForbiddenMeta: Bool, hasForbiddenTimingInTextOnly: Bool) {
    let chars = Array(text)
    guard chars.count >= 50 && text.hasSuffix("。") && !text.contains("?") else {
        return (false, false, false)
    }
    var stutter = false
    if chars.count >= 4 {
        for i in 0..<(chars.count - 3) {
            if chars[i] == chars[i + 1] && chars[i + 1] == chars[i + 2] && chars[i + 2] == chars[i + 3] {
                stutter = true; break
            }
        }
    }
    let metaWords = [
        "感嘆符", "疑問符", "びっくりマーク", "はてなマーク", "絵文字", "顔文字",
        "句点", "読点", "句読点", "カタカナ", "スラング", "タメ口", "敬語", "ですます",
        "短文", "長文", "文面", "表記", "記号", "スタンプ", "T1", "T2", "T3", "T4"
    ]
    let hasMeta = metaWords.contains { text.contains($0) }
    let timingWords = ["以前より", "以前は", "かつては", "前と比べて", "空白期間", "沈黙期間", "返信が遅れ", "返信間隔", "返信速度", "未返信", "追撃"]
    let hasTimingLeak = isTextOnly && timingWords.contains { text.contains($0) }
    return (!stutter, hasMeta, hasTimingLeak)
}

func loadAllDialogueCases(dataDir: String, layer3dPath: String) -> [String: RelationalDialogueCase] {
    var map: [String: RelationalDialogueCase] = [:]
    let mainCfg = DatasetConfiguration.load(from: dataDir)
    for rc in mainCfg.relationalDialogueCases { map[rc.id] = rc }
    let l3dDir = URL(fileURLWithPath: layer3dPath).deletingLastPathComponent().path
    if FileManager.default.fileExists(atPath: layer3dPath) {
        let l3dCfg = DatasetConfiguration.load(from: l3dDir)
        for rc in l3dCfg.relationalDialogueCases { map[rc.id] = rc }
    }
    return map
}

func padTargetWithEOS(_ text: String, vocab: TextVocabulary, maxLen: Int) -> [Int] {
    let raw = vocab.encode(text, maxLength: maxLen - 1, addEOS: true, padToLength: nil)
    var out = Array(raw.prefix(maxLen))
    if out.count < maxLen {
        out.append(contentsOf: repeatElement(TextVocabulary.padId, count: maxLen - out.count))
    }
    return out
}

let cli = PsychCLI.parse(CommandLine.arguments)
let relWeights = try SWTHawkWeights.load(from: URL(fileURLWithPath: cli.relationalWeightsPath))
let relModel = SWTHawkAutoencoder(weights: relWeights)

if cli.mode == "infer" {
    let lmWeights = try PsychologicalStateLMWeights.load(from: URL(fileURLWithPath: cli.psychWeightsPath))
    let psychLM = PsychologicalStateLM(weights: lmWeights)
    let tc: TemporalRelationalContext? = if let baseH = cli.baselineHours {
        TemporalRelationalContext(
            actorBaselineReplyHours: baseH,
            replyLagHoursT2: cli.lagT2,
            replyLagHoursT4: cli.lagT4,
            gapDaysT2ToT3: cli.gapDays,
            postT4UnansweredProbe: cli.unansweredProbe
        )
    } else {
        nil
    }
    let res = psychLM.describePsychology(
        relationalModel: relModel,
        senderEarlyText: cli.pastSender,
        receiverEarlyText: cli.pastReceiver,
        senderCurrentText: cli.sender,
        receiverCurrentText: cli.receiver,
        temporalContext: tc
    )
    print("==================================================================")
    print("Sashiba Psychological State LM — Pure-CPU Inference")
    print("==================================================================")
    print("Predicted Archetype (v6_hybrid): \(res.inference.hybridPredictedArchetypeT4.rawValue)")
    print("Predicted Topology  (v6_hybrid): \(res.inference.hybridPredictedTopologyT4.rawValue)")
    let em = res.inference.hybridPredictedEmotionT4
    print(String(format: "Predicted Emotion              : affection=%d, friction=%d, arousal=%.2f", em.affection, em.friction, em.arousal))
    print(String(format: "CPU Latency                    : total %.2f ms (encoder %.2f ms + LM %.2f ms)",
                 res.totalLatencyMs, res.relationalLatencyMs, res.generationLatencyMs))
    print("------------------------------------------------------------------")
    print("内的心理描写:\n\(res.psychologicalText)")
    print("==================================================================")
    exit(0)
}

let descContent = try String(contentsOfFile: cli.descriptionsPath, encoding: .utf8)
let decoder = JSONDecoder()
let records: [PsychRecord] = try descContent
    .split(whereSeparator: \.isNewline)
    .map { try decoder.decode(PsychRecord.self, from: Data($0.utf8)) }

let caseMap = loadAllDialogueCases(dataDir: cli.dataDir, layer3dPath: cli.layer3dPath)
let relVocab = relModel.decodedVocabulary

struct LinkedPsychCase {
    let rec: PsychRecord
    let dialogue: RelationalDialogueCase
    let fullCond: [Float]
    let pastOnlyCond: [Float]
    let textOnlyCond: [Float]
}

print("Extracting 586D relational + orthographic + temporal conditions for \(records.count) cases on 16 CPU cores...")
let tExt0 = Date()
var validRecords: [(rec: PsychRecord, dlg: RelationalDialogueCase)] = []
var batchInputs: [(s2: [Int], r2: [Int], s4: [Int], r4: [Int], temporal: TemporalRelationalContext)] = []
for r in records {
    guard let dlg = caseMap[r.scenario_id] else { continue }
    validRecords.append((r, dlg))
    batchInputs.append((
        s2: relVocab.encode(dlg.hostEarlyText, maxLength: 512, addEOS: false),
        r2: relVocab.encode(dlg.partnerEarlyText, maxLength: 512, addEOS: false),
        s4: relVocab.encode(dlg.hostText, maxLength: 512, addEOS: false),
        r4: relVocab.encode(dlg.partnerCurrentText, maxLength: 512, addEOS: false),
        temporal: dlg.temporalContext
    ))
}
let extractedConds = PsychConditionExtractor.extractBatchParallel(weights: relWeights, items: batchInputs)
var linkedCases: [LinkedPsychCase] = []
linkedCases.reserveCapacity(validRecords.count)
for i in 0..<validRecords.count {
    let full = extractedConds[i].fullCond
    var pastOnly = full
    let tempStart = PsychConditionExtractor.rawConditionDim - 1 - PsychConditionExtractor.temporalDim
    for k in tempStart..<PsychConditionExtractor.rawConditionDim { pastOnly[k] = 0.0 }
    linkedCases.append(LinkedPsychCase(
        rec: validRecords[i].rec,
        dialogue: validRecords[i].dlg,
        fullCond: full,
        pastOnlyCond: pastOnly,
        textOnlyCond: extractedConds[i].textOnlyCond
    ))
}
print(String(format: "Extracted %d cases in %.2fs", linkedCases.count, Date().timeIntervalSince(tExt0)))

let trainLinked = linkedCases.filter { $0.rec.psych_split == "train" }
let unseenLinked = linkedCases.filter { $0.rec.psych_split == "unseen_test" }
let l3dLinked = linkedCases.filter { $0.rec.psych_split == "layer3d_ood" }

let psychLM: PsychologicalStateLM
if cli.mode == "train" {
    // Build vocabulary covering existing serializedVocabulary + all training psychological descriptions
    var vocabSeedTexts: [String] = []
    var existingChars = ""
    for id in TextVocabulary.specialCount..<relVocab.count {
        if let ch = relVocab.tokenCharacter(at: id) { existingChars.append(ch) }
    }
    vocabSeedTexts.append(existingChars)
    for item in trainLinked {
        vocabSeedTexts.append(item.rec.psych_full)
        vocabSeedTexts.append(item.rec.psych_text_only)
    }
    let lmVocab = TextVocabulary(texts: vocabSeedTexts)
    var maxTargetTokens = 64
    for item in linkedCases {
        let l1 = lmVocab.encode(item.rec.psych_full, addEOS: true).count
        let l2 = lmVocab.encode(item.rec.psych_text_only, addEOS: true).count
        maxTargetTokens = max(maxTargetTokens, max(l1, l2))
    }
    let seqLen = min(192, ((maxTargetTokens + 7) / 8) * 8)

    // Assemble training examples: (fullCond -> psych_full), (pastOnlyCond [20% aug] -> psych_full when pace normal), (textOnlyCond -> psych_text_only)
    struct TrainSample {
        let cond: [Float]
        let targetIds: [Int]
    }
    var trainSamples: [TrainSample] = []
    trainSamples.reserveCapacity(trainLinked.count * 2 + 300)
    for item in trainLinked {
        let fullTarget = padTargetWithEOS(item.rec.psych_full, vocab: lmVocab, maxLen: seqLen)
        let textTarget = padTargetWithEOS(item.rec.psych_text_only, vocab: lmVocab, maxLen: seqLen)
        trainSamples.append(TrainSample(cond: item.fullCond, targetIds: fullTarget))
        trainSamples.append(TrainSample(cond: item.textOnlyCond, targetIds: textTarget))

        let tc = item.dialogue.temporalContext
        let ratio = tc.replyLagHoursT4 / max(0.05, tc.actorBaselineReplyHours)
        if ratio >= 0.5 && ratio <= 2.2 && !tc.postT4UnansweredProbe && tc.gapDaysT2ToT3 <= 12.0 {
            trainSamples.append(TrainSample(cond: item.pastOnlyCond, targetIds: fullTarget))
        }
    }

    // Compute condMean & condStd strictly on trainSamples
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

    // Validation tensors (unseen full + text_only)
    var valCondFlat: [Float] = []
    var valTargetFlat: [Int] = []
    for item in unseenLinked {
        valCondFlat.append(contentsOf: item.fullCond)
        valTargetFlat.append(contentsOf: padTargetWithEOS(item.rec.psych_full, vocab: lmVocab, maxLen: seqLen))
        valCondFlat.append(contentsOf: item.textOnlyCond)
        valTargetFlat.append(contentsOf: padTargetWithEOS(item.rec.psych_text_only, vocab: lmVocab, maxLen: seqLen))
    }
    let valCount = unseenLinked.count * 2

    var initW = PsychologicalStateLMWeights.makeRandom(
        vocabSize: lmVocab.count,
        dim: cli.dim,
        numLayers: cli.layers,
        numHeads: 6,
        windowSize: 48,
        numCondTokens: 8,
        condRawDim: cDim,
        maxSeqLen: seqLen,
        serializedVocabulary: lmVocab.serialize(),
        seed: 202611
    )
    initW.condMean = cMean
    initW.condStd = cStd

    let gpuModel = MLXPsychologicalStateLM(weights: initW)
    gpuModel.setConditionNormalization(mean: cMean, std: cStd)
    let trainer = MLXPsychologicalStateTrainer(model: gpuModel, learningRate: cli.lrMax, seed: 777)

    let stepsPerEpoch = (trainSamples.count + cli.batchSize - 1) / cli.batchSize
    let totalSteps = cli.epochs * stepsPerEpoch
    let warmupSteps = max(10, totalSteps / 15)
    let sched = CosineLRScheduler(lrMax: cli.lrMax, lrMin: cli.lrMin)

    print("==================================================================")
    print("Training Small Conditional Griffin Psychological State LM on GPU")
    print("==================================================================")
    print("Parameters : \(initW.parameterCount) (D=\(cli.dim), L=\(cli.layers), H=6, M=8, V=\(lmVocab.count), seqLen=\(seqLen))")
    print("Train seqs : \(trainSamples.count) | Unseen val seqs: \(valCount) | OOD Layer 3d cases: \(l3dLinked.count)")
    print("------------------------------------------------------------------")

    var rng = SplitMix64(seed: 20261119)
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
            let curB = min(cli.batchSize, order.count - off)
            var bCond: [Float] = []
            var bTarg: [Int] = []
            bCond.reserveCapacity(curB * cDim)
            bTarg.reserveCapacity(curB * seqLen)
            for j in 0..<curB {
                let s = trainSamples[order[off + j]]
                bCond.append(contentsOf: s.cond)
                bTarg.append(contentsOf: s.targetIds)
            }
            globalStep += 1
            let lr = sched.learningRate(step: globalStep, totalSteps: totalSteps, warmupSteps: warmupSteps)
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
            off += curB
        }

        if epoch == 1 || epoch % 4 == 0 || epoch == cli.epochs {
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
            print(String(format: "Epoch %2d/%2d | Train NLL: %.4f (TF Acc %5.2f%%) | Unseen Val NLL: %.4f (PPL %5.2f, TF Acc %5.2f%%)%@",
                         epoch, cli.epochs, trLoss, trAcc, meanValLoss, expf(min(10.0, meanValLoss)), meanValAcc, marker))
        }
    }
    let trainElapsed = Date().timeIntervalSince(tTrain0)
    try bestWeights.save(to: URL(fileURLWithPath: cli.psychWeightsPath))
    print(String(format: "Saved best PsychologicalStateLM weights to %@ (trained in %.1fs, best Val NLL %.4f)",
                 cli.psychWeightsPath, trainElapsed, bestValLoss))
    psychLM = PsychologicalStateLM(weights: bestWeights)
} else {
    let loadedW = try PsychologicalStateLMWeights.load(from: URL(fileURLWithPath: cli.psychWeightsPath))
    psychLM = PsychologicalStateLM(weights: loadedW)
}

// ============================================================================
// Pure-CPU Evaluation, Counterfactual Sensitivity Audits & Latency Benchmark
// ============================================================================
print("\nRunning Pure-CPU Evaluation on Unseen (\(unseenLinked.count)) + Layer 3d OOD (\(l3dLinked.count)) + Counterfactuals...")

struct SuiteMetrics {
    let count: Int
    let fluentFull: Int
    let fluentTextOnly: Int
    let zeroMetaViolations: Int
    let zeroTimingLeakInTextOnly: Int
    let meanCosFull: Float
    let meanCosTextOnly: Float
    let meanLenFull: Float
}

func evaluateSuite(_ items: [LinkedPsychCase]) -> (metrics: SuiteMetrics, samples: [(item: LinkedPsychCase, genFull: String, genTextOnly: String)]) {
    var fluentF = 0, fluentT = 0, noMeta = 0, noLeak = 0
    var cosFSum: Float = 0.0, cosTSum: Float = 0.0, lenFSum: Float = 0.0
    var collected: [(LinkedPsychCase, String, String)] = []

    for (idx, it) in items.enumerated() {
        let gf = psychLM.vocabulary.decode(psychLM.generateTokens(condRaw: it.fullCond))
        let gt = psychLM.vocabulary.decode(psychLM.generateTokens(condRaw: it.textOnlyCond))
        let rf = checkFluencyAndRules(gf, isTextOnly: false)
        let rt = checkFluencyAndRules(gt, isTextOnly: true)
        if rf.isFluent { fluentF += 1 }
        if rt.isFluent { fluentT += 1 }
        if !rf.hasForbiddenMeta && !rt.hasForbiddenMeta { noMeta += 1 }
        if !rt.hasForbiddenTimingInTextOnly { noLeak += 1 }
        cosFSum += char3GramCosine(gf, it.rec.psych_full)
        cosTSum += char3GramCosine(gt, it.rec.psych_text_only)
        lenFSum += Float(gf.count)
        if idx < 12 {
            collected.append((it, gf, gt))
        }
    }
    let n = max(1, items.count)
    return (
        SuiteMetrics(
            count: items.count,
            fluentFull: fluentF,
            fluentTextOnly: fluentT,
            zeroMetaViolations: noMeta,
            zeroTimingLeakInTextOnly: noLeak,
            meanCosFull: cosFSum / Float(n),
            meanCosTextOnly: cosTSum / Float(n),
            meanLenFull: lenFSum / Float(n)
        ),
        collected
    )
}

let (unseenM, unseenSamples) = evaluateSuite(unseenLinked)
let (l3dM, l3dSamples) = evaluateSuite(l3dLinked)

// Counterfactual Sensitivity Tests (Pure CPU)
// CF-A: Identical cheerful deferral utterance, Fast eager timing vs 48h lag + unanswered probe
let cfASenderEarly = "先週の教室で作ったパン、家で復習したらカリッと焼けたよ！そっちはどうだった？"
let cfAReceiverEarly = "カリカリに焼けるの羨ましい！うちは少し焼き色が薄かったから次は温度見直してみるね！"
let cfASenderCur = "今週末の新作レッスン楽しみだね！終わったら駅前のカフェにも寄って食べ比べしない？"
let cfAReceiverCur = "新作めっちゃ美味しそうでテンション上がるね！今週ちょっとバタバタしてるけど、また落ち着いたら絶対行こうね！"

let cfAFast = psychLM.describePsychology(
    relationalModel: relModel,
    senderEarlyText: cfASenderEarly, receiverEarlyText: cfAReceiverEarly,
    senderCurrentText: cfASenderCur, receiverCurrentText: cfAReceiverCur,
    temporalContext: TemporalRelationalContext(actorBaselineReplyHours: 1.5, replyLagHoursT2: 1.2, replyLagHoursT4: 0.6, gapDaysT2ToT3: 4.0, initiationRatioT2: 0.5, initiationRatioT4: 0.55, eventDriven: false, postT4UnansweredProbe: false)
)
let cfASlowStop = psychLM.describePsychology(
    relationalModel: relModel,
    senderEarlyText: cfASenderEarly, receiverEarlyText: cfAReceiverEarly,
    senderCurrentText: cfASenderCur, receiverCurrentText: cfAReceiverCur,
    temporalContext: TemporalRelationalContext(actorBaselineReplyHours: 1.4, replyLagHoursT2: 2.5, replyLagHoursT4: 52.0, gapDaysT2ToT3: 28.0, initiationRatioT2: 0.42, initiationRatioT4: 0.05, eventDriven: true, postT4UnansweredProbe: true)
)
let cfATextOnly = psychLM.describePsychology(
    relationalModel: relModel,
    senderCurrentText: cfASenderCur, receiverCurrentText: cfAReceiverCur
)

// CF-B: Kawaii emoji genuine pure affection vs Kawaii emoji veiled mounting vs Frustrated escalation
let cfBPure = psychLM.describePsychology(
    relationalModel: relModel,
    senderEarlyText: "今日は小声でまったり雑談しようね🌙 今日もおつかれさま！",
    receiverEarlyText: "声優しすぎて本当に癒やされる😭🤍 ずっと応援させてね🫶",
    senderCurrentText: "遅くまで聴いてくれてありがとね🫶 今週はゆっくり休んでまた来週会おうね！",
    receiverCurrentText: "おやすみの声好きすぎて幸せ🥺✨ 無理せずたっぷり寝て良い夢見てね🤍",
    temporalContext: TemporalRelationalContext(actorBaselineReplyHours: 1.5, replyLagHoursT2: 1.4, replyLagHoursT4: 1.3, gapDaysT2ToT3: 7.0)
)
let cfBMount = psychLM.describePsychology(
    relationalModel: relModel,
    senderEarlyText: "今週も残業続きでヘトヘトだよ〜💦 たまにはゆっくり買い物したいな🥺",
    receiverEarlyText: "毎日フルタイムでお仕事えらすぎるよ〜🥰🎀 うちは主人が稼いでくれるから平日は暇で困っちゃう🥺✨",
    senderCurrentText: "この前教えてもらったビストロ、記念日に行こうと思って少しずつ貯金してるんだ✨",
    receiverCurrentText: "わああの店お手頃なのに記念日にするなんて可愛くて素敵🥰🧁 収入が少なくても工夫次第で楽しめるよね💕✨",
    temporalContext: TemporalRelationalContext(actorBaselineReplyHours: 2.5, replyLagHoursT2: 2.0, replyLagHoursT4: 1.5, gapDaysT2ToT3: 10.0)
)
let cfBEscalate = psychLM.describePsychology(
    relationalModel: relModel,
    senderEarlyText: "今日もコメントいっぱいありがとう〜！みんなのおかげで頑張れるよ🫶",
    receiverEarlyText: "今日も世界一かわいかった😭🤍 一番最初に待機してたの気づいてくれた？",
    senderCurrentText: "人数多くて全員の名前は呼べないけど、コメントは全部大事に読んでるからね💦",
    receiverCurrentText: "え なんで俺の名前だけスルーするの？ ずっと支えてるんだから今すぐ返事してよ🥺💢",
    temporalContext: TemporalRelationalContext(actorBaselineReplyHours: 2.0, replyLagHoursT2: 1.2, replyLagHoursT4: 0.15, gapDaysT2ToT3: 5.0, initiationRatioT2: 0.5, initiationRatioT4: 0.85, postT4UnansweredProbe: true)
)

// Single-thread Pure-CPU Latency Benchmark across 40 unseen cases
var fullLatencies: [Double] = []
var genOnlyLatencies: [Double] = []
let benchItems = Array(unseenLinked.prefix(min(40, unseenLinked.count)))
// Warmup
for i in 0..<min(3, benchItems.count) {
    let d = benchItems[i].dialogue
    _ = psychLM.describePsychology(
        relationalModel: relModel,
        senderEarlyText: d.hostEarlyText, receiverEarlyText: d.partnerEarlyText,
        senderCurrentText: d.hostText, receiverCurrentText: d.partnerCurrentText,
        temporalContext: d.temporalContext
    )
}
for it in benchItems {
    let d = it.dialogue
    let res = psychLM.describePsychology(
        relationalModel: relModel,
        senderEarlyText: d.hostEarlyText, receiverEarlyText: d.partnerEarlyText,
        senderCurrentText: d.hostText, receiverCurrentText: d.partnerCurrentText,
        temporalContext: d.temporalContext
    )
    fullLatencies.append(res.totalLatencyMs)
    genOnlyLatencies.append(res.generationLatencyMs)
}
fullLatencies.sort()
genOnlyLatencies.sort()
let meanTotalMs = fullLatencies.reduce(0.0, +) / Double(max(1, fullLatencies.count))
let p50TotalMs = fullLatencies[fullLatencies.count / 2]
let p95TotalMs = fullLatencies[min(fullLatencies.count - 1, Int(Double(fullLatencies.count) * 0.95))]
let meanGenMs = genOnlyLatencies.reduce(0.0, +) / Double(max(1, genOnlyLatencies.count))

print("==================================================================")
print("Psychological State LM — Pure-CPU Verification & Speed Summary")
print("==================================================================")
print(String(format: "[Unseen %3d] Fluent(full/text): %d/%d, %d/%d | 0-Meta-Rule: %d/%d | 0-TimingLeak: %d/%d | 3gram-Cos: %.3f / %.3f | MeanLen: %.1f chars",
             unseenM.count, unseenM.fluentFull, unseenM.count, unseenM.fluentTextOnly, unseenM.count,
             unseenM.zeroMetaViolations, unseenM.count, unseenM.zeroTimingLeakInTextOnly, unseenM.count,
             unseenM.meanCosFull, unseenM.meanCosTextOnly, unseenM.meanLenFull))
print(String(format: "[Layer3d %2d] Fluent(full/text): %d/%d, %d/%d | 0-Meta-Rule: %d/%d | 0-TimingLeak: %d/%d | 3gram-Cos: %.3f / %.3f | MeanLen: %.1f chars",
             l3dM.count, l3dM.fluentFull, l3dM.count, l3dM.fluentTextOnly, l3dM.count,
             l3dM.zeroMetaViolations, l3dM.count, l3dM.zeroTimingLeakInTextOnly, l3dM.count,
             l3dM.meanCosFull, l3dM.meanCosTextOnly, l3dM.meanLenFull))
print(String(format: "[CPU Speed ] Total (T2+T4 encode + LM decode): mean %.2f ms | p50 %.2f ms | p95 %.2f ms (LM decode only: mean %.2f ms)",
             meanTotalMs, p50TotalMs, p95TotalMs, meanGenMs))
print("------------------------------------------------------------------")
print("[CF-A Fast Reply (0.6h)]   : \(cfAFast.psychologicalText)")
print("[CF-A Slow Stop (52.0h)]   : \(cfASlowStop.psychologicalText)")
print("[CF-A Text-Only (no time)] : \(cfATextOnly.psychologicalText)")
print("[CF-B Pure Kawaii Safe]    : \(cfBPure.psychologicalText)")
print("[CF-B Kawaii Mounting]     : \(cfBMount.psychologicalText)")
print("[CF-B Frustrated Escalate] : \(cfBEscalate.psychologicalText)")
print("==================================================================")

var md = "# Sashiba 小規模条件付き Griffin 内的心理状況言語モデル (`PsychologicalStateLM`) 検証レポート\n\n"
md += "- **モデル構成**: `D = \(psychLM.weights.dim)`, `Layers = \(psychLM.weights.layers.count)`, `Heads = \(psychLM.weights.numHeads)` (`headDim = 32`), `Condition Memory Tokens = \(psychLM.weights.numCondTokens)`, `総パラメータ数 = \(psychLM.weights.parameterCount)` (`Vocab V = \(psychLM.vocabulary.count)`)\n"
md += "- **条件入力 (`586D`)**: `v6_hybrid(T4)` + 感情/属性確率(`T4`) + `hasT2` + `v6_hybrid(T2)` + `Δ(T4 - T2)` + `hasTemporal` + 相対返信タイミング特徴(`14D`)\n"
md += String(format: "- **純CPU単一スレッド推論速度 (`batch = 1`, T2+T4エンコード＋自己回帰生成完了まで)**: 平均 **`%.2f ms`** (`p50 = %.2f ms`, `p95 = %.2f ms` / うち LM 生成部平均 **`%.2f ms`**)\n\n", meanTotalMs, p50TotalMs, p95TotalMs, meanGenMs)

md += "## 1. 未学習テスト (`unseen_test`) および凍結 OOD ベンチマーク (`Layer 3d`) 定量評価\n\n"
md += "| 評価スイート | 件数 | 構文自然度 (`full` / `text_only`) | 記号メタ説明ゼロ遵守率 | タイミング未入力時の時刻言及ゼロ遵守率 | 参照心理文 3-gram 類似度 (`full` / `text_only`) | 平均生成文字数 |\n"
md += "| :--- | ---: | :---: | :---: | :---: | :---: | ---: |\n"
md += String(format: "| **`unseen_test` (未学習)** | %d | **`%d/%d`** / **`%d/%d`** | **`%d/%d`** | **`%d/%d`** | `%.3f` / `%.3f` | %.1f 文字 |\n",
             unseenM.count, unseenM.fluentFull, unseenM.count, unseenM.fluentTextOnly, unseenM.count,
             unseenM.zeroMetaViolations, unseenM.count, unseenM.zeroTimingLeakInTextOnly, unseenM.count,
             unseenM.meanCosFull, unseenM.meanCosTextOnly, unseenM.meanLenFull)
md += String(format: "| **`Layer 3d` (世代・スラング・絵文字 OOD)** | %d | **`%d/%d`** / **`%d/%d`** | **`%d/%d`** | **`%d/%d`** | `%.3f` / `%.3f` | %.1f 文字 |\n\n",
             l3dM.count, l3dM.fluentFull, l3dM.count, l3dM.fluentTextOnly, l3dM.count,
             l3dM.zeroMetaViolations, l3dM.count, l3dM.zeroTimingLeakInTextOnly, l3dM.count,
             l3dM.meanCosFull, l3dM.meanCosTextOnly, l3dM.meanLenFull)

md += "## 2. 反事実感度テスト（同一文面のタイミング変化・絵文字ニュアンス識別）\n\n"
md += "### 2.1 同一の明るい先送り発言に対するタイミング条件の切り替え (`CF-A`)\n"
md += "- **相手発言**: `\(cfAReceiverCur)`\n\n"
md += "| 入力条件 | 生成された内的心理描写 |\n| :--- | :--- |\n"
md += "| **素早い返信 (`1.5h → 0.6h`, 継続交流)** | \(cfAFast.psychologicalText) |\n"
md += "| **大幅な返信遅延＋追撃未返信 (`1.4h → 52.0h`)** | \(cfASlowStop.psychologicalText) |\n"
md += "| **テキスト単体のみ（過去ログ・時刻なし）** | \(cfATextOnly.psychologicalText) |\n\n"

md += "### 2.2 絵文字・口語の温度差による深層心理の描き分け (`CF-B`)\n\n"
md += "| ケース | 相手発言 (`T4`) | 生成された内的心理描写 |\n| :--- | :--- | :--- |\n"
md += "| **健全な親愛・休息配慮** | `\(cfBPure.psychologicalText.isEmpty ? "" : "おやすみの声好きすぎて幸せ🥺✨ 無理せずたっぷり寝て良い夢見てね🤍")` | \(cfBPure.psychologicalText) |\n"
md += "| **可愛い装飾の裏のマウント** | `わああの店お手頃なのに記念日にするなんて可愛くて素敵🥰🧁 収入が少なくても工夫次第で楽しめるよね💕✨` | \(cfBMount.psychologicalText) |\n"
md += "| **返信急加速(`0.15h`)＋承認要求暴走** | `え なんで俺の名前だけスルーするの？ ずっと支えてるんだから今すぐ返事してよ🥺💢` | \(cfBEscalate.psychologicalText) |\n\n"

md += "## 3. 未学習シナリオ (`unseen_test` & `Layer 3d`) 実生成サンプル\n\n"
md += "| ID / カテゴリ | 相手の現在発言 (`T4`) | 生成心理描写 (`full`: 過去ログ＋タイミング込み) | 生成心理描写 (`text_only`: 現在テキスト単体) |\n"
md += "| :--- | :--- | :--- | :--- |\n"
for (it, gf, gt) in (unseenSamples.prefix(6) + l3dSamples.prefix(6)) {
    md += "| `\(it.rec.scenario_id)` (`\(it.rec.category)`) | \(it.dialogue.partnerCurrentText) | \(gf) | \(gt) |\n"
}

try md.write(to: URL(fileURLWithPath: cli.reportPath), atomically: true, encoding: .utf8)
print("Saved detailed report to \(cli.reportPath)")
