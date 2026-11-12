import Foundation
import Sashiba

struct TrainCLI {
    var dataPath: String = ".tmp"
    var epochs: Int = 60
    var batchSize: Int = 16
    var dim: Int = SWTHawkShape.defaultDim
    var encoderLayers: Int = SWTHawkShape.encoderLayers
    var decoderLayers: Int = SWTHawkShape.decoderLayers
    var lrMax: Float = 0.003
    var lrMin: Float = 0.0003
    var wordDropout: Float = 0.25
    var latentNoiseStd: Float = 0.10
    var regWeight: Float = 0.02
    var maxSeqLen: Int = 512
    var seed: UInt64 = 42
    var exportWeightsPath: String = ".tmp/weights.safetensors"
    var importWeightsPath: String? = nil
    var earlyStopOnExactGreedy: Bool = true
    var relEpochs: Int = 15
    var relBatch: Int = 32
    var relLR: Float = 1.4e-3
    var aeWeightsPath: String = ".tmp/weights_ae.safetensors"
    var recalibrateCFM: Bool = false

    static func parse(_ args: [String]) -> TrainCLI {
        var cli = TrainCLI()
        var i = 1
        while i < args.count {
            let arg = args[i]
            switch arg {
            case "-d", "--data", "--dataset":
                i += 1
                if i < args.count { cli.dataPath = args[i] }
            case "-e", "--epochs":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.epochs = max(0, v) }
            case "-b", "--batch", "--batch-size":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.batchSize = max(1, v) }
            case "--dim":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.dim = max(8, v) }
            case "--encoder-layers":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.encoderLayers = max(1, v) }
            case "--decoder-layers":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.decoderLayers = max(1, v) }
            case "--lr":
                i += 1
                if i < args.count, let v = Float(args[i]) { cli.lrMax = v }
            case "--min-lr":
                i += 1
                if i < args.count, let v = Float(args[i]) { cli.lrMin = v }
            case "--word-dropout":
                i += 1
                if i < args.count, let v = Float(args[i]) { cli.wordDropout = min(0.95, max(0.0, v)) }
            case "--latent-noise":
                i += 1
                if i < args.count, let v = Float(args[i]) { cli.latentNoiseStd = max(0.0, v) }
            case "--reg-weight":
                i += 1
                if i < args.count, let v = Float(args[i]) { cli.regWeight = max(0.0, v) }
            case "--max-len":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.maxSeqLen = max(4, v) }
            case "--seed":
                i += 1
                if i < args.count, let v = UInt64(args[i]) { cli.seed = v }
            case "-o", "--output", "--export-weights":
                i += 1
                if i < args.count { cli.exportWeightsPath = args[i] }
            case "--import-weights":
                i += 1
                if i < args.count { cli.importWeightsPath = args[i] }
            case "--no-early-stop":
                cli.earlyStopOnExactGreedy = false
            case "--rel-epochs":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.relEpochs = max(0, v) }
            case "--rel-batch":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.relBatch = max(2, v) }
            case "--rel-lr":
                i += 1
                if i < args.count, let v = Float(args[i]) { cli.relLR = v }
            case "--ae-weights":
                i += 1
                if i < args.count { cli.aeWeightsPath = args[i] }
            case "--recalibrate-cfm":
                cli.recalibrateCFM = true
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

/// Generates Compositional Epoch Augmentation texts (`Phase 5 Section 4.1`) from the loaded dialogue corpus and configuration.
func buildAugmentedCorpus(docs: [TextDocument], config: DatasetConfiguration) -> [String] {
    var augmented: [String] = []
    if !config.baseVocabularyCharacters.isEmpty {
        augmented.append(config.baseVocabularyCharacters)
    }
    for doc in docs {
        augmented.append(doc.text)
        if !doc.summaryLevel0.isEmpty { augmented.append(doc.summaryLevel0) }
        if !doc.summaryLevel1.isEmpty { augmented.append(doc.summaryLevel1) }
        if !doc.summary.isEmpty { augmented.append(doc.summary) }
    }

    let trainCases = config.relationalDialogueCases.filter { !$0.isZeroShotUnseen }
    for rc in trainCases {
        augmented.append(rc.hostText)
        augmented.append(rc.partnerEarlyText)
        augmented.append(rc.partnerCurrentText)
        augmented.append(rc.harmonizedPartnerText)
        augmented.append(rc.insightLevel0)
        augmented.append(rc.insightLevel1)
        augmented.append(rc.insightLevel2)
    }

    // Compositional cross-recombination across (T1..T2) context prefixes and (T3..T4) emotional reaction suffixes
    let delim = config.classifierRules.sentenceDelimiters.first ?? "。"
    var earlyPools: [String] = []
    var latePools: [String] = []
    for rc in trainCases {
        earlyPools.append(rc.partnerEarlyText)
        let parts = rc.partnerCurrentText.components(separatedBy: delim).filter { !$0.isEmpty }
        if parts.count >= 3 {
            let half = parts.count / 2
            let suffix = parts[half..<parts.count].map { $0 + delim }.joined()
            latePools.append(suffix)
        }
    }
    for i in 0..<earlyPools.count {
        for step in 1...min(8, max(1, latePools.count - 1)) {
            let j = (i + step) % max(1, latePools.count)
            if j < latePools.count {
                augmented.append(earlyPools[i] + latePools[j])
            }
        }
    }
    return augmented
}

MLXSWTHawkAutoencoder.configureLowMemoryCache()
let cli = TrainCLI.parse(CommandLine.arguments)
let dataURL = URL(fileURLWithPath: cli.dataPath)
let docs = try TextVocabulary.loadDocuments(from: dataURL)
let datasetConfig = DatasetConfiguration.load(from: cli.dataPath)

guard !docs.isEmpty else {
    print("No text documents found in \(cli.dataPath). Place .json or .txt files in .tmp/")
    exit(1)
}

let augmentedCorpus = buildAugmentedCorpus(docs: docs, config: datasetConfig)
let vocab = TextVocabulary(texts: augmentedCorpus)
let subbandClassifier = SubbandTokenClassifier(vocabulary: vocab, rules: datasetConfig.classifierRules)

var encodedDocs: [[Int]] = []
var maxObservedLen = 0
for doc in docs {
    let ids = vocab.encode(doc.text, maxLength: cli.maxSeqLen, addEOS: false, padToLength: nil)
    encodedDocs.append(ids)
    if ids.count > maxObservedLen {
        maxObservedLen = ids.count
    }
}
for rc in datasetConfig.relationalDialogueCases where !rc.isZeroShotUnseen {
    for text in [rc.hostText, rc.partnerEarlyText, rc.partnerCurrentText, rc.harmonizedPartnerText, rc.insightLevel2] {
        let count = vocab.encode(text, maxLength: cli.maxSeqLen, addEOS: false, padToLength: nil).count
        if count > maxObservedLen {
            maxObservedLen = count
        }
    }
}
let seqLen = min(cli.maxSeqLen, max(4, maxObservedLen))

func padRow(_ ids: [Int], to length: Int) -> [Int] {
    var row = Array(ids.prefix(length))
    if row.count < length {
        row.append(contentsOf: repeatElement(TextVocabulary.padId, count: length - row.count))
    }
    return row
}

var allTokens: [Int] = []
allTokens.reserveCapacity(docs.count * seqLen)
for ids in encodedDocs {
    allTokens.append(contentsOf: padRow(ids, to: seqLen))
}

let cfmDocs = docs
var cfmFullTokens: [Int] = []
var cfmSummaryTokens: [Int] = []
var cfmSummaryL1Tokens: [Int] = []
var cfmSummaryL0Tokens: [Int] = []
cfmFullTokens.reserveCapacity(cfmDocs.count * seqLen)
cfmSummaryTokens.reserveCapacity(cfmDocs.count * seqLen)
cfmSummaryL1Tokens.reserveCapacity(cfmDocs.count * seqLen)
cfmSummaryL0Tokens.reserveCapacity(cfmDocs.count * seqLen)

for doc in cfmDocs {
    let fullIds = vocab.encode(doc.text, maxLength: seqLen, addEOS: false, padToLength: nil)
    let sumL2Ids = vocab.encode(doc.summary(forGamma: 1.0), maxLength: seqLen, addEOS: false, padToLength: nil)
    let sumL1Ids = vocab.encode(doc.summary(forGamma: 0.5), maxLength: seqLen, addEOS: false, padToLength: nil)
    let sumL0Ids = vocab.encode(doc.summary(forGamma: 0.0), maxLength: seqLen, addEOS: false, padToLength: nil)
    cfmFullTokens.append(contentsOf: padRow(fullIds, to: seqLen))
    cfmSummaryTokens.append(contentsOf: padRow(sumL2Ids, to: seqLen))
    cfmSummaryL1Tokens.append(contentsOf: padRow(sumL1Ids, to: seqLen))
    cfmSummaryL0Tokens.append(contentsOf: padRow(sumL0Ids, to: seqLen))
}

var trainingRelationalCases: [(
    hostTokens: [Int],
    partnerEarlyTokens: [Int],
    partnerCurrentTokens: [Int],
    harmonizedTokens: [Int],
    insightL0Tokens: [Int],
    insightL1Tokens: [Int],
    insightL2Tokens: [Int]
)] = []
for rc in datasetConfig.relationalDialogueCases where !rc.isZeroShotUnseen {
    let hIds = padRow(vocab.encode(rc.hostText, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let p1Ids = padRow(vocab.encode(rc.partnerEarlyText, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let p2Ids = padRow(vocab.encode(rc.partnerCurrentText, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let harmIds = padRow(vocab.encode(rc.harmonizedPartnerText, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let l0Ids = padRow(vocab.encode(rc.insightLevel0, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let l1Ids = padRow(vocab.encode(rc.insightLevel1, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let l2Ids = padRow(vocab.encode(rc.insightLevel2, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    trainingRelationalCases.append((hIds, p1Ids, p2Ids, harmIds, l0Ids, l1Ids, l2Ids))
}

var weights: SWTHawkWeights = if let importPath = cli.importWeightsPath {
    try SWTHawkWeights.load(from: URL(fileURLWithPath: importPath))
} else {
    SWTHawkWeights.makeRandom(
        vocabSize: vocab.count,
        dim: cli.dim,
        latentDim: SWTHawkShape.latentDim,
        mlpDim: 2 * cli.dim,
        encoderLayers: cli.encoderLayers,
        decoderLayers: cli.decoderLayers,
        serializedVocabulary: vocab.serialize(),
        subbandClassifier: subbandClassifier,
        seed: cli.seed
    )
}
weights.subbandClassifier = subbandClassifier

let model = MLXSWTHawkAutoencoder(weights: weights)
let trainer = MLXSWTHawkTrainer(model: model, learningRate: cli.lrMax, seed: cli.seed &+ 99)

let numDocs = docs.count
let batchSize = min(cli.batchSize, numDocs)
let stepsPerEpoch = max(1, (numDocs + batchSize - 1) / batchSize)
let totalSteps = max(1, cli.epochs * stepsPerEpoch)
let warmupSteps = min(max(5, totalSteps / 20), max(1, totalSteps / 2))
let scheduler = CosineLRScheduler(lrMax: cli.lrMax, lrMin: cli.lrMin)

print("==================================================================")
print("SWT-Griffin Relational Dynamics Engine & CFM Training (Phase 5)")
print("==================================================================")
print("Backend          : MLX GPU (Metal / CUDA)")
print("Train Dialogues  : \(numDocs) multi-turn dialogues (\(augmentedCorpus.count) compositional combinations)")
print("Sequence length T: \(seqLen) tokens | Vocab size V: \(vocab.count)")
print("Model dim D      : \(weights.dim) | AE z: \(weights.latentDim) (z_LL: \(weights.latentDim / 2), z_HH: \(weights.latentDim / 2)) | Rel z_rel: \(weights.relational.latentDim)")
print("Griffin layers   : \(weights.hawkBranch.count) (RG-LRU + Local Window Attn W=\(SWTHawkShape.windowSize)) | Decoder layers: \(weights.decoder.layers.count)")
print("Word Dropout p   : \(String(format: "%.2f", cli.wordDropout)) | Latent Noise σ: \(String(format: "%.2f", cli.latentNoiseStd)) | Reg α: \(String(format: "%.4f", cli.regWeight))")
print("Parameters       : \(weights.parameterCount)")
print("------------------------------------------------------------------")

let startTime = Date()
var globalStep = 0

if cli.epochs > 0 {
    for epoch in 1...cli.epochs {
        var epochLoss: Float = 0.0
        var epochRecLoss: Float = 0.0
        var epochRegLoss: Float = 0.0
        var epochTokAcc: Float = 0.0
        var batchCount = 0

        var offset = 0
        while offset < numDocs {
            let curB = min(batchSize, numDocs - offset)
            let slice = Array(allTokens[(offset * seqLen)..<((offset + curB) * seqLen)])
            globalStep += 1
            let lr = scheduler.learningRate(step: globalStep, totalSteps: totalSteps, warmupSteps: warmupSteps)

            let metrics = trainer.trainStep(
                tokens: slice,
                batch: curB,
                time: seqLen,
                learningRate: lr,
                wordDropout: cli.wordDropout,
                regWeight: cli.regWeight,
                latentNoiseStd: cli.latentNoiseStd
            )
            epochLoss += metrics.totalLoss
            epochRecLoss += metrics.reconstructionLoss
            epochRegLoss += metrics.regularizationLoss
            epochTokAcc += metrics.tokenAccuracy
            batchCount += 1
            offset += curB
        }

        let invB = 1.0 / Float(max(1, batchCount))
        let avgLoss = epochLoss * invB
        let avgRec = epochRecLoss * invB
        let avgReg = epochRegLoss * invB
        let avgTokAcc = epochTokAcc * invB

        let shouldLog = (epoch == 1) || (epoch % 5 == 0) || (epoch == cli.epochs) || (avgTokAcc >= 0.9995)
        if shouldLog {
            let evalResult = trainer.evaluateGreedyReconstruction(
                tokens: allTokens,
                batch: numDocs,
                time: seqLen
            )
            print(String(
                format: "Epoch %4d/%4d | L_total: %.4f (L_rec: %.4f, L_reg: %.4f, |g|: %.4f) | TF Acc (w/ drop+noise): %6.2f%% | Greedy z->Text Acc: %6.2f%% (%d/%d exact)",
                epoch, cli.epochs,
                avgLoss, avgRec, avgReg, trainer.lastGradientNorm,
                avgTokAcc * 100.0,
                evalResult.tokenAccuracy * 100.0,
                Int((evalResult.exactMatchRate * Float(numDocs)).rounded()),
                numDocs
            ))

            if cli.earlyStopOnExactGreedy && epoch >= 20 && evalResult.exactMatchRate >= 1.0 {
                print("Reached 100.00% exact greedy reconstruction across all \(numDocs) documents from z = [z_LL(\(weights.latentDim / 2)); z_HH(\(weights.latentDim / 2))] ∈ R^\(weights.latentDim)!")
                break
            }
        }
    }
}

if cli.epochs > 0 || weights.flowMatching.numPairs == 0 || cli.recalibrateCFM {
    // Stage 2, Stage 3 & Phase 5: Multi-Resolution Relational Insight CFM (γ ∈ {1.0, 0.5, 0.0}) & Latent Harmonization CFM
    let cfmStats = trainer.trainAndCalibrateFlowMatching(
        fullTokens: cfmFullTokens,
        summaryTokens: cfmSummaryTokens,
        summaryLevel1Tokens: cfmSummaryL1Tokens,
        summaryLevel0Tokens: cfmSummaryL0Tokens,
        relationalCases: trainingRelationalCases,
        numDocs: cfmDocs.count,
        prototypeCount: numDocs,
        seqLen: seqLen
    )
    print(String(
        format: "[Phase 5 Relational CFM] Registered %d multi-res insight & harmonization pairs | L_align: %.4f | L_CFM: %.6f | L_polarity: %.4f",
        cfmStats.registeredPairs,
        cfmStats.alignLoss,
        cfmStats.cfmLoss,
        cfmStats.polarityLoss
    ))
} else {
    print("[Phase 5 Relational CFM] Reusing \(weights.flowMatching.numPairs) registered pairs from imported weights")
}

let elapsed = Date().timeIntervalSince(startTime)
if cli.epochs > 0 {
    let finalEval = trainer.evaluateGreedyReconstruction(
        tokens: allTokens,
        batch: numDocs,
        time: seqLen
    )
    print("------------------------------------------------------------------")
    print(String(
        format: "Training finished in %.2fs | Greedy Token Accuracy: %.2f%% | Exact Document Reconstruction: %d/%d (%.1f%%)",
        elapsed,
        finalEval.tokenAccuracy * 100.0,
        Int((finalEval.exactMatchRate * Float(numDocs)).rounded()),
        numDocs,
        finalEval.exactMatchRate * 100.0
    ))
}

if (cli.epochs > 0 || cli.importWeightsPath != nil) && !cli.exportWeightsPath.isEmpty {
    let exportURL = URL(fileURLWithPath: cli.exportWeightsPath)
    try FileManager.default.createDirectory(
        at: exportURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    // Autoencoder-stage snapshot (before relational E2E): starting point for per-fold retraining in `relational_head`.
    if !cli.aeWeightsPath.isEmpty {
        try trainer.weights.save(to: URL(fileURLWithPath: cli.aeWeightsPath))
        print("Saved autoencoder-stage weights to \(cli.aeWeightsPath)")
    }

    // Phase 5.1: End-to-End Dual-Stream relational training on the train split only (split == "train").
    if cli.relEpochs > 0 {
        let relTrainCases = datasetConfig.relationalDialogueCases.filter { !$0.isZeroShotUnseen }
        var relLen = 4
        for rc in relTrainCases {
            for text in [rc.hostText, rc.partnerEarlyText, rc.partnerCurrentText, rc.harmonizedPartnerText] {
                relLen = max(relLen, vocab.encode(text, maxLength: cli.maxSeqLen, addEOS: false, padToLength: nil).count)
            }
        }
        print("------------------------------------------------------------------")
        print("[Phase 5.1] Relational E2E stage: \(relTrainCases.count) train scenarios | T = \(relLen) | epochs = \(cli.relEpochs)")
        let relStart = Date()
        let telemetry = RelationalPipeline.trainRelationalStage(
            model: model,
            trainCases: relTrainCases,
            vocab: vocab,
            time: relLen,
            epochs: cli.relEpochs,
            batchSize: cli.relBatch,
            learningRate: cli.relLR,
            seed: cli.seed &+ 7
        )
        print(String(format: "[Phase 5.1] Relational stage finished in %.1fs", Date().timeIntervalSince(relStart)))
        for m in telemetry.modules {
            print(String(
                format: "  %-36@ params %8d | ||W_final - W_0||_F %.6f | mean ||grad||_2 %.6f",
                m.moduleName as NSString, m.parameterCount, m.weightDeltaFrobeniusNorm, m.meanGradientNorm
            ))
        }
    }

    let exported = trainer.weights
    try exported.save(to: exportURL)
    print("Saved trained weights to \(cli.exportWeightsPath)")
}
