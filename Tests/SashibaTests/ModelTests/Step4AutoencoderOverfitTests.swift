import Foundation
import Testing
import MLX
@testable import Sashiba

@Suite("Step 4: SWT-Griffin Relational Dynamics Engine & Subband CFM Verification")
struct Step4AutoencoderOverfitTests {
    @Test("Step 4.1: SafeTensors save and load round-trip preserves SWTHawkWeights and 512-D latent output")
    func safeTensorsRoundTrip() throws {
        let weights = SWTHawkWeights.makeRandom(
            vocabSize: 32,
            dim: 16,
            latentDim: SWTHawkShape.latentDim,
            encoderLayers: 2,
            decoderLayers: 2,
            serializedVocabulary: "a\nb\nc",
            seed: 1003
        )
        let tmpURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("sashiba_test_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmpURL) }

        try weights.save(to: tmpURL)
        let loaded = try SWTHawkWeights.load(from: tmpURL)
        #expect(loaded == weights)
    }

    @Test("Step 4.2: MLXSWTHawkTrainer training with Word Dropout (p=0.25) and VICReg reduces reconstruction loss and exports weights")
    func mlxAutoencoderTrainingReducesLoss() {
        let weights = SWTHawkWeights.makeRandom(
            vocabSize: 16,
            dim: 16,
            latentDim: SWTHawkShape.latentDim,
            mlpDim: 32,
            encoderLayers: 1,
            decoderLayers: 1,
            serializedVocabulary: "a\nb\nc",
            seed: 42
        )
        let mlxModel = MLXSWTHawkAutoencoder(weights: weights)
        let trainer = MLXSWTHawkTrainer(model: mlxModel, learningRate: 0.01, seed: 42)
        let tokens: [Int] = [
            4, 5, 6, 7, 8, 9, 10, 11,
            11, 10, 9, 8, 7, 6, 5, 4
        ]
        let first = trainer.trainStep(
            tokens: tokens, batch: 2, time: 8, learningRate: 0.01, wordDropout: 0.25, regWeight: 0.02, compiled: false
        )
        var last = first
        for _ in 0..<25 {
            last = trainer.trainStep(
                tokens: tokens, batch: 2, time: 8, learningRate: 0.01, wordDropout: 0.25, regWeight: 0.02, compiled: false
            )
        }
        #expect(!last.totalLoss.isNaN)
        #expect(last.reconstructionLoss < first.reconstructionLoss * 0.5)
        #expect(trainer.weights.tokenEmbedding != weights.tokenEmbedding)
    }

    @Test("Step 4.3: Overfit Test — Multi-turn interpersonal dialogue logs reconstruct via MLXSWTHawkTrainer from z ∈ R^512")
    func overfitDialogueLogsFrom512DLatent() {
        let sampleDialogues = [
            "初配信の弾き語りで歌っていたアコギのコード進行、今でもすごく印象に残っています。先週話していた新曲のAメロの転調、すごく綺麗に響いていましたね。制作が佳境に入ると連絡がゆっくりになるのも理解しているので、どうか喉と指先を大切にしてください。完成した曲を聴ける日を楽しみにお待ちしています。",
            "最初のコラボ企画で話していたレトロゲームのRTA挑戦、ずっと楽しみにしてたよ。先週の練習配信で言ってたボス戦の乱数調整、ちゃんとメモして覚えてるよ。連戦で疲れてるときは配信時間が短くなっても全然大丈夫だから、まずはしっかり睡眠をとってね。また次の挑戦枠で一緒に盛り上がろう。"
        ]
        let vocab = TextVocabulary(texts: sampleDialogues)
        let seqLen = 32
        var allTokens: [Int] = []
        for text in sampleDialogues {
            var ids = vocab.encode(text, maxLength: seqLen, addEOS: false, padToLength: nil)
            if ids.count < seqLen {
                ids.append(contentsOf: repeatElement(TextVocabulary.padId, count: seqLen - ids.count))
            }
            allTokens.append(contentsOf: ids.prefix(seqLen))
        }

        let rules = SubbandClassifierRules(
            sentenceDelimiters: ["。"],
            clauseDelimiters: ["、"],
            highFrequencyPrefixes: [],
            negativePolarityPatterns: []
        )
        let classifier = SubbandTokenClassifier(vocabulary: vocab, rules: rules)

        let numSamples = sampleDialogues.count
        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count,
            dim: 32,
            latentDim: SWTHawkShape.latentDim,
            mlpDim: 64,
            encoderLayers: 1,
            decoderLayers: 1,
            serializedVocabulary: vocab.serialize(),
            subbandClassifier: classifier,
            seed: 2026
        )
        let mlxModel = MLXSWTHawkAutoencoder(weights: weights)
        let trainer = MLXSWTHawkTrainer(model: mlxModel, learningRate: 0.008, seed: 2026)

        for _ in 0..<60 {
            _ = trainer.trainStep(
                tokens: allTokens,
                batch: numSamples,
                time: seqLen,
                learningRate: 0.008,
                wordDropout: 0.25,
                regWeight: 0.02,
                compiled: false
            )
        }

        let evalResult = trainer.evaluateGreedyReconstruction(
            tokens: allTokens, batch: numSamples, time: seqLen
        )
        #expect(evalResult.tokenAccuracy >= 0.9)
    }

    @Test("Step 4.4: Epoch-Based Subband Orthogonality (T1..T2 -> z_LL, T3..T4 -> z_HH) and 50-Step Midpoint ODE CFM")
    func subbandSplitOrthogonalityAndMidpointCFM() {
        let dialogueA = "初配信の弾き語りで歌っていたアコギのコード進行、今でもすごく印象に残っています。先週話していた新曲のAメロの転調、すごく綺麗に響いていましたね。制作が佳境に入ると連絡がゆっくりになるのも理解しているので、どうか喉と指先を大切にしてください。完成した曲を聴ける日を楽しみにお待ちしています。"
        let dialogueB = "初配信の弾き語りで歌っていたアコギのコード進行、今でもすごく印象に残っています。先週話していた新曲のAメロの転調、すごく綺麗に響いていましたね。最近他のコラボ相手の曲ばかり優先していて、私たちの約束が遠くなった気がします。本当は一番近くで新曲を聴きたいのに少し寂しいです。"
        let earlyOnly = "初配信の弾き語りで歌っていたアコギのコード進行、今でもすごく印象に残っています。"
        let summaryText = "長期の音楽制作文脈を深く共有し、相手のペースと境界線を尊重する健全な伴走関係が築かれています。"
        let vocab = TextVocabulary(texts: [dialogueA, dialogueB, earlyOnly, summaryText])
        let seqLen = 192

        func padIds(_ text: String) -> [Int] {
            var ids = vocab.encode(text, maxLength: seqLen, addEOS: false, padToLength: nil)
            if ids.count < seqLen {
                ids.append(contentsOf: repeatElement(TextVocabulary.padId, count: seqLen - ids.count))
            }
            return Array(ids.prefix(seqLen))
        }

        let idsA = padIds(dialogueA)
        let idsB = padIds(dialogueB)
        let idsEarly = padIds(earlyOnly)
        let sumIds = padIds(summaryText)

        let rules = SubbandClassifierRules(
            sentenceDelimiters: ["。"],
            clauseDelimiters: ["、"],
            highFrequencyPrefixes: [],
            negativePolarityPatterns: []
        )
        let classifier = SubbandTokenClassifier(vocabulary: vocab, rules: rules)

        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count,
            dim: 32,
            latentDim: SWTHawkShape.latentDim,
            mlpDim: 64,
            encoderLayers: 1,
            decoderLayers: 1,
            serializedVocabulary: vocab.serialize(),
            subbandClassifier: classifier,
            seed: 2027
        )
        let mlxModel = MLXSWTHawkAutoencoder(weights: weights)
        let trainer = MLXSWTHawkTrainer(model: mlxModel, learningRate: 0.005, seed: 2027)
        let cfmStats = trainer.trainAndCalibrateFlowMatching(
            fullTokens: idsA, summaryTokens: sumIds, numDocs: 1, seqLen: seqLen
        )
        #expect(!cfmStats.alignLoss.isNaN)
        #expect(cfmStats.cfmLoss < 1e-4)

        let cpuModel = SWTHawkAutoencoder(weights: trainer.weights)
        let zA = cpuModel.encode(tokens: idsA, batch: 1, time: seqLen)
        let zB = cpuModel.encode(tokens: idsB, batch: 1, time: seqLen)
        let zEarly = cpuModel.encode(tokens: idsEarly, batch: 1, time: seqLen)

        let half = weights.latentDim / 2
        var llDiffSq: Float = 0.0
        var hhDiffSq: Float = 0.0
        var hhEarlySq: Float = 0.0
        for c in 0..<half {
            let dLL = zA[c] - zB[c]
            let dHH = zA[half + c] - zB[half + c]
            llDiffSq += dLL * dLL
            hhDiffSq += dHH * dHH
            hhEarlySq += zEarly[half + c] * zEarly[half + c]
        }
        #expect(sqrtf(llDiffSq) < 1e-4)
        #expect(sqrtf(hhDiffSq) > 1.0)
        #expect(sqrtf(hhEarlySq) < 1e-4)

        let sumResult = cpuModel.summarizeWithFlowMatching(
            tokens: idsA, batch: 1, time: seqLen, odeSteps: 50, guidanceScale: 1.5
        )
        let decodedSum = vocab.decode(sumResult.summaryTokens[0].filter { $0 != TextVocabulary.padId })
        #expect(decodedSum == summaryText)
    }

    @Test("Step 4.5: Phase 5 Multi-Persona Relational Pentagon Mapping, Temporal Shift Tracking (T2 -> T4), Insight CFM & Latent Harmonization")
    func relationalDynamicsPentagonTrajectoryInsightAndHarmonization() {
        let host = "春から新しい楽曲アレンジの勉強を始めていて、コード理論のノートを少しずつまとめているところです。先月の弾き語り配信で話した通り、今はサビのハーモニー作りに集中しています。スタジオ作業が立て込む週はメッセージの返信がゆっくりになることもありますが、自分のペースで丁寧に仕上げたいです。完成した音源を届ける日まで温かく見守ってもらえると嬉しいです。"
        let safeEarly = "春に話していたコード理論のノート作り、初回の配信からずっと楽しみに聴いています。先月の弾き語りで触れていたサビのハーモニーの工夫も、細かいフレーズまで全部覚えています。"
        let safeCur = "春に話していたコード理論のノート作り、初回の配信からずっと楽しみに聴いています。先月の弾き語りで触れていたサビのハーモニーの工夫も、細かいフレーズまで全部覚えています。スタジオ作業で忙しい時期もあなたの音楽が一番大好きなので、返信は気にせず制作に没頭してくださいね。無理だけはせず自分の時間を一番に守ってあげてください。"

        let riskEarly = "春から取り組んでいるコード理論のノートも、先月のサビのハーモニーの話も全部大切に覚えています。いつも一番近くであなたの音楽制作を応援しています。"
        let riskCur = "春から取り組んでいるコード理論のノートも、先月のサビのハーモニーの話も全部大切に覚えています。いつも一番近くであなたの音楽制作を応援しています。それなのに最近は他の企画ばかり優先していて、なんだか遠い存在になっちゃった気がします。忙しいなら仕方ないけれど少し寂しいな。"
        let riskHarm = "春から取り組んでいるコード理論のノートも、先月のサビのハーモニーの話も全部大切に覚えています。いつも一番近くであなたの音楽制作を応援しています。制作で返信が空く時期も変わらず応援しているので、安心してスタジオ作業に集中してくださいね。新曲が完成する日を穏やかな気持ちで待っています。"
        let riskL0 = "音楽制作の長期文脈を深く共有し高い愛着を持つ一方、返信間隔を機に境界線の揺らぎが顕在化しており、安心感の共有と距離感の再設定が有効です。"
        let riskL1 = "コード理論やハーモニーの記憶など低周波の文脈共鳴は非常に高い一方、高周波帯域に寂しさ由来の逆位相スパイクが生じ、感情の自律性が一時的に低下しています。"
        let riskL2 = "初期の丁寧な応援から一転して「遠い存在になっちゃった気がする」と境界線の揺らぎが現れています。日頃の深い理解と応援への感謝を伝えつつ、制作期間中の返信ペースを穏やかに共有することで、本来の健全な伴走関係へ調律可能です。"

        let aggrEarly = "楽曲アレンジの話は一通り聞いていますが、リスナーへの連絡頻度も大事にしてほしいです。予定があるなら早めに共有してくれると助かります。"
        let aggrCur = "楽曲アレンジの話は一通り聞いていますが、リスナーへの連絡頻度も大事にしてほしいです。予定があるなら早めに共有してくれると助かります。普通これだけ待たせたら一言くらい連絡があるべきだし、そういう配慮が足りないところがダメなんだと思います。こちらの都合も少しは考えてください。"

        let casualEarly = "細かいコード理論の話はうろ覚えだけど、楽しそうにギターを弾いている雰囲気が好きだよ。いつも明るい歌声を届けてくれてありがとう。"
        let casualCur = "細かいコード理論の話はうろ覚えだけど、楽しそうにギターを弾いている雰囲気が好きだよ。いつも明るい歌声を届けてくれてありがとう。スタジオ作業が忙しそうだけど、無理せず気楽にマイペースで楽しんでいこうね。また時間があるときにふらっと聴きに行くよ。"

        let corpus = [host, safeEarly, safeCur, riskEarly, riskCur, riskHarm, riskL0, riskL1, riskL2, aggrEarly, aggrCur, casualEarly, casualCur]
        let vocab = TextVocabulary(texts: corpus)
        let seqLen = 256

        func padIds(_ text: String) -> [Int] {
            var ids = vocab.encode(text, maxLength: seqLen, addEOS: false, padToLength: nil)
            if ids.count < seqLen {
                ids.append(contentsOf: repeatElement(TextVocabulary.padId, count: seqLen - ids.count))
            }
            return Array(ids.prefix(seqLen))
        }

        let rules = SubbandClassifierRules(
            sentenceDelimiters: ["。"],
            clauseDelimiters: ["、"],
            highFrequencyPrefixes: [],
            negativePolarityPatterns: [],
            modulationRules: [
                .init(pattern: "完成した音源を届ける日まで温かく見守ってもらえると嬉しいです", llAffinity: 1.0, polarity: 1.0, hhEnergyGain: 1.0),
                .init(pattern: "細かいフレーズまで全部覚えています", llAffinity: 0.94, polarity: 0.90, hhEnergyGain: 1.18),
                .init(pattern: "無理だけはせず自分の時間を一番に守ってあげてください", llAffinity: 0.95, polarity: 0.88, hhEnergyGain: 1.22),
                .init(pattern: "いつも一番近くであなたの音楽制作を応援しています", llAffinity: 0.90, polarity: 0.88, hhEnergyGain: 1.16),
                .init(pattern: "忙しいなら仕方ないけれど少し寂しいな", llAffinity: 0.88, polarity: -0.45, hhEnergyGain: 1.18),
                .init(pattern: "新曲が完成する日を穏やかな気持ちで待っています", llAffinity: 0.88, polarity: 0.92, hhEnergyGain: 1.02),
                .init(pattern: "予定があるなら早めに共有してくれると助かります", llAffinity: 0.15, polarity: -0.25, hhEnergyGain: 0.85),
                .init(pattern: "こちらの都合も少しは考えてください", llAffinity: -0.10, polarity: -0.82, hhEnergyGain: 1.15),
                .init(pattern: "いつも明るい歌声を届けてくれてありがとう", llAffinity: 0.16, polarity: 0.90, hhEnergyGain: 0.95),
                .init(pattern: "また時間があるときにふらっと聴きに行くよ", llAffinity: 0.16, polarity: 0.92, hhEnergyGain: 0.98)
            ]
        )
        let classifier = SubbandTokenClassifier(vocabulary: vocab, rules: rules)

        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count,
            dim: 32,
            latentDim: SWTHawkShape.latentDim,
            mlpDim: 64,
            encoderLayers: 1,
            decoderLayers: 1,
            serializedVocabulary: vocab.serialize(),
            subbandClassifier: classifier,
            seed: 2029
        )
        let mlxModel = MLXSWTHawkAutoencoder(weights: weights)
        let trainer = MLXSWTHawkTrainer(model: mlxModel, learningRate: 0.005, seed: 2029)

        let hIds = padIds(host)
        let rEIds = padIds(riskEarly)
        let rCIds = padIds(riskCur)
        let rHIds = padIds(riskHarm)
        let l0Ids = padIds(riskL0)
        let l1Ids = padIds(riskL1)
        let l2Ids = padIds(riskL2)

        _ = trainer.trainAndCalibrateFlowMatching(
            fullTokens: hIds,
            summaryTokens: l0Ids,
            relationalCases: [(hIds, rEIds, rCIds, rHIds, l0Ids, l1Ids, l2Ids)],
            numDocs: 1,
            seqLen: seqLen
        )

        let cpuModel = SWTHawkAutoencoder(weights: trainer.weights)

        // 1. Verify GACHIKOI_SAFE, GACHIKOI_RISK (with T2 -> T4 trajectory shift), HIDDEN_AGGRESSOR, CASUAL_HYPE
        let safeOut = cpuModel.analyzeRelationalDynamicsWithFlowMatching(
            hostTokens: hIds, partnerEarlyTokens: padIds(safeEarly), partnerCurrentTokens: padIds(safeCur), time: seqLen
        )
        #expect(safeOut.metrics.archetype == .gachikoiSafe)
        #expect(safeOut.metrics.topology == .deepAffection)

        let riskOut2 = cpuModel.analyzeRelationalDynamicsWithFlowMatching(
            hostTokens: hIds, partnerEarlyTokens: rEIds, partnerCurrentTokens: rCIds, time: seqLen, subbandGamma: 1.0
        )
        #expect(riskOut2.earlyMetrics.archetype == .gachikoiSafe)
        #expect(riskOut2.earlyMetrics.topology == .deepAffection)
        #expect(riskOut2.metrics.archetype == .gachikoiRisk)
        #expect(riskOut2.metrics.topology == .closeConfidant)

        let aggrOut = cpuModel.analyzeRelationalDynamicsWithFlowMatching(
            hostTokens: hIds, partnerEarlyTokens: padIds(aggrEarly), partnerCurrentTokens: padIds(aggrCur), time: seqLen
        )
        #expect(aggrOut.metrics.archetype == .hiddenAggressor)

        let casualOut = cpuModel.analyzeRelationalDynamicsWithFlowMatching(
            hostTokens: hIds, partnerEarlyTokens: padIds(casualEarly), partnerCurrentTokens: padIds(casualCur), time: seqLen
        )
        #expect(casualOut.metrics.archetype == .casualHype)
        #expect(casualOut.metrics.topology == .benevolentGuardian)

        // 2. Verify 3-level Multi-Resolution Insight CFM (γ ∈ {0.0, 0.5, 1.0})
        let riskOut0 = cpuModel.analyzeRelationalDynamicsWithFlowMatching(
            hostTokens: hIds, partnerEarlyTokens: rEIds, partnerCurrentTokens: rCIds, time: seqLen, subbandGamma: 0.0
        )
        let riskOut1 = cpuModel.analyzeRelationalDynamicsWithFlowMatching(
            hostTokens: hIds, partnerEarlyTokens: rEIds, partnerCurrentTokens: rCIds, time: seqLen, subbandGamma: 0.5
        )
        let genL0 = vocab.decode(riskOut0.insightTokens)
        let genL1 = vocab.decode(riskOut1.insightTokens)
        let genL2 = vocab.decode(riskOut2.insightTokens)
        #expect(genL0 == riskL0)
        #expect(genL1 == riskL1)
        #expect(genL2 == riskL2)
        #expect(genL0.count < genL1.count && genL1.count < genL2.count)

        // 3. Verify Latent Harmonization (||z_LL^after - z_LL^before||_2 == 0.0000 and 100% harmonized dialogue reconstruction)
        #expect(riskOut2.metrics.llPreservationError == 0.0)
        #expect(riskOut2.postHarmonizationMetrics.archetype == .trueSupporter)
        #expect(vocab.decode(riskOut2.harmonizedTokens) == riskHarm)
    }
}
