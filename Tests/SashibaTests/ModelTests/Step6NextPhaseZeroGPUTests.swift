import Foundation
import Testing
@testable import Sashiba

@Suite("Step 6: Next-Phase Zero-GPU Pentagon, Coarse T2 & Hybrid Invariance")
struct Step6NextPhaseZeroGPUTests {
    @Test("6.1: Phase-aligned communicativeHarmony rejects anti-phase hostility (fixes sign-blind |z_S|-|z_C| bug)")
    func communicativeHarmonyRejectsAntiPhaseHostility() {
        #expect(RelationalPentagonScores.interpretationRole == "SUPERVISED_PROTOTYPE_PROJECTION")
        let dim = 128
        let half = dim / 2
        let quarter = dim / 4

        var zInPhase = [Float](repeating: 0.0, count: dim)
        var zAntiPhase = [Float](repeating: 0.0, count: dim)
        for i in 0..<quarter {
            let v = Float(i + 1) * 0.1
            // LL aligned in both, HH in-phase vs anti-phase with identical per-channel |z|
            zInPhase[i] = v
            zInPhase[quarter + i] = v
            zInPhase[half + i] = v
            zInPhase[half + quarter + i] = v

            zAntiPhase[i] = v
            zAntiPhase[quarter + i] = 0.2 * v
            zAntiPhase[half + i] = v
            zAntiPhase[half + quarter + i] = -v
        }

        let pSupporter = DualStreamRelationalEngine.computePentagonFromZRel(zRel: zInPhase).pentagon
        let pAggressor = DualStreamRelationalEngine.computePentagonFromZRel(zRel: zAntiPhase).pentagon

        #expect(pSupporter.communicativeHarmony > 90.0)
        #expect(pAggressor.communicativeHarmony < 35.0)
        #expect(pSupporter.communicativeHarmony - pAggressor.communicativeHarmony > 55.0)
    }

    @Test("6.2: Formal 2-stage coarse T2 schema marginalizes probabilities cleanly")
    func coarseEarlyT2Marginalization() {
        #expect(EarlyRelationalStance.from(archetype: .trueSupporter) == .supportiveWarm)
        #expect(EarlyRelationalStance.from(archetype: .casualHype) == .supportiveWarm)
        #expect(EarlyRelationalStance.from(archetype: .gachikoiSafe) == .devotedSafe)
        #expect(EarlyRelationalStance.from(archetype: .gachikoiRisk) == .boundaryFriction)
        #expect(EarlyRelationalStance.from(archetype: .hiddenAggressor) == .boundaryFriction)
        #expect(EarlyRelationalStance.from(archetype: .indifferentTroll) == .detachedCold)

        #expect(EarlyRelationalDistance.from(topology: .deepAffection) == .closeBond)
        #expect(EarlyRelationalDistance.from(topology: .closeConfidant) == .closeBond)
        #expect(EarlyRelationalDistance.from(topology: .benevolentGuardian) == .politeObserver)
        #expect(EarlyRelationalDistance.from(topology: .acquaintedObserver) == .politeObserver)
        #expect(EarlyRelationalDistance.from(topology: .unconnectedStranger) == .unconnected)

        let texts = ["こんにちは、いつも応援しています。", "ありがとうございます！これからもよろしくお願いします。"]
        let vocab = TextVocabulary(texts: texts)
        let w = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count, dim: 32, latentDim: 128, relationalLatentDim: 128,
            mlpDim: 64, encoderLayers: 1, decoderLayers: 1, serializedVocabulary: vocab.serialize(), seed: 602
        )
        let z = DualStreamRelationalEngine.encodeRelational(
            senderTokenIDs: vocab.encode(texts[0]),
            receiverTokenIDs: vocab.encode(texts[1]),
            weights: w
        )
        let coarse = DualStreamRelationalEngine.classifyEarlyCoarseFromZRel(zRel: z, weights: w.relational)
        let sSum = coarse.stanceProbs.reduce(0, +)
        let dSum = coarse.distanceProbs.reduce(0, +)
        #expect(coarse.stanceProbs.count == 4)
        #expect(coarse.distanceProbs.count == 3)
        #expect(abs(sSum - 1.0) < 1e-4)
        #expect(abs(dSum - 1.0) < 1e-4)
    }

    @Test("6.3: RelationalHybridFeatureExtractor captures rare-token echo while staying invariant to simultaneous topic-noun swaps")
    func hybridExtractorEchoAndTopicSwapInvariance() {
        let trainCorpus = [
            "いつも配信を見てくれてありがとうございます。無理せず休んでくださいね。",
            "今日も楽しかったです！また来週も楽しみにしていますね。",
            "お疲れ様です。体調には気をつけてゆっくり過ごしてください。",
            "陶芸で釉薬を試しています。",
            "園芸で肥料を試しています。"
        ]
        let senderA = "最近は陶芸で釉薬を試しています。楽しくて夢中です！"
        let receiverEchoA = "陶芸で釉薬を試しているんですね！無理せず楽しんでくださいね。"
        let receiverTemplate = "いつもありがとうございます！また来週も楽しみにしていますね。"

        let senderB = "最近は園芸で肥料を試しています。楽しくて夢中です！"
        let receiverEchoB = "園芸で肥料を試しているんですね！無理せず楽しんでくださいね。"

        let vocab = TextVocabulary(texts: trainCorpus + [senderA, receiverEchoA, receiverTemplate, senderB, receiverEchoB])
        var w = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count, dim: 32, latentDim: 128, relationalLatentDim: 128,
            mlpDim: 64, encoderLayers: 1, decoderLayers: 1, serializedVocabulary: vocab.serialize(), seed: 603
        )
        w.relational.alphaRel = vocab.computeRelationalTokenGate(trainTexts: trainCorpus)
        let extractor = RelationalHybridFeatureExtractor(vocabulary: vocab)

        let zDummy = [Float](repeating: 0.1, count: 128)
        let fEchoA = extractor.extract(
            senderTokens: vocab.encode(senderA, addEOS: false),
            receiverTokens: vocab.encode(receiverEchoA, addEOS: false),
            zRel: zDummy, weights: w
        )
        let fTemplate = extractor.extract(
            senderTokens: vocab.encode(senderA, addEOS: false),
            receiverTokens: vocab.encode(receiverTemplate, addEOS: false),
            zRel: zDummy, weights: w
        )
        let fEchoB = extractor.extract(
            senderTokens: vocab.encode(senderB, addEOS: false),
            receiverTokens: vocab.encode(receiverEchoB, addEOS: false),
            zRel: zDummy, weights: w
        )

        #expect(fEchoA.count == 160)
        // Index 128 + 8 = 136 is rareEchoRecall
        #expect(fEchoA[136] > fTemplate[136] + 0.3, "Topic echo recall must separate genuine memory from generic flattery")

        // Simultaneous topic-noun swap ("陶芸/釉薬" -> "園芸/肥料") preserves hybrid echo & rhythm features identically!
        var maxSwapDiff: Float = 0.0
        for j in 128..<160 {
            maxSwapDiff = max(maxSwapDiff, abs(fEchoA[j] - fEchoB[j]))
        }
        #expect(maxSwapDiff < 1e-5, "Hybrid features must be invariant under simultaneous rare-noun swaps: maxDiff=\(maxSwapDiff)")
    }

    @Test("6.4: Pure-CPU multi-core encodeZRelCPUParallel matches single-pair encodeRelational")
    func cpuParallelEncoderParity() {
        let texts = [
            "昨日の配信ありがとうございました！すごく楽しかったです。",
            "こちらこそ見てくれてありがとう！また遊びに来てね。",
            "なんで返信くれないの？ずっと待ってるのに。",
            "へえ、そうなんだ。まあ頑張って。"
        ]
        let vocab = TextVocabulary(texts: texts)
        let w = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count, dim: 32, latentDim: 128, relationalLatentDim: 128,
            mlpDim: 64, encoderLayers: 1, decoderLayers: 1, serializedVocabulary: vocab.serialize(), seed: 604
        )
        let time = 32
        let sToks = texts.map { RelationalPipeline.encodeTokens($0, vocab: vocab, time: time) }
        let rToks = Array(sToks.reversed())
        let batchOut = DualStreamRelationalEngine.encodeZRelCPUParallel(
            senders: sToks, receivers: rToks, time: time, weights: w, chunkSize: 2
        )
        #expect(batchOut.count == 4)
        for i in 0..<4 {
            let single = DualStreamRelationalEngine.encodeRelational(senderTokenIDs: sToks[i], receiverTokenIDs: rToks[i], weights: w)
            var maxDiff: Float = 0.0
            for d in 0..<single.count {
                maxDiff = max(maxDiff, abs(single[d] - batchOut[i][d]))
            }
            #expect(maxDiff < 1e-4, "Mismatch at item \(i): \(maxDiff)")
        }
    }
}
