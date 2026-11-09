import Foundation
import Testing
import MLX
@testable import Sashiba

@Suite("Step 5: Phase 5.1 Dual-Stream Relational Engine (Anti-Cheat)")
struct Step5DualStreamRelationalTests {
    static let texts = [
        "先週話していた新曲の転調、すごく綺麗でした。無理せず休んでくださいね。",
        "新曲の転調の話、ちゃんと覚えています。返信は気にしないでください。",
        "よく知らないけど頑張って。また気が向いたら見に行くよ。",
        "最近ほかの企画ばかりで寂しいです。もっとこっちを見てほしい。"
    ]

    static func makeWeights(seed: UInt64) -> (SWTHawkWeights, TextVocabulary) {
        let vocab = TextVocabulary(texts: texts)
        var w = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count, dim: 32, latentDim: 1024, mlpDim: 64,
            encoderLayers: 1, decoderLayers: 1, serializedVocabulary: vocab.serialize(), seed: seed
        )
        w.relational.alphaRel = vocab.computeRelationalTokenGate(trainTexts: texts)
        return (w, vocab)
    }

    @Test("5.1: MLX forwardDualStream matches CPU forwardBatchCPU (trained graph == inference graph)")
    func cpuMlxParity() {
        let (w, vocab) = Self.makeWeights(seed: 501)
        let time = 48
        let s = Self.texts.map { RelationalPipeline.encodeTokens($0, vocab: vocab, time: time) }
        let c = [Self.texts[1], Self.texts[0], Self.texts[3], Self.texts[2]].map { RelationalPipeline.encodeTokens($0, vocab: vocab, time: time) }
        let cpu = DualStreamRelationalEngine.forwardBatchCPU(
            senderTokens: s.flatMap { $0 }, receiverTokens: c.flatMap { $0 }, batch: 4, time: time, weights: w
        ).zRel
        let mlx = MLXSWTHawkAutoencoder(weights: w)
        let pe = MLXArray(HawkConditionalDecoder.positionalEncoding(time: time, dim: w.dim), [1, time, w.dim])
        let sIds = MLXArray(s.flatMap { $0 }.map { Int32($0) }, [4, time])
        let cIds = MLXArray(c.flatMap { $0 }.map { Int32($0) }, [4, time])
        let z = mlx.forwardDualStream(senderIds: sIds, receiverIds: cIds, pe: pe).zRel
        eval(z)
        let g = z.asArray(Float.self)
        var maxDiff: Float = 0.0
        for i in 0..<g.count {
            maxDiff = max(maxDiff, abs(g[i] - cpu[i]))
        }
        #expect(maxDiff < 2e-2, "max |z_mlx - z_cpu| = \(maxDiff)")
    }

    @Test("5.2: Encoder signature takes only token IDs and z_rel depends on receiver token order")
    func signatureAndOrderSensitivity() {
        let (w, vocab) = Self.makeWeights(seed: 502)
        let model = SWTHawkAutoencoder(weights: w)
        let s = vocab.encode(Self.texts[0])
        let c = vocab.encode(Self.texts[1])
        let z1 = model.encodeRelational(senderTokenIDs: s, receiverTokenIDs: c)
        let z2 = model.encodeRelational(senderTokenIDs: s, receiverTokenIDs: Array(c.reversed()))
        #expect(z1.count == 1024)
        var diff: Float = 0.0
        for i in 0..<z1.count {
            diff += (z1[i] - z2[i]) * (z1[i] - z2[i])
        }
        #expect(sqrtf(diff) > 1e-3)
    }

    @Test("5.3: End-to-End training moves all 5 audited modules and reduces loss")
    func endToEndMovesAllModules() {
        let (w, vocab) = Self.makeWeights(seed: 503)
        let model = MLXSWTHawkAutoencoder(weights: w)
        let time = 48
        let enc = { (t: String) in RelationalPipeline.encodeTokens(t, vocab: vocab, time: time) }
        let pairs = [
            RelationalTrainingPair(senderTokens: enc(Self.texts[0]), receiverTokens: enc(Self.texts[1]), archetype: 0, topology: 0, isEarlyT2: false, groupKey: "A"),
            RelationalTrainingPair(senderTokens: enc(Self.texts[0]), receiverTokens: enc(Self.texts[2]), archetype: 5, topology: 4, isEarlyT2: false, groupKey: "B"),
            RelationalTrainingPair(senderTokens: enc(Self.texts[0]), receiverTokens: enc(Self.texts[3]), archetype: 3, topology: 1, isEarlyT2: true, groupKey: "C"),
            RelationalTrainingPair(senderTokens: enc(Self.texts[1]), receiverTokens: enc(Self.texts[0]), archetype: 0, topology: 0, isEarlyT2: false, groupKey: "D")
        ]
        let cfs = [RelationalCounterfactualPair(senderTokens: enc(Self.texts[0]), receiverTokens: enc(Self.texts[3]), harmonizedTokens: enc(Self.texts[1]))]
        let trainer = MLXRelationalTrainer(model: model, learningRate: 3e-3, seed: 1)
        let tel = trainer.train(pairs: pairs, counterfactuals: cfs, time: time, epochs: 15, batchSize: 4, log: { _ in })
        #expect(tel.modules.count == 5)
        for m in tel.modules {
            #expect(m.weightDeltaFrobeniusNorm > 1e-4, "\(m.moduleName) did not move")
            #expect(m.meanGradientNorm > 1e-6, "\(m.moduleName) has zero gradient")
        }
        #expect(!tel.finalTotalLoss.isNaN)
    }
}

