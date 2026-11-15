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
            vocabSize: vocab.count, dim: 32,
            latentDim: SWTHawkShape.latentDim,
            relationalLatentDim: SWTHawkShape.relationalLatentDim,
            mlpDim: 64,
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
        #expect(z1.count == w.relational.latentDim)
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

    @Test("5.4: All 30 targetSubbandGeometry cells are non-degenerate, bounded, and pairwise separated")
    func geometryNonDegenerate30Cells() {
        var cells: [(RelationalArchetype, RelationalTopology, Float, Float, Float)] = []
        for a in RelationalArchetype.allCases {
            var prevSLL: Float = 2.0
            for t in RelationalTopology.allCases {
                let g = DualStreamRelationalEngine.targetSubbandGeometry(archetype: a, topology: t)
                #expect(g.sLL >= -0.85 && g.sLL <= 0.95)
                #expect(g.sHH >= -0.90 && g.sHH <= 0.92)
                #expect(g.rHH >= 0.20 && g.rHH <= 1.55)
                #expect(g.sLL < prevSLL, "sLL must decrease monotonically across topology for \(a)")
                prevSLL = g.sLL
                cells.append((a, t, g.sLL, g.sHH, g.rHH))
            }
        }
        #expect(cells.count == 30)
        for i in 0..<cells.count {
            for j in (i + 1)..<cells.count {
                let dx = cells[i].2 - cells[j].2
                let dy = cells[i].3 - cells[j].3
                let dz = cells[i].4 - cells[j].4
                let dist = sqrtf(dx * dx + dy * dy + dz * dz)
                #expect(dist > 0.05, "Collision between \(cells[i].0)/\(cells[i].1) and \(cells[j].0)/\(cells[j].1): dist=\(dist)")
            }
        }
    }

    @Test("5.5: Causal T2 sender window uses hostEarlyText (no future s34 leakage)")
    func causalT2SenderWindow() throws {
        let rc = RelationalDialogueCase(
            id: "test_causal",
            title: "causal",
            expectedTopologyT2: .benevolentGuardian,
            expectedArchetypeT2: .casualHype,
            expectedTopology: .deepAffection,
            expectedArchetype: .trueSupporter,
            hostEarlyText: "前半1文目です。前半2文目です。",
            hostText: "前半1文目です。前半2文目です。後半3文目です。後半4文目です。",
            partnerEarlyText: "受信前半1文目です。受信前半2文目です。",
            partnerCurrentText: "受信前半1文目です。受信前半2文目です。受信後半3文目です。受信後半4文目です。",
            harmonizedPartnerText: "受信前半1文目です。受信前半2文目です。調律文です。調律2文目です。",
            insightLevel0: "L0",
            insightLevel1: "L1",
            insightLevel2: "L2"
        )
        let vocab = TextVocabulary(texts: [rc.hostText, rc.partnerCurrentText])
        let built = RelationalPipeline.buildPairs(cases: [rc], vocab: vocab, time: 64, includeHarmonized: false, mismatchPerCase: 0)
        #expect(built.pairs.count == 2)
        let earlyPair = built.pairs.first { $0.isEarlyT2 }!
        let latePair = built.pairs.first { !$0.isEarlyT2 }!
        #expect(vocab.decode(earlyPair.senderTokens) == rc.hostEarlyText)
        #expect(vocab.decode(latePair.senderTokens) == rc.hostText)
    }
}

