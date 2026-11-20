import Foundation
import MLX
import Testing
@testable import Sashiba

@Suite("Step 9: Small Conditional Griffin Psychological State LM (Subword BPE, CPU/GPU Parity & KV-Cache)")
struct Step9PsychologicalStateLMTests {
    @Test("9.1: SubwordVocabulary BPE round-trips Japanese psychological prose with 0 <unk> and ~2.5x compression")
    func subwordBPERoundTripAndCompression() {
        let corpus = Array(repeating: "表面上は明るく合わせていますが、内心では少しずつ距離を置きたい心境です。", count: 8) +
                     Array(repeating: "相手の体調を気遣っており、無理のない距離感で温かく見守りたい心理状態です。", count: 8)
        let bpe = SubwordVocabulary.train(trainingTexts: corpus, baseCharacters: "未知漢字薔薇のように静かに", numMerges: 64, minPairFreq: 2)
        let sample = "表面上は明るく合わせていますが、薔薇のように静かに距離を置きたい心境です。"
        let ids = bpe.encode(sample, addEOS: true)
        #expect(!ids.contains(TextVocabulary.unkId))
        #expect(ids.count < sample.count)
        #expect(bpe.decode(ids) == sample)
    }

    @Test("9.2: PsychologicalStateLMWeights SafeTensors round-trip is bit-exact")
    func safeTensorsRoundTrip() throws {
        let bpe = SubwordVocabulary.train(trainingTexts: ["心の距離と安心感があふれています。"], numMerges: 16, minPairFreq: 1)
        let w = PsychologicalStateLMWeights.makeRandom(
            vocabSize: bpe.count,
            dim: 64,
            numLayers: 2,
            numHeads: 2,
            windowSize: 16,
            numCondTokens: 4,
            condRawDim: PsychConditionExtractor.rawConditionDim,
            maxSeqLen: 48,
            serializedSubwordVocabulary: bpe.serialize(),
            seed: 9001
        )
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("psych_lm_test_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try w.save(to: url)
        let loaded = try PsychologicalStateLMWeights.load(from: url)
        #expect(loaded == w)
    }

    @Test("9.3: CPU forwardTeacherForced matches MLX GPU forward logits to < 2e-3 (TF32) and KV-cached generateTokens matches argmax")
    func cpuMLXParityAndKVCacheDecode() {
        let bpe = SubwordVocabulary.train(trainingTexts: ["穏やかな信頼と親しみがあふれており、心地よい距離感で見守りたい心理状態です。"], numMerges: 8, minPairFreq: 1)
        let w = PsychologicalStateLMWeights.makeRandom(
            vocabSize: bpe.count,
            dim: 64,
            numLayers: 2,
            numHeads: 2,
            windowSize: 12,
            numCondTokens: 4,
            condRawDim: PsychConditionExtractor.rawConditionDim,
            maxSeqLen: 48,
            serializedSubwordVocabulary: bpe.serialize(),
            seed: 9002
        )
        let cpuLM = PsychologicalStateLM(weights: w)
        let mlxLM = MLXPsychologicalStateLM(weights: w)

        var rng = SplitMix64(seed: 1234)
        var cond = [Float](repeating: 0.0, count: PsychConditionExtractor.rawConditionDim)
        for i in 0..<cond.count { cond[i] = rng.nextSignedUniform(scale: 1.0) }

        let genIds = cpuLM.generateTokens(condRaw: cond, maxLength: 24, minCharLength: 200, repetitionPenalty: 1.0, temperature: 0.0)
        #expect(genIds.count == 24)

        var prevTokens = [Int](repeating: TextVocabulary.bosId, count: 24)
        for t in 1..<24 { prevTokens[t] = genIds[t - 1] }

        let cpuLogits = cpuLM.forwardTeacherForced(condRaw: cond, prevTokens: prevTokens, batch: 1, time: 24)
        let pe = MLXArray(HawkConditionalDecoder.positionalEncoding(time: 24, dim: 64), [1, 24, 64])
        let mlxLogits = mlxLM(
            condRaw: MLXArray(cond, [1, PsychConditionExtractor.rawConditionDim]),
            prevTokenIds: MLXArray(prevTokens.map { Int32($0) }, [1, 24]),
            pe: pe
        )
        eval(mlxLogits)
        let gpuArr = mlxLogits.asType(.float32).asArray(Float.self)
        #expect(cpuLogits.count == gpuArr.count)

        var maxDiff: Float = 0.0
        for (a, b) in zip(cpuLogits, gpuArr) {
            maxDiff = max(maxDiff, abs(a - b))
        }
        #expect(maxDiff < 2e-3)

        let vSize = bpe.count
        for t in 0..<24 {
            let base = t * vSize
            var best = TextVocabulary.specialCount
            for v in (TextVocabulary.specialCount + 1)..<vSize where cpuLogits[base + v] > cpuLogits[base + best] {
                best = v
            }
            #expect(best == genIds[t])
        }
    }

    @Test("9.4: TimestampedComment parser and sender-free 128D PsychConditionExtractor distinguish single vs steady vs decelerating streams")
    func timestampedCommentStreamModes() {
        let parsed = TimestampedComment.parseList([
            "2026-12-25 22:33:44 ううう",
            "2026-10-12 12:34:56 あああ",
            "2026-11-23 10:22:33 いいい"
        ])
        #expect(parsed.count == 3)
        #expect(parsed[0].text == "あああ" && parsed[0].rawTimestamp == "2026-10-12 12:34:56")
        #expect(parsed[1].text == "いいい" && parsed[1].rawTimestamp == "2026-11-23 10:22:33")
        #expect(parsed[2].text == "ううう" && parsed[2].rawTimestamp == "2026-12-25 22:33:44")

        let plainYear = TimestampedComment.parse("2026年もよろしくね！")
        #expect(plainYear.timestamp == nil && plainYear.text == "2026年もよろしくね！")

        let relW = SWTHawkWeights.makeRandom(vocabSize: 128, dim: 64, latentDim: 128, seed: 333)
        let relModel = SWTHawkAutoencoder(weights: relW)

        let single = PsychConditionExtractor.extractFromComments(
            relationalModel: relModel,
            comments: TimestampedComment.parseList(["2026-12-25 22:33:44 ううう"])
        ).rawCondition
        let steady = PsychConditionExtractor.extractFromComments(
            relationalModel: relModel,
            comments: parsed
        ).rawCondition
        let decel = PsychConditionExtractor.extractFromComments(
            relationalModel: relModel,
            comments: TimestampedComment.parseList([
                "2026-10-12 12:00:00 あああ",
                "2026-10-12 14:00:00 いいい",
                "2026-12-25 22:33:44 ううう"
            ])
        ).rawCondition
        let burst = PsychConditionExtractor.extractFromComments(
            relationalModel: relModel,
            comments: TimestampedComment.parseList([
                "2026-10-12 12:00:00 あああ",
                "2026-11-23 10:20:00 いいい",
                "2026-11-23 10:25:00 ううう"
            ])
        ).rawCondition

        #expect(single.count == PsychConditionExtractor.rawConditionDim)
        let hasPastIdx = 40 + 21 + 16
        let tempBaseIdx = hasPastIdx + 1 + 21 + 13
        #expect(single[hasPastIdx] == 0.0 && single[tempBaseIdx] == 0.0)
        #expect(steady[hasPastIdx] == 1.0 && steady[tempBaseIdx] == 1.0 && steady[tempBaseIdx + 13] == 1.0)
        #expect(decel[tempBaseIdx + 10] > 0.85) // isDecelerating
        #expect(burst[tempBaseIdx + 12] > 0.80) // isRapidBurst
    }

    @Test("9.5: Weight-tied MLX export matches CPU logits and AVX2-batched generateBeam avoids immediate self-repeats")
    func weightTiedExportAndBeamSearch() {
        let bpe = SubwordVocabulary.train(
            trainingTexts: ["穏やかな信頼と親しみがあふれており、心地よい距離感で見守りたい心理状態です。"],
            numMerges: 12,
            minPairFreq: 1
        )
        let initW = PsychologicalStateLMWeights.makeRandom(
            vocabSize: bpe.count,
            dim: 64,
            numLayers: 2,
            numHeads: 2,
            windowSize: 16,
            numCondTokens: 4,
            condRawDim: PsychConditionExtractor.rawConditionDim,
            maxSeqLen: 32,
            serializedSubwordVocabulary: bpe.serialize(),
            seed: 9005
        )
        let mlxTied = MLXPsychologicalStateLM(weights: initW, useTiedEmbeddings: true)
        let exported = mlxTied.exportWeights()
        let cpuTied = PsychologicalStateLM(weights: exported)

        let cond = [Float](repeating: 0.1, count: PsychConditionExtractor.rawConditionDim)
        let prev = [TextVocabulary.bosId, 4, 5, 6, 7, 8]
        let cpuLogits = cpuTied.forwardTeacherForced(condRaw: cond, prevTokens: prev, batch: 1, time: prev.count)
        let pe = MLXArray(HawkConditionalDecoder.positionalEncoding(time: prev.count, dim: 64), [1, prev.count, 64])
        let gpuLogits = mlxTied(
            condRaw: MLXArray(cond, [1, PsychConditionExtractor.rawConditionDim]),
            prevTokenIds: MLXArray(prev.map { Int32($0) }, [1, prev.count]),
            pe: pe
        )
        eval(gpuLogits)
        let gpuArr = gpuLogits.asType(.float32).asArray(Float.self)
        var maxDiff: Float = 0.0
        for (a, b) in zip(cpuLogits, gpuArr) { maxDiff = max(maxDiff, abs(a - b)) }
        #expect(maxDiff < 2e-3)

        let beamToks = cpuTied.generateBeam(condRaw: cond, beamWidth: 4, maxLength: 20, minCharLength: 10)
        #expect(!beamToks.isEmpty)
        for i in 1..<beamToks.count {
            #expect(beamToks[i] != beamToks[i - 1])
        }
    }
}
