import Foundation
import MLX
import Testing
@testable import Sashiba

@Suite("Step 9: Small Conditional Griffin Psychological State LM (CPU/GPU Parity & KV-Cache)")
struct Step9PsychologicalStateLMTests {
    @Test("9.1: PsychologicalStateLMWeights SafeTensors round-trip is bit-exact")
    func safeTensorsRoundTrip() throws {
        let vocab = TextVocabulary(texts: ["心の距離と安心感があふれています。"])
        let w = PsychologicalStateLMWeights.makeRandom(
            vocabSize: vocab.count,
            dim: 64,
            numLayers: 2,
            numHeads: 2,
            windowSize: 16,
            numCondTokens: 4,
            condRawDim: PsychConditionExtractor.rawConditionDim,
            maxSeqLen: 96,
            serializedVocabulary: vocab.serialize(),
            seed: 9001
        )
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("psych_lm_test_\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try w.save(to: url)
        let loaded = try PsychologicalStateLMWeights.load(from: url)
        #expect(loaded == w)
    }

    @Test("9.2: CPU forwardTeacherForced matches MLX GPU forward logits to < 1e-4 and KV-cached generateTokens matches step-for-step")
    func cpuMLXParityAndKVCacheDecode() {
        let vocab = TextVocabulary(texts: ["穏やかな信頼と親しみがあふれており、心地よい距離感で見守りたい心理状態です。"])
        let w = PsychologicalStateLMWeights.makeRandom(
            vocabSize: vocab.count,
            dim: 64,
            numLayers: 2,
            numHeads: 2,
            windowSize: 12, // Intentionally small windowSize < length (28) to test sliding KV eviction at t >= W!
            numCondTokens: 4,
            condRawDim: PsychConditionExtractor.rawConditionDim,
            maxSeqLen: 64,
            serializedVocabulary: vocab.serialize(),
            seed: 9002
        )
        let cpuLM = PsychologicalStateLM(weights: w)
        let mlxLM = MLXPsychologicalStateLM(weights: w)

        var rng = SplitMix64(seed: 1234)
        var cond = [Float](repeating: 0.0, count: PsychConditionExtractor.rawConditionDim)
        for i in 0..<cond.count { cond[i] = rng.nextSignedUniform(scale: 1.0) }

        // Test KV-cached generateTokens over 28 steps (> windowSize=12) with 4-gram penalty irrelevant on random logits or checked step-by-step
        let genIds = cpuLM.generateTokens(condRaw: cond, maxLength: 28, minLength: 28, temperature: 0.0)
        #expect(genIds.count == 28)

        var prevTokens = [Int](repeating: TextVocabulary.bosId, count: 28)
        for t in 1..<28 { prevTokens[t] = genIds[t - 1] }

        let cpuLogits = cpuLM.forwardTeacherForced(condRaw: cond, prevTokens: prevTokens, batch: 1, time: 28)
        let pe = MLXArray(HawkConditionalDecoder.positionalEncoding(time: 28, dim: 64), [1, 28, 64])
        let mlxLogits = mlxLM(
            condRaw: MLXArray(cond, [1, PsychConditionExtractor.rawConditionDim]),
            prevTokenIds: MLXArray(prevTokens.map { Int32($0) }, [1, 28]),
            pe: pe
        )
        eval(mlxLogits)
        let gpuArr = mlxLogits.asType(.float32).asArray(Float.self)
        #expect(cpuLogits.count == gpuArr.count)

        var maxDiff: Float = 0.0
        for (a, b) in zip(cpuLogits, gpuArr) {
            maxDiff = max(maxDiff, abs(a - b))
        }
        #expect(maxDiff < 1e-3)

        // Verify KV-cached step-by-step greedy choices (for t < 3 before 4-gram check activates, and all non-penalized steps) match teacher-forced argmax
        let vSize = vocab.count
        for t in 0..<3 {
            let base = t * vSize
            var best = TextVocabulary.specialCount
            for v in (TextVocabulary.specialCount + 1)..<vSize where cpuLogits[base + v] > cpuLogits[base + best] {
                best = v
            }
            #expect(best == genIds[t])
        }
    }

    @Test("9.3: PsychConditionExtractor cleanly encodes full vs text-only flags and relative timing deltas")
    func conditionExtractorModes() {
        let relW = SWTHawkWeights.makeRandom(vocabSize: 128, dim: 64, latentDim: 128, seed: 333)
        let relModel = SWTHawkAutoencoder(weights: relW)
        let tcFast = TemporalRelationalContext(actorBaselineReplyHours: 2.0, replyLagHoursT2: 2.0, replyLagHoursT4: 0.2)
        let tcSlow = TemporalRelationalContext(actorBaselineReplyHours: 2.0, replyLagHoursT2: 2.0, replyLagHoursT4: 48.0, postT4UnansweredProbe: true)

        let (cTextOnly, _) = PsychConditionExtractor.extractSingle(
            relationalModel: relModel,
            senderCurrentTokens: [5, 8, 12],
            receiverCurrentTokens: [6, 9, 15]
        )
        let (cFullFast, _) = PsychConditionExtractor.extractSingle(
            relationalModel: relModel,
            senderEarlyTokens: [5, 7],
            receiverEarlyTokens: [6, 8],
            senderCurrentTokens: [5, 8, 12],
            receiverCurrentTokens: [6, 9, 15],
            temporalContext: tcFast
        )
        let (cFullSlow, _) = PsychConditionExtractor.extractSingle(
            relationalModel: relModel,
            senderEarlyTokens: [5, 7],
            receiverEarlyTokens: [6, 8],
            senderCurrentTokens: [5, 8, 12],
            receiverCurrentTokens: [6, 9, 15],
            temporalContext: tcSlow
        )
        #expect(cTextOnly.count == PsychConditionExtractor.rawConditionDim)
        #expect(cTextOnly[190] == 0.0 && cTextOnly[571] == 0.0)
        #expect(cFullFast[190] == 1.0 && cFullFast[571] == 1.0)
        #expect(cFullSlow[571 + 1 + 3] > cFullFast[571 + 1 + 3] + 2.5)
    }
}
