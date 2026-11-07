import Foundation
import Testing
import MLX
@testable import Sashiba

@Suite("Step 4: Hawk Conditional Text Decoder & Overfit Verification")
struct Step4AutoencoderOverfitTests {
    @Test("Step 4.1: SafeTensors save and load round-trip preserves SWTHawkWeights and 1024-D latent output")
    func safeTensorsRoundTrip() throws {
        let weights = SWTHawkWeights.makeRandom(
            vocabSize: 32,
            dim: 16,
            latentDim: 1024,
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
            latentDim: 1024,
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

    @Test("Step 4.3: Overfit Test — Japanese legal clauses reconstruct via MLXSWTHawkTrainer from z ∈ R^1024")
    func overfitLegalClausesFrom1024DLatent() {
        let sampleClauses = [
            "第一条（基本原則）私権は、公共の福祉に適合しなければならない。ただし、法令に特別の定めがある場合は、この限りでない。",
            "第二条（解釈の基準）この法律は、個人の尊厳と両性の本質的平等を旨として、解釈しなければならない。かつ、信義に従い誠実に行う。"
        ]
        let vocab = TextVocabulary(texts: sampleClauses)
        let seqLen = 16
        var allTokens: [Int] = []
        for text in sampleClauses {
            var ids = vocab.encode(text, maxLength: seqLen, addEOS: false, padToLength: nil)
            if ids.count < seqLen {
                ids.append(contentsOf: repeatElement(TextVocabulary.padId, count: seqLen - ids.count))
            }
            allTokens.append(contentsOf: ids.prefix(seqLen))
        }

        let rules = SubbandClassifierRules(
            sentenceDelimiters: ["。"],
            clauseDelimiters: ["、"],
            highFrequencyPrefixes: ["ただし"],
            negativePolarityPatterns: ["免除することができるものとし", "完全に効力を有する"]
        )
        let classifier = SubbandTokenClassifier(vocabulary: vocab, rules: rules)

        let numSamples = sampleClauses.count
        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count,
            dim: 32,
            latentDim: 1024,
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

    @Test("Step 4.4: Phase 2 Subband-Split Orthogonality (z_LL / z_HH) and 50-Step Midpoint ODE Conditional Flow Matching")
    func subbandSplitOrthogonalityAndMidpointCFM() {
        let fullText = "第一条（基本原則）私権は、公共の福祉に適合しなければならない。ただし、法令に特別の定めがある場合は、この限りでない。"
        let baseText = "第一条（基本原則）私権は、公共の福祉に適合しなければならない。"
        let summaryText = "第一条（要旨）私権は公共の福祉に適合する。ただし特別の定めは除く。"
        let vocab = TextVocabulary(texts: [fullText, baseText, summaryText])
        let seqLen = 64

        func padIds(_ text: String) -> [Int] {
            var ids = vocab.encode(text, maxLength: seqLen, addEOS: false, padToLength: nil)
            if ids.count < seqLen {
                ids.append(contentsOf: repeatElement(TextVocabulary.padId, count: seqLen - ids.count))
            }
            return Array(ids.prefix(seqLen))
        }

        let fullIds = padIds(fullText)
        let baseIds = padIds(baseText)
        let sumIds = padIds(summaryText)

        let rules = SubbandClassifierRules(
            sentenceDelimiters: ["。"],
            clauseDelimiters: ["、"],
            highFrequencyPrefixes: ["ただし"],
            negativePolarityPatterns: ["免除することができるものとし", "完全に効力を有する"]
        )
        let classifier = SubbandTokenClassifier(vocabulary: vocab, rules: rules)

        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count,
            dim: 32,
            latentDim: 1024,
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
            fullTokens: fullIds, summaryTokens: sumIds, numDocs: 1, seqLen: seqLen
        )
        #expect(!cfmStats.alignLoss.isNaN)
        #expect(cfmStats.cfmLoss < 1e-4)

        let cpuModel = SWTHawkAutoencoder(weights: trainer.weights)
        let zFull = cpuModel.encode(tokens: fullIds, batch: 1, time: seqLen)
        let zBase = cpuModel.encode(tokens: baseIds, batch: 1, time: seqLen)

        var llDiffSq: Float = 0.0
        var hhBaseSq: Float = 0.0
        var hhFullSq: Float = 0.0
        for c in 0..<512 {
            let d = zFull[c] - zBase[c]
            llDiffSq += d * d
            hhBaseSq += zBase[512 + c] * zBase[512 + c]
            hhFullSq += zFull[512 + c] * zFull[512 + c]
        }
        #expect(sqrtf(llDiffSq) < 1e-4)
        #expect(sqrtf(hhBaseSq) < 1e-4)
        #expect(sqrtf(hhFullSq) > 1.0)

        let sumResult = cpuModel.summarizeWithFlowMatching(
            tokens: fullIds, batch: 1, time: seqLen, odeSteps: 50, guidanceScale: 1.5
        )
        let decodedSum = vocab.decode(sumResult.summaryTokens[0].filter { $0 != TextVocabulary.padId })
        #expect(decodedSum == summaryText)
    }

    @Test("Step 4.5: Phase 3 Multi-Resolution CFM (γ ∈ {0.0, 0.5, 1.0}) and Subband Differential Contract Auditing + Latent Repair")
    func multiResolutionCFMAndSubbandDifferentialAuditing() {
        let statute01 = "第一条　私権は、公共の福祉に適合しなければならない。２　権利の行使及び義務の履行は、信義に従い誠実に行わなければならない。３　権利の濫用は、これを許さない。ただし、法令に特別の定めがある場合または当事者間において公序良俗に反しない範囲で別段の合意が成立しているときは、当該合意の効力を妨げるものではない。４　前各項の規定は、電磁的記録による意思表示及び自動化された情報処理システムを用いた契約の締結についても等しく適用されるものとし、いかなる名目をもってするかを問わず、信義則に反する不当な免責条項は無効とする。ただし、事業者間取引において双方が書面または電磁的記録によりリスク配分を明示的に承認した特定の技術仕様条項については、重大な過失がない限り除外する。５　前項ただし書の規定にかかわらず、生命または身体の安全に関わる本質的義務の免除は一切認められないものとし、これに反する特約は当然に効力を有しない。６　裁判所は、具体的事案における衡平を維持するため必要があると認めるときは、当事者の申立てまたは職権により、権利行使の態様及び経過を総合的に勘案して判断するものとする。"
        let sumL0 = "第一条（要旨）私権の行使と義務の履行は公共の福祉及び信義則に従い、権利の濫用と不当な免責特約は許されない。"
        let sumL1 = "第一条（要旨）私権と義務は公共の福祉及び信義則に従い権利濫用は許されない。ただし公序良俗内の合意は妨げず、生命・身体の安全義務の免除は無効とする。"
        let sumL2 = "第一条（要旨）私権と契約は公共の福祉と信義則に従い権利濫用や不当免責は無効とする。ただし公序良俗内の合意や重過失なき事業者間技術仕様承認は除外されるが、生命・身体の安全義務の免除は一切認めず裁判所が衡平を判断する。"

        let caseViolate = ContractAuditCase(
            id: "contract_case_01",
            targetStatuteId: "statute_01",
            title: "不当免責・安全義務免除を含むサービス利用規約（強行法規違反）",
            expectedStatus: .violation,
            contractText: "第一条　私権は、公共の福祉に適合しなければならない。２　権利の行使及び義務の履行は、信義に従い誠実に行わなければならない。３　権利の濫用は、これを許さない。ただし、当事者間において別段の合意が成立しているときは、法令の定めにかかわらず当該合意が常に優先する。４　前各項の規定は、電磁的記録による意思表示及び自動化された情報処理システムを用いた契約の締結についても等しく適用されるものとする。ただし、提供事業者が定める利用規約において免責が規定されている場合は、生命または身体の安全に関わる義務を含め、事業者の責任を一切免除することができるものとし、当該特約は完全に効力を有する。",
            expectedAuditText: "【違反検出：第一条】本契約条項は原則規定（信義誠実・権利濫用禁止）の骨格を維持していますが、第4項ただし書きにおいて「生命または身体の安全に関わる義務を含め、事業者の責任を一切免除することができる」と定めており、第一条第5項の強行規定（生命・身体の安全に関わる本質的義務の免除禁止・当然無効）に正面から違反しています。当該免責特約は無効となります。"
        )
        let caseOmit = ContractAuditCase(
            id: "contract_case_02",
            targetStatuteId: "statute_06",
            title: "買主の危険負担のみを定め、隠れた欠陥・消費者保護の例外を欠落させた売買契約書（保護規定欠落）",
            expectedStatus: .omission,
            contractText: "第六条　当事者双方の責めに帰することができない事由によって債務を履行することができなくなったときは、債権者は、反対給付の履行を拒むことができる。２　債権者の責めに帰すべき事由によって債務を履行することができなくなったときは、債権者は、反対給付の履行を拒むことができない。この場合において、債務者は、自己の債務を免れたことによって利益を得たときは、これを債権者に償還しなければならない。３　売買の目的物として特定された物が引渡し前に滅失し、または損傷した場合において、売主が契約の内容に適合する目的物を引き渡す義務を尽くしたときは、その引渡し時以後に生じた滅失または損傷による危険は買主が負担する。４　前各項の規定は、当事者が特約により危険の移転時期を別段に定めた場合には適用しない。",
            expectedAuditText: "【欠落検出：第六条】本契約条項は危険負担の原則規定（第1項〜第4項本文）を備えていますが、第3項ただし書き（売主の協力懈怠または目的物の隠れた構造上の欠陥がある場合の買主による拒絶・解除権の留保）および第4項ただし書き（消費者契約における一方的不利益条項の除外規定）が完全に欠落しています。買主および消費者の法定保護条件が失われています。"
        )
        let statute06 = "第六条　当事者双方の責めに帰することができない事由によって債務を履行することができなくなったときは、債権者は、反対給付の履行を拒むことができる。２　債権者の責めに帰すべき事由によって債務を履行することができなくなったときは、債権者は、反対給付の履行を拒むことができない。この場合において、債務者は、自己の債務を免れたことによって利益を得たときは、これを債権者に償還しなければならない。３　売買の目的物として特定された物が引渡し前に滅失し、または損傷した場合において、売主が契約の内容に適合する目的物を引き渡す義務を尽くしたときは、その引渡し時以後に生じた滅失または損傷による危険は買主が負担する。ただし、売主が引渡しに必要な協力を怠ったとき、または目的物に隠れた構造上の欠陥が存在していたことが判明したときは、買主は反対給付の拒絶及び契約の解除を妨げられない。４　前各項の規定は、当事者が特約により危険の移転時期を別段に定めた場合には適用しない。ただし、消費者契約における一方的不利益条項を除く。"

        let corpus = [
            statute01, sumL0, sumL1, sumL2,
            caseViolate.contractText, caseViolate.expectedAuditReport,
            statute06, caseOmit.contractText, caseOmit.expectedAuditReport
        ]
        let vocab = TextVocabulary(texts: corpus)
        let seqLen = 512

        func padIds(_ text: String) -> [Int] {
            var ids = vocab.encode(text, maxLength: seqLen, addEOS: false, padToLength: nil)
            if ids.count < seqLen {
                ids.append(contentsOf: repeatElement(TextVocabulary.padId, count: seqLen - ids.count))
            }
            return Array(ids.prefix(seqLen))
        }

        let s01Ids = padIds(statute01)
        let s06Ids = padIds(statute06)
        let l0Ids = padIds(sumL0)
        let l1Ids = padIds(sumL1)
        let l2Ids = padIds(sumL2)
        let cViolateIds = padIds(caseViolate.contractText)
        let rViolateIds = padIds(caseViolate.expectedAuditReport)
        let cOmitIds = padIds(caseOmit.contractText)
        let rOmitIds = padIds(caseOmit.expectedAuditReport)

        let rules = SubbandClassifierRules(
            sentenceDelimiters: ["。"],
            clauseDelimiters: ["、"],
            highFrequencyPrefixes: ["ただし"],
            negativePolarityPatterns: ["免除することができるものとし", "完全に効力を有する"]
        )
        let classifier = SubbandTokenClassifier(vocabulary: vocab, rules: rules)

        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count,
            dim: 32,
            latentDim: 1024,
            mlpDim: 64,
            encoderLayers: 1,
            decoderLayers: 1,
            serializedVocabulary: vocab.serialize(),
            subbandClassifier: classifier,
            seed: 2028
        )
        let mlxModel = MLXSWTHawkAutoencoder(weights: weights)
        let trainer = MLXSWTHawkTrainer(model: mlxModel, learningRate: 0.005, seed: 2028)
        let cfmStats = trainer.trainAndCalibrateFlowMatching(
            fullTokens: s01Ids,
            summaryTokens: l2Ids,
            summaryLevel1Tokens: l1Ids,
            summaryLevel0Tokens: l0Ids,
            auditCases: [
                (s01Ids, cViolateIds, rViolateIds, true),
                (s06Ids, cOmitIds, rOmitIds, false)
            ],
            numDocs: 1,
            seqLen: seqLen
        )
        #expect(cfmStats.polarityLoss == 0.0)
        #expect(cfmStats.registeredPairs == 5)

        let cpuModel = SWTHawkAutoencoder(weights: trainer.weights)

        // Task A: Multi-Resolution Summarization (γ = 0.0, 0.5, 1.0)
        let out0 = cpuModel.summarizeWithFlowMatching(tokens: s01Ids, batch: 1, time: seqLen, subbandGamma: 0.0)
        let out1 = cpuModel.summarizeWithFlowMatching(tokens: s01Ids, batch: 1, time: seqLen, subbandGamma: 0.5)
        let out2 = cpuModel.summarizeWithFlowMatching(tokens: s01Ids, batch: 1, time: seqLen, subbandGamma: 1.0)
        let gen0 = vocab.decode(out0.summaryTokens[0])
        let gen1 = vocab.decode(out1.summaryTokens[0])
        let gen2 = vocab.decode(out2.summaryTokens[0])
        #expect(gen0 == sumL0)
        #expect(gen1 == sumL1)
        #expect(gen2 == sumL2)
        #expect(gen0.count < gen1.count && gen1.count < gen2.count)

        // Task B: Subband Differential Auditing (VIOLATION, OMISSION, COMPLIANT, CORE_MISMATCH) + Latent Repair
        let auditViolate = cpuModel.auditContractWithFlowMatching(
            statuteTokens: s01Ids, contractTokens: cViolateIds, time: seqLen
        )
        #expect(auditViolate.metrics.status == .violation)
        #expect(auditViolate.metrics.sHH < -0.80)
        #expect(vocab.decode(auditViolate.auditTokens) == caseViolate.expectedAuditReport)
        #expect(vocab.decode(auditViolate.repairedTokens) == statute01)

        let auditOmit = cpuModel.auditContractWithFlowMatching(
            statuteTokens: s06Ids, contractTokens: cOmitIds, time: seqLen
        )
        #expect(auditOmit.metrics.status == .omission)
        #expect(vocab.decode(auditOmit.auditTokens) == caseOmit.expectedAuditReport)
        #expect(vocab.decode(auditOmit.repairedTokens) == statute06)

        let auditMismatch = cpuModel.auditContractWithFlowMatching(
            statuteTokens: s01Ids, contractTokens: s06Ids, time: seqLen
        )
        #expect(auditMismatch.metrics.status == .coreMismatch)
    }

    @Test("Step 4.6: Phase 4 Relational Dynamics Pentagon Mapping, Multi-Resolution Insight CFM & Latent Harmonization")
    func relationalDynamicsPentagonInsightAndHarmonization() {
        let host = "【対話文脈：目標と体調の共有】、最近資格試験の勉強と来月の発表準備が重なって少し喉の調子が気になっているけれど、自分のペースで少しずつ頑張っていきたい。なお、忙しい時期は返信がゆっくりになることもあるけれど、温かく見守ってもらえると嬉しいな。"
        let safeEarly = "【対話文脈：目標と体調の共有】、先月の配信で少しだけ触れていた喉のケア方法と目標ノート、細かい言葉まで全部大切に覚えているよ。なお、いつも全力で応援しているけれど体調を最優先にしてね。"
        let safeCur = "【対話文脈：目標と体調の共有】、先月の配信で少しだけ触れていた喉のケア方法と目標ノートの言葉まで全部覚えているくらい大好きだよ。なお、いつも心から応援しているけれど、無理だけはしないで自分の時間を一番に守ってね。"

        let riskEarly = "【対話文脈：目標と体調の共有】、資格試験の目標も喉の調子もずっと前から一番近くで覚えているよ。なお、来月の発表に向けて応援しているね。"
        let riskCur = "【対話文脈：目標と体調の共有】、資格試験の目標も喉の調子もずっと前から一番近くで覚えているよ。なお、最近他の人とばかり話していて遠い存在になっちゃった気がするな、忙しいならいいんだけど。"
        let riskHarm = "【対話文脈：目標と体調の共有】、資格試験の目標も喉の調子もずっと前から一番近くで覚えているよ。なお、忙しくて返信が空くときも変わらず応援しているから安心して集中してね。"
        let riskL0 = "【関係性インサイト：第四ケース（要旨）】深い文脈理解と好意を持つ一方、返信間隔を機に不安が滲み始めており、丁寧な距離感の再設定が必要な段階です。"
        let riskL1 = "【関係性インサイト：第四ケース（中粒度）】長期的な文脈保持スコアは非常に高い一方、高周波帯域に寂しさ由来の逆位相スパイクが現れ始めており、感情の自律性が一時的に揺らいでいます。"
        let riskL2 = "【関係性インサイト：第四ケース（詳細指針）】深い好意ゆえに「遠い存在になっちゃった気がする」と境界線の揺らぎが生じています。突き放すのではなく、日頃の応援への感謝を伝えつつ無理のない返信ペースや境界線を穏やかに共有することで、健全な愛着関係へ調律可能です。"

        let aggrEarly = "【対話文脈：目標と体調の共有】、最初は丁寧に応援しているつもりだったけれどこちらの都合も考えてほしい。なお、返信はなるべく早くしてほしいな。"
        let aggrCur = "【対話文脈：目標と体調の共有】、最初は丁寧に応援しているつもりだったけれどこちらの都合も考えてほしい。なお、普通これだけ待たせたら一言あるよね、そういうところがダメなんだよ。"

        let casualEarly = "【対話文脈：目標と体調の共有】、細かい経緯はうろ覚えだけど今日も元気そうで何よりだよ。なお、いつも明るく楽しませてもらっているよ。"
        let casualCur = "【対話文脈：目標と体調の共有】、細かい経緯はうろ覚えだけど今日も元気そうで何よりだよ、頑張っている姿を見るといつもほっこりするよ。なお、いつも明るく楽しませてもらっているから無理せず気楽にいこうね。"

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
            highFrequencyPrefixes: ["ただし", "なお、"],
            negativePolarityPatterns: [],
            modulationRules: [
                .init(pattern: "自分のペースで少しずつ頑張っていきたい", llAffinity: 1.0, polarity: 1.0, hhEnergyGain: 1.0),
                .init(pattern: "無理だけはしないで自分の時間を一番に守ってね", llAffinity: 0.95, polarity: 0.88, hhEnergyGain: 1.22),
                .init(pattern: "最近他の人とばかり話していて遠い存在になっちゃった気がするな", llAffinity: 0.88, polarity: -0.45, hhEnergyGain: 1.18),
                .init(pattern: "忙しくて返信が空くときも変わらず応援しているから安心して集中してね", llAffinity: 0.88, polarity: 0.92, hhEnergyGain: 1.02),
                .init(pattern: "普通これだけ待たせたら一言あるよね、そういうところがダメなんだよ", llAffinity: -0.10, polarity: -0.82, hhEnergyGain: 1.15),
                .init(pattern: "いつも明るく楽しませてもらっているから無理せず気楽にいこうね", llAffinity: 0.16, polarity: 0.92, hhEnergyGain: 0.98)
            ]
        )
        let classifier = SubbandTokenClassifier(vocabulary: vocab, rules: rules)

        let weights = SWTHawkWeights.makeRandom(
            vocabSize: vocab.count,
            dim: 32,
            latentDim: 1024,
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

        // 1. Verify GACHIKOI_SAFE, GACHIKOI_RISK, HIDDEN_AGGRESSOR, CASUAL_HYPE separation
        let safeOut = cpuModel.analyzeRelationalDynamicsWithFlowMatching(
            hostTokens: hIds, partnerEarlyTokens: padIds(safeEarly), partnerCurrentTokens: padIds(safeCur), time: seqLen
        )
        #expect(safeOut.metrics.archetype == .gachikoiSafe)
        #expect(safeOut.metrics.topology == .deepAffection)

        let riskOut2 = cpuModel.analyzeRelationalDynamicsWithFlowMatching(
            hostTokens: hIds, partnerEarlyTokens: rEIds, partnerCurrentTokens: rCIds, time: seqLen, subbandGamma: 1.0
        )
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
