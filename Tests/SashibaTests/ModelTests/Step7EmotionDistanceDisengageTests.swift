import Foundation
import Testing
@testable import Sashiba

@Suite("Step 7: v5 Emotional Gradation, Relative Temporal Context & Disengagement Trajectories")
struct Step7EmotionDistanceDisengageTests {
    @Test("7.1: Unified v5 JSONL record decodes emotion, observations, temporal_context, and trajectory cleanly")
    func v5JSONLSchemaRoundTrip() throws {
        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("sashiba_step7_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let jsonLine = """
        {"scenario_id":"v5_test_001","split":"unseen_test","category_12":"LIGHT_SUPPORT_V5","archetype_6":"TRUE_SUPPORTER","topology_5":"BENEVOLENT_GUARDIAN","expected_archetype_t2":"TRUE_SUPPORTER","expected_topology_t2":"BENEVOLENT_GUARDIAN","modulation":{"ll_t2":0.82,"pol_t2":0.78,"gain_t2":0.85,"ll_t4":0.84,"pol_t4":0.80,"gain_t4":0.85},"persona_profile":{"age_group":"30代前半","tone":"穏やかな丁寧語","relationship_duration":"2年","context_domain":"地方移住エッセイ連載"},"sender_log_t1_t2":"秋の収穫祭エッセイを来週公開します。準備で返信は遅めになります。","sender_log_t3_t4":"冬の保存食コラムも書き上がりました。無理のない範囲で覗いてもらえたら嬉しいです。","receiver_log_t1_t2":"昨年の柿酢づくりの記事も素敵でしたし、収穫祭エッセイ楽しみにしています。返信はお気になさらず準備を優先してくださいね。","receiver_log_t3_t4":"保存食コラムの完成おめでとうございます。寒い季節ですので指先を冷やさないよう温かくしてお過ごしくださいね。","receiver_log_harmonized":"保存食コラムの完成おめでとうございます。寒い季節ですので指先を冷やさないよう温かくしてお過ごしくださいね。","insights":{"level_0_core":"適切な距離を保ちつつ誠実に応援する読者です。","level_1_mid":"過去記事を記憶しつつ返信負担をかけず節目の活動を見守る安定した支援関係です。","level_2_deep":"踏み込みすぎずイベントごとに温かい具体的な気遣いを届ける非エスカレート型の誠実な応援関係が維持されています。"},"observations_t2":{"o1_support":true,"o2_specific":true,"o3_neg_mix":false,"o4_boundary":"RESPECTED","o5_substantive":"ONE_WAY"},"observations_t4":{"o1_support":true,"o2_specific":true,"o3_neg_mix":false,"o4_boundary":"RESPECTED","o5_substantive":"ONE_WAY"},"emotion_t2":{"affection":2,"friction":0,"arousal":0.28},"emotion_t4":{"affection":2,"friction":0,"arousal":0.26},"temporal_context":{"actor_baseline_reply_hours":6.0,"actor_baseline_init_ratio":0.35,"reply_lag_hours_t2":12.0,"reply_lag_hours_t4":14.0,"gap_days_t2_to_t3":28.0,"initiation_ratio_t2":0.20,"initiation_ratio_t4":0.20,"event_driven":true,"post_t4_unanswered_probe":false},"trajectory":{"type":"STEADY_LIGHT_SUPPORT","terminal_utterance_kind":"NONE","retained_30d":true,"retained_90d":true}}
        """
        try jsonLine.write(to: tmpDir.appendingPathComponent("scenarios.jsonl"), atomically: true, encoding: .utf8)
        let cfg = DatasetConfiguration.load(from: tmpDir.path)
        #expect(cfg.relationalDialogueCases.count == 1)
        let rc = cfg.relationalDialogueCases[0]
        #expect(rc.isZeroShotUnseen == true)
        #expect(rc.emotionT2 == EmotionalGradation(affection: 2, friction: 0, arousal: 0.28))
        #expect(rc.emotionT4 == EmotionalGradation(affection: 2, friction: 0, arousal: 0.26))
        #expect(rc.observationsT4.o2Specific == true)
        #expect(rc.temporalContext.eventDriven == true)
        #expect(abs(rc.temporalContext.gapDaysT2ToT3 - 28.0) < 1e-4)
        #expect(rc.trajectory.type == .steadyLightSupport)
        #expect(rc.trajectory.retained90d == true)
    }

    @Test("7.2: Relative-baseline temporal context separates naturally slow replier from selective ghosting/fadeout")
    func relativeTemporalContextFeatures() {
        // Person A replies in 48h to everyone AND 48h to this partner (healthy slow-paced bond after 21d gap)
        let slowBaseline = TemporalRelationalContext(
            actorBaselineReplyHours: 48.0, actorBaselineInitRatio: 0.45,
            replyLagHoursT2: 46.0, replyLagHoursT4: 48.0, gapDaysT2ToT3: 21.0,
            initiationRatioT2: 0.45, initiationRatioT4: 0.45,
            eventDriven: false, postT4UnansweredProbe: false
        )
        // Person B replies in 1.0h to others, but 48h to this partner + unanswered probe
        let selectiveStop = TemporalRelationalContext(
            actorBaselineReplyHours: 1.0, actorBaselineInitRatio: 0.55,
            replyLagHoursT2: 2.0, replyLagHoursT4: 48.0, gapDaysT2ToT3: 21.0,
            initiationRatioT2: 0.50, initiationRatioT4: 0.05,
            eventDriven: false, postT4UnansweredProbe: true
        )
        let fSlow = RelationalHybridFeatureExtractor.extractTemporalContextFeatures(slowBaseline)
        let fStop = RelationalHybridFeatureExtractor.extractTemporalContextFeatures(selectiveStop)
        #expect(fSlow.count == 14)
        #expect(fStop.count == 14)
        // Index 3 is relative T4 lag log((1+lag4)/(1+baseline)): ~0 for slowBaseline vs >3.0 for selectiveStop!
        #expect(abs(fSlow[3]) < 0.05)
        #expect(fStop[3] > 3.0)
        // Index 12 is postT4UnansweredProbe, Index 13 is gapResumedFlag
        #expect(fSlow[12] == 0.0 && fSlow[13] == 1.0)
        #expect(fStop[12] == 1.0 && fStop[13] == 0.0)
    }

    @Test("7.3: Smooth intra-cell emotional prototype modulation preserves cell identity while ordering friction & arousal")
    func intraCellEmotionalPrototypeModulation() {
        let zCalmWarm = DualStreamRelationalEngine.targetPrototype(
            archetype: .trueSupporter, topology: .closeConfidant, latentDim: 128,
            emotion: EmotionalGradation(affection: 3, friction: 0, arousal: 0.20)
        )
        let zHighEnergyConflict = DualStreamRelationalEngine.targetPrototype(
            archetype: .trueSupporter, topology: .closeConfidant, latentDim: 128,
            emotion: EmotionalGradation(affection: 2, friction: 2, arousal: 0.85)
        )
        let gCalm = DualStreamRelationalEngine.computePentagonFromZRel(zRel: zCalmWarm)
        let gConf = DualStreamRelationalEngine.computePentagonFromZRel(zRel: zHighEnergyConflict)
        #expect(gCalm.sHH > gConf.sHH + 0.04)
        #expect(gConf.rHH > gCalm.rHH + 0.04)
    }

    @Test("7.4: §3.1 counterfactualMoveLastSentenceToMiddle preserves character multiset while moving terminal sentence")
    func counterfactualTerminalSentenceMove() {
        let log = "先日の展示会とても素敵でした。案内状も大切に飾っています。今回の新作も色使いが綺麗ですね。ただ、価値観が違いすぎるので連絡はこれで最後にします。"
        let moved = DualStreamRelationalEngine.counterfactualMoveLastSentenceToMiddle(log)
        #expect(moved != log)
        #expect(moved.sorted() == log.sorted())
        #expect(moved.hasPrefix("先日の展示会とても素敵でした。ただ、価値観が違いすぎるので連絡はこれで最後にします。"))
        #expect(moved.hasSuffix("今回の新作も色使いが綺麗ですね。"))

        let emojiLog = "昨日のソワレ最高だった😭✨ 差し入れも喜んでくれて嬉しかったよ🫶 新曲も楽しみにしてるね🎶 でも価値観が合わないから連絡はこれで最後にするね🙇"
        let emojiMoved = DualStreamRelationalEngine.counterfactualMoveLastSentenceToMiddle(emojiLog)
        #expect(emojiMoved.sorted() == emojiLog.sorted())
        #expect(emojiMoved.hasPrefix("昨日のソワレ最高だった😭✨ でも価値観が合わないから連絡はこれで最後にするね🙇"))
    }

    @Test("7.5: Emoji base-scalar fallback, expressive alphaRel=1.0 gate, and period-less chat preservation")
    func emojiEncodingGateAndPeriodlessChat() throws {
        let baseVocab = TextVocabulary(serializedCharacters: "あいうえお配信おつ歌枠たすかる❤❗🙇😭✨😆💦！？ ")
        // Variation-selector & ZWJ emojis ("❤️", "❗️", "🙇‍♂️") resolve to base scalar without <unk>
        let encoded = baseVocab.encode("配信おつ❤️❗️🙇‍♂️😭✨")
        #expect(!encoded.contains(TextVocabulary.unkId))

        let gate = baseVocab.computeRelationalTokenGate(trainTexts: ["あいうえお"])
        let cryID = baseVocab.encode("😭")[0]
        #expect(gate[cryID] == 1.0)

        let tmpDir = FileManager.default.temporaryDirectory.appendingPathComponent("sashiba_emoji_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }
        let cfgJSON = #"{"classifierRules":{"sentenceDelimiters":["。"],"clauseDelimiters":["、"]}}"#
        try cfgJSON.write(to: tmpDir.appendingPathComponent("dataset_config.json"), atomically: true, encoding: .utf8)
        let rowJSON = """
        {"scenario_id":"v6_chat_01","split":"unseen_test","category_12":"VTUBER_GENZ","archetype_6":"TRUE_SUPPORTER","topology_5":"CLOSE_CONFIDANT","expected_archetype_t2":"TRUE_SUPPORTER","expected_topology_t2":"CLOSE_CONFIDANT","modulation":{"ll_t2":0.8,"pol_t2":0.8,"gain_t2":1.0,"ll_t4":0.85,"pol_t4":0.85,"gain_t4":1.0},"persona_profile":{"age_group":"10代後半","tone":"Z世代スラング・絵文字","relationship_duration":"1年","context_domain":"VTuber歌枠"},"sender_log_t1_t2":"深夜のゲリラ歌枠来てくれてありがと〜🎤✨","sender_log_t3_t4":"来週の新衣装お披露目も楽しみにしててね🫶","receiver_log_t1_t2":"ゲリラ歌枠ガチたすかる😭✨ 高音メロすぎて鳥肌立ったよ🔥","receiver_log_t3_t4":"新衣装ビジュ絶対えぐいじゃん🥺 喉ケアしてゆっくり寝てね🫶🤍","receiver_log_harmonized":"新衣装ビジュ絶対えぐいじゃん🥺 喉ケアしてゆっくり寝てね🫶🤍","insights":{"level_0_core":"スラングと絵文字を使いつつ体調を気遣う10代ファン","level_1_mid":"Z世代特有の表現と絵文字で熱量高く応援しつつ睡眠を促す健全な支援者","level_2_deep":"句点を使わない現代チャット文体ながら配信内容を具体的に受け止め境界を尊重している"}}
        """
        try rowJSON.write(to: tmpDir.appendingPathComponent("scenarios.jsonl"), atomically: true, encoding: .utf8)
        let loaded = DatasetConfiguration.load(from: tmpDir.path).relationalDialogueCases
        #expect(loaded.count == 1)
        // Must NOT inject fake "。" in the middle of period-less emoji chat!
        #expect(!loaded[0].partnerCurrentText.contains("。"))
    }
}
