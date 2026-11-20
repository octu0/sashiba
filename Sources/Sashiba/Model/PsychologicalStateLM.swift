import Foundation

/// A single comment/utterance in a chronological single-actor stream (`--text "2026-10-12 12:34:56 あああ"` or `--text "あああ"`).
public struct TimestampedComment: Codable, Sendable, Equatable {
    public let timestamp: Date?
    public let rawTimestamp: String?
    public let text: String

    public init(timestamp: Date? = nil, rawTimestamp: String? = nil, text: String) {
        self.timestamp = timestamp
        self.rawTimestamp = rawTimestamp
        self.text = text
    }

    private static func daysFromCivil(year: Int, month: Int, day: Int) -> Int {
        let y = year - (month <= 2 ? 1 : 0)
        let era = (y >= 0 ? y : y - 399) / 400
        let yoe = y - era * 400
        let mAdj = month + (month > 2 ? -3 : 9)
        let doy = (153 * mAdj + 2) / 5 + day - 1
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy
        return era * 146097 + doe - 719468
    }

    private static func formatUTC(_ date: Date) -> String {
        var z = Int(floor(date.timeIntervalSince1970))
        var rem = z % 86400
        var days = z / 86400
        if rem < 0 { rem += 86400; days -= 1 }
        let hh = rem / 3600
        let mm = (rem % 3600) / 60
        let ss = rem % 60
        z = days + 719468
        let era = (z >= 0 ? z : z - 146096) / 146097
        let doe = z - era * 146097
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
        var y = yoe + era * 400
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100)
        let mp = (5 * doy + 2) / 153
        let d = doy - (153 * mp + 2) / 5 + 1
        let m = mp + (mp < 10 ? 3 : -9)
        if m <= 2 { y += 1 }
        return String(format: "%04d-%02d-%02d %02d:%02d:%02d", y, m, d, hh, mm, ss)
    }

    /// Parses `"2026-10-12 12:34:56 あああ"`, `"2026/10/12 12:34 あああ"`, `"2026-10-12T12:34:56+09:00 あああ"`, `"2026-10-12 あああ"`, or plain `"あああ"`.
    public static func parse(_ raw: String) -> TimestampedComment {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return TimestampedComment(text: "") }

        let chars = Array(trimmed.unicodeScalars)
        let n = chars.count
        var i = 0
        let hadBracket = (i < n && chars[i].value == 0x5B) // '['
        if hadBracket { i += 1 }

        func readDigits(maxCount: Int) -> (val: Int, count: Int) {
            var v = 0, c = 0
            while i < n && c < maxCount && chars[i].value >= 0x30 && chars[i].value <= 0x39 {
                v = v * 10 + Int(chars[i].value - 0x30)
                i += 1; c += 1
            }
            return (v, c)
        }

        let startIdx = i
        let (year, yLen) = readDigits(maxCount: 4)
        guard yLen == 4, year >= 1970, year <= 2199, i < n,
              (chars[i].value == 0x2D || chars[i].value == 0x2F) else {
            return TimestampedComment(text: trimmed)
        }
        let dateSep = chars[i].value
        i += 1
        let (month, mLen) = readDigits(maxCount: 2)
        guard (1...2).contains(mLen), (1...12).contains(month), i < n, chars[i].value == dateSep else {
            return TimestampedComment(text: trimmed)
        }
        i += 1
        let (day, dLen) = readDigits(maxCount: 2)
        guard (1...2).contains(dLen), (1...31).contains(day) else {
            return TimestampedComment(text: trimmed)
        }

        var hour = 0, minute = 0, second = 0, tzOffsetSec = 0
        let afterDateIdx = i
        if i < n && (chars[i].value == 0x20 || chars[i].value == 0x54 || chars[i].value == 0x74) {
            let saved = i
            i += 1
            let (hh, hLen) = readDigits(maxCount: 2)
            if (1...2).contains(hLen), (0...23).contains(hh), i < n, chars[i].value == 0x3A {
                i += 1
                let (mm, minLen) = readDigits(maxCount: 2)
                if minLen == 2, (0...59).contains(mm) {
                    hour = hh; minute = mm
                    if i < n && chars[i].value == 0x3A {
                        let secSaved = i
                        i += 1
                        let (ss, sLen) = readDigits(maxCount: 2)
                        if sLen == 2 && (0...60).contains(ss) {
                            second = min(59, ss)
                            if i < n && chars[i].value == 0x2E {
                                i += 1
                                _ = readDigits(maxCount: 9)
                            }
                        } else {
                            i = secSaved
                        }
                    }
                    // Optional Z or +/-HH(:?MM)
                    if i < n && (chars[i].value == 0x5A || chars[i].value == 0x7A) {
                        i += 1
                    } else if i < n && (chars[i].value == 0x2B || chars[i].value == 0x2D) {
                        let sign = chars[i].value == 0x2D ? -1 : 1
                        let tzSaved = i
                        i += 1
                        let (tzh, tzhL) = readDigits(maxCount: 2)
                        if tzhL == 2 && tzh <= 14 {
                            var tzm = 0
                            if i < n && chars[i].value == 0x3A {
                                i += 1
                                let (mVal, mL) = readDigits(maxCount: 2)
                                if mL == 2 { tzm = mVal }
                            } else {
                                let (mVal, mL) = readDigits(maxCount: 2)
                                if mL == 2 { tzm = mVal }
                            }
                            tzOffsetSec = sign * (tzh * 3600 + tzm * 60)
                        } else {
                            i = tzSaved
                        }
                    }
                } else {
                    i = afterDateIdx
                }
            } else {
                i = saved
            }
        }

        let tsEndIdx = i
        if hadBracket && i < n && chars[i].value == 0x5D { // ']'
            i += 1
        }

        // Must be followed by whitespace (or ']') + non-empty comment text
        let sepBeforeBody = hadBracket || (i < n && (chars[i].value == 0x20 || chars[i].value == 0x09 || chars[i].value == 0x3000))
        guard sepBeforeBody else {
            return TimestampedComment(text: trimmed)
        }
        while i < n && (chars[i].value == 0x20 || chars[i].value == 0x09 || chars[i].value == 0x3000) {
            i += 1
        }
        guard i < n else {
            return TimestampedComment(text: trimmed)
        }

        var rawTS = ""
        for k in startIdx..<tsEndIdx { rawTS.unicodeScalars.append(chars[k]) }
        var body = ""
        for k in i..<n { body.unicodeScalars.append(chars[k]) }

        let epochDays = daysFromCivil(year: year, month: month, day: day)
        let unixSec = Double(epochDays * 86400 + hour * 3600 + minute * 60 + second - tzOffsetSec)
        return TimestampedComment(
            timestamp: Date(timeIntervalSince1970: unixSec),
            rawTimestamp: rawTS,
            text: body.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }

    /// Parses multiple `--text` strings and stably orders them chronologically when timestamps are present.
    public static func parseList(_ rawInputs: [String]) -> [TimestampedComment] {
        let parsed = rawInputs
            .map { parse($0) }
            .filter { !$0.text.isEmpty }
        guard parsed.count >= 2, parsed.allSatisfy({ $0.timestamp != nil }) else {
            return parsed
        }
        return parsed.enumerated().sorted { lhs, rhs in
            let t0 = lhs.element.timestamp!.timeIntervalSince1970
            let t1 = rhs.element.timestamp!.timeIntervalSince1970
            if t0 != t1 { return t0 < t1 }
            return lhs.offset < rhs.offset
        }.map(\.element)
    }

    /// Synthesizes a pure partner-comment stream (`0%` sender/host text) with realistic timestamps from a `RelationalDialogueCase`.
    public static func synthesizeStream(from rc: RelationalDialogueCase, targetCount: Int? = nil) -> [TimestampedComment] {
        let eRaw = DualStreamRelationalEngine.splitChatSentences(rc.partnerEarlyText)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let cRaw = DualStreamRelationalEngine.splitChatSentences(rc.partnerCurrentText)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let earlyParts = eRaw.isEmpty ? [rc.partnerEarlyText] : eRaw
        let curParts = cRaw.isEmpty ? [rc.partnerCurrentText] : cRaw

        let texts: [String]
        if let tcCount = targetCount {
            if tcCount <= 1 {
                texts = [rc.partnerCurrentText]
            } else if tcCount == 2 {
                texts = [rc.partnerEarlyText, rc.partnerCurrentText]
            } else if tcCount == 3 {
                if curParts.count >= 2 {
                    texts = [rc.partnerEarlyText, curParts[0], curParts[1..<curParts.count].joined(separator: " ")]
                } else if earlyParts.count >= 2 {
                    texts = [earlyParts[0], earlyParts[1..<earlyParts.count].joined(separator: " "), rc.partnerCurrentText]
                } else {
                    texts = [rc.partnerEarlyText, rc.partnerCurrentText]
                }
            } else {
                texts = earlyParts + curParts
            }
        } else {
            texts = earlyParts + curParts
        }

        if texts.count <= 1 {
            let d = Date(timeIntervalSince1970: 1_792_000_000)
            return [TimestampedComment(timestamp: d, rawTimestamp: formatUTC(d), text: texts.first ?? rc.partnerCurrentText)]
        }

        let tc = rc.temporalContext
        let lagRatio = Double(tc.replyLagHoursT4 / max(0.05, tc.actorBaselineReplyHours))

        // Deterministic hash from scenario ID to vary steady baseline cadence across hours / days / weeks / months (e.g. Oct->Nov->Dec)
        var idHash: UInt64 = 1469598103934665603
        for b in rc.id.utf8 { idHash = (idHash ^ UInt64(b)) &* 1099511628211 }
        let scaleBucket = Int(idHash % 4)
        let steadyScalesHours: [Double] = [3.0, 26.0, 168.0, 840.0] // 3h, ~1d, 7d, 35d (monthly)

        let baseEarlyHours: Double
        if lagRatio >= 3.5 {
            // Decelerating / fade-out: early cadence was active (2h to 48h), then stretched dramatically
            baseEarlyHours = max(1.0, min(48.0, Double(max(tc.actorBaselineReplyHours, tc.replyLagHoursT2)) * 4.0))
        } else if tc.eventDriven || tc.gapDaysT2ToT3 >= 14.0 {
            baseEarlyHours = max(168.0, Double(tc.gapDaysT2ToT3) * 24.0)
        } else {
            baseEarlyHours = steadyScalesHours[scaleBucket]
        }

        let targetLastHours: Double
        if tc.postT4UnansweredProbe && lagRatio <= 1.8 {
            // Rapid consecutive follow-up / burst (3 to 12 minutes)
            targetLastHours = min(0.20, max(0.05, Double(tc.replyLagHoursT4) * 0.12))
        } else if lagRatio <= 0.40 {
            targetLastHours = max(0.08, min(baseEarlyHours * 0.20, Double(tc.replyLagHoursT4)))
        } else if lagRatio >= 3.5 {
            // 8x to 60x slowdown relative to own early baseline
            let mult = max(8.0, min(65.0, lagRatio * 2.2))
            targetLastHours = max(baseEarlyHours * mult, Double(tc.gapDaysT2ToT3) * 24.0)
        } else {
            // Steady cadence (0.72x to 1.35x of own baseline, just like Oct 12 -> Nov 23 -> Dec 25!)
            let jitter = 0.72 + Double(idHash % 64) * 0.01
            targetLastHours = max(0.5, baseEarlyHours * jitter)
        }

        var cursorSec: Double = 1_791_806_096 // 2026-10-12 12:34:56 UTC
        var out: [TimestampedComment] = []
        out.reserveCapacity(texts.count)
        let n = texts.count

        for idx in 0..<n {
            if idx > 0 {
                let stepH: Double
                if idx == n - 1 {
                    stepH = targetLastHours
                } else if idx == 1 {
                    stepH = baseEarlyHours
                } else {
                    stepH = sqrt(baseEarlyHours * max(baseEarlyHours, targetLastHours))
                }
                cursorSec += stepH * 3600.0
            }
            let d = Date(timeIntervalSince1970: cursorSec)
            out.append(TimestampedComment(timestamp: d, rawTimestamp: formatUTC(d), text: texts[idx]))
        }
        return out
    }
}

/// Output returned by `PsychologicalStateLM.describeComments(...)` / `describeTexts(...)`.
public struct PsychologicalStateResult: Sendable {
    public let psychologicalText: String
    public let predictedArchetype: RelationalArchetype
    public let predictedTopology: RelationalTopology
    public let predictedEmotion: EmotionalGradation
    public let inference: RelationalInferenceOutput
    public let commentCount: Int
    public let usedPastLog: Bool
    public let usedTemporalContext: Bool
    public let relationalLatencyMs: Double
    public let generationLatencyMs: Double

    public var totalLatencyMs: Double { relationalLatencyMs + generationLatencyMs }
}

/// Deterministic BPE Subword Tokenizer built over character primitives (`0%` `<unk>` on any base character, ~3x shorter sequences).
public struct SubwordVocabulary: Codable, Sendable, Equatable {
    public struct MergeRule: Codable, Sendable, Equatable {
        public let left: Int
        public let right: Int
        public let merged: Int
    }

    private struct PairTarget: Sendable, Equatable {
        let rank: Int
        let merged: Int
    }

    public let idToPiece: [String]
    public let merges: [MergeRule]
    private let charToId: [String: Int]
    private let pairToMerged: [UInt64: PairTarget]

    private enum CodingKeys: String, CodingKey {
        case idToPiece
        case merges
    }

    public var count: Int { idToPiece.count }

    public static func == (lhs: SubwordVocabulary, rhs: SubwordVocabulary) -> Bool {
        lhs.idToPiece == rhs.idToPiece && lhs.merges == rhs.merges
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let pieces = try c.decode([String].self, forKey: .idToPiece)
        let m = try c.decode([MergeRule].self, forKey: .merges)
        self.init(idToPiece: pieces, merges: m)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(idToPiece, forKey: .idToPiece)
        try c.encode(merges, forKey: .merges)
    }

    public init(idToPiece: [String], merges: [MergeRule]) {
        self.idToPiece = idToPiece
        self.merges = merges
        var cMap: [String: Int] = [:]
        for (idx, piece) in idToPiece.enumerated() where idx >= TextVocabulary.specialCount && piece.count == 1 {
            cMap[piece] = idx
        }
        self.charToId = cMap
        var pMap: [UInt64: PairTarget] = [:]
        pMap.reserveCapacity(merges.count)
        for (rank, m) in merges.enumerated() {
            let key = (UInt64(UInt32(m.left)) << 32) | UInt64(UInt32(m.right))
            pMap[key] = PairTarget(rank: rank, merged: m.merged)
        }
        self.pairToMerged = pMap
    }

    /// Splits Japanese prose along clause & bunsetsu boundaries (`、` / `。` or `[Hiragana ≠ っ] -> [Kanji/Katakana]`, keeping polite prefixes `お`/`ご`).
    public static func splitBunsetsu(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        func isHira(_ ch: Character) -> Bool {
            guard let v = ch.unicodeScalars.first?.value else { return false }
            return (0x3040...0x309F).contains(v)
        }
        func isCJKOrKata(_ ch: Character) -> Bool {
            if ch == "々" { return true }
            guard let v = ch.unicodeScalars.first?.value else { return false }
            return (0x30A0...0x30FF).contains(v) || (0x4E00...0x9FFF).contains(v)
        }
        var chunks: [String] = []
        var cur = ""
        for ch in text {
            if cur.isEmpty {
                cur.append(ch)
                continue
            }
            let prev = cur.last!
            if prev == "、" || prev == "。" {
                chunks.append(cur)
                cur = String(ch)
            } else if isHira(prev) && isCJKOrKata(ch) && prev != "っ" && cur != "お" && cur != "ご" {
                chunks.append(cur)
                cur = String(ch)
            } else {
                cur.append(ch)
            }
        }
        if !cur.isEmpty { chunks.append(cur) }
        return chunks
    }

    /// Trains a deterministic bunsetsu-pretokenized BPE vocabulary from `trainingTexts` plus `baseCharacters`.
    public static func train(
        trainingTexts: [String],
        baseCharacters: String = "",
        numMerges: Int = 1200,
        minPairFreq: Int = 4,
        maxPieceLen: Int = 16
    ) -> SubwordVocabulary {
        // Must match TextVocabulary special token IDs: pad=0, unk=1, bos=2, eos=3
        var pieces: [String] = ["<pad>", "<unk>", "<bos>", "<eos>"]
        precondition(
            pieces[TextVocabulary.padId] == "<pad>" &&
            pieces[TextVocabulary.unkId] == "<unk>" &&
            pieces[TextVocabulary.bosId] == "<bos>" &&
            pieces[TextVocabulary.eosId] == "<eos>"
        )
        var seenChars: Set<Character> = []
        var orderedChars: [Character] = []
        for ch in baseCharacters where !seenChars.contains(ch) {
            seenChars.insert(ch)
            orderedChars.append(ch)
        }
        for t in trainingTexts {
            for ch in t where !seenChars.contains(ch) {
                seenChars.insert(ch)
                orderedChars.append(ch)
            }
        }
        var charIdMap: [Character: Int] = [:]
        var endsWithPeriod: [Bool] = [false, false, false, false]
        var pieceCharLen: [Int] = [0, 0, 0, 0]
        endsWithPeriod.reserveCapacity(orderedChars.count + numMerges + 8)
        pieceCharLen.reserveCapacity(orderedChars.count + numMerges + 8)
        for ch in orderedChars {
            let id = pieces.count
            pieces.append(String(ch))
            endsWithPeriod.append(ch == "。")
            pieceCharLen.append(1)
            charIdMap[ch] = id
        }

        // Deduplicate unique bunsetsu chunks weighted by frequency (10x faster BPE & zero cross-bunsetsu monster merges)
        var wordFreq: [String: Int] = [:]
        for t in trainingTexts {
            for w in splitBunsetsu(t) {
                wordFreq[w, default: 0] += 1
            }
        }
        let sortedWords = wordFreq.keys.sorted()
        var wordSeqs: [[Int]] = sortedWords.map { w in
            w.map { charIdMap[$0] ?? TextVocabulary.unkId }
        }
        let freqs: [Int] = sortedWords.map { wordFreq[$0]! }

        var learnedMerges: [MergeRule] = []
        learnedMerges.reserveCapacity(numMerges)

        for _ in 0..<numMerges {
            var counts: [UInt64: Int] = [:]
            counts.reserveCapacity(4096)
            for wIdx in 0..<wordSeqs.count {
                let seq = wordSeqs[wIdx]
                if seq.count < 2 { continue }
                let f = freqs[wIdx]
                for i in 0..<(seq.count - 1) {
                    let a = seq[i]
                    let b = seq[i + 1]
                    if endsWithPeriod[a] { continue }
                    if pieceCharLen[a] + pieceCharLen[b] > maxPieceLen { continue }
                    let key = (UInt64(UInt32(a)) << 32) | UInt64(UInt32(b))
                    counts[key, default: 0] += f
                }
            }
            var bestKey: UInt64 = 0
            var bestCount = 0
            for (k, c) in counts {
                if c > bestCount || (c == bestCount && k < bestKey) {
                    bestCount = c
                    bestKey = k
                }
            }
            if bestCount < minPairFreq { break }

            let left = Int(UInt32(bestKey >> 32))
            let right = Int(UInt32(bestKey & 0xFFFF_FFFF))
            let newId = pieces.count
            pieces.append(pieces[left] + pieces[right])
            endsWithPeriod.append(endsWithPeriod[right])
            pieceCharLen.append(pieceCharLen[left] + pieceCharLen[right])
            learnedMerges.append(MergeRule(left: left, right: right, merged: newId))

            for wIdx in 0..<wordSeqs.count where wordSeqs[wIdx].count >= 2 {
                let old = wordSeqs[wIdx]
                var updated: [Int] = []
                updated.reserveCapacity(old.count)
                var i = 0
                while i < old.count {
                    if i + 1 < old.count && old[i] == left && old[i + 1] == right {
                        updated.append(newId)
                        i += 2
                    } else {
                        updated.append(old[i])
                        i += 1
                    }
                }
                wordSeqs[wIdx] = updated
            }
        }
        return SubwordVocabulary(idToPiece: pieces, merges: learnedMerges)
    }

    public func encode(_ text: String, addEOS: Bool = false) -> [Int] {
        var result: [Int] = []
        result.reserveCapacity(text.count / 2 + 2)
        for chunk in Self.splitBunsetsu(text) {
            var ids: [Int] = []
            ids.reserveCapacity(chunk.count)
            for ch in chunk {
                let s = String(ch)
                if let id = charToId[s] {
                    ids.append(id)
                } else if let firstScalar = ch.unicodeScalars.first, let fb = charToId[String(firstScalar)] {
                    ids.append(fb)
                } else {
                    ids.append(TextVocabulary.unkId)
                }
            }
            while ids.count >= 2 {
                var bestPos = -1
                var bestRank = Int.max
                var bestMerged = 0
                for i in 0..<(ids.count - 1) {
                    let key = (UInt64(UInt32(ids[i])) << 32) | UInt64(UInt32(ids[i + 1]))
                    if let hit = pairToMerged[key], hit.rank < bestRank {
                        bestRank = hit.rank
                        bestPos = i
                        bestMerged = hit.merged
                    }
                }
                if bestPos < 0 { break }
                ids[bestPos] = bestMerged
                ids.remove(at: bestPos + 1)
            }
            result.append(contentsOf: ids)
        }
        if addEOS { result.append(TextVocabulary.eosId) }
        return result
    }

    public func decode(_ ids: [Int]) -> String {
        var out = ""
        for id in ids {
            if id == TextVocabulary.eosId { break }
            if id < TextVocabulary.specialCount || id >= idToPiece.count { continue }
            out += idToPiece[id]
        }
        return out
    }

    public func serialize() -> String {
        guard let data = try? JSONEncoder().encode(self),
              let s = String(data: data, encoding: .utf8) else { return "" }
        return s
    }

    public static func deserialize(_ str: String) -> SubwordVocabulary {
        guard !str.isEmpty,
              let data = str.data(using: .utf8),
              let v = try? JSONDecoder().decode(SubwordVocabulary.self, from: data) else {
            return SubwordVocabulary.train(trainingTexts: [""], numMerges: 0)
        }
        return v
    }
}

/// Calibrated single-actor comment-stream semantic probe heads serialized with `PsychologicalStateLM` (`0%` sender text required).
public struct PsychAuxiliaryHeads: Codable, Sendable, Equatable {
    public var archCur: SoftmaxClassifierHead?            // 6-way on hybCur (168D)
    public var topoCur: SoftmaxClassifierHead?            // 5-way on hybCur (168D)
    public var affCur: SoftmaxClassifierHead?             // 5-way on hybCur (168D)
    public var fricCur: SoftmaxClassifierHead?            // 5-way on hybCur (168D)
    public var negMixT4: SoftmaxClassifierHead?           // 2-way o3_neg_mix on hybCur (168D)
    public var boundaryT4: SoftmaxClassifierHead?         // 4-way o4_boundary on hybCur (168D)
    public var trajectoryTextOnly: SoftmaxClassifierHead? // 10-way on hybCur (168D)

    public var archEarly: SoftmaxClassifierHead?          // 6-way on hybEarly (168D)
    public var topoEarly: SoftmaxClassifierHead?          // 5-way on hybEarly (168D)
    public var affEarly: SoftmaxClassifierHead?           // 5-way on hybEarly (168D)
    public var fricEarly: SoftmaxClassifierHead?          // 5-way on hybEarly (168D)

    public var trajectoryFull: SoftmaxClassifierHead?     // 10-way on [hybEarly(168)+hybCur(168)+hybCross(168)+temp(16)=520D]
    public var archFull: SoftmaxClassifierHead?           // 6-way on [520D]
    public var topoFull: SoftmaxClassifierHead?           // 5-way on [520D]

    public init(
        archCur: SoftmaxClassifierHead? = nil,
        topoCur: SoftmaxClassifierHead? = nil,
        affCur: SoftmaxClassifierHead? = nil,
        fricCur: SoftmaxClassifierHead? = nil,
        negMixT4: SoftmaxClassifierHead? = nil,
        boundaryT4: SoftmaxClassifierHead? = nil,
        trajectoryTextOnly: SoftmaxClassifierHead? = nil,
        archEarly: SoftmaxClassifierHead? = nil,
        topoEarly: SoftmaxClassifierHead? = nil,
        affEarly: SoftmaxClassifierHead? = nil,
        fricEarly: SoftmaxClassifierHead? = nil,
        trajectoryFull: SoftmaxClassifierHead? = nil,
        archFull: SoftmaxClassifierHead? = nil,
        topoFull: SoftmaxClassifierHead? = nil
    ) {
        self.archCur = archCur
        self.topoCur = topoCur
        self.affCur = affCur
        self.fricCur = fricCur
        self.negMixT4 = negMixT4
        self.boundaryT4 = boundaryT4
        self.trajectoryTextOnly = trajectoryTextOnly
        self.archEarly = archEarly
        self.topoEarly = topoEarly
        self.affEarly = affEarly
        self.fricEarly = fricEarly
        self.trajectoryFull = trajectoryFull
        self.archFull = archFull
        self.topoFull = topoFull
    }

    public func serialize() -> String {
        guard let data = try? JSONEncoder().encode(self),
              let s = String(data: data, encoding: .utf8) else { return "" }
        return s
    }

    public static func deserialize(_ str: String) -> PsychAuxiliaryHeads {
        guard !str.isEmpty,
              let data = str.data(using: .utf8),
              let h = try? JSONDecoder().decode(PsychAuxiliaryHeads.self, from: data) else {
            return PsychAuxiliaryHeads()
        }
        return h
    }
}

/// Extracts a compact, high-SNR `128`-dimensional psychological state vector purely from a comment stream `[TimestampedComment]` (`0%` sender text).
public enum PsychConditionExtractor {
    public static let prosodySliceDim: Int = 40   // hybCur[128..<168]: subband geometry + internal shift + rhythm + emoji/orthography
    public static let coreProbsDim: Int = 21      // arch(6) + topo(5) + affection(5) + friction(5)
    public static let auxProbsDim: Int = 16       // trajectory(10) + negMix(2) + boundary(4)
    public static let shiftProbsDim: Int = 21     // cur - early over arch(6)+topo(5)+aff(5)+fric(5)
    public static let crossAndDeltaDim: Int = 13  // crossCoherence(5) + deltaProsody(8)
    public static let streamTemporalDim: Int = 16 // self-calibrated comment timestamp cadence features
    /// `40 (prosodyCur) + 21 (coreProbs) + 16 (auxProbs) + 1 (hasPast) + 21 (deltaProbs) + 13 (cross+deltaProsody) + 16 (streamTemporal) = 128`
    public static let rawConditionDim: Int = 128

    /// Splits a single text into `(firstHalf, secondHalf)` clauses so `DualStreamRelationalEngine` captures internal turn dynamics without any `sender`.
    public static func selfSplitTokens(_ text: String, vocab: TextVocabulary) -> (a: [Int], b: [Int]) {
        let clauses = DualStreamRelationalEngine.splitChatSentences(text)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if clauses.count >= 2 {
            let mid = clauses.count / 2
            let first = clauses[0..<mid].joined(separator: " ")
            let second = clauses[mid..<clauses.count].joined(separator: " ")
            let a = vocab.encode(first, maxLength: 512, addEOS: false)
            let b = vocab.encode(second, maxLength: 512, addEOS: false)
            if !a.isEmpty && !b.isEmpty { return (a, b) }
        }
        let toks = vocab.encode(text, maxLength: 512, addEOS: false)
        let safe = toks.isEmpty ? [TextVocabulary.padId] : toks
        return (safe, safe)
    }

    /// Splits a chronological comment list into `(earlyText?, currentText)` (`earlyText == nil` when `comments.count <= 1`).
    public static func splitCommentHistory(_ comments: [TimestampedComment]) -> (earlyText: String?, currentText: String) {
        guard !comments.isEmpty else { return (nil, "") }
        if comments.count == 1 {
            return (nil, comments[0].text)
        }
        let n = comments.count
        let splitIdx = (n == 2 || n == 3) ? 1 : (n / 2)
        let early = comments[0..<splitIdx].map(\.text).joined(separator: " ")
        let current = comments[splitIdx..<n].map(\.text).joined(separator: " ")
        return (early, current)
    }

    /// Extracts `16D` self-calibrated timestamp cadence & length-shift features directly from `[TimestampedComment]`.
    public static func extractStreamTemporalFeatures(_ comments: [TimestampedComment]) -> [Float] {
        var f = [Float](repeating: 0.0, count: streamTemporalDim)
        let n = comments.count
        guard n >= 1 else { return f }
        f[2] = log1pf(Float(n))

        if n >= 2 {
            let splitIdx = (n == 2 || n == 3) ? 1 : (n / 2)
            let earlyMeanChars = Float(comments[0..<splitIdx].reduce(0) { $0 + $1.text.count }) / Float(max(1, splitIdx))
            let lastChars = Float(comments[n - 1].text.count)
            let lenLog = logf((lastChars + 5.0) / (earlyMeanChars + 5.0))
            f[15] = max(-3.0, min(3.0, lenLog))
        }

        let validDates = comments.compactMap(\.timestamp)
        guard n >= 2, validDates.count == n else {
            return f
        }

        var gapsHours: [Double] = []
        gapsHours.reserveCapacity(n - 1)
        for i in 0..<(n - 1) {
            let dt = max(0.01, (validDates[i + 1].timeIntervalSince1970 - validDates[i].timeIntervalSince1970) / 3600.0)
            gapsHours.append(dt)
        }

        let lastGapH = Float(gapsHours.last!)
        let earlyCount = max(1, gapsHours.count - 1)
        let earlyGapH: Float
        if gapsHours.count >= 2 {
            var logSum = 0.0
            for k in 0..<earlyCount { logSum += log(max(0.01, gapsHours[k])) }
            earlyGapH = Float(exp(logSum / Double(earlyCount)))
        } else {
            earlyGapH = lastGapH
        }
        let maxGapDays = Float(gapsHours.max()! / 24.0)
        let totalSpanDays = Float(max(0.0, validDates.last!.timeIntervalSince1970 - validDates.first!.timeIntervalSince1970) / 86400.0)

        f[0] = 1.0 // hasTimestamps (>= 2 comments with valid timestamps)
        f[1] = gapsHours.count >= 2 ? 1.0 : 0.0 // hasMultiIntervals (>= 3 comments -> cadence acceleration observable)
        // Normalize log-intervals softly and center dynamics on time-scale-invariant relative ratios
        f[3] = log1pf(min(earlyGapH, 168.0)) * 0.25
        f[4] = earlyGapH >= 240.0 ? 1.0 : Float(earlyGapH / 240.0) // seasonal/monthly rhythm indicator
        f[5] = log1pf(min(lastGapH, 168.0)) * 0.25
        f[6] = lastGapH <= 1.0 ? (1.0 - lastGapH) : 0.0
        f[7] = gapsHours.count == 1 ? tanhf(maxGapDays / 21.0) : 0.0 // raw gap only when N==2 (no baseline interval)
        f[8] = log1pf(min(totalSpanDays, 90.0)) * 0.20

        var decelFlag: Float = 0.0
        if gapsHours.count >= 2 {
            let cadRatio = max(-6.0, min(6.0, logf((lastGapH + 0.25) / (earlyGapH + 0.25))))
            f[9] = cadRatio
            decelFlag = max(0.0, tanhf((cadRatio - 1.35) * 0.85)) // decelerating (> ~3.8x slower than own baseline)
            f[10] = decelFlag
            f[11] = max(0.0, tanhf((-cadRatio - 1.35) * 0.85))    // accelerating (> ~3.8x faster than own baseline)
            f[13] = abs(cadRatio) < 0.95 ? 1.0 : 0.0              // steady periodic commenter (daily, weekly, OR monthly!)
        } else if maxGapDays >= 25.0 {
            decelFlag = tanhf((maxGapDays - 20.0) / 20.0)
            f[10] = decelFlag
        }
        f[12] = expf(-lastGapH / 0.45) // rapid consecutive burst (<= ~25 min)
        f[14] = decelFlag * tanhf(maxGapDays / 14.0) // long silence ONLY when decelerating!
        return f
    }

    private static func keyProsody8(_ hyb: [Float]) -> [Float] {
        guard hyb.count >= 168 else { return [Float](repeating: 0.0, count: 8) }
        return [hyb[128], hyb[129], hyb[130], hyb[148], hyb[150], hyb[153], hyb[145], hyb[161]]
    }

    /// Ultra-fast (`~0.01 ms`) Stage 1 `SubbandClauseSplitter` LL/HH proviso semantic features (`2 * min(64, D) + 4` floats) for probe heads (`0%` raw topic coordinates passed to LM).
    public static func extractAutoencoderSemantic(tokens: [Int], weights: SWTHawkWeights) -> [Float] {
        let d = weights.dim
        let kDim = min(64, d)
        guard kDim > 0, !weights.tokenEmbedding.isEmpty else { return [] }
        let safeToks = tokens.isEmpty ? [TextVocabulary.padId] : Array(tokens.prefix(256))
        let vSize = weights.vocabSize
        var out = [Float](repeating: 0.0, count: 2 * kDim + 4)

        if weights.subbandClassifier.isConfigured {
            let split = SubbandClauseSplitter.split(tokens: safeToks, classifier: weights.subbandClassifier)
            if !split.llTokens.isEmpty {
                let inv = 1.0 / Float(split.llTokens.count)
                for tok in split.llTokens {
                    let base = max(0, min(vSize - 1, tok)) * d
                    for c in 0..<kDim { out[c] += weights.tokenEmbedding[base + c] * inv }
                }
            }
            let hhToks = split.provisos.flatMap(\.tokens)
            if !hhToks.isEmpty {
                let inv = 1.0 / Float(hhToks.count)
                for tok in hhToks {
                    let base = max(0, min(vSize - 1, tok)) * d
                    for c in 0..<kDim { out[kDim + c] += weights.tokenEmbedding[base + c] * inv }
                }
            }
            let total = Float(max(1, safeToks.count))
            out[2 * kDim] = Float(split.llTokens.count) / total
            out[2 * kDim + 1] = Float(hhToks.count) / total
            out[2 * kDim + 2] = Float(split.provisos.count) * 0.25
            out[2 * kDim + 3] = hhToks.isEmpty ? 0.0 : 1.0
        } else {
            let inv = 1.0 / Float(safeToks.count)
            for tok in safeToks {
                let base = max(0, min(vSize - 1, tok)) * d
                for c in 0..<kDim { out[c] += weights.tokenEmbedding[base + c] * inv }
            }
        }
        return out
    }

    private static func safeProbs(_ head: SoftmaxClassifierHead?, _ x: [Float], count: Int, fallback: [Float]) -> [Float] {
        if let h = head, h.featureDim == x.count {
            let p = h.probabilities(x)
            if p.count == count { return p }
        }
        if fallback.count == count { return fallback }
        var u = [Float](repeating: 1.0 / Float(max(1, count)), count: count)
        if !fallback.isEmpty {
            for i in 0..<min(count, fallback.count) { u[i] = fallback[i] }
        }
        return u
    }

    public static func buildCommentStreamCondition(
        infCur: RelationalInferenceOutput,
        semCur: [Float] = [],
        infEarly: RelationalInferenceOutput?,
        semEarly: [Float] = [],
        infCross: RelationalInferenceOutput?,
        streamTemporal: [Float],
        auxHeads: PsychAuxiliaryHeads
    ) -> (raw: [Float], arch: RelationalArchetype, topo: RelationalTopology, emotion: EmotionalGradation) {
        var raw = [Float](repeating: 0.0, count: rawConditionDim)
        var cursor = 0
        let hybCur = infCur.hybridFeatures
        let probeCur = hybCur + semCur
        let temp16 = streamTemporal.count == streamTemporalDim ? streamTemporal : [Float](repeating: 0.0, count: streamTemporalDim)

        // 1. Current Comment Subband & Prosody Geometry (40D)
        if hybCur.count >= 128 + prosodySliceDim {
            for i in 0..<prosodySliceDim { raw[cursor + i] = hybCur[128 + i] }
        }
        cursor += prosodySliceDim

        // 2. Core Probabilities (21D: arch 6 + topo 5 + affection 5 + friction 5)
        let archCurP = safeProbs(auxHeads.archCur, probeCur, count: 6, fallback: infCur.hybridArchetypeProbsT4)
        let topoCurP = safeProbs(auxHeads.topoCur, probeCur, count: 5, fallback: infCur.hybridTopologyProbsT4)
        let affCurP = safeProbs(auxHeads.affCur, probeCur, count: 5, fallback: (0..<5).map { $0 == infCur.hybridPredictedEmotionT4.affection ? 1.0 : 0.0 })
        let fricCurP = safeProbs(auxHeads.fricCur, probeCur, count: 5, fallback: (0..<5).map { $0 == infCur.hybridPredictedEmotionT4.friction ? 1.0 : 0.0 })

        let fullFeat: [Float]? = if let e = infEarly, let cr = infCross {
            (e.hybridFeatures + semEarly) + probeCur + cr.hybridFeatures + temp16
        } else {
            nil
        }
        let finalArchP = fullFeat.map { safeProbs(auxHeads.archFull, $0, count: 6, fallback: archCurP) } ?? archCurP
        let finalTopoP = fullFeat.map { safeProbs(auxHeads.topoFull, $0, count: 5, fallback: topoCurP) } ?? topoCurP

        for v in finalArchP { raw[cursor] = v; cursor += 1 }
        for v in finalTopoP { raw[cursor] = v; cursor += 1 }
        for v in affCurP { raw[cursor] = v; cursor += 1 }
        for v in fricCurP { raw[cursor] = v; cursor += 1 }

        // 3. Auxiliary Trajectory (10D) + Veiled NegMix (2D) + Boundary (4D) = 16D
        let trajTextP = safeProbs(auxHeads.trajectoryTextOnly, probeCur, count: 10, fallback: [Float](repeating: 0.1, count: 10))
        let trajP = fullFeat.map { safeProbs(auxHeads.trajectoryFull, $0, count: 10, fallback: trajTextP) } ?? trajTextP
        let negP = safeProbs(auxHeads.negMixT4, probeCur, count: 2, fallback: [0.8, 0.2])
        let bndP = safeProbs(auxHeads.boundaryT4, probeCur, count: 4, fallback: [0.25, 0.5, 0.15, 0.1])
        for v in trajP { raw[cursor] = v; cursor += 1 }
        for v in negP { raw[cursor] = v; cursor += 1 }
        for v in bndP { raw[cursor] = v; cursor += 1 }

        // 4. Past-to-Present Comment Shift Features (1 + 21 + 13 = 35D)
        if let e = infEarly, let cr = infCross {
            raw[cursor] = 1.0
            cursor += 1
            let hybE = e.hybridFeatures
            let probeE = hybE + semEarly
            let archEP = safeProbs(auxHeads.archEarly, probeE, count: 6, fallback: e.hybridArchetypeProbsT4)
            let topoEP = safeProbs(auxHeads.topoEarly, probeE, count: 5, fallback: e.hybridTopologyProbsT4)
            let affEP = safeProbs(auxHeads.affEarly, probeE, count: 5, fallback: (0..<5).map { $0 == e.hybridPredictedEmotionT4.affection ? 1.0 : 0.0 })
            let fricEP = safeProbs(auxHeads.fricEarly, probeE, count: 5, fallback: (0..<5).map { $0 == e.hybridPredictedEmotionT4.friction ? 1.0 : 0.0 })

            for i in 0..<6 { raw[cursor + i] = archCurP[i] - archEP[i] }; cursor += 6
            for i in 0..<5 { raw[cursor + i] = topoCurP[i] - topoEP[i] }; cursor += 5
            for i in 0..<5 { raw[cursor + i] = affCurP[i] - affEP[i] }; cursor += 5
            for i in 0..<5 { raw[cursor + i] = fricCurP[i] - fricEP[i] }; cursor += 5

            let hybCr = cr.hybridFeatures
            for i in 0..<5 {
                raw[cursor + i] = (128 + i < hybCr.count) ? hybCr[128 + i] : 0.0
            }
            cursor += 5

            let kpCur = keyProsody8(hybCur)
            let kpEar = keyProsody8(hybE)
            for i in 0..<8 { raw[cursor + i] = kpCur[i] - kpEar[i] }
            cursor += 8
        } else {
            raw[cursor] = 0.0
            cursor += 1 + shiftProbsDim + crossAndDeltaDim
        }

        // 5. Comment-Stream Timestamp Cadence Features (16D)
        for i in 0..<streamTemporalDim { raw[cursor + i] = temp16[i] }
        cursor += streamTemporalDim
        precondition(cursor == rawConditionDim)

        let archCases = RelationalArchetype.allCases
        var bestArch = 0
        for i in 1..<min(archCases.count, finalArchP.count) where finalArchP[i] > finalArchP[bestArch] { bestArch = i }
        let topoCases = RelationalTopology.allCases
        var bestTopo = 0
        for i in 1..<min(topoCases.count, finalTopoP.count) where finalTopoP[i] > finalTopoP[bestTopo] { bestTopo = i }
        var bestAff = 0
        for i in 1..<5 where affCurP[i] > affCurP[bestAff] { bestAff = i }
        var bestFric = 0
        for i in 1..<5 where fricCurP[i] > fricCurP[bestFric] { bestFric = i }

        let predEmo = EmotionalGradation(
            affection: bestAff,
            friction: bestFric,
            arousal: infCur.hybridPredictedEmotionT4.arousal
        )
        return (raw, archCases[bestArch], topoCases[bestTopo], predEmo)
    }

    /// Single-stream pure-CPU extraction from `[TimestampedComment]` (`0%` sender text).
    public static func extractFromComments(
        relationalModel: SWTHawkAutoencoder,
        auxHeads: PsychAuxiliaryHeads = PsychAuxiliaryHeads(),
        comments: [TimestampedComment]
    ) -> (rawCondition: [Float], predictedArchetype: RelationalArchetype, predictedTopology: RelationalTopology, predictedEmotion: EmotionalGradation, inferenceCur: RelationalInferenceOutput) {
        let relVocab = relationalModel.decodedVocabulary
        let (earlyText, currentText) = splitCommentHistory(comments)
        let (curA, curB) = selfSplitTokens(currentText, vocab: relVocab)
        let infCur = relationalModel.inferRelational(senderTokenIDs: curA, receiverTokenIDs: curB)

        let cFull = relVocab.encode(currentText, maxLength: 512, addEOS: false)
        let semCur = extractAutoencoderSemantic(tokens: cFull, weights: relationalModel.weights)

        let infEarly: RelationalInferenceOutput?
        let infCross: RelationalInferenceOutput?
        let semEarly: [Float]
        if let eTxt = earlyText, !eTxt.isEmpty {
            let (eA, eB) = selfSplitTokens(eTxt, vocab: relVocab)
            infEarly = relationalModel.inferRelational(senderTokenIDs: eA, receiverTokenIDs: eB)
            let eFull = relVocab.encode(eTxt, maxLength: 512, addEOS: false)
            semEarly = extractAutoencoderSemantic(tokens: eFull, weights: relationalModel.weights)
            infCross = relationalModel.inferRelational(
                senderTokenIDs: eFull.isEmpty ? [TextVocabulary.padId] : eFull,
                receiverTokenIDs: cFull.isEmpty ? [TextVocabulary.padId] : cFull
            )
        } else {
            infEarly = nil
            infCross = nil
            semEarly = []
        }

        let temp16 = extractStreamTemporalFeatures(comments)
        let built = buildCommentStreamCondition(
            infCur: infCur,
            semCur: semCur,
            infEarly: infEarly,
            semEarly: semEarly,
            infCross: infCross,
            streamTemporal: temp16,
            auxHeads: auxHeads
        )
        return (built.raw, built.arch, built.topo, built.emotion, infCur)
    }

    /// Multi-core CPU batch condition extractor for `[[TimestampedComment]]` (`0%` sender text, `1,000+` pairs/s).
    public static func extractCommentsBatchParallel(
        weights: SWTHawkWeights,
        auxHeads: PsychAuxiliaryHeads,
        streams: [[TimestampedComment]]
    ) -> [(fullCond: [Float], noTimeCond: [Float], singleLastCond: [Float], probeEarly: [Float], probeCur: [Float], hybCross: [Float], temp16: [Float])] {
        let n = streams.count
        guard n > 0 else { return [] }
        let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)
        let hybExt = RelationalHybridFeatureExtractor(vocabulary: vocab)

        var allS: [[Int]] = []
        var allR: [[Int]] = []
        allS.reserveCapacity(3 * n)
        allR.reserveCapacity(3 * n)
        var maxLen = 8
        var streamMeta: [(hasEarly: Bool, temp16: [Float])] = []
        streamMeta.reserveCapacity(n)

        for comments in streams {
            let (earlyText, currentText) = splitCommentHistory(comments)
            let (curA, curB) = selfSplitTokens(currentText, vocab: vocab)
            let eTxt = earlyText ?? currentText
            let (eA, eB) = selfSplitTokens(eTxt, vocab: vocab)
            let eFullRaw = vocab.encode(eTxt, maxLength: 512, addEOS: false)
            let cFullRaw = vocab.encode(currentText, maxLength: 512, addEOS: false)
            let eFull = eFullRaw.isEmpty ? [TextVocabulary.padId] : eFullRaw
            let cFull = cFullRaw.isEmpty ? [TextVocabulary.padId] : cFullRaw

            allS.append(eA); allR.append(eB)
            allS.append(curA); allR.append(curB)
            allS.append(eFull); allR.append(cFull)
            maxLen = max(maxLen, max(max(eA.count, eB.count), max(max(curA.count, curB.count), max(eFull.count, cFull.count))))
            streamMeta.append((earlyText != nil, extractStreamTemporalFeatures(comments)))
        }

        let allZ = DualStreamRelationalEngine.encodeZRelCPUParallel(
            senders: allS, receivers: allR, time: maxLen, weights: weights
        )

        var semPairs = Array(repeating: (semE: [Float](), semC: [Float]()), count: n)
        let semBox = UnsafeMutablePointer<(semE: [Float], semC: [Float])>.allocate(capacity: n)
        semBox.initialize(repeating: ([], []), count: n)
        defer {
            semBox.deinitialize(count: n)
            semBox.deallocate()
        }
        struct SendableSemPtr: @unchecked Sendable { let ptr: UnsafeMutablePointer<(semE: [Float], semC: [Float])> }
        let sPtr = SendableSemPtr(ptr: semBox)
        let allSRef = allS
        let allRRef = allR
        let chunkSz = max(1, (n + 15) / 16)
        let numChunks = (n + chunkSz - 1) / chunkSz
        DispatchQueue.concurrentPerform(iterations: numChunks) { cIdx in
            let i0 = cIdx * chunkSz
            let i1 = min(n, i0 + chunkSz)
            for i in i0..<i1 {
                let sE = Self.extractAutoencoderSemantic(tokens: allSRef[3 * i + 2], weights: weights)
                let sC = Self.extractAutoencoderSemantic(tokens: allRRef[3 * i + 2], weights: weights)
                sPtr.ptr[i] = (sE, sC)
            }
        }
        for i in 0..<n { semPairs[i] = semBox[i] }

        let zeroContent = [Float](repeating: 0.0, count: weights.relational.latentDim / 2)
        var results: [(fullCond: [Float], noTimeCond: [Float], singleLastCond: [Float], probeEarly: [Float], probeCur: [Float], hybCross: [Float], temp16: [Float])] = []
        results.reserveCapacity(n)

        for i in 0..<n {
            let zE = allZ[3 * i]
            let zC = allZ[3 * i + 1]
            let zX = allZ[3 * i + 2]
            let hybE = hybExt.extract(senderTokens: allS[3 * i], receiverTokens: allR[3 * i], zRel: zE, weights: weights)
            let hybC = hybExt.extract(senderTokens: allS[3 * i + 1], receiverTokens: allR[3 * i + 1], zRel: zC, weights: weights)
            let hybX = hybExt.extract(senderTokens: allS[3 * i + 2], receiverTokens: allR[3 * i + 2], zRel: zX, weights: weights)
            let semE = semPairs[i].semE
            let semC = semPairs[i].semC

            let infE = DualStreamRelationalEngine.makeInferenceOutput(zRel: zE, hybridFeatures: hybE, zContent: zeroContent, weights: weights.relational)
            let infC = DualStreamRelationalEngine.makeInferenceOutput(zRel: zC, hybridFeatures: hybC, zContent: zeroContent, weights: weights.relational)
            let infX = DualStreamRelationalEngine.makeInferenceOutput(zRel: zX, hybridFeatures: hybX, zContent: zeroContent, weights: weights.relational)

            let meta = streamMeta[i]
            var noTimeTemp = [Float](repeating: 0.0, count: streamTemporalDim)
            noTimeTemp[2] = meta.temp16[2]
            noTimeTemp[15] = meta.temp16[15]
            var singleTemp = [Float](repeating: 0.0, count: streamTemporalDim)
            singleTemp[2] = log1pf(1.0)

            let full = buildCommentStreamCondition(
                infCur: infC,
                semCur: semC,
                infEarly: meta.hasEarly ? infE : nil,
                semEarly: meta.hasEarly ? semE : [],
                infCross: meta.hasEarly ? infX : nil,
                streamTemporal: meta.temp16,
                auxHeads: auxHeads
            ).raw
            let noTime = buildCommentStreamCondition(
                infCur: infC,
                semCur: semC,
                infEarly: meta.hasEarly ? infE : nil,
                semEarly: meta.hasEarly ? semE : [],
                infCross: meta.hasEarly ? infX : nil,
                streamTemporal: noTimeTemp,
                auxHeads: auxHeads
            ).raw
            let singleLast = buildCommentStreamCondition(
                infCur: infC,
                semCur: semC,
                infEarly: nil,
                semEarly: [],
                infCross: nil,
                streamTemporal: singleTemp,
                auxHeads: auxHeads
            ).raw

            results.append((full, noTime, singleLast, hybE + semE, hybC + semC, hybX, meta.temp16))
        }
        return results
    }
}

/// Weights for one Conditioned Griffin + Multi-Token Cross-Attention Decoder Layer.
public struct PsychDecoderLayerWeights: Sendable, Equatable {
    public let dim: Int
    public let numHeads: Int
    public let block: HawkBlockWeights
    public let wH0: [Float]
    public let bH0: [Float]
    public let wAdaGamma: [Float]
    public let bAdaGamma: [Float]
    public let wAdaBeta: [Float]
    public let bAdaBeta: [Float]
    public let crossNormGamma: [Float]
    public let crossNormBeta: [Float]
    public let wCrossQ: [Float]
    public let bCrossQ: [Float]
    public let wCrossK: [Float]
    public let bCrossK: [Float]
    public let wCrossV: [Float]
    public let bCrossV: [Float]
    public let wCrossOut: [Float]
    public let bCrossOut: [Float]

    public var parameterCount: Int {
        return block.parameterCount + wH0.count + bH0.count + wAdaGamma.count + bAdaGamma.count +
               wAdaBeta.count + bAdaBeta.count + crossNormGamma.count + crossNormBeta.count +
               wCrossQ.count + bCrossQ.count + wCrossK.count + bCrossK.count +
               wCrossV.count + bCrossV.count + wCrossOut.count + bCrossOut.count
    }

    public static func makeRandom(
        dim: Int,
        numHeads: Int = 5,
        windowSize: Int = 64,
        mlpDim: Int? = nil,
        residualScale: Float = 1.0,
        rng: inout SplitMix64
    ) -> PsychDecoderLayerWeights {
        let m = mlpDim ?? (2 * dim)
        let dBound = sqrtf(6.0 / Float(dim + dim))
        let outBound = dBound * residualScale
        let condBound = dBound * 0.35

        func randMat(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count { a[i] = rng.nextSignedUniform(scale: bound) }
            return a
        }

        let la = LocalAttentionWeights(
            dim: dim,
            numHeads: numHeads,
            windowSize: windowSize,
            normGamma: [Float](repeating: 1.0, count: dim),
            normBeta: [Float](repeating: 0.0, count: dim),
            wQ: randMat(dim * dim, dBound),
            bQ: [Float](repeating: 0.0, count: dim),
            wK: randMat(dim * dim, dBound),
            bK: [Float](repeating: 0.0, count: dim),
            wV: randMat(dim * dim, dBound),
            bV: [Float](repeating: 0.0, count: dim),
            wOut: randMat(dim * dim, outBound),
            bOut: [Float](repeating: 0.0, count: dim)
        )
        let upBound = sqrtf(6.0 / Float(dim + m))
        let downBound = sqrtf(6.0 / Float(m + dim)) * residualScale
        let hb = HawkBlockWeights(
            dim: dim,
            mlpDim: m,
            norm1Gamma: [Float](repeating: 1.0, count: dim),
            norm1Beta: [Float](repeating: 0.0, count: dim),
            wGate: randMat(dim * dim, dBound),
            bGate: [Float](repeating: 0.0, count: dim),
            rglru: RGLRUWeights.makeRandom(dim: dim, rng: &rng),
            wOut: randMat(dim * dim, outBound),
            bOut: [Float](repeating: 0.0, count: dim),
            localAttn: la,
            norm2Gamma: [Float](repeating: 1.0, count: dim),
            norm2Beta: [Float](repeating: 0.0, count: dim),
            wMLPUp: randMat(dim * m, upBound),
            bMLPUp: [Float](repeating: 0.0, count: m),
            wMLPDown: randMat(m * dim, downBound),
            bMLPDown: [Float](repeating: 0.0, count: dim)
        )

        return PsychDecoderLayerWeights(
            dim: dim,
            numHeads: numHeads,
            block: hb,
            wH0: randMat(dim * dim, condBound),
            bH0: [Float](repeating: 0.0, count: dim),
            wAdaGamma: randMat(dim * dim, condBound * 0.5),
            bAdaGamma: [Float](repeating: 0.0, count: dim),
            wAdaBeta: randMat(dim * dim, condBound * 0.5),
            bAdaBeta: [Float](repeating: 0.0, count: dim),
            crossNormGamma: [Float](repeating: 1.0, count: dim),
            crossNormBeta: [Float](repeating: 0.0, count: dim),
            wCrossQ: randMat(dim * dim, dBound),
            bCrossQ: [Float](repeating: 0.0, count: dim),
            wCrossK: randMat(dim * dim, dBound),
            bCrossK: [Float](repeating: 0.0, count: dim),
            wCrossV: randMat(dim * dim, dBound),
            bCrossV: [Float](repeating: 0.0, count: dim),
            wCrossOut: randMat(dim * dim, outBound),
            bCrossOut: [Float](repeating: 0.0, count: dim)
        )
    }
}

/// Complete weights of the Subword-Tokenized Small Conditional Griffin Psychological State Language Model.
public struct PsychologicalStateLMWeights: Sendable, Equatable {
    public let vocabSize: Int
    public let dim: Int
    public let numHeads: Int
    public let windowSize: Int
    public let numCondTokens: Int
    public let condRawDim: Int
    public let maxSeqLen: Int
    public let serializedSubwordVocabulary: String
    public var auxiliaryHeads: PsychAuxiliaryHeads

    public var condMean: [Float]
    public var condStd: [Float]
    public var wCondIn: [Float]
    public var bCondIn: [Float]
    public var wCondOut: [Float]
    public var bCondOut: [Float]
    public var condNormGamma: [Float]
    public var condNormBeta: [Float]

    public var tokenEmbedding: [Float]
    public var layers: [PsychDecoderLayerWeights]
    public var finalNormGamma: [Float]
    public var finalNormBeta: [Float]
    public var wReadout: [Float]
    public var bReadout: [Float]

    public var parameterCount: Int {
        let base = wCondIn.count + bCondIn.count + wCondOut.count + bCondOut.count +
                   condNormGamma.count + condNormBeta.count + tokenEmbedding.count +
                   finalNormGamma.count + finalNormBeta.count + wReadout.count + bReadout.count
        return base + layers.reduce(0) { $0 + $1.parameterCount }
    }

    public static func makeRandom(
        vocabSize: Int,
        dim: Int = 160,
        numLayers: Int = 3,
        numHeads: Int = 5,
        windowSize: Int = 64,
        numCondTokens: Int = 6,
        condRawDim: Int = PsychConditionExtractor.rawConditionDim,
        maxSeqLen: Int = 64,
        serializedSubwordVocabulary: String = "",
        auxiliaryHeads: PsychAuxiliaryHeads = PsychAuxiliaryHeads(),
        seed: UInt64 = 2026
    ) -> PsychologicalStateLMWeights {
        precondition(dim % numHeads == 0)
        var rng = SplitMix64(seed: seed)
        let resScale = 1.0 / sqrtf(Float(max(1, 3 * numLayers)))
        let embScale: Float = 0.04
        let cInBound = sqrtf(6.0 / Float(condRawDim + dim))
        let cOutBound = sqrtf(6.0 / Float(dim + (numCondTokens + 1) * dim))
        let readBound = sqrtf(6.0 / Float(dim + vocabSize))

        func randVec(_ count: Int, _ bound: Float) -> [Float] {
            var a = [Float](repeating: 0.0, count: count)
            for i in 0..<count { a[i] = rng.nextSignedUniform(scale: bound) }
            return a
        }

        var builtLayers: [PsychDecoderLayerWeights] = []
        builtLayers.reserveCapacity(numLayers)
        for _ in 0..<numLayers {
            builtLayers.append(PsychDecoderLayerWeights.makeRandom(
                dim: dim, numHeads: numHeads, windowSize: windowSize,
                mlpDim: 2 * dim, residualScale: resScale, rng: &rng
            ))
        }

        return PsychologicalStateLMWeights(
            vocabSize: vocabSize,
            dim: dim,
            numHeads: numHeads,
            windowSize: windowSize,
            numCondTokens: numCondTokens,
            condRawDim: condRawDim,
            maxSeqLen: maxSeqLen,
            serializedSubwordVocabulary: serializedSubwordVocabulary,
            auxiliaryHeads: auxiliaryHeads,
            condMean: [Float](repeating: 0.0, count: condRawDim),
            condStd: [Float](repeating: 1.0, count: condRawDim),
            wCondIn: randVec(condRawDim * dim, cInBound),
            bCondIn: [Float](repeating: 0.0, count: dim),
            wCondOut: randVec(dim * (numCondTokens + 1) * dim, cOutBound),
            bCondOut: [Float](repeating: 0.0, count: (numCondTokens + 1) * dim),
            condNormGamma: [Float](repeating: 1.0, count: dim),
            condNormBeta: [Float](repeating: 0.0, count: dim),
            tokenEmbedding: randVec(vocabSize * dim, embScale),
            layers: builtLayers,
            finalNormGamma: [Float](repeating: 1.0, count: dim),
            finalNormBeta: [Float](repeating: 0.0, count: dim),
            wReadout: randVec(dim * vocabSize, readBound),
            bReadout: [Float](repeating: 0.0, count: vocabSize)
        )
    }

    public func save(to url: URL) throws {
        let d = dim
        let totalCondToks = numCondTokens + 1
        var tensors: [(name: String, tensor: SafeTensors.Tensor)] = [
            ("condMean", .init(shape: [condRawDim], values: condMean)),
            ("condStd", .init(shape: [condRawDim], values: condStd)),
            ("wCondIn", .init(shape: [condRawDim, d], values: wCondIn)),
            ("bCondIn", .init(shape: [d], values: bCondIn)),
            ("wCondOut", .init(shape: [d, totalCondToks * d], values: wCondOut)),
            ("bCondOut", .init(shape: [totalCondToks * d], values: bCondOut)),
            ("condNormGamma", .init(shape: [d], values: condNormGamma)),
            ("condNormBeta", .init(shape: [d], values: condNormBeta)),
            ("tokenEmbedding", .init(shape: [vocabSize, d], values: tokenEmbedding)),
            ("finalNormGamma", .init(shape: [d], values: finalNormGamma)),
            ("finalNormBeta", .init(shape: [d], values: finalNormBeta)),
            ("wReadout", .init(shape: [d, vocabSize], values: wReadout)),
            ("bReadout", .init(shape: [vocabSize], values: bReadout))
        ]
        for (i, lw) in layers.enumerated() {
            let p = "layers.\(i)"
            let m = lw.block.mlpDim
            tensors.append(contentsOf: [
                ("\(p).norm1Gamma", .init(shape: [d], values: lw.block.norm1Gamma)),
                ("\(p).norm1Beta", .init(shape: [d], values: lw.block.norm1Beta)),
                ("\(p).wGate", .init(shape: [d, d], values: lw.block.wGate)),
                ("\(p).bGate", .init(shape: [d], values: lw.block.bGate)),
                ("\(p).rglru.wA", .init(shape: [d, d], values: lw.block.rglru.wA)),
                ("\(p).rglru.bA", .init(shape: [d], values: lw.block.rglru.bA)),
                ("\(p).rglru.wX", .init(shape: [d, d], values: lw.block.rglru.wX)),
                ("\(p).rglru.bX", .init(shape: [d], values: lw.block.rglru.bX)),
                ("\(p).rglru.lambda", .init(shape: [d], values: lw.block.rglru.lambda)),
                ("\(p).wOut", .init(shape: [d, d], values: lw.block.wOut)),
                ("\(p).bOut", .init(shape: [d], values: lw.block.bOut)),
                ("\(p).la.normGamma", .init(shape: [d], values: lw.block.localAttn.normGamma)),
                ("\(p).la.normBeta", .init(shape: [d], values: lw.block.localAttn.normBeta)),
                ("\(p).la.wQ", .init(shape: [d, d], values: lw.block.localAttn.wQ)),
                ("\(p).la.bQ", .init(shape: [d], values: lw.block.localAttn.bQ)),
                ("\(p).la.wK", .init(shape: [d, d], values: lw.block.localAttn.wK)),
                ("\(p).la.bK", .init(shape: [d], values: lw.block.localAttn.bK)),
                ("\(p).la.wV", .init(shape: [d, d], values: lw.block.localAttn.wV)),
                ("\(p).la.bV", .init(shape: [d], values: lw.block.localAttn.bV)),
                ("\(p).la.wOut", .init(shape: [d, d], values: lw.block.localAttn.wOut)),
                ("\(p).la.bOut", .init(shape: [d], values: lw.block.localAttn.bOut)),
                ("\(p).norm2Gamma", .init(shape: [d], values: lw.block.norm2Gamma)),
                ("\(p).norm2Beta", .init(shape: [d], values: lw.block.norm2Beta)),
                ("\(p).wMLPUp", .init(shape: [d, m], values: lw.block.wMLPUp)),
                ("\(p).bMLPUp", .init(shape: [m], values: lw.block.bMLPUp)),
                ("\(p).wMLPDown", .init(shape: [m, d], values: lw.block.wMLPDown)),
                ("\(p).bMLPDown", .init(shape: [d], values: lw.block.bMLPDown)),
                ("\(p).wH0", .init(shape: [d, d], values: lw.wH0)),
                ("\(p).bH0", .init(shape: [d], values: lw.bH0)),
                ("\(p).wAdaGamma", .init(shape: [d, d], values: lw.wAdaGamma)),
                ("\(p).bAdaGamma", .init(shape: [d], values: lw.bAdaGamma)),
                ("\(p).wAdaBeta", .init(shape: [d, d], values: lw.wAdaBeta)),
                ("\(p).bAdaBeta", .init(shape: [d], values: lw.bAdaBeta)),
                ("\(p).crossNormGamma", .init(shape: [d], values: lw.crossNormGamma)),
                ("\(p).crossNormBeta", .init(shape: [d], values: lw.crossNormBeta)),
                ("\(p).wCrossQ", .init(shape: [d, d], values: lw.wCrossQ)),
                ("\(p).bCrossQ", .init(shape: [d], values: lw.bCrossQ)),
                ("\(p).wCrossK", .init(shape: [d, d], values: lw.wCrossK)),
                ("\(p).bCrossK", .init(shape: [d], values: lw.bCrossK)),
                ("\(p).wCrossV", .init(shape: [d, d], values: lw.wCrossV)),
                ("\(p).bCrossV", .init(shape: [d], values: lw.bCrossV)),
                ("\(p).wCrossOut", .init(shape: [d, d], values: lw.wCrossOut)),
                ("\(p).bCrossOut", .init(shape: [d], values: lw.bCrossOut))
            ])
        }
        let meta: [String: String] = [
            "vocabSize": String(vocabSize),
            "dim": String(dim),
            "numLayers": String(layers.count),
            "numHeads": String(numHeads),
            "windowSize": String(windowSize),
            "numCondTokens": String(numCondTokens),
            "condRawDim": String(condRawDim),
            "maxSeqLen": String(maxSeqLen),
            "serializedSubwordVocabulary": serializedSubwordVocabulary,
            "auxiliaryHeads": auxiliaryHeads.serialize()
        ]
        try SafeTensors.write(tensors, metadata: meta, to: url)
    }

    public static func load(from url: URL) throws -> PsychologicalStateLMWeights {
        let (tensors, meta) = try SafeTensors.read(from: url)
        func req(_ key: String) throws -> [Float] {
            guard let t = tensors[key] else { throw SafeTensors.Failure.badShape(key) }
            return t.values
        }
        let vocabSize = Int(meta["vocabSize"] ?? "") ?? 0
        let dim = Int(meta["dim"] ?? "") ?? 160
        let numLayers = Int(meta["numLayers"] ?? "") ?? 3
        let numHeads = Int(meta["numHeads"] ?? "") ?? 5
        let windowSize = Int(meta["windowSize"] ?? "") ?? 64
        let numCondTokens = Int(meta["numCondTokens"] ?? "") ?? 6
        let condRawDim = Int(meta["condRawDim"] ?? "") ?? PsychConditionExtractor.rawConditionDim
        let maxSeqLen = Int(meta["maxSeqLen"] ?? "") ?? 64
        let vocabStr = meta["serializedSubwordVocabulary"] ?? ""
        let auxHeads = PsychAuxiliaryHeads.deserialize(meta["auxiliaryHeads"] ?? "")

        var builtLayers: [PsychDecoderLayerWeights] = []
        builtLayers.reserveCapacity(numLayers)
        for i in 0..<numLayers {
            let p = "layers.\(i)"
            let bMLPUp = try req("\(p).bMLPUp")
            let la = LocalAttentionWeights(
                dim: dim,
                numHeads: numHeads,
                windowSize: windowSize,
                normGamma: try req("\(p).la.normGamma"),
                normBeta: try req("\(p).la.normBeta"),
                wQ: try req("\(p).la.wQ"),
                bQ: try req("\(p).la.bQ"),
                wK: try req("\(p).la.wK"),
                bK: try req("\(p).la.bK"),
                wV: try req("\(p).la.wV"),
                bV: try req("\(p).la.bV"),
                wOut: try req("\(p).la.wOut"),
                bOut: try req("\(p).la.bOut")
            )
            let rg = RGLRUWeights(
                dim: dim,
                wA: try req("\(p).rglru.wA"),
                bA: try req("\(p).rglru.bA"),
                wX: try req("\(p).rglru.wX"),
                bX: try req("\(p).rglru.bX"),
                lambda: try req("\(p).rglru.lambda")
            )
            let hb = HawkBlockWeights(
                dim: dim,
                mlpDim: bMLPUp.count,
                norm1Gamma: try req("\(p).norm1Gamma"),
                norm1Beta: try req("\(p).norm1Beta"),
                wGate: try req("\(p).wGate"),
                bGate: try req("\(p).bGate"),
                rglru: rg,
                wOut: try req("\(p).wOut"),
                bOut: try req("\(p).bOut"),
                localAttn: la,
                norm2Gamma: try req("\(p).norm2Gamma"),
                norm2Beta: try req("\(p).norm2Beta"),
                wMLPUp: try req("\(p).wMLPUp"),
                bMLPUp: bMLPUp,
                wMLPDown: try req("\(p).wMLPDown"),
                bMLPDown: try req("\(p).bMLPDown")
            )
            builtLayers.append(PsychDecoderLayerWeights(
                dim: dim,
                numHeads: numHeads,
                block: hb,
                wH0: try req("\(p).wH0"),
                bH0: try req("\(p).bH0"),
                wAdaGamma: try req("\(p).wAdaGamma"),
                bAdaGamma: try req("\(p).bAdaGamma"),
                wAdaBeta: try req("\(p).wAdaBeta"),
                bAdaBeta: try req("\(p).bAdaBeta"),
                crossNormGamma: try req("\(p).crossNormGamma"),
                crossNormBeta: try req("\(p).crossNormBeta"),
                wCrossQ: try req("\(p).wCrossQ"),
                bCrossQ: try req("\(p).bCrossQ"),
                wCrossK: try req("\(p).wCrossK"),
                bCrossK: try req("\(p).bCrossK"),
                wCrossV: try req("\(p).wCrossV"),
                bCrossV: try req("\(p).bCrossV"),
                wCrossOut: try req("\(p).wCrossOut"),
                bCrossOut: try req("\(p).bCrossOut")
            ))
        }

        return PsychologicalStateLMWeights(
            vocabSize: vocabSize,
            dim: dim,
            numHeads: numHeads,
            windowSize: windowSize,
            numCondTokens: numCondTokens,
            condRawDim: condRawDim,
            maxSeqLen: maxSeqLen,
            serializedSubwordVocabulary: vocabStr,
            auxiliaryHeads: auxHeads,
            condMean: try req("condMean"),
            condStd: try req("condStd"),
            wCondIn: try req("wCondIn"),
            bCondIn: try req("bCondIn"),
            wCondOut: try req("wCondOut"),
            bCondOut: try req("bCondOut"),
            condNormGamma: try req("condNormGamma"),
            condNormBeta: try req("condNormBeta"),
            tokenEmbedding: try req("tokenEmbedding"),
            layers: builtLayers,
            finalNormGamma: try req("finalNormGamma"),
            finalNormBeta: try req("finalNormBeta"),
            wReadout: try req("wReadout"),
            bReadout: try req("bReadout")
        )
    }
}

/// Pure-CPU AVX2/SIMD8 Subword-Tokenized Conditional Griffin Language Model (`~15–25 ms` total inference).
public final class PsychologicalStateLM: @unchecked Sendable {
    public let weights: PsychologicalStateLMWeights
    public let vocabulary: SubwordVocabulary
    private let pieceCharCount: [Int]
    private let pieceEndsWithPeriod: [Bool]
    private let isPenalizableToken: [Bool]
    private let pieceStemKey: [UInt64]
    private let pieceHasPartnerWord: [Bool]

    public init(weights: PsychologicalStateLMWeights) {
        self.weights = weights
        let vocab = SubwordVocabulary.deserialize(weights.serializedSubwordVocabulary)
        self.vocabulary = vocab
        let vSize = weights.vocabSize
        var charCounts = [Int](repeating: 0, count: vSize)
        var endsPeriod = [Bool](repeating: false, count: vSize)
        var penalizable = [Bool](repeating: false, count: vSize)
        var stemKeys = [UInt64](repeating: 0, count: vSize)
        var hasPartner = [Bool](repeating: false, count: vSize)
        let exempt: Set<String> = ["相手", "おり、", "です。", "ます。", "います。", "ています。", "ながら、", "ものの、", "つつも、"]
        for id in TextVocabulary.specialCount..<min(vSize, vocab.idToPiece.count) {
            let p = vocab.idToPiece[id]
            let cLen = p.count
            charCounts[id] = cLen
            endsPeriod[id] = p.hasSuffix("。")
            hasPartner[id] = p.contains("相手")
            if !exempt.contains(p) {
                let scalars = Array(p.unicodeScalars)
                var hasCJK = false
                for idx in 0..<scalars.count {
                    let v = scalars[idx].value
                    if (0x30A0...0x30FF).contains(v) || (0x4E00...0x9FFF).contains(v) {
                        hasCJK = true
                        if stemKeys[id] == 0 && idx + 1 < scalars.count {
                            let v2 = scalars[idx + 1].value
                            let pairStr = String(scalars[idx]) + String(scalars[idx + 1])
                            if pairStr != "相手" && pairStr != "距離" && v2 != 0x3001 && v2 != 0x3002 {
                                stemKeys[id] = (UInt64(v) << 32) | UInt64(v2)
                            }
                        }
                    }
                }
                penalizable[id] = hasCJK || cLen >= 3
            }
        }
        self.pieceCharCount = charCounts
        self.pieceEndsWithPeriod = endsPeriod
        self.isPenalizableToken = penalizable
        self.pieceStemKey = stemKeys
        self.pieceHasPartnerWord = hasPartner
    }

    public func projectCondition(condRaw: [Float], batch: Int) -> (cGlobal: [Float], condMem: [Float]) {
        let cDim = weights.condRawDim
        let d = weights.dim
        let mTok = weights.numCondTokens
        let totalToks = mTok + 1
        precondition(condRaw.count == batch * cDim)

        var normCond = [Float](repeating: 0.0, count: batch * cDim)
        for b in 0..<batch {
            let off = b * cDim
            for j in 0..<cDim {
                let z = (condRaw[off + j] - weights.condMean[j]) / max(weights.condStd[j], 1e-4)
                normCond[off + j] = max(-6.0, min(6.0, z))
            }
        }
        let hPre = SWTHawkAutoencoder.linear(
            x: normCond, w: weights.wCondIn, b: weights.bCondIn, rows: batch, inDim: cDim, outDim: d
        )
        let hAct = SWTHawkAutoencoder.silu(hPre)
        let projRaw = SWTHawkAutoencoder.linear(
            x: hAct, w: weights.wCondOut, b: weights.bCondOut, rows: batch, inDim: d, outDim: totalToks * d
        )
        let projNorm = SWTHawkAutoencoder.layerNorm(
            x: projRaw, gamma: weights.condNormGamma, beta: weights.condNormBeta, rows: batch * totalToks, dim: d
        )

        var cGlobal = [Float](repeating: 0.0, count: batch * d)
        var condMem = [Float](repeating: 0.0, count: batch * mTok * d)
        for b in 0..<batch {
            let srcBase = b * totalToks * d
            for c in 0..<d { cGlobal[b * d + c] = projNorm[srcBase + c] }
            let dstMem = b * mTok * d
            for k in 0..<(mTok * d) { condMem[dstMem + k] = projNorm[srcBase + d + k] }
        }
        return (cGlobal, condMem)
    }

    private func crossAttendCondition(
        x: [Float],
        condMem: [Float],
        lw: PsychDecoderLayerWeights,
        batch: Int,
        time: Int
    ) -> [Float] {
        let d = weights.dim
        let mTok = weights.numCondTokens
        let nH = weights.numHeads
        let hDim = d / nH
        let rows = batch * time
        let scale = 1.0 / sqrtf(Float(max(1, hDim)))

        let uCross = SWTHawkAutoencoder.layerNorm(
            x: x, gamma: lw.crossNormGamma, beta: lw.crossNormBeta, rows: rows, dim: d
        )
        let q = SWTHawkAutoencoder.linear(x: uCross, w: lw.wCrossQ, b: lw.bCrossQ, rows: rows, inDim: d, outDim: d)
        let k = SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossK, b: lw.bCrossK, rows: batch * mTok, inDim: d, outDim: d)
        let v = SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossV, b: lw.bCrossV, rows: batch * mTok, inDim: d, outDim: d)

        var ctx = [Float](repeating: 0.0, count: rows * d)
        var sc = [Float](repeating: 0.0, count: mTok)
        for b in 0..<batch {
            let kvBase = b * mTok * d
            for t in 0..<time {
                let qBase = (b * time + t) * d
                for h in 0..<nH {
                    let hOff = h * hDim
                    var maxS: Float = -.greatestFiniteMagnitude
                    for m in 0..<mTok {
                        var dot: Float = 0.0
                        let kOff = kvBase + m * d + hOff
                        for c in 0..<hDim { dot += q[qBase + hOff + c] * k[kOff + c] }
                        let s = dot * scale
                        sc[m] = s
                        if s > maxS { maxS = s }
                    }
                    var sumE: Float = 0.0
                    for m in 0..<mTok {
                        let e = expf(sc[m] - maxS)
                        sc[m] = e
                        sumE += e
                    }
                    let inv = 1.0 / max(sumE, 1e-12)
                    for m in 0..<mTok {
                        let p = sc[m] * inv
                        let vOff = kvBase + m * d + hOff
                        for c in 0..<hDim { ctx[qBase + hOff + c] += p * v[vOff + c] }
                    }
                }
            }
        }
        let crossOut = SWTHawkAutoencoder.linear(x: ctx, w: lw.wCrossOut, b: lw.bCrossOut, rows: rows, inDim: d, outDim: d)
        var out = [Float](repeating: 0.0, count: rows * d)
        for i in 0..<(rows * d) { out[i] = x[i] + crossOut[i] }
        return out
    }

    public func forwardTeacherForced(
        condRaw: [Float],
        prevTokens: [Int],
        batch: Int,
        time: Int
    ) -> [Float] {
        let d = weights.dim
        let vSize = weights.vocabSize
        let rows = batch * time
        precondition(prevTokens.count == rows)

        let (cGlobal, condMem) = projectCondition(condRaw: condRaw, batch: batch)
        let pe = HawkConditionalDecoder.positionalEncoding(time: time, dim: d)

        let embScale: Float = 2.5
        var x = [Float](repeating: 0.0, count: rows * d)
        for b in 0..<batch {
            for t in 0..<time {
                let rIdx = b * time + t
                let tok = max(0, min(vSize - 1, prevTokens[rIdx]))
                let eOff = tok * d
                let peOff = t * d
                let dst = rIdx * d
                for c in 0..<d {
                    x[dst + c] = weights.tokenEmbedding[eOff + c] * embScale + pe[peOff + c]
                }
            }
        }

        for lw in weights.layers {
            let h0 = SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wH0, b: lw.bH0, rows: batch, inDim: d, outDim: d)
            let adaG = SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaGamma, b: lw.bAdaGamma, rows: batch, inDim: d, outDim: d)
            let adaB = SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaBeta, b: lw.bAdaBeta, rows: batch, inDim: d, outDim: d)

            let norm1 = SWTHawkAutoencoder.layerNorm(x: x, gamma: lw.block.norm1Gamma, beta: lw.block.norm1Beta, rows: rows, dim: d)
            var u1 = [Float](repeating: 0.0, count: rows * d)
            for b in 0..<batch {
                let gOff = b * d
                for t in 0..<time {
                    let rOff = (b * time + t) * d
                    for c in 0..<d {
                        u1[rOff + c] = (1.0 + adaG[gOff + c]) * norm1[rOff + c] + adaB[gOff + c]
                    }
                }
            }
            let gate = SWTHawkAutoencoder.silu(
                SWTHawkAutoencoder.linear(x: u1, w: lw.block.wGate, b: lw.block.bGate, rows: rows, inDim: d, outDim: d)
            )
            let (hRec, _) = RGLRULayer.forward(x: u1, weights: lw.block.rglru, h0: h0, batch: batch, time: time)
            var gatedH = [Float](repeating: 0.0, count: rows * d)
            for i in 0..<(rows * d) { gatedH[i] = gate[i] * hRec[i] }
            let recOut = SWTHawkAutoencoder.linear(x: gatedH, w: lw.block.wOut, b: lw.block.bOut, rows: rows, inDim: d, outDim: d)
            var x1 = [Float](repeating: 0.0, count: rows * d)
            for i in 0..<(rows * d) { x1[i] = x[i] + recOut[i] }

            let x2 = LocalSlidingWindowAttention.forward(x: x1, weights: lw.block.localAttn, batch: batch, time: time)
            let x3 = crossAttendCondition(x: x2, condMem: condMem, lw: lw, batch: batch, time: time)

            let norm2 = SWTHawkAutoencoder.layerNorm(x: x3, gamma: lw.block.norm2Gamma, beta: lw.block.norm2Beta, rows: rows, dim: d)
            let mlpAct = SWTHawkAutoencoder.silu(
                SWTHawkAutoencoder.linear(x: norm2, w: lw.block.wMLPUp, b: lw.block.bMLPUp, rows: rows, inDim: d, outDim: lw.block.mlpDim)
            )
            let mlpDown = SWTHawkAutoencoder.linear(x: mlpAct, w: lw.block.wMLPDown, b: lw.block.bMLPDown, rows: rows, inDim: lw.block.mlpDim, outDim: d)
            for i in 0..<(rows * d) { x[i] = x3[i] + mlpDown[i] }
        }

        let finalNorm = SWTHawkAutoencoder.layerNorm(x: x, gamma: weights.finalNormGamma, beta: weights.finalNormBeta, rows: rows, dim: d)
        return SWTHawkAutoencoder.linear(x: finalNorm, w: weights.wReadout, b: weights.bReadout, rows: rows, inDim: d, outDim: vSize)
    }

    /// Fast KV-cached single-thread pure-CPU subword generator (`~8–15 ms` for ~40 subwords = ~115 chars).
    public func generateTokens(
        condRaw: [Float],
        maxLength: Int? = nil,
        minCharLength: Int = 72,
        repetitionPenalty: Float = 1.35,
        temperature: Float = 0.0,
        seed: UInt64 = 42
    ) -> [Int] {
        let d = weights.dim
        let vSize = weights.vocabSize
        let nLayers = weights.layers.count
        let nHeads = weights.numHeads
        let hDim = d / nHeads
        let halfH = hDim / 2
        let winSize = weights.windowSize
        let mTok = weights.numCondTokens
        let maxT = max(8, min(maxLength ?? weights.maxSeqLen, 128))
        let attnScale = 1.0 / sqrtf(Float(max(1, hDim)))

        let (cGlobal, condMem) = projectCondition(condRaw: condRaw, batch: 1)
        let pe = HawkConditionalDecoder.positionalEncoding(time: maxT, dim: d)
        let (cosTable, sinTable) = LocalSlidingWindowAttention.ropeTables(time: maxT, headDim: hDim)

        var hStates: [[Float]] = []
        var adaGammas: [[Float]] = []
        var adaBetas: [[Float]] = []
        var condKs: [[Float]] = []
        var condVs: [[Float]] = []
        hStates.reserveCapacity(nLayers)
        adaGammas.reserveCapacity(nLayers)
        adaBetas.reserveCapacity(nLayers)
        condKs.reserveCapacity(nLayers)
        condVs.reserveCapacity(nLayers)

        for lw in weights.layers {
            hStates.append(SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wH0, b: lw.bH0, rows: 1, inDim: d, outDim: d))
            adaGammas.append(SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaGamma, b: lw.bAdaGamma, rows: 1, inDim: d, outDim: d))
            adaBetas.append(SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaBeta, b: lw.bAdaBeta, rows: 1, inDim: d, outDim: d))
            condKs.append(SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossK, b: lw.bCrossK, rows: mTok, inDim: d, outDim: d))
            condVs.append(SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossV, b: lw.bCrossV, rows: mTok, inDim: d, outDim: d))
        }

        var kvCaches = [[Float]](repeating: [], count: nLayers)
        for l in 0..<nLayers { kvCaches[l].reserveCapacity(winSize * 2 * d) }

        var rng = SplitMix64(seed: seed)
        var prevTok = TextVocabulary.bosId
        var generated: [Int] = []
        generated.reserveCapacity(maxT)
        var emittedCharCount = 0
        var seenSubwordBigrams: Set<UInt64> = []
        var seenContentTokens: [Int: Int] = [:]

        var xStep = [Float](repeating: 0.0, count: d)
        var u1 = [Float](repeating: 0.0, count: d)
        var gatedH = [Float](repeating: 0.0, count: d)
        var x1 = [Float](repeating: 0.0, count: d)
        var qRot = [Float](repeating: 0.0, count: d)
        var kRot = [Float](repeating: 0.0, count: d)
        var scores = [Float](repeating: 0.0, count: max(winSize, mTok))

        let embScale: Float = 2.5
        for t in 0..<maxT {
            let embOff = max(0, min(vSize - 1, prevTok)) * d
            let peOff = t * d
            for c in 0..<d {
                xStep[c] = weights.tokenEmbedding[embOff + c] * embScale + pe[peOff + c]
            }

            let ropeOff = t * halfH
            for l in 0..<nLayers {
                let lw = weights.layers[l]
                let adaG = adaGammas[l]
                let adaB = adaBetas[l]

                let norm1 = SWTHawkAutoencoder.layerNorm(x: xStep, gamma: lw.block.norm1Gamma, beta: lw.block.norm1Beta, rows: 1, dim: d)
                for c in 0..<d { u1[c] = (1.0 + adaG[c]) * norm1[c] + adaB[c] }
                let gate = SWTHawkAutoencoder.silu(
                    SWTHawkAutoencoder.linear(x: u1, w: lw.block.wGate, b: lw.block.bGate, rows: 1, inDim: d, outDim: d)
                )
                let (_, hNext) = RGLRULayer.forward(x: u1, weights: lw.block.rglru, h0: hStates[l], batch: 1, time: 1)
                hStates[l] = hNext
                for c in 0..<d { gatedH[c] = gate[c] * hNext[c] }
                let recOut = SWTHawkAutoencoder.linear(x: gatedH, w: lw.block.wOut, b: lw.block.bOut, rows: 1, inDim: d, outDim: d)
                for c in 0..<d { x1[c] = xStep[c] + recOut[c] }

                let la = lw.block.localAttn
                let normLA = SWTHawkAutoencoder.layerNorm(x: x1, gamma: la.normGamma, beta: la.normBeta, rows: 1, dim: d)
                let qRaw = SWTHawkAutoencoder.linear(x: normLA, w: la.wQ, b: la.bQ, rows: 1, inDim: d, outDim: d)
                let kRaw = SWTHawkAutoencoder.linear(x: normLA, w: la.wK, b: la.bK, rows: 1, inDim: d, outDim: d)
                let vStep = SWTHawkAutoencoder.linear(x: normLA, w: la.wV, b: la.bV, rows: 1, inDim: d, outDim: d)

                for h in 0..<nHeads {
                    let hOff = h * hDim
                    for i in 0..<halfH {
                        let cs = cosTable[ropeOff + i], sn = sinTable[ropeOff + i]
                        let q1 = qRaw[hOff + i], q2 = qRaw[hOff + halfH + i]
                        qRot[hOff + i] = q1 * cs - q2 * sn
                        qRot[hOff + halfH + i] = q1 * sn + q2 * cs
                        let k1 = kRaw[hOff + i], k2 = kRaw[hOff + halfH + i]
                        kRot[hOff + i] = k1 * cs - k2 * sn
                        kRot[hOff + halfH + i] = k1 * sn + k2 * cs
                    }
                }
                kvCaches[l].append(contentsOf: kRot)
                kvCaches[l].append(contentsOf: vStep)
                let curSteps = kvCaches[l].count / (2 * d)
                if curSteps > winSize {
                    kvCaches[l].removeFirst((curSteps - winSize) * 2 * d)
                }
                let wSteps = kvCaches[l].count / (2 * d)

                var ctxLA = [Float](repeating: 0.0, count: d)
                kvCaches[l].withUnsafeBufferPointer { kvBuf in
                    qRot.withUnsafeBufferPointer { qrBuf in
                        ctxLA.withUnsafeMutableBufferPointer { ctxBuf in
                            let kvp = kvBuf.baseAddress!, qrp = qrBuf.baseAddress!, cp = ctxBuf.baseAddress!
                            for h in 0..<nHeads {
                                let hOff = h * hDim
                                var maxS: Float = -.greatestFiniteMagnitude
                                for idx in 0..<wSteps {
                                    let dot = VectorOperations.dotProduct(a: qrp + hOff, b: kvp + idx * 2 * d + hOff, count: hDim)
                                    let s = dot * attnScale
                                    scores[idx] = s
                                    if s > maxS { maxS = s }
                                }
                                var sumE: Float = 0.0
                                for idx in 0..<wSteps {
                                    let e = expf(scores[idx] - maxS)
                                    scores[idx] = e
                                    sumE += e
                                }
                                let inv = 1.0 / max(sumE, 1e-12)
                                for idx in 0..<wSteps {
                                    let p = scores[idx] * inv
                                    let vp = kvp + idx * 2 * d + d + hOff
                                    for c in 0..<hDim { cp[hOff + c] += p * vp[c] }
                                }
                            }
                        }
                    }
                }
                let laOut = SWTHawkAutoencoder.linear(x: ctxLA, w: la.wOut, b: la.bOut, rows: 1, inDim: d, outDim: d)
                var x2 = [Float](repeating: 0.0, count: d)
                for c in 0..<d { x2[c] = x1[c] + laOut[c] }

                let uCross = SWTHawkAutoencoder.layerNorm(x: x2, gamma: lw.crossNormGamma, beta: lw.crossNormBeta, rows: 1, dim: d)
                let qCross = SWTHawkAutoencoder.linear(x: uCross, w: lw.wCrossQ, b: lw.bCrossQ, rows: 1, inDim: d, outDim: d)
                var ctxCross = [Float](repeating: 0.0, count: d)
                condKs[l].withUnsafeBufferPointer { ckBuf in
                    condVs[l].withUnsafeBufferPointer { cvBuf in
                        qCross.withUnsafeBufferPointer { qcBuf in
                            ctxCross.withUnsafeMutableBufferPointer { ccBuf in
                                let ckp = ckBuf.baseAddress!, cvp = cvBuf.baseAddress!
                                let qcp = qcBuf.baseAddress!, ccp = ccBuf.baseAddress!
                                for h in 0..<nHeads {
                                    let hOff = h * hDim
                                    var maxS: Float = -.greatestFiniteMagnitude
                                    for m in 0..<mTok {
                                        let dot = VectorOperations.dotProduct(a: qcp + hOff, b: ckp + m * d + hOff, count: hDim)
                                        let s = dot * attnScale
                                        scores[m] = s
                                        if s > maxS { maxS = s }
                                    }
                                    var sumE: Float = 0.0
                                    for m in 0..<mTok {
                                        let e = expf(scores[m] - maxS)
                                        scores[m] = e
                                        sumE += e
                                    }
                                    let inv = 1.0 / max(sumE, 1e-12)
                                    for m in 0..<mTok {
                                        let p = scores[m] * inv
                                        let vp = cvp + m * d + hOff
                                        for c in 0..<hDim { ccp[hOff + c] += p * vp[c] }
                                    }
                                }
                            }
                        }
                    }
                }
                let crossOut = SWTHawkAutoencoder.linear(x: ctxCross, w: lw.wCrossOut, b: lw.bCrossOut, rows: 1, inDim: d, outDim: d)
                var x3 = [Float](repeating: 0.0, count: d)
                for c in 0..<d { x3[c] = x2[c] + crossOut[c] }

                let norm2 = SWTHawkAutoencoder.layerNorm(x: x3, gamma: lw.block.norm2Gamma, beta: lw.block.norm2Beta, rows: 1, dim: d)
                let mlpAct = SWTHawkAutoencoder.silu(
                    SWTHawkAutoencoder.linear(x: norm2, w: lw.block.wMLPUp, b: lw.block.bMLPUp, rows: 1, inDim: d, outDim: lw.block.mlpDim)
                )
                let mlpDown = SWTHawkAutoencoder.linear(x: mlpAct, w: lw.block.wMLPDown, b: lw.block.bMLPDown, rows: 1, inDim: lw.block.mlpDim, outDim: d)
                for c in 0..<d { xStep[c] = x3[c] + mlpDown[c] }
            }

            let finalNorm = SWTHawkAutoencoder.layerNorm(x: xStep, gamma: weights.finalNormGamma, beta: weights.finalNormBeta, rows: 1, dim: d)
            var logits = SWTHawkAutoencoder.linear(x: finalNorm, w: weights.wReadout, b: weights.bReadout, rows: 1, inDim: d, outDim: vSize)

            logits[TextVocabulary.padId] = -1e9
            logits[TextVocabulary.bosId] = -1e9
            logits[TextVocabulary.unkId] = -1e9
            if emittedCharCount < minCharLength {
                logits[TextVocabulary.eosId] = -1e9
            }

            // Penalize repeated content subwords & forbid immediate self-repeats / repeated bigrams
            if repetitionPenalty > 1.001 {
                for (tok, cnt) in seenContentTokens {
                    let pen = Float(cnt) * (repetitionPenalty - 1.0) * 3.5
                    logits[tok] -= pen
                }
                if let last = generated.last {
                    if last < vSize { logits[last] = -1e9 }
                    let prefix = UInt64(UInt32(last)) << 32
                    for cand in TextVocabulary.specialCount..<vSize {
                        if seenSubwordBigrams.contains(prefix | UInt64(UInt32(cand))) {
                            logits[cand] = -1e9
                        }
                    }
                }
            }

            let nextId: Int
            if temperature <= 1e-4 {
                var best = 0
                for v in 1..<vSize where logits[v] > logits[best] { best = v }
                nextId = best
            } else {
                let invTemp = 1.0 / temperature
                var maxL: Float = -.greatestFiniteMagnitude
                for v in 0..<vSize {
                    logits[v] *= invTemp
                    if logits[v] > maxL { maxL = logits[v] }
                }
                var sumP: Float = 0.0
                for v in 0..<vSize {
                    let e = expf(logits[v] - maxL)
                    logits[v] = e
                    sumP += e
                }
                var r = rng.nextUnitFloat() * sumP
                var chosen = TextVocabulary.eosId
                for v in 0..<vSize {
                    r -= logits[v]
                    if r <= 0.0 { chosen = v; break }
                }
                nextId = chosen
            }

            if nextId == TextVocabulary.eosId {
                break
            }
            if let last = generated.last {
                seenSubwordBigrams.insert((UInt64(UInt32(last)) << 32) | UInt64(UInt32(nextId)))
            }
            if nextId < pieceCharCount.count {
                emittedCharCount += pieceCharCount[nextId]
                if isPenalizableToken[nextId] {
                    seenContentTokens[nextId, default: 0] += 1
                }
            }
            generated.append(nextId)
            prevTok = nextId
        }
        return generated
    }

    /// AVX2-batched (`rows = B`) KV-cached Beam Search enforcing clean 2-sentence (`...。...。`) Japanese psychological prose.
    public func generateBeam(
        condRaw: [Float],
        beamWidth: Int = 2,
        maxLength: Int? = nil,
        minCharLength: Int = 68,
        repetitionPenalty: Float = 1.30
    ) -> [Int] {
        if beamWidth <= 1 {
            return generateTokens(condRaw: condRaw, maxLength: maxLength, minCharLength: minCharLength, repetitionPenalty: repetitionPenalty, temperature: 0.0)
        }
        let d = weights.dim
        let vSize = weights.vocabSize
        let nLayers = weights.layers.count
        let nHeads = weights.numHeads
        let hDim = d / nHeads
        let halfH = hDim / 2
        let winSize = weights.windowSize
        let mTok = weights.numCondTokens
        let maxT = max(12, min(maxLength ?? weights.maxSeqLen, 128))
        let attnScale = 1.0 / sqrtf(Float(max(1, hDim)))
        let kW = min(8, max(2, beamWidth))

        let (cGlobal, condMem) = projectCondition(condRaw: condRaw, batch: 1)
        let pe = HawkConditionalDecoder.positionalEncoding(time: maxT, dim: d)
        let (cosTable, sinTable) = LocalSlidingWindowAttention.ropeTables(time: maxT, headDim: hDim)

        var baseH0: [[Float]] = []
        var adaGammas: [[Float]] = []
        var adaBetas: [[Float]] = []
        var condKs: [[Float]] = []
        var condVs: [[Float]] = []
        baseH0.reserveCapacity(nLayers)
        adaGammas.reserveCapacity(nLayers)
        adaBetas.reserveCapacity(nLayers)
        condKs.reserveCapacity(nLayers)
        condVs.reserveCapacity(nLayers)

        for lw in weights.layers {
            baseH0.append(SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wH0, b: lw.bH0, rows: 1, inDim: d, outDim: d))
            adaGammas.append(SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaGamma, b: lw.bAdaGamma, rows: 1, inDim: d, outDim: d))
            adaBetas.append(SWTHawkAutoencoder.linear(x: cGlobal, w: lw.wAdaBeta, b: lw.bAdaBeta, rows: 1, inDim: d, outDim: d))
            condKs.append(SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossK, b: lw.bCrossK, rows: mTok, inDim: d, outDim: d))
            condVs.append(SWTHawkAutoencoder.linear(x: condMem, w: lw.wCrossV, b: lw.bCrossV, rows: mTok, inDim: d, outDim: d))
        }

        struct BeamState {
            var tokens: [Int]
            var logProb: Float
            var penalizedScore: Float
            var charCount: Int
            var periodCount: Int
            var hStates: [[Float]]
            var kvCaches: [[Float]]
            var seenBigrams: Set<UInt64>
            var seenCounts: [Int: Int]
            var seenStems: Set<UInt64>
            var usedPartnerInClause: Bool
        }

        var emptyKV = [[Float]](repeating: [], count: nLayers)
        for l in 0..<nLayers { emptyKV[l].reserveCapacity(winSize * 2 * d) }

        var beams: [BeamState] = [
            BeamState(
                tokens: [],
                logProb: 0.0,
                penalizedScore: 0.0,
                charCount: 0,
                periodCount: 0,
                hStates: baseH0,
                kvCaches: emptyKV,
                seenBigrams: [],
                seenCounts: [:],
                seenStems: [],
                usedPartnerInClause: false
            )
        ]
        var completed: [(tokens: [Int], normScore: Float)] = []
        var scores = [Float](repeating: 0.0, count: max(winSize, mTok))
        let embScale: Float = 2.5

        for t in 0..<maxT {
            let B = beams.count
            if B == 0 { break }

            var xBatch = [Float](repeating: 0.0, count: B * d)
            let peOff = t * d
            for b in 0..<B {
                let prevTok = beams[b].tokens.last ?? TextVocabulary.bosId
                let embOff = max(0, min(vSize - 1, prevTok)) * d
                let dst = b * d
                for c in 0..<d {
                    xBatch[dst + c] = weights.tokenEmbedding[embOff + c] * embScale + pe[peOff + c]
                }
            }

            let ropeOff = t * halfH
            var nextHStates = Array(repeating: Array(repeating: [Float](), count: nLayers), count: B)
            var nextKVCaches = Array(repeating: Array(repeating: [Float](), count: nLayers), count: B)

            for l in 0..<nLayers {
                let lw = weights.layers[l]
                let adaG = adaGammas[l]
                let adaB = adaBetas[l]

                let norm1 = SWTHawkAutoencoder.layerNorm(x: xBatch, gamma: lw.block.norm1Gamma, beta: lw.block.norm1Beta, rows: B, dim: d)
                var u1 = [Float](repeating: 0.0, count: B * d)
                for b in 0..<B {
                    let rOff = b * d
                    for c in 0..<d { u1[rOff + c] = (1.0 + adaG[c]) * norm1[rOff + c] + adaB[c] }
                }
                let gate = SWTHawkAutoencoder.silu(
                    SWTHawkAutoencoder.linear(x: u1, w: lw.block.wGate, b: lw.block.bGate, rows: B, inDim: d, outDim: d)
                )
                var h0Stack = [Float](repeating: 0.0, count: B * d)
                for b in 0..<B {
                    let src = beams[b].hStates[l]
                    for c in 0..<d { h0Stack[b * d + c] = src[c] }
                }
                let (_, hNextStack) = RGLRULayer.forward(x: u1, weights: lw.block.rglru, h0: h0Stack, batch: B, time: 1)
                for b in 0..<B {
                    nextHStates[b][l] = Array(hNextStack[(b * d)..<((b + 1) * d)])
                }
                var gatedH = [Float](repeating: 0.0, count: B * d)
                for i in 0..<(B * d) { gatedH[i] = gate[i] * hNextStack[i] }
                let recOut = SWTHawkAutoencoder.linear(x: gatedH, w: lw.block.wOut, b: lw.block.bOut, rows: B, inDim: d, outDim: d)
                var x1 = [Float](repeating: 0.0, count: B * d)
                for i in 0..<(B * d) { x1[i] = xBatch[i] + recOut[i] }

                let la = lw.block.localAttn
                let normLA = SWTHawkAutoencoder.layerNorm(x: x1, gamma: la.normGamma, beta: la.normBeta, rows: B, dim: d)
                let qRaw = SWTHawkAutoencoder.linear(x: normLA, w: la.wQ, b: la.bQ, rows: B, inDim: d, outDim: d)
                let kRaw = SWTHawkAutoencoder.linear(x: normLA, w: la.wK, b: la.bK, rows: B, inDim: d, outDim: d)
                let vStep = SWTHawkAutoencoder.linear(x: normLA, w: la.wV, b: la.bV, rows: B, inDim: d, outDim: d)

                var ctxLA = [Float](repeating: 0.0, count: B * d)
                var qRot = [Float](repeating: 0.0, count: d)
                var kRot = [Float](repeating: 0.0, count: d)
                for b in 0..<B {
                    let bOff = b * d
                    for h in 0..<nHeads {
                        let hOff = h * hDim
                        for i in 0..<halfH {
                            let cs = cosTable[ropeOff + i], sn = sinTable[ropeOff + i]
                            let q1 = qRaw[bOff + hOff + i], q2 = qRaw[bOff + hOff + halfH + i]
                            qRot[hOff + i] = q1 * cs - q2 * sn
                            qRot[hOff + halfH + i] = q1 * sn + q2 * cs
                            let k1 = kRaw[bOff + hOff + i], k2 = kRaw[bOff + hOff + halfH + i]
                            kRot[hOff + i] = k1 * cs - k2 * sn
                            kRot[hOff + halfH + i] = k1 * sn + k2 * cs
                        }
                    }
                    var kv = beams[b].kvCaches[l]
                    kv.append(contentsOf: kRot)
                    kv.append(contentsOf: vStep[bOff..<(bOff + d)])
                    let curSteps = kv.count / (2 * d)
                    if curSteps > winSize {
                        kv.removeFirst((curSteps - winSize) * 2 * d)
                    }
                    nextKVCaches[b][l] = kv
                    let wSteps = kv.count / (2 * d)

                    kv.withUnsafeBufferPointer { kvBuf in
                        qRot.withUnsafeBufferPointer { qrBuf in
                            ctxLA.withUnsafeMutableBufferPointer { ctxBuf in
                                let kvp = kvBuf.baseAddress!, qrp = qrBuf.baseAddress!, cp = ctxBuf.baseAddress! + bOff
                                for h in 0..<nHeads {
                                    let hOff = h * hDim
                                    var maxS: Float = -.greatestFiniteMagnitude
                                    for idx in 0..<wSteps {
                                        let dot = VectorOperations.dotProduct(a: qrp + hOff, b: kvp + idx * 2 * d + hOff, count: hDim)
                                        let s = dot * attnScale
                                        scores[idx] = s
                                        if s > maxS { maxS = s }
                                    }
                                    var sumE: Float = 0.0
                                    for idx in 0..<wSteps {
                                        let e = expf(scores[idx] - maxS)
                                        scores[idx] = e
                                        sumE += e
                                    }
                                    let inv = 1.0 / max(sumE, 1e-12)
                                    for idx in 0..<wSteps {
                                        let p = scores[idx] * inv
                                        let vp = kvp + idx * 2 * d + d + hOff
                                        for c in 0..<hDim { cp[hOff + c] += p * vp[c] }
                                    }
                                }
                            }
                        }
                    }
                }
                let laOut = SWTHawkAutoencoder.linear(x: ctxLA, w: la.wOut, b: la.bOut, rows: B, inDim: d, outDim: d)
                var x2 = [Float](repeating: 0.0, count: B * d)
                for i in 0..<(B * d) { x2[i] = x1[i] + laOut[i] }

                let uCross = SWTHawkAutoencoder.layerNorm(x: x2, gamma: lw.crossNormGamma, beta: lw.crossNormBeta, rows: B, dim: d)
                let qCross = SWTHawkAutoencoder.linear(x: uCross, w: lw.wCrossQ, b: lw.bCrossQ, rows: B, inDim: d, outDim: d)
                var ctxCross = [Float](repeating: 0.0, count: B * d)
                condKs[l].withUnsafeBufferPointer { ckBuf in
                    condVs[l].withUnsafeBufferPointer { cvBuf in
                        qCross.withUnsafeBufferPointer { qcBuf in
                            ctxCross.withUnsafeMutableBufferPointer { ccBuf in
                                let ckp = ckBuf.baseAddress!, cvp = cvBuf.baseAddress!
                                let qcp = qcBuf.baseAddress!, ccp = ccBuf.baseAddress!
                                for b in 0..<B {
                                    let bOff = b * d
                                    for h in 0..<nHeads {
                                        let hOff = h * hDim
                                        var maxS: Float = -.greatestFiniteMagnitude
                                        for m in 0..<mTok {
                                            let dot = VectorOperations.dotProduct(a: qcp + bOff + hOff, b: ckp + m * d + hOff, count: hDim)
                                            let s = dot * attnScale
                                            scores[m] = s
                                            if s > maxS { maxS = s }
                                        }
                                        var sumE: Float = 0.0
                                        for m in 0..<mTok {
                                            let e = expf(scores[m] - maxS)
                                            scores[m] = e
                                            sumE += e
                                        }
                                        let inv = 1.0 / max(sumE, 1e-12)
                                        for m in 0..<mTok {
                                            let p = scores[m] * inv
                                            let vp = cvp + m * d + hOff
                                            for c in 0..<hDim { ccp[bOff + hOff + c] += p * vp[c] }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                let crossOut = SWTHawkAutoencoder.linear(x: ctxCross, w: lw.wCrossOut, b: lw.bCrossOut, rows: B, inDim: d, outDim: d)
                var x3 = [Float](repeating: 0.0, count: B * d)
                for i in 0..<(B * d) { x3[i] = x2[i] + crossOut[i] }

                let norm2 = SWTHawkAutoencoder.layerNorm(x: x3, gamma: lw.block.norm2Gamma, beta: lw.block.norm2Beta, rows: B, dim: d)
                let mlpAct = SWTHawkAutoencoder.silu(
                    SWTHawkAutoencoder.linear(x: norm2, w: lw.block.wMLPUp, b: lw.block.bMLPUp, rows: B, inDim: d, outDim: lw.block.mlpDim)
                )
                let mlpDown = SWTHawkAutoencoder.linear(x: mlpAct, w: lw.block.wMLPDown, b: lw.block.bMLPDown, rows: B, inDim: lw.block.mlpDim, outDim: d)
                for i in 0..<(B * d) { xBatch[i] = x3[i] + mlpDown[i] }
            }

            let finalNorm = SWTHawkAutoencoder.layerNorm(x: xBatch, gamma: weights.finalNormGamma, beta: weights.finalNormBeta, rows: B, dim: d)
            let logitsBatch = SWTHawkAutoencoder.linear(x: finalNorm, w: weights.wReadout, b: weights.bReadout, rows: B, inDim: d, outDim: vSize)

            var candidates: [BeamState] = []
            candidates.reserveCapacity(B * (kW + 2))
            let topPerBeam = kW + 2

            for b in 0..<B {
                let lOff = b * vSize
                var maxL: Float = -.greatestFiniteMagnitude
                for v in TextVocabulary.specialCount..<vSize {
                    let val = logitsBatch[lOff + v]
                    if val > maxL { maxL = val }
                }
                let eosVal = logitsBatch[lOff + TextVocabulary.eosId]
                if eosVal > maxL { maxL = eosVal }

                var sumExp: Float = 0.0
                for v in TextVocabulary.specialCount..<vSize {
                    sumExp += expf(logitsBatch[lOff + v] - maxL)
                }
                sumExp += expf(eosVal - maxL)
                let logZ = maxL + logf(max(sumExp, 1e-12))

                let bm = beams[b]
                let lastTok = bm.tokens.last ?? -1
                let bigPrefix = lastTok >= 0 ? (UInt64(UInt32(lastTok)) << 32) : 0

                // Track top-(kW+2) candidate tokens for beam b
                var topIds = [Int](repeating: -1, count: topPerBeam)
                var topPenScores = [Float](repeating: -1e9, count: topPerBeam)
                var topRawLogP = [Float](repeating: -1e9, count: topPerBeam)

                for v in TextVocabulary.specialCount..<vSize {
                    if v == lastTok { continue }
                    if lastTok >= 0 && bm.seenBigrams.contains(bigPrefix | UInt64(UInt32(v))) { continue }

                    let addChars = pieceCharCount[v]
                    let newChars = bm.charCount + addChars
                    let endsP = pieceEndsWithPeriod[v]
                    if endsP {
                        if bm.periodCount == 0 && newChars < 38 { continue }
                        if bm.periodCount == 1 && newChars < minCharLength { continue }
                    }

                    let rawLP = logitsBatch[lOff + v] - logZ
                    var penLP = rawLP
                    if repetitionPenalty > 1.001 {
                        if let cnt = bm.seenCounts[v] {
                            penLP -= Float(cnt) * (repetitionPenalty - 1.0) * 3.5
                        }
                        let sKey = pieceStemKey[v]
                        if sKey != 0 && bm.seenStems.contains(sKey) {
                            penLP -= 5.5
                        }
                        if bm.usedPartnerInClause && pieceHasPartnerWord[v] {
                            penLP -= 7.5
                        }
                    }
                    if newChars > 136 && !endsP {
                        penLP -= Float(newChars - 136) * 0.15
                    }

                    if penLP > topPenScores[topPerBeam - 1] {
                        var pos = topPerBeam - 1
                        while pos > 0 && penLP > topPenScores[pos - 1] {
                            topPenScores[pos] = topPenScores[pos - 1]
                            topIds[pos] = topIds[pos - 1]
                            topRawLogP[pos] = topRawLogP[pos - 1]
                            pos -= 1
                        }
                        topPenScores[pos] = penLP
                        topIds[pos] = v
                        topRawLogP[pos] = rawLP
                    }
                }

                for k in 0..<topPerBeam {
                    let v = topIds[k]
                    if v < 0 { break }
                    var newToks = bm.tokens
                    newToks.append(v)
                    let newLP = bm.logProb + topRawLogP[k]
                    let newPen = bm.penalizedScore + topPenScores[k]
                    let newChars = bm.charCount + pieceCharCount[v]
                    let newPeriods = bm.periodCount + (pieceEndsWithPeriod[v] ? 1 : 0)

                    if newPeriods >= 2 {
                        let normS = newPen / powf(Float(5 + newToks.count), 0.75)
                        completed.append((tokens: newToks, normScore: normS))
                    } else {
                        var newBig = bm.seenBigrams
                        if lastTok >= 0 {
                            newBig.insert(bigPrefix | UInt64(UInt32(v)))
                        }
                        var newCnts = bm.seenCounts
                        if isPenalizableToken[v] {
                            newCnts[v, default: 0] += 1
                        }
                        var newStems = bm.seenStems
                        let sKey = pieceStemKey[v]
                        if sKey != 0 {
                            newStems.insert(sKey)
                        }
                        let nextPartnerFlag = pieceEndsWithPeriod[v] ? false : (bm.usedPartnerInClause || pieceHasPartnerWord[v])
                        candidates.append(BeamState(
                            tokens: newToks,
                            logProb: newLP,
                            penalizedScore: newPen,
                            charCount: newChars,
                            periodCount: newPeriods,
                            hStates: nextHStates[b],
                            kvCaches: nextKVCaches[b],
                            seenBigrams: newBig,
                            seenCounts: newCnts,
                            seenStems: newStems,
                            usedPartnerInClause: nextPartnerFlag
                        ))
                    }
                }
            }

            if candidates.isEmpty { break }
            candidates.sort { $0.penalizedScore > $1.penalizedScore }
            beams = Array(candidates.prefix(kW))

            if completed.count >= kW {
                let bestDone = completed.max(by: { $0.normScore < $1.normScore })!.normScore
                let optimisticActive = beams[0].penalizedScore / powf(Float(5 + beams[0].tokens.count), 0.75)
                if optimisticActive <= bestDone { break }
            }
        }

        if let best = completed.max(by: { $0.normScore < $1.normScore }) {
            return best.tokens
        }
        return beams.first?.tokens ?? []
    }

    /// Primary end-to-end pure-CPU inference from raw `--text` strings (`"YYYY-MM-DD HH:MM:SS comment"` or plain `"comment"`, `0%` sender text).
    public func describeTexts(
        relationalModel: SWTHawkAutoencoder,
        texts: [String],
        beamWidth: Int = 2,
        temperature: Float = 0.0
    ) -> PsychologicalStateResult {
        let parsed = TimestampedComment.parseList(texts)
        return describeComments(
            relationalModel: relationalModel,
            comments: parsed,
            beamWidth: beamWidth,
            temperature: temperature
        )
    }

    /// Primary end-to-end pure-CPU inference from chronological `[TimestampedComment]` (`0%` sender text).
    public func describeComments(
        relationalModel: SWTHawkAutoencoder,
        comments: [TimestampedComment],
        beamWidth: Int = 2,
        temperature: Float = 0.0
    ) -> PsychologicalStateResult {
        let safeComments = comments.isEmpty ? [TimestampedComment(text: "")] : comments
        let t0 = Date()
        let extracted = PsychConditionExtractor.extractFromComments(
            relationalModel: relationalModel,
            auxHeads: weights.auxiliaryHeads,
            comments: safeComments
        )
        let relMs = Date().timeIntervalSince(t0) * 1000.0

        let t1 = Date()
        let outIds: [Int]
        if temperature <= 1e-4 && beamWidth > 1 {
            outIds = generateBeam(condRaw: extracted.rawCondition, beamWidth: beamWidth)
        } else {
            outIds = generateTokens(condRaw: extracted.rawCondition, temperature: temperature)
        }
        let text = vocabulary.decode(outIds)
        let genMs = Date().timeIntervalSince(t1) * 1000.0
        let hasPast = safeComments.count >= 2
        let hasTimestamps = safeComments.count >= 2 && safeComments.allSatisfy { $0.timestamp != nil }

        return PsychologicalStateResult(
            psychologicalText: text,
            predictedArchetype: extracted.predictedArchetype,
            predictedTopology: extracted.predictedTopology,
            predictedEmotion: extracted.predictedEmotion,
            inference: extracted.inferenceCur,
            commentCount: safeComments.count,
            usedPastLog: hasPast,
            usedTemporalContext: hasTimestamps,
            relationalLatencyMs: relMs,
            generationLatencyMs: genMs
        )
    }

    /// Legacy compatibility wrapper delegating purely to receiver comments (`sender` arguments are unused).
    public func describePsychology(
        relationalModel: SWTHawkAutoencoder,
        senderEarlyText: String? = nil,
        receiverEarlyText: String? = nil,
        senderCurrentText: String = "",
        receiverCurrentText: String,
        temporalContext: TemporalRelationalContext? = nil,
        beamWidth: Int = 2,
        temperature: Float = 0.0
    ) -> PsychologicalStateResult {
        var comments: [TimestampedComment] = []
        if let early = receiverEarlyText, !early.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            if let tc = temporalContext {
                let dummyCase = RelationalDialogueCase(
                    id: "compat",
                    title: "compat",
                    expectedTopology: .acquaintedObserver,
                    expectedArchetype: .trueSupporter,
                    hostText: "",
                    partnerEarlyText: early,
                    partnerCurrentText: receiverCurrentText,
                    harmonizedPartnerText: receiverCurrentText,
                    insightLevel0: "", insightLevel1: "", insightLevel2: "",
                    temporalContext: tc
                )
                comments = TimestampedComment.synthesizeStream(from: dummyCase)
            } else {
                comments = [TimestampedComment(text: early), TimestampedComment(text: receiverCurrentText)]
            }
        } else {
            comments = [TimestampedComment(text: receiverCurrentText)]
        }
        return describeComments(
            relationalModel: relationalModel,
            comments: comments,
            beamWidth: beamWidth,
            temperature: temperature
        )
    }
}
