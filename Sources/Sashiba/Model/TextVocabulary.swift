import Foundation

/// Document record for training and evaluation.
public struct TextDocument: Codable, Sendable, Equatable {
    public let id: String
    public let title: String
    public let text: String
    public let summary: String
    public let summaryLevel0: String
    public let summaryLevel1: String

    public init(
        id: String,
        title: String = "",
        text: String,
        summary: String = "",
        summaryLevel0: String = "",
        summaryLevel1: String = ""
    ) {
        self.id = id
        self.title = title
        self.text = text
        self.summary = summary
        self.summaryLevel0 = summaryLevel0
        self.summaryLevel1 = summaryLevel1
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = (try? container.decode(String.self, forKey: .id)) ?? UUID().uuidString
        self.title = (try? container.decode(String.self, forKey: .title)) ?? ""
        self.text = try container.decode(String.self, forKey: .text)
        self.summary = (try? container.decode(String.self, forKey: .summary)) ?? ""
        self.summaryLevel0 = (try? container.decode(String.self, forKey: .summaryLevel0)) ?? ""
        self.summaryLevel1 = (try? container.decode(String.self, forKey: .summaryLevel1)) ?? ""
    }

    /// Returns the document summary corresponding to subband scale `gamma ∈ [0.0, 1.0]`.
    public func summary(forGamma gamma: Float) -> String {
        if gamma < 0.25 {
            return !summaryLevel0.isEmpty ? summaryLevel0 : (!summary.isEmpty ? summary : text)
        } else if gamma < 0.75 {
            return !summaryLevel1.isEmpty ? summaryLevel1 : (!summary.isEmpty ? summary : text)
        } else {
            return !summary.isEmpty ? summary : text
        }
    }

    /// Loads `TextDocument` items from a JSON file, a plain text file, or a directory of `.json` / `.txt` files.
    public static func loadDocuments(from paths: [String]) -> [(source: String, documents: [TextDocument])] {
        let fm = FileManager.default
        var results: [(source: String, documents: [TextDocument])] = []

        func loadFile(_ filePath: String) {
            let url = URL(fileURLWithPath: filePath)
            let ext = url.pathExtension.lowercased()
            guard let data = try? Data(contentsOf: url) else { return }

            if ext == "json" {
                if let docs = try? JSONDecoder().decode([TextDocument].self, from: data), !docs.isEmpty {
                    results.append((source: filePath, documents: docs))
                    return
                }
                if let strings = try? JSONDecoder().decode([String].self, from: data), !strings.isEmpty {
                    let docs = strings.enumerated().map { idx, s in
                        TextDocument(id: "\(url.deletingPathExtension().lastPathComponent)_\(idx + 1)", text: s)
                    }
                    results.append((source: filePath, documents: docs))
                    return
                }
            } else if ext == "txt" {
                if let content = String(data: data, encoding: .utf8) {
                    let lines = content
                        .components(separatedBy: .newlines)
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
                    if !lines.isEmpty {
                        let docs = lines.enumerated().map { idx, s in
                            TextDocument(id: "\(url.deletingPathExtension().lastPathComponent)_\(idx + 1)", text: s)
                        }
                        results.append((source: filePath, documents: docs))
                    }
                }
            }
        }

        for path in paths {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: path, isDirectory: &isDir) else { continue }
            if isDir.boolValue {
                if let items = try? fm.contentsOfDirectory(atPath: path).sorted() {
                    for item in items where (item.hasSuffix(".json") || item.hasSuffix(".txt")) && item != "dataset_config.json" {
                        loadFile((path as NSString).appendingPathComponent(item))
                    }
                    if results.isEmpty {
                        let jsonlDocs = DatasetConfiguration.loadAnchorDocumentsFromJSONL(inDirectory: path)
                        if !jsonlDocs.isEmpty {
                            results.append((source: path, documents: jsonlDocs))
                        }
                    }
                }
            } else if path.hasSuffix(".jsonl") {
                let jsonlDocs = DatasetConfiguration.loadAnchorDocumentsFromJSONL(inDirectory: (path as NSString).deletingLastPathComponent)
                if !jsonlDocs.isEmpty {
                    results.append((source: path, documents: jsonlDocs))
                }
            } else {
                loadFile(path)
            }
        }
        return results
    }
}

/// Character/token vocabulary with reserved special tokens: `padId = 0`, `unkId = 1`, `bosId = 2`, `eosId = 3`.
public struct TextVocabulary: Sendable, Equatable {
    public static let padId = 0
    public static let unkId = 1
    public static let bosId = 2
    public static let eosId = 3
    public static let specialCount = 4

    public let size: Int
    public var count: Int { size }
    private let tokenList: [Character]
    private let charToIdTable: [Character: Int]

    public init(corpus: [String] = []) {
        var set: Set<Character> = []
        for text in corpus {
            for ch in text {
                set.insert(ch)
            }
        }
        let specialChars: [Character] = ["\u{0000}", "\u{0001}", "\u{0002}", "\u{0003}"]
        let sorted = set.sorted()
        var list = specialChars
        for ch in sorted where !specialChars.contains(ch) {
            list.append(ch)
        }
        self.size = list.count
        self.tokenList = list

        var table: [Character: Int] = [:]
        for (idx, ch) in list.enumerated() {
            table[ch] = idx
        }
        self.charToIdTable = table
    }

    public init(texts: [String]) {
        self.init(corpus: texts)
    }

    public init(serializedCharacters: String) {
        let specialChars: [Character] = ["\u{0000}", "\u{0001}", "\u{0002}", "\u{0003}"]
        var list = specialChars
        for ch in serializedCharacters where !specialChars.contains(ch) {
            list.append(ch)
        }
        self.size = list.count
        self.tokenList = list

        var table: [Character: Int] = [:]
        for (idx, ch) in list.enumerated() {
            table[ch] = idx
        }
        self.charToIdTable = table
    }

    public var serializedCharacters: String {
        let specialChars: Set<Character> = ["\u{0000}", "\u{0001}", "\u{0002}", "\u{0003}"]
        var s = ""
        for ch in tokenList where !specialChars.contains(ch) {
            s.append(ch)
        }
        return s
    }

    public func serialize() -> String {
        return serializedCharacters
    }

    public static func deserialize(_ s: String) -> TextVocabulary {
        return TextVocabulary(serializedCharacters: s)
    }

    public static func loadDocuments(from url: URL) throws -> [TextDocument] {
        return TextDocument.loadDocuments(from: [url.path]).flatMap(\.documents)
    }

    public func encode(
        _ text: String,
        maxLength: Int? = nil,
        addEOS: Bool = false,
        padToLength: Int? = nil
    ) -> [Int] {
        var ids = text.map { charToIdTable[$0] ?? Self.unkId }
        if addEOS {
            ids.append(Self.eosId)
        }
        if let maxLen = maxLength, ids.count > maxLen {
            ids = Array(ids.prefix(maxLen))
        }
        if let padLen = padToLength, ids.count < padLen {
            ids.append(contentsOf: repeatElement(Self.padId, count: padLen - ids.count))
        }
        return ids
    }

    public func decode(_ ids: [Int]) -> String {
        var s = ""
        for id in ids {
            if id == Self.eosId || id == Self.padId {
                continue
            }
            if id == Self.bosId {
                continue
            }
            if id == Self.unkId {
                s.append("?")
            } else if id >= Self.specialCount && id < tokenList.count {
                s.append(tokenList[id])
            }
        }
        return s
    }

    /// Computes a corpus-statistics disentanglement gate `alphaRel ∈ [0, 1]^size` separating domain-general
    /// relational/grammatical characters (`1.0`) from domain-specific topic-noun characters (`0.02`) using only
    /// raw training text document frequencies and Unicode script ranges (no labels or category metadata).
    public func computeRelationalTokenGate(trainTexts: [String]) -> [Float] {
        let uniqueTexts = Array(Set(trainTexts))
        var docFreq = [Int](repeating: 0, count: size)
        for text in uniqueTexts {
            var seen: Set<Int> = []
            for ch in text {
                if let id = charToIdTable[ch], id >= Self.specialCount {
                    seen.insert(id)
                }
            }
            for id in seen {
                docFreq[id] += 1
            }
        }
        let totalDocs = Float(max(1, uniqueTexts.count))
        var gate = [Float](repeating: 0.0, count: size)
        for idx in Self.specialCount..<size {
            let ch = tokenList[idx]
            let scalar = ch.unicodeScalars.first?.value ?? 0
            let isHiragana = (scalar >= 0x3040 && scalar <= 0x309F)
            let isPunctOrSym = (scalar >= 0x3000 && scalar <= 0x303F) || (scalar >= 0xFF01 && scalar <= 0xFF60) || (scalar >= 0x2000 && scalar <= 0x206F) || (scalar == 0x30FC)
            let isKatakana = (scalar >= 0x30A1 && scalar <= 0x30FA)
            let dfRatio = Float(docFreq[idx]) / totalDocs
            if isHiragana || isPunctOrSym {
                gate[idx] = 1.0
            } else if isKatakana {
                gate[idx] = 0.04
            } else {
                gate[idx] = (dfRatio >= 0.012) ? 0.88 : 0.04
            }
        }
        return gate
    }
}

/// String-level subband classification rules loaded from dataset configuration when building model weights.
public struct SubbandClassifierRules: Codable, Sendable, Equatable {
    public struct SubbandModulationRule: Codable, Sendable, Equatable {
        public let pattern: String
        public let llAffinity: Float
        public let polarity: Float
        public let hhEnergyGain: Float

        public init(
            pattern: String,
            llAffinity: Float = 1.0,
            polarity: Float = 1.0,
            hhEnergyGain: Float = 1.0
        ) {
            self.pattern = pattern
            self.llAffinity = llAffinity
            self.polarity = polarity
            self.hhEnergyGain = hhEnergyGain
        }
    }

    public let sentenceDelimiters: [String]
    public let clauseDelimiters: [String]
    public let highFrequencyPrefixes: [String]
    public let negativePolarityPatterns: [String]
    public let modulationRules: [SubbandModulationRule]

    private enum CodingKeys: String, CodingKey {
        case sentenceDelimiters
        case clauseDelimiters
        case highFrequencyPrefixes
        case negativePolarityPatterns
        case modulationRules
    }

    public init(
        sentenceDelimiters: [String] = [],
        clauseDelimiters: [String] = [],
        highFrequencyPrefixes: [String] = [],
        negativePolarityPatterns: [String] = [],
        modulationRules: [SubbandModulationRule] = []
    ) {
        self.sentenceDelimiters = sentenceDelimiters
        self.clauseDelimiters = clauseDelimiters
        self.highFrequencyPrefixes = highFrequencyPrefixes
        self.negativePolarityPatterns = negativePolarityPatterns
        self.modulationRules = modulationRules
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.sentenceDelimiters = (try? container.decode([String].self, forKey: .sentenceDelimiters)) ?? []
        self.clauseDelimiters = (try? container.decode([String].self, forKey: .clauseDelimiters)) ?? []
        self.highFrequencyPrefixes = (try? container.decode([String].self, forKey: .highFrequencyPrefixes)) ?? []
        self.negativePolarityPatterns = (try? container.decode([String].self, forKey: .negativePolarityPatterns)) ?? []
        self.modulationRules = (try? container.decode([SubbandModulationRule].self, forKey: .modulationRules)) ?? []
    }

    /// Splits `text` into low-frequency base text and high-frequency clauses using `sentenceDelimiters` and `highFrequencyPrefixes`.
    public func splitBaseAndHighFrequency(_ text: String) -> (base: String, highFrequencyClauses: [String]) {
        guard let delim = sentenceDelimiters.first, !delim.isEmpty else {
            return (text, [])
        }
        let parts = text.components(separatedBy: delim).filter { !$0.isEmpty }
        var kept: [String] = []
        var highFreq: [String] = []
        for s in parts {
            let matchedPrefix = highFrequencyPrefixes.first { s.contains($0) }
            if let prefix = matchedPrefix, let r = s.range(of: prefix) {
                let head = String(s[..<r.lowerBound])
                let tail = String(s[r.lowerBound...]) + delim
                if !head.isEmpty {
                    kept.append(head + delim)
                }
                highFreq.append(tail)
            } else {
                kept.append(s + delim)
            }
        }
        return (kept.joined(), highFreq)
    }

    /// Counts occurrences of any `highFrequencyPrefixes` in `text`.
    public func countHighFrequencyOccurrences(in text: String) -> Int {
        return highFrequencyPrefixes.reduce(0) { acc, prefix in
            guard !prefix.isEmpty else { return acc }
            return acc + max(0, text.components(separatedBy: prefix).count - 1)
        }
    }
}

/// Dataset-level configuration loaded from `<dataPath>/dataset_config.json` (kept out of source code).
public struct DatasetConfiguration: Codable, Sendable, Equatable {
    public struct VerificationPairSpec: Codable, Sendable, Equatable {
        public let label: String
        public let category: String

        public init(label: String, category: String) {
            self.label = label
            self.category = category
        }
    }

    public let classifierRules: SubbandClassifierRules
    public let relationalDialogueCases: [RelationalDialogueCase]
    public let baseVocabularyCharacters: String
    public let test10Specs: [VerificationPairSpec]
    public let test11Specs: [VerificationPairSpec]
    public let test12Specs: [VerificationPairSpec]

    private enum CodingKeys: String, CodingKey {
        case classifierRules
        case relationalDialogueCases
        case baseVocabularyCharacters
        case test10Specs
        case test11Specs
        case test12Specs
    }

    public init(
        classifierRules: SubbandClassifierRules = SubbandClassifierRules(),
        relationalDialogueCases: [RelationalDialogueCase] = [],
        baseVocabularyCharacters: String = "",
        test10Specs: [VerificationPairSpec] = [],
        test11Specs: [VerificationPairSpec] = [],
        test12Specs: [VerificationPairSpec] = []
    ) {
        self.classifierRules = classifierRules
        self.relationalDialogueCases = relationalDialogueCases
        self.baseVocabularyCharacters = baseVocabularyCharacters
        self.test10Specs = test10Specs
        self.test11Specs = test11Specs
        self.test12Specs = test12Specs
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.classifierRules = (try? container.decode(SubbandClassifierRules.self, forKey: .classifierRules)) ?? SubbandClassifierRules()
        self.relationalDialogueCases = (try? container.decode([RelationalDialogueCase].self, forKey: .relationalDialogueCases)) ?? []
        self.baseVocabularyCharacters = (try? container.decode(String.self, forKey: .baseVocabularyCharacters)) ?? ""
        self.test10Specs = (try? container.decode([VerificationPairSpec].self, forKey: .test10Specs)) ?? []
        self.test11Specs = (try? container.decode([VerificationPairSpec].self, forKey: .test11Specs)) ?? []
        self.test12Specs = (try? container.decode([VerificationPairSpec].self, forKey: .test12Specs)) ?? []
    }

    private struct ScenarioJSONLRecord: Codable {
        struct ModulationValues: Codable {
            let ll_t2: Float
            let pol_t2: Float
            let gain_t2: Float
            let ll_t4: Float
            let pol_t4: Float
            let gain_t4: Float
        }
        /// Optional persona metadata; every field falls back to an empty string when absent.
        struct PersonaProfile: Codable {
            let age_group: String
            let tone: String
            let relationship_duration: String
            let context_domain: String

            init(age_group: String = "", tone: String = "", relationship_duration: String = "", context_domain: String = "") {
                self.age_group = age_group
                self.tone = tone
                self.relationship_duration = relationship_duration
                self.context_domain = context_domain
            }

            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                self.age_group = (try? container.decode(String.self, forKey: .age_group)) ?? ""
                self.tone = (try? container.decode(String.self, forKey: .tone)) ?? ""
                self.relationship_duration = (try? container.decode(String.self, forKey: .relationship_duration)) ?? ""
                self.context_domain = (try? container.decode(String.self, forKey: .context_domain)) ?? ""
            }
        }
        struct Insights: Codable {
            let level_0_core: String
            let level_1_mid: String
            let level_2_deep: String
        }
        let scenario_id: String
        let split: String
        let category_12: String
        let archetype_6: RelationalArchetype
        let topology_5: RelationalTopology
        let expected_archetype_t2: RelationalArchetype
        let expected_topology_t2: RelationalTopology
        let modulation: ModulationValues
        let persona_profile: PersonaProfile
        let sender_log_t1_t2: String
        let sender_log_t3_t4: String
        let receiver_log_t1_t2: String
        let receiver_log_t3_t4: String
        let receiver_log_harmonized: String
        let insights: Insights

        enum CodingKeys: String, CodingKey {
            case scenario_id
            case split
            case category_12
            case archetype_6
            case topology_5
            case expected_archetype_t2
            case expected_topology_t2
            case modulation
            case persona_profile
            case sender_log_t1_t2
            case sender_log_t3_t4
            case receiver_log_t1_t2
            case receiver_log_t3_t4
            case receiver_log_harmonized
            case insights
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.scenario_id = try c.decode(String.self, forKey: .scenario_id)
            self.split = try c.decode(String.self, forKey: .split)
            let cat = try c.decode(String.self, forKey: .category_12)
            self.category_12 = cat
            self.archetype_6 = try c.decode(RelationalArchetype.self, forKey: .archetype_6)
            self.topology_5 = try c.decode(RelationalTopology.self, forKey: .topology_5)
            self.expected_archetype_t2 = try c.decode(RelationalArchetype.self, forKey: .expected_archetype_t2)
            self.expected_topology_t2 = try c.decode(RelationalTopology.self, forKey: .expected_topology_t2)
            self.modulation = try c.decode(ModulationValues.self, forKey: .modulation)
            self.persona_profile = (try? c.decode(PersonaProfile.self, forKey: .persona_profile)) ?? PersonaProfile(context_domain: cat)
            self.sender_log_t1_t2 = try c.decode(String.self, forKey: .sender_log_t1_t2)
            self.sender_log_t3_t4 = try c.decode(String.self, forKey: .sender_log_t3_t4)
            self.receiver_log_t1_t2 = try c.decode(String.self, forKey: .receiver_log_t1_t2)
            self.receiver_log_t3_t4 = try c.decode(String.self, forKey: .receiver_log_t3_t4)
            self.receiver_log_harmonized = try c.decode(String.self, forKey: .receiver_log_harmonized)
            self.insights = try c.decode(Insights.self, forKey: .insights)
        }

        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(scenario_id, forKey: .scenario_id)
            try c.encode(split, forKey: .split)
            try c.encode(category_12, forKey: .category_12)
            try c.encode(archetype_6, forKey: .archetype_6)
            try c.encode(topology_5, forKey: .topology_5)
            try c.encode(expected_archetype_t2, forKey: .expected_archetype_t2)
            try c.encode(expected_topology_t2, forKey: .expected_topology_t2)
            try c.encode(modulation, forKey: .modulation)
            try c.encode(persona_profile, forKey: .persona_profile)
            try c.encode(sender_log_t1_t2, forKey: .sender_log_t1_t2)
            try c.encode(sender_log_t3_t4, forKey: .sender_log_t3_t4)
            try c.encode(receiver_log_t1_t2, forKey: .receiver_log_t1_t2)
            try c.encode(receiver_log_t3_t4, forKey: .receiver_log_t3_t4)
            try c.encode(receiver_log_harmonized, forKey: .receiver_log_harmonized)
            try c.encode(insights, forKey: .insights)
        }
    }

    private static func normalizeTwoSentences(_ text: String, sentenceDelim: String, clauseDelim: String) -> String {
        guard !sentenceDelim.isEmpty else { return text }
        let parts = text
            .components(separatedBy: sentenceDelim)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if parts.count == 2 {
            return parts[0] + sentenceDelim + parts[1] + sentenceDelim
        } else if parts.count > 2 {
            let first = parts[0]
            let rest = parts[1..<parts.count].joined(separator: clauseDelim)
            return first + sentenceDelim + rest + sentenceDelim
        } else if let single = parts.first {
            let clauses = single.components(separatedBy: clauseDelim).filter { !$0.isEmpty }
            if clauses.count >= 2 {
                let mid = clauses.count / 2
                let s1 = clauses[0..<mid].joined(separator: clauseDelim)
                let s2 = clauses[mid..<clauses.count].joined(separator: clauseDelim)
                return s1 + sentenceDelim + s2 + sentenceDelim
            }
            let midIdx = single.index(single.startIndex, offsetBy: max(1, single.count / 2))
            return String(single[..<midIdx]) + sentenceDelim + String(single[midIdx...]) + sentenceDelim
        }
        return text
    }

    private static func readJSONLRecords(inDirectory dirPath: String) -> [ScenarioJSONLRecord] {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: dirPath).sorted() else { return [] }
        let jsonlFiles: [String] = if items.contains("scenarios.jsonl") {
            ["scenarios.jsonl"]
        } else {
            items.filter { $0.hasSuffix(".jsonl") }
        }
        let decoder = JSONDecoder()
        var records: [ScenarioJSONLRecord] = []
        for file in jsonlFiles {
            let fullPath = (dirPath as NSString).appendingPathComponent(file)
            guard let content = try? String(contentsOfFile: fullPath, encoding: .utf8) else { continue }
            for line in content.components(separatedBy: .newlines) {
                let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty, let data = trimmed.data(using: .utf8),
                      let rec = try? decoder.decode(ScenarioJSONLRecord.self, from: data) else {
                    continue
                }
                records.append(rec)
            }
        }
        return records
    }

    public static func loadAnchorDocumentsFromJSONL(inDirectory dirPath: String) -> [TextDocument] {
        let baseCfg = loadBaseConfig(fromDirectory: dirPath)
        let sDelim = baseCfg.classifierRules.sentenceDelimiters.first ?? ""
        let cDelim = baseCfg.classifierRules.clauseDelimiters.first ?? ""
        let records = readJSONLRecords(inDirectory: dirPath)
        var seenCats: Set<String> = []
        var docs: [TextDocument] = []
        for rec in records where rec.split == "train" && !seenCats.contains(rec.category_12) {
            seenCats.insert(rec.category_12)
            let r12 = normalizeTwoSentences(rec.receiver_log_t1_t2, sentenceDelim: sDelim, clauseDelim: cDelim)
            let r34 = normalizeTwoSentences(rec.receiver_log_t3_t4, sentenceDelim: sDelim, clauseDelim: cDelim)
            docs.append(
                TextDocument(
                    id: rec.scenario_id,
                    title: "\(rec.category_12) - \(rec.persona_profile.context_domain) (\(rec.persona_profile.age_group))",
                    text: r12 + r34,
                    summary: rec.insights.level_2_deep,
                    summaryLevel0: rec.insights.level_0_core,
                    summaryLevel1: rec.insights.level_1_mid
                )
            )
        }
        return docs
    }

    private static func loadBaseConfig(fromDirectory dirPath: String) -> DatasetConfiguration {
        let candidatePath = (dirPath as NSString).appendingPathComponent("dataset_config.json")
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: candidatePath)),
              let decoded = try? JSONDecoder().decode(DatasetConfiguration.self, from: data) else {
            return DatasetConfiguration()
        }
        return decoded
    }

    public static func load(from directoryOrFilePath: String) -> DatasetConfiguration {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        let isDirectory = fm.fileExists(atPath: directoryOrFilePath, isDirectory: &isDir) && isDir.boolValue
        let dirPath = isDirectory ? directoryOrFilePath : (directoryOrFilePath as NSString).deletingLastPathComponent
        let candidatePath = isDirectory ? (directoryOrFilePath as NSString).appendingPathComponent("dataset_config.json") : directoryOrFilePath
        let baseConfig: DatasetConfiguration = if let data = try? Data(contentsOf: URL(fileURLWithPath: candidatePath)),
                                                  let decoded = try? JSONDecoder().decode(DatasetConfiguration.self, from: data) {
            decoded
        } else {
            DatasetConfiguration()
        }

        guard baseConfig.relationalDialogueCases.isEmpty else {
            return baseConfig
        }

        let records = readJSONLRecords(inDirectory: dirPath)
        guard !records.isEmpty else {
            return baseConfig
        }

        let sDelim = baseConfig.classifierRules.sentenceDelimiters.first ?? ""
        let cDelim = baseConfig.classifierRules.clauseDelimiters.first ?? ""
        var cases: [RelationalDialogueCase] = []
        cases.reserveCapacity(records.count)

        for rec in records {
            let s12 = normalizeTwoSentences(rec.sender_log_t1_t2, sentenceDelim: sDelim, clauseDelim: cDelim)
            let s34 = normalizeTwoSentences(rec.sender_log_t3_t4, sentenceDelim: sDelim, clauseDelim: cDelim)
            let r12 = normalizeTwoSentences(rec.receiver_log_t1_t2, sentenceDelim: sDelim, clauseDelim: cDelim)
            let r34 = normalizeTwoSentences(rec.receiver_log_t3_t4, sentenceDelim: sDelim, clauseDelim: cDelim)
            let rharm = normalizeTwoSentences(rec.receiver_log_harmonized, sentenceDelim: sDelim, clauseDelim: cDelim)

            let hostText = s12 + s34
            let earlyText = r12
            let curText = r12 + r34
            let harmText = r12 + rharm
            let prof = rec.persona_profile

            cases.append(
                RelationalDialogueCase(
                    id: rec.scenario_id,
                    group: rec.category_12,
                    personaStyle: "\(prof.age_group)・\(prof.tone) (\(prof.relationship_duration))",
                    topicDomain: prof.context_domain,
                    title: "\(rec.category_12) - \(prof.context_domain)",
                    expectedTopologyT2: rec.expected_topology_t2,
                    expectedArchetypeT2: rec.expected_archetype_t2,
                    expectedTopology: rec.topology_5,
                    expectedArchetype: rec.archetype_6,
                    hostText: hostText,
                    partnerEarlyText: earlyText,
                    partnerCurrentText: curText,
                    harmonizedPartnerText: harmText,
                    insightLevel0: rec.insights.level_0_core,
                    insightLevel1: rec.insights.level_1_mid,
                    insightLevel2: rec.insights.level_2_deep,
                    isZeroShotUnseen: rec.split != "train"
                )
            )
        }

        return DatasetConfiguration(
            classifierRules: baseConfig.classifierRules,
            relationalDialogueCases: cases,
            baseVocabularyCharacters: baseConfig.baseVocabularyCharacters,
            test10Specs: baseConfig.test10Specs,
            test11Specs: baseConfig.test11Specs,
            test12Specs: baseConfig.test12Specs
        )
    }
}

/// Vocabulary-encoded token-ID subband classifier stored in model weights alongside `TextVocabulary`.
/// Separates token sequences into Low-Frequency (`LL`) vs High-Frequency (`HH`) subbands and positive/negative polarity
/// purely on token IDs without any dataset-dependent strings in the model code.
public struct SubbandTokenClassifier: Codable, Sendable, Equatable {
    public struct CompiledModulationRule: Codable, Sendable, Equatable {
        public let patternTokens: [Int]
        public let patternHash: UInt64
        public let llAffinity: Float
        public let polarity: Float
        public let hhEnergyGain: Float

        public init(
            patternTokens: [Int],
            patternHash: UInt64? = nil,
            llAffinity: Float = 1.0,
            polarity: Float = 1.0,
            hhEnergyGain: Float = 1.0
        ) {
            self.patternTokens = patternTokens
            self.patternHash = patternHash ?? SubbandTokenClassifier.fnv1aHash(patternTokens)
            self.llAffinity = llAffinity
            self.polarity = polarity
            self.hhEnergyGain = hhEnergyGain
        }
    }

    public static func fnv1aHash(_ tokens: [Int]) -> UInt64 {
        var h: UInt64 = 0xcbf2_9ce4_8422_2325
        for tok in tokens {
            h ^= UInt64(bitPattern: Int64(tok))
            h = h &* 0x0000_0100_0000_01b3
        }
        let masked = h & 0x7fff_ffff_ffff_ffff
        return masked == 0 ? 1 : masked
    }

    public let sentenceDelimiters: [[Int]]
    public let clauseDelimiters: [[Int]]
    public let highFrequencyPrefixes: [[Int]]
    public let negativePolarityPatterns: [[Int]]
    public let modulationRules: [CompiledModulationRule]
    private let ruleByHash: [UInt64: CompiledModulationRule]

    private enum CodingKeys: String, CodingKey {
        case sentenceDelimiters
        case clauseDelimiters
        case highFrequencyPrefixes
        case negativePolarityPatterns
        case modulationRules
    }

    public init(
        sentenceDelimiters: [[Int]] = [],
        clauseDelimiters: [[Int]] = [],
        highFrequencyPrefixes: [[Int]] = [],
        negativePolarityPatterns: [[Int]] = [],
        modulationRules: [CompiledModulationRule] = []
    ) {
        self.sentenceDelimiters = sentenceDelimiters.filter { !$0.isEmpty }
        self.clauseDelimiters = clauseDelimiters.filter { !$0.isEmpty }
        self.highFrequencyPrefixes = highFrequencyPrefixes.filter { !$0.isEmpty }
        self.negativePolarityPatterns = negativePolarityPatterns.filter { !$0.isEmpty }
        let validRules = modulationRules.filter { !$0.patternTokens.isEmpty || $0.patternHash != 0 }
        self.modulationRules = validRules
        var table: [UInt64: CompiledModulationRule] = [:]
        table.reserveCapacity(validRules.count)
        for r in validRules {
            table[r.patternHash] = r
        }
        self.ruleByHash = table
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let sDelims = ((try? container.decode([[Int]].self, forKey: .sentenceDelimiters)) ?? []).filter { !$0.isEmpty }
        let cDelims = ((try? container.decode([[Int]].self, forKey: .clauseDelimiters)) ?? []).filter { !$0.isEmpty }
        let hfPrefixes = ((try? container.decode([[Int]].self, forKey: .highFrequencyPrefixes)) ?? []).filter { !$0.isEmpty }
        let negPatterns = ((try? container.decode([[Int]].self, forKey: .negativePolarityPatterns)) ?? []).filter { !$0.isEmpty }
        let modRules = ((try? container.decode([CompiledModulationRule].self, forKey: .modulationRules)) ?? []).filter { !$0.patternTokens.isEmpty || $0.patternHash != 0 }
        self.init(
            sentenceDelimiters: sDelims,
            clauseDelimiters: cDelims,
            highFrequencyPrefixes: hfPrefixes,
            negativePolarityPatterns: negPatterns,
            modulationRules: modRules
        )
    }

    public init(vocabulary: TextVocabulary, rules: SubbandClassifierRules) {
        func encodeNonEmpty(_ items: [String]) -> [[Int]] {
            return items.compactMap { str in
                let ids = vocabulary.encode(str)
                return (!ids.isEmpty && !ids.contains(TextVocabulary.unkId)) ? ids : nil
            }
        }
        let compactMode = rules.modulationRules.count > 128
        let compiledModulation: [CompiledModulationRule] = rules.modulationRules.compactMap { rule in
            let ids = vocabulary.encode(rule.pattern)
            guard !ids.isEmpty && !ids.contains(TextVocabulary.unkId) else { return nil }
            let h = Self.fnv1aHash(ids)
            return CompiledModulationRule(
                patternTokens: compactMode ? [] : ids,
                patternHash: h,
                llAffinity: rule.llAffinity,
                polarity: rule.polarity,
                hhEnergyGain: rule.hhEnergyGain
            )
        }
        self.init(
            sentenceDelimiters: encodeNonEmpty(rules.sentenceDelimiters),
            clauseDelimiters: encodeNonEmpty(rules.clauseDelimiters),
            highFrequencyPrefixes: encodeNonEmpty(rules.highFrequencyPrefixes),
            negativePolarityPatterns: encodeNonEmpty(rules.negativePolarityPatterns),
            modulationRules: compiledModulation
        )
    }

    public var isConfigured: Bool {
        return !sentenceDelimiters.isEmpty
    }

    public var primarySentenceDelimiterId: Int? {
        return sentenceDelimiters.first?.first
    }

    public var primaryClauseDelimiterId: Int? {
        return clauseDelimiters.first?.first
    }

    public func serialize() -> String {
        guard let data = try? JSONEncoder().encode(self),
              let str = String(data: data, encoding: .utf8) else {
            return ""
        }
        return str
    }

    public static func deserialize(_ str: String) -> SubbandTokenClassifier {
        guard !str.isEmpty,
              let data = str.data(using: .utf8),
              let decoded = try? JSONDecoder().decode(SubbandTokenClassifier.self, from: data) else {
            return SubbandTokenClassifier()
        }
        return decoded
    }

    /// Checks whether `tokens` starts with `prefixSeq`.
    public static func hasPrefix(_ tokens: [Int], prefix prefixSeq: [Int]) -> Bool {
        guard !prefixSeq.isEmpty && tokens.count >= prefixSeq.count else { return false }
        for i in 0..<prefixSeq.count where tokens[i] != prefixSeq[i] {
            return false
        }
        return true
    }

    /// Checks whether `tokens` contains `subseq` as a contiguous slice.
    public static func containsSubsequence(_ tokens: [Int], subsequence subseq: [Int]) -> Bool {
        guard !subseq.isEmpty && tokens.count >= subseq.count else { return false }
        let limit = tokens.count - subseq.count
        for start in 0...limit {
            var matched = true
            for j in 0..<subseq.count where tokens[start + j] != subseq[j] {
                matched = false
                break
            }
            if matched {
                return true
            }
        }
        return false
    }

    /// Splits a token sequence into sentences terminated by any sequence in `sentenceDelimiters`.
    public func splitSentences(_ tokens: [Int]) -> [[Int]] {
        guard !tokens.isEmpty else { return [] }
        guard !sentenceDelimiters.isEmpty else { return [tokens] }
        var sentences: [[Int]] = []
        var current: [Int] = []
        for tok in tokens {
            current.append(tok)
            let endsWithDelimiter = sentenceDelimiters.contains { delim in
                guard current.count >= delim.count else { return false }
                let offset = current.count - delim.count
                for i in 0..<delim.count where current[offset + i] != delim[i] {
                    return false
                }
                return true
            }
            if endsWithDelimiter {
                sentences.append(current)
                current = []
            }
        }
        if !current.isEmpty {
            sentences.append(current)
        }
        return sentences
    }

    /// Returns `true` if `sentence` begins with any registered `highFrequencyPrefixes`.
    public func isHighFrequencySentence(_ sentence: [Int]) -> Bool {
        return highFrequencyPrefixes.contains { Self.hasPrefix(sentence, prefix: $0) }
    }

    /// Returns `true` if `sentence` contains any registered `negativePolarityPatterns`.
    public func isNegativePolaritySentence(_ sentence: [Int]) -> Bool {
        return negativePolarityPatterns.contains { Self.containsSubsequence(sentence, subsequence: $0) }
    }

    /// Returns the last (most specific) matching `CompiledModulationRule` found in `tokens`, if any.
    public func matchModulationRule(in tokens: [Int]) -> CompiledModulationRule? {
        guard !modulationRules.isEmpty && !tokens.isEmpty else { return nil }
        if let exact = ruleByHash[Self.fnv1aHash(tokens)] {
            return exact
        }
        if let lastSent = splitSentences(tokens).last {
            if let exactLast = ruleByHash[Self.fnv1aHash(lastSent)] {
                return exactLast
            }
            if lastSent.count > 1, let exactTrimmed = ruleByHash[Self.fnv1aHash(Array(lastSent.dropLast()))] {
                return exactTrimmed
            }
        }
        return modulationRules.last { !$0.patternTokens.isEmpty && Self.containsSubsequence(tokens, subsequence: $0.patternTokens) }
    }

    /// Extracts the prefix of `sentence` up to and including the first `clauseDelimiters` token sequence, or the full `sentence` if none occurs.
    public func extractFirstClause(_ sentence: [Int]) -> [Int] {
        guard !sentence.isEmpty, let delim = clauseDelimiters.first, !delim.isEmpty else {
            return sentence
        }
        var current: [Int] = []
        for tok in sentence {
            current.append(tok)
            if current.count >= delim.count {
                let offset = current.count - delim.count
                var matched = true
                for i in 0..<delim.count where current[offset + i] != delim[i] {
                    matched = false
                    break
                }
                if matched {
                    return current
                }
            }
        }
        return sentence
    }
}
