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
                    for item in items where item.hasSuffix(".json") || item.hasSuffix(".txt") {
                        loadFile((path as NSString).appendingPathComponent(item))
                    }
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
    public struct PairSpec: Codable, Sendable, Equatable {
        public let label: String
        public let idA: String
        public let idB: String

        public init(label: String, idA: String, idB: String) {
            self.label = label
            self.idA = idA
            self.idB = idB
        }
    }

    public struct ExceptionArithmeticSpec: Codable, Sendable, Equatable {
        public let baseTargetId: String
        public let sourceAId: String
        public let sourceALabel: String
        public let sourceBId: String
        public let sourceBLabel: String

        public init(
            baseTargetId: String,
            sourceAId: String,
            sourceALabel: String,
            sourceBId: String,
            sourceBLabel: String
        ) {
            self.baseTargetId = baseTargetId
            self.sourceAId = sourceAId
            self.sourceALabel = sourceALabel
            self.sourceBId = sourceBId
            self.sourceBLabel = sourceBLabel
        }
    }

    public struct CoreMismatchSpec: Codable, Sendable, Equatable {
        public let statuteId: String
        public let contractId: String

        public init(statuteId: String, contractId: String) {
            self.statuteId = statuteId
            self.contractId = contractId
        }
    }

    public let classifierRules: SubbandClassifierRules
    public let unseenDocuments: [TextDocument]
    public let contractAuditCases: [ContractAuditCase]
    public let relationalDialogueCases: [RelationalDialogueCase]
    public let interpolationPairs: [PairSpec]
    public let crossSynthesisPairs: [PairSpec]
    public let exceptionArithmetic: ExceptionArithmeticSpec?
    public let coreMismatchPair: CoreMismatchSpec?

    private enum CodingKeys: String, CodingKey {
        case classifierRules
        case unseenDocuments
        case contractAuditCases
        case relationalDialogueCases
        case interpolationPairs
        case crossSynthesisPairs
        case exceptionArithmetic
        case coreMismatchPair
    }

    public init(
        classifierRules: SubbandClassifierRules = SubbandClassifierRules(),
        unseenDocuments: [TextDocument] = [],
        contractAuditCases: [ContractAuditCase] = [],
        relationalDialogueCases: [RelationalDialogueCase] = [],
        interpolationPairs: [PairSpec] = [],
        crossSynthesisPairs: [PairSpec] = [],
        exceptionArithmetic: ExceptionArithmeticSpec? = nil,
        coreMismatchPair: CoreMismatchSpec? = nil
    ) {
        self.classifierRules = classifierRules
        self.unseenDocuments = unseenDocuments
        self.contractAuditCases = contractAuditCases
        self.relationalDialogueCases = relationalDialogueCases
        self.interpolationPairs = interpolationPairs
        self.crossSynthesisPairs = crossSynthesisPairs
        self.exceptionArithmetic = exceptionArithmetic
        self.coreMismatchPair = coreMismatchPair
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.classifierRules = (try? container.decode(SubbandClassifierRules.self, forKey: .classifierRules)) ?? SubbandClassifierRules()
        self.unseenDocuments = (try? container.decode([TextDocument].self, forKey: .unseenDocuments)) ?? []
        self.contractAuditCases = (try? container.decode([ContractAuditCase].self, forKey: .contractAuditCases)) ?? []
        self.relationalDialogueCases = (try? container.decode([RelationalDialogueCase].self, forKey: .relationalDialogueCases)) ?? []
        self.interpolationPairs = (try? container.decode([PairSpec].self, forKey: .interpolationPairs)) ?? []
        self.crossSynthesisPairs = (try? container.decode([PairSpec].self, forKey: .crossSynthesisPairs)) ?? []
        self.exceptionArithmetic = try? container.decode(ExceptionArithmeticSpec.self, forKey: .exceptionArithmetic)
        self.coreMismatchPair = try? container.decode(CoreMismatchSpec.self, forKey: .coreMismatchPair)
    }

    public static func load(from directoryOrFilePath: String) -> DatasetConfiguration {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        let candidatePath = if fm.fileExists(atPath: directoryOrFilePath, isDirectory: &isDir), isDir.boolValue {
            (directoryOrFilePath as NSString).appendingPathComponent("dataset_config.json")
        } else {
            directoryOrFilePath
        }
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: candidatePath)),
              let decoded = try? JSONDecoder().decode(DatasetConfiguration.self, from: data) else {
            return DatasetConfiguration()
        }
        return decoded
    }
}

/// Vocabulary-encoded token-ID subband classifier stored in model weights alongside `TextVocabulary`.
/// Separates token sequences into Low-Frequency (`LL`) vs High-Frequency (`HH`) subbands and positive/negative polarity
/// purely on token IDs without any dataset-dependent strings in the model code.
public struct SubbandTokenClassifier: Codable, Sendable, Equatable {
    public struct CompiledModulationRule: Codable, Sendable, Equatable {
        public let patternTokens: [Int]
        public let llAffinity: Float
        public let polarity: Float
        public let hhEnergyGain: Float

        public init(
            patternTokens: [Int],
            llAffinity: Float = 1.0,
            polarity: Float = 1.0,
            hhEnergyGain: Float = 1.0
        ) {
            self.patternTokens = patternTokens
            self.llAffinity = llAffinity
            self.polarity = polarity
            self.hhEnergyGain = hhEnergyGain
        }
    }

    public let sentenceDelimiters: [[Int]]
    public let clauseDelimiters: [[Int]]
    public let highFrequencyPrefixes: [[Int]]
    public let negativePolarityPatterns: [[Int]]
    public let modulationRules: [CompiledModulationRule]

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
        self.modulationRules = modulationRules.filter { !$0.patternTokens.isEmpty }
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.sentenceDelimiters = ((try? container.decode([[Int]].self, forKey: .sentenceDelimiters)) ?? []).filter { !$0.isEmpty }
        self.clauseDelimiters = ((try? container.decode([[Int]].self, forKey: .clauseDelimiters)) ?? []).filter { !$0.isEmpty }
        self.highFrequencyPrefixes = ((try? container.decode([[Int]].self, forKey: .highFrequencyPrefixes)) ?? []).filter { !$0.isEmpty }
        self.negativePolarityPatterns = ((try? container.decode([[Int]].self, forKey: .negativePolarityPatterns)) ?? []).filter { !$0.isEmpty }
        self.modulationRules = ((try? container.decode([CompiledModulationRule].self, forKey: .modulationRules)) ?? []).filter { !$0.patternTokens.isEmpty }
    }

    public init(vocabulary: TextVocabulary, rules: SubbandClassifierRules) {
        func encodeNonEmpty(_ items: [String]) -> [[Int]] {
            return items.compactMap { str in
                let ids = vocabulary.encode(str)
                return (!ids.isEmpty && !ids.contains(TextVocabulary.unkId)) ? ids : nil
            }
        }
        let compiledModulation: [CompiledModulationRule] = rules.modulationRules.compactMap { rule in
            let ids = vocabulary.encode(rule.pattern)
            guard !ids.isEmpty && !ids.contains(TextVocabulary.unkId) else { return nil }
            return CompiledModulationRule(
                patternTokens: ids,
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
        return modulationRules.last { Self.containsSubsequence(tokens, subsequence: $0.patternTokens) }
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
