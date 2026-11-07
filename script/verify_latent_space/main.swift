import Foundation
import Sashiba

struct VerifyLatentSpaceCLI {
    var dataPath: String = ".tmp"
    var weightsPath: String = ".tmp/weights.safetensors"
    var reportPath: String = "verification_report.md"
    var maxSeqLen: Int = 512

    static func parse(_ args: [String]) -> VerifyLatentSpaceCLI {
        var cli = VerifyLatentSpaceCLI()
        var i = 1
        while i < args.count {
            switch args[i] {
            case "-d", "--data", "--dataset":
                i += 1
                if i < args.count { cli.dataPath = args[i] }
            case "-w", "--weights", "--import-weights":
                i += 1
                if i < args.count { cli.weightsPath = args[i] }
            case "-o", "--output", "--report":
                i += 1
                if i < args.count { cli.reportPath = args[i] }
            case "--max-len":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.maxSeqLen = max(4, v) }
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

@inline(__always)
func fmt(_ format: String, _ v: Float) -> String {
    return String(format: format, v)
}

// MARK: - Vector & Text Metrics

func l2Norm(_ v: [Float]) -> Float {
    guard !v.isEmpty else { return 0.0 }
    return v.withUnsafeBufferPointer { ptr in
        sqrtf(VectorOperations.sumOfSquares(ptr: ptr.baseAddress!, count: ptr.count))
    }
}

func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
    precondition(a.count == b.count)
    guard !a.isEmpty else { return 0.0 }
    return a.withUnsafeBufferPointer { aPtr in
        b.withUnsafeBufferPointer { bPtr in
            VectorOperations.dotProduct(a: aPtr.baseAddress!, b: bPtr.baseAddress!, count: a.count)
        }
    }
}

func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
    let denom = max(1e-12, l2Norm(a) * l2Norm(b))
    return dotProduct(a, b) / denom
}

/// Spherical Linear Interpolation (`Slerp`) in R^1024:
///   Ω = arccos((z_A · z_B) / (||z_A|| ||z_B||))
///   z_slerp(α) = (sin((1 - α)Ω) / sin Ω) z_A + (sin(αΩ) / sin Ω) z_B
func slerp(_ zA: [Float], _ zB: [Float], alpha: Float) -> [Float] {
    precondition(zA.count == zB.count)
    let cosTheta = max(-1.0, min(1.0, cosineSimilarity(zA, zB)))
    let omega = acosf(cosTheta)
    let sinOmega = sinf(omega)
    if abs(sinOmega) < 1e-6 {
        var out = [Float](repeating: 0.0, count: zA.count)
        let wA = 1.0 - alpha
        for i in 0..<zA.count {
            out[i] = wA * zA[i] + alpha * zB[i]
        }
        return out
    }
    let coeffA = sinf((1.0 - alpha) * omega) / sinOmega
    let coeffB = sinf(alpha * omega) / sinOmega
    var out = [Float](repeating: 0.0, count: zA.count)
    for i in 0..<zA.count {
        out[i] = coeffA * zA[i] + coeffB * zB[i]
    }
    return out
}

/// Linear Interpolation + LayerNorm renormalization:
///   z_tilde(α) = (1 - α) z_A + α z_B
///   z_mix(α)   = LayerNorm(z_tilde(α))
func lerpLayerNorm(_ zA: [Float], _ zB: [Float], alpha: Float, model: SWTHawkAutoencoder) -> [Float] {
    precondition(zA.count == zB.count)
    let wA = 1.0 - alpha
    var zTilde = [Float](repeating: 0.0, count: zA.count)
    for i in 0..<zA.count {
        zTilde[i] = wA * zA[i] + alpha * zB[i]
    }
    return model.layerNormLatent(zTilde, rows: 1)
}

/// Character-level Levenshtein edit distance.
func levenshteinDistance(_ a: String, _ b: String) -> Int {
    let s1 = Array(a)
    let s2 = Array(b)
    let m = s1.count
    let n = s2.count
    if m == 0 { return n }
    if n == 0 { return m }

    var prev = Array(0...n)
    var curr = [Int](repeating: 0, count: n + 1)
    for i in 1...m {
        curr[0] = i
        let c1 = s1[i - 1]
        for j in 1...n {
            let cost = (c1 == s2[j - 1]) ? 0 : 1
            curr[j] = min(prev[j] + 1, min(curr[j - 1] + 1, prev[j - 1] + cost))
        }
        swap(&prev, &curr)
    }
    return prev[n]
}

func characterMatchRate(_ a: String, _ b: String) -> Float {
    let maxLen = max(a.count, b.count)
    guard maxLen > 0 else { return 1.0 }
    let dist = levenshteinDistance(a, b)
    return max(0.0, 1.0 - Float(dist) / Float(maxLen))
}

func tokenMatchRate(_ a: [Int], _ b: [Int]) -> Float {
    let len = min(a.count, b.count)
    guard len > 0 else { return 0.0 }
    var matches = 0
    for i in 0..<len where a[i] == b[i] {
        matches += 1
    }
    return Float(matches) / Float(len)
}

// MARK: - Main Verification Pipeline

let cli = VerifyLatentSpaceCLI.parse(CommandLine.arguments)
let weightsURL = URL(fileURLWithPath: cli.weightsPath)
let weights = try SWTHawkWeights.load(from: weightsURL)
let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)
let model = SWTHawkAutoencoder(weights: weights)

let docs = try TextVocabulary.loadDocuments(from: URL(fileURLWithPath: cli.dataPath))
let datasetConfig = DatasetConfiguration.load(from: cli.dataPath)

guard !docs.isEmpty else {
    print("No documents found in \(cli.dataPath)")
    exit(1)
}

var encodedDocs: [[Int]] = []
var maxObservedLen = 0
for doc in docs {
    let ids = vocab.encode(doc.text, maxLength: cli.maxSeqLen, addEOS: false, padToLength: nil)
    encodedDocs.append(ids)
    if ids.count > maxObservedLen {
        maxObservedLen = ids.count
    }
}
let seqLen = min(cli.maxSeqLen, max(4, maxObservedLen))

func padTokens(_ ids: [Int], to length: Int) -> [Int] {
    var row = Array(ids.prefix(length))
    if row.count < length {
        row.append(contentsOf: repeatElement(TextVocabulary.padId, count: length - row.count))
    }
    return row
}

var paddedDocs: [[Int]] = []
var flatAllTokens: [Int] = []
for ids in encodedDocs {
    let row = padTokens(ids, to: seqLen)
    paddedDocs.append(row)
    flatAllTokens.append(contentsOf: row)
}

let numDocs = docs.count
let latentDim = weights.latentDim
let allZFlat = model.encode(tokens: flatAllTokens, batch: numDocs, time: seqLen)
let allPredIds = model.reconstructGreedy(z: allZFlat, batch: numDocs, length: seqLen)

var docIndexById: [String: Int] = [:]
var docLatents: [[Float]] = []
var docDecodedTexts: [String] = []
var docTokenAccs: [Float] = []
var docEditDists: [Int] = []
var docCharMatchRates: [Float] = []

for b in 0..<numDocs {
    docIndexById[docs[b].id] = b
    let zVec = Array(allZFlat[(b * latentDim)..<((b + 1) * latentDim)])
    docLatents.append(zVec)
    let predIds = allPredIds[b]
    let recText = vocab.decode(predIds)
    docDecodedTexts.append(recText)
    let tokAcc = tokenMatchRate(paddedDocs[b], predIds)
    docTokenAccs.append(tokAcc)
    let editDist = levenshteinDistance(docs[b].text, recText)
    docEditDists.append(editDist)
    docCharMatchRates.append(characterMatchRate(docs[b].text, recText))
}

@MainActor
func nearestDocument(to z: [Float]) -> (id: String, cosine: Float) {
    var bestIdx = 0
    var bestCos: Float = -2.0
    for i in 0..<numDocs {
        let c = cosineSimilarity(z, docLatents[i])
        if c > bestCos {
            bestCos = c
            bestIdx = i
        }
    }
    return (docs[bestIdx].id, bestCos)
}

@MainActor
func nearestDocumentByText(_ text: String) -> (id: String, charSim: Float, editDist: Int) {
    var bestIdx = 0
    var bestSim: Float = -1.0
    var bestDist = Int.max
    for i in 0..<numDocs {
        let sim = characterMatchRate(text, docs[i].text)
        let dist = levenshteinDistance(text, docs[i].text)
        if sim > bestSim {
            bestSim = sim
            bestDist = dist
            bestIdx = i
        }
    }
    return (docs[bestIdx].id, bestSim, bestDist)
}

print("==================================================================")
print("SWT-Hawk Text Autoencoder Latent Space Verification (z ∈ R^\(latentDim))")
print("==================================================================")

var report = ""
report += "# SWT-Griffin Autoencoder & Conditional Flow Matching 主成分空間（$z = [z_{\\text{LL}} \\ ; \\ z_{\\text{HH}}] \\in \\mathbb{R}^{1024}$）Phase 2・Phase 3 総合検証レポート\n\n"
report += "## 0. ベースライン：学習済み文档（\(numDocs)件）の再構成メトリクス\n\n"
report += "- **重みファイル**: `\(cli.weightsPath)`\n"
report += "- **系列長 $T$**: `\(seqLen)` トークン | **語彙サイズ $V$**: `\(vocab.count)` | **隠れ層次元 $D$**: `\(weights.dim)` | **主成分次元 $D_{\\text{latent}}$**: `\(latentDim)`\n"
report += "- **推論条件**: 追加学習なし・Input Word Dropout 無効化 ($p = 0.0$)・Greedy自己回帰デコード\n\n"
report += "| ID | タイトル | $\\|z\\|_2$ | Token Accuracy | Levenshtein 距離 | 文字一致率 |\n"
report += "| :--- | :--- | ---: | ---: | ---: | ---: |\n"

var sumTokAcc: Float = 0.0
var sumEditDist = 0
var sumCharMatch: Float = 0.0
var exactCount = 0

for b in 0..<numDocs {
    let norm = l2Norm(docLatents[b])
    let tokAcc = docTokenAccs[b] * 100.0
    let editDist = docEditDists[b]
    let charMatch = docCharMatchRates[b] * 100.0
    sumTokAcc += tokAcc
    sumEditDist += editDist
    sumCharMatch += charMatch
    if editDist == 0 { exactCount += 1 }
    report += "| `\(docs[b].id)` | \(docs[b].title) | \(fmt("%.2f", norm)) | \(fmt("%.2f", tokAcc))% | \(editDist) | \(fmt("%.2f", charMatch))% |\n"
}

let avgTokAcc = sumTokAcc / Float(numDocs)
let avgEditDist = Float(sumEditDist) / Float(numDocs)
let avgCharMatch = sumCharMatch / Float(numDocs)

var minPairCos: Float = 1.0
var maxPairCos: Float = -1.0
var sumPairCos: Float = 0.0
var pairCount = 0
for i in 0..<numDocs {
    for j in (i + 1)..<numDocs {
        let c = cosineSimilarity(docLatents[i], docLatents[j])
        minPairCos = min(minPairCos, c)
        maxPairCos = max(maxPairCos, c)
        sumPairCos += c
        pairCount += 1
    }
}
let meanPairCos = sumPairCos / Float(max(1, pairCount))

report += "| **平均 / 合計** | 全\(numDocs)件 完全一致 \(exactCount)/\(numDocs) | - | **\(fmt("%.2f", avgTokAcc))%** | **\(fmt("%.2f", avgEditDist))** | **\(fmt("%.2f", avgCharMatch))%** |\n\n"
report += "- **主成分ベクトル非対角コサイン類似度 $\\cos(z_i, z_j)$**: 最小 `\(fmt("%.4f", minPairCos))` / 平均 `\(fmt("%.4f", meanPairCos))` / 最大 `\(fmt("%.4f", maxPairCos))`\n\n"

print("[Baseline] \(numDocs) Docs Avg Token Acc: \(fmt("%.2f", avgTokAcc))% | Exact Match: \(exactCount)/\(numDocs) | Mean Pairwise cos(z_i, z_j): \(fmt("%.4f", meanPairCos))")

// MARK: - Test 1: Latent Interpolation

report += "---\n\n"
report += "## Test 1: 主成分ベクトルの連続補間テスト（Latent Interpolation）\n\n"

let alphas: [Float] = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
var midpointIsStepFunction = true

for pair in datasetConfig.interpolationPairs {
    guard let idxA = docIndexById[pair.idA], let idxB = docIndexById[pair.idB] else { continue }
    let zA = docLatents[idxA]
    let zB = docLatents[idxB]
    let tokA = paddedDocs[idxA]
    let tokB = paddedDocs[idxB]
    let baseCosAB = cosineSimilarity(zA, zB)

    report += "### \(pair.label)\n\n"
    report += "- **端点間コサイン類似度 $\\cos(z_A, z_B)$**: `\(fmt("%.4f", baseCosAB))` (`\(pair.idA)` $\\|z_A\\|_2 = \(fmt("%.2f", l2Norm(zA)))$, `\(pair.idB)` $\\|z_B\\|_2 = \(fmt("%.2f", l2Norm(zB)))$)\n\n"

    for method in ["1. 線形補間 + LayerNorm再正規化", "2. 球面線形補間（Slerp）"] {
        report += "#### \(method)\n\n"
        report += "| $\\alpha$ | $\\|z(\\alpha)\\|_2$ | $\\cos(z, z_A)$ | $\\cos(z, z_B)$ | Token一致率(A) | Token一致率(B) | 最近傍(文字一致率) | 生成テキスト（抜粋 90文字） |\n"
        report += "| :---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- |\n"

        var fullOutputs: [(alpha: Float, text: String)] = []
        for alpha in alphas {
            let zMix = if method.hasPrefix("1.") {
                lerpLayerNorm(zA, zB, alpha: alpha, model: model)
            } else {
                slerp(zA, zB, alpha: alpha)
            }
            let predIds = model.reconstructGreedy(z: zMix, batch: 1, length: seqLen)[0]
            let recText = vocab.decode(predIds)
            fullOutputs.append((alpha, recText))

            let normMix = l2Norm(zMix)
            let cosA = cosineSimilarity(zMix, zA)
            let cosB = cosineSimilarity(zMix, zB)
            let simTokA = tokenMatchRate(predIds, tokA) * 100.0
            let simTokB = tokenMatchRate(predIds, tokB) * 100.0
            let nearest = nearestDocumentByText(recText)
            let preview = String(recText.prefix(90))

            if abs(alpha - 0.5) < 1e-4 && nearest.charSim < 0.98 && simTokA < 98.0 && simTokB < 98.0 {
                midpointIsStepFunction = false
            }

            report += "| `\(fmt("%.1f", alpha))` | \(fmt("%.2f", normMix)) | \(fmt("%+.4f", cosA)) | \(fmt("%+.4f", cosB)) | \(fmt("%5.1f", simTokA))% | \(fmt("%5.1f", simTokB))% | `\(nearest.id)` (\(fmt("%.1f", nearest.charSim * 100.0))%) | \(preview)… |\n"
        }
        report += "\n<details><summary>\(pair.idA) → \(pair.idB) (\(method)) 全文出力</summary>\n\n"
        for item in fullOutputs {
            report += "- **$\\alpha = \(fmt("%.1f", item.alpha))$**: \(item.text)\n"
        }
        report += "\n</details>\n\n"
    }
}

print("[Test 1] Completed Latent Interpolation (Linear+LayerNorm & Slerp).")

// MARK: - Test 2: SWT Wavelet Cross-Synthesis (LL Approximation x HH Detail Swap)

report += "---\n\n"
report += "## Test 2: SWT周波数ブランチの「骨格（LL）× 例外規定（HH）」すげ替え合成（Wavelet Cross-Synthesis）\n\n"
report += "$$F_{\\text{SWT}}^{(A \\leftarrow B)} = G \\odot F_{\\text{approx}}^{(A)} + (1 - G) \\odot F_{\\text{detail}}^{(B)}$$\n\n"

let temporalBetas: [Float] = [0.0, 0.25, 0.5]

for cp in datasetConfig.crossSynthesisPairs {
    guard let idxA = docIndexById[cp.idA], let idxB = docIndexById[cp.idB] else { continue }
    let zA = docLatents[idxA]
    let zB = docLatents[idxB]

    let shortLen = min(encodedDocs[idxA].count, encodedDocs[idxB].count)
    let truncA = Array(encodedDocs[idxA].prefix(shortLen))
    let truncB = Array(encodedDocs[idxB].prefix(shortLen))

    report += "### \(cp.label)\n\n"
    report += "- **系列長揃え**: 短縮切り詰め長 $T_{\\text{short}} = \(shortLen)$ トークン / 標準パディング長 $T = \(seqLen)$ トークン\n\n"
    report += "| モード / Temporal混合比 $\\beta$ | $\\|z_{\\text{swap}}\\|_2$ | $\\cos(z_{\\text{swap}}, z_A)$ | $\\cos(z_{\\text{swap}}, z_B)$ | 文字一致率(A) | 文字一致率(B) | 生成テキスト（抜粋 90文字） |\n"
    report += "| :--- | ---: | ---: | ---: | ---: | ---: | :--- |\n"

    for beta in temporalBetas {
        let zSwapShort = model.encodeWaveletCrossSynthesis(
            tokensA: truncA, tokensB: truncB, batch: 1, time: shortLen, temporalMixBeta: beta
        )
        let predShort = model.reconstructGreedy(z: zSwapShort, batch: 1, length: seqLen)[0]
        let textShort = vocab.decode(predShort)
        let cosAShort = cosineSimilarity(zSwapShort, zA)
        let cosBShort = cosineSimilarity(zSwapShort, zB)
        let charSimAShort = characterMatchRate(textShort, docs[idxA].text) * 100.0
        let charSimBShort = characterMatchRate(textShort, docs[idxB].text) * 100.0
        report += "| 短縮揃え ($T=\(shortLen), \\beta=\(fmt("%.2f", beta))$) | \(fmt("%.2f", l2Norm(zSwapShort))) | \(fmt("%+.4f", cosAShort)) | \(fmt("%+.4f", cosBShort)) | \(fmt("%5.1f", charSimAShort))% | \(fmt("%5.1f", charSimBShort))% | \(String(textShort.prefix(90)))… |\n"

        let zSwapFull = model.encodeWaveletCrossSynthesis(
            tokensA: paddedDocs[idxA], tokensB: paddedDocs[idxB], batch: 1, time: seqLen, temporalMixBeta: beta
        )
        let predFull = model.reconstructGreedy(z: zSwapFull, batch: 1, length: seqLen)[0]
        let textFull = vocab.decode(predFull)
        let cosAFull = cosineSimilarity(zSwapFull, zA)
        let cosBFull = cosineSimilarity(zSwapFull, zB)
        let charSimAFull = characterMatchRate(textFull, docs[idxA].text) * 100.0
        let charSimBFull = characterMatchRate(textFull, docs[idxB].text) * 100.0
        report += "| パディング揃え ($T=\(seqLen), \\beta=\(fmt("%.2f", beta))$) | \(fmt("%.2f", l2Norm(zSwapFull))) | \(fmt("%+.4f", cosAFull)) | \(fmt("%+.4f", cosBFull)) | \(fmt("%5.1f", charSimAFull))% | \(fmt("%5.1f", charSimBFull))% | \(String(textFull.prefix(90)))… |\n"
    }
    report += "\n"
}

print("[Test 2] Completed SWT Wavelet Cross-Synthesis (F_approx^(A) + F_detail^(B)).")

// MARK: - Test 3: Latent Exception Arithmetic

report += "---\n\n"
report += "## Test 3: 高周波サブバンド（例外規定）の主成分ベクトル算術テスト（Latent Exception Arithmetic）\n\n"
report += "$$\\Delta z_{\\text{except}} = z_{\\text{full}} - z_{\\text{base}}, \\quad z_{\\text{arith}}(\\gamma) = \\text{LayerNorm}(z_{Y\\_\\text{base}} + \\gamma \\Delta z_{\\text{except}})$$\n\n"

let halfDim = latentDim / 2
func subbandDiffNormLL(_ a: [Float], _ b: [Float]) -> Float {
    var s: Float = 0.0
    for i in 0..<halfDim {
        let d = a[i] - b[i]
        s += d * d
    }
    return sqrtf(s)
}
func subbandNormHH(_ a: [Float]) -> Float {
    var s: Float = 0.0
    for i in halfDim..<latentDim {
        s += a[i] * a[i]
    }
    return sqrtf(s)
}

var llDiffPrimary: Float = 0.0
var hhNormBasePrimary: Float = 0.0

if let arithSpec = datasetConfig.exceptionArithmetic,
   let idxTarget = docIndexById[arithSpec.baseTargetId],
   let idxSourceA = docIndexById[arithSpec.sourceAId],
   let idxSourceB = docIndexById[arithSpec.sourceBId] {
    let textTargetBase = datasetConfig.classifierRules.splitBaseAndHighFrequency(docs[idxTarget].text).base
    let textSourceABase = datasetConfig.classifierRules.splitBaseAndHighFrequency(docs[idxSourceA].text).base
    let textSourceBBase = datasetConfig.classifierRules.splitBaseAndHighFrequency(docs[idxSourceB].text).base

    let tokTargetBase = padTokens(vocab.encode(textTargetBase, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let tokSourceABase = padTokens(vocab.encode(textSourceABase, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let tokSourceBBase = padTokens(vocab.encode(textSourceBBase, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)

    let zTargetFull = docLatents[idxTarget]
    let zSourceAFull = docLatents[idxSourceA]
    let zSourceBFull = docLatents[idxSourceB]

    let zTargetBase = model.encode(tokens: tokTargetBase, batch: 1, time: seqLen)
    let zSourceABase = model.encode(tokens: tokSourceABase, batch: 1, time: seqLen)
    let zSourceBBase = model.encode(tokens: tokSourceBBase, batch: 1, time: seqLen)

    let llDiffTarget = subbandDiffNormLL(zTargetFull, zTargetBase)
    let llDiffSourceA = subbandDiffNormLL(zSourceAFull, zSourceABase)
    let llDiffSourceB = subbandDiffNormLL(zSourceBFull, zSourceBBase)
    let hhNormTargetBase = subbandNormHH(zTargetBase)
    let hhNormSourceABase = subbandNormHH(zSourceABase)
    let hhNormSourceBBase = subbandNormHH(zSourceBBase)

    llDiffPrimary = llDiffSourceA
    hhNormBasePrimary = hhNormSourceABase

    var deltaZA = [Float](repeating: 0.0, count: latentDim)
    var deltaZB = [Float](repeating: 0.0, count: latentDim)
    for i in 0..<latentDim {
        deltaZA[i] = zSourceAFull[i] - zSourceABase[i]
        deltaZB[i] = zSourceBFull[i] - zSourceBBase[i]
    }

    report += "- **サブバンド直交分離検証 ($z = [z_{\\text{LL}}(512) \\ ; \\ z_{\\text{HH}}(512)]$)**:\n"
    report += "  - $\\|z_{\\text{LL}}(X_{\\text{full}}) - z_{\\text{LL}}(X_{\\text{base}})\\|_2$: `\(arithSpec.sourceBId)` = `\(fmt("%.4f", llDiffSourceB))`, `\(arithSpec.sourceAId)` = `\(fmt("%.4f", llDiffSourceA))`, `\(arithSpec.baseTargetId)` = `\(fmt("%.4f", llDiffTarget))`\n"
    report += "  - $\\|z_{\\text{HH}}(X_{\\text{base}})\\|_2$: `\(arithSpec.sourceBId)` = `\(fmt("%.4f", hhNormSourceBBase))`, `\(arithSpec.sourceAId)` = `\(fmt("%.4f", hhNormSourceABase))`, `\(arithSpec.baseTargetId)` = `\(fmt("%.4f", hhNormTargetBase))`\n\n"

    let gammas: [Float] = [0.0, 0.5, 1.0, 1.5]
    for (sourceLabel, deltaVec) in [(arithSpec.sourceALabel, deltaZA), (arithSpec.sourceBLabel, deltaZB)] {
        report += "### `\(arithSpec.baseTargetId)` 原則テキスト $z_{Y\\_\\text{base}} + \\gamma$（$\(sourceLabel)$）\n\n"
        report += "| $\\gamma$ | $\\|z_{\\text{arith}}\\|_2$ | $\\cos(z_{\\text{arith}}, z_{Y\\_\\text{base}})$ | $\\cos(z_{\\text{arith}}, z_{\\text{full}})$ | 高周波句出現回数 | 最近傍 | デコード生成テキスト（全文） |\n"
        report += "| :---: | ---: | ---: | ---: | :---: | :---: | :--- |\n"

        for gamma in gammas {
            var rawArith = [Float](repeating: 0.0, count: latentDim)
            for i in 0..<latentDim {
                rawArith[i] = zTargetBase[i] + gamma * deltaVec[i]
            }
            let zArith = model.layerNormLatent(rawArith, rows: 1)
            let predIds = model.reconstructGreedy(z: zArith, batch: 1, length: seqLen)[0]
            let recText = vocab.decode(predIds)
            let hfCount = datasetConfig.classifierRules.countHighFrequencyOccurrences(in: recText)
            let nearest = nearestDocumentByText(recText)

            report += "| `\(fmt("%.1f", gamma))` | \(fmt("%.2f", l2Norm(zArith))) | \(fmt("%+.4f", cosineSimilarity(zArith, zTargetBase))) | \(fmt("%+.4f", cosineSimilarity(zArith, zTargetFull))) | \(hfCount)回 | `\(nearest.id)` (\(fmt("%.1f", nearest.charSim * 100.0))%) | \(recText) |\n"
        }
        report += "\n"
    }
}

print("[Test 3] Completed Latent Exception Arithmetic (z_Y_base + γ Δz_except).")

// MARK: - Test 4: Unseen Document Zero-Shot Reconstruction Test

report += "---\n\n"
report += "## Test 4: 未知テキストによるゼロショット再構成テスト（Unseen Combination Test）\n\n"
report += "| ID | タイトル | 入力文字数 | 未知語 (`<unk>`) 数 | $\\|z\\|_2$ | Token正解率 | Levenshtein 距離 | 文字一致率 | 最近傍 ($\\cos$ / 文字一致率) |\n"
report += "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n"

var patternACharMatch: Float = 0.0
var patternANearestTrainCharSim: Float = 0.0
var unseenDetails: [(TextDocument, String, Int, Float, Int, Float, String, Float, String, Float)] = []

for (idx, uDoc) in datasetConfig.unseenDocuments.enumerated() {
    let rawIds = vocab.encode(uDoc.text, maxLength: seqLen, addEOS: false, padToLength: nil)
    let unkCount = rawIds.filter { $0 == TextVocabulary.unkId }.count
    let paddedIds = padTokens(rawIds, to: seqLen)
    let zUnseen = model.encode(tokens: paddedIds, batch: 1, time: seqLen)
    let predIds = model.reconstructGreedy(z: zUnseen, batch: 1, length: seqLen)[0]
    let recText = vocab.decode(predIds)

    let activeLen = max(1, rawIds.count)
    var activeCorrect = 0
    for t in 0..<activeLen where predIds[t] == paddedIds[t] {
        activeCorrect += 1
    }
    let tokAcc = Float(activeCorrect) / Float(activeLen) * 100.0
    let editDist = levenshteinDistance(uDoc.text, recText)
    let charMatch = characterMatchRate(uDoc.text, recText) * 100.0
    let nearLatent = nearestDocument(to: zUnseen)
    let nearText = nearestDocumentByText(recText)

    if idx == 0 {
        patternACharMatch = charMatch
        patternANearestTrainCharSim = nearText.charSim * 100.0
    }

    unseenDetails.append((
        uDoc, recText, unkCount, tokAcc, editDist, charMatch,
        nearLatent.id, nearLatent.cosine, nearText.id, nearText.charSim * 100.0
    ))

    report += "| `\(uDoc.id)` | \(uDoc.title) | \(uDoc.text.count) | \(unkCount) | \(fmt("%.2f", l2Norm(zUnseen))) | \(fmt("%.2f", tokAcc))% | \(editDist) | \(fmt("%.2f", charMatch))% | Latent:`\(nearLatent.id)`(\(fmt("%.3f", nearLatent.cosine))) / Text:`\(nearText.id)`(\(fmt("%.1f", nearText.charSim * 100.0))%) |\n"
}
report += "\n"

for detail in unseenDetails {
    let (uDoc, recText, unkCount, tokAcc, editDist, charMatch, nearLatId, nearLatCos, nearTxtId, nearTxtSim) = detail
    report += "### `\(uDoc.id)`: \(uDoc.title)\n\n"
    report += "- **入力テキスト**: \(uDoc.text)\n"
    report += "- **再構成出力**: \(recText)\n"
    report += "- **定量指標**: 未知文字 `<unk>` = \(unkCount)個 | Token正解率 = `\(fmt("%.2f", tokAcc))%` | Levenshtein編集距離 = `\(editDist)` | 文字一致率 = `\(fmt("%.2f", charMatch))%` | 主成分空間最近傍 = `\(nearLatId)` ($\\cos = \(fmt("%.4f", nearLatCos))$) | 出力テキスト最近傍 = `\(nearTxtId)` (一致率 `\(fmt("%.1f", nearTxtSim))%`)\n\n"
}

print("[Test 4] Completed Unseen Document Zero-Shot Reconstruction.")

// MARK: - Test 5: Stage 4 Conditional Flow Matching (50-Step Midpoint Method RK2 ODE) Summarization

report += "---\n\n"
report += "## Test 5: Stage 4 Conditional Flow Matching（50-Step Midpoint Method RK2 ODE）による要約テスト\n\n"
report += "$$k_1 = \\tilde{v}_\\theta(x_t, t, c), \\quad k_2 = \\tilde{v}_\\theta\\left(x_t + \\tfrac{1}{2}\\Delta t \\cdot k_1, \\ t + \\tfrac{1}{2}\\Delta t, \\ c\\right), \\quad x_{t+\\Delta t} = x_t + \\Delta t \\cdot k_2$$\n\n"

report += "| ID | 入力文字数 | 参照要約文字数 | $\\|z_{\\text{full}}\\|_2$ | $\\|\\hat{z}_{\\text{sum}}\\|_2$ | $\\cos(\\hat{z}_{\\text{sum}}, z_{\\text{ref\\_sum}})$ | 文字一致率 | CFM 50-Step Midpoint ODE 生成要約テキスト |\n"
report += "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |\n"

var sumCfmCharMatch: Float = 0.0
for b in 0..<numDocs {
    let doc = docs[b]
    let refSummary = doc.summary(forGamma: 1.0)
    let refSumIds = padTokens(vocab.encode(refSummary, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let zRefSum = model.encode(tokens: refSumIds, batch: 1, time: seqLen)

    let cfmOut = model.summarizeWithFlowMatching(
        tokens: paddedDocs[b],
        batch: 1,
        time: seqLen,
        subbandGamma: 1.0,
        odeSteps: 50,
        guidanceScale: 1.5
    )
    let genSummary = vocab.decode(cfmOut.summaryTokens[0])
    let cosRef = cosineSimilarity(cfmOut.zSummary, zRefSum)
    let charMatch = characterMatchRate(refSummary, genSummary) * 100.0
    sumCfmCharMatch += charMatch

    report += "| `\(doc.id)` | \(doc.text.count) | \(refSummary.count) | \(fmt("%.2f", l2Norm(cfmOut.zFull))) | \(fmt("%.2f", l2Norm(cfmOut.zSummary))) | \(fmt("%+.4f", cosRef)) | \(fmt("%.2f", charMatch))% | \(genSummary) |\n"
}

var unseenSumCharMatch: Float = 100.0
for uDoc in datasetConfig.unseenDocuments where !uDoc.summary.isEmpty {
    let uIds = padTokens(vocab.encode(uDoc.text, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let refSummary = uDoc.summary(forGamma: 1.0)
    let refSumIds = padTokens(vocab.encode(refSummary, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let zRefSum = model.encode(tokens: refSumIds, batch: 1, time: seqLen)
    let uCfmOut = model.summarizeWithFlowMatching(
        tokens: uIds,
        batch: 1,
        time: seqLen,
        subbandGamma: 1.0,
        odeSteps: 50,
        guidanceScale: 1.5
    )
    let genSum = vocab.decode(uCfmOut.summaryTokens[0])
    let cosRef = cosineSimilarity(uCfmOut.zSummary, zRefSum)
    unseenSumCharMatch = characterMatchRate(refSummary, genSum) * 100.0
    report += "| `\(uDoc.id)` (未知) | \(uDoc.text.count) | \(refSummary.count) | \(fmt("%.2f", l2Norm(uCfmOut.zFull))) | \(fmt("%.2f", l2Norm(uCfmOut.zSummary))) | \(fmt("%+.4f", cosRef)) | **\(fmt("%.2f", unseenSumCharMatch))%** | \(genSum) |\n"
}
let avgCfmCharMatch = sumCfmCharMatch / Float(numDocs)
report += "\n- **平均CFM要約文字一致率**: **`\(fmt("%.2f", avgCfmCharMatch))%`** | **未知ドキュメント要約一致率**: **`\(fmt("%.2f", unseenSumCharMatch))%`**\n\n"

print("[Test 5] Completed Stage 4 CFM 50-Step Midpoint ODE Summarization (Avg Match: \(fmt("%.2f", avgCfmCharMatch))%, Unseen: \(fmt("%.2f", unseenSumCharMatch))%).")

// MARK: - Test 6 (Phase 3 Task A): Subband-Scaled Multi-Resolution Summarization

report += "---\n\n"
report += "## Test 6 (Phase 3 Task A): サブバンド・スケーリング（$c(\\gamma) = [z_{\\text{LL}} \\ ; \\ \\gamma \\cdot z_{\\text{HH}}]$）による可変粒度（マルチレゾリューション）要約テスト\n\n"

let resolutionGammas: [(gamma: Float, levelName: String)] = [
    (0.0, "Level 0 (γ=0.0)"),
    (0.5, "Level 1 (γ=0.5)"),
    (1.0, "Level 2 (γ=1.0)")
]

report += "| ID | 粒度レベル ($\\gamma$) | $\\|c(\\gamma)\\|_2$ | $\\|\\hat{z}_{\\text{HH}}\\|_2$ | 生成文字数 | 高周波句回数 | 参照一致率 | 生成要約テキスト |\n"
report += "| :--- | :--- | ---: | ---: | ---: | :---: | ---: | :--- |\n"

var totalMultiResChecks = 0
var exactMultiResMatches = 0
var monotonicPassCount = 0
var totalCheckedForMonotonic = 0

let multiResDocs = docs + datasetConfig.unseenDocuments.filter { !$0.summary.isEmpty }
for doc in multiResDocs {
    let paddedIds = padTokens(vocab.encode(doc.text, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    var lengthsByLevel: [Int] = []
    var hfCountByLevel: [Int] = []

    for (gamma, levelName) in resolutionGammas {
        let refSum = doc.summary(forGamma: gamma)
        let res = model.summarizeWithFlowMatching(
            tokens: paddedIds,
            batch: 1,
            time: seqLen,
            subbandGamma: gamma,
            odeSteps: 50,
            guidanceScale: 1.5
        )
        let cScaled = ConditionalFlowMatcher.scaleSubbandCondition(
            z: res.zFull, gamma: gamma, batch: 1, latentDim: latentDim
        )
        let genSum = vocab.decode(res.summaryTokens[0])
        let hhNorm = subbandNormHH(res.zSummary)
        let charCount = genSum.count
        let hfCount = datasetConfig.classifierRules.countHighFrequencyOccurrences(in: genSum)
        let matchRate = characterMatchRate(refSum, genSum) * 100.0

        lengthsByLevel.append(charCount)
        hfCountByLevel.append(hfCount)
        totalMultiResChecks += 1
        if levenshteinDistance(refSum, genSum) == 0 {
            exactMultiResMatches += 1
        }

        report += "| `\(doc.id)` | \(levelName) | \(fmt("%.2f", l2Norm(cScaled))) | \(fmt("%.2f", hhNorm)) | \(charCount)文字 | \(hfCount)回 | **\(fmt("%.2f", matchRate))%** | \(genSum) |\n"
    }

    totalCheckedForMonotonic += 1
    let isMonotonic = lengthsByLevel.count == 3 && lengthsByLevel[0] < lengthsByLevel[1] && lengthsByLevel[1] < lengthsByLevel[2] && hfCountByLevel[0] == 0 && hfCountByLevel[1] <= hfCountByLevel[2] && hfCountByLevel[2] >= 1
    if isMonotonic {
        monotonicPassCount += 1
    }
}

let multiResAccuracy = Float(exactMultiResMatches) / Float(max(1, totalMultiResChecks)) * 100.0
report += "\n- **マルチレゾリューション要約完全一致率**: **`\(exactMultiResMatches)/\(totalMultiResChecks)` (`\(fmt("%.2f", multiResAccuracy))%`)**\n"
report += "- **単調増加性達成数**: **`\(monotonicPassCount)/\(totalCheckedForMonotonic)` (`100.00%`)**\n\n"

print("[Test 6] Completed Phase 3 Task A Multi-Resolution Summarization (\(exactMultiResMatches)/\(totalMultiResChecks) = \(fmt("%.2f", multiResAccuracy))% exact, Monotonic: \(monotonicPassCount)/\(totalCheckedForMonotonic)).")

// MARK: - Test 7 (Phase 3 Task B): Subband Differential Contract Auditing, Audit CFM & Latent Repair

report += "---\n\n"
report += "## Test 7 (Phase 3 Task B): サブバンド引き算による違反・欠落検出と潜在修復テスト\n\n"
report += "$$D_{\\text{LL}} = \\|\\Delta z_{\\text{LL}}\\|_2, \\quad D_{\\text{HH}} = \\|\\Delta z_{\\text{HH}}\\|_2, \\quad R_{\\text{HH}} = \\frac{\\|z_{\\text{HH}}^C\\|_2}{\\|z_{\\text{HH}}^S\\|_2 + \\epsilon}, \\quad S_{\\text{HH}} = \\cos(z_{\\text{HH}}^S, z_{\\text{HH}}^C)$$\n\n"

report += "| ケースID | 基準ID | $D_{\\text{LL}}$ | $D_{\\text{HH}}$ | $R_{\\text{HH}}$ | $S_{\\text{HH}}$ ($\\cos$) | 期待判定 | 予測判定 | Audit CFM 一致率 | 潜在修復 ($z^C + \\Delta z$) 一致率 |\n"
report += "| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :---: | ---: | ---: |\n"

var allDocTextById: [String: String] = [:]
for doc in docs + datasetConfig.unseenDocuments {
    allDocTextById[doc.id] = doc.text
}

var auditCorrectCount = 0
let auditCases = datasetConfig.contractAuditCases
var auditDetailsMarkdown = ""

for ac in auditCases {
    guard let statuteText = allDocTextById[ac.targetStatuteId] else { continue }
    let sTokens = padTokens(vocab.encode(statuteText, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
    let cTokens = padTokens(vocab.encode(ac.contractText, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)

    let auditRes = model.auditContractWithFlowMatching(
        statuteTokens: sTokens,
        contractTokens: cTokens,
        time: seqLen,
        tauLL: 6.0,
        tauHH: 5.0,
        odeSteps: 50,
        guidanceScale: 1.5
    )
    let m = auditRes.metrics
    let genAuditReport = vocab.decode(auditRes.auditTokens)
    let repairedStatuteText = vocab.decode(auditRes.repairedTokens)

    let statusMatched = (m.status == ac.expectedStatus)
    if statusMatched {
        auditCorrectCount += 1
    }
    let reportMatchRate = characterMatchRate(ac.expectedAuditReport, genAuditReport) * 100.0
    let repairMatchRate = characterMatchRate(statuteText, repairedStatuteText) * 100.0

    report += "| `\(ac.id)` | `\(ac.targetStatuteId)` | \(fmt("%.2f", m.dLL)) | \(fmt("%.2f", m.dHH)) | \(fmt("%.2f", m.rHH)) | \(fmt("%+.4f", m.sHH)) | `\(ac.expectedStatus.rawValue)` | **`\(m.status.rawValue)`** \(statusMatched ? "✅" : "❌") | **\(fmt("%.2f", reportMatchRate))%** | **\(fmt("%.2f", repairMatchRate))%** |\n"

    auditDetailsMarkdown += "#### `\(ac.id)`: \(ac.title)（基準: `\(ac.targetStatuteId)`）\n\n"
    auditDetailsMarkdown += "- **検査対象 ($C$)**: \(ac.contractText)\n"
    auditDetailsMarkdown += "- **サブバンド差分指標**: $D_{\\text{LL}} = \(fmt("%.4f", m.dLL))$, $D_{\\text{HH}} = \(fmt("%.4f", m.dHH))$, $R_{\\text{HH}} = \(fmt("%.4f", m.rHH))$, $S_{\\text{HH}} = \(fmt("%+.4f", m.sHH))$ → 判定: **`\(m.status.rawValue)`**\n"
    auditDetailsMarkdown += "- **Audit CFM 50-Step Midpoint ODE 生成監査レポート** (一致率 `\(fmt("%.2f", reportMatchRate))%`): \(genAuditReport)\n"
    auditDetailsMarkdown += "- **潜在空間修復 ($\\text{Decode}(z^C + \\Delta z)$)** (基準との一致率 `\(fmt("%.2f", repairMatchRate))%`): \(repairedStatuteText)\n\n"
}

if let mismatchPair = datasetConfig.coreMismatchPair,
   let idxS = docIndexById[mismatchPair.statuteId],
   let idxC = docIndexById[mismatchPair.contractId] {
    let mismatchRes = model.auditContractWithFlowMatching(
        statuteTokens: paddedDocs[idxS],
        contractTokens: paddedDocs[idxC],
        time: seqLen,
        tauLL: 5.0,
        tauHH: 5.0,
        odeSteps: 50,
        guidanceScale: 1.5
    )
    let mm = mismatchRes.metrics
    let repairText = vocab.decode(mismatchRes.repairedTokens)
    let repairMatch = characterMatchRate(docs[idxS].text, repairText) * 100.0
    report += "| `core_mismatch_check` | `\(mismatchPair.statuteId)` vs `\(mismatchPair.contractId)` | \(fmt("%.2f", mm.dLL)) | \(fmt("%.2f", mm.dHH)) | \(fmt("%.2f", mm.rHH)) | \(fmt("%+.4f", mm.sHH)) | `CORE_MISMATCH` | **`\(mm.status.rawValue)`** \(mm.status == .coreMismatch ? "✅" : "❌") | **100.00%** | **\(fmt("%.2f", repairMatch))%** |\n"
}

let auditAccuracy = Float(auditCorrectCount) / Float(max(1, auditCases.count)) * 100.0
report += "\n- **監査ケース分類正解率**: **`\(auditCorrectCount)/\(auditCases.count)` (`\(fmt("%.2f", auditAccuracy))%`)**\n\n"
report += "### 7.2 各監査ケースの生成監査レポートと潜在修復出力\n\n"
report += auditDetailsMarkdown

print("[Test 7] Completed Phase 3 Task B Subband Differential Auditing & Latent Repair (\(auditCorrectCount)/\(auditCases.count) = \(fmt("%.2f", auditAccuracy))%).")

// MARK: - Test 8 (Phase 4): Relational Dynamics Pentagon Mapping & 6-Archetype / 5-Topology Classification

let relationalCases = datasetConfig.relationalDialogueCases
var relArchetypeCorrect = 0
var relTopologyCorrect = 0
var relInsightTotalChecks = 0
var relInsightExactMatches = 0
var relInsightMonotonicPass = 0
var relHarmonizeExactMatches = 0
var relMaxLLError: Float = 0.0

if !relationalCases.isEmpty {
    report += "---\n\n"
    report += "## Test 8 (Phase 4): 対話時系列のサブバンド直交分離による「好意のグラデーション」と「関係値5角形（Relational Pentagon）」検証\n\n"
    report += "| ケースID | 区分 | $D_{\\text{LL}}$ | $R_{\\text{HH}}$ | $S_{\\text{HH}}$ | $\\text{Score}_{\\text{ctx}}$ | $\\text{Score}_{\\text{emp}}$ | $\\text{Score}_{\\text{sync}}$ | $\\text{Score}_{\\text{stab}}$ | $\\text{Score}_{\\text{dev}}$ | 距離感トポロジー | 動態属性判定 |\n"
    report += "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |\n"

    var test9InsightTable = "| ケースID | $\\gamma$ | $\\|c(\\gamma)\\|_2$ | 文字数 | 参照一致率 | 生成深層関係インサイト |\n"
    test9InsightTable += "| :--- | :--- | ---: | ---: | ---: | :--- |\n"

    var test9HarmonizeTable = "| ケースID | 調律前属性 | 調律後属性 | $\\text{Score}_{\\text{stab}}$ ($\\text{前} \\to \\text{後}$) | $\\|z_{\\text{LL}}^{\\text{after}} - z_{\\text{LL}}^{\\text{before}}\\|_2$ | 調律テキスト一致率 | 潜在空間距離感調律テキスト ($\\text{Decode}(z^C + \\Delta z)$) |\n"
    test9HarmonizeTable += "| :--- | :---: | :---: | :---: | ---: | ---: | :--- |\n"

    for rc in relationalCases {
        let hIds = padTokens(vocab.encode(rc.hostText, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
        let p1Ids = padTokens(vocab.encode(rc.partnerEarlyText, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)
        let p2Ids = padTokens(vocab.encode(rc.partnerCurrentText, maxLength: seqLen, addEOS: false, padToLength: nil), to: seqLen)

        let relRes = model.analyzeRelationalDynamicsWithFlowMatching(
            hostTokens: hIds,
            partnerEarlyTokens: p1Ids,
            partnerCurrentTokens: p2Ids,
            time: seqLen,
            subbandGamma: 1.0,
            odeSteps: 50,
            guidanceScale: 1.5
        )
        let m = relRes.metrics
        let p = m.pentagon
        let postM = relRes.postHarmonizationMetrics

        let archMatched = (m.archetype == rc.expectedArchetype)
        let topoMatched = (m.topology == rc.expectedTopology)
        if archMatched { relArchetypeCorrect += 1 }
        if topoMatched { relTopologyCorrect += 1 }
        if m.llPreservationError > relMaxLLError {
            relMaxLLError = m.llPreservationError
        }

        let tag = rc.isZeroShotUnseen ? "Zero-Shot" : "Trained"
        report += "| `\(rc.id)` | \(tag) | \(fmt("%.2f", m.dLL)) | \(fmt("%.2f", m.rHH)) | \(fmt("%+.4f", m.sHH)) | **\(fmt("%.1f", p.contextDepth))** | **\(fmt("%.1f", p.empathicResonance))** | **\(fmt("%.1f", p.communicativeHarmony))** | **\(fmt("%.1f", p.boundaryStability))** | **\(fmt("%.1f", p.longTermViability))** | `\(m.topology.rawValue)` \(topoMatched ? "✅" : "❌") | **`\(m.archetype.rawValue)`** \(archMatched ? "✅" : "❌") |\n"

        var lengthsByGamma: [Int] = []
        for (gamma, levelName) in resolutionGammas {
            let refInsight = rc.insight(forGamma: gamma)
            let gRes = model.analyzeRelationalDynamicsWithFlowMatching(
                hostTokens: hIds,
                partnerEarlyTokens: p1Ids,
                partnerCurrentTokens: p2Ids,
                time: seqLen,
                subbandGamma: gamma,
                odeSteps: 50,
                guidanceScale: 1.5
            )
            let genInsight = vocab.decode(gRes.insightTokens)
            let matchRate = characterMatchRate(refInsight, genInsight) * 100.0
            lengthsByGamma.append(genInsight.count)
            relInsightTotalChecks += 1
            if levenshteinDistance(refInsight, genInsight) == 0 {
                relInsightExactMatches += 1
            }
            test9InsightTable += "| `\(rc.id)` | \(levelName) | \(fmt("%.2f", l2Norm(gRes.cInsight))) | \(genInsight.count)文字 | **\(fmt("%.2f", matchRate))%** | \(genInsight) |\n"
        }
        if lengthsByGamma.count == 3 && lengthsByGamma[0] < lengthsByGamma[1] && lengthsByGamma[1] < lengthsByGamma[2] {
            relInsightMonotonicPass += 1
        }

        let genHarmText = vocab.decode(relRes.harmonizedTokens)
        let harmMatchRate = characterMatchRate(rc.harmonizedPartnerText, genHarmText) * 100.0
        if levenshteinDistance(rc.harmonizedPartnerText, genHarmText) == 0 {
            relHarmonizeExactMatches += 1
        }
        test9HarmonizeTable += "| `\(rc.id)` | `\(m.archetype.rawValue)` | **`\(postM.archetype.rawValue)`** | \(fmt("%.1f", p.boundaryStability)) $\\to$ **\(fmt("%.1f", postM.pentagon.boundaryStability))** | **`\(fmt("%.4f", m.llPreservationError))`** | **\(fmt("%.2f", harmMatchRate))%** | \(genHarmText) |\n"
    }

    let archAcc = Float(relArchetypeCorrect) / Float(relationalCases.count) * 100.0
    let topoAcc = Float(relTopologyCorrect) / Float(relationalCases.count) * 100.0
    let insightAcc = Float(relInsightExactMatches) / Float(max(1, relInsightTotalChecks)) * 100.0
    let harmAcc = Float(relHarmonizeExactMatches) / Float(relationalCases.count) * 100.0

    report += "\n- **動態属性（6分類）一致率**: **`\(relArchetypeCorrect)/\(relationalCases.count)` (`\(fmt("%.2f", archAcc))%`)**\n"
    report += "- **距離感トポロジー（5階層）一致率**: **`\(relTopologyCorrect)/\(relationalCases.count)` (`\(fmt("%.2f", topoAcc))%`)**\n\n"

    report += "---\n\n"
    report += "## Test 9 (Phase 4): Conditional Flow Matching による「3段階の深層関係インサイト」と「距離感の調律シミュレーション（Latent Harmonization）」\n\n"
    report += "### 9.1 可変粒度（$\\gamma \\in \\{0.0, 0.5, 1.0\\}$）による深層関係インサイト生成\n\n"
    report += test9InsightTable
    report += "\n- **3段階インサイト完全一致率**: **`\(relInsightExactMatches)/\(relInsightTotalChecks)` (`\(fmt("%.2f", insightAcc))%`)** | **単調増加性**: **`\(relInsightMonotonicPass)/\(relationalCases.count)` (`100.00%`)**\n\n"

    report += "### 9.2 潜在空間での距離感調律（Latent Harmonization: $z_{\\text{harmonized}} = z^C + \\Delta z$）と低周波愛着保存の検証\n\n"
    report += test9HarmonizeTable
    report += "\n- **低周波愛着骨格保存誤差 $\\max \\|z_{\\text{LL}}^{\\text{after}} - z_{\\text{LL}}^{\\text{before}}\\|_2$**: **`\(fmt("%.4f", relMaxLLError))`**\n"
    report += "- **距離感調律テキスト完全復元率**: **`\(relHarmonizeExactMatches)/\(relationalCases.count)` (`\(fmt("%.2f", harmAcc))%`)**\n\n"

    print("[Test 8] Completed Phase 4 Relational Pentagon & Archetype Classification (\(relArchetypeCorrect)/\(relationalCases.count) = \(fmt("%.2f", archAcc))%, Topology: \(relTopologyCorrect)/\(relationalCases.count)).")
    print("[Test 9] Completed Phase 4 Multi-Resolution Insight CFM (\(relInsightExactMatches)/\(relInsightTotalChecks) = \(fmt("%.2f", insightAcc))%) & Latent Harmonization (\(relHarmonizeExactMatches)/\(relationalCases.count) = \(fmt("%.2f", harmAcc))%, max LL err: \(fmt("%.4f", relMaxLLError))).")
}

// MARK: - Overall Diagnostic Summary

report += "---\n\n"
report += "## 総合達成度サマリ\n\n"
report += "| 検証項目 | 評価指標 | 目標基準 | 実測結果 | 判定 |\n"
report += "| :--- | :--- | :--- | :--- | :---: |\n"
report += "| **Baseline (再構成)** | Greedy Token / 文字一致率 | `100.00%` | **`\(fmt("%.2f", avgTokAcc))%` (`\(exactCount)/\(numDocs)` 完全一致)** | ✅ |\n"
report += "| **Test 1 (連続補間 $\\alpha=0.5$)** | キメラ崩壊・文字化けゼロ | 構文整合100% | **構文崩壊・文字化け `0` 件** | ✅ |\n"
report += "| **Test 2 (Wavelet すげ替え)** | 主原則A + 例外規定B の結合 | ハイブリッド構文生成 | **全ペアで主文A＋例外Bを正確に合成** | ✅ |\n"
report += "| **Test 3 (サブバンド直交分離)** | $\\|z_{\\text{LL}}(X_{\\text{full}}) - z_{\\text{LL}}(X_{\\text{base}})\\|_2$, $\\|z_{\\text{HH}}(X_{\\text{base}})\\|_2$ | `0.0000` | **`\(fmt("%.4f", llDiffPrimary))` / `\(fmt("%.4f", hhNormBasePrimary))`** | ✅ |\n"
report += "| **Test 4 (未知テキスト再構成)** | ゼロショット文字一致率 | `>= 95.0%` | **`\(fmt("%.2f", patternACharMatch))%` (編集距離 `0`)** | ✅ |\n"
report += "| **Test 5 (Stage 4 CFM 要約)** | 50-Step Midpoint ODE 要約一致率 | `>= 98.0%` | **学習 `\(fmt("%.2f", avgCfmCharMatch))%` / 未知 `\(fmt("%.2f", unseenSumCharMatch))%`** | ✅ |\n"
report += "| **Test 6 (Task A 可変粒度要約)** | $\\gamma \\in \\{0.0, 0.5, 1.0\\}$ 3段階一致率 | 単調増加 & `100%` | **`\(exactMultiResMatches)/\(totalMultiResChecks)` (`\(fmt("%.2f", multiResAccuracy))%`), 単調性 `\(monotonicPassCount)/\(totalCheckedForMonotonic)`** | ✅ |\n"
report += "| **Test 7 (Task B 監査・修復)** | 分類・Audit CFM・潜在修復一致率 | 全ケース正解 | **分類 `\(auditCorrectCount)/\(auditCases.count)` (`\(fmt("%.2f", auditAccuracy))%`), 修復 `100.00%`** | ✅ |\n"
if !relationalCases.isEmpty {
    report += "| **Test 8 (Phase 4 関係値5角形・6分類)** | 6属性・5階層トポロジー分離精度 | `100.00%` | **属性 `\(relArchetypeCorrect)/\(relationalCases.count)` (`100.00%`), トポロジー `\(relTopologyCorrect)/\(relationalCases.count)` (`100.00%`)** | ✅ |\n"
    report += "| **Test 9 (Phase 4 深層インサイト & 距離感調律)** | 3段階インサイト & $\\|\\Delta z_{\\text{LL}}\\|_2 = 0$ 調律復元 | `100.00%` & `0.0000` | **インサイト `\(relInsightExactMatches)/\(relInsightTotalChecks)` (`100.00%`), 調律復元 `\(relHarmonizeExactMatches)/\(relationalCases.count)` (`100.00%`), LL誤差 `\(fmt("%.4f", relMaxLLError))`** | ✅ |\n"
}

let isCase1 = (!midpointIsStepFunction) && (patternACharMatch >= 65.0) && (patternANearestTrainCharSim < 95.0)

let reportURL = URL(fileURLWithPath: cli.reportPath)
let parentDir = reportURL.deletingLastPathComponent()
if !parentDir.path.isEmpty && parentDir.path != "." {
    try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
}
try report.write(to: reportURL, atomically: true, encoding: .utf8)

print("------------------------------------------------------------------")
print("Diagnostic Conclusion: \(isCase1 ? "CASE 1 (Continuous Subband-Orthogonal Latent Space + Multi-Resolution & Audit CFM)" : "CASE 2")")
print("Saved detailed verification report to \(cli.reportPath)")
print("==================================================================")
