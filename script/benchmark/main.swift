import Foundation
import Sashiba

struct BenchCLI {
    var weightsPath: String = ".tmp/weights.safetensors"
    var dataPath: String = ".tmp"
    var iterations: Int = 40
    var unseenSampleCount: Int = 200
    var dumpSnapshotPath: String? = nil
    var verifySnapshotPath: String? = nil

    static func parse(_ args: [String]) -> BenchCLI {
        var cli = BenchCLI()
        var i = 1
        while i < args.count {
            switch args[i] {
            case "-w", "--weights":
                i += 1
                if i < args.count { cli.weightsPath = args[i] }
            case "-d", "--data":
                i += 1
                if i < args.count { cli.dataPath = args[i] }
            case "-n", "--iterations":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.iterations = max(5, v) }
            case "--unseen-samples":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.unseenSampleCount = max(10, v) }
            case "--dump-snapshot":
                i += 1
                if i < args.count { cli.dumpSnapshotPath = args[i] }
            case "--verify-snapshot":
                i += 1
                if i < args.count { cli.verifySnapshotPath = args[i] }
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

struct ParitySnapshotItem: Codable {
    let id: String
    let zRel: [Float]
    let hybridFeatures: [Float]
    let archT4: String
    let topoT4: String
    let hybArchT4: String
    let hybTopoT4: String
    let affectionT4: Int
    let frictionT4: Int
}

func percentile(_ sortedMs: [Double], _ q: Double) -> Double {
    guard !sortedMs.isEmpty else { return 0.0 }
    let pos = max(0.0, min(Double(sortedMs.count - 1), q * Double(sortedMs.count - 1)))
    let lo = Int(floor(pos))
    let hi = Int(ceil(pos))
    if lo == hi { return sortedMs[lo] }
    let frac = pos - Double(lo)
    return sortedMs[lo] * (1.0 - frac) + sortedMs[hi] * frac
}

@MainActor
func measureLatencies(warmup: Int = 3, iterations: Int, _ block: () -> Void) -> (meanMs: Double, p50Ms: Double, p95Ms: Double, pairsPerSec: Double) {
    for _ in 0..<warmup { block() }
    var samples: [Double] = []
    samples.reserveCapacity(iterations)
    let tAll0 = Date()
    for _ in 0..<iterations {
        let t0 = Date()
        block()
        samples.append(Date().timeIntervalSince(t0) * 1000.0)
    }
    let totalSec = max(1e-9, Date().timeIntervalSince(tAll0))
    samples.sort()
    let meanMs = samples.reduce(0.0, +) / Double(samples.count)
    return (meanMs, percentile(samples, 0.50), percentile(samples, 0.95), Double(iterations) / totalSec)
}

let cli = BenchCLI.parse(CommandLine.arguments)

print("==================================================================")
print("Sashiba v6 / v6_hybrid Pure-CPU Inference & Throughput Benchmark")
print("==================================================================")

if FileManager.default.fileExists(atPath: cli.weightsPath) {
    let tLoad0 = Date()
    let weights = try SWTHawkWeights.load(from: URL(fileURLWithPath: cli.weightsPath))
    let loadMs = Date().timeIntervalSince(tLoad0) * 1000.0
    let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)
    let model = SWTHawkAutoencoder(weights: weights)

    print(String(format: "Loaded weights: %@ (%.1f ms) | Params: %d | Vocab V = %d | Dim D = %d | z_rel = %d",
                 cli.weightsPath, loadMs, weights.parameterCount, vocab.count, weights.dim, weights.relational.latentDim))

    let workloads: [(label: String, sender: String, receiver: String)] = [
        (
            "Short (~32 chars, emoji/slang)",
            "今日も歌枠おつかれ〜！高音めっちゃ刺さった✨",
            "ありがとう〜！！喉ギリギリだったけど聴いてくれて嬉しい😭✨"
        ),
        (
            "Medium (~95 chars, 2-turn chat)",
            "昨日の弾き語りアーカイブ、移動中に3回聴き返したよ。アコギのアルペジオすごく丁寧だったね！週末の路上ライブも冷えるから暖かくしてね。",
            "3回も聴いてくれたの！？指先のタッチまで気づいてもらえてめちゃくちゃ報われる…🎸週末の駅前、カイロ多めに貼って頑張るね！"
        ),
        (
            "Long (~210 chars, 4-turn history)",
            "先月から現場の工程管理が立て込んでてリアタイ減っちゃってごめんね。でも毎週水曜の雑談だけは帰りの電車で必ず追ってるよ！新しい機材トラブル大変そうだったけど、無理しすぎないで自分のペースで続けてくれたら十分だからね。",
            "忙しい現場の合間に水曜の雑談まで追ってくれて本当にありがとう…！工程管理って気を張る場面ばっかりだと思うから、むしろそっちの体調が一番心配だよ💦機材の配線はやっと落ち着いたから安心してね、落ち着いたらまたゆっくり報告待ってる！"
        )
    ]

    print("\n--- [1] Single-Thread (batch = 1) Latency by Chat Length ---")
    for wl in workloads {
        let sIds = vocab.encode(wl.sender, addEOS: false)
        let rIds = vocab.encode(wl.receiver, addEOS: false)
        let encStat = measureLatencies(iterations: cli.iterations) {
            _ = model.encodeRelational(senderTokenIDs: sIds, receiverTokenIDs: rIds)
        }
        let infStat = measureLatencies(iterations: cli.iterations) {
            _ = model.inferRelational(senderTokenIDs: sIds, receiverTokenIDs: rIds)
        }
        print(String(format: "%@ (tokS=%d, tokR=%d):", wl.label, sIds.count, rIds.count))
        print(String(format: "  encodeRelational : mean %6.2f ms | p50 %6.2f ms | p95 %6.2f ms (%6.1f pairs/s)",
                     encStat.meanMs, encStat.p50Ms, encStat.p95Ms, encStat.pairsPerSec))
        print(String(format: "  inferRelational  : mean %6.2f ms | p50 %6.2f ms | p95 %6.2f ms (%6.1f pairs/s)",
                     infStat.meanMs, infStat.p50Ms, infStat.p95Ms, infStat.pairsPerSec))
    }

    let datasetConfig = DatasetConfiguration.load(from: cli.dataPath)
    let unseenCases = datasetConfig.relationalDialogueCases.filter { $0.isZeroShotUnseen }
    if !unseenCases.isEmpty {
        let sampleCount = min(cli.unseenSampleCount, unseenCases.count)
        let sampled = Array(unseenCases.prefix(sampleCount))
        let sampledIds: [(s: [Int], r: [Int])] = sampled.map {
            (vocab.encode($0.hostText, maxLength: 512, addEOS: false),
             vocab.encode($0.partnerCurrentText, maxLength: 512, addEOS: false))
        }

        // Warmup
        for i in 0..<min(5, sampledIds.count) {
            _ = model.inferRelational(senderTokenIDs: sampledIds[i].s, receiverTokenIDs: sampledIds[i].r)
        }

        var unseenLatencies: [Double] = []
        unseenLatencies.reserveCapacity(sampleCount)
        let tUnseen0 = Date()
        for pair in sampledIds {
            let t0 = Date()
            _ = model.inferRelational(senderTokenIDs: pair.s, receiverTokenIDs: pair.r)
            unseenLatencies.append(Date().timeIntervalSince(t0) * 1000.0)
        }
        let unseenTotalSec = max(1e-9, Date().timeIntervalSince(tUnseen0))
        unseenLatencies.sort()
        let unseenMean = unseenLatencies.reduce(0.0, +) / Double(unseenLatencies.count)
        print("\n--- [2] Single-Thread (batch = 1) Real Unseen Suite (\(sampleCount) v6 pairs) ---")
        print(String(format: "  inferRelational  : mean %6.2f ms | p50 %6.2f ms | p95 %6.2f ms (%6.1f pairs/s)",
                     unseenMean, percentile(unseenLatencies, 0.50), percentile(unseenLatencies, 0.95), Double(sampleCount) / unseenTotalSec))

        // Multi-core batch throughput across all unseen T2 + T4 pairs (754 * 2 = 1508 pairs)
        var maxLen = 4
        var allSenders: [[Int]] = []
        var allReceivers: [[Int]] = []
        allSenders.reserveCapacity(unseenCases.count * 2)
        allReceivers.reserveCapacity(unseenCases.count * 2)
        for rc in unseenCases {
            let s2 = vocab.encode(rc.hostEarlyText, maxLength: 512, addEOS: false)
            let r2 = vocab.encode(rc.partnerEarlyText, maxLength: 512, addEOS: false)
            let s4 = vocab.encode(rc.hostText, maxLength: 512, addEOS: false)
            let r4 = vocab.encode(rc.partnerCurrentText, maxLength: 512, addEOS: false)
            maxLen = max(maxLen, max(max(s2.count, r2.count), max(s4.count, r4.count)))
            allSenders.append(s2); allReceivers.append(r2)
            allSenders.append(s4); allReceivers.append(r4)
        }
        // Warmup 16 pairs
        _ = DualStreamRelationalEngine.encodeZRelCPUParallel(
            senders: Array(allSenders.prefix(16)),
            receivers: Array(allReceivers.prefix(16)),
            time: maxLen,
            weights: weights
        )
        let tPar0 = Date()
        let parZ = DualStreamRelationalEngine.encodeZRelCPUParallel(
            senders: allSenders,
            receivers: allReceivers,
            time: maxLen,
            weights: weights
        )
        let parSec = max(1e-9, Date().timeIntervalSince(tPar0))
        let parPairsPerSec = Double(parZ.count) / parSec
        let parEffMs = (parSec * 1000.0) / Double(parZ.count)
        print("\n--- [3] Multi-Core CPU Parallel Batch Throughput (\(parZ.count) pairs, maxSeqLen=\(maxLen)) ---")
        print(String(format: "  encodeZRelCPUParallel : total %7.1f ms | %7.1f pairs/s | effective %5.2f ms/pair",
                     parSec * 1000.0, parPairsPerSec, parEffMs))

        // Snapshot dump / verify for numerical parity
        let snapCases = Array(unseenCases.prefix(min(64, unseenCases.count)))
        if let dumpPath = cli.dumpSnapshotPath {
            var items: [ParitySnapshotItem] = []
            for rc in snapCases {
                let s = vocab.encode(rc.hostText, maxLength: 512, addEOS: false)
                let r = vocab.encode(rc.partnerCurrentText, maxLength: 512, addEOS: false)
                let out = model.inferRelational(senderTokenIDs: s, receiverTokenIDs: r)
                items.append(ParitySnapshotItem(
                    id: rc.id,
                    zRel: out.zRel,
                    hybridFeatures: out.hybridFeatures,
                    archT4: out.predictedArchetypeT4.rawValue,
                    topoT4: out.predictedTopologyT4.rawValue,
                    hybArchT4: out.hybridPredictedArchetypeT4.rawValue,
                    hybTopoT4: out.hybridPredictedTopologyT4.rawValue,
                    affectionT4: out.hybridPredictedEmotionT4.affection,
                    frictionT4: out.hybridPredictedEmotionT4.friction
                ))
            }
            let data = try JSONEncoder().encode(items)
            try data.write(to: URL(fileURLWithPath: dumpPath))
            print("\n[Snapshot] Dumped \(items.count) reference parity outputs to \(dumpPath)")
        }

        if let verifyPath = cli.verifySnapshotPath,
           let data = FileManager.default.contents(atPath: verifyPath) {
            let refItems = try JSONDecoder().decode([ParitySnapshotItem].self, from: data)
            let byId = Dictionary(uniqueKeysWithValues: snapCases.map { ($0.id, $0) })
            var maxZDiff: Float = 0.0
            var maxHybDiff: Float = 0.0
            var labelMatches = 0
            for ref in refItems {
                guard let rc = byId[ref.id] else { continue }
                let s = vocab.encode(rc.hostText, maxLength: 512, addEOS: false)
                let r = vocab.encode(rc.partnerCurrentText, maxLength: 512, addEOS: false)
                let out = model.inferRelational(senderTokenIDs: s, receiverTokenIDs: r)
                for (a, b) in zip(ref.zRel, out.zRel) {
                    maxZDiff = max(maxZDiff, abs(a - b))
                }
                for (a, b) in zip(ref.hybridFeatures, out.hybridFeatures) {
                    maxHybDiff = max(maxHybDiff, abs(a - b))
                }
                if ref.archT4 == out.predictedArchetypeT4.rawValue &&
                   ref.topoT4 == out.predictedTopologyT4.rawValue &&
                   ref.hybArchT4 == out.hybridPredictedArchetypeT4.rawValue &&
                   ref.hybTopoT4 == out.hybridPredictedTopologyT4.rawValue &&
                   ref.affectionT4 == out.hybridPredictedEmotionT4.affection &&
                   ref.frictionT4 == out.hybridPredictedEmotionT4.friction {
                    labelMatches += 1
                }
            }
            print(String(format: "\n[Parity Check vs %@] cases=%d | max|Δz_rel|=%.6e | max|Δhybrid|=%.6e | label agreement=%d/%d",
                         verifyPath, refItems.count, maxZDiff, maxHybDiff, labelMatches, refItems.count))
            if maxZDiff > 1e-4 || labelMatches != refItems.count {
                print("ERROR: Numerical parity check failed!")
                exit(2)
            }
        }
    }
}

// Synthetic SWT-Hawk Autoencoder Neural Path Benchmark (for pure neural backbone comparison)
let synVocab = 256
let synWeights = SWTHawkWeights.makeRandom(vocabSize: synVocab, dim: SWTHawkShape.defaultDim, latentDim: SWTHawkShape.latentDim, seed: 123)
let synModel = SWTHawkAutoencoder(weights: synWeights)
let synB = 2
let synT = 288
var synTokens = [Int](repeating: 0, count: synB * synT)
for i in 0..<synTokens.count { synTokens[i] = 4 + (i % (synVocab - 4)) }
let zWarm = synModel.encode(tokens: synTokens, batch: synB, time: synT)
_ = synModel.reconstructGreedy(z: zWarm, batch: synB, length: synT)

let aeIters = 10
let tAE0 = Date()
var zLat = zWarm
for _ in 0..<aeIters {
    zLat = synModel.encode(tokens: synTokens, batch: synB, time: synT)
}
let aeEncMs = (Date().timeIntervalSince(tAE0) / Double(aeIters)) * 1000.0
let tAE1 = Date()
for _ in 0..<aeIters {
    _ = synModel.reconstructGreedy(z: zLat, batch: synB, length: synT)
}
let aeDecMs = (Date().timeIntervalSince(tAE1) / Double(aeIters)) * 1000.0
print("\n--- [4] Pure Neural Autoencoder Backbone (B=\(synB), T=\(synT), D=\(SWTHawkShape.defaultDim)) ---")
print(String(format: "  AE Encode : %7.2f ms/iter | AE Greedy Decode : %7.2f ms/iter", aeEncMs, aeDecMs))
print("==================================================================")
