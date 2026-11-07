import Foundation
import Sashiba

struct BenchCLI {
    var batch: Int = 2
    var seqLen: Int = 512
    var dim: Int = SWTHawkShape.defaultDim
    var iterations: Int = 10

    static func parse(_ args: [String]) -> BenchCLI {
        var cli = BenchCLI()
        var i = 1
        while i < args.count {
            switch args[i] {
            case "-b", "--batch":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.batch = max(1, v) }
            case "-t", "--seq-len":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.seqLen = max(8, v) }
            case "--dim":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.dim = max(8, v) }
            case "-n", "--iterations":
                i += 1
                if i < args.count, let v = Int(args[i]) { cli.iterations = max(1, v) }
            default:
                break
            }
            i += 1
        }
        return cli
    }
}

let cli = BenchCLI.parse(CommandLine.arguments)
let vocabSize = 128
let weights = SWTHawkWeights.makeRandom(
    vocabSize: vocabSize,
    dim: cli.dim,
    latentDim: SWTHawkShape.latentDim,
    seed: 123
)
let model = SWTHawkAutoencoder(weights: weights)

var tokens = [Int](repeating: 0, count: cli.batch * cli.seqLen)
for i in 0..<tokens.count {
    tokens[i] = 4 + (i % (vocabSize - 4))
}

let zWarmup = model.encode(tokens: tokens, batch: cli.batch, time: cli.seqLen)
_ = model.reconstructGreedy(z: zWarmup, batch: cli.batch, length: cli.seqLen)

let t0 = Date()
var zLatest = zWarmup
for _ in 0..<cli.iterations {
    zLatest = model.encode(tokens: tokens, batch: cli.batch, time: cli.seqLen)
}
let encElapsed = Date().timeIntervalSince(t0)
let encMs = (encElapsed / Double(cli.iterations)) * 1000.0
let encTokPerSec = Double(cli.batch * cli.seqLen * cli.iterations) / max(encElapsed, 1e-9)

let t1 = Date()
for _ in 0..<cli.iterations {
    _ = model.reconstructGreedy(z: zLatest, batch: cli.batch, length: cli.seqLen)
}
let decElapsed = Date().timeIntervalSince(t1)
let decMs = (decElapsed / Double(cli.iterations)) * 1000.0
let decTokPerSec = Double(cli.batch * cli.seqLen * cli.iterations) / max(decElapsed, 1e-9)

print("==================================================================")
print("SWT-Hawk Text Autoencoder Benchmark")
print("==================================================================")
print("Model parameters: \(weights.parameterCount)")
print("Batch B = \(cli.batch) | Sequence Length T = \(cli.seqLen) | Dim D = \(cli.dim) | Latent z = \(weights.latentDim)")
print(String(format: "Encode (Tokens -> z ∈ R^1024) : %7.2f ms/iter (%8.0f tokens/s)", encMs, encTokPerSec))
print(String(format: "Decode (z ∈ R^1024 -> Tokens) : %7.2f ms/iter (%8.0f tokens/s)", decMs, decTokPerSec))
print("==================================================================")
