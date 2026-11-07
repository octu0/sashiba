import Foundation
import Sashiba

struct ReconstructCLI {
    var dataPath: String = ".tmp"
    var weightsPath: String = ".tmp/weights.safetensors"
    var maxSeqLen: Int = 512

    static func parse(_ args: [String]) -> ReconstructCLI {
        var cli = ReconstructCLI()
        var i = 1
        while i < args.count {
            switch args[i] {
            case "-d", "--data", "--dataset":
                i += 1
                if i < args.count { cli.dataPath = args[i] }
            case "-w", "--weights", "--import-weights":
                i += 1
                if i < args.count { cli.weightsPath = args[i] }
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

let cli = ReconstructCLI.parse(CommandLine.arguments)
let weightsURL = URL(fileURLWithPath: cli.weightsPath)
let weights = try SWTHawkWeights.load(from: weightsURL)
let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)

let docs = try TextVocabulary.loadDocuments(from: URL(fileURLWithPath: cli.dataPath))
guard !docs.isEmpty else {
    print("No documents found at \(cli.dataPath)")
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

var allTokens: [Int] = []
for ids in encodedDocs {
    var row = Array(ids.prefix(seqLen))
    if row.count < seqLen {
        row.append(contentsOf: repeatElement(TextVocabulary.padId, count: seqLen - row.count))
    }
    allTokens.append(contentsOf: row)
}

let model = SWTHawkAutoencoder(weights: weights)
let numDocs = docs.count
let z = model.encode(tokens: allTokens, batch: numDocs, time: seqLen)
let reconstructedIds = model.reconstructGreedy(z: z, batch: numDocs, length: seqLen)

print("==================================================================")
print("SWT-Hawk Text Autoencoder Reconstruction from z ∈ R^\(weights.latentDim)")
print("==================================================================")

var totalValid = 0
var totalCorrect = 0
var exactDocs = 0

for b in 0..<numDocs {
    let zSlice = Array(z[(b * weights.latentDim)..<((b + 1) * weights.latentDim)])
    let norm = zSlice.withUnsafeBufferPointer { sqrtf(VectorOperations.sumOfSquares(ptr: $0.baseAddress!, count: $0.count)) }
    let targetIds = Array(allTokens[(b * seqLen)..<((b + 1) * seqLen)])
    let predIds = reconstructedIds[b]

    var docCorrect = 0
    for t in 0..<seqLen {
        totalValid += 1
        if predIds[t] == targetIds[t] {
            docCorrect += 1
            totalCorrect += 1
        }
    }
    let isExact = (docCorrect == seqLen)
    if isExact { exactDocs += 1 }

    let origText = vocab.decode(targetIds)
    let recText = vocab.decode(predIds)
    let previewOrig = String(origText.prefix(72))
    let previewRec = String(recText.prefix(72))
    let zHead = zSlice.prefix(4).map { String(format: "%+.3f", $0) }.joined(separator: ", ")

    print(String(
        format: "[%02d] %-16@ | ||z||_2 = %6.2f | z[0..3] = [%@] | Match: %3d/%3d (%@)",
        b + 1,
        (docs[b].id as NSString),
        norm,
        zHead,
        docCorrect,
        seqLen,
        isExact ? "EXACT" : "DIFF"
    ))
    print("     Orig: \(previewOrig)...")
    print("     Rec : \(previewRec)...")
}

let tokAcc = Float(totalCorrect) / Float(max(1, totalValid)) * 100.0
let docAcc = Float(exactDocs) / Float(max(1, numDocs)) * 100.0
print("------------------------------------------------------------------")
print(String(format: "Summary: Token Accuracy = %.2f%% | Exact Document Match = %d/%d (%.1f%%)", tokAcc, exactDocs, numDocs, docAcc))
