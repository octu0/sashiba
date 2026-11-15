import Foundation
import MLX
import Sashiba

/// 12-dimensional surface-form baseline featurizer:
/// character counts, length ratios, sentence-length ratios, punctuation (`！`, `？`, `、`) frequencies,
/// and Unicode script ratios (Hiragana / Kanji / Katakana) — zero lexical or semantic features.
struct SurfaceStatsFeaturizer: Sendable {
    static let dim: Int = 12

    private static func meanSentenceLength(_ text: String) -> Float {
        let terminators: Set<Character> = ["。", "！", "？", "!", "?"]
        var sentences = 0
        var chars = 0
        var cur = 0
        for ch in text {
            if ch.isWhitespace || ch.isNewline { continue }
            chars += 1
            cur += 1
            if terminators.contains(ch) {
                sentences += 1
                cur = 0
            }
        }
        if cur > 0 { sentences += 1 }
        return Float(chars) / Float(max(1, sentences))
    }

    static func features(sender: String, receiver: String) -> [Float] {
        let sChars = Array(sender)
        let rChars = Array(receiver)
        let lenS = max(1, sChars.count)
        let lenR = max(1, rChars.count)
        let invS = 1.0 / Float(lenS)
        let invR = 1.0 / Float(lenR)

        let mslS = max(1.0, meanSentenceLength(sender))
        let mslR = meanSentenceLength(receiver)

        var rExclam = 0
        var rQuestion = 0
        var rComma = 0
        var rHira = 0
        var rKanji = 0
        var rKata = 0
        for ch in rChars {
            if ch == "！" || ch == "!" { rExclam += 1 }
            if ch == "？" || ch == "?" { rQuestion += 1 }
            if ch == "、" || ch == "," { rComma += 1 }
            if let sc = ch.unicodeScalars.first {
                let v = sc.value
                if v >= 0x3040 && v <= 0x309F {
                    rHira += 1
                } else if v >= 0x4E00 && v <= 0x9FFF {
                    rKanji += 1
                } else if v >= 0x30A0 && v <= 0x30FF {
                    rKata += 1
                }
            }
        }

        var sPunct = 0
        for ch in sChars {
            if ch == "！" || ch == "!" || ch == "？" || ch == "?" || ch == "、" || ch == "," {
                sPunct += 1
            }
        }

        return [
            Float(sChars.count),
            Float(rChars.count),
            Float(rChars.count) * invS,
            mslR / mslS,
            mslR,
            Float(rExclam) * invR,
            Float(rQuestion) * invR,
            Float(rComma) * invR,
            Float(rHira) * invR,
            Float(rKanji) * invR,
            Float(rKata) * invR,
            Float(sPunct) * invS
        ]
    }
}

enum BenchmarkMath {
    static func pearsonR(_ xs: [Float], _ ys: [Float]) -> Float {
        let n = min(xs.count, ys.count)
        guard n >= 2 else { return 0.0 }
        let invN = 1.0 / Float(n)
        var mx: Float = 0.0
        var my: Float = 0.0
        for i in 0..<n {
            mx += xs[i]
            my += ys[i]
        }
        mx *= invN
        my *= invN
        var cov: Float = 0.0
        var vx: Float = 0.0
        var vy: Float = 0.0
        for i in 0..<n {
            let dx = xs[i] - mx
            let dy = ys[i] - my
            cov += dx * dy
            vx += dx * dx
            vy += dy * dy
        }
        let denom = sqrtf(max(1e-12, vx * vy))
        return cov / denom
    }

    static func meanAbsoluteError(_ preds: [Float], _ targets: [Float]) -> Float {
        let n = min(preds.count, targets.count)
        guard n > 0 else { return 0.0 }
        var sum: Float = 0.0
        for i in 0..<n {
            sum += abs(preds[i] - targets[i])
        }
        return sum / Float(n)
    }

    /// Exact tie-aware Mann-Whitney ROC-AUC for binary labels.
    static func rocAUC(scores: [Float], labels: [Bool]) -> Float {
        let n = min(scores.count, labels.count)
        var posScores: [Float] = []
        var negScores: [Float] = []
        for i in 0..<n {
            if labels[i] {
                posScores.append(scores[i])
            } else {
                negScores.append(scores[i])
            }
        }
        guard !posScores.isEmpty && !negScores.isEmpty else { return 0.5 }
        var wins: Double = 0.0
        for ps in posScores {
            for ns in negScores {
                if ps > ns {
                    wins += 1.0
                } else if ps == ns {
                    wins += 0.5
                }
            }
        }
        return Float(wins / Double(posScores.count * negScores.count))
    }

    /// Discretizes `(sLL, sHH, rHH)` into `binsPerAxis^3` cells over physical ranges
    /// `sLL ∈ [-1, 1], sHH ∈ [-1, 1], rHH ∈ [0, 1.8]` and returns `MI(X_bin; Y)` and `H(Y)` in bits (`log2`).
    static func discretizedSubbandMI(
        triples: [(sLL: Float, sHH: Float, rHH: Float)],
        labels: [Int],
        classCount: Int,
        binsPerAxis: Int = 4
    ) -> (miBits: Float, labelEntropyBits: Float) {
        let n = min(triples.count, labels.count)
        guard n > 0 else { return (0.0, 0.0) }

        func binIdx(_ val: Float, lo: Float, hi: Float) -> Int {
            let t = (val - lo) / max(1e-6, hi - lo)
            let b = Int(floorf(t * Float(binsPerAxis)))
            return max(0, min(binsPerAxis - 1, b))
        }

        let totalBins = binsPerAxis * binsPerAxis * binsPerAxis
        var joint = [Int](repeating: 0, count: totalBins * classCount)
        var xCounts = [Int](repeating: 0, count: totalBins)
        var yCounts = [Int](repeating: 0, count: classCount)

        for i in 0..<n {
            let y = labels[i]
            guard y >= 0 && y < classCount else { continue }
            let bx = binIdx(triples[i].sLL, lo: -1.0, hi: 1.0)
            let by = binIdx(triples[i].sHH, lo: -1.0, hi: 1.0)
            let bz = binIdx(triples[i].rHH, lo: 0.0, hi: 1.8)
            let cell = (bx * binsPerAxis + by) * binsPerAxis + bz
            joint[cell * classCount + y] += 1
            xCounts[cell] += 1
            yCounts[y] += 1
        }

        let invN = 1.0 / Double(n)
        var mi: Double = 0.0
        for cell in 0..<totalBins where xCounts[cell] > 0 {
            let px = Double(xCounts[cell]) * invN
            for y in 0..<classCount {
                let c = joint[cell * classCount + y]
                guard c > 0 && yCounts[y] > 0 else { continue }
                let pxy = Double(c) * invN
                let py = Double(yCounts[y]) * invN
                mi += pxy * log2(pxy / (px * py))
            }
        }

        var hy: Double = 0.0
        for y in 0..<classCount where yCounts[y] > 0 {
            let py = Double(yCounts[y]) * invN
            hy -= py * log2(py)
        }
        return (Float(max(0.0, mi)), Float(max(0.0, hy)))
    }

    static func stratifiedBinaryFolds(labels: [Bool], k: Int = 5, seed: UInt64 = 2026) -> [[Int]] {
        var rng = SplitMix64(seed: seed)
        var pos = labels.indices.filter { labels[$0] }
        var neg = labels.indices.filter { !labels[$0] }
        func shuffle(_ arr: inout [Int]) {
            guard arr.count > 1 else { return }
            for i in stride(from: arr.count - 1, to: 0, by: -1) {
                let j = Int(rng.next() % UInt64(i + 1))
                arr.swapAt(i, j)
            }
        }
        shuffle(&pos)
        shuffle(&neg)
        var folds = Array(repeating: [Int](), count: k)
        var rr = 0
        for idx in pos {
            folds[rr % k].append(idx)
            rr += 1
        }
        for idx in neg {
            folds[rr % k].append(idx)
            rr += 1
        }
        return folds.map { $0.sorted() }
    }

    static func stratifiedContinuousFolds(targets: [Float], k: Int = 5, seed: UInt64 = 2026) -> [[Int]] {
        var rng = SplitMix64(seed: seed)
        let sortedIdx = targets.indices.sorted { targets[$0] < targets[$1] }
        var folds = Array(repeating: [Int](), count: k)
        var offset = 0
        while offset < sortedIdx.count {
            let end = min(sortedIdx.count, offset + k)
            var chunk = Array(sortedIdx[offset..<end])
            if chunk.count > 1 {
                for i in stride(from: chunk.count - 1, to: 0, by: -1) {
                    let j = Int(rng.next() % UInt64(i + 1))
                    chunk.swapAt(i, j)
                }
            }
            for (f, idx) in chunk.enumerated() {
                folds[f % k].append(idx)
            }
            offset = end
        }
        return folds.map { $0.sorted() }
    }

    /// Solves `A x = b` (`n x n` dense system) using Gaussian elimination with partial pivoting.
    private static func solveLinearSystem(matrix: [Double], rhs: [Double], n: Int) -> [Double] {
        var a = matrix
        var b = rhs
        for col in 0..<n {
            var pivot = col
            var maxVal = abs(a[col * n + col])
            for row in (col + 1)..<n {
                let v = abs(a[row * n + col])
                if v > maxVal {
                    maxVal = v
                    pivot = row
                }
            }
            if pivot != col {
                for j in col..<n {
                    a.swapAt(col * n + j, pivot * n + j)
                }
                b.swapAt(col, pivot)
            }
            let diag = abs(a[col * n + col]) < 1e-12 ? 1e-12 : a[col * n + col]
            for row in (col + 1)..<n {
                let factor = a[row * n + col] / diag
                guard factor != 0.0 else { continue }
                for j in col..<n {
                    a[row * n + j] -= factor * a[col * n + j]
                }
                b[row] -= factor * b[col]
            }
        }
        var x = [Double](repeating: 0.0, count: n)
        for row in stride(from: n - 1, through: 0, by: -1) {
            var s = b[row]
            for j in (row + 1)..<n {
                s -= a[row * n + j] * x[j]
            }
            let diag = abs(a[row * n + row]) < 1e-12 ? 1e-12 : a[row * n + row]
            x[row] = s / diag
        }
        return x
    }

    /// Stratified 5-Fold L2-regularized Logistic Regression with inner 3-Fold L2 selection strictly on `trainX`.
    static func evaluateBinaryLogisticCV(
        features: [[Float]],
        labels: [Bool],
        folds: [[Int]]
    ) -> (rocAUC: Float, accuracy: Float) {
        let totalN = features.count
        let allSet = Set(0..<totalN)
        var oofScores = [Float](repeating: 0.5, count: totalN)
        var oofPreds = [Bool](repeating: false, count: totalN)
        let fDim = features.first?.count ?? 1
        let dimScale = Float(max(1, fDim)) / 128.0
        let candidateL2s: [Float] = [0.005, 0.05, 0.5].map { max(1e-3, $0 * dimScale) }

        for testIdx in folds {
            let testSet = Set(testIdx)
            let trainIdx = allSet.subtracting(testSet).sorted()
            let trainX = trainIdx.map { features[$0] }
            let trainYBool = trainIdx.map { labels[$0] }
            let trainYInt = trainYBool.map { $0 ? 1 : 0 }

            // Inner 3-fold CV inside trainIdx to pick L2 fairly across 12D, 128D, and 4096D features
            let innerFolds = stratifiedBinaryFolds(labels: trainYBool, k: 3, seed: 91)
            let innerSetAll = Set(0..<trainIdx.count)
            var bestL2 = candidateL2s[1]
            var bestInnerAUC: Float = -1.0
            for cand in candidateL2s {
                var innerScores = [Float](repeating: 0.5, count: trainIdx.count)
                for inTest in innerFolds {
                    let inTrain = innerSetAll.subtracting(Set(inTest)).sorted()
                    let h = SoftmaxClassifierHead.train(
                        features: inTrain.map { trainX[$0] },
                        labels: inTrain.map { trainYInt[$0] },
                        classCount: 2,
                        steps: 180,
                        learningRate: 0.05,
                        l2: cand
                    )
                    for idx in inTest {
                        innerScores[idx] = h.probabilities(trainX[idx])[1]
                    }
                }
                let auc = rocAUC(scores: innerScores, labels: trainYBool)
                if auc > bestInnerAUC {
                    bestInnerAUC = auc
                    bestL2 = cand
                }
            }

            let head = SoftmaxClassifierHead.train(
                features: trainX,
                labels: trainYInt,
                classCount: 2,
                steps: 240,
                learningRate: 0.05,
                l2: bestL2
            )
            for idx in testIdx {
                let p1 = head.probabilities(features[idx])[1]
                oofScores[idx] = p1
                oofPreds[idx] = p1 >= 0.5
            }
        }
        let auc = rocAUC(scores: oofScores, labels: labels)
        let correct = zip(oofPreds, labels).filter { $0 == $1 }.count
        return (auc, Float(correct) / Float(max(1, totalN)))
    }

    private static func fitAndPredictDualRidge(
        trainX: [[Float]],
        trainY: [Float],
        testX: [[Float]],
        lambda: Double
    ) -> [Float] {
        let n = trainX.count
        let f = trainX[0].count
        let invN = 1.0 / Double(n)

        var meanX = [Double](repeating: 0.0, count: f)
        var meanY: Double = 0.0
        for i in 0..<n {
            meanY += Double(trainY[i])
            for j in 0..<f {
                meanX[j] += Double(trainX[i][j])
            }
        }
        meanY *= invN
        for j in 0..<f { meanX[j] *= invN }

        var stdX = [Double](repeating: 0.0, count: f)
        for i in 0..<n {
            for j in 0..<f {
                let d = Double(trainX[i][j]) - meanX[j]
                stdX[j] += d * d
            }
        }
        for j in 0..<f {
            stdX[j] = max(1e-4, sqrt(stdX[j] * invN))
        }

        let zTrain: [[Double]] = trainX.map { row in
            (0..<f).map { (Double(row[$0]) - meanX[$0]) / stdX[$0] }
        }
        var gram = [Double](repeating: 0.0, count: n * n)
        for i in 0..<n {
            for k in i..<n {
                var dot: Double = 0.0
                for j in 0..<f {
                    dot += zTrain[i][j] * zTrain[k][j]
                }
                gram[i * n + k] = dot
                gram[k * n + i] = dot
            }
            gram[i * n + i] += lambda
        }
        let centeredY = trainY.map { Double($0) - meanY }
        let alpha = solveLinearSystem(matrix: gram, rhs: centeredY, n: n)

        return testX.map { row in
            var pred = meanY
            for i in 0..<n {
                var dot: Double = 0.0
                for j in 0..<f {
                    let zj = (Double(row[j]) - meanX[j]) / stdX[j]
                    dot += zj * zTrain[i][j]
                }
                pred += alpha[i] * dot
            }
            return Float(pred)
        }
    }

    /// Stratified 5-Fold Exact Dual Ridge Regression with inner 3-Fold lambda selection strictly on `trainX`.
    static func evaluateRidgeRegressionCV(
        features: [[Float]],
        targets: [Float],
        folds: [[Int]]
    ) -> (pearsonR: Float, mae: Float) {
        let totalN = features.count
        let allSet = Set(0..<totalN)
        var oofPreds = [Float](repeating: 0.0, count: totalN)
        let fDim = Double(max(1, features.first?.count ?? 1))
        let lambdas: [Double] = [0.2, 1.0, 5.0, 25.0, 125.0].map { $0 * (fDim / 128.0) }

        for testIdx in folds {
            let trainIdx = allSet.subtracting(Set(testIdx)).sorted()
            let trainX = trainIdx.map { features[$0] }
            let trainY = trainIdx.map { targets[$0] }

            let innerFolds = stratifiedContinuousFolds(targets: trainY, k: 3, seed: 97)
            let innerAll = Set(0..<trainIdx.count)
            var bestLambda = lambdas[2]
            var bestMAE: Float = .greatestFiniteMagnitude
            for cand in lambdas {
                var inPreds = [Float](repeating: 0.0, count: trainIdx.count)
                for inTest in innerFolds {
                    let inTrain = innerAll.subtracting(Set(inTest)).sorted()
                    let p = fitAndPredictDualRidge(
                        trainX: inTrain.map { trainX[$0] },
                        trainY: inTrain.map { trainY[$0] },
                        testX: inTest.map { trainX[$0] },
                        lambda: cand
                    )
                    for (k, idx) in inTest.enumerated() {
                        inPreds[idx] = p[k]
                    }
                }
                let mae = meanAbsoluteError(inPreds, trainY)
                if mae < bestMAE {
                    bestMAE = mae
                    bestLambda = cand
                }
            }

            let foldPreds = fitAndPredictDualRidge(
                trainX: trainX,
                trainY: trainY,
                testX: testIdx.map { features[$0] },
                lambda: bestLambda
            )
            for (k, idx) in testIdx.enumerated() {
                oofPreds[idx] = foldPreds[k]
            }
        }
        return (pearsonR(oofPreds, targets), meanAbsoluteError(oofPreds, targets))
    }
}
