import Foundation

/// Multinomial logistic regression head trained on fixed feature vectors.
/// Features are standardized with the training mean / standard deviation stored in the head.
public struct SoftmaxClassifierHead: Codable, Sendable, Equatable {
    public let classCount: Int
    public let featureDim: Int
    public let featureMean: [Float]
    public let featureStd: [Float]
    public let weights: [Float]
    public let bias: [Float]

    public init(classCount: Int, featureDim: Int, featureMean: [Float], featureStd: [Float], weights: [Float], bias: [Float]) {
        self.classCount = classCount
        self.featureDim = featureDim
        self.featureMean = featureMean
        self.featureStd = featureStd
        self.weights = weights
        self.bias = bias
    }

    /// Trains the head with full-batch Adam on softmax cross-entropy plus L2 weight decay (optionally sample-weighted / class-balanced).
    public static func train(
        features: [[Float]],
        labels: [Int],
        classCount: Int,
        steps: Int = 300,
        learningRate: Float = 0.05,
        l2: Float = 1e-3,
        sampleWeights: [Float]? = nil,
        classBalanced: Bool = false
    ) -> SoftmaxClassifierHead {
        precondition(!features.isEmpty && features.count == labels.count)
        let n = features.count
        let f = features[0].count

        var weightsPerSample = [Float](repeating: 1.0, count: n)
        if let sw = sampleWeights, sw.count == n {
            weightsPerSample = sw
        }
        if classBalanced {
            var counts = [Int](repeating: 0, count: classCount)
            for y in labels where y >= 0 && y < classCount { counts[y] += 1 }
            let active = max(1, counts.filter { $0 > 0 }.count)
            let targetPerClass = Float(n) / Float(active)
            for i in 0..<n {
                let c = max(1, counts[labels[i]])
                let cb = min(3.0, max(0.5, sqrtf(targetPerClass / Float(c))))
                weightsPerSample[i] *= cb
            }
        }
        let wSum = max(1e-6, weightsPerSample.reduce(0.0, +))
        let invWSum = 1.0 / wSum

        var mean = [Float](repeating: 0.0, count: f)
        for x in features {
            for j in 0..<f {
                mean[j] += x[j]
            }
        }
        let invN = 1.0 / Float(n)
        for j in 0..<f {
            mean[j] *= invN
        }
        var std = [Float](repeating: 0.0, count: f)
        for x in features {
            for j in 0..<f {
                let d = x[j] - mean[j]
                std[j] += d * d
            }
        }
        for j in 0..<f {
            std[j] = max(1e-4, sqrtf(std[j] * invN))
        }
        var xFlat = [Float](repeating: 0.0, count: n * f)
        for i in 0..<n {
            let x = features[i]
            let base = i * f
            for j in 0..<f {
                xFlat[base + j] = (x[j] - mean[j]) / std[j]
            }
        }

        var w = [Float](repeating: 0.0, count: classCount * f)
        var b = [Float](repeating: 0.0, count: classCount)
        var mW = [Float](repeating: 0.0, count: w.count)
        var vW = [Float](repeating: 0.0, count: w.count)
        var mB = [Float](repeating: 0.0, count: classCount)
        var vB = [Float](repeating: 0.0, count: classCount)
        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let eps: Float = 1e-8

        var logits = [Float](repeating: 0.0, count: classCount)
        var probs = [Float](repeating: 0.0, count: classCount)
        var gW = [Float](repeating: 0.0, count: w.count)
        var gB = [Float](repeating: 0.0, count: classCount)

        xFlat.withUnsafeBufferPointer { xBuf in
            labels.withUnsafeBufferPointer { yBuf in
                weightsPerSample.withUnsafeBufferPointer { swBuf in
                    w.withUnsafeMutableBufferPointer { wBuf in
                        b.withUnsafeMutableBufferPointer { bBuf in
                            gW.withUnsafeMutableBufferPointer { gWBuf in
                                gB.withUnsafeMutableBufferPointer { gBBuf in
                                    mW.withUnsafeMutableBufferPointer { mWBuf in
                                        vW.withUnsafeMutableBufferPointer { vWBuf in
                                            mB.withUnsafeMutableBufferPointer { mBBuf in
                                                vB.withUnsafeMutableBufferPointer { vBBuf in
                                                    let xBase = xBuf.baseAddress!
                                                    let yBase = yBuf.baseAddress!
                                                    let swBase = swBuf.baseAddress!
                                                    let wBase = wBuf.baseAddress!
                                                    let bBase = bBuf.baseAddress!
                                                    let gWBase = gWBuf.baseAddress!
                                                    let gBBase = gBBuf.baseAddress!
                                                    let mWBase = mWBuf.baseAddress!
                                                    let vWBase = vWBuf.baseAddress!
                                                    let mBBase = mBBuf.baseAddress!
                                                    let vBBase = vBBuf.baseAddress!
                                                    let wCount = wBuf.count

                                                    for step in 1...max(1, steps) {
                                                        gWBase.update(repeating: 0.0, count: wCount)
                                                        gBBase.update(repeating: 0.0, count: classCount)

                                                        for i in 0..<n {
                                                            let xPtr = xBase + i * f
                                                            var maxLogit: Float = -Float.greatestFiniteMagnitude
                                                            for c in 0..<classCount {
                                                                let lg = bBase[c] + VectorOperations.dotProduct(a: wBase + c * f, b: xPtr, count: f)
                                                                logits[c] = lg
                                                                if lg > maxLogit { maxLogit = lg }
                                                            }
                                                            var expSum: Float = 0.0
                                                            for c in 0..<classCount {
                                                                let ev = expf(logits[c] - maxLogit)
                                                                probs[c] = ev
                                                                expSum += ev
                                                            }
                                                            let invExp = 1.0 / max(1e-12, expSum)
                                                            let sw = swBase[i] * invWSum
                                                            let yi = yBase[i]
                                                            for c in 0..<classCount {
                                                                let err = (probs[c] * invExp - (yi == c ? 1.0 : 0.0)) * sw
                                                                gBBase[c] += err
                                                                if abs(err) > 1e-12 {
                                                                    let dst = gWBase + c * f
                                                                    for j in 0..<f {
                                                                        dst[j] += err * xPtr[j]
                                                                    }
                                                                }
                                                            }
                                                        }

                                                        let bc1 = 1.0 - powf(beta1, Float(step))
                                                        let bc2 = 1.0 - powf(beta2, Float(step))
                                                        let lrEff = learningRate / bc1
                                                        let invBc2 = 1.0 / bc2
                                                        for k in 0..<wCount {
                                                            let g = gWBase[k] + l2 * wBase[k]
                                                            let mk = beta1 * mWBase[k] + (1.0 - beta1) * g
                                                            let vk = beta2 * vWBase[k] + (1.0 - beta2) * g * g
                                                            mWBase[k] = mk
                                                            vWBase[k] = vk
                                                            wBase[k] -= lrEff * mk / (sqrtf(vk * invBc2) + eps)
                                                        }
                                                        for c in 0..<classCount {
                                                            let g = gBBase[c]
                                                            let mc = beta1 * mBBase[c] + (1.0 - beta1) * g
                                                            let vc = beta2 * vBBase[c] + (1.0 - beta2) * g * g
                                                            mBBase[c] = mc
                                                            vBBase[c] = vc
                                                            bBase[c] -= lrEff * mc / (sqrtf(vc * invBc2) + eps)
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return SoftmaxClassifierHead(classCount: classCount, featureDim: f, featureMean: mean, featureStd: std, weights: w, bias: b)
    }

    public func probabilities(_ x: [Float]) -> [Float] {
        precondition(x.count == featureDim)
        var logits = [Float](repeating: 0.0, count: classCount)
        for c in 0..<classCount {
            var s = bias[c]
            let off = c * featureDim
            for j in 0..<featureDim {
                s += weights[off + j] * ((x[j] - featureMean[j]) / featureStd[j])
            }
            logits[c] = s
        }
        return Self.softmax(logits)
    }

    public func predict(_ x: [Float]) -> Int {
        let p = probabilities(x)
        var best = 0
        for c in 1..<p.count where p[c] > p[best] {
            best = c
        }
        return best
    }

    static func softmax(_ logits: [Float]) -> [Float] {
        let m = logits.max() ?? 0.0
        let e = logits.map { expf($0 - m) }
        let s = max(1e-12, e.reduce(0.0, +))
        return e.map { $0 / s }
    }

    private struct SendableScorePtr: @unchecked Sendable {
        let base: UnsafeMutablePointer<Float>
    }

    /// Pure-CPU Group 3-Fold CV hyperparameter-calibrated Softmax head (`0` GPU memory, parallelized across CPU cores).
    public static func trainGroupCV(
        features: [[Float]],
        labels: [Int],
        groups: [String],
        classCount: Int,
        steps: Int = 350
    ) -> SoftmaxClassifierHead {
        guard groups.count == features.count else {
            return train(features: features, labels: labels, classCount: classCount, steps: steps, l2: 0.05)
        }
        var gCounts: [String: Int] = [:]
        for g in groups { gCounts[g, default: 0] += 1 }
        let maxGC = Float(max(1, gCounts.values.max() ?? 1))
        let allGroupWeights = groups.map { powf(maxGC / Float(max(1, gCounts[$0] ?? 1)), 0.30) }
        let uniqueGroups = Array(Set(groups)).sorted()
        guard uniqueGroups.count >= 3 else {
            return train(features: features, labels: labels, classCount: classCount, steps: steps, l2: 0.05, sampleWeights: allGroupWeights)
        }

        // Stratify groups by majority label so every fold sees all classes
        var labelOfGroup: [String: Int] = [:]
        for (i, g) in groups.enumerated() where labelOfGroup[g] == nil {
            labelOfGroup[g] = labels[i]
        }
        let byLabel = Dictionary(grouping: uniqueGroups) { labelOfGroup[$0] ?? 0 }
        var groupFold: [String: Int] = [:]
        var rr = 0
        for lbl in byLabel.keys.sorted() {
            for g in (byLabel[lbl] ?? []).sorted() {
                groupFold[g] = rr % 3
                rr += 1
            }
        }
        let foldOfRow = groups.map { groupFold[$0] ?? 0 }
        var foldSplits: [(trX: [[Float]], trY: [Int], trW: [Float], vaX: [[Float]], vaY: [Int], vaW: [Float])] = []
        for fold in 0..<3 {
            var trX: [[Float]] = [], trY: [Int] = [], trW: [Float] = []
            var vaX: [[Float]] = [], vaY: [Int] = [], vaW: [Float] = []
            for i in 0..<features.count {
                if foldOfRow[i] == fold {
                    vaX.append(features[i]); vaY.append(labels[i]); vaW.append(allGroupWeights[i])
                } else {
                    trX.append(features[i]); trY.append(labels[i]); trW.append(allGroupWeights[i])
                }
            }
            if !trX.isEmpty && !vaX.isEmpty {
                foldSplits.append((trX, trY, trW, vaX, vaY, vaW))
            }
        }
        guard !foldSplits.isEmpty else {
            return train(features: features, labels: labels, classCount: classCount, steps: steps, l2: 0.05, sampleWeights: allGroupWeights)
        }

        let candidates: [(l2: Float, bal: Bool)] = [
            (0.01, false), (0.06, false), (0.35, false),
            (0.01, true),  (0.06, true),  (0.35, true)
        ]
        let rawScores = UnsafeMutablePointer<Float>.allocate(capacity: candidates.count)
        rawScores.initialize(repeating: -1e9, count: candidates.count)
        defer {
            rawScores.deinitialize(count: candidates.count)
            rawScores.deallocate()
        }
        let scorePtr = SendableScorePtr(base: rawScores)
        let splitsCopy = foldSplits
        DispatchQueue.concurrentPerform(iterations: candidates.count) { cIdx in
            let cand = candidates[cIdx]
            var wCorrect: Float = 0.0
            var wTotal: Float = 0.0
            var logProbSum: Float = 0.0
            for sp in splitsCopy {
                let h = Self.train(
                    features: sp.trX, labels: sp.trY, classCount: classCount,
                    steps: 100, learningRate: 0.08, l2: cand.l2,
                    sampleWeights: sp.trW, classBalanced: cand.bal
                )
                for i in 0..<sp.vaX.count {
                    let probs = h.probabilities(sp.vaX[i])
                    var pred = 0
                    for c in 1..<probs.count where probs[c] > probs[pred] { pred = c }
                    let rw = sp.vaW[i]
                    if pred == sp.vaY[i] { wCorrect += rw }
                    let pTrue = (sp.vaY[i] >= 0 && sp.vaY[i] < probs.count) ? max(1e-7, probs[sp.vaY[i]]) : 1e-7
                    logProbSum += rw * logf(pTrue)
                    wTotal += rw
                }
            }
            if wTotal > 0 {
                scorePtr.base[cIdx] = (wCorrect / wTotal) + 1e-4 * (logProbSum / wTotal)
            }
        }
        var bestIdx = 1
        for i in 0..<candidates.count where rawScores[i] > rawScores[bestIdx] {
            bestIdx = i
        }
        let best = candidates[bestIdx]
        return train(
            features: features, labels: labels, classCount: classCount,
            steps: min(steps, 240), learningRate: 0.07, l2: best.l2,
            sampleWeights: allGroupWeights, classBalanced: best.bal
        )
    }
}

/// Pure token-ID, topic-noun-swap-invariant hybrid relational descriptor (`160` dims for `z_rel ∈ R^128`):
/// combines `z_rel` (`128`), nonlinear subband geometry (`8`), cross-stream content-echo & subband rule modulation (`8`),
/// and non-lexical dialogue turn-taking / punctuation / sentence-trajectory cadence (`16`).
/// Requires ZERO GPU retraining and preserves Audit 1 (token-ID-only signature) & Audit 4 (topic-swap invariance).
public struct RelationalHybridFeatureExtractor: Sendable {
    public let questionIDs: Set<Int>
    public let exclamIDs: Set<Int>
    public let commaIDs: Set<Int>
    public let periodIDs: Set<Int>

    public init(vocabulary: TextVocabulary) {
        func ids(for chars: [String]) -> Set<Int> {
            var out: Set<Int> = []
            for ch in chars {
                let enc = vocabulary.encode(ch, addEOS: false)
                if enc.count == 1 && enc[0] >= TextVocabulary.specialCount {
                    out.insert(enc[0])
                }
            }
            return out
        }
        self.questionIDs = ids(for: ["？", "?"])
        self.exclamIDs = ids(for: ["！", "!", "♪"])
        self.commaIDs = ids(for: ["、", ","])
        self.periodIDs = ids(for: ["。", "."])
    }

    public func extract(
        senderTokens: [Int],
        receiverTokens: [Int],
        zRel: [Float],
        weights: SWTHawkWeights
    ) -> [Float] {
        let cleanS = senderTokens.filter { $0 >= TextVocabulary.specialCount }
        let cleanR = receiverTokens.filter { $0 >= TextVocabulary.specialCount }
        let alpha = weights.relational.alphaRel
        let subCls = weights.subbandClassifier

        // 1. Subband nonlinear geometry (8 dims)
        let p = DualStreamRelationalEngine.computePentagonFromZRel(zRel: zRel, zRelEarly: nil)
        let sLL = p.sLL
        let sHH = p.sHH
        let rHH = p.rHH
        let geomFeats: [Float] = [
            sLL,
            sHH,
            rHH,
            logf(max(0.05, rHH)),
            sLL * sHH,
            max(0.0, -sHH) * rHH,
            max(0.0, sHH) * max(0.0, rHH - 1.05),
            abs(rHH - 1.0)
        ]

        // 2. Topic-swap-invariant cross-stream content-echo & subband rule modulation (8 dims)
        var rareS: Set<Int> = []
        var rareR: Set<Int> = []
        var funcS = 0
        var funcR = 0
        for id in cleanS {
            let a = (id < alpha.count) ? alpha[id] : 0.04
            if a < 0.5 { rareS.insert(id) }
            if a >= 0.85 { funcS += 1 }
        }
        for id in cleanR {
            let a = (id < alpha.count) ? alpha[id] : 0.04
            if a < 0.5 { rareR.insert(id) }
            if a >= 0.85 { funcR += 1 }
        }
        let sharedRare = rareS.intersection(rareR).count
        let unionRare = rareS.union(rareR).count
        let novelRare = rareR.subtracting(rareS).count
        let rareEchoRecall = Float(sharedRare) / Float(max(1, rareS.count))
        let rareEchoJaccard = Float(sharedRare) / Float(max(1, unionRare))
        let novelRareRatio = Float(novelRare) / Float(max(1, cleanR.count))

        func funcBigrams(_ toks: [Int]) -> Set<UInt64> {
            let fToks = toks.filter { $0 < alpha.count && alpha[$0] >= 0.5 }
            guard fToks.count >= 2 else { return [] }
            var out: Set<UInt64> = []
            for i in 0..<(fToks.count - 1) {
                out.insert((UInt64(fToks[i]) << 32) | UInt64(fToks[i + 1]))
            }
            return out
        }
        let bgS = funcBigrams(cleanS)
        let bgR = funcBigrams(cleanR)
        let funcBigramEcho = Float(bgS.intersection(bgR).count) / Float(max(1, bgS.count))

        let sentsS = subCls.splitSentences(cleanS)
        let sentsR = subCls.splitSentences(cleanR)

        let echoFeats: [Float] = [
            rareEchoRecall,
            rareEchoJaccard,
            novelRareRatio,
            funcBigramEcho,
            p.pentagon.contextDepth * 0.01,
            p.pentagon.empathicResonance * 0.01,
            p.pentagon.communicativeHarmony * 0.01,
            p.pentagon.boundaryStability * 0.01
        ]

        // 3. Non-lexical dialogue turn-taking, punctuation & sentence-trajectory rhythm (16 dims)
        let lenS = max(1, cleanS.count)
        let lenR = max(1, cleanR.count)
        let invS = 1.0 / Float(lenS)
        let invR = 1.0 / Float(lenR)
        let meanSentS = Float(cleanS.count) / Float(max(1, sentsS.count))
        let meanSentR = Float(cleanR.count) / Float(max(1, sentsR.count))
        let firstLenR = Float(sentsR.first?.count ?? cleanR.count)
        let lastLenR = Float(sentsR.last?.count ?? cleanR.count)
        let sentAccelR = lastLenR / max(1.0, firstLenR)

        let qS = Float(cleanS.filter { questionIDs.contains($0) }.count) * invS
        let qR = Float(cleanR.filter { questionIDs.contains($0) }.count) * invR
        let exS = Float(cleanS.filter { exclamIDs.contains($0) }.count) * invS
        let exR = Float(cleanR.filter { exclamIDs.contains($0) }.count) * invR
        let comR = Float(cleanR.filter { commaIDs.contains($0) }.count) * invR
        let funcRatioS = Float(funcS) * invS
        let funcRatioR = Float(funcR) * invR
        let qIgnored = (qS > 0.0 ? 1.0 : 0.0) * max(0.0, 1.0 - Float(cleanR.count) / 28.0)

        let rhythmFeats: [Float] = [
            logf(1.0 + Float(cleanS.count)),
            logf(1.0 + Float(cleanR.count)),
            Float(cleanR.count) * invS,
            Float(cleanR.count - cleanS.count) / Float(lenS + lenR),
            meanSentR / max(1.0, meanSentS),
            sentAccelR,
            qS,
            qR,
            qR - qS,
            exS,
            exR,
            exR - exS,
            comR,
            funcRatioR,
            funcRatioR - funcRatioS,
            qIgnored
        ]

        var out: [Float] = []
        out.reserveCapacity(zRel.count + 32)
        out.append(contentsOf: zRel)
        out.append(contentsOf: geomFeats)
        out.append(contentsOf: echoFeats)
        out.append(contentsOf: rhythmFeats)
        return out
    }
}

