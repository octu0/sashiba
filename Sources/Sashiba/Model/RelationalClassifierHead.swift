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

    /// Trains the head with full-batch Adam on softmax cross-entropy plus L2 weight decay.
    public static func train(
        features: [[Float]],
        labels: [Int],
        classCount: Int,
        steps: Int = 300,
        learningRate: Float = 0.05,
        l2: Float = 1e-3
    ) -> SoftmaxClassifierHead {
        precondition(!features.isEmpty && features.count == labels.count)
        let n = features.count
        let f = features[0].count

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
        let xs: [[Float]] = features.map { x in
            (0..<f).map { (x[$0] - mean[$0]) / std[$0] }
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
        for step in 1...max(1, steps) {
            var gW = [Float](repeating: 0.0, count: w.count)
            var gB = [Float](repeating: 0.0, count: classCount)
            for i in 0..<n {
                let x = xs[i]
                for c in 0..<classCount {
                    var s = b[c]
                    let off = c * f
                    for j in 0..<f {
                        s += w[off + j] * x[j]
                    }
                    logits[c] = s
                }
                let probs = softmax(logits)
                for c in 0..<classCount {
                    let err = (probs[c] - (labels[i] == c ? 1.0 : 0.0)) * invN
                    gB[c] += err
                    let off = c * f
                    for j in 0..<f {
                        gW[off + j] += err * x[j]
                    }
                }
            }
            for k in 0..<w.count {
                gW[k] += l2 * w[k]
            }
            let bc1 = 1.0 - powf(beta1, Float(step))
            let bc2 = 1.0 - powf(beta2, Float(step))
            for k in 0..<w.count {
                mW[k] = beta1 * mW[k] + (1.0 - beta1) * gW[k]
                vW[k] = beta2 * vW[k] + (1.0 - beta2) * gW[k] * gW[k]
                w[k] -= learningRate * (mW[k] / bc1) / (sqrtf(vW[k] / bc2) + eps)
            }
            for c in 0..<classCount {
                mB[c] = beta1 * mB[c] + (1.0 - beta1) * gB[c]
                vB[c] = beta2 * vB[c] + (1.0 - beta2) * gB[c] * gB[c]
                b[c] -= learningRate * (mB[c] / bc1) / (sqrtf(vB[c] / bc2) + eps)
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
}

/// Hashed character uni-gram + bi-gram frequency features, L2-normalized. Used as a non-neural reference baseline.
public struct CharNGramFeaturizer: Sendable {
    public let dim: Int

    public init(dim: Int = 2048) {
        self.dim = dim
    }

    public func features(_ text: String) -> [Float] {
        var v = [Float](repeating: 0.0, count: dim)
        let scalars = text.unicodeScalars.map { UInt64($0.value) }
        for (i, s) in scalars.enumerated() {
            v[Int(Self.mix(s) % UInt64(dim))] += 1.0
            if i + 1 < scalars.count {
                let pair = (s << 21) ^ scalars[i + 1] ^ 0x9e37_79b9_7f4a_7c15
                v[Int(Self.mix(pair) % UInt64(dim))] += 1.0
            }
        }
        let norm = sqrtf(v.reduce(0.0) { $0 + $1 * $1 })
        guard norm > 0 else { return v }
        return v.map { $0 / norm }
    }

    static func mix(_ x: UInt64) -> UInt64 {
        var z = x &+ 0x9e37_79b9_7f4a_7c15
        z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z >> 31)
    }
}

