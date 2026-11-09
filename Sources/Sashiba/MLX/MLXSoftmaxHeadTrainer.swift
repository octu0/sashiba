import Foundation
import MLX

extension SoftmaxClassifierHead {
    /// MLX GPU version of `SoftmaxClassifierHead.train`: identical objective (standardized features,
    /// softmax cross-entropy + L2 weight decay, full-batch Adam), computed with closed-form gradients on GPU.
    public static func trainMLX(
        features: [[Float]],
        labels: [Int],
        classCount: Int,
        steps: Int = 300,
        learningRate: Float = 0.05,
        l2: Float = 2.5e-3
    ) -> SoftmaxClassifierHead {
        precondition(!features.isEmpty && features.count == labels.count)
        let n = features.count
        let f = features[0].count

        let x = MLXArray(features.flatMap { $0 }, [n, f])
        let mean = MLX.mean(x, axis: 0, keepDims: true)
        let centered = x - mean
        let std = maximum(sqrt(MLX.mean(centered * centered, axis: 0, keepDims: true)), MLXArray(Float(1e-4)))
        let xs = centered / std
        var oneHotFlat = [Float](repeating: 0.0, count: n * classCount)
        for i in 0..<n {
            oneHotFlat[i * classCount + labels[i]] = 1.0
        }
        let y = MLXArray(oneHotFlat, [n, classCount])

        var w = MLXArray.zeros([classCount, f])
        var b = MLXArray.zeros([classCount])
        var mW = MLXArray.zeros([classCount, f])
        var vW = MLXArray.zeros([classCount, f])
        var mB = MLXArray.zeros([classCount])
        var vB = MLXArray.zeros([classCount])
        let beta1: Float = 0.9
        let beta2: Float = 0.999
        let eps: Float = 1e-8
        let invN = 1.0 / Float(n)

        for step in 1...max(1, steps) {
            let logits = matmul(xs, w.transposed(1, 0)) + b
            let probs = MLX.softmax(logits, axis: -1)
            let err = (probs - y) * invN
            let gW = matmul(err.transposed(1, 0), xs) + l2 * w
            let gB = MLX.sum(err, axis: 0)
            let bc1 = 1.0 - powf(beta1, Float(step))
            let bc2 = 1.0 - powf(beta2, Float(step))
            mW = beta1 * mW + (1.0 - beta1) * gW
            vW = beta2 * vW + (1.0 - beta2) * gW * gW
            w = w - learningRate * (mW / bc1) / (sqrt(vW / bc2) + eps)
            mB = beta1 * mB + (1.0 - beta1) * gB
            vB = beta2 * vB + (1.0 - beta2) * gB * gB
            b = b - learningRate * (mB / bc1) / (sqrt(vB / bc2) + eps)
            if step % 25 == 0 {
                eval(w, b, mW, vW, mB, vB)
            }
        }
        eval(w, b, mean, std)
        return SoftmaxClassifierHead(
            classCount: classCount,
            featureDim: f,
            featureMean: mean.reshaped([f]).asArray(Float.self),
            featureStd: std.reshaped([f]).asArray(Float.self),
            weights: w.asArray(Float.self),
            bias: b.asArray(Float.self)
        )
    }
}
