import Foundation
import Testing
import MLX
import MLXNN
@testable import Sashiba

@Suite("Step 2: Hawk (RG-LRU) Temporal Branch Verification")
struct Step2HawkBlockTests {
    @Test("Step 2.1: Multi-timescale Lambda initialization covers short-term (0.5..0.9) and long-term (0.9..0.999) memory")
    func multiTimescaleLambdaInitialization() {
        let dim = 64
        var rng = SplitMix64(seed: 404)
        let weights = RGLRUWeights.makeRandom(dim: dim, rng: &rng)
        #expect(weights.lambda.count == dim)

        let half = dim / 2
        let halfC = RGLRUWeights.decayConstant * 0.5
        for i in 0..<half {
            let baseA = 1.0 / (1.0 + expf(-weights.lambda[i]))
            let effectiveA = powf(baseA, halfC)
            #expect(effectiveA >= 0.499 && effectiveA <= 0.901)
        }
        for i in half..<dim {
            let baseA = 1.0 / (1.0 + expf(-weights.lambda[i]))
            let effectiveA = powf(baseA, halfC)
            #expect(effectiveA >= 0.899 && effectiveA <= 0.9995)
        }
    }

    @Test("Step 2.2: HawkBlock CPU forward matches MLXHawkBlock and MLX autograd propagates gradients on (B=2, T=128, D=64)")
    func hawkBlockForwardAndMLXBackward() {
        let batch = 2
        let time = 128
        let dim = 64
        let count = batch * time * dim

        var rng = SplitMix64(seed: 505)
        let blockWeights = HawkBlockWeights.makeRandom(dim: dim, rng: &rng)

        var x = [Float](repeating: 0.0, count: count)
        for i in 0..<count {
            x[i] = rng.nextSignedUniform(scale: 0.5)
        }

        let (cpuOut, cpuHLast) = HawkBlock.forward(x: x, weights: blockWeights, batch: batch, time: time)
        #expect(cpuOut.count == count)
        #expect(cpuHLast.count == batch * dim)

        let mlxBlock = MLXHawkBlock(weights: blockWeights)
        let xArr = MLXArray(x, [batch, time, dim])
        let (mlxOut, mlxHLast) = mlxBlock.forward(xArr)
        eval(mlxOut, mlxHLast)

        let gpuOut = mlxOut.asArray(Float.self)
        let gpuHLast = mlxHLast.asArray(Float.self)
        var maxDiff: Float = 0.0
        for i in 0..<count {
            maxDiff = max(maxDiff, abs(cpuOut[i] - gpuOut[i]))
        }
        #expect(maxDiff < 1e-3)
        for i in 0..<(batch * dim) {
            #expect(abs(cpuHLast[i] - gpuHLast[i]) < 1e-3)
        }

        let lossAndGrad = valueAndGrad(model: mlxBlock) { (block: MLXHawkBlock, inputs: [MLXArray]) -> [MLXArray] in
            let (y, h) = block.forward(inputs[0])
            return [sum(y * y) + sum(h * h)]
        }
        let (losses, grads) = lossAndGrad(mlxBlock, [xArr])
        eval(losses[0], grads)
        #expect(losses[0].item(Float.self) > 0.0)
        #expect(!grads.flattened().isEmpty)
    }
}
