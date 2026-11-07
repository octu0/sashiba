import Foundation
import Testing
import MLX
import MLXNN
@testable import Sashiba

@Suite("Step 1: 1D-SWT (db1 / Haar) Frequency Branch Verification")
struct Step1SWTBranchTests {
    @Test("Step 1.1: Dummy tensor (B=2, T=128, D=64) Level 1 & Level 2 decomposition preserves shape (2, 128, 64) and is invertible")
    func swtDecompositionShapeAndInverse() {
        let batch = 2
        let time = 128
        let dim = 64
        let count = batch * time * dim

        var rng = SplitMix64(seed: 101)
        var x = [Float](repeating: 0.0, count: count)
        for i in 0..<count {
            x[i] = rng.nextSignedUniform(scale: 1.0)
        }

        let decomp = SWTFrequencyBranch.decompose(x: x, batch: batch, time: time, dim: dim)

        // Verify CA1, CD1, CA2, CD2 all preserve shape (B=2, T=128, D=64)
        #expect(decomp.ca1.count == count)
        #expect(decomp.cd1.count == count)
        #expect(decomp.ca2.count == count)
        #expect(decomp.cd2.count == count)

        // Verify exact reconstruction from (CA2, CD2, CD1) -> CA1 -> X
        let reconCA1 = SWTFrequencyBranch.reconstructLevel1(ca1: decomp.ca2, cd1: decomp.cd2)
        let reconstructed = SWTFrequencyBranch.reconstructLevel1(ca1: reconCA1, cd1: decomp.cd1)
        #expect(reconstructed.count == count)
        var maxErr: Float = 0.0
        for i in 0..<count {
            let err = abs(reconstructed[i] - x[i])
            if err > maxErr { maxErr = err }
        }
        #expect(maxErr < 1e-5)
    }

    @Test("Step 1.2: SWTFrequencyBranch CPU forward matches MLXSWTFrequencyBranch and MLX autograd propagates gradients on (B=2, T=128, D=64)")
    func swtForwardAndMLXBackward() {
        let batch = 2
        let time = 128
        let dim = 64
        let count = batch * time * dim

        var rng = SplitMix64(seed: 202)
        let weights = SWTBranchWeights.makeRandom(dim: dim, rng: &rng)
        var x = [Float](repeating: 0.0, count: count)
        for i in 0..<count {
            x[i] = rng.nextSignedUniform(scale: 0.8)
        }

        let cpuOut = SWTFrequencyBranch.forward(x: x, weights: weights, batch: batch, time: time)
        #expect(cpuOut.count == count)

        let mlxBranch = MLXSWTFrequencyBranch(weights: weights)
        let xArr = MLXArray(x, [batch, time, dim])
        let mlxOut = mlxBranch(xArr)
        eval(mlxOut)
        let gpuFloats = mlxOut.asArray(Float.self)
        #expect(gpuFloats.count == count)

        var maxDiff: Float = 0.0
        for i in 0..<count {
            let diff = abs(cpuOut[i] - gpuFloats[i])
            if diff > maxDiff { maxDiff = diff }
        }
        #expect(maxDiff < 1e-3)

        let lossAndGrad = valueAndGrad(model: mlxBranch) { (branch: MLXSWTFrequencyBranch, inputs: [MLXArray]) -> [MLXArray] in
            let y = branch(inputs[0])
            return [sum(y * y)]
        }
        let (losses, grads) = lossAndGrad(mlxBranch, [xArr])
        eval(losses[0], grads)
        #expect(losses[0].item(Float.self) > 0.0)
        #expect(!grads.flattened().isEmpty)
    }
}
