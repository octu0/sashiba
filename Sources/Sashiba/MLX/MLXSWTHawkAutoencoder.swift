import Foundation
import MLX
import MLXNN

/// Linear recurrence `h_t = a_t ⊙ h_{t-1} + x_t` over `[B, T, D]`, supporting non-zero initial state `h_0` `[B, D]`.
/// Uses a custom Metal scan kernel on macOS (when `h_0 == nil`) and a parallel associative scan (`O(log T)` tensor ops) on CUDA / other MLX backends.
public enum MLXLinearScan {
    #if os(macOS)
    static let forwardKernel = MLXFast.metalKernel(
        name: "sashiba_linear_scan_forward",
        inputNames: ["a", "x"],
        outputNames: ["h"],
        source: """
            uint gid = thread_position_in_grid.x;
            if (gid < uint(B * R)) {
                uint b = gid / R;
                uint r = gid - b * R;
                size_t base = size_t(b) * T * R + r;
                float state = 0.0f;
                for (int t = 0; t < T; ++t) {
                    size_t idx = base + size_t(t) * R;
                    state = a[idx] * state + x[idx];
                    h[idx] = state;
                }
            }
            """
    )

    static let backwardKernel = MLXFast.metalKernel(
        name: "sashiba_linear_scan_backward",
        inputNames: ["a", "h", "g"],
        outputNames: ["da", "dx"],
        source: """
            uint gid = thread_position_in_grid.x;
            if (gid < uint(B * R)) {
                uint b = gid / R;
                uint r = gid - b * R;
                size_t base = size_t(b) * T * R + r;
                float carry = 0.0f;
                for (int t = T - 1; 0 <= t; --t) {
                    size_t idx = base + size_t(t) * R;
                    float d = g[idx] + carry;
                    dx[idx] = d;
                    float previous = 0.0f;
                    if (0 < t) {
                        previous = h[idx - R];
                    }
                    da[idx] = d * previous;
                    carry = d * a[idx];
                }
            }
            """
    )

    static func launch(_ shape: [Int]) -> (template: [(String, any KernelTemplateArg)], grid: (Int, Int, Int), group: (Int, Int, Int)) {
        let threads = shape[0] * shape[2]
        return (
            template: [("B", shape[0]), ("T", shape[1]), ("R", shape[2])],
            grid: (threads, 1, 1),
            group: (min(256, threads), 1, 1)
        )
    }

    static func forward(_ a: MLXArray, _ x: MLXArray) -> MLXArray {
        let p = launch(a.shape)
        return forwardKernel(
            [a, x], template: p.template, grid: p.grid, threadGroup: p.group,
            outputShapes: [a.shape], outputDTypes: [.float32]
        )[0]
    }

    nonisolated(unsafe) static let function = CustomFunction {
        Forward { inputs in
            return [MLXLinearScan.forward(inputs[0], inputs[1])]
        }
        VJP { primals, cotangents in
            let a = primals[0]
            let h = MLXLinearScan.forward(a, primals[1])
            let p = launch(a.shape)
            return backwardKernel(
                [a, h, cotangents[0]], template: p.template, grid: p.grid, threadGroup: p.group,
                outputShapes: [a.shape, a.shape], outputDTypes: [.float32, .float32]
            )
        }
    }
    #endif

    /// Parallel associative scan over time axis 1 (`O(log T)` parallel depth) using standard `MLXArray` primitives (runs natively on CUDA & Metal).
    public static func associativeScan(_ aIn: MLXArray, _ xIn: MLXArray, h0: MLXArray? = nil) -> MLXArray {
        var curA = aIn.asType(.float32)
        var curX = xIn.asType(.float32)
        let frames = curX.shape[1]

        if let initH = h0, frames > 0 {
            let h0Exp = initH.asType(.float32).expandedDimensions(axis: 1) // [B, 1, D]
            let firstA = curA[0..., 0..<1, 0...]
            let firstX = curX[0..., 0..<1, 0...] + firstA * h0Exp
            if frames > 1 {
                curX = concatenated([firstX, curX[0..., 1..<frames, 0...]], axis: 1)
            } else {
                curX = firstX
            }
        }

        var stride = 1
        while stride < frames {
            let headX = curX[0..., 0..<stride, 0...]
            let tailX = curX[0..., stride..<frames, 0...]
            let prevX = curX[0..., 0..<(frames - stride), 0...]
            let tailA = curA[0..., stride..<frames, 0...]
            curX = concatenated([headX, tailX + tailA * prevX], axis: 1)

            if stride * 2 < frames {
                let headA = curA[0..., 0..<stride, 0...]
                let prevA = curA[0..., 0..<(frames - stride), 0...]
                curA = concatenated([headA, tailA * prevA], axis: 1)
            }
            stride *= 2
        }
        return curX
    }

    /// `a, x: [B, T, D]` (`float32`), optional `h0: [B, D]`. Returns `h: [B, T, D]`.
    public static func scan(_ a: MLXArray, _ x: MLXArray, h0: MLXArray? = nil) -> MLXArray {
        #if os(macOS)
        if h0 == nil {
            return function([a.asType(.float32), x.asType(.float32)])[0]
        }
        return associativeScan(a, x, h0: h0)
        #else
        return associativeScan(a, x, h0: h0)
        #endif
    }
}

/// Standard feature-wise LayerNorm over the last axis (`eps = 1e-5`).
@inline(__always)
func mlxLayerNorm(_ x: MLXArray, gamma: MLXArray, beta: MLXArray, eps: Float = SWTHawkShape.layerNormEps) -> MLXArray {
    let mu = mean(x, axis: -1, keepDims: true)
    let centered = x - mu
    let variance = mean(centered * centered, axis: -1, keepDims: true)
    return centered * rsqrt(variance + eps) * gamma + beta
}

/// SiLU activation `x * sigmoid(x)`.
@inline(__always)
func mlxSiLU(_ x: MLXArray) -> MLXArray {
    return x * sigmoid(x)
}

/// 1D convolution (`kernel_size = 3, padding = 1`) over `[B, T, inDim] -> [B, T, outDim]` with weight `[3, inDim, outDim]` and bias `[outDim]`.
public final class MLXConv1D3: Module, @unchecked Sendable {
    public let inDim: Int
    public let outDim: Int
    public var w: MLXArray // [3, inDim, outDim]
    public var b: MLXArray // [outDim]

    public init(inDim: Int, outDim: Int, w: MLXArray, b: MLXArray) {
        self.inDim = inDim
        self.outDim = outDim
        self.w = w
        self.b = b
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let time = x.shape[1]
        let dIn = x.shape[2]
        let padZero = MLXArray.zeros([batch, 1, dIn])
        let padded = concatenated([padZero, x, padZero], axis: 1) // [B, T + 2, inDim]
        let x0 = padded[0..., 0..<time, 0...]
        let x1 = padded[0..., 1..<(time + 1), 0...]
        let x2 = padded[0..., 2..<(time + 2), 0...]
        return matmul(x0, w[0]) + matmul(x1, w[1]) + matmul(x2, w[2]) + b
    }
}

/// Branch A: Energy-Preserving 1D Stationary Wavelet Transform (`db1` / Haar) Frequency Branch on MLX GPU.
public final class MLXSWTFrequencyBranch: Module, @unchecked Sendable {
    public let dim: Int
    public var convCD1: MLXConv1D3
    public var convCD2: MLXConv1D3
    public var convCA2: MLXConv1D3
    public var wDetailRouter: MLXArray // [2D, 2D]
    public var bDetailRouter: MLXArray // [2D]
    public var wGate: MLXArray         // [2D, D]
    public var bGate: MLXArray         // [D]

    public init(weights: SWTBranchWeights) {
        let d = weights.dim
        let cd1In = weights.convCD1.inDim
        let cd2In = weights.convCD2.inDim
        self.dim = d
        self.convCD1 = MLXConv1D3(
            inDim: cd1In,
            outDim: d,
            w: MLXArray(weights.convCD1.w, [3, cd1In, d]),
            b: MLXArray(weights.convCD1.b, [d])
        )
        self.convCD2 = MLXConv1D3(
            inDim: cd2In,
            outDim: d,
            w: MLXArray(weights.convCD2.w, [3, cd2In, d]),
            b: MLXArray(weights.convCD2.b, [d])
        )
        self.convCA2 = MLXConv1D3(
            inDim: d,
            outDim: d,
            w: MLXArray(weights.convCA2.w, [3, d, d]),
            b: MLXArray(weights.convCA2.b, [d])
        )
        self.wDetailRouter = MLXArray(weights.wDetailRouter, [2 * d, 2 * d])
        self.bDetailRouter = MLXArray(weights.bDetailRouter, [2 * d])
        self.wGate = MLXArray(weights.wGate, [2 * d, d])
        self.bGate = MLXArray(weights.bGate, [d])
        super.init()
    }

    /// 2-level undecimated Haar (`db1`) wavelet decomposition with replication padding, preserving shape `[B, T, D]`.
    public func decompose(_ x: MLXArray) -> (ca1: MLXArray, cd1: MLXArray, ca2: MLXArray, cd2: MLXArray) {
        let time = x.shape[1]
        let invSqrt2 = SWTBranchWeights.invSqrt2

        // Level 1 (dilation d = 1): left replication pad of length 1
        let first1 = x[0..., 0..<1, 0...]
        let xPrev1 = time > 1 ? concatenated([first1, x[0..., 0..<(time - 1), 0...]], axis: 1) : first1
        let ca1 = (xPrev1 + x) * invSqrt2
        let cd1 = (x - xPrev1) * invSqrt2

        // Level 2 (dilation d = 2): left replication pad of length 2 on CA1
        let firstCA1 = ca1[0..., 0..<1, 0...]
        let pad2 = concatenated([firstCA1, firstCA1], axis: 1)
        let ca1Prev2: MLXArray = if time > 2 {
            concatenated([pad2, ca1[0..., 0..<(time - 2), 0...]], axis: 1)
        } else if time == 2 {
            pad2
        } else {
            firstCA1
        }
        let ca2 = (ca1Prev2 + ca1) * invSqrt2
        let cd2 = (ca1 - ca1Prev2) * invSqrt2
        return (ca1, cd1, ca2, cd2)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let time = x.shape[1]
        let (_, cd1, ca2, cd2) = decompose(x)

        let cd1In = (convCD1.inDim == 2 * dim) ? concatenated([cd1, abs(cd1)], axis: -1) : cd1
        let cd2In = (convCD2.inDim == 2 * dim) ? concatenated([cd2, abs(cd2)], axis: -1) : cd2

        let uCD1 = mlxSiLU(convCD1(cd1In))
        let uCD2 = mlxSiLU(convCD2(cd2In))
        let detailCat = concatenated([uCD1, uCD2], axis: -1) // [B, T, 2D]
        let routerLogits = (matmul(detailCat, wDetailRouter) + bDetailRouter).reshaped([batch, time, 2, dim])
        let routerWeights = softmax(routerLogits, axis: 2)
        let alpha1 = routerWeights[0..., 0..., 0, 0...]
        let alpha2 = routerWeights[0..., 0..., 1, 0...]
        let fDetail = alpha1 * uCD1 + alpha2 * uCD2

        let fApprox = convCA2(ca2)
        let gateCat = concatenated([fApprox, fDetail], axis: -1) // [B, T, 2D]
        let gate = sigmoid(matmul(gateCat, wGate) + bGate)
        return gate * fApprox + (1.0 - gate) * fDetail
    }
}

/// Causal Local Sliding-Window Attention (`W = 32`) with Rotary Position Embedding (`RoPE`) on MLX GPU.
public final class MLXLocalAttention: Module, @unchecked Sendable {
    public let dim: Int
    public let numHeads: Int
    public let windowSize: Int
    public var normGamma: MLXArray
    public var normBeta: MLXArray
    public var wQ: MLXArray
    public var bQ: MLXArray
    public var wK: MLXArray
    public var bK: MLXArray
    public var wV: MLXArray
    public var bV: MLXArray
    public var wOut: MLXArray
    public var bOut: MLXArray

    public init(weights: LocalAttentionWeights) {
        let d = weights.dim
        self.dim = d
        self.numHeads = weights.numHeads
        self.windowSize = weights.windowSize
        self.normGamma = MLXArray(weights.normGamma, [d])
        self.normBeta = MLXArray(weights.normBeta, [d])
        self.wQ = MLXArray(weights.wQ, [d, d])
        self.bQ = MLXArray(weights.bQ, [d])
        self.wK = MLXArray(weights.wK, [d, d])
        self.bK = MLXArray(weights.bK, [d])
        self.wV = MLXArray(weights.wV, [d, d])
        self.bV = MLXArray(weights.bV, [d])
        self.wOut = MLXArray(weights.wOut, [d, d])
        self.bOut = MLXArray(weights.bOut, [d])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.shape[0]
        let time = x.shape[1]
        let headDim = dim / numHeads
        let half = headDim / 2

        let normed = mlxLayerNorm(x, gamma: normGamma, beta: normBeta)
        var q = (matmul(normed, wQ) + bQ).reshaped([batch, time, numHeads, headDim])
        var k = (matmul(normed, wK) + bK).reshaped([batch, time, numHeads, headDim])
        let v = (matmul(normed, wV) + bV).reshaped([batch, time, numHeads, headDim]).transposed(0, 2, 1, 3) // [B, H, T, headDim]

        if half > 0 {
            let (cosFlat, sinFlat) = LocalSlidingWindowAttention.ropeTables(time: time, headDim: headDim)
            let cosArr = MLXArray(cosFlat, [1, time, 1, half])
            let sinArr = MLXArray(sinFlat, [1, time, 1, half])

            let q1 = q[0..., 0..., 0..., 0..<half]
            let q2 = q[0..., 0..., 0..., half..<headDim]
            q = concatenated([q1 * cosArr - q2 * sinArr, q1 * sinArr + q2 * cosArr], axis: -1)

            let k1 = k[0..., 0..., 0..., 0..<half]
            let k2 = k[0..., 0..., 0..., half..<headDim]
            k = concatenated([k1 * cosArr - k2 * sinArr, k1 * sinArr + k2 * cosArr], axis: -1)
        }

        let qHeads = q.transposed(0, 2, 1, 3) // [B, H, T, headDim]
        let kHeadsT = k.transposed(0, 2, 3, 1) // [B, H, headDim, T]
        let scale = 1.0 / sqrt(Float(max(1, headDim)))

        var maskFlat = [Float](repeating: -1e9, count: time * time)
        for t in 0..<time {
            let sStart = max(0, t - windowSize + 1)
            for s in sStart...t {
                maskFlat[t * time + s] = 0.0
            }
        }
        let mask = MLXArray(maskFlat, [1, 1, time, time])
        let scores = matmul(qHeads, kHeadsT) * scale + mask
        let probs = softmax(scores, axis: -1)
        let attended = matmul(probs, v).transposed(0, 2, 1, 3).reshaped([batch, time, dim])
        let attnOut = matmul(attended, wOut) + bOut
        return x + attnOut
    }
}

/// Branch B: Griffin (`RG-LRU` + `Local Sliding-Window Attention W=32 with RoPE` + `Gated MLP`) Temporal Block on MLX GPU.
public final class MLXHawkBlock: Module, @unchecked Sendable {
    public let dim: Int
    public let mlpDim: Int
    public var norm1Gamma: MLXArray
    public var norm1Beta: MLXArray
    public var wGate: MLXArray
    public var bGate: MLXArray
    public var wA: MLXArray
    public var bA: MLXArray
    public var wX: MLXArray
    public var bX: MLXArray
    public var lambda: MLXArray
    public var wOut: MLXArray
    public var bOut: MLXArray
    public var localAttn: MLXLocalAttention
    public var norm2Gamma: MLXArray
    public var norm2Beta: MLXArray
    public var wMLPUp: MLXArray
    public var bMLPUp: MLXArray
    public var wMLPDown: MLXArray
    public var bMLPDown: MLXArray

    public init(weights: HawkBlockWeights) {
        let d = weights.dim
        let m = weights.mlpDim
        self.dim = d
        self.mlpDim = m
        self.norm1Gamma = MLXArray(weights.norm1Gamma, [d])
        self.norm1Beta = MLXArray(weights.norm1Beta, [d])
        self.wGate = MLXArray(weights.wGate, [d, d])
        self.bGate = MLXArray(weights.bGate, [d])
        self.wA = MLXArray(weights.rglru.wA, [d, d])
        self.bA = MLXArray(weights.rglru.bA, [d])
        self.wX = MLXArray(weights.rglru.wX, [d, d])
        self.bX = MLXArray(weights.rglru.bX, [d])
        self.lambda = MLXArray(weights.rglru.lambda, [d])
        self.wOut = MLXArray(weights.wOut, [d, d])
        self.bOut = MLXArray(weights.bOut, [d])
        self.localAttn = MLXLocalAttention(weights: weights.localAttn)
        self.norm2Gamma = MLXArray(weights.norm2Gamma, [d])
        self.norm2Beta = MLXArray(weights.norm2Beta, [d])
        self.wMLPUp = MLXArray(weights.wMLPUp, [d, m])
        self.bMLPUp = MLXArray(weights.bMLPUp, [m])
        self.wMLPDown = MLXArray(weights.wMLPDown, [m, d])
        self.bMLPDown = MLXArray(weights.bMLPDown, [d])
        super.init()
    }

    public func forward(_ x: MLXArray, h0: MLXArray? = nil, adaGamma: MLXArray? = nil, adaBeta: MLXArray? = nil) -> (out: MLXArray, hLast: MLXArray) {
        let normed1Raw = mlxLayerNorm(x, gamma: norm1Gamma, beta: norm1Beta)
        let normed1: MLXArray = if let g = adaGamma, let b = adaBeta {
            (1.0 + (g.ndim == 2 ? g.expandedDimensions(axis: 1) : g)) * normed1Raw + (b.ndim == 2 ? b.expandedDimensions(axis: 1) : b)
        } else {
            normed1Raw
        }

        let gateAct = mlxSiLU(matmul(normed1, wGate) + bGate)
        let r = sigmoid(matmul(normed1, wA) + bA)
        let i = sigmoid(matmul(normed1, wX) + bX)
        let logA = (RGLRUWeights.decayConstant * r) * logSigmoid(lambda)
        let a = exp(logA)
        let scale = sqrt(maximum(1.0 - exp(2.0 * logA) + RGLRUWeights.scaleEps, MLXArray(RGLRUWeights.scaleEps)))
        let u = scale * (i * normed1)
        let h = MLXLinearScan.scan(a, u, h0: h0)
        let time = x.shape[1]
        let hLast = h[0..., (time - 1), 0...]

        let xMid1 = x + matmul(gateAct * h, wOut) + bOut
        let xMid2 = localAttn(xMid1)
        let normed2 = mlxLayerNorm(xMid2, gamma: norm2Gamma, beta: norm2Beta)
        let mlpAct = mlxSiLU(matmul(normed2, wMLPUp) + bMLPUp)
        let out = xMid2 + matmul(mlpAct, wMLPDown) + bMLPDown
        return (out, hLast)
    }
}

/// Bidirectional Cross-Attention & Gated Time-Frequency Fusion on MLX GPU.
public final class MLXGatedTimeFrequencyFusion: Module, @unchecked Sendable {
    public let dim: Int
    public var wSwiGLUGate: MLXArray
    public var bSwiGLUGate: MLXArray
    public var wSwiGLUUp: MLXArray
    public var bSwiGLUUp: MLXArray
    public var normGamma: MLXArray
    public var normBeta: MLXArray

    public init(weights: GatedFusionWeights) {
        let d = weights.dim
        self.dim = d
        self.wSwiGLUGate = MLXArray(weights.wSwiGLUGate, [2 * d, d])
        self.bSwiGLUGate = MLXArray(weights.bSwiGLUGate, [d])
        self.wSwiGLUUp = MLXArray(weights.wSwiGLUUp, [2 * d, d])
        self.bSwiGLUUp = MLXArray(weights.bSwiGLUUp, [d])
        self.normGamma = MLXArray(weights.normGamma, [d])
        self.normBeta = MLXArray(weights.normBeta, [d])
        super.init()
    }

    public func callAsFunction(fSWT: MLXArray, tFeat: MLXArray) -> MLXArray {
        let cat = concatenated([fSWT, tFeat], axis: -1) // [B, T, 2D]
        let gateAct = mlxSiLU(matmul(cat, wSwiGLUGate) + bSwiGLUGate)
        let up = matmul(cat, wSwiGLUUp) + bSwiGLUUp
        let hFusion = gateAct * up
        return mlxLayerNorm(fSWT + tFeat + hFusion, gamma: normGamma, beta: normBeta)
    }
}

/// Subband-Split Multi-Scale Attention Pooling (`-> z ∈ R^(B × 1024)`) on MLX GPU.
public final class MLXMultiScaleAttentionPooling: Module, @unchecked Sendable {
    public let dim: Int
    public let latentDim: Int
    public let numQueries: Int
    public let numHeads: Int
    public let l2Normalize: Bool
    public var qPool: MLXArray
    public var wK: MLXArray
    public var bK: MLXArray
    public var wV: MLXArray
    public var bV: MLXArray
    public var wO: MLXArray
    public var bO: MLXArray
    public var wProj: MLXArray
    public var bProj: MLXArray
    public var normGamma: MLXArray
    public var normBeta: MLXArray

    public init(weights: AttentionPoolingWeights) {
        let d = weights.dim
        let lDim = weights.latentDim
        let k = weights.numQueries
        self.dim = d
        self.latentDim = lDim
        self.numQueries = k
        self.numHeads = weights.numHeads
        self.l2Normalize = weights.l2Normalize
        self.qPool = MLXArray(weights.qPool, [k, d])
        self.wK = MLXArray(weights.wK, [d, d])
        self.bK = MLXArray(weights.bK, [d])
        self.wV = MLXArray(weights.wV, [d, d])
        self.bV = MLXArray(weights.bV, [d])
        self.wO = MLXArray(weights.wO, [d, d])
        self.bO = MLXArray(weights.bO, [d])
        self.wProj = MLXArray(weights.wProj, [3 * d, lDim])
        self.bProj = MLXArray(weights.bProj, [lDim])
        self.normGamma = MLXArray(weights.normGamma, [lDim])
        self.normBeta = MLXArray(weights.normBeta, [lDim])
        super.init()
    }

    public func callAsFunction(uFused: MLXArray, hLast: MLXArray, phi: MLXArray) -> MLXArray {
        let batch = uFused.shape[0]
        let time = uFused.shape[1]
        let headDim = dim / numHeads

        // 1. v_mean: temporal mean over [0, T)
        let vMean = mean(uFused, axis: 1) // [B, D]

        // 2. v_last = h_T
        let vLast = hLast

        // 3. v_attn: Multi-head Attention Pooling with learnable queries Q_pool [K, D]
        let keys = (matmul(uFused, wK) + bK).reshaped([batch, time, numHeads, headDim]).transposed(0, 2, 3, 1) // [B, H, headDim, T]
        let values = (matmul(uFused, wV) + bV).reshaped([batch, time, numHeads, headDim]).transposed(0, 2, 1, 3) // [B, H, T, headDim]
        let q = qPool.reshaped([1, numQueries, numHeads, headDim]).transposed(0, 2, 1, 3) // [1, H, K, headDim]

        let scale = 1.0 / sqrt(Float(max(1, headDim)))
        let scores = matmul(q, keys) * scale // [B, H, K, T]
        let probs = softmax(scores, axis: -1)
        let attended = matmul(probs, values) // [B, H, K, headDim]
        let meanOverQueries = mean(attended, axis: 2).transposed(0, 1, 2).reshaped([batch, dim])
        let vAttn = matmul(meanOverQueries, wO) + bO

        // 4. Projection + Nyquist harmonic pooling + LayerNorm -> z ∈ R^(B × 1024)
        let concatPooled = concatenated([vMean, vLast, vAttn], axis: -1) // [B, 3D]
        let wHarmonic = wProj[0..<dim, 0...] // [D, latentDim]
        let uProj = matmul(uFused, wHarmonic) // [B, T, latentDim]
        let harmScale = 2.0 / sqrt(Float(max(1, time)))
        let zHarmonic = sum(uProj * phi, axis: 1) * harmScale // [B, latentDim]
        let zRaw = matmul(concatPooled, wProj) + bProj + zHarmonic
        let zNormed = mlxLayerNorm(zRaw, gamma: normGamma, beta: normBeta)
        if !l2Normalize {
            return zNormed
        }
        return zNormed * rsqrt(sum(zNormed * zNormed, axis: -1, keepDims: true) + 1e-12)
    }
}

/// One AdaLN + `h_0(z)` conditioned Griffin Decoder Layer on MLX GPU.
public final class MLXConditionalHawkLayer: Module, @unchecked Sendable {
    public var block: MLXHawkBlock
    public var wH0: MLXArray
    public var bH0: MLXArray
    public var wAdaGamma: MLXArray
    public var bAdaGamma: MLXArray
    public var wAdaBeta: MLXArray
    public var bAdaBeta: MLXArray

    public init(weights: ConditionalHawkLayerWeights) {
        let d = weights.dim
        let lDim = weights.latentDim
        self.block = MLXHawkBlock(weights: weights.block)
        self.wH0 = MLXArray(weights.wH0, [lDim, d])
        self.bH0 = MLXArray(weights.bH0, [d])
        self.wAdaGamma = MLXArray(weights.wAdaGamma, [lDim, d])
        self.bAdaGamma = MLXArray(weights.bAdaGamma, [d])
        self.wAdaBeta = MLXArray(weights.wAdaBeta, [lDim, d])
        self.bAdaBeta = MLXArray(weights.bAdaBeta, [d])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, z: MLXArray, zMod: MLXArray) -> MLXArray {
        let h0 = matmul(z, wH0) + bH0
        let adaGamma = matmul(zMod, wAdaGamma) + bAdaGamma
        let adaBeta = matmul(zMod, wAdaBeta) + bAdaBeta
        return block.forward(x, h0: h0, adaGamma: adaGamma, adaBeta: adaBeta).out
    }
}

/// Griffin Conditional Text Decoder on MLX GPU (`Section 2.2`).
public final class MLXHawkConditionalDecoder: Module, @unchecked Sendable {
    public let dim: Int
    public let latentDim: Int
    public let vocabSize: Int

    public var wCross: MLXArray
    public var bCross: MLXArray
    public var layers: [MLXConditionalHawkLayer]
    public var finalNormGamma: MLXArray
    public var finalNormBeta: MLXArray
    public var wReadout: MLXArray
    public var bReadout: MLXArray

    public init(weights: HawkDecoderWeights) {
        let d = weights.dim
        let lDim = weights.latentDim
        let v = weights.vocabSize

        self.dim = d
        self.latentDim = lDim
        self.vocabSize = v
        self.wCross = MLXArray(weights.wCross, [lDim, d])
        self.bCross = MLXArray(weights.bCross, [d])
        self.layers = weights.layers.map { MLXConditionalHawkLayer(weights: $0) }
        self.finalNormGamma = MLXArray(weights.finalNormGamma, [d])
        self.finalNormBeta = MLXArray(weights.finalNormBeta, [d])
        self.wReadout = MLXArray(weights.wReadout, [d, v])
        self.bReadout = MLXArray(weights.bReadout, [v])
        super.init()
    }

    public func callAsFunction(
        z: MLXArray,
        prevEmbedded: MLXArray,
        dropoutKeepMask: MLXArray,
        pe: MLXArray,
        phi: MLXArray
    ) -> MLXArray {
        let zMod = z.expandedDimensions(axis: 1) * (1.0 + phi) // [B, T, latentDim]
        let zCond = matmul(zMod, wCross) + bCross // [B, T, D]
        let tokVal = prevEmbedded * dropoutKeepMask.expandedDimensions(axis: -1)
        var x = tokVal + pe + zCond
        for layer in layers {
            x = layer(x, z: z, zMod: zMod)
        }
        let finalNormed = mlxLayerNorm(x, gamma: finalNormGamma, beta: finalNormBeta)
        return matmul(finalNormed, wReadout) + bReadout
    }
}

/// Phase 5.1: Dual-Stream Subband Bidirectional Cross-Attention & Disentangled Latent Projection Module on MLX GPU (`Section 2 & 3`).
public final class MLXDualStreamRelationalModule: Module, @unchecked Sendable {
    public let vocabSize: Int
    public let dim: Int
    public let latentDim: Int
    public var alphaRel: [Float]
    public var calibratedHeads: RelationalCalibratedHeads
    public var auditTelemetry: RelationalAuditTelemetry

    // Bidirectional_CrossAttention_Layer parameters
    public var eEcho: MLXArray
    public var eTopic: MLXArray
    public var wQLL: MLXArray
    public var bQLL: MLXArray
    public var wKLL: MLXArray
    public var bKLL: MLXArray
    public var wVLL: MLXArray
    public var bVLL: MLXArray
    public var wCohLL: MLXArray
    public var bCohLL: MLXArray

    public var wQHH: MLXArray
    public var bQHH: MLXArray
    public var wKHH: MLXArray
    public var bKHH: MLXArray
    public var wVHH: MLXArray
    public var bVHH: MLXArray
    public var wCohHH: MLXArray
    public var bCohHH: MLXArray

    // Latent_Projection_Layer parameters
    public var wProjLL: MLXArray
    public var bProjLL: MLXArray
    public var normLLGamma: MLXArray
    public var normLLBeta: MLXArray

    public var wProjHH: MLXArray
    public var bProjHH: MLXArray
    public var normHHGamma: MLXArray
    public var normHHBeta: MLXArray

    public var wContent: MLXArray
    public var bContent: MLXArray

    public var wArchT4: MLXArray
    public var bArchT4: MLXArray
    public var wTopoT4: MLXArray
    public var bTopoT4: MLXArray
    public var wArchT2: MLXArray
    public var bArchT2: MLXArray
    public var wTopoT2: MLXArray
    public var bTopoT2: MLXArray

    public init(weights: DualStreamRelationalWeights) {
        let d = weights.dim
        let lDim = weights.latentDim
        let half = lDim / 2
        let cohDim = DualStreamRelationalWeights.coherenceDim

        self.vocabSize = weights.vocabSize
        self.dim = d
        self.latentDim = lDim
        self.alphaRel = weights.alphaRel
        self.calibratedHeads = weights.calibratedHeads
        self.auditTelemetry = weights.auditTelemetry

        self.eEcho = MLXArray(weights.eEcho, [d])
        self.eTopic = MLXArray(weights.eTopic, [d])
        self.wQLL = MLXArray(weights.wQLL, [d, d])
        self.bQLL = MLXArray(weights.bQLL, [d])
        self.wKLL = MLXArray(weights.wKLL, [d, d])
        self.bKLL = MLXArray(weights.bKLL, [d])
        self.wVLL = MLXArray(weights.wVLL, [d, d])
        self.bVLL = MLXArray(weights.bVLL, [d])
        self.wCohLL = MLXArray(weights.wCohLL, [cohDim, d])
        self.bCohLL = MLXArray(weights.bCohLL, [d])

        self.wQHH = MLXArray(weights.wQHH, [d, d])
        self.bQHH = MLXArray(weights.bQHH, [d])
        self.wKHH = MLXArray(weights.wKHH, [d, d])
        self.bKHH = MLXArray(weights.bKHH, [d])
        self.wVHH = MLXArray(weights.wVHH, [d, d])
        self.bVHH = MLXArray(weights.bVHH, [d])
        self.wCohHH = MLXArray(weights.wCohHH, [cohDim, d])
        self.bCohHH = MLXArray(weights.bCohHH, [d])

        self.wProjLL = MLXArray(weights.wProjLL, [5 * d, half])
        self.bProjLL = MLXArray(weights.bProjLL, [half])
        self.normLLGamma = MLXArray(weights.normLLGamma, [half])
        self.normLLBeta = MLXArray(weights.normLLBeta, [half])

        self.wProjHH = MLXArray(weights.wProjHH, [5 * d, half])
        self.bProjHH = MLXArray(weights.bProjHH, [half])
        self.normHHGamma = MLXArray(weights.normHHGamma, [half])
        self.normHHBeta = MLXArray(weights.normHHBeta, [half])

        self.wContent = MLXArray(weights.wContent, [2 * d, half])
        self.bContent = MLXArray(weights.bContent, [half])

        self.wArchT4 = MLXArray(weights.wArchT4, [lDim, 6])
        self.bArchT4 = MLXArray(weights.bArchT4, [6])
        self.wTopoT4 = MLXArray(weights.wTopoT4, [lDim, 5])
        self.bTopoT4 = MLXArray(weights.bTopoT4, [5])
        self.wArchT2 = MLXArray(weights.wArchT2, [lDim, 6])
        self.bArchT2 = MLXArray(weights.bArchT2, [6])
        self.wTopoT2 = MLXArray(weights.wTopoT2, [lDim, 5])
        self.bTopoT2 = MLXArray(weights.bTopoT2, [5])
        super.init()
    }
}

/// Complete SWT-Griffin Text Autoencoder (`MLXSWTFrequencyBranch` + `MLXHawkBlock` + `MLXGatedTimeFrequencyFusion`
/// + `MLXMultiScaleAttentionPooling` + `MLXHawkConditionalDecoder` + `MLXDualStreamRelationalModule`) on MLX GPU (Metal / CUDA).
public final class MLXSWTHawkAutoencoder: Module, @unchecked Sendable {
    public let vocabSize: Int
    public let dim: Int
    public let latentDim: Int
    public let mlpDim: Int
    public let serializedVocabulary: String
    public var subbandClassifier: SubbandTokenClassifier

    public var tokenEmbedding: MLXArray
    public var swtBranch: MLXSWTFrequencyBranch
    public var hawkBranch: [MLXHawkBlock]
    public var fusion: MLXGatedTimeFrequencyFusion
    public var pooling: MLXMultiScaleAttentionPooling
    public var decoder: MLXHawkConditionalDecoder
    public var relational: MLXDualStreamRelationalModule
    public var flowMatchingWeights: FlowMatchingWeights

    public init(weights: SWTHawkWeights) {
        self.vocabSize = weights.vocabSize
        self.dim = weights.dim
        self.latentDim = weights.latentDim
        self.mlpDim = weights.mlpDim
        self.serializedVocabulary = weights.serializedVocabulary
        self.subbandClassifier = weights.subbandClassifier

        self.tokenEmbedding = MLXArray(weights.tokenEmbedding, [weights.vocabSize, weights.dim])
        self.swtBranch = MLXSWTFrequencyBranch(weights: weights.swtBranch)
        self.hawkBranch = weights.hawkBranch.map { MLXHawkBlock(weights: $0) }
        self.fusion = MLXGatedTimeFrequencyFusion(weights: weights.fusion)
        self.pooling = MLXMultiScaleAttentionPooling(weights: weights.pooling)
        self.decoder = MLXHawkConditionalDecoder(weights: weights.decoder)
        self.relational = MLXDualStreamRelationalModule(weights: weights.relational)
        self.flowMatchingWeights = weights.flowMatching
        super.init()
    }

    /// Encodes `tokenIds: [B, T]` (`int32`) with sinusoidal positional encoding `pe: [1, T, D]` into `z: [B, 1024]`.
    public func encode(_ tokenIds: MLXArray, pe: MLXArray, phi: MLXArray? = nil) -> MLXArray {
        let time = tokenIds.shape[1]
        let latentPhi = phi ?? MLXArray(HawkConditionalDecoder.latentBasis(time: time, latentDim: latentDim), [1, time, latentDim])
        let embedded = tokenEmbedding[tokenIds] + pe // [B, T, D]
        let fSWT = swtBranch(embedded)

        var tFeat = embedded
        var hLast = MLXArray.zeros([tokenIds.shape[0], dim])
        for block in hawkBranch {
            let res = block.forward(tFeat)
            tFeat = res.out
            hLast = res.hLast
        }

        let uFused = fusion(fSWT: fSWT, tFeat: tFeat)
        return pooling(uFused: uFused, hLast: hLast, phi: latentPhi)
    }

    /// Runs only the neural encoder (no token-ID lattice, no modulation rules) over `tokens: [B * T]` in chunks.
    /// Returns `[B * latentDim]` features suitable for downstream learned heads.
    public func encodeNeural(tokens: [Int], batch: Int, time: Int, chunk: Int = 64) -> [Float] {
        let peFlat = HawkConditionalDecoder.positionalEncoding(time: time, dim: dim)
        let pe = MLXArray(peFlat, [1, time, dim])
        let phi = MLXArray(HawkConditionalDecoder.latentBasis(time: time, latentDim: latentDim), [1, time, latentDim])
        var out: [Float] = []
        out.reserveCapacity(batch * latentDim)
        var start = 0
        while start < batch {
            let cur = min(chunk, batch - start)
            let slice = tokens[(start * time)..<((start + cur) * time)].map { Int32($0) }
            let ids = MLXArray(slice, [cur, time])
            let z = encode(ids, pe: pe, phi: phi)
            eval(z)
            out.append(contentsOf: z.asArray(Float.self))
            start += cur
        }
        return out
    }

    // MARK: - Phase 5.1: Differentiable Dual-Stream SWT-Griffin & Bidirectional Cross-Attention Forward Pass

    /// Runs the differentiable Dual-Stream SWT-Griffin & Subband Bidirectional Cross-Attention forward pass on MLX GPU.
    /// Mirrors `DualStreamRelationalEngine.forwardBatchCPU` operation-for-operation so that the graph optimized by
    /// End-to-End training is exactly the graph used for inference, routing gradients through
    /// `tokenEmbedding`, `swtBranch`, `hawkBranch`, `fusion`, and `relational` (`Section 2 & 3`).
    public func forwardDualStream(
        senderIds: MLXArray,
        receiverIds: MLXArray,
        pe: MLXArray
    ) -> (
        phiLL: MLXArray,
        phiHH: MLXArray,
        zRel: MLXArray,
        zContent: MLXArray,
        logitsArchT4: MLXArray,
        logitsTopoT4: MLXArray,
        logitsArchT2: MLXArray,
        logitsTopoT2: MLXArray
    ) {
        let batch = senderIds.shape[0]
        let time = senderIds.shape[1]
        let d = dim
        let cohProjDim = min(32, d)

        func shiftTime(_ x: MLXArray) -> MLXArray {
            // x_{t-1} with zero at t = 0 ([B, T, F])
            let f = x.shape[2]
            if time <= 1 {
                return MLXArray.zeros([batch, time, f])
            }
            return concatenated([MLXArray.zeros([batch, 1, f]), x[0..., 0..<(time - 1), 0...]], axis: 1)
        }
        func shiftTimeReplicate(_ x: MLXArray) -> MLXArray {
            // x_{max(0, t-1)} ([B, T, F])
            if time <= 1 {
                return x
            }
            return concatenated([x[0..., 0..<1, 0...], x[0..., 0..<(time - 1), 0...]], axis: 1)
        }
        func nextTime(_ x: MLXArray) -> MLXArray {
            // x_{t+1} with zero at t = T - 1
            let f = x.shape[2]
            if time <= 1 {
                return MLXArray.zeros([batch, time, f])
            }
            return concatenated([x[0..., 1..<time, 0...], MLXArray.zeros([batch, 1, f])], axis: 1)
        }
        func rollChannels(_ x: MLXArray, _ k: Int) -> MLXArray {
            // y[..., c] = x[..., (c + k) % D]
            let kk = k % d
            if kk == 0 {
                return x
            }
            return concatenated([x[0..., 0..., kk..<d], x[0..., 0..., 0..<kk]], axis: -1)
        }
        func channelMix(_ x: MLXArray) -> MLXArray {
            return 0.6 * rollChannels(x, 1) - 0.4 * rollChannels(x, 3) + 0.5 * x
        }
        func channelMix2(_ x: MLXArray) -> MLXArray {
            return 0.6 * rollChannels(x, 5) - 0.4 * rollChannels(x, 9) + 0.5 * rollChannels(x, 2)
        }
        func channelMixSkip(_ x: MLXArray) -> MLXArray {
            return 0.6 * rollChannels(x, 5) - 0.4 * rollChannels(x, 1) + 0.5 * rollChannels(x, 3)
        }
        func l2NormalizeRows(_ x: MLXArray) -> MLXArray {
            return x / sqrt(maximum(sum(x * x, axis: -1, keepDims: true), MLXArray(Float(1e-12))))
        }
        func clip01(_ x: MLXArray) -> MLXArray {
            return minimum(MLXArray(Float(1.0)), maximum(MLXArray(Float(0.0)), x))
        }

        // 1. Masks, pair masks and normalized temporal pooling weights
        let maskS = (senderIds .> 0).asType(.float32).expandedDimensions(axis: -1) // [B, T, 1]
        let maskC = (receiverIds .> 0).asType(.float32).expandedDimensions(axis: -1)
        let pairS = maskS * shiftTime(maskS)
        let pairC = maskC * shiftTime(maskC)
        let lenS = maximum(sum(maskS, axis: 1, keepDims: true), MLXArray(Float(1.0))) // [B, 1, 1]
        let lenC = maximum(sum(maskC, axis: 1, keepDims: true), MLXArray(Float(1.0)))
        let tauS = (cumsum(maskS, axis: 1) / lenS) * maskS
        let tauC = (cumsum(maskC, axis: 1) / lenC) * maskC

        func normalizeOverTime(_ w: MLXArray) -> MLXArray {
            return w / maximum(sum(w, axis: 1, keepDims: true), MLXArray(Float(1e-6)))
        }
        let wMeanS = normalizeOverTime(pairS)
        let wMeanC = normalizeOverTime(pairC)
        let omS = maximum(MLXArray(Float(0.0)), 1.0 - tauS)
        let omC = maximum(MLXArray(Float(0.0)), 1.0 - tauC)
        let wEarlyS = normalizeOverTime(pairS * omS * omS)
        let wEarlyC = normalizeOverTime(pairC * omC * omC)
        let wLateS = normalizeOverTime(pairS * tauS * tauS * tauS)
        let wLateC = normalizeOverTime(pairC * tauC * tauC * tauC)

        // 2. Raw embeddings & Multi-Scale Haar Wavelet Cross-Coherence on the first cohProjDim channels
        let eRawS = tokenEmbedding[senderIds] * maskS
        let eRawC = tokenEmbedding[receiverIds] * maskC
        let eCohS = eRawS[0..., 0..., 0..<cohProjDim]
        let eCohC = eRawC[0..., 0..., 0..<cohProjDim]
        let (ca1RawS, cd1RawS, ca2RawS, _) = swtBranch.decompose(eCohS)
        let (ca1RawC, cd1RawC, ca2RawC, _) = swtBranch.decompose(eCohC)
        let altSign = MLXArray(DualStreamRelationalEngine.alternatingSigns(dim: cohProjDim), [1, 1, cohProjDim])

        let w1S = l2NormalizeRows((ca1RawS + 0.75 * cd1RawS * altSign) * pairS)
        let w1C = l2NormalizeRows((ca1RawC + 0.75 * cd1RawC * altSign) * pairC)
        let w2S = l2NormalizeRows(ca2RawS * pairS)
        let w2C = l2NormalizeRows(ca2RawC * pairC)
        let dS = l2NormalizeRows(cd1RawS * pairS)
        let dC = l2NormalizeRows(cd1RawC * pairC)

        func coherence(_ a: MLXArray, _ b: MLXArray) -> (cC: MLXArray, cS: MLXArray) {
            let sim = matmul(a, b.transposed(0, 2, 1)) // [B, T_C, T_S]
            let zero = MLXArray(Float(0.0))
            let cC = maximum(max(sim, axis: 2, keepDims: true), zero) * pairC
            let cS = maximum(max(sim, axis: 1), zero).expandedDimensions(axis: -1) * pairS
            return (cC, cS)
        }
        let (c1C, c1S) = coherence(w1C, w1S)
        let (c2C, c2S) = coherence(w2C, w2S)
        let (cDC, cDS) = coherence(dC, dS)

        let gPairC = clip01((c1C - 0.72) / 0.28) * pairC
        let gPairS = clip01((c1S - 0.72) / 0.28) * pairS
        let gC = maximum(gPairC, nextTime(gPairC)) * maskC
        let gS = maximum(gPairS, nextTime(gPairS)) * maskS

        func pool(_ w: MLXArray, _ x: MLXArray) -> MLXArray {
            return sum(w * x, axis: 1) // [B, F]
        }
        let s0 = pool(wMeanC, c1C)
        let s1 = pool(wMeanS, c1S)
        let s2 = pool(wMeanC, c2C)
        let s3 = pool(wMeanS, c2S)
        let s4 = pool(wMeanC, cDC)
        let s5 = pool(wMeanS, cDS)
        let s6 = pool(wMeanC, gC)
        let s7 = pool(wMeanS, gS)
        let s8 = pool(wEarlyC, gC)
        let s9 = pool(wLateC, gC)
        let s10 = pool(wMeanC, clip01((c1C - 0.90) / 0.10))
        let s11 = pool(wMeanS, clip01((c1S - 0.90) / 0.10))
        let s12 = pool(wMeanC, clip01((c2C - 0.80) / 0.20))
        let lenS2 = lenS.reshaped([batch, 1])
        let lenC2 = lenC.reshaped([batch, 1])
        let s13 = minimum(MLXArray(Float(2.0)), lenC2 / lenS2)
        let s14 = minimum(MLXArray(Float(1.0)), lenC2 / 160.0)
        let s15 = minimum(MLXArray(Float(1.0)), lenS2 / 160.0)
        let psiCoh = concatenated([s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15], axis: -1) // [B, 16]

        // 3. Disentangled relational input streams & content stream input
        let alphaArr = MLXArray(relational.alphaRel, [vocabSize, 1])
        let alphaS = alphaArr[senderIds] // [B, T, 1]
        let alphaC = alphaArr[receiverIds]
        let rootD = sqrt(Float(d))
        let eNormS = l2NormalizeRows(eRawS) * rootD * maskS
        let eNormC = l2NormalizeRows(eRawC) * rootD * maskC
        let echo = l2NormalizeRows(relational.eEcho.reshaped([1, 1, d])) * (0.18 * rootD)
        let topic = l2NormalizeRows(relational.eTopic.reshaped([1, 1, d])) * (0.12 * rootD)
        let eRelS = (alphaS * eNormS + (1.0 - alphaS) * (gS * echo + (1.0 - gS) * topic) + 0.05 * pe) * maskS
        let eRelC = (alphaC * eNormC + (1.0 - alphaC) * (gC * echo + (1.0 - gC) * topic) + 0.05 * pe) * maskC
        let uContS = sum(wMeanS * (1.0 - 0.5 * alphaS) * eRawS, axis: 1)
        let uContC = sum(wMeanC * (1.0 - 0.5 * alphaC) * eRawC, axis: 1)

        // 4. Wavelet transition kernels derived from swtBranch.convCA2 taps & RG-LRU decay from hawkBranch[0].lambda
        let diagIdx = MLXArray((0..<d).map { Int32($0 * d + $0) })
        let nextIdx = MLXArray((0..<d).map { Int32((($0 + 1) % d) * d + $0) })
        func kernel(_ tap: Int) -> MLXArray {
            let flat = swtBranch.convCA2.w[tap].reshaped([d * d])
            return (1.0 + 0.5 * (flat[diagIdx] + flat[nextIdx])).reshaped([1, 1, d])
        }
        let k0 = kernel(0)
        let k1 = kernel(1)
        let k2 = kernel(2)
        let (rDecay, rInput): (MLXArray, MLXArray) = if let hb0 = hawkBranch.first {
            (pow(sigmoid(hb0.lambda), Float(0.25)).reshaped([1, 1, d]),
             sqrt(maximum(MLXArray(Float(1e-6)), 1.0 - pow(sigmoid(hb0.lambda), Float(0.5)))).reshaped([1, 1, d]))
        } else {
            (MLXArray.full([1, 1, d], values: MLXArray(Float(0.85))), MLXArray.full([1, 1, d], values: MLXArray(Float(0.52))))
        }

        func buildStream(
            _ eRel: MLXArray, _ pm: MLXArray, _ wm: MLXArray, _ we: MLXArray, _ wl: MLXArray
        ) -> (hLL: MLXArray, hHH: MLXArray) {
            let (ca1, cd1, ca2, cd2) = swtBranch.decompose(eRel)
            let aFull = broadcast(rDecay, to: [batch, time, d])
            let h = MLXLinearScan.scan(aFull, rInput * eRel)
            let hPrev = shiftTime(h)
            let ePrev1 = shiftTimeReplicate(eRel)
            let ePrev2 = shiftTimeReplicate(ePrev1)
            let pBi = channelMix(ePrev1) * k0 * (eRel * k1) * pm
            let pBi2 = channelMix2(ePrev1) * k2 * (eRel * k0) * pm
            let pSkip = channelMixSkip(ePrev2) * k1 * (eRel * k2) * pm
            let pTri = channelMix(shiftTimeReplicate(ca1)) * k1 * (cd1 * k2) * pm
            let pCD2 = channelMix(ca2) * k0 * (cd2 * k2) * pm
            let dT = (h - hPrev) * eRel * pm
            let rawVLL = pBi + 0.55 * pSkip + 0.5 * pCD2 + 0.35 * dT
            let rawVHH = pTri + 0.85 * pBi2 + 0.45 * pSkip + 0.35 * dT
            let tau = MLXArray(Float(0.15))
            let zero = MLXArray(Float(0.0))
            let stLL = maximum(zero, rawVLL - tau) - maximum(zero, -rawVLL - tau)
            let stHH = maximum(zero, rawVHH - tau) - maximum(zero, -rawVHH - tau)
            let vLL = stLL / (1.0 + 0.25 * abs(stLL))
            let vHH = stHH / (1.0 + 0.25 * abs(stHH))

            let wMidRaw = maximum(MLXArray(Float(0.0)), wm - 0.35 * (we + wl))
            let wMid = normalizeOverTime(wMidRaw)
            let winLL = stacked([pool(we, vLL), pool(wMid, vLL), pool(wl, vLL), pool(wm, vLL)], axis: 1) // [B, 4, D]
            let winHH = stacked([pool(we, vHH), pool(wMid, vHH), pool(wl, vHH), pool(wm, vHH)], axis: 1)

            func backbone(_ x: MLXArray) -> MLXArray {
                let fSWT = swtBranch(x)
                var tFeat = x
                for block in hawkBranch {
                    tFeat = block.forward(tFeat).out
                }
                return x + 0.25 * fusion(fSWT: fSWT, tFeat: tFeat)
            }
            return (backbone(winLL), backbone(winHH))
        }

        let (hLLS, hHHS) = buildStream(eRelS, pairS, wMeanS, wEarlyS, wLateS)
        let (hLLC, hHHC) = buildStream(eRelC, pairC, wMeanC, wEarlyC, wLateC)

        // 5. Subband Bidirectional Cross-Attention over the 4 temporal windows (0 = early, 1 = mid, 2 = late, 3 = mean)
        let qLLC = matmul(hLLC, relational.wQLL) + relational.bQLL
        let kLLS = matmul(hLLS, relational.wKLL) + relational.bKLL
        let vLLS = matmul(hLLS, relational.wVLL) + relational.bVLL
        let qHHC = matmul(hHHC, relational.wQHH) + relational.bQHH
        let kHHS = matmul(hHHS, relational.wKHH) + relational.bKHH
        let vHHS = matmul(hHHS, relational.wVHH) + relational.bVHH
        let cohProjLL = matmul(psiCoh, relational.wCohLL) + relational.bCohLL
        let cohProjHH = matmul(psiCoh, relational.wCohHH) + relational.bCohHH

        let scale = 1.0 / sqrt(Float(max(1, d)))
        let attnLL = matmul(softmax(matmul(qLLC, kLLS.transposed(0, 2, 1)) * scale, axis: -1), vLLS)
        let attnHH = matmul(softmax(matmul(qHHC, kHHS.transposed(0, 2, 1)) * scale, axis: -1), vHHS)

        func win(_ x: MLXArray, _ i: Int) -> MLXArray {
            return x[0..., i, 0...] // [B, D]
        }
        let gBar = 0.2 + 0.8 * s6
        let betaLate = clip01((lenC2 - 110.0) / 20.0)

        let sLEarly = win(hLLS, 0) + win(kLLS, 0)
        let sLMid = win(hLLS, 1) + win(kLLS, 1)
        let sLLate = win(hLLS, 2) + win(kLLS, 2)
        let sLMean = win(hLLS, 3) + win(kLLS, 3)
        let sL = 0.5 * sLMean + 0.5 * sLLate
        let cLMean = win(hLLC, 3) + win(qLLC, 3)
        let cLEarly = win(hLLC, 0) + win(qLLC, 0)
        let cLMid = win(hLLC, 1) + win(qLLC, 1)
        let cLLate = win(hLLC, 2) + win(qLLC, 2)
        let cL = (1.0 - 0.70 * betaLate) * cLMean + (0.70 * betaLate) * cLLate
        let diffL = (sL - cL) + (1.0 - betaLate) * (cLMean + 0.5 * cLEarly - sLEarly) + betaLate * (cLLate - cLEarly + 0.5 * (cLMid - sLMid))
        let prodL = sL * cL
        let crossL = gBar * win(attnLL, 3) + (0.45 * betaLate) * (cLLate - cLEarly) + cohProjLL
        let phiLL = concatenated([sL, cL, diffL, prodL, crossL], axis: -1) // [B, 5D]

        let sHEarly = win(hHHS, 0) + win(kHHS, 0)
        let sHMid = win(hHHS, 1) + win(kHHS, 1)
        let sHLate = win(hHHS, 2) + win(kHHS, 2)
        let sHMean = win(hHHS, 3) + win(kHHS, 3)
        let sH = 0.4 * sHMean + 0.6 * sHLate
        let cHMean = win(hHHC, 3) + win(qHHC, 3)
        let cHEarly = win(hHHC, 0) + win(qHHC, 0)
        let cHMid = win(hHHC, 1) + win(qHHC, 1)
        let cHLate = win(hHHC, 2) + win(qHHC, 2)
        let cHT2 = cHMean
        let cHT4 = 0.85 * cHLate + 0.15 * cHMean + 0.35 * (cHLate - cHEarly)
        let cH = (1.0 - betaLate) * cHT2 + betaLate * cHT4
        let diffH = (sH - cH) + (1.0 - betaLate) * (cHMean + 0.5 * cHEarly - sHEarly) + betaLate * (cHLate - cHEarly + 0.5 * (cHMid - sHMid))
        let prodH = sH * cH
        let crossH = (1.0 - betaLate) * win(attnHH, 3) + betaLate * (win(attnHH, 2) + 0.55 * cHEarly) + cohProjHH
        let phiHH = concatenated([sH, cH, diffH, prodH, crossH], axis: -1)

        // 6. Latent Projection Layer & LayerNorm
        let zLL = mlxLayerNorm(matmul(phiLL, relational.wProjLL) + relational.bProjLL, gamma: relational.normLLGamma, beta: relational.normLLBeta)
        let zHH = mlxLayerNorm(matmul(phiHH, relational.wProjHH) + relational.bProjHH, gamma: relational.normHHGamma, beta: relational.normHHBeta)
        let zRel = concatenated([zLL, zHH], axis: -1) // [B, 1024]

        let half = latentDim / 2
        let zContent = mlxLayerNorm(
            matmul(concatenated([uContS, uContC], axis: -1), relational.wContent) + relational.bContent,
            gamma: MLXArray.ones([half]),
            beta: MLXArray.zeros([half])
        )

        let logitsArchT4 = matmul(zRel, relational.wArchT4) + relational.bArchT4
        let logitsTopoT4 = matmul(zRel, relational.wTopoT4) + relational.bTopoT4
        let logitsArchT2 = matmul(zRel, relational.wArchT2) + relational.bArchT2
        let logitsTopoT2 = matmul(zRel, relational.wTopoT2) + relational.bTopoT2
        return (phiLL, phiHH, zRel, zContent, logitsArchT4, logitsTopoT4, logitsArchT2, logitsTopoT2)
    }

    /// Chunked MLX GPU extraction of the pre-projection subband interaction tensors `(phiLL: [B × 5D], phiHH: [B × 5D])`.
    public func extractRelationalPhiGPU(
        senderTokens: [Int],
        receiverTokens: [Int],
        batch: Int,
        time: Int,
        chunk: Int = 64
    ) -> (phiLL: [Float], phiHH: [Float]) {
        precondition(senderTokens.count == batch * time && receiverTokens.count == batch * time)
        let pe = MLXArray(HawkConditionalDecoder.positionalEncoding(time: time, dim: dim), [1, time, dim])
        var phiLLOut: [Float] = []
        var phiHHOut: [Float] = []
        phiLLOut.reserveCapacity(batch * 5 * dim)
        phiHHOut.reserveCapacity(batch * 5 * dim)
        var start = 0
        while start < batch {
            let cur = min(chunk, batch - start)
            let sIds = MLXArray(senderTokens[(start * time)..<((start + cur) * time)].map { Int32($0) }, [cur, time])
            let cIds = MLXArray(receiverTokens[(start * time)..<((start + cur) * time)].map { Int32($0) }, [cur, time])
            let out = forwardDualStream(senderIds: sIds, receiverIds: cIds, pe: pe)
            eval(out.phiLL, out.phiHH)
            phiLLOut.append(contentsOf: out.phiLL.asArray(Float.self))
            phiHHOut.append(contentsOf: out.phiHH.asArray(Float.self))
            start += cur
        }
        return (phiLLOut, phiHHOut)
    }

    /// Pure-neural Dual-Stream Relational Encoder taking strictly `(senderTokenIDs: [Int], receiverTokenIDs: [Int])` (`Phase 5.1 Audit 1`), on MLX GPU.
    public func encodeRelational(
        senderTokenIDs: [Int],
        receiverTokenIDs: [Int]
    ) -> [Float] {
        let time = max(DualStreamRelationalWeights.canonicalSeqLen, max(senderTokenIDs.count, receiverTokenIDs.count))
        return encodeRelationalGPU(
            senderTokens: DualStreamRelationalEngine.padOrTruncate(senderTokenIDs, to: time),
            receiverTokens: DualStreamRelationalEngine.padOrTruncate(receiverTokenIDs, to: time),
            batch: 1,
            time: time
        ).zRel
    }

    /// Batched pure-neural Dual-Stream Relational Encoder returning `z_rel = [z_LL^rel(512); z_HH^rel(512)] ∈ R^(B × 1024)` (`Phase 5.1 Audit 5`).
    public func encodeRelationalBatch(
        senderTokens: [Int],
        receiverTokens: [Int],
        batch: Int,
        time: Int
    ) -> [Float] {
        return encodeRelationalGPU(senderTokens: senderTokens, receiverTokens: receiverTokens, batch: batch, time: time).zRel
    }

    /// Chunked MLX GPU evaluation of `forwardDualStream` returning host `(zRel: [B × 1024], zContent: [B × 512])`.
    public func encodeRelationalGPU(
        senderTokens: [Int],
        receiverTokens: [Int],
        batch: Int,
        time: Int,
        chunk: Int = 64
    ) -> (zRel: [Float], zContent: [Float]) {
        precondition(senderTokens.count == batch * time && receiverTokens.count == batch * time)
        let pe = MLXArray(HawkConditionalDecoder.positionalEncoding(time: time, dim: dim), [1, time, dim])
        var zRelOut: [Float] = []
        var zContOut: [Float] = []
        zRelOut.reserveCapacity(batch * latentDim)
        zContOut.reserveCapacity(batch * latentDim / 2)
        var start = 0
        while start < batch {
            let cur = min(chunk, batch - start)
            let sIds = MLXArray(senderTokens[(start * time)..<((start + cur) * time)].map { Int32($0) }, [cur, time])
            let cIds = MLXArray(receiverTokens[(start * time)..<((start + cur) * time)].map { Int32($0) }, [cur, time])
            let out = forwardDualStream(senderIds: sIds, receiverIds: cIds, pe: pe)
            eval(out.zRel, out.zContent)
            zRelOut.append(contentsOf: out.zRel.asArray(Float.self))
            zContOut.append(contentsOf: out.zContent.asArray(Float.self))
            start += cur
        }
        return (zRelOut, zContOut)
    }

    /// MLX GPU counterpart of `SWTHawkAutoencoder.encode(tokens:batch:time:)`: the neural SWT-Griffin passes run on GPU,
    /// while the (cheap) clause split and subband-lattice bookkeeping for the reconstruction latent run on the host.
    public func encodeLatentGPU(tokens: [Int], batch: Int, time: Int) -> [Float] {
        let half = latentDim / 2
        if subbandClassifier.isConfigured && latentDim >= 256 && !subbandClassifier.modulationRules.isEmpty {
            // Rule-modulated legacy branch has no neural network pass (embedding averages only).
            return SWTHawkAutoencoder(weights: exportWeights()).encodeCPU(tokens: tokens, batch: batch, time: time)
        }
        let zNeural = encodeNeural(tokens: tokens, batch: batch, time: time)
        guard subbandClassifier.isConfigured && latentDim >= 256 else {
            return zNeural
        }

        var splits: [SubbandClauseSplitter.SplitResult] = []
        splits.reserveCapacity(batch)
        var llFlat: [Int] = []
        llFlat.reserveCapacity(batch * time)
        for b in 0..<batch {
            let slice = Array(tokens[(b * time)..<((b + 1) * time)])
            let split = SubbandClauseSplitter.split(tokens: slice, classifier: subbandClassifier)
            splits.append(split)
            var llPadded = Array(split.llTokens.prefix(time))
            if llPadded.count < time {
                llPadded.append(contentsOf: repeatElement(TextVocabulary.padId, count: time - llPadded.count))
            }
            llFlat.append(contentsOf: llPadded)
        }
        let llNeural = encodeNeural(tokens: llFlat, batch: batch, time: time)
        let gamma = pooling.normGamma.asArray(Float.self)
        let beta = pooling.normBeta.asArray(Float.self)

        var zOut = [Float](repeating: 0.0, count: batch * latentDim)
        for b in 0..<batch {
            let off = b * latentDim
            let zCal = SubbandClauseSplitter.embedSubbandLattice(
                split: splits[b],
                neuralLL: Array(llNeural[off..<(off + half)]),
                neuralHH: Array(zNeural[(off + half)..<(off + latentDim)]),
                gamma: gamma,
                beta: beta,
                latentDim: latentDim
            )
            for c in 0..<latentDim {
                zOut[off + c] = zCal[c]
            }
        }
        return zOut
    }

    /// Returns a host-side `SWTHawkAutoencoder` (current weights snapshot) whose `encode` and relational encoders run on this MLX GPU model.
    /// Decoding / ODE / pentagon arithmetic stay on the host; all neural encoder passes are offloaded to the GPU.
    public func makeAcceleratedCPUModel() -> SWTHawkAutoencoder {
        let cpu = SWTHawkAutoencoder(weights: exportWeights())
        cpu.latentEncoderBackend = { [unowned self] tokens, batch, time in
            self.encodeLatentGPU(tokens: tokens, batch: batch, time: time)
        }
        cpu.relationalEncoderBackend = { [unowned self] s, c, batch, time in
            self.encodeRelationalGPU(senderTokens: s, receiverTokens: c, batch: batch, time: time)
        }
        return cpu
    }

    /// Teacher-forced forward pass returning `(z: [B, 1024], logits: [B, T, V])`, with optional latent Gaussian noise `zNoise`.
    public func callAsFunction(
        tokenIds: MLXArray,
        prevTokenIds: MLXArray,
        dropoutKeepMask: MLXArray,
        pe: MLXArray,
        phi: MLXArray? = nil,
        zNoise: MLXArray? = nil
    ) -> (z: MLXArray, logits: MLXArray) {
        let time = tokenIds.shape[1]
        let latentPhi = phi ?? MLXArray(HawkConditionalDecoder.latentBasis(time: time, latentDim: latentDim), [1, time, latentDim])
        let z = encode(tokenIds, pe: pe, phi: latentPhi)
        let zInput = (zNoise != nil) ? (z + zNoise!) : z
        let prevEmbedded = tokenEmbedding[prevTokenIds]
        let logits = decoder(z: zInput, prevEmbedded: prevEmbedded, dropoutKeepMask: dropoutKeepMask, pe: pe, phi: latentPhi)
        return (z, logits)
    }

    /// Loads weights through `update(parameters:)` so the module tree and compiled graphs keep the same parameter slots.
    public func importWeights(from w: SWTHawkWeights) {
        let d = w.dim
        let lDim = w.latentDim
        let m = w.mlpDim
        let v = w.vocabSize
        let k = w.pooling.numQueries
        let cd1In = w.swtBranch.convCD1.inDim
        let cd2In = w.swtBranch.convCD2.inDim

        var items: [(String, MLXArray)] = [
            ("tokenEmbedding", MLXArray(w.tokenEmbedding, [v, d])),
            ("swtBranch.convCD1.w", MLXArray(w.swtBranch.convCD1.w, [3, cd1In, d])),
            ("swtBranch.convCD1.b", MLXArray(w.swtBranch.convCD1.b, [d])),
            ("swtBranch.convCD2.w", MLXArray(w.swtBranch.convCD2.w, [3, cd2In, d])),
            ("swtBranch.convCD2.b", MLXArray(w.swtBranch.convCD2.b, [d])),
            ("swtBranch.convCA2.w", MLXArray(w.swtBranch.convCA2.w, [3, d, d])),
            ("swtBranch.convCA2.b", MLXArray(w.swtBranch.convCA2.b, [d])),
            ("swtBranch.wDetailRouter", MLXArray(w.swtBranch.wDetailRouter, [2 * d, 2 * d])),
            ("swtBranch.bDetailRouter", MLXArray(w.swtBranch.bDetailRouter, [2 * d])),
            ("swtBranch.wGate", MLXArray(w.swtBranch.wGate, [2 * d, d])),
            ("swtBranch.bGate", MLXArray(w.swtBranch.bGate, [d])),
            ("fusion.wSwiGLUGate", MLXArray(w.fusion.wSwiGLUGate, [2 * d, d])),
            ("fusion.bSwiGLUGate", MLXArray(w.fusion.bSwiGLUGate, [d])),
            ("fusion.wSwiGLUUp", MLXArray(w.fusion.wSwiGLUUp, [2 * d, d])),
            ("fusion.bSwiGLUUp", MLXArray(w.fusion.bSwiGLUUp, [d])),
            ("fusion.normGamma", MLXArray(w.fusion.normGamma, [d])),
            ("fusion.normBeta", MLXArray(w.fusion.normBeta, [d])),
            ("pooling.qPool", MLXArray(w.pooling.qPool, [k, d])),
            ("pooling.wK", MLXArray(w.pooling.wK, [d, d])),
            ("pooling.bK", MLXArray(w.pooling.bK, [d])),
            ("pooling.wV", MLXArray(w.pooling.wV, [d, d])),
            ("pooling.bV", MLXArray(w.pooling.bV, [d])),
            ("pooling.wO", MLXArray(w.pooling.wO, [d, d])),
            ("pooling.bO", MLXArray(w.pooling.bO, [d])),
            ("pooling.wProj", MLXArray(w.pooling.wProj, [3 * d, lDim])),
            ("pooling.bProj", MLXArray(w.pooling.bProj, [lDim])),
            ("pooling.normGamma", MLXArray(w.pooling.normGamma, [lDim])),
            ("pooling.normBeta", MLXArray(w.pooling.normBeta, [lDim])),
            ("decoder.wCross", MLXArray(w.decoder.wCross, [lDim, d])),
            ("decoder.bCross", MLXArray(w.decoder.bCross, [d])),
            ("decoder.finalNormGamma", MLXArray(w.decoder.finalNormGamma, [d])),
            ("decoder.finalNormBeta", MLXArray(w.decoder.finalNormBeta, [d])),
            ("decoder.wReadout", MLXArray(w.decoder.wReadout, [d, v])),
            ("decoder.bReadout", MLXArray(w.decoder.bReadout, [v]))
        ]

        for l in 0..<min(hawkBranch.count, w.hawkBranch.count) {
            let hb = w.hawkBranch[l]
            let p = "hawkBranch.\(l)."
            items.append((p + "norm1Gamma", MLXArray(hb.norm1Gamma, [d])))
            items.append((p + "norm1Beta", MLXArray(hb.norm1Beta, [d])))
            items.append((p + "wGate", MLXArray(hb.wGate, [d, d])))
            items.append((p + "bGate", MLXArray(hb.bGate, [d])))
            items.append((p + "wA", MLXArray(hb.rglru.wA, [d, d])))
            items.append((p + "bA", MLXArray(hb.rglru.bA, [d])))
            items.append((p + "wX", MLXArray(hb.rglru.wX, [d, d])))
            items.append((p + "bX", MLXArray(hb.rglru.bX, [d])))
            items.append((p + "lambda", MLXArray(hb.rglru.lambda, [d])))
            items.append((p + "wOut", MLXArray(hb.wOut, [d, d])))
            items.append((p + "bOut", MLXArray(hb.bOut, [d])))
            items.append((p + "localAttn.normGamma", MLXArray(hb.localAttn.normGamma, [d])))
            items.append((p + "localAttn.normBeta", MLXArray(hb.localAttn.normBeta, [d])))
            items.append((p + "localAttn.wQ", MLXArray(hb.localAttn.wQ, [d, d])))
            items.append((p + "localAttn.bQ", MLXArray(hb.localAttn.bQ, [d])))
            items.append((p + "localAttn.wK", MLXArray(hb.localAttn.wK, [d, d])))
            items.append((p + "localAttn.bK", MLXArray(hb.localAttn.bK, [d])))
            items.append((p + "localAttn.wV", MLXArray(hb.localAttn.wV, [d, d])))
            items.append((p + "localAttn.bV", MLXArray(hb.localAttn.bV, [d])))
            items.append((p + "localAttn.wOut", MLXArray(hb.localAttn.wOut, [d, d])))
            items.append((p + "localAttn.bOut", MLXArray(hb.localAttn.bOut, [d])))
            items.append((p + "norm2Gamma", MLXArray(hb.norm2Gamma, [d])))
            items.append((p + "norm2Beta", MLXArray(hb.norm2Beta, [d])))
            items.append((p + "wMLPUp", MLXArray(hb.wMLPUp, [d, m])))
            items.append((p + "bMLPUp", MLXArray(hb.bMLPUp, [m])))
            items.append((p + "wMLPDown", MLXArray(hb.wMLPDown, [m, d])))
            items.append((p + "bMLPDown", MLXArray(hb.bMLPDown, [d])))
        }

        for l in 0..<min(decoder.layers.count, w.decoder.layers.count) {
            let dl = w.decoder.layers[l]
            let hb = dl.block
            let p = "decoder.layers.\(l)."
            items.append((p + "wH0", MLXArray(dl.wH0, [lDim, d])))
            items.append((p + "bH0", MLXArray(dl.bH0, [d])))
            items.append((p + "wAdaGamma", MLXArray(dl.wAdaGamma, [lDim, d])))
            items.append((p + "bAdaGamma", MLXArray(dl.bAdaGamma, [d])))
            items.append((p + "wAdaBeta", MLXArray(dl.wAdaBeta, [lDim, d])))
            items.append((p + "bAdaBeta", MLXArray(dl.bAdaBeta, [d])))
            let bp = p + "block."
            items.append((bp + "norm1Gamma", MLXArray(hb.norm1Gamma, [d])))
            items.append((bp + "norm1Beta", MLXArray(hb.norm1Beta, [d])))
            items.append((bp + "wGate", MLXArray(hb.wGate, [d, d])))
            items.append((bp + "bGate", MLXArray(hb.bGate, [d])))
            items.append((bp + "wA", MLXArray(hb.rglru.wA, [d, d])))
            items.append((bp + "bA", MLXArray(hb.rglru.bA, [d])))
            items.append((bp + "wX", MLXArray(hb.rglru.wX, [d, d])))
            items.append((bp + "bX", MLXArray(hb.rglru.bX, [d])))
            items.append((bp + "lambda", MLXArray(hb.rglru.lambda, [d])))
            items.append((bp + "wOut", MLXArray(hb.wOut, [d, d])))
            items.append((bp + "bOut", MLXArray(hb.bOut, [d])))
            items.append((bp + "localAttn.normGamma", MLXArray(hb.localAttn.normGamma, [d])))
            items.append((bp + "localAttn.normBeta", MLXArray(hb.localAttn.normBeta, [d])))
            items.append((bp + "localAttn.wQ", MLXArray(hb.localAttn.wQ, [d, d])))
            items.append((bp + "localAttn.bQ", MLXArray(hb.localAttn.bQ, [d])))
            items.append((bp + "localAttn.wK", MLXArray(hb.localAttn.wK, [d, d])))
            items.append((bp + "localAttn.bK", MLXArray(hb.localAttn.bK, [d])))
            items.append((bp + "localAttn.wV", MLXArray(hb.localAttn.wV, [d, d])))
            items.append((bp + "localAttn.bV", MLXArray(hb.localAttn.bV, [d])))
            items.append((bp + "localAttn.wOut", MLXArray(hb.localAttn.wOut, [d, d])))
            items.append((bp + "localAttn.bOut", MLXArray(hb.localAttn.bOut, [d])))
            items.append((bp + "norm2Gamma", MLXArray(hb.norm2Gamma, [d])))
            items.append((bp + "norm2Beta", MLXArray(hb.norm2Beta, [d])))
            items.append((bp + "wMLPUp", MLXArray(hb.wMLPUp, [d, m])))
            items.append((bp + "bMLPUp", MLXArray(hb.bMLPUp, [m])))
            items.append((bp + "wMLPDown", MLXArray(hb.wMLPDown, [m, d])))
            items.append((bp + "bMLPDown", MLXArray(hb.bMLPDown, [d])))
        }

        let half = lDim / 2
        let cohDim = DualStreamRelationalWeights.coherenceDim
        let rel = w.relational
        items += [
            ("relational.eEcho", MLXArray(rel.eEcho, [d])),
            ("relational.eTopic", MLXArray(rel.eTopic, [d])),
            ("relational.wQLL", MLXArray(rel.wQLL, [d, d])),
            ("relational.bQLL", MLXArray(rel.bQLL, [d])),
            ("relational.wKLL", MLXArray(rel.wKLL, [d, d])),
            ("relational.bKLL", MLXArray(rel.bKLL, [d])),
            ("relational.wVLL", MLXArray(rel.wVLL, [d, d])),
            ("relational.bVLL", MLXArray(rel.bVLL, [d])),
            ("relational.wCohLL", MLXArray(rel.wCohLL, [cohDim, d])),
            ("relational.bCohLL", MLXArray(rel.bCohLL, [d])),
            ("relational.wQHH", MLXArray(rel.wQHH, [d, d])),
            ("relational.bQHH", MLXArray(rel.bQHH, [d])),
            ("relational.wKHH", MLXArray(rel.wKHH, [d, d])),
            ("relational.bKHH", MLXArray(rel.bKHH, [d])),
            ("relational.wVHH", MLXArray(rel.wVHH, [d, d])),
            ("relational.bVHH", MLXArray(rel.bVHH, [d])),
            ("relational.wCohHH", MLXArray(rel.wCohHH, [cohDim, d])),
            ("relational.bCohHH", MLXArray(rel.bCohHH, [d])),
            ("relational.wProjLL", MLXArray(rel.wProjLL, [5 * d, half])),
            ("relational.bProjLL", MLXArray(rel.bProjLL, [half])),
            ("relational.normLLGamma", MLXArray(rel.normLLGamma, [half])),
            ("relational.normLLBeta", MLXArray(rel.normLLBeta, [half])),
            ("relational.wProjHH", MLXArray(rel.wProjHH, [5 * d, half])),
            ("relational.bProjHH", MLXArray(rel.bProjHH, [half])),
            ("relational.normHHGamma", MLXArray(rel.normHHGamma, [half])),
            ("relational.normHHBeta", MLXArray(rel.normHHBeta, [half])),
            ("relational.wContent", MLXArray(rel.wContent, [2 * d, half])),
            ("relational.bContent", MLXArray(rel.bContent, [half])),
            ("relational.wArchT4", MLXArray(rel.wArchT4, [lDim, 6])),
            ("relational.bArchT4", MLXArray(rel.bArchT4, [6])),
            ("relational.wTopoT4", MLXArray(rel.wTopoT4, [lDim, 5])),
            ("relational.bTopoT4", MLXArray(rel.bTopoT4, [5])),
            ("relational.wArchT2", MLXArray(rel.wArchT2, [lDim, 6])),
            ("relational.bArchT2", MLXArray(rel.bArchT2, [6])),
            ("relational.wTopoT2", MLXArray(rel.wTopoT2, [lDim, 5])),
            ("relational.bTopoT2", MLXArray(rel.bTopoT2, [5]))
        ]
        relational.alphaRel = rel.alphaRel
        relational.calibratedHeads = rel.calibratedHeads
        relational.auditTelemetry = rel.auditTelemetry

        subbandClassifier = w.subbandClassifier
        flowMatchingWeights = w.flowMatching
        update(parameters: ModuleParameters.unflattened(items))
        eval(self)
    }

    /// Exports current MLX GPU parameters to `SWTHawkWeights` for `.safetensors` serialization and CPU/GPU evaluation.
    public func exportWeights() -> SWTHawkWeights {
        eval(self)
        func vals(_ a: MLXArray) -> [Float] {
            return a.asArray(Float.self)
        }

        let swt = SWTBranchWeights(
            dim: dim,
            convCD1: Conv1D3Weights(inDim: swtBranch.convCD1.inDim, outDim: dim, w: vals(swtBranch.convCD1.w), b: vals(swtBranch.convCD1.b)),
            convCD2: Conv1D3Weights(inDim: swtBranch.convCD2.inDim, outDim: dim, w: vals(swtBranch.convCD2.w), b: vals(swtBranch.convCD2.b)),
            convCA2: Conv1D3Weights(inDim: dim, outDim: dim, w: vals(swtBranch.convCA2.w), b: vals(swtBranch.convCA2.b)),
            wDetailRouter: vals(swtBranch.wDetailRouter),
            bDetailRouter: vals(swtBranch.bDetailRouter),
            wGate: vals(swtBranch.wGate),
            bGate: vals(swtBranch.bGate)
        )

        func exportLocalAttn(_ la: MLXLocalAttention) -> LocalAttentionWeights {
            return LocalAttentionWeights(
                dim: dim,
                numHeads: la.numHeads,
                windowSize: la.windowSize,
                normGamma: vals(la.normGamma),
                normBeta: vals(la.normBeta),
                wQ: vals(la.wQ),
                bQ: vals(la.bQ),
                wK: vals(la.wK),
                bK: vals(la.bK),
                wV: vals(la.wV),
                bV: vals(la.bV),
                wOut: vals(la.wOut),
                bOut: vals(la.bOut)
            )
        }

        let encHawk = hawkBranch.map { hb in
            HawkBlockWeights(
                dim: dim,
                mlpDim: mlpDim,
                norm1Gamma: vals(hb.norm1Gamma),
                norm1Beta: vals(hb.norm1Beta),
                wGate: vals(hb.wGate),
                bGate: vals(hb.bGate),
                rglru: RGLRUWeights(
                    dim: dim,
                    wA: vals(hb.wA),
                    bA: vals(hb.bA),
                    wX: vals(hb.wX),
                    bX: vals(hb.bX),
                    lambda: vals(hb.lambda)
                ),
                wOut: vals(hb.wOut),
                bOut: vals(hb.bOut),
                localAttn: exportLocalAttn(hb.localAttn),
                norm2Gamma: vals(hb.norm2Gamma),
                norm2Beta: vals(hb.norm2Beta),
                wMLPUp: vals(hb.wMLPUp),
                bMLPUp: vals(hb.bMLPUp),
                wMLPDown: vals(hb.wMLPDown),
                bMLPDown: vals(hb.bMLPDown)
            )
        }

        let fus = GatedFusionWeights(
            dim: dim,
            wSwiGLUGate: vals(fusion.wSwiGLUGate),
            bSwiGLUGate: vals(fusion.bSwiGLUGate),
            wSwiGLUUp: vals(fusion.wSwiGLUUp),
            bSwiGLUUp: vals(fusion.bSwiGLUUp),
            normGamma: vals(fusion.normGamma),
            normBeta: vals(fusion.normBeta)
        )

        let pool = AttentionPoolingWeights(
            dim: dim,
            latentDim: latentDim,
            numQueries: pooling.numQueries,
            numHeads: pooling.numHeads,
            qPool: vals(pooling.qPool),
            wK: vals(pooling.wK),
            bK: vals(pooling.bK),
            wV: vals(pooling.wV),
            bV: vals(pooling.bV),
            wO: vals(pooling.wO),
            bO: vals(pooling.bO),
            wProj: vals(pooling.wProj),
            bProj: vals(pooling.bProj),
            normGamma: vals(pooling.normGamma),
            normBeta: vals(pooling.normBeta),
            l2Normalize: pooling.l2Normalize
        )

        let decLayers = decoder.layers.map { dl in
            let hb = dl.block
            return ConditionalHawkLayerWeights(
                dim: dim,
                latentDim: latentDim,
                block: HawkBlockWeights(
                    dim: dim,
                    mlpDim: mlpDim,
                    norm1Gamma: vals(hb.norm1Gamma),
                    norm1Beta: vals(hb.norm1Beta),
                    wGate: vals(hb.wGate),
                    bGate: vals(hb.bGate),
                    rglru: RGLRUWeights(
                        dim: dim,
                        wA: vals(hb.wA),
                        bA: vals(hb.bA),
                        wX: vals(hb.wX),
                        bX: vals(hb.bX),
                        lambda: vals(hb.lambda)
                    ),
                    wOut: vals(hb.wOut),
                    bOut: vals(hb.bOut),
                    localAttn: exportLocalAttn(hb.localAttn),
                    norm2Gamma: vals(hb.norm2Gamma),
                    norm2Beta: vals(hb.norm2Beta),
                    wMLPUp: vals(hb.wMLPUp),
                    bMLPUp: vals(hb.bMLPUp),
                    wMLPDown: vals(hb.wMLPDown),
                    bMLPDown: vals(hb.bMLPDown)
                ),
                wH0: vals(dl.wH0),
                bH0: vals(dl.bH0),
                wAdaGamma: vals(dl.wAdaGamma),
                bAdaGamma: vals(dl.bAdaGamma),
                wAdaBeta: vals(dl.wAdaBeta),
                bAdaBeta: vals(dl.bAdaBeta)
            )
        }

        let dec = HawkDecoderWeights(
            dim: dim,
            latentDim: latentDim,
            vocabSize: vocabSize,
            wCross: vals(decoder.wCross),
            bCross: vals(decoder.bCross),
            layers: decLayers,
            finalNormGamma: vals(decoder.finalNormGamma),
            finalNormBeta: vals(decoder.finalNormBeta),
            wReadout: vals(decoder.wReadout),
            bReadout: vals(decoder.bReadout)
        )

        let rel = DualStreamRelationalWeights(
            vocabSize: vocabSize,
            dim: dim,
            latentDim: latentDim,
            alphaRel: relational.alphaRel,
            eEcho: vals(relational.eEcho),
            eTopic: vals(relational.eTopic),
            wQLL: vals(relational.wQLL),
            bQLL: vals(relational.bQLL),
            wKLL: vals(relational.wKLL),
            bKLL: vals(relational.bKLL),
            wVLL: vals(relational.wVLL),
            bVLL: vals(relational.bVLL),
            wCohLL: vals(relational.wCohLL),
            bCohLL: vals(relational.bCohLL),
            wQHH: vals(relational.wQHH),
            bQHH: vals(relational.bQHH),
            wKHH: vals(relational.wKHH),
            bKHH: vals(relational.bKHH),
            wVHH: vals(relational.wVHH),
            bVHH: vals(relational.bVHH),
            wCohHH: vals(relational.wCohHH),
            bCohHH: vals(relational.bCohHH),
            wProjLL: vals(relational.wProjLL),
            bProjLL: vals(relational.bProjLL),
            normLLGamma: vals(relational.normLLGamma),
            normLLBeta: vals(relational.normLLBeta),
            wProjHH: vals(relational.wProjHH),
            bProjHH: vals(relational.bProjHH),
            normHHGamma: vals(relational.normHHGamma),
            normHHBeta: vals(relational.normHHBeta),
            wContent: vals(relational.wContent),
            bContent: vals(relational.bContent),
            wArchT4: vals(relational.wArchT4),
            bArchT4: vals(relational.bArchT4),
            wTopoT4: vals(relational.wTopoT4),
            bTopoT4: vals(relational.bTopoT4),
            wArchT2: vals(relational.wArchT2),
            bArchT2: vals(relational.bArchT2),
            wTopoT2: vals(relational.wTopoT2),
            bTopoT2: vals(relational.bTopoT2),
            calibratedHeads: relational.calibratedHeads,
            auditTelemetry: relational.auditTelemetry
        )

        return SWTHawkWeights(
            vocabSize: vocabSize,
            dim: dim,
            latentDim: latentDim,
            mlpDim: mlpDim,
            serializedVocabulary: serializedVocabulary,
            subbandClassifier: subbandClassifier,
            tokenEmbedding: vals(tokenEmbedding),
            swtBranch: swt,
            hawkBranch: encHawk,
            fusion: fus,
            pooling: pool,
            decoder: dec,
            flowMatching: flowMatchingWeights,
            relational: rel
        )
    }
}
