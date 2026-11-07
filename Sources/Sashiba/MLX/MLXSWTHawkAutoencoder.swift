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

/// Complete SWT-Griffin Text Autoencoder (`MLXSWTFrequencyBranch` + `MLXHawkBlock` + `MLXGatedTimeFrequencyFusion`
/// + `MLXMultiScaleAttentionPooling` + `MLXHawkConditionalDecoder`) on MLX GPU (Metal / CUDA).
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
            flowMatching: flowMatchingWeights
        )
    }
}
