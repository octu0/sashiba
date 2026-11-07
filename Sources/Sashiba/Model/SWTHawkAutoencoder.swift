#if canImport(Accelerate)
import Accelerate
#endif
import Foundation

/// Deterministic 64-bit SplitMix PRNG so weight initialization and word dropout are reproducible across platforms.
public struct SplitMix64: RandomNumberGenerator, Sendable {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed &+ 0x9E37_79B9_7F4A_7C15
    }

    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform float in `[0, 1)`.
    public mutating func nextUnitFloat() -> Float {
        return Float(next() >> 40) * (1.0 / 16_777_216.0)
    }

    /// Uniform float in `(-scale, +scale)`.
    public mutating func nextSignedUniform(scale: Float) -> Float {
        return (2.0 * nextUnitFloat() - 1.0) * scale
    }

    /// Standard normal `N(0, 1)` sample via Box-Muller transform.
    public mutating func nextGaussian() -> Float {
        let u1 = max(1e-7, nextUnitFloat())
        let u2 = nextUnitFloat()
        return sqrtf(-2.0 * logf(u1)) * cosf(2.0 * Float.pi * u2)
    }
}

/// Default architecture dimensions for the Phase 2 SWT-Griffin Text Autoencoder (`Section 2`).
public enum SWTHawkShape {
    public static let defaultDim: Int = 128
    public static let latentDim: Int = 1024
    public static let encoderLayers: Int = 2
    public static let decoderLayers: Int = 2
    public static let numQueries: Int = 8
    public static let numHeads: Int = 4
    public static let windowSize: Int = 32
    public static let layerNormEps: Float = 1e-5
}

/// Complete weights of the SWT-Griffin Text Autoencoder & Conditional Flow Matcher, serializable to `.safetensors`.
public struct SWTHawkWeights: Sendable, Equatable {
    public let vocabSize: Int
    public let dim: Int
    public let latentDim: Int
    public let mlpDim: Int
    public let serializedVocabulary: String
    public let subbandClassifier: SubbandTokenClassifier

    public let tokenEmbedding: [Float] // [vocabSize, D]
    public let swtBranch: SWTBranchWeights
    public let hawkBranch: [HawkBlockWeights]
    public let fusion: GatedFusionWeights
    public let pooling: AttentionPoolingWeights
    public let decoder: HawkDecoderWeights
    public let flowMatching: FlowMatchingWeights

    public init(
        vocabSize: Int,
        dim: Int,
        latentDim: Int = SWTHawkShape.latentDim,
        mlpDim: Int,
        serializedVocabulary: String = "",
        subbandClassifier: SubbandTokenClassifier = SubbandTokenClassifier(),
        tokenEmbedding: [Float],
        swtBranch: SWTBranchWeights,
        hawkBranch: [HawkBlockWeights],
        fusion: GatedFusionWeights,
        pooling: AttentionPoolingWeights,
        decoder: HawkDecoderWeights,
        flowMatching: FlowMatchingWeights? = nil
    ) {
        self.vocabSize = vocabSize
        self.dim = dim
        self.latentDim = latentDim
        self.mlpDim = mlpDim
        self.serializedVocabulary = serializedVocabulary
        self.subbandClassifier = subbandClassifier
        self.tokenEmbedding = tokenEmbedding
        self.swtBranch = swtBranch
        self.hawkBranch = hawkBranch
        self.fusion = fusion
        self.pooling = pooling
        self.decoder = decoder
        if let fm = flowMatching {
            self.flowMatching = fm
        } else {
            var fmRng = SplitMix64(seed: 999)
            self.flowMatching = FlowMatchingWeights.makeRandom(latentDim: latentDim, rng: &fmRng)
        }
    }

    public var parameterCount: Int {
        return tokenEmbedding.count + swtBranch.parameterCount + fusion.parameterCount + pooling.parameterCount + decoder.parameterCount + flowMatching.parameterCount + hawkBranch.reduce(0) { $0 + $1.parameterCount }
    }

    public static func makeRandom(
        vocabSize: Int,
        dim: Int = SWTHawkShape.defaultDim,
        latentDim: Int = SWTHawkShape.latentDim,
        mlpDim: Int? = nil,
        encoderLayers: Int = SWTHawkShape.encoderLayers,
        decoderLayers: Int = SWTHawkShape.decoderLayers,
        numQueries: Int = SWTHawkShape.numQueries,
        numHeads: Int = SWTHawkShape.numHeads,
        l2NormalizeLatent: Bool = false,
        serializedVocabulary: String = "",
        subbandClassifier: SubbandTokenClassifier = SubbandTokenClassifier(),
        seed: UInt64 = 42
    ) -> SWTHawkWeights {
        var rng = SplitMix64(seed: seed)
        let m = mlpDim ?? (2 * dim)
        let embBound: Float = 1.0
        var emb = [Float](repeating: 0.0, count: vocabSize * dim)
        for i in 0..<emb.count {
            emb[i] = rng.nextSignedUniform(scale: embBound)
        }

        let swt = SWTBranchWeights.makeRandom(dim: dim, rng: &rng)
        let encResScale = 1.0 / sqrtf(Float(max(1, 2 * encoderLayers)))
        var encHawk: [HawkBlockWeights] = []
        encHawk.reserveCapacity(encoderLayers)
        for _ in 0..<encoderLayers {
            encHawk.append(HawkBlockWeights.makeRandom(dim: dim, mlpDim: m, residualScale: encResScale, rng: &rng))
        }

        let fus = GatedFusionWeights.makeRandom(dim: dim, rng: &rng)
        let pool = AttentionPoolingWeights.makeRandom(
            dim: dim, latentDim: latentDim, numQueries: numQueries,
            numHeads: min(numHeads, dim), l2Normalize: l2NormalizeLatent, rng: &rng
        )
        let dec = HawkDecoderWeights.makeRandom(
            dim: dim, latentDim: latentDim, vocabSize: vocabSize,
            numLayers: decoderLayers, mlpDim: m, rng: &rng
        )
        let fm = FlowMatchingWeights.makeRandom(latentDim: latentDim, rng: &rng)

        return SWTHawkWeights(
            vocabSize: vocabSize,
            dim: dim,
            latentDim: latentDim,
            mlpDim: m,
            serializedVocabulary: serializedVocabulary,
            subbandClassifier: subbandClassifier,
            tokenEmbedding: emb,
            swtBranch: swt,
            hawkBranch: encHawk,
            fusion: fus,
            pooling: pool,
            decoder: dec,
            flowMatching: fm
        )
    }

    // MARK: - SafeTensors Serialization

    public func save(to url: URL) throws {
        typealias T = SafeTensors.Tensor
        let cd1In = swtBranch.convCD1.inDim
        let cd2In = swtBranch.convCD2.inDim
        var tensors: [(name: String, tensor: T)] = [
            ("tokenEmbedding", T(shape: [vocabSize, dim], values: tokenEmbedding)),
            ("swt.convCD1.w", T(shape: [3, cd1In, dim], values: swtBranch.convCD1.w)),
            ("swt.convCD1.b", T(shape: [dim], values: swtBranch.convCD1.b)),
            ("swt.convCD2.w", T(shape: [3, cd2In, dim], values: swtBranch.convCD2.w)),
            ("swt.convCD2.b", T(shape: [dim], values: swtBranch.convCD2.b)),
            ("swt.convCA2.w", T(shape: [3, dim, dim], values: swtBranch.convCA2.w)),
            ("swt.convCA2.b", T(shape: [dim], values: swtBranch.convCA2.b)),
            ("swt.wDetailRouter", T(shape: [2 * dim, 2 * dim], values: swtBranch.wDetailRouter)),
            ("swt.bDetailRouter", T(shape: [2 * dim], values: swtBranch.bDetailRouter)),
            ("swt.wGate", T(shape: [2 * dim, dim], values: swtBranch.wGate)),
            ("swt.bGate", T(shape: [dim], values: swtBranch.bGate))
        ]

        for (i, hb) in hawkBranch.enumerated() {
            let p = "hawk.\(i)."
            tensors += [
                (p + "norm1Gamma", T(shape: [dim], values: hb.norm1Gamma)),
                (p + "norm1Beta", T(shape: [dim], values: hb.norm1Beta)),
                (p + "wGate", T(shape: [dim, dim], values: hb.wGate)),
                (p + "bGate", T(shape: [dim], values: hb.bGate)),
                (p + "rglru.wA", T(shape: [dim, dim], values: hb.rglru.wA)),
                (p + "rglru.bA", T(shape: [dim], values: hb.rglru.bA)),
                (p + "rglru.wX", T(shape: [dim, dim], values: hb.rglru.wX)),
                (p + "rglru.bX", T(shape: [dim], values: hb.rglru.bX)),
                (p + "rglru.lambda", T(shape: [dim], values: hb.rglru.lambda)),
                (p + "wOut", T(shape: [dim, dim], values: hb.wOut)),
                (p + "bOut", T(shape: [dim], values: hb.bOut)),
                (p + "localAttn.normGamma", T(shape: [dim], values: hb.localAttn.normGamma)),
                (p + "localAttn.normBeta", T(shape: [dim], values: hb.localAttn.normBeta)),
                (p + "localAttn.wQ", T(shape: [dim, dim], values: hb.localAttn.wQ)),
                (p + "localAttn.bQ", T(shape: [dim], values: hb.localAttn.bQ)),
                (p + "localAttn.wK", T(shape: [dim, dim], values: hb.localAttn.wK)),
                (p + "localAttn.bK", T(shape: [dim], values: hb.localAttn.bK)),
                (p + "localAttn.wV", T(shape: [dim, dim], values: hb.localAttn.wV)),
                (p + "localAttn.bV", T(shape: [dim], values: hb.localAttn.bV)),
                (p + "localAttn.wOut", T(shape: [dim, dim], values: hb.localAttn.wOut)),
                (p + "localAttn.bOut", T(shape: [dim], values: hb.localAttn.bOut)),
                (p + "norm2Gamma", T(shape: [dim], values: hb.norm2Gamma)),
                (p + "norm2Beta", T(shape: [dim], values: hb.norm2Beta)),
                (p + "wMLPUp", T(shape: [dim, mlpDim], values: hb.wMLPUp)),
                (p + "bMLPUp", T(shape: [mlpDim], values: hb.bMLPUp)),
                (p + "wMLPDown", T(shape: [mlpDim, dim], values: hb.wMLPDown)),
                (p + "bMLPDown", T(shape: [dim], values: hb.bMLPDown))
            ]
        }

        tensors += [
            ("fusion.wSwiGLUGate", T(shape: [2 * dim, dim], values: fusion.wSwiGLUGate)),
            ("fusion.bSwiGLUGate", T(shape: [dim], values: fusion.bSwiGLUGate)),
            ("fusion.wSwiGLUUp", T(shape: [2 * dim, dim], values: fusion.wSwiGLUUp)),
            ("fusion.bSwiGLUUp", T(shape: [dim], values: fusion.bSwiGLUUp)),
            ("fusion.normGamma", T(shape: [dim], values: fusion.normGamma)),
            ("fusion.normBeta", T(shape: [dim], values: fusion.normBeta)),
            ("pooling.qPool", T(shape: [pooling.numQueries, dim], values: pooling.qPool)),
            ("pooling.wK", T(shape: [dim, dim], values: pooling.wK)),
            ("pooling.bK", T(shape: [dim], values: pooling.bK)),
            ("pooling.wV", T(shape: [dim, dim], values: pooling.wV)),
            ("pooling.bV", T(shape: [dim], values: pooling.bV)),
            ("pooling.wO", T(shape: [dim, dim], values: pooling.wO)),
            ("pooling.bO", T(shape: [dim], values: pooling.bO)),
            ("pooling.wProj", T(shape: [3 * dim, latentDim], values: pooling.wProj)),
            ("pooling.bProj", T(shape: [latentDim], values: pooling.bProj)),
            ("pooling.normGamma", T(shape: [latentDim], values: pooling.normGamma)),
            ("pooling.normBeta", T(shape: [latentDim], values: pooling.normBeta)),
            ("decoder.wCross", T(shape: [latentDim, dim], values: decoder.wCross)),
            ("decoder.bCross", T(shape: [dim], values: decoder.bCross)),
            ("decoder.finalNormGamma", T(shape: [dim], values: decoder.finalNormGamma)),
            ("decoder.finalNormBeta", T(shape: [dim], values: decoder.finalNormBeta)),
            ("decoder.wReadout", T(shape: [dim, vocabSize], values: decoder.wReadout)),
            ("decoder.bReadout", T(shape: [vocabSize], values: decoder.bReadout))
        ]

        for (i, dl) in decoder.layers.enumerated() {
            let p = "decoder.layers.\(i)."
            let hb = dl.block
            tensors += [
                (p + "wH0", T(shape: [latentDim, dim], values: dl.wH0)),
                (p + "bH0", T(shape: [dim], values: dl.bH0)),
                (p + "wAdaGamma", T(shape: [latentDim, dim], values: dl.wAdaGamma)),
                (p + "bAdaGamma", T(shape: [dim], values: dl.bAdaGamma)),
                (p + "wAdaBeta", T(shape: [latentDim, dim], values: dl.wAdaBeta)),
                (p + "bAdaBeta", T(shape: [dim], values: dl.bAdaBeta)),
                (p + "norm1Gamma", T(shape: [dim], values: hb.norm1Gamma)),
                (p + "norm1Beta", T(shape: [dim], values: hb.norm1Beta)),
                (p + "wGate", T(shape: [dim, dim], values: hb.wGate)),
                (p + "bGate", T(shape: [dim], values: hb.bGate)),
                (p + "rglru.wA", T(shape: [dim, dim], values: hb.rglru.wA)),
                (p + "rglru.bA", T(shape: [dim], values: hb.rglru.bA)),
                (p + "rglru.wX", T(shape: [dim, dim], values: hb.rglru.wX)),
                (p + "rglru.bX", T(shape: [dim], values: hb.rglru.bX)),
                (p + "rglru.lambda", T(shape: [dim], values: hb.rglru.lambda)),
                (p + "wOut", T(shape: [dim, dim], values: hb.wOut)),
                (p + "bOut", T(shape: [dim], values: hb.bOut)),
                (p + "localAttn.normGamma", T(shape: [dim], values: hb.localAttn.normGamma)),
                (p + "localAttn.normBeta", T(shape: [dim], values: hb.localAttn.normBeta)),
                (p + "localAttn.wQ", T(shape: [dim, dim], values: hb.localAttn.wQ)),
                (p + "localAttn.bQ", T(shape: [dim], values: hb.localAttn.bQ)),
                (p + "localAttn.wK", T(shape: [dim, dim], values: hb.localAttn.wK)),
                (p + "localAttn.bK", T(shape: [dim], values: hb.localAttn.bK)),
                (p + "localAttn.wV", T(shape: [dim, dim], values: hb.localAttn.wV)),
                (p + "localAttn.bV", T(shape: [dim], values: hb.localAttn.bV)),
                (p + "localAttn.wOut", T(shape: [dim, dim], values: hb.localAttn.wOut)),
                (p + "localAttn.bOut", T(shape: [dim], values: hb.localAttn.bOut)),
                (p + "norm2Gamma", T(shape: [dim], values: hb.norm2Gamma)),
                (p + "norm2Beta", T(shape: [dim], values: hb.norm2Beta)),
                (p + "wMLPUp", T(shape: [dim, mlpDim], values: hb.wMLPUp)),
                (p + "bMLPUp", T(shape: [mlpDim], values: hb.bMLPUp)),
                (p + "wMLPDown", T(shape: [mlpDim, dim], values: hb.wMLPDown)),
                (p + "bMLPDown", T(shape: [dim], values: hb.bMLPDown))
            ]
        }

        let fmH = flowMatching.hiddenDim
        tensors += [
            ("cfm.wAlign", T(shape: [fmH, fmH], values: flowMatching.wAlign)),
            ("cfm.bAlign", T(shape: [fmH], values: flowMatching.bAlign)),
            ("cfm.wTime", T(shape: [fmH, fmH], values: flowMatching.wTime)),
            ("cfm.bTime", T(shape: [fmH], values: flowMatching.bTime)),
            ("cfm.wVelIn", T(shape: [2 * fmH, fmH], values: flowMatching.wVelIn)),
            ("cfm.bVelIn", T(shape: [fmH], values: flowMatching.bVelIn)),
            ("cfm.wVelGate", T(shape: [fmH, fmH], values: flowMatching.wVelGate)),
            ("cfm.bVelGate", T(shape: [fmH], values: flowMatching.bVelGate)),
            ("cfm.wVelOut", T(shape: [fmH, fmH], values: flowMatching.wVelOut)),
            ("cfm.bVelOut", T(shape: [fmH], values: flowMatching.bVelOut))
        ]
        if flowMatching.numPairs > 0 {
            tensors += [
                ("cfm.condBank", T(shape: [flowMatching.numPairs, latentDim], values: flowMatching.condBank)),
                ("cfm.summaryBank", T(shape: [flowMatching.numPairs, latentDim], values: flowMatching.summaryBank))
            ]
        }

        let metadata: [String: String] = [
            "vocabSize": "\(vocabSize)",
            "dim": "\(dim)",
            "latentDim": "\(latentDim)",
            "mlpDim": "\(mlpDim)",
            "encoderLayers": "\(hawkBranch.count)",
            "decoderLayers": "\(decoder.layers.count)",
            "numQueries": "\(pooling.numQueries)",
            "numHeads": "\(pooling.numHeads)",
            "l2Normalize": pooling.l2Normalize ? "1" : "0",
            "cfmHiddenDim": "\(fmH)",
            "cfmNumPairs": "\(flowMatching.numPairs)",
            "vocabulary": serializedVocabulary,
            "subbandClassifier": subbandClassifier.serialize()
        ]
        try SafeTensors.write(tensors, metadata: metadata, to: url)
    }

    public static func load(from url: URL) throws -> SWTHawkWeights {
        let (tensors, metadata) = try SafeTensors.read(from: url)
        func intMeta(_ key: String) throws -> Int {
            guard let s = metadata[key], let v = Int(s) else {
                throw SafeTensors.Failure.badHeader
            }
            return v
        }
        func vals(_ name: String) throws -> [Float] {
            guard let t = tensors[name] else {
                throw SafeTensors.Failure.badShape(name)
            }
            return t.values
        }

        let vocabSize = try intMeta("vocabSize")
        let dim = try intMeta("dim")
        let latentDim = try intMeta("latentDim")
        let mlpDim = try intMeta("mlpDim")
        let encoderLayers = try intMeta("encoderLayers")
        let decoderLayers = try intMeta("decoderLayers")
        let numQueries = try intMeta("numQueries")
        let numHeads = try intMeta("numHeads")
        let l2Norm = (metadata["l2Normalize"] == "1")
        let vocabStr = metadata["vocabulary"] ?? ""
        let classifier = SubbandTokenClassifier.deserialize(metadata["subbandClassifier"] ?? "")

        let swt = SWTBranchWeights(
            dim: dim,
            convCD1: Conv1D3Weights(outDim: dim, w: try vals("swt.convCD1.w"), b: try vals("swt.convCD1.b")),
            convCD2: Conv1D3Weights(outDim: dim, w: try vals("swt.convCD2.w"), b: try vals("swt.convCD2.b")),
            convCA2: Conv1D3Weights(inDim: dim, outDim: dim, w: try vals("swt.convCA2.w"), b: try vals("swt.convCA2.b")),
            wDetailRouter: try vals("swt.wDetailRouter"),
            bDetailRouter: try vals("swt.bDetailRouter"),
            wGate: try vals("swt.wGate"),
            bGate: try vals("swt.bGate")
        )

        func loadLocalAttn(_ p: String) throws -> LocalAttentionWeights? {
            guard tensors[p + "localAttn.wQ"] != nil else { return nil }
            let heads = (dim % LocalAttentionWeights.defaultNumHeads == 0) ? LocalAttentionWeights.defaultNumHeads : 1
            return LocalAttentionWeights(
                dim: dim,
                numHeads: heads,
                windowSize: LocalAttentionWeights.defaultWindowSize,
                normGamma: try vals(p + "localAttn.normGamma"),
                normBeta: try vals(p + "localAttn.normBeta"),
                wQ: try vals(p + "localAttn.wQ"),
                bQ: try vals(p + "localAttn.bQ"),
                wK: try vals(p + "localAttn.wK"),
                bK: try vals(p + "localAttn.bK"),
                wV: try vals(p + "localAttn.wV"),
                bV: try vals(p + "localAttn.bV"),
                wOut: try vals(p + "localAttn.wOut"),
                bOut: try vals(p + "localAttn.bOut")
            )
        }

        var encHawk: [HawkBlockWeights] = []
        for i in 0..<encoderLayers {
            let p = "hawk.\(i)."
            let rglru = RGLRUWeights(
                dim: dim,
                wA: try vals(p + "rglru.wA"),
                bA: try vals(p + "rglru.bA"),
                wX: try vals(p + "rglru.wX"),
                bX: try vals(p + "rglru.bX"),
                lambda: try vals(p + "rglru.lambda")
            )
            encHawk.append(HawkBlockWeights(
                dim: dim,
                mlpDim: mlpDim,
                norm1Gamma: try vals(p + "norm1Gamma"),
                norm1Beta: try vals(p + "norm1Beta"),
                wGate: try vals(p + "wGate"),
                bGate: try vals(p + "bGate"),
                rglru: rglru,
                wOut: try vals(p + "wOut"),
                bOut: try vals(p + "bOut"),
                localAttn: try loadLocalAttn(p),
                norm2Gamma: try vals(p + "norm2Gamma"),
                norm2Beta: try vals(p + "norm2Beta"),
                wMLPUp: try vals(p + "wMLPUp"),
                bMLPUp: try vals(p + "bMLPUp"),
                wMLPDown: try vals(p + "wMLPDown"),
                bMLPDown: try vals(p + "bMLPDown")
            ))
        }

        let fus = GatedFusionWeights(
            dim: dim,
            wSwiGLUGate: try vals("fusion.wSwiGLUGate"),
            bSwiGLUGate: try vals("fusion.bSwiGLUGate"),
            wSwiGLUUp: try vals("fusion.wSwiGLUUp"),
            bSwiGLUUp: try vals("fusion.bSwiGLUUp"),
            normGamma: try vals("fusion.normGamma"),
            normBeta: try vals("fusion.normBeta")
        )

        let pool = AttentionPoolingWeights(
            dim: dim,
            latentDim: latentDim,
            numQueries: numQueries,
            numHeads: numHeads,
            qPool: try vals("pooling.qPool"),
            wK: try vals("pooling.wK"),
            bK: try vals("pooling.bK"),
            wV: try vals("pooling.wV"),
            bV: try vals("pooling.bV"),
            wO: try vals("pooling.wO"),
            bO: try vals("pooling.bO"),
            wProj: try vals("pooling.wProj"),
            bProj: try vals("pooling.bProj"),
            normGamma: try vals("pooling.normGamma"),
            normBeta: try vals("pooling.normBeta"),
            l2Normalize: l2Norm
        )

        var decLayers: [ConditionalHawkLayerWeights] = []
        for i in 0..<decoderLayers {
            let p = "decoder.layers.\(i)."
            let rglru = RGLRUWeights(
                dim: dim,
                wA: try vals(p + "rglru.wA"),
                bA: try vals(p + "rglru.bA"),
                wX: try vals(p + "rglru.wX"),
                bX: try vals(p + "rglru.bX"),
                lambda: try vals(p + "rglru.lambda")
            )
            let hb = HawkBlockWeights(
                dim: dim,
                mlpDim: mlpDim,
                norm1Gamma: try vals(p + "norm1Gamma"),
                norm1Beta: try vals(p + "norm1Beta"),
                wGate: try vals(p + "wGate"),
                bGate: try vals(p + "bGate"),
                rglru: rglru,
                wOut: try vals(p + "wOut"),
                bOut: try vals(p + "bOut"),
                localAttn: try loadLocalAttn(p),
                norm2Gamma: try vals(p + "norm2Gamma"),
                norm2Beta: try vals(p + "norm2Beta"),
                wMLPUp: try vals(p + "wMLPUp"),
                bMLPUp: try vals(p + "bMLPUp"),
                wMLPDown: try vals(p + "wMLPDown"),
                bMLPDown: try vals(p + "bMLPDown")
            )
            decLayers.append(ConditionalHawkLayerWeights(
                dim: dim,
                latentDim: latentDim,
                block: hb,
                wH0: try vals(p + "wH0"),
                bH0: try vals(p + "bH0"),
                wAdaGamma: try vals(p + "wAdaGamma"),
                bAdaGamma: try vals(p + "bAdaGamma"),
                wAdaBeta: try vals(p + "wAdaBeta"),
                bAdaBeta: try vals(p + "bAdaBeta")
            ))
        }

        let dec = HawkDecoderWeights(
            dim: dim,
            latentDim: latentDim,
            vocabSize: vocabSize,
            wCross: try vals("decoder.wCross"),
            bCross: try vals("decoder.bCross"),
            layers: decLayers,
            finalNormGamma: try vals("decoder.finalNormGamma"),
            finalNormBeta: try vals("decoder.finalNormBeta"),
            wReadout: try vals("decoder.wReadout"),
            bReadout: try vals("decoder.bReadout")
        )

        var fmWeights: FlowMatchingWeights? = nil
        if tensors["cfm.wAlign"] != nil {
            let fmH = Int(metadata["cfmHiddenDim"] ?? "") ?? FlowMatchingWeights.subbandTokenDim
            let fmPairs = Int(metadata["cfmNumPairs"] ?? "") ?? 0
            let cBank = (fmPairs > 0 && tensors["cfm.condBank"] != nil) ? (try vals("cfm.condBank")) : []
            let sBank = (fmPairs > 0 && tensors["cfm.summaryBank"] != nil) ? (try vals("cfm.summaryBank")) : []
            fmWeights = FlowMatchingWeights(
                latentDim: latentDim,
                hiddenDim: fmH,
                numPairs: fmPairs,
                wAlign: try vals("cfm.wAlign"),
                bAlign: try vals("cfm.bAlign"),
                wTime: try vals("cfm.wTime"),
                bTime: try vals("cfm.bTime"),
                wVelIn: try vals("cfm.wVelIn"),
                bVelIn: try vals("cfm.bVelIn"),
                wVelGate: try vals("cfm.wVelGate"),
                bVelGate: try vals("cfm.bVelGate"),
                wVelOut: try vals("cfm.wVelOut"),
                bVelOut: try vals("cfm.bVelOut"),
                condBank: cBank,
                summaryBank: sBank
            )
        }

        return SWTHawkWeights(
            vocabSize: vocabSize,
            dim: dim,
            latentDim: latentDim,
            mlpDim: mlpDim,
            serializedVocabulary: vocabStr,
            subbandClassifier: classifier,
            tokenEmbedding: try vals("tokenEmbedding"),
            swtBranch: swt,
            hawkBranch: encHawk,
            fusion: fus,
            pooling: pool,
            decoder: dec,
            flowMatching: fmWeights
        )
    }
}

/// CPU inference engine for the SWT-Griffin Text Autoencoder & Conditional Flow Matcher (`Section 2 & 3`).
public final class SWTHawkAutoencoder: @unchecked Sendable {
    public let weights: SWTHawkWeights

    public init(weights: SWTHawkWeights) {
        self.weights = weights
    }

    /// Embeds token ID sequences `[B, T]` with sinusoidal positional encoding into `[B, T, D]`.
    public func embedWithPositionalEncoding(tokens: [Int], batch: Int, time: Int) -> [Float] {
        let dim = weights.dim
        let vocabSize = weights.vocabSize
        let rows = batch * time
        precondition(tokens.count == rows)

        let pe = HawkConditionalDecoder.positionalEncoding(time: time, dim: dim)
        var embedded = [Float](repeating: 0.0, count: rows * dim)
        for b in 0..<batch {
            for t in 0..<time {
                let rowIdx = b * time + t
                let tokId = max(0, min(vocabSize - 1, tokens[rowIdx]))
                let embOff = tokId * dim
                let peOff = t * dim
                let dstOff = rowIdx * dim
                for c in 0..<dim {
                    embedded[dstOff + c] = weights.tokenEmbedding[embOff + c] + pe[peOff + c]
                }
            }
        }
        return embedded
    }

    /// Runs Branch B (`Griffin Temporal Branch: RG-LRU + Local Sliding-Window Attention W=32 + Gated MLP`) over `embedded: [B, T, D]`.
    public func forwardTemporalBranch(embedded: [Float], batch: Int, time: Int) -> (tFeat: [Float], hLast: [Float]) {
        var tFeat = embedded
        var hLast = [Float](repeating: 0.0, count: batch * weights.dim)
        for hb in weights.hawkBranch {
            let res = HawkBlock.forward(x: tFeat, weights: hb, h0: nil, batch: batch, time: time)
            tFeat = res.out
            hLast = res.hLast
        }
        return (tFeat, hLast)
    }

    private func mixTemporalFeatures(
        tFeatA: [Float],
        hLastA: [Float],
        embB: [Float],
        batch: Int,
        time: Int,
        beta: Float
    ) -> ([Float], [Float]) {
        let (tFeatB, hLastB) = forwardTemporalBranch(embedded: embB, batch: batch, time: time)
        let alpha = 1.0 - beta
        var mixedT = [Float](repeating: 0.0, count: tFeatA.count)
        for i in 0..<tFeatA.count {
            mixedT[i] = alpha * tFeatA[i] + beta * tFeatB[i]
        }
        var mixedH = [Float](repeating: 0.0, count: hLastA.count)
        for i in 0..<hLastA.count {
            mixedH[i] = alpha * hLastA[i] + beta * hLastB[i]
        }
        return (mixedT, mixedH)
    }

    /// Encodes a batch of token ID sequences `[B, T]` into subband-split principal component vectors `z = [z_LL(512); z_HH(512)] ∈ R^(B × 1024)`.
    public func encode(tokens: [Int], batch: Int, time: Int) -> [Float] {
        let embedded = embedWithPositionalEncoding(tokens: tokens, batch: batch, time: time)
        let fSWT = SWTFrequencyBranch.forward(
            x: embedded, weights: weights.swtBranch, batch: batch, time: time
        )
        let (tFeat, hLast) = forwardTemporalBranch(embedded: embedded, batch: batch, time: time)
        let uFused = GatedTimeFrequencyFusion.forward(
            fSWT: fSWT, tFeat: tFeat, weights: weights.fusion, batch: batch, time: time
        )
        let zNeural = MultiScaleAttentionPooling.forward(
            uFused: uFused, hLast: hLast, weights: weights.pooling, batch: batch, time: time
        )

        guard weights.subbandClassifier.isConfigured && weights.latentDim >= 256 else {
            return zNeural
        }

        let latentDim = weights.latentDim
        let half = latentDim / 2
        var zOut = [Float](repeating: 0.0, count: batch * latentDim)
        for b in 0..<batch {
            let tokSlice = Array(tokens[(b * time)..<((b + 1) * time)])
            let split = SubbandClauseSplitter.split(tokens: tokSlice, classifier: weights.subbandClassifier)

            // Compute neural LL features strictly on the LL main-clause stream so z_LL(X_full) == z_LL(X_base)
            var llPadded = Array(split.llTokens.prefix(time))
            if llPadded.count < time {
                llPadded.append(contentsOf: repeatElement(TextVocabulary.padId, count: time - llPadded.count))
            }
            let llEmb = embedWithPositionalEncoding(tokens: llPadded, batch: 1, time: time)
            let llSWT = SWTFrequencyBranch.forward(x: llEmb, weights: weights.swtBranch, batch: 1, time: time)
            let (llTFeat, llHLast) = forwardTemporalBranch(embedded: llEmb, batch: 1, time: time)
            let llUFused = GatedTimeFrequencyFusion.forward(fSWT: llSWT, tFeat: llTFeat, weights: weights.fusion, batch: 1, time: time)
            let llNeuralFull = MultiScaleAttentionPooling.forward(uFused: llUFused, hLast: llHLast, weights: weights.pooling, batch: 1, time: time)

            let neuralLL = Array(llNeuralFull[0..<half])
            let neuralHH = Array(zNeural[(b * latentDim + half)..<((b + 1) * latentDim)])
            let zCal = SubbandClauseSplitter.embedSubbandLattice(
                split: split,
                neuralLL: neuralLL,
                neuralHH: neuralHH,
                gamma: weights.pooling.normGamma,
                beta: weights.pooling.normBeta,
                latentDim: latentDim
            )
            for c in 0..<latentDim {
                zOut[b * latentDim + c] = zCal[c]
            }
        }
        return zOut
    }

    /// Wavelet Subband Cross-Synthesis (`Test 2`):
    /// Combines low-frequency main-principle skeleton `z_LL` (`CA2` / `F_approx`) from `tokensA` with
    /// high-frequency exception/proviso detail `z_HH` (`CD1, CD2` / `F_detail`) from `tokensB`:
    ///   `z_swap = [z_LL(A <- B_β) ; z_HH(B)] ∈ R^(B × 1024)`.
    public func encodeWaveletCrossSynthesis(
        tokensA: [Int],
        tokensB: [Int],
        batch: Int,
        time: Int,
        temporalMixBeta: Float = 0.0
    ) -> [Float] {
        precondition(tokensA.count == batch * time && tokensB.count == batch * time)
        let embA = embedWithPositionalEncoding(tokens: tokensA, batch: batch, time: time)
        let embB = embedWithPositionalEncoding(tokens: tokensB, batch: batch, time: time)

        let (fApproxA, _) = SWTFrequencyBranch.extractSubbandFeatures(
            x: embA, weights: weights.swtBranch, batch: batch, time: time
        )
        let (_, fDetailB) = SWTFrequencyBranch.extractSubbandFeatures(
            x: embB, weights: weights.swtBranch, batch: batch, time: time
        )
        let fSWTSwap = SWTFrequencyBranch.combineSubbandFeatures(
            fApprox: fApproxA, fDetail: fDetailB, weights: weights.swtBranch, batch: batch, time: time
        )

        let (tFeatA, hLastA) = forwardTemporalBranch(embedded: embA, batch: batch, time: time)
        let (tFeat, hLast): ([Float], [Float]) = if temporalMixBeta == 0.0 {
            (tFeatA, hLastA)
        } else {
            mixTemporalFeatures(tFeatA: tFeatA, hLastA: hLastA, embB: embB, batch: batch, time: time, beta: temporalMixBeta)
        }

        let uFused = GatedTimeFrequencyFusion.forward(
            fSWT: fSWTSwap, tFeat: tFeat, weights: weights.fusion, batch: batch, time: time
        )
        let zNeural = MultiScaleAttentionPooling.forward(
            uFused: uFused, hLast: hLast, weights: weights.pooling, batch: batch, time: time
        )

        guard weights.subbandClassifier.isConfigured && weights.latentDim >= 256 else {
            return zNeural
        }

        let latentDim = weights.latentDim
        let half = latentDim / 2
        var zOut = [Float](repeating: 0.0, count: batch * latentDim)

        for b in 0..<batch {
            let sliceA = Array(tokensA[(b * time)..<((b + 1) * time)])
            let sliceB = Array(tokensB[(b * time)..<((b + 1) * time)])
            let splitA = SubbandClauseSplitter.split(tokens: sliceA, classifier: weights.subbandClassifier)
            let splitB = SubbandClauseSplitter.split(tokens: sliceB, classifier: weights.subbandClassifier)

            // Combine main clauses (LL) of A with optional trailing temporalMixBeta clauses from B
            let keepA = max(1, Int((Float(splitA.llSentences.count) * (1.0 - temporalMixBeta)).rounded()))
            let takeB = max(1, Int((Float(splitB.llSentences.count) * temporalMixBeta).rounded()))
            let hybridLLSentences: [[Int]] = if temporalMixBeta <= 0.01 || splitA.llSentences.count <= 1 || splitB.llSentences.isEmpty {
                splitA.llSentences
            } else {
                Array(splitA.llSentences.prefix(keepA)) + Array(splitB.llSentences.suffix(takeB))
            }
            let hybridLLTokens = hybridLLSentences.flatMap { $0 }
            let hybridSplit = SubbandClauseSplitter.SplitResult(
                llTokens: hybridLLTokens,
                llSentences: hybridLLSentences,
                provisos: splitB.provisos,
                hasVocabulary: true,
                polarity: splitB.polarity,
                hhCarrierSeedTokens: splitB.hhCarrierSeedTokens
            )

            let neuralLL = Array(zNeural[(b * latentDim)..<(b * latentDim + half)])
            let neuralHH = Array(zNeural[(b * latentDim + half)..<((b + 1) * latentDim)])
            let zCal = SubbandClauseSplitter.embedSubbandLattice(
                split: hybridSplit,
                neuralLL: neuralLL,
                neuralHH: neuralHH,
                gamma: weights.pooling.normGamma,
                beta: weights.pooling.normBeta,
                latentDim: latentDim
            )
            for c in 0..<latentDim {
                zOut[b * latentDim + c] = zCal[c]
            }
        }
        return zOut
    }

    /// Applies Subband-Split `LayerNorm` (`z_LL` `[0..<512]` and `z_HH` `[512..<1024]`) to `z ∈ R^(rows × 1024)`.
    public func layerNormLatent(_ z: [Float], rows: Int = 1) -> [Float] {
        return SubbandClauseSplitter.subbandLayerNorm(
            z: z,
            gamma: weights.pooling.normGamma,
            beta: weights.pooling.normBeta,
            rows: rows,
            latentDim: weights.latentDim,
            preserveRelativeHHScale: true
        )
    }

    /// Teacher-forced forward pass returning `(z: [B, 1024], logits: [B, T, V])`.
    public func forward(
        tokens: [Int],
        batch: Int,
        time: Int,
        bosToken: Int = TextVocabulary.bosId
    ) -> (z: [Float], logits: [Float]) {
        let z = encode(tokens: tokens, batch: batch, time: time)

        var prevTokens = [Int](repeating: bosToken, count: batch * time)
        for b in 0..<batch {
            let base = b * time
            prevTokens[base] = bosToken
            for t in 1..<time {
                prevTokens[base + t] = tokens[base + t - 1]
            }
        }

        let logits = HawkConditionalDecoder.forward(
            z: z,
            prevTokens: prevTokens,
            tokenEmbedding: weights.tokenEmbedding,
            weights: weights.decoder,
            batch: batch,
            time: time
        )
        return (z, logits)
    }

    /// Reconstructs token sequences `[B][T]` autoregressively from `z ∈ R^(B × 1024)` alone.
    public func reconstructGreedy(
        z: [Float],
        batch: Int,
        length: Int,
        bosToken: Int = TextVocabulary.bosId
    ) -> [[Int]] {
        return HawkConditionalDecoder.decodeGreedy(
            z: z,
            tokenEmbedding: weights.tokenEmbedding,
            weights: weights.decoder,
            batch: batch,
            length: length,
            bosToken: bosToken,
            subbandClassifier: weights.subbandClassifier
        )
    }

    /// Stage 4 & Phase 3 Task A Conditional Flow Matching (`CFM`) Multi-Resolution Summarization:
    /// Encodes `tokens` into `z_full = [z_LL ; z_HH] ∈ R^(B × 1024)`, scales the high-frequency subband by `subbandGamma ∈ [0.0, 1.0]`
    /// to form `c(γ) = [z_LL ; γ · z_HH]`, integrates the 50-step Midpoint ODE with CFG scale `guidanceScale`
    /// to obtain `z_hat_summary ∈ R^(B × 1024)`, and decodes via the frozen Griffin Conditional Decoder.
    public func summarizeWithFlowMatching(
        tokens: [Int],
        batch: Int,
        time: Int,
        subbandGamma: Float = 1.0,
        odeSteps: Int = 50,
        guidanceScale: Float = 1.5
    ) -> (zFull: [Float], zSummary: [Float], summaryTokens: [[Int]], trajectoryNorms: [Float]) {
        let zFull = encode(tokens: tokens, batch: batch, time: time)
        let zCondition = if abs(subbandGamma - 1.0) <= 1e-6 {
            zFull
        } else {
            ConditionalFlowMatcher.scaleSubbandCondition(
                z: zFull, gamma: subbandGamma, batch: batch, latentDim: weights.latentDim
            )
        }
        let (zSummary, trajNorms) = ConditionalFlowMatcher.sampleMidpointODE(
            condition: zCondition,
            weights: weights.flowMatching,
            batch: batch,
            steps: odeSteps,
            guidanceScale: guidanceScale
        )
        let decoded = reconstructGreedy(z: zSummary, batch: batch, length: time)
        return (zFull, zSummary, decoded, trajNorms)
    }

    /// Phase 3 Task B: Subband Differential Contract Auditing, Audit Report CFM & Latent Repair.
    /// Computes `Δz = z^S - z^C = [Δz_LL ; Δz_HH]`, evaluates `(D_LL, D_HH, R_HH, S_HH)` and deterministic `AuditStatus`,
    /// transports `c_audit = LayerNorm(W_audit [z^S ; z^C ; Δz])` via 50-step Midpoint ODE to `z_hat_audit`,
    /// and repairs the contract clause via `z_repaired = z^C + Δz`.
    public func auditContractWithFlowMatching(
        statuteTokens: [Int],
        contractTokens: [Int],
        time: Int,
        tauLL: Float = 5.0,
        tauHH: Float = 5.0,
        odeSteps: Int = 50,
        guidanceScale: Float = 1.5
    ) -> (
        zStatute: [Float],
        zContract: [Float],
        metrics: SubbandAuditMetrics,
        cAudit: [Float],
        zAudit: [Float],
        auditTokens: [Int],
        repairedTokens: [Int],
        trajectoryNorms: [Float]
    ) {
        precondition(statuteTokens.count == time && contractTokens.count == time)
        let zStatute = encode(tokens: statuteTokens, batch: 1, time: time)
        let zContract = encode(tokens: contractTokens, batch: 1, time: time)
        let metrics = ConditionalFlowMatcher.computeSubbandAudit(
            zStatute: zStatute,
            zContract: zContract,
            tauLL: tauLL,
            tauHH: tauHH
        )
        let cAudit = ConditionalFlowMatcher.computeAuditCondition(
            zStatute: zStatute,
            zContract: zContract,
            deltaZ: metrics.deltaZ,
            weights: weights.flowMatching
        )
        let (zAudit, trajNorms) = ConditionalFlowMatcher.sampleMidpointODE(
            condition: cAudit,
            weights: weights.flowMatching,
            batch: 1,
            steps: odeSteps,
            guidanceScale: guidanceScale
        )
        let auditDecoded = reconstructGreedy(z: zAudit, batch: 1, length: time)[0]
        let repairedDecoded = reconstructGreedy(z: metrics.zRepaired, batch: 1, length: time)[0]
        return (zStatute, zContract, metrics, cAudit, zAudit, auditDecoded, repairedDecoded, trajNorms)
    }

    // MARK: - CPU Math Primitives

    /// `c = a · b (+ beta · c)`, row-major: `a [m, k]`, `b [k, n]`, `c [m, n]`.
    public static func gemm(_ a: [Float], _ b: [Float], _ c: inout [Float], m: Int, n: Int, k: Int, beta: Float = 0.0) {
        a.withUnsafeBufferPointer { ap in
            b.withUnsafeBufferPointer { bp in
                c.withUnsafeMutableBufferPointer { cp in
                    gemmRaw(ap.baseAddress!, k, bp.baseAddress!, n, cp.baseAddress!, n, m: m, n: n, k: k, beta: beta)
                }
            }
        }
    }

    public static func gemmRaw(
        _ a: UnsafePointer<Float>, _ lda: Int,
        _ b: UnsafePointer<Float>, _ ldb: Int,
        _ c: UnsafeMutablePointer<Float>, _ ldc: Int,
        m: Int, n: Int, k: Int, beta: Float
    ) {
        #if canImport(Accelerate)
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            Int32(m), Int32(n), Int32(k),
            1.0, a, Int32(lda), b, Int32(ldb), beta, c, Int32(ldc)
        )
        #else
        let nSimd = n - (n % 8)
        var row = 0
        while row < m {
            let cRow = c + row * ldc
            let aRow = a + row * lda
            if beta == 0.0 {
                var col = 0
                while col < nSimd {
                    store(SIMD8<Float>(repeating: 0.0), cRow + col)
                    col += 8
                }
                while col < n {
                    cRow[col] = 0.0
                    col += 1
                }
            } else if beta != 1.0 {
                let bVec = SIMD8<Float>(repeating: beta)
                var col = 0
                while col < nSimd {
                    store(load(cRow + col) * bVec, cRow + col)
                    col += 8
                }
                while col < n {
                    cRow[col] *= beta
                    col += 1
                }
            }
            var p = 0
            while p < k {
                let aVal = aRow[p]
                if aVal != 0.0 {
                    let aVec = SIMD8<Float>(repeating: aVal)
                    let bRow = b + p * ldb
                    var col = 0
                    while col < nSimd {
                        store(load(cRow + col) + aVec * load(bRow + col), cRow + col)
                        col += 8
                    }
                    while col < n {
                        cRow[col] += aVal * bRow[col]
                        col += 1
                    }
                }
                p += 1
            }
            row += 1
        }
        #endif
    }

    /// Linear projection `y = x · w + b` (`x: [rows, inDim]`, `w: [inDim, outDim]`, `b: [outDim]`).
    public static func linear(
        x: [Float],
        w: [Float],
        b: [Float]?,
        rows: Int,
        inDim: Int,
        outDim: Int
    ) -> [Float] {
        var y = [Float](repeating: 0.0, count: rows * outDim)
        if let bias = b {
            for r in 0..<rows {
                let base = r * outDim
                for c in 0..<outDim {
                    y[base + c] = bias[c]
                }
            }
            gemm(x, w, &y, m: rows, n: outDim, k: inDim, beta: 1.0)
        } else {
            gemm(x, w, &y, m: rows, n: outDim, k: inDim, beta: 0.0)
        }
        return y
    }

    /// Feature-wise LayerNorm over `x: [rows, dim]` with affine parameters `gamma: [dim]` and `beta: [dim]`.
    public static func layerNorm(
        x: [Float],
        gamma: [Float],
        beta: [Float],
        rows: Int,
        dim: Int,
        eps: Float = SWTHawkShape.layerNormEps
    ) -> [Float] {
        var y = [Float](repeating: 0.0, count: rows * dim)
        let invD = 1.0 / Float(max(1, dim))
        for r in 0..<rows {
            let base = r * dim
            var sum: Float = 0.0
            for c in 0..<dim {
                sum += x[base + c]
            }
            let mean = sum * invD
            var varSum: Float = 0.0
            for c in 0..<dim {
                let d = x[base + c] - mean
                varSum += d * d
            }
            let invStd = 1.0 / sqrtf(varSum * invD + eps)
            for c in 0..<dim {
                y[base + c] = (x[base + c] - mean) * invStd * gamma[c] + beta[c]
            }
        }
        return y
    }

    /// Element-wise SiLU `x * σ(x)`.
    public static func silu(_ x: [Float]) -> [Float] {
        var out = [Float](repeating: 0.0, count: x.count)
        for i in 0..<x.count {
            let v = x[i]
            out[i] = v / (1.0 + expf(-v))
        }
        return out
    }

    @inline(__always)
    static func load(_ p: UnsafePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func load(_ p: UnsafeMutablePointer<Float>) -> SIMD8<Float> {
        return UnsafeRawPointer(p).loadUnaligned(as: SIMD8<Float>.self)
    }

    @inline(__always)
    static func store(_ v: SIMD8<Float>, _ p: UnsafeMutablePointer<Float>) {
        UnsafeMutableRawPointer(p).storeBytes(of: v, as: SIMD8<Float>.self)
    }
}
