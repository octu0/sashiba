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
    public static let latentDim: Int = 512
    public static let relationalLatentDim: Int = 128
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
    public var subbandClassifier: SubbandTokenClassifier

    public let tokenEmbedding: [Float] // [vocabSize, D]
    public let swtBranch: SWTBranchWeights
    public let hawkBranch: [HawkBlockWeights]
    public let fusion: GatedFusionWeights
    public let pooling: AttentionPoolingWeights
    public let decoder: HawkDecoderWeights
    public let flowMatching: FlowMatchingWeights
    public var relational: DualStreamRelationalWeights

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
        flowMatching: FlowMatchingWeights? = nil,
        relational: DualStreamRelationalWeights? = nil
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
        if let rel = relational {
            self.relational = rel
        } else {
            var relRng = SplitMix64(seed: 1777)
            self.relational = DualStreamRelationalWeights.makeRandom(
                vocabSize: vocabSize, dim: dim, latentDim: SWTHawkShape.relationalLatentDim, rng: &relRng
            )
        }
    }

    public var parameterCount: Int {
        return tokenEmbedding.count + swtBranch.parameterCount + fusion.parameterCount + pooling.parameterCount + decoder.parameterCount + flowMatching.parameterCount + relational.parameterCount + hawkBranch.reduce(0) { $0 + $1.parameterCount }
    }

    public static func makeRandom(
        vocabSize: Int,
        dim: Int = SWTHawkShape.defaultDim,
        latentDim: Int = SWTHawkShape.latentDim,
        relationalLatentDim: Int = SWTHawkShape.relationalLatentDim,
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
        let rel = DualStreamRelationalWeights.makeRandom(
            vocabSize: vocabSize, dim: dim, latentDim: relationalLatentDim, rng: &rng
        )

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
            flowMatching: fm,
            relational: rel
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

        let cohDim = DualStreamRelationalWeights.coherenceDim
        tensors += [
            ("rel.alphaRel", T(shape: [vocabSize], values: relational.alphaRel)),
            ("rel.eEcho", T(shape: [dim], values: relational.eEcho)),
            ("rel.eTopic", T(shape: [dim], values: relational.eTopic)),
            ("rel.convRelLL1.w", T(shape: [3, dim, dim], values: relational.convRelLL1.w)),
            ("rel.convRelLL1.b", T(shape: [dim], values: relational.convRelLL1.b)),
            ("rel.convRelLL2.w", T(shape: [3, dim, dim], values: relational.convRelLL2.w)),
            ("rel.convRelLL2.b", T(shape: [dim], values: relational.convRelLL2.b)),
            ("rel.convRelHH1.w", T(shape: [3, 2 * dim, dim], values: relational.convRelHH1.w)),
            ("rel.convRelHH1.b", T(shape: [dim], values: relational.convRelHH1.b)),
            ("rel.convRelHH2.w", T(shape: [3, dim, dim], values: relational.convRelHH2.w)),
            ("rel.convRelHH2.b", T(shape: [dim], values: relational.convRelHH2.b)),
            ("rel.wPoolLL", T(shape: [dim, 2], values: relational.wPoolLL)),
            ("rel.wPoolHH", T(shape: [dim, 2], values: relational.wPoolHH)),
            ("rel.wQLL", T(shape: [dim, dim], values: relational.wQLL)),
            ("rel.bQLL", T(shape: [dim], values: relational.bQLL)),
            ("rel.wKLL", T(shape: [dim, dim], values: relational.wKLL)),
            ("rel.bKLL", T(shape: [dim], values: relational.bKLL)),
            ("rel.wVLL", T(shape: [dim, dim], values: relational.wVLL)),
            ("rel.bVLL", T(shape: [dim], values: relational.bVLL)),
            ("rel.wCohLL", T(shape: [cohDim, dim], values: relational.wCohLL)),
            ("rel.bCohLL", T(shape: [dim], values: relational.bCohLL)),
            ("rel.wQHH", T(shape: [dim, dim], values: relational.wQHH)),
            ("rel.bQHH", T(shape: [dim], values: relational.bQHH)),
            ("rel.wKHH", T(shape: [dim, dim], values: relational.wKHH)),
            ("rel.bKHH", T(shape: [dim], values: relational.bKHH)),
            ("rel.wVHH", T(shape: [dim, dim], values: relational.wVHH)),
            ("rel.bVHH", T(shape: [dim], values: relational.bVHH)),
            ("rel.wCohHH", T(shape: [cohDim, dim], values: relational.wCohHH)),
            ("rel.bCohHH", T(shape: [dim], values: relational.bCohHH)),
            ("rel.wProjLL", T(shape: [5 * dim, relational.latentDim / 2], values: relational.wProjLL)),
            ("rel.bProjLL", T(shape: [relational.latentDim / 2], values: relational.bProjLL)),
            ("rel.normLLGamma", T(shape: [relational.latentDim / 2], values: relational.normLLGamma)),
            ("rel.normLLBeta", T(shape: [relational.latentDim / 2], values: relational.normLLBeta)),
            ("rel.wProjHH", T(shape: [5 * dim, relational.latentDim / 2], values: relational.wProjHH)),
            ("rel.bProjHH", T(shape: [relational.latentDim / 2], values: relational.bProjHH)),
            ("rel.normHHGamma", T(shape: [relational.latentDim / 2], values: relational.normHHGamma)),
            ("rel.normHHBeta", T(shape: [relational.latentDim / 2], values: relational.normHHBeta)),
            ("rel.wContent", T(shape: [2 * dim, relational.latentDim / 2], values: relational.wContent)),
            ("rel.bContent", T(shape: [relational.latentDim / 2], values: relational.bContent)),
            ("rel.wArchT4", T(shape: [relational.latentDim, 6], values: relational.wArchT4)),
            ("rel.bArchT4", T(shape: [6], values: relational.bArchT4)),
            ("rel.wTopoT4", T(shape: [relational.latentDim, 5], values: relational.wTopoT4)),
            ("rel.bTopoT4", T(shape: [5], values: relational.bTopoT4)),
            ("rel.wArchT2", T(shape: [relational.latentDim, 6], values: relational.wArchT2)),
            ("rel.bArchT2", T(shape: [6], values: relational.bArchT2)),
            ("rel.wTopoT2", T(shape: [relational.latentDim, 5], values: relational.wTopoT2)),
            ("rel.bTopoT2", T(shape: [5], values: relational.bTopoT2))
        ]

        let metadata: [String: String] = [
            "vocabSize": "\(vocabSize)",
            "dim": "\(dim)",
            "latentDim": "\(latentDim)",
            "relationalLatentDim": "\(relational.latentDim)",
            "mlpDim": "\(mlpDim)",
            "encoderLayers": "\(hawkBranch.count)",
            "decoderLayers": "\(decoder.layers.count)",
            "numQueries": "\(pooling.numQueries)",
            "numHeads": "\(pooling.numHeads)",
            "l2Normalize": pooling.l2Normalize ? "1" : "0",
            "cfmHiddenDim": "\(fmH)",
            "cfmNumPairs": "\(flowMatching.numPairs)",
            "vocabulary": serializedVocabulary,
            "subbandClassifier": subbandClassifier.serialize(),
            "calibratedHeads": relational.calibratedHeads.serialize(),
            "auditTelemetry": relational.auditTelemetry.serialize()
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
            let fmH = Int(metadata["cfmHiddenDim"] ?? "") ?? max(8, latentDim / FlowMatchingWeights.numSubbandTokens)
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

        var relWeights: DualStreamRelationalWeights? = nil
        if tensors["rel.wQLL"] != nil {
            let calHeads = RelationalCalibratedHeads.deserialize(metadata["calibratedHeads"] ?? "")
            let telemetry = RelationalAuditTelemetry.deserialize(metadata["auditTelemetry"] ?? "")
            let defSeq = DualStreamRelationalWeights.makeDefaultSequenceFilters(dim: dim)
            let cLL1: Conv1D3Weights = if tensors["rel.convRelLL1.w"] != nil, tensors["rel.convRelLL1.b"] != nil,
                let w = try? vals("rel.convRelLL1.w"), w.count == 3 * dim * dim,
                let b = try? vals("rel.convRelLL1.b"), b.count == dim
            {
                Conv1D3Weights(inDim: dim, outDim: dim, w: w, b: b)
            } else {
                defSeq.convRelLL1
            }
            let cLL2: Conv1D3Weights = if tensors["rel.convRelLL2.w"] != nil, tensors["rel.convRelLL2.b"] != nil,
                let w = try? vals("rel.convRelLL2.w"), w.count == 3 * dim * dim,
                let b = try? vals("rel.convRelLL2.b"), b.count == dim
            {
                Conv1D3Weights(inDim: dim, outDim: dim, w: w, b: b)
            } else {
                defSeq.convRelLL2
            }
            let cHH1: Conv1D3Weights = if tensors["rel.convRelHH1.w"] != nil, tensors["rel.convRelHH1.b"] != nil,
                let w = try? vals("rel.convRelHH1.w"), w.count == 3 * 2 * dim * dim,
                let b = try? vals("rel.convRelHH1.b"), b.count == dim
            {
                Conv1D3Weights(inDim: 2 * dim, outDim: dim, w: w, b: b)
            } else {
                defSeq.convRelHH1
            }
            let cHH2: Conv1D3Weights = if tensors["rel.convRelHH2.w"] != nil, tensors["rel.convRelHH2.b"] != nil,
                let w = try? vals("rel.convRelHH2.w"), w.count == 3 * dim * dim,
                let b = try? vals("rel.convRelHH2.b"), b.count == dim
            {
                Conv1D3Weights(inDim: dim, outDim: dim, w: w, b: b)
            } else {
                defSeq.convRelHH2
            }
            let wPoolLL: [Float] = if tensors["rel.wPoolLL"] != nil,
                let w = try? vals("rel.wPoolLL"), w.count == dim * 2
            {
                w
            } else {
                defSeq.wPoolLL
            }
            let wPoolHH: [Float] = if tensors["rel.wPoolHH"] != nil,
                let w = try? vals("rel.wPoolHH"), w.count == dim * 2
            {
                w
            } else {
                defSeq.wPoolHH
            }
            let relLatentDim = Int(metadata["relationalLatentDim"] ?? "")
                ?? ((try? vals("rel.bProjLL").count).map { $0 * 2 } ?? SWTHawkShape.relationalLatentDim)
            relWeights = DualStreamRelationalWeights(
                vocabSize: vocabSize,
                dim: dim,
                latentDim: relLatentDim,
                alphaRel: try vals("rel.alphaRel"),
                eEcho: try vals("rel.eEcho"),
                eTopic: try vals("rel.eTopic"),
                convRelLL1: cLL1,
                convRelLL2: cLL2,
                convRelHH1: cHH1,
                convRelHH2: cHH2,
                wPoolLL: wPoolLL,
                wPoolHH: wPoolHH,
                wQLL: try vals("rel.wQLL"),
                bQLL: try vals("rel.bQLL"),
                wKLL: try vals("rel.wKLL"),
                bKLL: try vals("rel.bKLL"),
                wVLL: try vals("rel.wVLL"),
                bVLL: try vals("rel.bVLL"),
                wCohLL: try vals("rel.wCohLL"),
                bCohLL: try vals("rel.bCohLL"),
                wQHH: try vals("rel.wQHH"),
                bQHH: try vals("rel.bQHH"),
                wKHH: try vals("rel.wKHH"),
                bKHH: try vals("rel.bKHH"),
                wVHH: try vals("rel.wVHH"),
                bVHH: try vals("rel.bVHH"),
                wCohHH: try vals("rel.wCohHH"),
                bCohHH: try vals("rel.bCohHH"),
                wProjLL: try vals("rel.wProjLL"),
                bProjLL: try vals("rel.bProjLL"),
                normLLGamma: try vals("rel.normLLGamma"),
                normLLBeta: try vals("rel.normLLBeta"),
                wProjHH: try vals("rel.wProjHH"),
                bProjHH: try vals("rel.bProjHH"),
                normHHGamma: try vals("rel.normHHGamma"),
                normHHBeta: try vals("rel.normHHBeta"),
                wContent: try vals("rel.wContent"),
                bContent: try vals("rel.bContent"),
                wArchT4: try vals("rel.wArchT4"),
                bArchT4: try vals("rel.bArchT4"),
                wTopoT4: try vals("rel.wTopoT4"),
                bTopoT4: try vals("rel.bTopoT4"),
                wArchT2: try vals("rel.wArchT2"),
                bArchT2: try vals("rel.bArchT2"),
                wTopoT2: try vals("rel.wTopoT2"),
                bTopoT2: try vals("rel.bTopoT2"),
                calibratedHeads: calHeads,
                auditTelemetry: telemetry
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
            flowMatching: fmWeights,
            relational: relWeights
        )
    }
}

/// CPU inference engine for the SWT-Griffin Text Autoencoder & Conditional Flow Matcher (`Section 2 & 3`).
public final class SWTHawkAutoencoder: @unchecked Sendable {
    public let weights: SWTHawkWeights
    public let decodedVocabulary: TextVocabulary
    public let hybridExtractor: RelationalHybridFeatureExtractor

    /// Optional pure-CPU Small Conditional Griffin LM for generating inner-psychology descriptions.
    public var psychologicalLM: PsychologicalStateLM? = nil

    /// Optional accelerated (MLX GPU) backend for `encode(tokens:batch:time:)`; when set, replaces the CPU encoder.
    public var latentEncoderBackend: (([Int], Int, Int) -> [Float])? = nil

    /// Optional accelerated (MLX GPU) backend for the Dual-Stream relational encoder returning `(zRel, zContent)`.
    public var relationalEncoderBackend: (([Int], [Int], Int, Int) -> (zRel: [Float], zContent: [Float]))? = nil

    public init(weights: SWTHawkWeights, psychologicalLM: PsychologicalStateLM? = nil) {
        self.weights = weights
        let vocab = TextVocabulary.deserialize(weights.serializedVocabulary)
        self.decodedVocabulary = vocab
        self.hybridExtractor = RelationalHybridFeatureExtractor(vocabulary: vocab)
        self.psychologicalLM = psychologicalLM
    }

    /// Loads and attaches a trained `PsychologicalStateLM` from a `.safetensors` file.
    public func loadPsychologicalStateLM(from url: URL) throws {
        let lmWeights = try PsychologicalStateLMWeights.load(from: url)
        self.psychologicalLM = PsychologicalStateLM(weights: lmWeights)
    }

    /// Evaluates `v6 / v6_hybrid` relational state and generates a natural inner-psychology description using the attached `PsychologicalStateLM`.
    public func inferRelationalWithPsychology(
        senderEarlyText: String? = nil,
        receiverEarlyText: String? = nil,
        senderCurrentText: String,
        receiverCurrentText: String,
        temporalContext: TemporalRelationalContext? = nil,
        temperature: Float = 0.0
    ) -> PsychologicalStateResult? {
        guard let lm = psychologicalLM else { return nil }
        return lm.describePsychology(
            relationalModel: self,
            senderEarlyText: senderEarlyText,
            receiverEarlyText: receiverEarlyText,
            senderCurrentText: senderCurrentText,
            receiverCurrentText: receiverCurrentText,
            temporalContext: temporalContext,
            temperature: temperature
        )
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

    private struct SendableFloatBuf: @unchecked Sendable {
        let base: UnsafeMutablePointer<Float>
    }

    /// Encodes a batch of token ID sequences `[B, T]` into subband-split principal component vectors `z = [z_LL(512); z_HH(512)] ∈ R^(B × 1024)`.
    public func encode(tokens: [Int], batch: Int, time: Int) -> [Float] {
        if let backend = latentEncoderBackend {
            return backend(tokens, batch, time)
        }
        if batch >= 32 {
            let chunkSize = 32
            let numChunks = (batch + chunkSize - 1) / chunkSize
            let lDim = weights.latentDim
            let totalFloats = batch * lDim
            let raw = UnsafeMutablePointer<Float>.allocate(capacity: totalFloats)
            raw.initialize(repeating: 0.0, count: totalFloats)
            defer {
                raw.deinitialize(count: totalFloats)
                raw.deallocate()
            }
            let dst = SendableFloatBuf(base: raw)
            DispatchQueue.concurrentPerform(iterations: numChunks) { cIdx in
                let b0 = cIdx * chunkSize
                let b1 = min(batch, b0 + chunkSize)
                let slice = Array(tokens[(b0 * time)..<(b1 * time)])
                let zPart = self.encodeCPU(tokens: slice, batch: b1 - b0, time: time)
                let off = b0 * lDim
                for k in 0..<zPart.count {
                    dst.base[off + k] = zPart[k]
                }
            }
            return Array(UnsafeBufferPointer(start: raw, count: totalFloats))
        }
        return encodeCPU(tokens: tokens, batch: batch, time: time)
    }

    /// CPU reference implementation of `encode(tokens:batch:time:)`.
    public func encodeCPU(tokens: [Int], batch: Int, time: Int) -> [Float] {
        let latentDim = weights.latentDim
        let half = latentDim / 2
        if weights.subbandClassifier.isConfigured && latentDim >= 256 && !weights.subbandClassifier.modulationRules.isEmpty {
            var zOut = [Float](repeating: 0.0, count: batch * latentDim)
            let dim = weights.dim
            let vocabSize = weights.vocabSize
            let embDim = min(44, min(dim, half))
            for b in 0..<batch {
                let tokSlice = Array(tokens[(b * time)..<((b + 1) * time)])
                let split = SubbandClauseSplitter.split(tokens: tokSlice, classifier: weights.subbandClassifier)
                var neuralLL = [Float](repeating: 0.0, count: half)
                if !split.llTokens.isEmpty {
                    let invN = 1.0 / Float(split.llTokens.count)
                    for tok in split.llTokens {
                        let safeTok = max(0, min(vocabSize - 1, tok))
                        let base = safeTok * dim
                        for c in 0..<embDim {
                            neuralLL[c] += weights.tokenEmbedding[base + c] * invN
                        }
                    }
                }
                var neuralHH = [Float](repeating: 0.0, count: half)
                let hhToks = split.provisos.flatMap(\.tokens)
                if !hhToks.isEmpty {
                    let invH = 1.0 / Float(hhToks.count)
                    for tok in hhToks {
                        let safeTok = max(0, min(vocabSize - 1, tok))
                        let base = safeTok * dim
                        for c in 0..<embDim {
                            neuralHH[c] += weights.tokenEmbedding[base + c] * invH
                        }
                    }
                }
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

        guard weights.subbandClassifier.isConfigured && latentDim >= 256 else {
            return zNeural
        }

        var splits: [SubbandClauseSplitter.SplitResult] = []
        splits.reserveCapacity(batch)
        var llPaddedBatch = [Int](repeating: TextVocabulary.padId, count: batch * time)
        for b in 0..<batch {
            let tokSlice = Array(tokens[(b * time)..<((b + 1) * time)])
            let split = SubbandClauseSplitter.split(tokens: tokSlice, classifier: weights.subbandClassifier)
            splits.append(split)
            let copyLen = min(time, split.llTokens.count)
            let rowOff = b * time
            for k in 0..<copyLen {
                llPaddedBatch[rowOff + k] = split.llTokens[k]
            }
        }

        // Batched neural LL forward pass across all B rows simultaneously
        let llEmb = embedWithPositionalEncoding(tokens: llPaddedBatch, batch: batch, time: time)
        let llSWT = SWTFrequencyBranch.forward(x: llEmb, weights: weights.swtBranch, batch: batch, time: time)
        let (llTFeat, llHLast) = forwardTemporalBranch(embedded: llEmb, batch: batch, time: time)
        let llUFused = GatedTimeFrequencyFusion.forward(fSWT: llSWT, tFeat: llTFeat, weights: weights.fusion, batch: batch, time: time)
        let llNeuralAll = MultiScaleAttentionPooling.forward(uFused: llUFused, hLast: llHLast, weights: weights.pooling, batch: batch, time: time)

        var zOut = [Float](repeating: 0.0, count: batch * latentDim)
        for b in 0..<batch {
            let bOff = b * latentDim
            let neuralLL = Array(llNeuralAll[bOff..<(bOff + half)])
            let neuralHH = Array(zNeural[(bOff + half)..<(bOff + latentDim)])
            let zCal = SubbandClauseSplitter.embedSubbandLattice(
                split: splits[b],
                neuralLL: neuralLL,
                neuralHH: neuralHH,
                gamma: weights.pooling.normGamma,
                beta: weights.pooling.normBeta,
                latentDim: latentDim
            )
            for c in 0..<latentDim {
                zOut[bOff + c] = zCal[c]
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

    private struct SendableIntBuf: @unchecked Sendable {
        let base: UnsafeMutablePointer<Int>
    }

    /// Reconstructs token sequences `[B][T]` autoregressively from `z ∈ R^(B × 1024)` alone.
    public func reconstructGreedy(
        z: [Float],
        batch: Int,
        length: Int,
        bosToken: Int = TextVocabulary.bosId
    ) -> [[Int]] {
        if batch >= 32 {
            let chunkSize = 32
            let numChunks = (batch + chunkSize - 1) / chunkSize
            let lDim = weights.latentDim
            let totalInts = batch * length
            let raw = UnsafeMutablePointer<Int>.allocate(capacity: totalInts)
            raw.initialize(repeating: TextVocabulary.padId, count: totalInts)
            defer {
                raw.deinitialize(count: totalInts)
                raw.deallocate()
            }
            let dst = SendableIntBuf(base: raw)
            DispatchQueue.concurrentPerform(iterations: numChunks) { cIdx in
                let b0 = cIdx * chunkSize
                let b1 = min(batch, b0 + chunkSize)
                let zSlice = Array(z[(b0 * lDim)..<(b1 * lDim)])
                let preds = HawkConditionalDecoder.decodeGreedy(
                    z: zSlice,
                    tokenEmbedding: self.weights.tokenEmbedding,
                    weights: self.weights.decoder,
                    batch: b1 - b0,
                    length: length,
                    bosToken: bosToken,
                    subbandClassifier: self.weights.subbandClassifier
                )
                for (localIdx, seq) in preds.enumerated() {
                    let rowOff = (b0 + localIdx) * length
                    let copyCount = min(length, seq.count)
                    for k in 0..<copyCount {
                        dst.base[rowOff + k] = seq[k]
                    }
                }
            }
            return (0..<batch).map { b in
                Array(UnsafeBufferPointer(start: raw + b * length, count: length))
            }
        }
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

    // MARK: - Phase 5.1: Pure-Neural Dual-Stream Relational Engine (Zero-Metadata Signature)

    /// Encodes `(senderTokenIDs: [Int], receiverTokenIDs: [Int])` into the pure-neural relational vector
    /// `z_rel = [z_LL^rel(512) ; z_HH^rel(512)] ∈ R^1024` (`Phase 5.1 Audit 1`).
    public func encodeRelational(
        senderTokenIDs: [Int],
        receiverTokenIDs: [Int]
    ) -> [Float] {
        let rawMax = max(senderTokenIDs.count, receiverTokenIDs.count)
        let time: Int = if relationalEncoderBackend != nil {
            max(DualStreamRelationalWeights.canonicalSeqLen, rawMax)
        } else {
            max(8, ((rawMax + 8 + 3) / 4) * 4)
        }
        return encodeRelationalBatch(
            senderTokens: DualStreamRelationalEngine.padOrTruncate(senderTokenIDs, to: time),
            receiverTokens: DualStreamRelationalEngine.padOrTruncate(receiverTokenIDs, to: time),
            batch: 1,
            time: time
        )
    }

    /// Batched pure-neural Dual-Stream Relational Encoder returning `z_rel ∈ R^(B × 1024)`.
    public func encodeRelationalBatch(
        senderTokens: [Int],
        receiverTokens: [Int],
        batch: Int,
        time: Int
    ) -> [Float] {
        return encodeRelationalBatchWithContent(senderTokens: senderTokens, receiverTokens: receiverTokens, batch: batch, time: time).zRel
    }

    /// Batched Dual-Stream encoder returning `(zRel: [B × 1024], zContent: [B × 512])` (GPU backend when available).
    public func encodeRelationalBatchWithContent(
        senderTokens: [Int],
        receiverTokens: [Int],
        batch: Int,
        time: Int
    ) -> (zRel: [Float], zContent: [Float]) {
        if let backend = relationalEncoderBackend {
            return backend(senderTokens, receiverTokens, batch, time)
        }
        let res = DualStreamRelationalEngine.forwardBatchCPU(
            senderTokens: senderTokens,
            receiverTokens: receiverTokens,
            batch: batch,
            time: time,
            weights: weights
        )
        return (res.zRel, res.zContent)
    }

    /// Runs Dual-Stream Relational Inference (`v6` + `v6_hybrid`) taking strictly `(senderTokenIDs: [Int], receiverTokenIDs: [Int])`.
    public func inferRelational(
        senderTokenIDs: [Int],
        receiverTokenIDs: [Int]
    ) -> RelationalInferenceOutput {
        let rawMax = max(senderTokenIDs.count, receiverTokenIDs.count)
        let time: Int = if relationalEncoderBackend != nil {
            max(DualStreamRelationalWeights.canonicalSeqLen, rawMax)
        } else {
            max(8, ((rawMax + 8 + 3) / 4) * 4)
        }
        let sPad = DualStreamRelationalEngine.padOrTruncate(senderTokenIDs, to: time)
        let rPad = DualStreamRelationalEngine.padOrTruncate(receiverTokenIDs, to: time)
        let res = encodeRelationalBatchWithContent(
            senderTokens: sPad,
            receiverTokens: rPad,
            batch: 1,
            time: time
        )
        let hyb = hybridExtractor.extract(
            senderTokens: sPad, receiverTokens: rPad, zRel: res.zRel, weights: weights
        )
        return DualStreamRelationalEngine.makeInferenceOutput(zRel: res.zRel, hybridFeatures: hyb, zContent: res.zContent, weights: weights.relational)
    }

    /// Phase 5 & 5.1: Relational Dynamics Engine:
    /// Encodes `(X^S, X^C(T_2), X^C(T_4))` via the Dual-Stream Subband Bidirectional Cross-Attention Encoder into
    /// `z_rel(T_2), z_rel(T_4) ∈ R^1024`, evaluates the 5-axis Relational Pentagon (`Score_ctx, Score_emp, Score_sync, Score_stab, Score_dev`),
    /// 5-tier `RelationalTopology`, and 6-class `RelationalArchetype` from the identical `z_rel` tensor (`Audit 5`),
    /// integrates the 50-Step Midpoint ODE for Multi-Resolution Insight Generation (`γ ∈ {0.0, 0.5, 1.0}`),
    /// and performs subband-orthogonal Latent Harmonization.
    public func analyzeRelationalDynamicsWithFlowMatching(
        hostEarlyTokens: [Int]? = nil,
        hostTokens: [Int],
        partnerEarlyTokens: [Int],
        partnerCurrentTokens: [Int],
        time: Int,
        subbandGamma: Float = 1.0,
        odeSteps: Int = 50,
        guidanceScale: Float = 1.5
    ) -> (
        zHost: [Float],
        zPartnerEarly: [Float],
        zPartnerCurrent: [Float],
        earlyMetrics: RelationalDynamicsMetrics,
        metrics: RelationalDynamicsMetrics,
        postHarmonizationMetrics: RelationalDynamicsMetrics,
        cInsight: [Float],
        zInsight: [Float],
        insightTokens: [Int],
        harmonizedTokens: [Int],
        trajectoryNorms: [Float]
    ) {
        let hEarlyTokens = hostEarlyTokens ?? hostTokens
        precondition(hEarlyTokens.count == time && hostTokens.count == time && partnerEarlyTokens.count == time && partnerCurrentTokens.count == time)
        let zTriple = encode(
            tokens: hostTokens + partnerEarlyTokens + partnerCurrentTokens,
            batch: 3,
            time: time
        )
        let lDimAE = weights.latentDim
        let zHost = Array(zTriple[0..<lDimAE])
        let zPartnerEarly = Array(zTriple[lDimAE..<(2 * lDimAE)])
        let zPartnerCurrent = Array(zTriple[(2 * lDimAE)..<(3 * lDimAE)])

        let cHarm = ConditionalFlowMatcher.computeRelationalHarmonizationCondition(
            zHost: zHost,
            zPartnerCurrent: zPartnerCurrent,
            weights: weights.flowMatching
        )
        let (zHarmTarget, _) = ConditionalFlowMatcher.sampleMidpointODE(
            condition: cHarm,
            weights: weights.flowMatching,
            batch: 1,
            steps: odeSteps,
            guidanceScale: guidanceScale
        )

        let legacyMetrics = ConditionalFlowMatcher.computeRelationalDynamics(
            zHost: zHost,
            zPartnerEarly: zPartnerEarly,
            zPartnerCurrent: zPartnerCurrent,
            zHarmonizedTarget: zHarmTarget
        )

        let earlyMetrics: RelationalDynamicsMetrics
        let metrics: RelationalDynamicsMetrics
        let postHarmonizationMetrics: RelationalDynamicsMetrics

        if weights.relational.calibratedHeads.archetypeT4 != nil || weights.subbandClassifier.modulationRules.isEmpty {
            let zRelT2 = encodeRelationalBatch(
                senderTokens: hEarlyTokens, receiverTokens: partnerEarlyTokens, batch: 1, time: time
            )
            let zRelT4 = encodeRelationalBatch(
                senderTokens: hostTokens, receiverTokens: partnerCurrentTokens, batch: 1, time: time
            )
            let p2 = DualStreamRelationalEngine.computePentagonFromZRel(zRel: zRelT2, zRelEarly: nil)
            let c2 = DualStreamRelationalEngine.classifyFromZRel(
                zRel: zRelT2, isEarlyT2: true, weights: weights.relational
            )
            let p4 = DualStreamRelationalEngine.computePentagonFromZRel(zRel: zRelT4, zRelEarly: zRelT2)
            let c4 = DualStreamRelationalEngine.classifyFromZRel(
                zRel: zRelT4, isEarlyT2: false, weights: weights.relational
            )

            // Subband-orthogonal Latent Harmonization in z_rel: preserve z_LL^rel [0..<relHalf] exactly, harmonize z_HH^rel [relHalf..<relDim]
            let relDim = weights.relational.latentDim
            let relHalf = relDim / 2
            let zHarmProto = DualStreamRelationalEngine.targetPrototype(
                archetype: .trueSupporter, topology: .deepAffection, latentDim: relDim
            )
            var zRelHarmonized = zRelT4
            var deltaZRel = [Float](repeating: 0.0, count: relDim)
            for i in relHalf..<relDim {
                let d = zHarmProto[i] - zRelT4[i]
                deltaZRel[i] = d
                zRelHarmonized[i] = zRelT4[i] + d
            }
            var llErrSq: Float = 0.0
            for i in 0..<relHalf {
                let e = zRelHarmonized[i] - zRelT4[i]
                llErrSq += e * e
            }
            let llPreservationError = sqrtf(llErrSq)

            let pHarm = DualStreamRelationalEngine.computePentagonFromZRel(
                zRel: zRelHarmonized, zRelEarly: zRelT2
            )
            let cHarm = DualStreamRelationalEngine.classifyFromZRel(
                zRel: zRelHarmonized, isEarlyT2: false, weights: weights.relational
            )
            earlyMetrics = RelationalDynamicsMetrics(
                dLL: p2.dLL,
                dHH: p2.dHH,
                rHH: p2.rHH,
                sHH: p2.sHH,
                sLL: p2.sLL,
                temporalConsistencyLL: p2.temporalConsistencyLL,
                pentagon: p2.pentagon,
                topology: c2.topology,
                archetype: c2.archetype,
                deltaZHarmonize: [Float](repeating: 0.0, count: weights.latentDim),
                zHarmonized: zPartnerEarly,
                llPreservationError: 0.0
            )
            metrics = RelationalDynamicsMetrics(
                dLL: p4.dLL,
                dHH: p4.dHH,
                rHH: p4.rHH,
                sHH: p4.sHH,
                sLL: p4.sLL,
                temporalConsistencyLL: p4.temporalConsistencyLL,
                pentagon: p4.pentagon,
                topology: c4.topology,
                archetype: c4.archetype,
                deltaZHarmonize: legacyMetrics.deltaZHarmonize,
                zHarmonized: legacyMetrics.zHarmonized,
                llPreservationError: max(llPreservationError, legacyMetrics.llPreservationError)
            )
            postHarmonizationMetrics = RelationalDynamicsMetrics(
                dLL: pHarm.dLL,
                dHH: pHarm.dHH,
                rHH: pHarm.rHH,
                sHH: pHarm.sHH,
                sLL: pHarm.sLL,
                temporalConsistencyLL: pHarm.temporalConsistencyLL,
                pentagon: pHarm.pentagon,
                topology: cHarm.topology,
                archetype: cHarm.archetype,
                deltaZHarmonize: deltaZRel,
                zHarmonized: legacyMetrics.zHarmonized,
                llPreservationError: llPreservationError
            )
        } else {
            earlyMetrics = ConditionalFlowMatcher.computeRelationalDynamics(
                zHost: zHost,
                zPartnerEarly: zPartnerEarly,
                zPartnerCurrent: zPartnerEarly,
                zHarmonizedTarget: nil
            )
            metrics = legacyMetrics
            postHarmonizationMetrics = ConditionalFlowMatcher.computeRelationalDynamics(
                zHost: zHost,
                zPartnerEarly: zPartnerEarly,
                zPartnerCurrent: legacyMetrics.zHarmonized,
                zHarmonizedTarget: zHarmTarget
            )
        }

        let cInsight = ConditionalFlowMatcher.computeRelationalInsightCondition(
            zHost: zHost,
            zPartnerCurrent: zPartnerCurrent,
            gamma: subbandGamma,
            weights: weights.flowMatching
        )
        let (zInsightRaw, trajNorms) = ConditionalFlowMatcher.sampleMidpointODE(
            condition: cInsight,
            weights: weights.flowMatching,
            batch: 1,
            steps: odeSteps,
            guidanceScale: guidanceScale
        )
        var zInsight = zInsightRaw
        let half = weights.latentDim / 2
        let carrierDims = min(44, half)
        for c in 0..<carrierDims {
            zInsight[c] = 0.94 * zInsightRaw[c] + 0.06 * cInsight[c]
        }

        let insightDecoded = reconstructGreedy(z: zInsight, batch: 1, length: time)[0]
        let harmonizedDecoded = reconstructGreedy(z: metrics.zHarmonized, batch: 1, length: time)[0]
        return (
            zHost,
            zPartnerEarly,
            zPartnerCurrent,
            earlyMetrics,
            metrics,
            postHarmonizationMetrics,
            cInsight,
            zInsight,
            insightDecoded,
            harmonizedDecoded,
            trajNorms
        )
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
        guard m > 0 && n > 0 else { return }
        if k == 0 {
            if beta == 0.0 {
                for r in 0..<m { (c + r * ldc).update(repeating: 0.0, count: n) }
            } else if beta != 1.0 {
                for r in 0..<m {
                    let cRow = c + r * ldc
                    for col in 0..<n { cRow[col] *= beta }
                }
            }
            return
        }

        let bVec = SIMD8<Float>(repeating: beta)
        var row = 0

        // Register-blocked 4x16 (4 rows x 2 SIMD8<Float>) AVX2 FMA micro-kernel
        while row + 4 <= m {
            let a0 = a + (row + 0) * lda
            let a1 = a + (row + 1) * lda
            let a2 = a + (row + 2) * lda
            let a3 = a + (row + 3) * lda
            let c0 = c + (row + 0) * ldc
            let c1 = c + (row + 1) * ldc
            let c2 = c + (row + 2) * ldc
            let c3 = c + (row + 3) * ldc

            var col = 0
            while col + 16 <= n {
                var acc00 = SIMD8<Float>(repeating: 0.0), acc01 = SIMD8<Float>(repeating: 0.0)
                var acc10 = SIMD8<Float>(repeating: 0.0), acc11 = SIMD8<Float>(repeating: 0.0)
                var acc20 = SIMD8<Float>(repeating: 0.0), acc21 = SIMD8<Float>(repeating: 0.0)
                var acc30 = SIMD8<Float>(repeating: 0.0), acc31 = SIMD8<Float>(repeating: 0.0)
                var p = 0
                while p < k {
                    let bPtr = b + p * ldb + col
                    let b0 = load(bPtr)
                    let b1 = load(bPtr + 8)
                    let v0 = SIMD8<Float>(repeating: a0[p])
                    let v1 = SIMD8<Float>(repeating: a1[p])
                    let v2 = SIMD8<Float>(repeating: a2[p])
                    let v3 = SIMD8<Float>(repeating: a3[p])
                    acc00.addProduct(v0, b0); acc01.addProduct(v0, b1)
                    acc10.addProduct(v1, b0); acc11.addProduct(v1, b1)
                    acc20.addProduct(v2, b0); acc21.addProduct(v2, b1)
                    acc30.addProduct(v3, b0); acc31.addProduct(v3, b1)
                    p += 1
                }
                if beta == 0.0 {
                    store(acc00, c0 + col); store(acc01, c0 + col + 8)
                    store(acc10, c1 + col); store(acc11, c1 + col + 8)
                    store(acc20, c2 + col); store(acc21, c2 + col + 8)
                    store(acc30, c3 + col); store(acc31, c3 + col + 8)
                } else if beta == 1.0 {
                    store(load(c0 + col) + acc00, c0 + col); store(load(c0 + col + 8) + acc01, c0 + col + 8)
                    store(load(c1 + col) + acc10, c1 + col); store(load(c1 + col + 8) + acc11, c1 + col + 8)
                    store(load(c2 + col) + acc20, c2 + col); store(load(c2 + col + 8) + acc21, c2 + col + 8)
                    store(load(c3 + col) + acc30, c3 + col); store(load(c3 + col + 8) + acc31, c3 + col + 8)
                } else {
                    store(load(c0 + col) * bVec + acc00, c0 + col); store(load(c0 + col + 8) * bVec + acc01, c0 + col + 8)
                    store(load(c1 + col) * bVec + acc10, c1 + col); store(load(c1 + col + 8) * bVec + acc11, c1 + col + 8)
                    store(load(c2 + col) * bVec + acc20, c2 + col); store(load(c2 + col + 8) * bVec + acc21, c2 + col + 8)
                    store(load(c3 + col) * bVec + acc30, c3 + col); store(load(c3 + col + 8) * bVec + acc31, c3 + col + 8)
                }
                col += 16
            }

            while col + 8 <= n {
                var acc0 = SIMD8<Float>(repeating: 0.0)
                var acc1 = SIMD8<Float>(repeating: 0.0)
                var acc2 = SIMD8<Float>(repeating: 0.0)
                var acc3 = SIMD8<Float>(repeating: 0.0)
                var p = 0
                while p < k {
                    let b0 = load(b + p * ldb + col)
                    acc0.addProduct(SIMD8<Float>(repeating: a0[p]), b0)
                    acc1.addProduct(SIMD8<Float>(repeating: a1[p]), b0)
                    acc2.addProduct(SIMD8<Float>(repeating: a2[p]), b0)
                    acc3.addProduct(SIMD8<Float>(repeating: a3[p]), b0)
                    p += 1
                }
                if beta == 0.0 {
                    store(acc0, c0 + col); store(acc1, c1 + col); store(acc2, c2 + col); store(acc3, c3 + col)
                } else if beta == 1.0 {
                    store(load(c0 + col) + acc0, c0 + col); store(load(c1 + col) + acc1, c1 + col)
                    store(load(c2 + col) + acc2, c2 + col); store(load(c3 + col) + acc3, c3 + col)
                } else {
                    store(load(c0 + col) * bVec + acc0, c0 + col); store(load(c1 + col) * bVec + acc1, c1 + col)
                    store(load(c2 + col) * bVec + acc2, c2 + col); store(load(c3 + col) * bVec + acc3, c3 + col)
                }
                col += 8
            }

            while col < n {
                var s0: Float = 0.0, s1: Float = 0.0, s2: Float = 0.0, s3: Float = 0.0
                var p = 0
                while p < k {
                    let bv = b[p * ldb + col]
                    s0 += a0[p] * bv; s1 += a1[p] * bv; s2 += a2[p] * bv; s3 += a3[p] * bv
                    p += 1
                }
                if beta == 0.0 {
                    c0[col] = s0; c1[col] = s1; c2[col] = s2; c3[col] = s3
                } else {
                    c0[col] = beta * c0[col] + s0; c1[col] = beta * c1[col] + s1
                    c2[col] = beta * c2[col] + s2; c3[col] = beta * c3[col] + s3
                }
                col += 1
            }
            row += 4
        }

        // Remaining rows (0..3)
        while row < m {
            let aRow = a + row * lda
            let cRow = c + row * ldc
            var col = 0
            while col + 16 <= n {
                var acc0 = SIMD8<Float>(repeating: 0.0)
                var acc1 = SIMD8<Float>(repeating: 0.0)
                var p = 0
                while p < k {
                    let aVec = SIMD8<Float>(repeating: aRow[p])
                    let bPtr = b + p * ldb + col
                    acc0.addProduct(aVec, load(bPtr))
                    acc1.addProduct(aVec, load(bPtr + 8))
                    p += 1
                }
                if beta == 0.0 {
                    store(acc0, cRow + col); store(acc1, cRow + col + 8)
                } else if beta == 1.0 {
                    store(load(cRow + col) + acc0, cRow + col); store(load(cRow + col + 8) + acc1, cRow + col + 8)
                } else {
                    store(load(cRow + col) * bVec + acc0, cRow + col); store(load(cRow + col + 8) * bVec + acc1, cRow + col + 8)
                }
                col += 16
            }
            while col + 8 <= n {
                var acc0 = SIMD8<Float>(repeating: 0.0)
                var p = 0
                while p < k {
                    acc0.addProduct(SIMD8<Float>(repeating: aRow[p]), load(b + p * ldb + col))
                    p += 1
                }
                if beta == 0.0 {
                    store(acc0, cRow + col)
                } else if beta == 1.0 {
                    store(load(cRow + col) + acc0, cRow + col)
                } else {
                    store(load(cRow + col) * bVec + acc0, cRow + col)
                }
                col += 8
            }
            while col < n {
                var s: Float = 0.0
                var p = 0
                while p < k {
                    s += aRow[p] * b[p * ldb + col]
                    p += 1
                }
                cRow[col] = (beta == 0.0) ? s : (beta * cRow[col] + s)
                col += 1
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
        guard rows > 0 && outDim > 0 else { return [] }
        if let bias = b {
            return [Float](unsafeUninitializedCapacity: rows * outDim) { yBuf, initializedCount in
                let yPtr = yBuf.baseAddress!
                bias.withUnsafeBufferPointer { bBuf in
                    let bPtr = bBuf.baseAddress!
                    for r in 0..<rows {
                        (yPtr + r * outDim).initialize(from: bPtr, count: outDim)
                    }
                }
                x.withUnsafeBufferPointer { xBuf in
                    w.withUnsafeBufferPointer { wBuf in
                        gemmRaw(xBuf.baseAddress!, inDim, wBuf.baseAddress!, outDim, yPtr, outDim, m: rows, n: outDim, k: inDim, beta: 1.0)
                    }
                }
                initializedCount = rows * outDim
            }
        } else {
            return [Float](unsafeUninitializedCapacity: rows * outDim) { yBuf, initializedCount in
                let yPtr = yBuf.baseAddress!
                x.withUnsafeBufferPointer { xBuf in
                    w.withUnsafeBufferPointer { wBuf in
                        gemmRaw(xBuf.baseAddress!, inDim, wBuf.baseAddress!, outDim, yPtr, outDim, m: rows, n: outDim, k: inDim, beta: 0.0)
                    }
                }
                initializedCount = rows * outDim
            }
        }
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
        guard rows > 0 && dim > 0 else { return [] }
        let invD = 1.0 / Float(dim)
        return [Float](unsafeUninitializedCapacity: rows * dim) { yBuf, initializedCount in
            let yBase = yBuf.baseAddress!
            x.withUnsafeBufferPointer { xBuf in
                gamma.withUnsafeBufferPointer { gBuf in
                    beta.withUnsafeBufferPointer { bBuf in
                        let xBase = xBuf.baseAddress!
                        let gBase = gBuf.baseAddress!
                        let bBase = bBuf.baseAddress!
                        for r in 0..<rows {
                            let xRow = xBase + r * dim
                            let yRow = yBase + r * dim
                            var sumVec = SIMD8<Float>(repeating: 0.0)
                            var c = 0
                            while c + 8 <= dim {
                                sumVec += load(xRow + c)
                                c += 8
                            }
                            var sum = sumVec.sum()
                            while c < dim {
                                sum += xRow[c]
                                c += 1
                            }
                            let mean = sum * invD
                            let meanVec = SIMD8<Float>(repeating: mean)

                            var varVec = SIMD8<Float>(repeating: 0.0)
                            c = 0
                            while c + 8 <= dim {
                                let d = load(xRow + c) - meanVec
                                varVec.addProduct(d, d)
                                c += 8
                            }
                            var varSum = varVec.sum()
                            while c < dim {
                                let d = xRow[c] - mean
                                varSum += d * d
                                c += 1
                            }
                            let invStd = 1.0 / sqrtf(varSum * invD + eps)
                            let invStdVec = SIMD8<Float>(repeating: invStd)

                            c = 0
                            while c + 8 <= dim {
                                let norm = (load(xRow + c) - meanVec) * invStdVec
                                store(norm * load(gBase + c) + load(bBase + c), yRow + c)
                                c += 8
                            }
                            while c < dim {
                                yRow[c] = (xRow[c] - mean) * invStd * gBase[c] + bBase[c]
                                c += 1
                            }
                        }
                    }
                }
            }
            initializedCount = rows * dim
        }
    }

    /// Element-wise SiLU `x * σ(x)` (`SIMD8<Float>` via `SIMDMath.sigmoid`).
    public static func silu(_ x: [Float]) -> [Float] {
        let n = x.count
        guard n > 0 else { return [] }
        return [Float](unsafeUninitializedCapacity: n) { outBuf, initializedCount in
            let outPtr = outBuf.baseAddress!
            x.withUnsafeBufferPointer { xBuf in
                let xPtr = xBuf.baseAddress!
                var i = 0
                while i + 8 <= n {
                    let v = load(xPtr + i)
                    store(v * SIMDMath.sigmoid(v), outPtr + i)
                    i += 8
                }
                while i < n {
                    let v = xPtr[i]
                    outPtr[i] = v / (1.0 + expf(-v))
                    i += 1
                }
            }
            initializedCount = n
        }
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
