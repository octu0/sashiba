// swift-tools-version: 6.0
import PackageDescription

let linuxSIMDFlags: [SwiftSetting] = [
    .unsafeFlags(["-Xcc", "-mavx2", "-Xcc", "-mfma", "-Xllvm", "-mattr=+avx2,+fma"], .when(platforms: [.linux]))
]

let package = Package(
    name: "sashiba",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "Sashiba", targets: ["Sashiba"]),
        .executable(name: "train", targets: ["train"]),
        .executable(name: "reconstruct", targets: ["reconstruct"]),
        .executable(name: "verify_latent_space", targets: ["verify_latent_space"]),
        .executable(name: "benchmark", targets: ["benchmark"]),
        .executable(name: "relational_head", targets: ["relational_head"]),
        .executable(name: "validate_benchmarks", targets: ["validate_benchmarks"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.33.0")
    ],
    targets: [
        .target(
            name: "Sashiba",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift")
            ],
            path: "Sources/Sashiba",
            swiftSettings: linuxSIMDFlags
        ),
        .executableTarget(
            name: "train",
            dependencies: [
                "Sashiba",
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "script/train",
            swiftSettings: linuxSIMDFlags
        ),
        .executableTarget(
            name: "reconstruct",
            dependencies: ["Sashiba"],
            path: "script/reconstruct",
            swiftSettings: linuxSIMDFlags
        ),
        .executableTarget(
            name: "verify_latent_space",
            dependencies: ["Sashiba"],
            path: "script/verify_latent_space",
            swiftSettings: linuxSIMDFlags
        ),
        .executableTarget(
            name: "benchmark",
            dependencies: ["Sashiba"],
            path: "script/benchmark",
            swiftSettings: linuxSIMDFlags
        ),
        .executableTarget(
            name: "relational_head",
            dependencies: [
                "Sashiba",
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "script/relational_head",
            swiftSettings: linuxSIMDFlags
        ),
        .executableTarget(
            name: "validate_benchmarks",
            dependencies: [
                "Sashiba",
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "script/validate_benchmarks",
            swiftSettings: linuxSIMDFlags
        ),
        .testTarget(
            name: "SashibaTests",
            dependencies: [
                "Sashiba",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ],
            path: "Tests/SashibaTests",
            swiftSettings: linuxSIMDFlags
        )
    ]
)
