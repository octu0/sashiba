FROM swift:6.0-jammy AS builder
WORKDIR /app

COPY Package.swift ./
COPY Sources ./Sources
COPY script ./script

RUN swift build -c release --product reconstruct

FROM swift:6.0-jammy-slim
WORKDIR /app

COPY --from=builder /app/.build/release/reconstruct /app/reconstruct
COPY assets ./assets
COPY .tmp/weights.safetensors ./.tmp/weights.safetensors

ENV WEIGHTS_PATH=/app/.tmp/weights.safetensors

CMD ["/app/reconstruct"]

