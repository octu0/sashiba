SWIFT ?= swift
DATA ?= .tmp
EPOCHS ?= 60
WEIGHTS ?= .tmp/weights.safetensors
REPORT ?= .tmp/verification_report.md
DROPOUT ?= 0.25

export PATH := /usr/local/cuda/bin:$(PATH)
export CUDA_ARCH ?= sm_86
export MLX_CCCL_DIR ?= $(HOME)/.local/share/cccl/include
export MLX_CUDA_INCLUDE_PATHS ?= $(HOME)/.local/share/cudnn-frontend/include:$(HOME)/.local/share/cutlass/include
export NVCC_PREPEND_FLAGS ?= -allow-unsupported-compiler --extended-lambda

.PHONY: all build test train eval reconstruct verify benchmark clean help

all: build test

build:
	$(SWIFT) build -c release

test:
	$(SWIFT) test

train:
	$(SWIFT) run -c release train -d $(DATA) -e $(EPOCHS) --word-dropout $(DROPOUT) --export-weights $(WEIGHTS)

eval:
	$(SWIFT) run -c release train -d $(DATA) -e 0 --import-weights $(WEIGHTS)

reconstruct:
	$(SWIFT) run -c release reconstruct -d $(DATA) -w $(WEIGHTS)

verify:
	$(SWIFT) run -c release verify_latent_space -d $(DATA) -w $(WEIGHTS) -o $(REPORT)

benchmark:
	$(SWIFT) run -c release benchmark

clean:
	$(SWIFT) package clean

help:
	@echo "make build        Build all targets in release mode"
	@echo "make test         Run unit test suite"
	@echo "make train        Train SWT-Hawk autoencoder on $(DATA) -> $(WEIGHTS)"
	@echo "make eval         Evaluate trained weights on $(DATA)"
	@echo "make reconstruct  Encode text to 1024-D vector z and reconstruct text"
	@echo "make verify       Run 4 latent space verification tests -> $(REPORT)"
	@echo "make benchmark    Run SWT-Hawk autoencoder throughput benchmark"
	@echo "make clean        Clean Swift build artifacts"

