SWIFT ?= swift
DATA ?= .tmp
EPOCHS ?= 60
WEIGHTS ?= .tmp/weights.safetensors
REPORT ?= .tmp/verification_report.md
DROPOUT ?= 0.25
REL_EPOCHS ?= 15
AE_WEIGHTS ?= .tmp/weights_ae.safetensors

export PATH := /usr/local/cuda/bin:$(PATH)
export CUDA_ARCH ?= sm_86
export MLX_CCCL_DIR ?= $(HOME)/.local/share/cccl/include
export MLX_CUDA_INCLUDE_PATHS ?= $(HOME)/.local/share/cudnn-frontend/include:$(HOME)/.local/share/cutlass/include
export NVCC_PREPEND_FLAGS ?= -allow-unsupported-compiler --extended-lambda

.PHONY: all build test train eval reconstruct verify verify-fast head benchmarks verify-all-fast head-gpu-cv verify-gpu-retrain benchmark psych-train psych-eval clean help

all: build test

build:
	$(SWIFT) build -c release

test:
	$(SWIFT) test

train:
	$(SWIFT) run -c release train -d $(DATA) -e $(EPOCHS) --word-dropout $(DROPOUT) --rel-epochs $(REL_EPOCHS) --ae-weights $(AE_WEIGHTS) --export-weights $(WEIGHTS)

train-rel:
	$(SWIFT) run -c release train -d $(DATA) -e 0 --import-weights $(AE_WEIGHTS) --rel-epochs $(REL_EPOCHS) --ae-weights "" --export-weights $(WEIGHTS)

eval:
	$(SWIFT) run -c release train -d $(DATA) -e 0 --rel-epochs 0 --import-weights $(WEIGHTS)

reconstruct:
	$(SWIFT) run -c release reconstruct -d $(DATA) -w $(WEIGHTS)

verify:
	$(SWIFT) run -c release verify_latent_space --cpu --fast -d $(DATA) -w $(WEIGHTS) -o $(REPORT) --ae-weights $(AE_WEIGHTS) --audit-report .tmp/audit_and_verification_report.md

verify-fast: verify

head:
	$(SWIFT) run -c release relational_head --cpu -d $(DATA) -w $(WEIGHTS) -o .tmp/relational_head_report.md --heads .tmp/relational_heads.json

benchmarks:
	$(SWIFT) run -c release validate_benchmarks --cpu -d $(DATA)

verify-all-fast: head verify benchmarks

head-gpu-cv:
	$(SWIFT) run -c release relational_head --retrain-folds -d $(DATA) -w $(WEIGHTS) --ae-weights $(AE_WEIGHTS) --fold-epochs $(REL_EPOCHS) -o .tmp/relational_head_report.md --heads .tmp/relational_heads.json

verify-gpu-retrain:
	$(SWIFT) run -c release verify_latent_space --run-permutation-audit -d $(DATA) -w $(WEIGHTS) -o $(REPORT) --ae-weights $(AE_WEIGHTS) --perm-epochs $(REL_EPOCHS) --audit-report .tmp/audit_and_verification_report.md

benchmark:
	$(SWIFT) run -c release benchmark

psych-train:
	$(SWIFT) run -c release psych_lm train --rel-weights $(WEIGHTS) --psych-weights .tmp/weights_psych_lm.safetensors --epochs 22

psych-eval:
	$(SWIFT) run -c release psych_lm eval --rel-weights $(WEIGHTS) --psych-weights .tmp/weights_psych_lm.safetensors

clean:
	$(SWIFT) package clean

help:
	@echo "make build             Build all targets in release mode"
	@echo "make test              Run unit test suite"
	@echo "make verify-all-fast   Run full pure-CPU verification (head + verify + benchmarks, 0 GPU)"
	@echo "make benchmarks        Run 5-layer external validation benchmarks on CPU (cached <2s)"
	@echo "make head              Evaluate relational heads + fast frozen 6-fold CV on CPU (0 GPU)"
	@echo "make verify            Run fast latent space + Audit 1-5 verification on CPU (0 GPU)"
	@echo "make head-gpu-cv       [Heavy Opt-in] Run 6-fold end-to-end GPU encoder retraining CV"
	@echo "make verify-gpu-retrain[Heavy Opt-in] Run Audit 3c full GPU permuted-label retraining"
	@echo "make train             Train SWT-Hawk autoencoder + relational stage on GPU"
	@echo "make benchmark         Run SWT-Hawk autoencoder throughput benchmark"
	@echo "make clean             Clean Swift build artifacts"

