SHELL := /bin/bash

UV ?= uv
PYTHON_VERSION ?= 3.12
BENCH_OPPONENT ?= baselines/minimax
BENCH_ROUNDS ?= 1
BENCH_BASE_MS ?= 10000
GPU_IMAGE ?= aichessathon-training:local
STOCKFISH ?= stockfish

.PHONY: setup play arena bench test distill-smoke distill-e2e-smoke aws-deploy aws-status aws-cpu aws-gpu aws-bench aws-teacher-pilot train-setup train-smoke aws-train aws-train-distilled gpu-build gpu-smoke gpu-train zip gate

setup:
	$(UV) sync --python $(PYTHON_VERSION)

play:
	$(UV) run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	$(UV) run python -m harness.arena --opponent baselines/greedy --games 20

bench:
	$(UV) run python -m benchmarks.run --opponent $(BENCH_OPPONENT) --rounds $(BENCH_ROUNDS) --base-ms $(BENCH_BASE_MS) --output benchmark-results/latest.json

test:
	$(UV) run python -m unittest discover -s tests -v

distill-smoke:
	$(UV) run python -m training.train_distilled --smoke --device cpu --epochs 1 --batch-size 32 --output /tmp/aichessathon-distilled-smoke/nnue.npz

distill-e2e-smoke:
	$(UV) run python -m distill.annotate --input benchmarks/openings.epd --output /tmp/aichessathon-teacher-smoke.jsonl.gz --stockfish $(STOCKFISH) --nodes 1000 --multipv 4 --limit 2 --keep-pv
	$(UV) run python -m distill.build_dataset /tmp/aichessathon-teacher-smoke.jsonl.gz --output /tmp/aichessathon-teacher-dataset --records-per-shard 2
	$(UV) run python -m training.train_distilled --data /tmp/aichessathon-teacher-dataset --device cpu --epochs 1 --batch-size 1 --accumulator 32 --hidden 24 --bottleneck 16 --output /tmp/aichessathon-teacher-model/nnue.npz

aws-deploy:
	bash infra/aws/compute.sh deploy

aws-status:
	bash infra/aws/compute.sh status

aws-cpu:
	bash infra/aws/compute.sh launch-cpu

aws-gpu:
	bash infra/aws/compute.sh launch-gpu

aws-bench:
	bash infra/aws/benchmark.sh

aws-teacher-pilot:
	bash infra/aws/teacher-pilot.sh

train-setup:
	$(UV) sync --python $(PYTHON_VERSION) --group training

train-smoke:
	$(UV) run --group training python training/train.py --smoke --device cpu --epochs 1 --output /tmp/aichessathon-smoke/value.onnx

aws-train:
	bash infra/aws/train.sh

aws-train-distilled:
	bash infra/aws/train-distilled.sh

gpu-build:
	docker build -f training/Dockerfile -t $(GPU_IMAGE) .

gpu-smoke:
	docker run --rm --gpus all $(GPU_IMAGE) python training/device_check.py --require-cuda

gpu-train:
	docker run --rm --gpus all -v "$(CURDIR):/workspace" $(GPU_IMAGE) python training/train.py --device cuda --data training/data/positions.npz --output weights/value.onnx

zip:
	$(UV) run python -m harness.package

gate:
	$(UV) run ruff check .
	$(UV) run mypy
	$(UV) run python -m unittest discover -s tests
	$(UV) run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
