.PHONY: install lint test train deploy pipeline clean

PYTHON := python
PIP    := pip

install:
	$(PIP) install -e ".[dev,api]"
	pre-commit install

lint:
	ruff check src/ scripts/ tests/
	black --check src/ scripts/ tests/
	mypy src/

format:
	black src/ scripts/ tests/
	ruff check --fix src/ scripts/ tests/

test:
	pytest tests/unit/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ── Local training (small subset for smoke-testing) ─────────────────────────
train-local:
	SM_CHANNEL_TRAIN=data/processed/train \
	SM_CHANNEL_VAL=data/processed/val \
	SM_MODEL_DIR=artifacts/model \
	$(PYTHON) -m src.training.train training.epochs=2 data.dataloader.batch_size=4

# ── Processing job (local) ───────────────────────────────────────────────────
process-local:
	$(PYTHON) scripts/processing_job.py \
		--input-dir data/raw \
		--output-train data/processed/train \
		--output-val data/processed/val \
		--output-test data/processed/test

# ── SageMaker pipeline ───────────────────────────────────────────────────────
pipeline-dry:
	$(PYTHON) scripts/run_pipeline.py --dry-run

pipeline-run:
	$(PYTHON) scripts/run_pipeline.py

# ── Endpoint deploy ──────────────────────────────────────────────────────────
deploy:
	$(PYTHON) scripts/deploy.py

deploy-update:
	$(PYTHON) scripts/deploy.py --update-only

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile -t road-seg-inference:local .

docker-run:
	docker run -p 8000:8000 \
		-e SAGEMAKER_ENDPOINT_NAME=$(SAGEMAKER_ENDPOINT_NAME) \
		-e AWS_DEFAULT_REGION=$(AWS_DEFAULT_REGION) \
		road-seg-inference:local

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf artifacts/checkpoints artifacts/output logs .coverage htmlcov
