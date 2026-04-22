# 🚗 Self-Driving Perceptron — Road Segmentation DLOps Pipeline

Production-grade road segmentation on the KITTI dataset, migrated from a Kaggle notebook to a fully automated, end-to-end DLOps pipeline on **AWS SageMaker**.

---

## Architecture

```
S3 (raw-data) → SM Processing Job → SM Training Job (g4dn.4xlarge)
     ↓                                      ↓
processed-data/                    SM Model Registry
                                           ↓
                               SM Real-time Endpoint  +  Batch Transform
                                           ↓
                               CloudWatch + Model Monitor + MLflow
                                           ↓
                            CodePipeline (CI/CD on git push)
```

---

## Project Structure

```
self-driving-perceptron/
├── config/
│   ├── config.yaml            # Master config: AWS, data, model, training, inference
│   └── config_loader.py       # Pydantic-validated loader with lru_cache
│
├── src/
│   ├── data/
│   │   ├── s3_io.py           # S3 read/write helpers (streaming, pagination)
│   │   └── preprocessing.py   # KITTI loader, augmentation pipeline, train/val split
│   ├── models/
│   │   ├── resnet_unet.py     # ResNet50-backbone UNet (modular decoder blocks)
│   │   └── metrics.py         # BinaryIoU, DiceLoss, CombinedLoss
│   ├── training/
│   │   └── train.py           # SM Training Job entry point + MLflow integration
│   ├── inference/
│   │   ├── predictor.py       # LocalPredictor + SageMakerPredictor
│   │   └── serve.py           # SageMaker model_fn / input_fn / predict_fn / output_fn
│   └── utils/
│       ├── callbacks.py       # Checkpoint, EarlyStopping, ReduceLR, MLflow, CSVLogger
│       └── logging_utils.py   # Structured logging setup
│
├── pipelines/
│   └── pipeline.py            # SageMaker Pipeline: Preprocess→Train→Eval→Register
│
├── scripts/
│   ├── preprocess_job.py      # SK-Learn Processing container script
│   ├── evaluate_job.py        # Eval container script → evaluation.json
│   ├── deploy.py              # Deploy/update/delete SM endpoint
│   ├── run_video_inference.py # Batch video overlay (testing/ → model-artifacts/)
│   ├── setup_monitoring.py    # Model Monitor + CloudWatch alarm setup
│   └── ecr_push.sh            # Build & push Docker images to ECR
│
├── docker/
│   ├── Dockerfile.training    # GPU training image (TF 2.13 + albumentations)
│   └── Dockerfile.inference   # CPU inference image (lightweight)
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   └── test_model.py
│   └── integration/
│       └── test_s3_io.py      # moto-mocked S3 tests
│
├── .github/workflows/ci.yml   # CI (lint+test) + CD (ECR push + SM Pipeline)
├── .pre-commit-config.yaml
└── pyproject.toml
```

---

## Quick Start

### 1. Install

```bash
git clone <your-repo>
cd self-driving-perceptron
pip install -e ".[dev]"
pre-commit install
```

### 2. Configure

Edit `config/config.yaml`:
```yaml
aws:
  region: us-east-1
  s3_bucket: self-driving-perceptron
  sagemaker:
    role_name: SageMakerExecutionRole   # your IAM role
  mlflow:
    tracking_uri: http://<your-ec2>:5000
```

### 3. Run tests locally

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

### 4. Build & push Docker images

```bash
chmod +x scripts/ecr_push.sh
./scripts/ecr_push.sh all
```

### 5. Create & run the SageMaker Pipeline

```bash
# Upsert (create or update) the pipeline definition
python pipelines/pipeline.py --action upsert

# Start a run
python pipelines/pipeline.py --action start --wait
```

This executes: **Preprocess → Train → Evaluate → (if IoU ≥ 0.60) → Register**

### 6. Deploy the model

```bash
# Deploy latest approved model to real-time endpoint
python scripts/deploy.py --action deploy

# Update endpoint with a newer model version
python scripts/deploy.py --action update
```

### 7. Set up monitoring

```bash
python scripts/setup_monitoring.py
```

### 8. Run video inference (batch)

```bash
python scripts/run_video_inference.py \
    --model-path /path/to/road_seg_savedmodel
```

---

## CI/CD Flow

Every `git push` to `main`:
1. **CI** — Black, isort, flake8, pytest (unit + integration with moto mocking)
2. **Build** — Docker images pushed to ECR with `git sha` tag
3. **Retrain** — SageMaker Pipeline upserted + execution triggered

Uses GitHub OIDC → AWS role assumption (no long-lived keys in GitHub secrets).

---

## AWS Services Used

| Service | Purpose |
|---------|---------|
| S3 | Raw data, processed splits, model artifacts, monitoring reports |
| SageMaker Processing | Preprocessing and evaluation jobs |
| SageMaker Training | GPU-backed model training (g4dn.4xlarge) |
| SageMaker Model Registry | Model versioning and approval gate |
| SageMaker Pipelines | Orchestrates the full ML workflow |
| SageMaker Endpoints | Real-time inference REST API |
| SageMaker Batch Transform | Offline video segmentation |
| SageMaker Model Monitor | Data drift + model quality detection |
| ECR | Docker image registry |
| CloudWatch | Logs, metrics, alarms |
| MLflow | Experiment tracking (runs on EC2 or SM-hosted) |
| IAM | Least-privilege execution roles |
| CodePipeline / CodeBuild | Alternative CI/CD trigger (see `.github/`) |

---

## Hyperparameter Tuning (optional)

SageMaker Automatic Model Tuning can be enabled by adding a `HyperparameterTuner`
around the `TensorFlow` estimator in `pipelines/pipeline.py`. Tune `learning_rate`,
`batch_size`, and `dropout_rate` using Bayesian optimization.

---

## License

MIT
