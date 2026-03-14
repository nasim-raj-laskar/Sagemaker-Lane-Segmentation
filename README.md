# Road Segmentation — DLOps

Binary road segmentation on the KITTI dataset using a ResNet50-UNet architecture,
deployed as a production ML system on AWS SageMaker.

## Project structure

```
road_segmentation/
├── configs/                    # Hydra config tree (never hardcode values)
│   ├── config.yaml             # root — composes all sub-configs
│   ├── data/kitti.yaml         # dataset paths, splits, augmentation
│   ├── model/resnet_unet.yaml  # architecture, backbone, decoder
│   ├── training/default.yaml   # epochs, optimizer, callbacks, SM instance
│   ├── inference/default.yaml  # threshold, postprocessing
│   ├── monitoring/default.yaml # CloudWatch, Model Monitor, drift thresholds
│   └── aws/sagemaker.yaml      # region, S3, ECR, endpoint, pipeline
│
├── src/
│   ├── data/
│   │   ├── loader.py           # load_dataset(), stream_from_s3()
│   │   └── preprocessing.py    # split_dataset(), build_tf_dataset()
│   ├── models/
│   │   ├── resnet_unet.py      # build_resnet_unet(), unfreeze_encoder()
│   │   └── losses.py           # bce_dice_loss, IoUScore, DiceCoefficient
│   ├── training/
│   │   ├── trainer.py          # Trainer class — compile, fit, MLflow, save
│   │   └── train.py            # SageMaker Training Job entrypoint
│   ├── inference/
│   │   └── handler.py          # model_fn / input_fn / predict_fn / output_fn
│   ├── evaluation/
│   │   └── evaluator.py        # Evaluator class + save_report()
│   ├── monitoring/
│   │   └── monitor.py          # CloudWatchReporter, ModelMonitorSetup
│   ├── pipelines/
│   │   └── sagemaker_pipeline.py  # full 5-step Pipeline definition
│   ├── api/
│   │   └── main.py             # FastAPI service (POST /segment)
│   └── utils/
│       ├── config.py           # load_config() for scripts/notebooks
│       ├── logging.py          # get_logger()
│       ├── s3.py               # S3Client, package_model_for_sagemaker()
│       └── seed.py             # set_global_seed()
│
├── scripts/
│   ├── processing_job.py       # SageMaker ProcessingStep entrypoint
│   ├── evaluate.py             # SageMaker evaluation step entrypoint
│   ├── deploy.py               # deploy / update SageMaker endpoint
│   └── run_pipeline.py         # upsert + start the full pipeline
│
├── tests/
│   ├── unit/
│   │   ├── test_losses.py
│   │   └── test_preprocessing.py
│   └── integration/            # add SM endpoint smoke tests here
│
├── docker/
│   └── Dockerfile              # FastAPI inference image
│
├── notebooks/                  # exploratory only — no production logic here
│
├── .github/workflows/ci.yml    # lint → test → docker build → ECS deploy
├── .env.example                # copy to .env, fill values, never commit
├── .gitignore
├── Makefile                    # dev shortcuts
└── pyproject.toml              # packaging, deps, tool config
```

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/your-org/road-segmentation
cd road-segmentation
cp .env.example .env            # fill in AWS_ACCOUNT_ID, S3_BUCKET, etc.
make install

# 2. Download KITTI and preprocess locally
make process-local              # reads data/raw/, writes data/processed/

# 3. Smoke-test training (2 epochs, no GPU needed)
make train-local

# 4. Run the full SageMaker pipeline
make pipeline-run               # upserts definition + starts execution

# 5. Deploy endpoint
make deploy

# 6. Run tests
make test
```

## Configuration

All values live in `configs/`. Override anything at runtime with Hydra syntax:

```bash
# Change epochs and batch size for a quick run
python -m src.training.train training.epochs=5 data.dataloader.batch_size=4

# Point at a different S3 bucket
python scripts/run_pipeline.py data.s3.bucket=my-other-bucket

# Use spot instances for training
python scripts/run_pipeline.py training.sagemaker.use_spot=true
```

Config values flow through the entire codebase — `configs/training/default.yaml`
controls both the local `Trainer` and the SageMaker `TensorFlow` estimator
hyperparameters. You never touch two files to change one setting.

## DLOps principles followed

| Principle | Implementation |
|---|---|
| Config-driven | Hydra + OmegaConf — zero hardcoded values |
| Reproducibility | `set_global_seed()`, config logged as artifact every run |
| Experiment tracking | MLflow logs params, metrics, config YAML, model artifact |
| Data versioning | S3 prefixes with `raw/`, `processed/`, timestamped splits |
| Model versioning | SageMaker Model Registry with approval gate |
| Automated pipeline | SageMaker Pipelines — process → train → evaluate → register |
| Quality gate | `ConditionStep` blocks registration if IoU < threshold |
| Monitoring | Model Monitor (drift) + CloudWatch (latency/errors) + SNS alerts |
| CI/CD | GitHub Actions: lint → test → build → push ECR → deploy ECS |
| Containerisation | Reproducible inference image, pushed to ECR on every main merge |
| Least privilege | Separate IAM roles for training, inference, deployment |

## Environment variables

See `.env.example` for the full list. Required before any AWS operation:

| Variable | Purpose |
|---|---|
| `AWS_DEFAULT_REGION` | All boto3/SM SDK calls |
| `S3_BUCKET` | Data and model artifact storage |
| `SAGEMAKER_ROLE_ARN` | Execution role for training jobs and pipelines |
| `MLFLOW_TRACKING_URI` | Experiment tracking server |
| `SAGEMAKER_ENDPOINT_NAME` | Name of the deployed inference endpoint |
