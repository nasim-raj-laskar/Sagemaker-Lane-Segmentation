"""src/training/train.py — SageMaker Training Job entrypoint.

SageMaker injects these environment variables at runtime:
    SM_CHANNEL_TRAIN  → path to training data mounted from S3
    SM_CHANNEL_VAL    → path to validation data mounted from S3
    SM_MODEL_DIR      → where to write the saved model artifact
    SM_OUTPUT_DATA_DIR → where to write any other output files
    SM_NUM_GPUS        → number of GPUs on the instance

Run locally:
    SM_CHANNEL_TRAIN=data/processed/train \
    SM_CHANNEL_VAL=data/processed/val \
    SM_MODEL_DIR=artifacts/model \
    python -m src.training.train
"""

from __future__ import annotations

import os
import sys

import hydra
import numpy as np
from omegaconf import DictConfig

from src.data.preprocessing import build_tf_dataset, split_dataset
from src.training.trainer import Trainer
from src.utils.logging import get_logger
from src.utils.s3 import package_model_for_sagemaker

logger = get_logger(__name__)

SM_CHANNEL_TRAIN = os.environ.get("SM_CHANNEL_TRAIN", "data/processed/train")
SM_CHANNEL_VAL = os.environ.get("SM_CHANNEL_VAL", "data/processed/val")
SM_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "artifacts/model")
SM_OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "artifacts/output")


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logger.info("Starting training job")
    logger.info("Train channel: %s", SM_CHANNEL_TRAIN)
    logger.info("Model dir:     %s", SM_MODEL_DIR)

    # ── Load pre-processed numpy arrays written by the Processing Job ───────
    X_train = np.load(os.path.join(SM_CHANNEL_TRAIN, "X_train.npy"))
    Y_train = np.load(os.path.join(SM_CHANNEL_TRAIN, "Y_train.npy"))
    X_val = np.load(os.path.join(SM_CHANNEL_VAL, "X_val.npy"))
    Y_val = np.load(os.path.join(SM_CHANNEL_VAL, "Y_val.npy"))
    logger.info("Loaded arrays — X_train: %s  X_val: %s", X_train.shape, X_val.shape)

    train_ds = build_tf_dataset(X_train, Y_train, cfg, training=True)
    val_ds = build_tf_dataset(X_val, Y_val, cfg, training=False)

    trainer = Trainer(cfg)
    trainer.setup()
    run = trainer.train(train_ds, val_ds)
    logger.info("MLflow run id: %s", run.info.run_id)

    # Save model to SM_MODEL_DIR so SageMaker packages it as model.tar.gz
    trainer.save(SM_MODEL_DIR)
    logger.info("Training complete")


if __name__ == "__main__":
    main()
