"""
src/training/train.py
SageMaker Training Job entry point.

SageMaker injects env vars:
  SM_CHANNEL_TRAIN  → /opt/ml/input/data/train
  SM_CHANNEL_VAL    → /opt/ml/input/data/val
  SM_OUTPUT_DATA_DIR → /opt/ml/output/data
  SM_MODEL_DIR       → /opt/ml/model   (saved model goes here)
  SM_HP_*            → hyperparameters passed at job launch

MLflow is used for experiment tracking.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0'
os.environ['XLA_FLAGS'] = '--xla_disable_all_hlo_passes'

import yaml
import mlflow
import numpy as np
import tensorflow as tf

tf.config.optimizer.set_jit(False)


#  path hack for local dev 
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config_loader import load_config
from src.data.preprocessing import build_dataset, split_dataset
from src.models.metrics import BinaryIoU, CombinedLoss
from src.models.resnet_unet import build_model_from_config
from src.utils.callbacks import build_callbacks
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


#  Hyperparameter parser (SM injects these) 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Directories (SM sets these automatically)
    p.add_argument("--train",     default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    p.add_argument("--val",       default=os.environ.get("SM_CHANNEL_VAL",   "/opt/ml/input/data/val"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR",     "/opt/ml/model"))
    p.add_argument("--output-dir",default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    # Hyperparams (can be overridden at job launch)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--batch-size",     type=int,   default=None)
    p.add_argument("--learning-rate",  type=float, default=None)
    p.add_argument("--img-size",       type=int,   default=None)

    # MLflow
    p.add_argument("--mlflow-tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", ""))

    # Config path
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[2] / "config/config.yaml"))

    return p.parse_args()


#  Main 

def main() -> None:
    setup_logging()
    args = parse_args()

    cfg = load_config(args.config)

    # Override config with CLI args if provided
    if args.epochs:        cfg.training.epochs        = args.epochs
    if args.batch_size:    cfg.training.batch_size     = args.batch_size
    if args.learning_rate: cfg.training.learning_rate  = args.learning_rate
    if args.img_size:      cfg.data.img_size            = args.img_size

    logger.info("Training config: epochs=%d  bs=%d  lr=%g  img_size=%d",
                cfg.training.epochs, cfg.training.batch_size,
                cfg.training.learning_rate, cfg.data.img_size)

    #  MLflow 
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(cfg.aws.mlflow.get("experiment_name", "road-segmentation"))

    with mlflow.start_run() as run:
        mlflow.log_params({
            "epochs":        cfg.training.epochs,
            "batch_size":    cfg.training.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "img_size":      cfg.data.img_size,
            "backbone":      cfg.model.backbone,
            "loss":          cfg.training.loss,
        })

        #  Load data 
        logger.info("Loading training data from %s", args.train)
        img_dir  = Path(args.train) / "image"
        mask_dir = Path(args.train) / "mask"

        X, Y = build_dataset(
            image_dir=img_dir,
            mask_dir=mask_dir,
            img_size=cfg.data.img_size,
            augment=cfg.data.augmentation.get("enabled", True),
            aug_cfg=cfg.data.augmentation,
        )

        if args.val and Path(args.val).exists() and any(Path(args.val).iterdir()):
            logger.info("Loading val data from %s", args.val)
            X_val, Y_val = build_dataset(
                image_dir=Path(args.val) / "image",
                mask_dir=Path(args.val) / "mask",
                img_size=cfg.data.img_size,
            )
        else:
            logger.info("No val channel; splitting from train (%g)", cfg.data.val_split)
            X, X_val, Y, Y_val = split_dataset(X, Y, cfg.data.val_split, cfg.data.random_seed)

        logger.info("Train: %s  Val: %s", X.shape, X_val.shape)

        #  Build model 
        model = build_model_from_config(cfg.data, cfg.model)

        loss_fn = CombinedLoss() if cfg.training.loss == "combined" else cfg.training.loss

        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg.training.learning_rate),
            loss=loss_fn,
            metrics=["accuracy", BinaryIoU(name="iou")],
        )

        #  Callbacks 
        callbacks = build_callbacks(
            cfg=cfg,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            mlflow_run_id=run.info.run_id,
        )

        #  Train 
        history = model.fit(
            X, Y,
            validation_data=(X_val, Y_val),
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        #  Log final metrics 
        final_metrics = {k: float(v[-1]) for k, v in history.history.items()}
        mlflow.log_metrics(final_metrics)
        logger.info("Final metrics: %s", json.dumps(final_metrics, indent=2))

        #  Save model 
        model_dir = Path(args.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        saved_model_path = str(model_dir / "road_seg_savedmodel")
        model.export(saved_model_path)
        logger.info("Model exported to %s", saved_model_path)

        # Save metrics.json for SageMaker Model Registry
        metrics_path = model_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_path))
        mlflow.tensorflow.log_model(model, artifact_path="model")

    logger.info("Training complete. Run ID: %s", run.info.run_id)


if __name__ == "__main__":
    main()
