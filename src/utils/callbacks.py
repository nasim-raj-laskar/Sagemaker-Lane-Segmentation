"""
src/utils/callbacks.py
Training callbacks: checkpoint, early stopping, reduce LR, MLflow logger.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class MLflowMetricsCallback(tf.keras.callbacks.Callback):
    """Log per-epoch metrics to MLflow (works even if MLflow URI is not set)."""

    def __init__(self, run_id: str | None = None):
        super().__init__()
        self.run_id = run_id

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        if logs is None:
            return
        try:
            mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)
        except Exception as exc:
            logger.warning("MLflow log_metrics failed: %s", exc)


class BestModelS3Uploader(tf.keras.callbacks.Callback):
    """
    After each epoch, if val_iou improved, write a signal file so the
    post-training script knows to upload the checkpoint to S3.
    (Actual S3 upload happens in the finalizer, not per-epoch, to avoid
     hammering the API on every epoch.)
    """

    def __init__(self, output_dir: str | Path, monitor: str = "val_iou"):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.monitor    = monitor
        self.best       = -np.inf

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        current = (logs or {}).get(self.monitor, -np.inf)
        if current > self.best:
            self.best = current
            flag = self.output_dir / ".best_epoch"
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.write_text(f"epoch={epoch}  {self.monitor}={current:.4f}\n")


def build_callbacks(
    cfg,
    model_dir: str | Path,
    output_dir: str | Path,
    mlflow_run_id: str | None = None,
) -> list[tf.keras.callbacks.Callback]:
    """Assemble all callbacks from config."""
    model_dir  = Path(model_dir)
    output_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[tf.keras.callbacks.Callback] = []

    # 1. Model checkpoint
    ckpt_cfg = cfg.training.checkpoint
    callbacks.append(
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_checkpoint.keras"),
            monitor=ckpt_cfg.get("monitor", "val_iou"),
            save_best_only=ckpt_cfg.get("save_best_only", True),
            mode="max",
            verbose=1,
        )
    )

    # 2. Early stopping
    es_cfg = cfg.training.early_stopping
    if es_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=es_cfg.get("monitor", "val_iou"),
                patience=es_cfg.get("patience", 8),
                mode=es_cfg.get("mode", "max"),
                restore_best_weights=True,
                verbose=1,
            )
        )

    # 3. ReduceLROnPlateau
    rlr_cfg = cfg.training.reduce_lr
    if rlr_cfg.get("enabled", True):
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=rlr_cfg.get("factor", 0.5),
                patience=rlr_cfg.get("patience", 4),
                min_lr=rlr_cfg.get("min_lr", 1e-7),
                verbose=1,
            )
        )

    # 4. CSV logger (for SageMaker metrics regex)
    callbacks.append(
        tf.keras.callbacks.CSVLogger(str(output_dir / "training_log.csv"))
    )

    # 5. TensorBoard
    tb_dir = str(output_dir / "tensorboard")
    callbacks.append(
        tf.keras.callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=0)
    )

    # 6. MLflow per-epoch logger
    callbacks.append(MLflowMetricsCallback(run_id=mlflow_run_id))

    # 7. Best-model S3 upload signal
    callbacks.append(BestModelS3Uploader(output_dir=output_dir))

    return callbacks
