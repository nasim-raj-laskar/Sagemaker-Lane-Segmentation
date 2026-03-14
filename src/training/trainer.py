"""src/training/trainer.py — training loop, callbacks, MLflow integration."""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.tensorflow
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from src.models.losses import bce_dice_loss, get_metrics
from src.models.resnet_unet import build_resnet_unet, unfreeze_encoder
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _build_optimizer(cfg: DictConfig) -> tf.keras.optimizers.Optimizer:
    opt_cfg = cfg.training.optimizer
    return tf.keras.optimizers.Adam(
        learning_rate=opt_cfg.learning_rate,
        beta_1=opt_cfg.beta_1,
        beta_2=opt_cfg.beta_2,
        epsilon=opt_cfg.epsilon,
    )


def _build_callbacks(cfg: DictConfig) -> list[tf.keras.callbacks.Callback]:
    tc = cfg.training.callbacks
    cbs = []

    if tc.early_stopping.enabled:
        cbs.append(tf.keras.callbacks.EarlyStopping(
            monitor=tc.early_stopping.monitor,
            patience=tc.early_stopping.patience,
            mode=tc.early_stopping.mode,
            restore_best_weights=tc.early_stopping.restore_best_weights,
            verbose=1,
        ))

    if tc.reduce_lr.enabled:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=tc.reduce_lr.monitor,
            factor=tc.reduce_lr.factor,
            patience=tc.reduce_lr.patience,
            min_lr=tc.reduce_lr.min_lr,
            verbose=1,
        ))

    if tc.model_checkpoint.enabled:
        Path(tc.model_checkpoint.filepath).parent.mkdir(parents=True, exist_ok=True)
        cbs.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=tc.model_checkpoint.filepath,
            monitor=tc.model_checkpoint.monitor,
            mode=tc.model_checkpoint.mode,
            save_best_only=tc.model_checkpoint.save_best_only,
            verbose=1,
        ))

    if tc.tensorboard.enabled:
        Path(tc.tensorboard.log_dir).mkdir(parents=True, exist_ok=True)
        cbs.append(tf.keras.callbacks.TensorBoard(
            log_dir=tc.tensorboard.log_dir,
            histogram_freq=tc.tensorboard.histogram_freq,
        ))

    return cbs


class Trainer:
    """Orchestrates model build → compile → fit → fine-tune → MLflow logging."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.model: tf.keras.Model | None = None

    def setup(self) -> None:
        self.model = build_resnet_unet(self.cfg)
        self.model.compile(
            optimizer=_build_optimizer(self.cfg),
            loss=bce_dice_loss(self.cfg),
            metrics=get_metrics(self.cfg),
        )

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
    ) -> mlflow.ActiveRun:
        cfg = self.cfg
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        with mlflow.start_run(run_name=cfg.mlflow.run_name, tags=dict(cfg.mlflow.tags)) as run:
            # Log full resolved config as artifact
            mlflow.log_text(OmegaConf.to_yaml(cfg), "config.yaml")
            mlflow.log_params({
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "learning_rate": cfg.training.optimizer.learning_rate,
                "backbone": cfg.model.backbone.name,
                "img_size": cfg.data.preprocessing.img_size,
                "loss": cfg.training.loss.name,
            })

            logger.info("Phase 1: training with frozen encoder")
            history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg.training.epochs,
                callbacks=_build_callbacks(cfg),
            )
            self._log_history(history, prefix="frozen")

            # ── Fine-tuning phase ────────────────────────────────────────────
            if cfg.training.fine_tuning.enabled:
                logger.info("Phase 2: fine-tuning encoder from %s", cfg.model.fine_tuning.unfreeze_from_layer)
                self.model = unfreeze_encoder(self.model, cfg)
                self.model.compile(
                    optimizer=tf.keras.optimizers.Adam(cfg.training.fine_tuning.learning_rate),
                    loss=bce_dice_loss(cfg),
                    metrics=get_metrics(cfg),
                )
                ft_history = self.model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=cfg.training.fine_tuning.epochs,
                    callbacks=_build_callbacks(cfg),
                )
                self._log_history(ft_history, prefix="finetune")

            if cfg.mlflow.log_model:
                mlflow.tensorflow.log_model(self.model, "model")

        return run

    def _log_history(self, history, prefix: str = "") -> None:
        for metric, values in history.history.items():
            for step, value in enumerate(values):
                mlflow.log_metric(f"{prefix}_{metric}" if prefix else metric, value, step=step)

    def save(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "model.keras"
        self.model.save(str(path))
        logger.info("Model saved to %s", path)
        return path
