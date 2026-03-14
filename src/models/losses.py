"""src/models/losses.py — custom losses and metrics for binary segmentation."""

from __future__ import annotations

import tensorflow as tf
from omegaconf import DictConfig


# ── Losses ───────────────────────────────────────────────────────────────────

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def bce_dice_loss(cfg: DictConfig):
    """Combined binary cross-entropy + Dice loss, weights from config."""
    bce_w = cfg.training.loss.bce_weight
    dice_w = cfg.training.loss.dice_weight

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return bce_w * tf.reduce_mean(bce) + dice_w * dice_loss(y_true, y_pred)

    loss.__name__ = "bce_dice_loss"
    return loss


# ── Metrics ──────────────────────────────────────────────────────────────────

class IoUScore(tf.keras.metrics.Metric):
    """Intersection-over-Union for binary segmentation masks."""

    def __init__(self, threshold: float = 0.5, name: str = "iou_score", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred > self.threshold, tf.float32)
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.reshape(y_pred_bin, [-1])
        self.intersection.assign_add(tf.reduce_sum(y_true_f * y_pred_f))
        self.union.assign_add(
            tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - tf.reduce_sum(y_true_f * y_pred_f)
        )

    def result(self):
        return self.intersection / (self.union + 1e-6)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, threshold: float = 0.5, name: str = "dice_coefficient", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.numerator = self.add_weight(name="num", initializer="zeros")
        self.denominator = self.add_weight(name="den", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred > self.threshold, tf.float32)
        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.reshape(y_pred_bin, [-1])
        self.numerator.assign_add(2.0 * tf.reduce_sum(y_true_f * y_pred_f))
        self.denominator.assign_add(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))

    def result(self):
        return self.numerator / (self.denominator + 1e-6)

    def reset_state(self):
        self.numerator.assign(0.0)
        self.denominator.assign(0.0)


def get_metrics(cfg: DictConfig) -> list:
    return [
        "accuracy",
        IoUScore(threshold=cfg.inference.threshold),
        DiceCoefficient(threshold=cfg.inference.threshold),
        tf.keras.metrics.Precision(thresholds=cfg.inference.threshold, name="precision"),
        tf.keras.metrics.Recall(thresholds=cfg.inference.threshold, name="recall"),
    ]
