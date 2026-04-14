"""
src/models/metrics.py
Custom Keras metrics for segmentation tasks.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K


class BinaryIoU(tf.keras.metrics.Metric):
    """
    Intersection over Union for binary segmentation.
    Threshold defaults to 0.5.
    """

    def __init__(self, threshold: float = 0.5, name: str = "iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union        = self.add_weight(name="union",        initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred  = tf.cast(y_pred > self.threshold, tf.float32)
        y_true  = tf.cast(y_true, tf.float32)
        inter   = tf.reduce_sum(y_true * y_pred)
        union   = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
        self.intersection.assign_add(inter)
        self.union.assign_add(union + K.epsilon())

    def result(self):
        return self.intersection / self.union

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class DiceLoss(tf.keras.losses.Loss):
    """
    Soft Dice loss for binary segmentation.
    """

    def __init__(self, smooth: float = 1.0, name: str = "dice_loss"):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        inter  = tf.reduce_sum(y_true * y_pred)
        denom  = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth
        return 1.0 - (2.0 * inter + self.smooth) / denom


class CombinedLoss(tf.keras.losses.Loss):
    """BCE + Dice hybrid (often outperforms either alone for segmentation)."""

    def __init__(self, dice_weight: float = 0.5, name: str = "combined_loss"):
        super().__init__(name=name)
        self.dice_weight = dice_weight
        self._dice = DiceLoss()
        self._bce  = tf.keras.losses.BinaryCrossentropy()

    def call(self, y_true, y_pred):
        return (1 - self.dice_weight) * self._bce(y_true, y_pred) + \
               self.dice_weight * self._dice(y_true, y_pred)
