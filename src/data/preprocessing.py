"""src/data/preprocessing.py — dataset splitting and tf.data pipeline."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.utils.logging import get_logger

logger = get_logger(__name__)


def split_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train / val / test.

    Returns: X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    test_size = cfg.data.splits.test_size
    val_size = cfg.data.splits.val_size
    seed = cfg.data.splits.random_state

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )
    # val_size is fraction of the *full* dataset, convert to fraction of train_val
    adjusted_val = val_size / (1.0 - test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=adjusted_val, random_state=seed
    )
    logger.info(
        "Split — train: %d  val: %d  test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def _augment(image: tf.Tensor, mask: tf.Tensor, cfg: DictConfig) -> tuple[tf.Tensor, tf.Tensor]:
    aug = cfg.data.augmentation
    if aug.horizontal_flip:
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
    if aug.vertical_flip:
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
    low, high = aug.brightness_range
    image = tf.image.random_brightness(image, max_delta=(high - low) / 2)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, mask


def build_tf_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    cfg: DictConfig,
    training: bool = False,
) -> tf.data.Dataset:
    """Convert numpy arrays into a batched, optionally augmented tf.data.Dataset."""
    dl = cfg.data.dataloader
    ds = tf.data.Dataset.from_tensor_slices((X, Y))

    if training and cfg.data.augmentation.enabled:
        ds = ds.map(
            lambda x, y: _augment(x, y, cfg),
            num_parallel_calls=dl.num_parallel_calls,
        )
    if training and dl.shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    if dl.cache:
        ds = ds.cache()

    ds = ds.batch(dl.batch_size)
    ds = ds.prefetch(dl.prefetch)
    return ds
