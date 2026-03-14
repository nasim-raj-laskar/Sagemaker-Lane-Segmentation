"""src/data/loader.py — dataset loading from local disk or S3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from omegaconf import DictConfig

from src.utils.logging import get_logger
from src.utils.s3 import S3Client

logger = get_logger(__name__)


def _mask_name_from_image(image_name: str) -> str:
    """Convert image filename to corresponding mask filename.

    Example: um_000000.png -> um_road_000000.png
    """
    prefix, rest = image_name.split("_", 1)
    return f"{prefix}_road_{rest}"


def load_sample(
    image_name: str,
    image_dir: str | Path,
    mask_dir: str | Path,
    img_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a single (image, mask) pair, resize, and normalise mask.

    Returns:
        img:  float32 array (H, W, 3) — NOT normalised (caller handles /255)
        mask: float32 array (H, W, 1) — binary {0.0, 1.0}
    """
    image_dir, mask_dir = Path(image_dir), Path(mask_dir)
    mask_name = _mask_name_from_image(image_name)

    img = cv2.imread(str(image_dir / image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_bgr = cv2.imread(str(mask_dir / mask_name))
    road = (mask_bgr[:, :, 0] == 255).astype(np.float32)

    img = cv2.resize(img, (img_size, img_size)).astype(np.float32)
    mask = cv2.resize(road, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask = np.expand_dims(mask, axis=-1)

    return img, mask


def load_dataset(cfg: DictConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load the full dataset from local disk.

    Returns:
        X: float32 array (N, H, W, 3) — pixel values in [0, 1]
        Y: float32 array (N, H, W, 1) — binary masks
    """
    image_dir = cfg.data.local.image_dir
    mask_dir = cfg.data.local.mask_dir
    img_size = cfg.data.preprocessing.img_size

    image_names = sorted(os.listdir(image_dir))
    logger.info("Loading %d samples from %s", len(image_names), image_dir)

    X, Y = [], []
    for name in image_names:
        img, mask = load_sample(name, image_dir, mask_dir, img_size)
        X.append(img)
        Y.append(mask)

    X = np.array(X) / 255.0
    Y = np.array(Y)
    logger.info("Dataset loaded — X: %s  Y: %s", X.shape, Y.shape)
    return X, Y


def stream_from_s3(cfg: DictConfig) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Stream (image, mask) pairs directly from S3 without full download.

    Useful for large datasets that don't fit in local storage.
    """
    s3 = S3Client(cfg)
    bucket = cfg.data.s3.bucket
    img_prefix = cfg.data.s3.raw_prefix + "/image_2/"
    mask_prefix = cfg.data.s3.raw_prefix + "/gt_image_2/"
    img_size = cfg.data.preprocessing.img_size

    image_keys = s3.list_keys(bucket, img_prefix)
    logger.info("Streaming %d samples from s3://%s/%s", len(image_keys), bucket, img_prefix)

    for key in image_keys:
        name = Path(key).name
        img_bytes = s3.read_bytes(bucket, key)
        mask_key = mask_prefix + _mask_name_from_image(name)
        mask_bytes = s3.read_bytes(bucket, mask_key)

        img_arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        mask_arr = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_COLOR)

        road = (mask_arr[:, :, 0] == 255).astype(np.float32)
        img_arr = cv2.resize(img_arr, (img_size, img_size)).astype(np.float32) / 255.0
        mask_resized = cv2.resize(road, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        yield img_arr, np.expand_dims(mask_resized, axis=-1)
