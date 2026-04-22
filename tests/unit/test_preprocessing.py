"""
tests/unit/test_preprocessing.py
Unit tests for data loading and preprocessing utilities.
Uses synthetic numpy data — no S3 or real files required.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.preprocessing import (
    build_dataset,
    img_to_mask_name,
    load_sample,
    split_dataset,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def synthetic_dataset(tmp_path: Path):
    """
    Creates a tiny synthetic KITTI-style dataset:
      image_dir/ um_000000.png um_000001.png um_000002.png
      mask_dir/  um_road_000000.png ...
    """
    img_dir  = tmp_path / "image"
    mask_dir = tmp_path / "mask"
    img_dir.mkdir()
    mask_dir.mkdir()

    for i in range(6):
        name      = f"um_{i:06d}.png"
        mask_name = f"um_road_{i:06d}.png"

        # Random BGR image 375×1242
        img  = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        # Mask: red channel == 255 for road pixels
        mask = np.zeros((375, 1242, 3), dtype=np.uint8)
        mask[:100, :, 0] = 255  # top strip is "road"

        cv2.imwrite(str(img_dir  / name),      img)
        cv2.imwrite(str(mask_dir / mask_name), mask)

    return img_dir, mask_dir


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestMaskNaming:
    def test_um_prefix(self):
        assert img_to_mask_name("um_000000.png") == "um_road_000000.png"

    def test_umm_prefix(self):
        assert img_to_mask_name("umm_000042.png") == "umm_road_000042.png"

    def test_uu_prefix(self):
        assert img_to_mask_name("uu_000010.png") == "uu_road_000010.png"


class TestLoadSample:
    def test_output_shapes(self, synthetic_dataset):
        img_dir, mask_dir = synthetic_dataset
        name = "um_000000.png"
        img, mask = load_sample(
            img_dir / name,
            mask_dir / f"um_road_000000.png",
            img_size=128,
        )
        assert img.shape  == (128, 128, 3), f"img shape: {img.shape}"
        assert mask.shape == (128, 128, 1), f"mask shape: {mask.shape}"

    def test_image_range(self, synthetic_dataset):
        img_dir, mask_dir = synthetic_dataset
        img, _ = load_sample(img_dir / "um_000000.png", mask_dir / "um_road_000000.png", 64)
        assert img.min() >= 0.0
        assert img.max() <= 1.0
        assert img.dtype == np.float32

    def test_mask_binary(self, synthetic_dataset):
        img_dir, mask_dir = synthetic_dataset
        _, mask = load_sample(img_dir / "um_000000.png", mask_dir / "um_road_000000.png", 64)
        unique = np.unique(mask)
        assert set(unique).issubset({0.0, 1.0}), f"Non-binary mask values: {unique}"


class TestBuildDataset:
    def test_dataset_size(self, synthetic_dataset):
        img_dir, mask_dir = synthetic_dataset
        X, Y = build_dataset(img_dir, mask_dir, img_size=64)
        assert X.shape[0] == 6
        assert Y.shape[0] == 6

    def test_dataset_shape(self, synthetic_dataset):
        img_dir, mask_dir = synthetic_dataset
        X, Y = build_dataset(img_dir, mask_dir, img_size=64)
        assert X.shape[1:] == (64, 64, 3)
        assert Y.shape[1:] == (64, 64, 1)

    def test_dataset_dtypes(self, synthetic_dataset):
        img_dir, mask_dir = synthetic_dataset
        X, Y = build_dataset(img_dir, mask_dir, img_size=64)
        assert X.dtype == np.float32
        assert Y.dtype == np.float32

    def test_missing_mask_skipped(self, synthetic_dataset):
        img_dir, mask_dir = synthetic_dataset
        # Add an image with no matching mask
        extra = img_dir / "um_999999.png"
        cv2.imwrite(str(extra), np.zeros((100, 100, 3), dtype=np.uint8))
        # Should load 6 valid + skip 1
        X, Y = build_dataset(img_dir, mask_dir, img_size=32)
        assert X.shape[0] == 6


class TestSplitDataset:
    def test_split_sizes(self):
        X = np.random.rand(100, 64, 64, 3).astype(np.float32)
        Y = np.random.rand(100, 64, 64, 1).astype(np.float32)
        X_train, X_val, Y_train, Y_val = split_dataset(X, Y, val_split=0.2)
        assert X_train.shape[0] == 80
        assert X_val.shape[0]   == 20
        assert Y_train.shape[0] == 80
        assert Y_val.shape[0]   == 20

    def test_split_reproducible(self):
        X = np.arange(50).reshape(50, 1).astype(np.float32)
        Y = X.copy()
        split_a = split_dataset(X, Y, val_split=0.2, random_seed=42)
        split_b = split_dataset(X, Y, val_split=0.2, random_seed=42)
        np.testing.assert_array_equal(split_a[0], split_b[0])
