"""
src/data/preprocessing.py
Handles:
 - Mask name resolution (KITTI naming convention)
 - Single-sample loading + resizing
 - Dataset-level loading from local dirs (post-download from S3)
 - Train/val split
 - Augmentation pipeline (albumentations)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


#  KITTI naming helpers 
def img_to_mask_name(img_path: str) -> str:
    """Convert image filename to corresponding mask filename."""
    img_path = Path(img_path)
    filename = img_path.name  
    prefix, rest = filename.split("_", 1)
    
    # Map image prefixes to mask prefixes
    if prefix == "um":
        # Try both lane and road masks for um_ images
        lane_mask = f"{prefix}_lane_{rest}"
        road_mask = f"{prefix}_road_{rest}"
        return [lane_mask, road_mask]  # Return both options
    elif prefix == "umm":
        return [f"{prefix}_road_{rest}"]
    elif prefix == "uu":
        return [f"{prefix}_road_{rest}"]
    else:
        return None


#  Augmentation pipeline 

def build_augmentation_pipeline(cfg: dict) -> A.Compose:
    transforms = []
    if cfg.get("horizontal_flip"):
        transforms.append(A.HorizontalFlip(p=0.5))
    br = cfg.get("brightness_range", [1.0, 1.0])
    cr = cfg.get("contrast_range",   [1.0, 1.0])
    if br != [1.0, 1.0] or cr != [1.0, 1.0]:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=(br[0] - 1, br[1] - 1),
                contrast_limit=(cr[0] - 1, cr[1] - 1),
                p=0.5,
            )
        )
    transforms.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4))
    transforms.append(A.GaussNoise(p=0.2))
    return A.Compose(transforms, additional_targets={"mask": "mask"})


#  Single sample 

def load_sample(
    img_path: str | Path,
    mask_path: str | Path,
    img_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and resize one (image, mask) pair.
    Returns:
        img  : (H, W, 3) float32 in [0, 1]
        mask : (H, W, 1) float32 in {0, 1}
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_bgr = cv2.imread(str(mask_path))
    road = (mask_bgr[:, :, 0] == 255).astype(np.float32)

    img  = cv2.resize(img,  (img_size, img_size))
    road = cv2.resize(road, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

    img  = img.astype(np.float32) / 255.0
    mask = np.expand_dims(road, axis=-1)

    return img, mask


#  Dataset builder 

def build_dataset(
    image_dir: str | Path,
    mask_dir:  str | Path,
    img_size:  int,
    augment:   bool = False,
    aug_cfg:   dict | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load every image from image_dir, pair with its mask, and return (X, Y).
    X: (N, H, W, 3)   float32 [0,1]
    Y: (N, H, W, 1)   float32 {0,1}
    """
    image_dir = Path(image_dir)
    mask_dir  = Path(mask_dir)

    img_names = sorted(os.listdir(image_dir))
    aug_fn = build_augmentation_pipeline(aug_cfg or {}) if augment else None

    X, Y = [], []
    skipped = 0

    for img_name in img_names:
        mask_candidates = img_to_mask_name(img_name)
        if mask_candidates is None:
            logger.warning("No mask available for %s — skipping", img_name)
            skipped += 1
            continue
            
        img_path = image_dir / img_name
        
        # Try to find any available mask
        mask_path = None
        for mask_name in mask_candidates:
            candidate_path = mask_dir / mask_name
            if candidate_path.exists():
                mask_path = candidate_path
                break
                
        if mask_path is None:
            logger.warning("Missing mask for %s — skipping", img_name)
            skipped += 1
            continue

        img, mask = load_sample(img_path, mask_path, img_size)

        if aug_fn is not None:
            aug = aug_fn(image=(img * 255).astype(np.uint8), mask=mask[:, :, 0])
            img  = aug["image"].astype(np.float32) / 255.0
            mask = np.expand_dims(aug["mask"], -1)

        X.append(img)
        Y.append(mask)

    logger.info("Loaded %d samples (%d skipped) from %s", len(X), skipped, image_dir)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def split_dataset(X: np.ndarray,Y: np.ndarray,val_split: float = 0.2,random_seed: int = 42,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, Y, test_size=val_split, random_state=random_seed)
