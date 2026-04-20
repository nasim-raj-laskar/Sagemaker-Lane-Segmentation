"""
scripts/preprocess_job.py
Runs inside a SageMaker SKLearn Processing container.
Reads raw images/masks from /opt/ml/processing/input/,
splits into train/val, saves to /opt/ml/processing/output/.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INPUT_IMAGE = Path("/opt/ml/processing/input/image")
INPUT_MASK  = Path("/opt/ml/processing/input/mask")
OUTPUT_TRAIN = Path("/opt/ml/processing/output/train")
OUTPUT_VAL   = Path("/opt/ml/processing/output/val")


def img_to_mask_name(img_name: str) -> str:
    prefix, rest = img_name.split("_", 1)
    return f"{prefix}_road_{rest}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img-size",  type=int,   default=256)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed",      type=int,   default=42)
    args = p.parse_args()

    img_names = sorted(os.listdir(INPUT_IMAGE))
    logger.info("Found %d images", len(img_names))

    # Filter to images that have a mask
    valid = [n for n in img_names if (INPUT_MASK / img_to_mask_name(n)).exists()]
    logger.info("%d images have matching masks", len(valid))

    train_names, val_names = train_test_split(valid, test_size=args.val_split, random_state=args.seed)
    logger.info("Split: train=%d  val=%d", len(train_names), len(val_names))

    for split_names, out_dir in [(train_names, OUTPUT_TRAIN), (val_names, OUTPUT_VAL)]:
        img_out  = out_dir / "image"
        mask_out = out_dir / "mask"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for name in split_names:
            # Copy original files (training script does resize on-the-fly)
            shutil.copy2(INPUT_IMAGE / name, img_out / name)
            shutil.copy2(INPUT_MASK / img_to_mask_name(name), mask_out / img_to_mask_name(name))

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
