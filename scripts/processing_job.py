"""scripts/processing_job.py — SageMaker Processing Job entrypoint.

Reads raw KITTI images from /opt/ml/processing/input,
outputs train/val/test .npy arrays to separate output channels.

Run locally for testing:
    python scripts/processing_job.py \
        --input-dir data/raw \
        --output-dir data/processed \
        --img-size 256 \
        --test-size 0.2 \
        --val-size 0.1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

INPUT_DIR = "/opt/ml/processing/input"
OUTPUT_TRAIN = "/opt/ml/processing/train"
OUTPUT_VAL = "/opt/ml/processing/val"
OUTPUT_TEST = "/opt/ml/processing/test"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=INPUT_DIR)
    p.add_argument("--output-train", default=OUTPUT_TRAIN)
    p.add_argument("--output-val", default=OUTPUT_VAL)
    p.add_argument("--output-test", default=OUTPUT_TEST)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def mask_name(image_name: str) -> str:
    prefix, rest = image_name.split("_", 1)
    return f"{prefix}_road_{rest}"


def load_all(args: argparse.Namespace):
    image_dir = Path(args.input_dir) / "image_2"
    mask_dir = Path(args.input_dir) / "gt_image_2"
    img_size = args.img_size

    names = sorted(os.listdir(image_dir))
    print(f"Found {len(names)} images")

    X, Y = [], []
    for name in names:
        img = cv2.imread(str(image_dir / name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(str(mask_dir / mask_name(name)))
        road = (msk[:, :, 0] == 255).astype(np.float32)

        img = cv2.resize(img, (img_size, img_size)).astype(np.float32) / 255.0
        road = cv2.resize(road, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        X.append(img)
        Y.append(np.expand_dims(road, -1))

    return np.array(X), np.array(Y)


def save_split(X: np.ndarray, Y: np.ndarray, out_dir: str, tag: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.save(os.path.join(out_dir, f"X_{tag}.npy"), X)
    np.save(os.path.join(out_dir, f"Y_{tag}.npy"), Y)
    print(f"Saved {tag}: X={X.shape}  Y={Y.shape}  → {out_dir}")


def main() -> None:
    args = parse_args()
    print("Loading dataset...")
    X, Y = load_all(args)

    X_tv, X_test, Y_tv, Y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.random_state
    )
    adj_val = args.val_size / (1.0 - args.test_size)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tv, Y_tv, test_size=adj_val, random_state=args.random_state
    )

    save_split(X_train, Y_train, args.output_train, "train")
    save_split(X_val, Y_val, args.output_val, "val")
    save_split(X_test, Y_test, args.output_test, "test")
    print("Processing complete.")


if __name__ == "__main__":
    main()
