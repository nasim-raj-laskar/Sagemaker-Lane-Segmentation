"""
scripts/evaluate_job.py
SageMaker Processing evaluation step.
Loads best checkpoint, runs inference on val set, writes evaluation.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tarfile
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR   = Path("/opt/ml/processing/model")
VAL_DIR     = Path("/opt/ml/processing/input/val")
EVAL_DIR    = Path("/opt/ml/processing/evaluation")


def img_to_mask_name(name: str) -> str:
    prefix, rest = name.split("_", 1)
    return f"{prefix}_road_{rest}"


def load_model() -> tf.saved_model.load:
    # SageMaker stores model as model.tar.gz
    archives = list(MODEL_DIR.glob("*.tar.gz"))
    if archives:
        with tarfile.open(archives[0]) as t:
            t.extractall(MODEL_DIR)
    # Find SavedModel
    candidates = [MODEL_DIR / "road_seg_savedmodel", MODEL_DIR]
    for c in candidates:
        if c.exists():
            return tf.saved_model.load(str(c))
    raise FileNotFoundError("No SavedModel found in %s" % MODEL_DIR)


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return float(inter / (union + 1e-7))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img-size",   type=int,   default=256)
    p.add_argument("--threshold",  type=float, default=0.5)
    args = p.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model...")
    model = load_model()
    infer = model.signatures["serving_default"]

    img_dir  = VAL_DIR / "image"
    mask_dir = VAL_DIR / "mask"
    img_names = sorted(os.listdir(img_dir))
    logger.info("Evaluating on %d images", len(img_names))

    ious, accs = [], []

    for name in img_names:
        mask_name = img_to_mask_name(name)
        if not (mask_dir / mask_name).exists():
            continue

        img = cv2.imread(str(img_dir / name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.img_size, args.img_size))
        img_f = img.astype(np.float32) / 255.0

        mask_bgr = cv2.imread(str(mask_dir / mask_name))
        road = (mask_bgr[:, :, 0] == 255).astype(np.float32)
        road = cv2.resize(road, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)

        inp  = tf.constant(img_f[np.newaxis, ...], dtype=tf.float32)
        out  = infer(inp)
        pred = list(out.values())[0].numpy()[0, :, :, 0]
        pred_bin = (pred > args.threshold)
        road_bin = road.astype(bool)

        ious.append(compute_iou(road_bin, pred_bin))
        accs.append(compute_accuracy(road_bin, pred_bin))

    val_iou = float(np.mean(ious))
    val_acc = float(np.mean(accs))
    logger.info("val_iou=%.4f  val_accuracy=%.4f", val_iou, val_acc)

    report = {
        "metrics": {
            "val_iou":      {"value": val_iou},
            "val_accuracy": {"value": val_acc},
        }
    }

    out_path = EVAL_DIR / "evaluation.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Evaluation report written to %s", out_path)


if __name__ == "__main__":
    main()
