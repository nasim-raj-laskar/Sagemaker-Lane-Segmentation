"""scripts/evaluate.py — SageMaker Processing Job for model evaluation.

Called by the pipeline EvaluationStep. Writes evaluation.json to
/opt/ml/processing/evaluation/ for the ConditionStep to read via JsonGet.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

MODEL_DIR = "/opt/ml/processing/model"
TEST_DIR = "/opt/ml/processing/test"
OUTPUT_DIR = "/opt/ml/processing/evaluation"
THRESHOLD = float(os.environ.get("MASK_THRESHOLD", "0.5"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=MODEL_DIR)
    p.add_argument("--test-dir", default=TEST_DIR)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    return p.parse_args()


def iou(y_true, y_pred, threshold):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    intersection = np.sum(y_true * y_pred_bin)
    union = np.sum(y_true) + np.sum(y_pred_bin) - intersection
    return float(intersection / (union + 1e-6))


def dice(y_true, y_pred, threshold):
    y_pred_bin = (y_pred > threshold).astype(np.float32)
    num = 2.0 * np.sum(y_true * y_pred_bin)
    den = np.sum(y_true) + np.sum(y_pred_bin)
    return float(num / (den + 1e-6))


def main():
    args = parse_args()

    print("Loading model...")
    model = tf.keras.models.load_model(
        os.path.join(args.model_dir, "model.keras"), compile=False
    )

    print("Loading test data...")
    X_test = np.load(os.path.join(args.test_dir, "X_test.npy"))
    Y_test = np.load(os.path.join(args.test_dir, "Y_test.npy"))
    print(f"Test set: {X_test.shape}")

    print("Running predictions...")
    preds = model.predict(X_test, batch_size=8, verbose=1)

    y_true_flat = Y_test.reshape(-1)
    y_pred_flat = preds.reshape(-1)

    iou_score = iou(y_true_flat, y_pred_flat, args.threshold)
    dice_score = dice(y_true_flat, y_pred_flat, args.threshold)
    y_pred_bin = (y_pred_flat > args.threshold).astype(np.float32)
    tp = np.sum(y_true_flat * y_pred_bin)
    p = float(tp / (np.sum(y_pred_bin) + 1e-6))
    r = float(tp / (np.sum(y_true_flat) + 1e-6))
    f1 = 2 * p * r / (p + r + 1e-6)

    metrics = {
        "iou_score": round(iou_score, 4),
        "dice_coefficient": round(dice_score, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    report = {
        "metrics": {k: {"value": v} for k, v in metrics.items()},
        **metrics,   # top-level for JsonGet in pipeline ConditionStep
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
