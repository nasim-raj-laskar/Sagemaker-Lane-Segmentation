"""src/evaluation/evaluator.py — model evaluation on test split."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from omegaconf import DictConfig

from src.models.losses import DiceCoefficient, IoUScore
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Runs inference on a test dataset and computes segmentation metrics."""

    def __init__(self, model: tf.keras.Model, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self.threshold = cfg.inference.threshold

    def evaluate(self, X_test: np.ndarray, Y_test: np.ndarray) -> dict[str, float]:
        """Run full evaluation. Returns metrics dict."""
        logger.info("Evaluating on %d samples", len(X_test))
        preds = self.model.predict(X_test, batch_size=self.cfg.training.batch_size, verbose=1)

        iou = IoUScore(threshold=self.threshold)
        dice = DiceCoefficient(threshold=self.threshold)
        precision = tf.keras.metrics.Precision(thresholds=self.threshold)
        recall = tf.keras.metrics.Recall(thresholds=self.threshold)

        iou.update_state(Y_test, preds)
        dice.update_state(Y_test, preds)
        precision.update_state(Y_test, preds)
        recall.update_state(Y_test, preds)

        p = float(precision.result())
        r = float(recall.result())
        f1 = 2 * p * r / (p + r + 1e-6)

        metrics = {
            "iou_score": round(float(iou.result()), 4),
            "dice_coefficient": round(float(dice.result()), 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
        }
        logger.info("Evaluation results: %s", metrics)
        return metrics

    def save_report(self, metrics: dict, output_dir: str | Path) -> Path:
        """Write evaluation.json in the format SageMaker Model Monitor expects."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "metrics": {k: {"value": v} for k, v in metrics.items()},
            # Top-level iou_score for JsonGet condition in pipeline
            **{k: v for k, v in metrics.items()},
        }
        path = output_dir / "evaluation.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Evaluation report saved to %s", path)
        return path
