"""
tests/unit/test_model.py
Tests model construction, output shapes, and custom metrics.
Downloads NO weights (backbone_weights=None) so these run fast without network.
"""
from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from src.models.metrics import BinaryIoU, CombinedLoss, DiceLoss
from src.models.resnet_unet import build_resnet_unet


# ─── Model construction ───────────────────────────────────────────────────────

class TestResNetUNet:
    @pytest.fixture(scope="class")
    def model(self):
        return build_resnet_unet(
            img_size=128,
            decoder_filters=[256, 128, 64, 32],
            dropout_rate=0.0,
            backbone_weights=None,
            freeze_encoder=False,
        )

    def test_output_shape(self, model):
        batch = np.random.rand(2, 128, 128, 3).astype(np.float32)
        out   = model(batch, training=False)
        assert out.shape == (2, 128, 128, 1), f"Got: {out.shape}"

    def test_output_range(self, model):
        batch = np.random.rand(1, 128, 128, 3).astype(np.float32)
        out   = model(batch, training=False).numpy()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_trainable_params(self, model):
        trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
        assert trainable > 0, "Model has no trainable parameters"

    def test_frozen_encoder(self):
        frozen_model = build_resnet_unet(
            img_size=64,
            decoder_filters=[64, 32, 16, 8],
            dropout_rate=0.0,
            backbone_weights=None,
            freeze_encoder=True,
        )
        # Decoder layers should still be trainable
        trainable_names = [v.name for v in frozen_model.trainable_variables]
        assert any("dec" in n for n in trainable_names), "Decoder is frozen — shouldn't be"


# ─── Metrics ─────────────────────────────────────────────────────────────────

class TestBinaryIoU:
    def test_perfect_prediction(self):
        metric = BinaryIoU(threshold=0.5)
        y_true = tf.constant([[[[1.0]], [[0.0]], [[1.0]]]])
        y_pred = tf.constant([[[[0.9]], [[0.1]], [[0.8]]]])
        metric.update_state(y_true, y_pred)
        assert float(metric.result()) == pytest.approx(1.0, abs=1e-4)

    def test_zero_overlap(self):
        metric = BinaryIoU(threshold=0.5)
        y_true = tf.constant([[[[1.0], [0.0]]]])
        y_pred = tf.constant([[[[0.1], [0.9]]]])
        metric.update_state(y_true, y_pred)
        assert float(metric.result()) == pytest.approx(0.0, abs=1e-4)

    def test_reset(self):
        metric = BinaryIoU()
        y = tf.constant([[[[1.0]]]])
        metric.update_state(y, y)
        metric.reset_state()
        metric.update_state(y, y)
        assert float(metric.result()) == pytest.approx(1.0, abs=1e-4)


class TestDiceLoss:
    def test_perfect_prediction(self):
        loss_fn = DiceLoss()
        y = tf.constant([1.0, 0.0, 1.0, 0.0])
        p = tf.constant([1.0, 0.0, 1.0, 0.0])
        val = float(loss_fn(y, p))
        assert val == pytest.approx(0.0, abs=1e-3)

    def test_worst_prediction(self):
        loss_fn = DiceLoss()
        y = tf.constant([1.0, 1.0])
        p = tf.constant([0.0, 0.0])
        val = float(loss_fn(y, p))
        assert val > 0.5


class TestCombinedLoss:
    def test_output_range(self):
        loss_fn = CombinedLoss()
        y_true  = tf.random.uniform((4, 64, 64, 1))
        y_pred  = tf.random.uniform((4, 64, 64, 1))
        val = float(loss_fn(y_true, y_pred))
        assert 0.0 <= val <= 2.0
