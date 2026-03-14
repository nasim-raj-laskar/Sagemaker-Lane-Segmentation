"""tests/unit/test_losses.py"""
import numpy as np
import pytest
import tensorflow as tf

from src.models.losses import DiceCoefficient, IoUScore, dice_loss


def _ones(shape=(2, 4, 4, 1)):
    return tf.ones(shape, dtype=tf.float32)


def _zeros(shape=(2, 4, 4, 1)):
    return tf.zeros(shape, dtype=tf.float32)


class TestDiceLoss:
    def test_perfect_prediction_is_zero(self):
        y = _ones()
        assert float(dice_loss(y, y)) == pytest.approx(0.0, abs=1e-4)

    def test_worst_prediction_is_one(self):
        assert float(dice_loss(_ones(), _zeros())) == pytest.approx(1.0, abs=1e-3)

    def test_range(self):
        y_true = tf.constant([[[1.0]], [[0.0]]])
        y_pred = tf.constant([[[0.8]], [[0.3]]])
        loss = float(dice_loss(y_true, y_pred))
        assert 0.0 <= loss <= 1.0


class TestIoUScore:
    def test_perfect_score(self):
        m = IoUScore(threshold=0.5)
        m.update_state(_ones(), _ones())
        assert float(m.result()) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        m = IoUScore(threshold=0.5)
        m.update_state(_ones(), _zeros())
        assert float(m.result()) == pytest.approx(0.0, abs=1e-4)

    def test_reset(self):
        m = IoUScore(threshold=0.5)
        m.update_state(_ones(), _ones())
        m.reset_state()
        m.update_state(_ones(), _zeros())
        assert float(m.result()) == pytest.approx(0.0, abs=1e-4)


class TestDiceCoefficient:
    def test_perfect_score(self):
        m = DiceCoefficient(threshold=0.5)
        m.update_state(_ones(), _ones())
        assert float(m.result()) == pytest.approx(1.0, abs=1e-4)
