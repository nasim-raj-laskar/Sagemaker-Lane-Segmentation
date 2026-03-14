"""tests/unit/test_preprocessing.py"""
import numpy as np
import pytest

from src.utils.config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config(overrides=["training.epochs=2", "data.dataloader.batch_size=2"])


def _dummy_arrays(n=10, h=256, w=256):
    X = np.random.rand(n, h, w, 3).astype(np.float32)
    Y = np.random.randint(0, 2, (n, h, w, 1)).astype(np.float32)
    return X, Y


class TestSplitDataset:
    def test_split_sizes(self, cfg):
        from src.data.preprocessing import split_dataset
        X, Y = _dummy_arrays(50)
        X_tr, X_v, X_te, Y_tr, Y_v, Y_te = split_dataset(X, Y, cfg)
        assert len(X_tr) + len(X_v) + len(X_te) == 50
        assert X_tr.shape[1:] == (256, 256, 3)

    def test_no_data_leakage(self, cfg):
        from src.data.preprocessing import split_dataset
        X, Y = _dummy_arrays(30)
        X_tr, X_v, X_te, *_ = split_dataset(X, Y, cfg)
        # Check no array appears in two splits (by identity)
        assert not any(np.array_equal(a, b) for a in X_tr for b in X_te)


class TestBuildTfDataset:
    def test_batch_shape(self, cfg):
        from src.data.preprocessing import build_tf_dataset
        X, Y = _dummy_arrays(8)
        ds = build_tf_dataset(X, Y, cfg, training=False)
        batch_x, batch_y = next(iter(ds))
        assert batch_x.shape[0] == cfg.data.dataloader.batch_size
        assert batch_x.shape[1:] == (256, 256, 3)

    def test_pixel_range(self, cfg):
        from src.data.preprocessing import build_tf_dataset
        X, Y = _dummy_arrays(4)
        ds = build_tf_dataset(X, Y, cfg, training=False)
        batch_x, _ = next(iter(ds))
        assert float(batch_x.numpy().min()) >= 0.0
        assert float(batch_x.numpy().max()) <= 1.0
