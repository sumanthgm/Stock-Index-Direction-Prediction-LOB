"""Unit tests for label construction, windowing and model forward passes (CPU, no data download)."""

import models
import numpy as np
import pytest
import torch
from nse_dataset import _make_labels, _make_windows


def test_labels_up_down_stationary_and_edges():
    k = 3
    up = _make_labels(np.linspace(100, 110, 30), k=k, alpha=1e-4)
    down = _make_labels(np.linspace(110, 100, 30), k=k, alpha=1e-4)
    flat = _make_labels(np.full(30, 100.0), k=k, alpha=1e-4)
    valid = slice(k - 1, 30 - k)
    assert set(up[valid]) == {2} and set(down[valid]) == {0} and set(flat[valid]) == {1}
    # edges without k past / k future points are marked -1
    assert (up[: k - 1] == -1).all() and (up[-k:] == -1).all()


def test_labels_never_cross_segment_boundary():
    mid = np.concatenate([np.full(20, 100.0), np.full(20, 200.0)])  # big jump at the "day" boundary
    seg = np.repeat([0, 1], 20)
    lab = _make_labels(mid, k=3, alpha=1e-4, seg=seg)
    assert set(lab[lab >= 0]) == {1}  # the jump must not create Up labels
    assert (lab[17:20] == -1).all()  # last k rows of segment 0


def test_windows_drop_cross_segment_and_invalid_labels():
    feats = np.arange(40, dtype=np.float32).reshape(20, 2)
    labels = np.ones(20, dtype=np.int8)
    labels[-1] = -1
    seg = np.repeat([0, 1], 10)
    X, y = _make_windows(feats, labels, seq_len=5, seg=seg)
    # 6 windows per segment, minus the one ending on the invalid last label
    assert X.shape == (11, 5, 2) and (y == 1).all()


@pytest.mark.parametrize("name", sorted(models.MODEL_REGISTRY))
def test_model_forward_shape(name):
    torch.manual_seed(0)
    model = models.build_model(name).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 100, 40))
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()
