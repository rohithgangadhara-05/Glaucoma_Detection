"""Tests for the model definition and the evaluation helpers in train.py, and for the saved results."""
import csv
import json
import os

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import train
from tests.conftest import ROOT, needs_data, needs_model


def test_threshold_metrics_on_a_known_example():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.3, 0.4])
    m = train.threshold_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(4 / 5)
    assert m["accuracy"] == pytest.approx(6 / 8) and m["precision"] == pytest.approx(2 / 3)


def test_sensitivity_at_specificity():
    y = np.array([0] * 20 + [1] * 10)
    p = np.concatenate([np.linspace(0, 0.5, 20), np.linspace(0.45, 1.0, 10)])
    se = train.sensitivity_at_specificity(y, p, 0.85)
    assert 0.0 <= se <= 1.0
    assert train.sensitivity_at_specificity(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_bootstrap_interval_contains_the_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.4 + rng.random(200) * 0.6, 0, 1)
    lo, hi = train.bootstrap_auc_ci(y, p, n=300)
    assert lo <= roc_auc_score(y, p) <= hi


def test_model_architecture():
    model, base = train.build_model()
    assert model.input_shape == (None, 224, 224, 3) and model.output_shape == (None, 1)
    assert model.count_params() == 2_264_385
    assert base.trainable is False                                   # phase 1: backbone frozen
    rescale = model.get_layer("to_minus1_plus1")
    assert np.allclose(rescale(np.array([[0.0, 0.5, 1.0]])).numpy(), [[-1.0, 0.0, 1.0]])


@needs_model
def test_saved_metrics_are_consistent():
    m = json.load(open(os.path.join(ROOT, "models", "metrics.json"), encoding="utf-8"))
    t = m["at_threshold_0.5"]
    assert t["tp"] + t["fn"] == m["test_glaucoma"] and t["tn"] + t["fp"] == m["test_normal"]
    assert m["auc_95ci"][0] <= m["auc"] <= m["auc_95ci"][1]


@needs_model
@needs_data
def test_saved_predictions_reproduce_from_the_model(model):
    from utils.preprocessing import preprocess
    rows = list(csv.DictReader(open(os.path.join(ROOT, "models", "test_predictions.csv"), encoding="utf-8")))[:16]
    x = np.stack([preprocess(os.path.join(ROOT, "data", "raw", *r["file"].split("/"))) for r in rows])
    p = model.predict(x, verbose=0).ravel()
    assert np.allclose(p, [float(r["p_glaucoma"]) for r in rows], atol=1e-4)
