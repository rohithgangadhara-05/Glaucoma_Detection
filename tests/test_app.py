"""End-to-end tests of the Streamlit app with streamlit.testing (no browser needed)."""
import csv
import os
import re

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT, needs_data, needs_model

APP = os.path.join(ROOT, "app.py")


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=240).run()


@needs_model
def test_app_starts_without_errors(app):
    assert not app.exception
    assert any("Glaucoma Detection AI" in m.value for m in app.markdown)
    assert any("Upload a retinal image" in i.value for i in app.info)


@needs_model
@needs_data
def test_app_result_matches_saved_prediction_for_test_images(app):
    saved = {os.path.basename(r["file"]): (float(r["p_glaucoma"]), int(r["label"]))
             for r in csv.DictReader(open(os.path.join(ROOT, "models", "test_predictions.csv"), encoding="utf-8"))}
    options = app.selectbox[0].options[1:]
    assert options and all(o.startswith("Test set") for o in options)
    for choice in (options[0], options[-1]):
        app.selectbox[0].set_value(choice).run()
        assert not app.exception
        name = choice.split(": ")[1]
        shown = float(re.search(r"P\(glaucoma\) = ([0-9.]+)", " ".join(m.value for m in app.markdown)).group(1))
        assert shown == pytest.approx(saved[name][0], abs=1e-3)
        expert = "Glaucoma" if saved[name][1] else "Normal"
        assert any(f"Expert label in RIM-ONE DL: **{expert}**" in c.value for c in app.caption)
        assert len(app.metric) == 3
