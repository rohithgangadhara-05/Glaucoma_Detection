import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pytest  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "models", "glaucoma_model.h5")
SPLITS_CSV = os.path.join(ROOT, "data", "splits.csv")
needs_model = pytest.mark.skipif(not os.path.isfile(MODEL_PATH), reason="models/glaucoma_model.h5 not found (run train.py)")
needs_data = pytest.mark.skipif(not os.path.isfile(SPLITS_CSV), reason="dataset not prepared (run fetch_data.py)")


@pytest.fixture(scope="session")
def model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)
