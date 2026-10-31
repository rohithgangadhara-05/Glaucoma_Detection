"""Integrity of the prepared RIM-ONE DL dataset and of its official partitions (requires fetch_data.py to have run)."""
import csv
import os

from tests.conftest import ROOT, SPLITS_CSV, needs_data


def _rows():
    return list(csv.DictReader(open(SPLITS_CSV, encoding="utf-8")))


@needs_data
def test_dataset_counts_match_the_paper():
    rows = _rows()
    assert len(rows) == 485
    assert sum(int(r["label"]) for r in rows) == 172                      # 313 normal + 172 glaucoma


@needs_data
def test_official_partition_sizes():
    rows = _rows()
    count = lambda col, split, lab: sum(1 for r in rows if r[col] == split and int(r["label"]) == lab)  # noqa: E731
    assert (count("random_split", "test", 0), count("random_split", "test", 1)) == (94, 52)
    assert (count("hospital_split", "test", 0), count("hospital_split", "test", 1)) == (118, 56)


@needs_data
def test_every_image_is_in_exactly_one_set_and_exists():
    for col in ("random_split", "hospital_split"):
        assert all(r[col] in ("train", "test") for r in _rows())
    assert all(os.path.isfile(os.path.join(ROOT, "data", "raw", *r["file"].split("/"))) for r in _rows())


@needs_data
def test_label_matches_folder_and_no_synthetic_images():
    for r in _rows():
        folder, name = r["file"].split("/")
        assert folder == ("1_glaucoma" if r["label"] == "1" else "0_healthy")
        assert not name.startswith("synth")
