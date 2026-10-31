"""Tests for fetch_data.py: safe extraction and reading of the official partitions."""
import zipfile

import fetch_data


def test_safe_extract_keeps_images_and_blocks_unsafe_paths(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("RIM-ONE_DL_images/partitioned_randomly/test_set/normal/r1_Im001.png", b"png-bytes")
        z.writestr("../evil.png", b"x")                  # path traversal attempt
        z.writestr("readme.txt", b"not an image")        # non-image file
    dest = tmp_path / "out"
    fetch_data.safe_extract(str(archive), str(dest))
    files = [p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()]
    assert files == ["RIM-ONE_DL_images/partitioned_randomly/test_set/normal/r1_Im001.png"]
    assert not (tmp_path / "evil.png").exists()


def test_scan_variant_reads_labels_and_splits(tmp_path, monkeypatch):
    layout = {
        "partitioned_randomly/training_set/normal/a.png": (0, "train"),
        "partitioned_randomly/training_set/glaucoma/b.png": (1, "train"),
        "partitioned_randomly/test_set/glaucoma/c.png": (1, "test"),
        "partitioned_by_hospital/test_set/normal/a.png": (0, "test"),
    }
    for rel in layout:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    monkeypatch.setattr(fetch_data, "EXTRACT_DIR", str(tmp_path))
    random_part = fetch_data.scan_variant("random")
    hospital_part = fetch_data.scan_variant("hospital")
    assert {k: v[:2] for k, v in random_part.items()} == {"a.png": (0, "train"), "b.png": (1, "train"), "c.png": (1, "test")}
    assert {k: v[:2] for k, v in hospital_part.items()} == {"a.png": (0, "test")}
