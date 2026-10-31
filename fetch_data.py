"""
fetch_data.py
=============
Downloads the public RIM-ONE DL retinal image database and prepares it for train.py and app.py.

RIM-ONE DL (Fumero Batista et al., Image Analysis & Stereology 39(3):161-167, 2020, doi:10.5566/ias.2346) contains
485 retinal fundus images cropped around the optic nerve head: 313 normal and 172 glaucoma, labelled by glaucoma experts
from three Spanish hospitals. Official page: https://github.com/miag-ull/rim-one-dl

LICENCE: the images may be used free of charge for research and education, but copying and redistribution are
prohibited. They are therefore NOT included in the project zip; this script downloads them from the official link.

Output:
    data/raw/0_healthy/   normal images     (label 0)
    data/raw/1_glaucoma/  glaucoma images   (label 1)
    data/splits.csv       file, label, random_split, hospital_split   (the two OFFICIAL train/test partitions)

If the automatic download fails (no internet / Google Drive quota), download the ZIP manually from
https://bit.ly/rim-one-dl-images, save it as data/RIM-ONE_DL_images.zip and run this script again.
"""

import csv
import logging
import os
import shutil
import sys
import zipfile

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_RAW = os.path.join(DATA_DIR, "raw")
HEALTHY = os.path.join(DATA_RAW, "0_healthy")
GLAUCOMA = os.path.join(DATA_RAW, "1_glaucoma")
SPLITS_CSV = os.path.join(DATA_DIR, "splits.csv")
ZIP_PATH = os.path.join(DATA_DIR, "RIM-ONE_DL_images.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "_extracted")

GDRIVE_FILE_ID = "1teYi_smpLiNZNJcTWdxXgKKLW2fkUQr4"   # target of https://bit.ly/rim-one-dl-images
EXPECTED = {"normal": 313, "glaucoma": 172}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_DIRS = {"normal": 0, "glaucoma": 1}


def _count(folder):
    return sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMG_EXTS) if os.path.isdir(folder) else 0


def download_zip():
    if os.path.isfile(ZIP_PATH) and zipfile.is_zipfile(ZIP_PATH):
        log.info("Using existing archive %s", ZIP_PATH)
        return True
    try:
        import gdown
    except ImportError:
        log.error("gdown is not installed. Run: pip install -r requirements.txt")
        return False
    os.makedirs(DATA_DIR, exist_ok=True)
    log.info("Downloading RIM-ONE DL from the official Google Drive link (about 100 MB) ...")
    try:
        gdown.download(id=GDRIVE_FILE_ID, output=ZIP_PATH, quiet=False)
    except Exception as exc:  # network errors, Drive quota, etc.
        log.error("Download failed: %s", exc)
        return False
    if not (os.path.isfile(ZIP_PATH) and zipfile.is_zipfile(ZIP_PATH)):
        log.error("The downloaded file is not a valid ZIP archive.")
        return False
    return True


def safe_extract(zip_path, dest):
    """Extract only image files, rejecting absolute paths and '..' (zip-slip protection)."""
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or os.path.splitext(name)[1].lower() not in IMG_EXTS:
                continue
            if name.startswith("/") or ".." in name.split("/"):
                log.warning("Skipping unsafe path in archive: %s", name)
                continue
            target = os.path.join(dest, *name.split("/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)


def scan_variant(variant_keyword):
    """Map file name -> (label, split) for one official partition (folders .../<variant>/<split>_set/<class>/img)."""
    result = {}
    for root, _, files in os.walk(EXTRACT_DIR):
        parts = [p.lower() for p in os.path.relpath(root, EXTRACT_DIR).split(os.sep)]
        if not any(variant_keyword in p for p in parts):
            continue
        cls = parts[-1]
        split = next(("train" if "train" in p else "test" for p in parts if p.endswith("_set") or p in ("training", "test")), None)
        if cls not in CLASS_DIRS or split is None:
            continue
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                result[f] = (CLASS_DIRS[cls], split, os.path.join(root, f))
    return result


def main():
    if _count(HEALTHY) >= EXPECTED["normal"] and _count(GLAUCOMA) >= EXPECTED["glaucoma"] and os.path.isfile(SPLITS_CSV):
        log.info("Dataset already prepared (%d normal, %d glaucoma). Nothing to do.", _count(HEALTHY), _count(GLAUCOMA))
        return
    if not download_zip():
        log.error("Could not obtain RIM-ONE DL automatically.\n"
                  "  1. Open https://bit.ly/rim-one-dl-images in a browser and download the ZIP.\n"
                  "  2. Save it as %s\n  3. Run: python fetch_data.py", ZIP_PATH)
        sys.exit(1)

    log.info("Extracting images ...")
    safe_extract(ZIP_PATH, EXTRACT_DIR)
    random_part = scan_variant("random")
    hospital_part = scan_variant("hospital")
    if not random_part:
        log.error("Unexpected archive layout: no 'partitioned_randomly' folders with normal/glaucoma images found.")
        sys.exit(1)

    for folder in (HEALTHY, GLAUCOMA):
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
    rows = []
    for fname in sorted(random_part):
        label, split, src = random_part[fname]
        dst_dir = GLAUCOMA if label == 1 else HEALTHY
        shutil.copy2(src, os.path.join(dst_dir, fname))
        h = hospital_part.get(fname)
        rows.append({"file": f"{os.path.basename(dst_dir)}/{fname}", "label": label, "random_split": split,
                     "hospital_split": h[1] if h else ""})
    with open(SPLITS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "label", "random_split", "hospital_split"])
        writer.writeheader()
        writer.writerows(rows)
    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)

    n0, n1 = _count(HEALTHY), _count(GLAUCOMA)
    log.info("Dataset ready: %d normal | %d glaucoma | splits -> %s", n0, n1, SPLITS_CSV)
    for name, key in (("random", "random_split"), ("hospital", "hospital_split")):
        counts = {}
        for r in rows:
            if r[key]:
                counts[(r[key], r["label"])] = counts.get((r[key], r["label"]), 0) + 1
        log.info("  %-8s partition: train %d normal / %d glaucoma | test %d normal / %d glaucoma", name,
                 counts.get(("train", 0), 0), counts.get(("train", 1), 0), counts.get(("test", 0), 0), counts.get(("test", 1), 0))
    if (n0, n1) != (EXPECTED["normal"], EXPECTED["glaucoma"]):
        log.warning("Expected %d normal and %d glaucoma images (RIM-ONE DL paper).", EXPECTED["normal"], EXPECTED["glaucoma"])


if __name__ == "__main__":
    main()
