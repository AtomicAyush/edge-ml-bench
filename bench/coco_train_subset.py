"""Download a fixed random subset of COCO train2017 for quantization-aware training.

    python -m bench.coco_train_subset                  # 5,000 images, seed 0 -> coco-train5k (~800 MB)
    python -m bench.coco_train_subset --n 64 --name coco-train-smoke \\
        --root models/_smoke_qat --yaml models/_smoke_qat/coco-train-smoke.yaml

bench/qat.py fine-tunes YOLOv8n with fake quantization and needs training images.
Ultralytics' coco.yaml downloads all of train2017 (19 GB, 118,287 images); QAT at
a small learning rate needs a few thousand. This shuffles the train2017 file list
with a fixed seed and keeps the first n, so a smaller subset is always a prefix of
a larger one. Labels come from Ultralytics' coco2017labels.zip (the same YOLO-format
labels bench.coco_val uses); each image is fetched on its own from
images.cocodataset.org. Images already on disk are skipped, so an interrupted run
resumes where it stopped.

The subset is a separate dataset, <root>/<name>/{images,labels}/train2017 plus a
train.txt list, with its own dataset file (data/<name>.yaml by default). Only
train2017 images are picked: the val2017 images behind every mAP in the study are
never trained on.
"""

import argparse
import http.client
import random
import shutil
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ultralytics.utils import ASSETS_URL, DATASETS_DIR
from ultralytics.utils.downloads import download

ROOT = Path(__file__).resolve().parent.parent
LABELS_ZIP = Path(DATASETS_DIR) / "coco2017labels.zip"
IMAGE_URL = "http://images.cocodataset.org/train2017/{}"


def labels_zip() -> zipfile.ZipFile:
    """Ultralytics' COCO labels archive (~50 MB); bench.coco_val leaves a copy in the datasets dir."""
    if not LABELS_ZIP.exists():
        download([ASSETS_URL + "/coco2017labels.zip"], dir=DATASETS_DIR, unzip=False)
    return zipfile.ZipFile(LABELS_ZIP)


def pick(zf: zipfile.ZipFile, n: int, seed: int) -> list[str]:
    """The first n train2017 file names in one seeded shuffle of the sorted list."""
    names = lambda split: {Path(line).name for line in zf.read(f"coco/{split}2017.txt").decode().split()}
    order = sorted(names("train"))
    random.Random(seed).shuffle(order)
    picked = order[:n]
    if len(picked) < n:
        raise SystemExit(f"train2017 has only {len(order)} images")
    overlap = names("val").intersection(picked)
    assert not overlap, f"val2017 images in the training subset: {sorted(overlap)[:5]}"
    return picked


def fetch(name: str, dest: Path, retries: int = 4) -> int:
    """Download one image unless it is already there; returns bytes written."""
    path = dest / name
    if path.exists() and path.stat().st_size > 0:
        return 0
    tmp = path.with_suffix(".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(IMAGE_URL.format(name), timeout=60) as r, tmp.open("wb") as fh:
                shutil.copyfileobj(r, fh)
            with tmp.open("rb") as fh:
                if fh.read(2) != b"\xff\xd8":
                    raise ValueError("not a JPEG")
            tmp.replace(path)
            return path.stat().st_size
        except (OSError, ValueError, http.client.HTTPException) as e:  # e.g. IncompleteRead on a cut connection
            error = e
            time.sleep(2 ** attempt)
    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"{name}: {error}")


def write_yaml(path: Path, name: str, root: Path, n: int, seed: int) -> None:
    """Dataset file for the subset, with the class names of data/coco-val2017.yaml."""
    src = (ROOT / "data" / "coco-val2017.yaml").read_text()
    names = src[src.index("names:"):]
    rel = root.resolve() == Path(DATASETS_DIR).resolve()
    val = "../coco/val2017.txt" if rel else Path(DATASETS_DIR) / "coco" / "val2017.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# COCO train2017, {n:,} images picked with seed {seed}. Download with\n"
        f"# `python -m bench.coco_train_subset --n {n} --seed {seed} --name {name}"
        + ("`.\n" if rel else f" --root {root}`.\n")
        + ("# Relative paths resolve against Ultralytics' datasets directory.\n" if rel else "")
        + f"path: {name if rel else (root / name).resolve()}\n"
        "train: train.txt\n"
        f"val: {val}  # COCO val2017 (bench.coco_val); bench.qat never reads it\n"
        f"\n{names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=5000, help="number of train2017 images (default 5000)")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed (default 0)")
    parser.add_argument("--name", default="coco-train5k", help="dataset folder name (default coco-train5k)")
    parser.add_argument("--root", type=Path, default=Path(DATASETS_DIR),
                        help="parent folder of the dataset (default: Ultralytics' datasets dir)")
    parser.add_argument("--yaml", type=Path, help="dataset file to write (default data/<name>.yaml)")
    parser.add_argument("--workers", type=int, default=16, help="parallel downloads (default 16)")
    args = parser.parse_args()

    dataset = args.root / args.name
    images, labels = dataset / "images" / "train2017", dataset / "labels" / "train2017"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    with labels_zip() as zf:
        picked = pick(zf, args.n, args.seed)
        members = set(zf.namelist())
        empty = 0
        for name in picked:
            member = f"coco/labels/train2017/{Path(name).stem}.txt"
            if member in members:
                (labels / Path(member).name).write_bytes(zf.read(member))
            else:
                empty += 1  # no objects: Ultralytics reads a missing label file as a background image

    t0, done, total = time.time(), 0, 0
    with ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed(pool.submit(fetch, name, images) for name in picked):
            total += future.result()
            done += 1
            if done % 500 == 0 or done == len(picked):
                print(f"{done}/{len(picked)} images, {total / 1e6:.0f} MB new, {time.time() - t0:.0f} s", flush=True)

    (dataset / "train.txt").write_text("".join(f"./images/train2017/{name}\n" for name in picked))
    yaml_path = args.yaml or ROOT / "data" / f"{args.name}.yaml"
    write_yaml(yaml_path, args.name, args.root, args.n, args.seed)
    print(f"{dataset}: {len(picked)} images ({empty} without objects) -> {yaml_path}")
