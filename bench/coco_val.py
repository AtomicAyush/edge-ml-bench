"""Download COCO val2017 (5,000 images, ~1 GB) for the final accuracy numbers.

    python -m bench.coco_val
    python -m bench.accuracy --data data/coco-val2017.yaml

Ultralytics' own coco.yaml also downloads train2017 (19 GB), which the study
never uses. This fetches the same labels and only the validation images.
"""

from ultralytics.utils import ASSETS_URL, DATASETS_DIR
from ultralytics.utils.downloads import download

if __name__ == "__main__":
    root = DATASETS_DIR / "coco"
    download([ASSETS_URL + "/coco2017labels.zip"], dir=DATASETS_DIR)
    download(["http://images.cocodataset.org/zips/val2017.zip"], dir=root / "images")
    print(root)
