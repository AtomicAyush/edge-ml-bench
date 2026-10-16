"""Per-variant metadata, so one harness can measure several models and input sizes.

Each folder under models/ may hold a meta.json:

    {"base": "yolov8s.pt", "imgsz": 480}

Folders without one are the original study: YOLOv8n at 640x640.
"""

import json
from pathlib import Path

DEFAULT = {"base": "yolov8n.pt", "imgsz": 640}


def read_meta(variant_dir: Path) -> dict:
    path = Path(variant_dir) / "meta.json"
    meta = dict(DEFAULT)
    if path.exists():
        meta.update(json.loads(path.read_text()))
    return meta


def write_meta(variant_dir: Path, **fields) -> None:
    variant_dir = Path(variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    meta = {**read_meta(variant_dir), **fields}
    (variant_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def variant_name(kind: str, base: str = DEFAULT["base"], imgsz: int = DEFAULT["imgsz"]) -> str:
    """Folder name for a variant: the original names for YOLOv8n at 640, prefixed otherwise.

    variant_name("coreml_fp16")                       -> "coreml_fp16"
    variant_name("coreml_fp16", "yolov8s.pt")         -> "yolov8s_coreml_fp16"
    variant_name("coreml_fp16", imgsz=320)            -> "coreml_fp16_320"
    """
    stem = Path(base).stem
    name = kind if stem == Path(DEFAULT["base"]).stem else f"{stem}_{kind}"
    return name if imgsz == DEFAULT["imgsz"] else f"{name}_{imgsz}"
