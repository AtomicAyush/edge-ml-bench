"""COCO mAP for a YOLOv8-style ONNX detector, with the input stretched or letterboxed.

    python -m bench.stretch_eval --model models/onnx_int8_tuned/yolov8n.onnx \\
        --images path/to/coco/val2017_500.txt --limit 300 --prep stretch
    python -m bench.stretch_eval --model models/onnx_fp32/yolov8n.onnx \\
        --images path/to/presized_images --prep stretch
    python -m bench.stretch_eval ... --csv results/stretch_accuracy.csv

Some deployment pipelines feed a YOLOv8 model a 640x640 picture made by resizing
the whole image to a square, so the aspect ratio is lost. Ultralytics, and
every mAP in results/accuracy.csv, letterbox instead. A model trained on letterboxed
images sees squashed or widened objects when the input is stretched, so the same
weights score differently. This measures both, with everything else the same:

  stretch    cv2.INTER_LINEAR resize of the RGB image to the model's input size.
             Boxes map back by w0/640 across and h0/640 down.
  letterbox  Ultralytics' val preprocessing (rect=False): long side to 640 with
             cv2.INTER_LINEAR, then centred on a 640x640 canvas of grey (114).
             Boxes map back by removing the padding and dividing by the scale.

Either way the input is RGB, divided by 255, NCHW float32, and the model's output is
[1, 4 + 80, N]: cx, cy, w, h in input pixels, then 80 class scores, with no NMS.
Postprocessing follows Ultralytics val: every (box, class) pair scoring above --conf
(multi-label; --single-label keeps each box's best class only, as Ultralytics predict
does), class-aware NMS at --iou, the --max-det best kept, boxes clipped to the original
image. Class index k becomes the k-th category id in the annotation file (COCO's ids
run 1 to 90 with gaps, so k + 1 is wrong from stop sign onward).

Scoring is pycocotools COCOeval (bbox) over exactly the images evaluated, reporting
AP (0.5:0.95), AP50 and AP75. These are the official COCO numbers. Ultralytics' own
printed mAP uses its own matcher and labels without crowd regions; on the same
detections it reads slightly lower.

--images takes a list file in Ultralytics' format (one path per line; relative paths
are relative to the list file) or a folder of .jpg, .png or .ppm files. The COCO
image id is the file name (000000000139.jpg is image 139). The original size always
comes from the annotation file, so an image already resized to the model's input
size (any of those formats) is used as is and scored against the original boxes
(640x426 for image 139). Letterbox refuses such an image, since its aspect ratio
is already gone.

Prints one line per run. --csv appends a row (device, images, prep, variant,
map50_95, map50, map75), replacing an earlier row with the same device, images,
prep and variant. The variant defaults to the model file's stem plus any setting
that differs from the defaults (yolov8n_int8 conf0.25), so a run with another
--conf, --iou, --max-det, --single-label or --agnostic never replaces a default row.
--save-json writes the detections in COCO results format.
"""

import argparse
import ast
import contextlib
import csv
import io
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from bench.latency import device_name

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}
PREPS = ("stretch", "letterbox")
PAD_VALUE = 114
MAX_NMS = 30000  # Ultralytics: boxes entering NMS, highest scores first
MAX_WH = 7680    # Ultralytics: per-class box offset that makes one NMS call class-aware
CSV_FIELDS = ["device", "images", "prep", "variant", "map50_95", "map50", "map75"]


def default_ann() -> Path:
    from ultralytics.utils import DATASETS_DIR
    return Path(DATASETS_DIR) / "coco" / "annotations" / "instances_val2017.json"


def image_paths(src: Path, limit: int | None = None) -> list[Path]:
    """Images from a folder (sorted by name) or a list file (in file order)."""
    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        lines = [line.strip() for line in src.read_text().splitlines() if line.strip()]
        paths = [p if p.is_absolute() else src.parent / p for p in map(Path, lines)]
    return paths[:limit] if limit else paths


def coco_id(path: Path) -> int:
    if not path.stem.isdigit():
        raise ValueError(f"{path.name}: the file name must be the COCO image id, e.g. 000000000139.jpg")
    return int(path.stem)


def load_coco(ann: Path):
    from pycocotools.coco import COCO
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(ann))


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR, for PPM too
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def stretch(img: np.ndarray, size: tuple[int, int], orig: tuple[int, int]):
    """Plain resize to size (h, w). Returns the input image and a function mapping boxes back to orig (h0, w0)."""
    (h, w), (h0, w0) = size, orig
    if img.shape[:2] == (h0, w0):
        x = img if (h0, w0) == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    elif img.shape[:2] == (h, w):
        x = img  # already resized to the model's input size
    else:
        raise ValueError(f"image is {img.shape[1]}x{img.shape[0]}; expected the original {w0}x{h0} or {w}x{h}")
    scale = np.array([w0 / w, h0 / h, w0 / w, h0 / h], dtype=np.float32)
    return x, lambda boxes: boxes * scale


def letterbox(img: np.ndarray, size: tuple[int, int], orig: tuple[int, int]):
    """Ultralytics' val preprocessing (BaseDataset.load_image, then LetterBox(scaleup=False)).

    Returns the input image and a function mapping boxes back to orig (h0, w0).
    """
    (h, w), (h0, w0) = size, orig
    if img.shape[:2] != (h0, w0):
        raise ValueError(f"image is {img.shape[1]}x{img.shape[0]}, not its original {w0}x{h0}: "
                         "letterbox needs the original aspect ratio")
    # load_image: long side to the input size (ceil, capped), shrinking or enlarging
    r = max(h, w) / max(h0, w0)
    if r != 1:
        img = cv2.resize(img, (min(math.ceil(w0 * r), max(h, w)), min(math.ceil(h0 * r), max(h, w))),
                         interpolation=cv2.INTER_LINEAR)
    h1, w1 = img.shape[:2]
    # LetterBox: shrink to fit if needed (never enlarge), then pad centred
    r2 = min(h / h1, w / w1, 1.0)
    unpad = round(w1 * r2), round(h1 * r2)
    if (w1, h1) != unpad:
        img = cv2.resize(img, unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = (w - unpad[0]) / 2, (h - unpad[1]) / 2
    top, bottom, left, right = round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1)
    x = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    gain = np.array([w1 / w0 * r2, h1 / h0 * r2] * 2, dtype=np.float32)  # Ultralytics' ratio_pad gains
    pad = np.array([left, top, left, top], dtype=np.float32)
    return x, lambda boxes: (boxes - pad) / gain


PREPROCESS = {"stretch": stretch, "letterbox": letterbox}


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        order = rest[inter / (areas[i] + areas[rest] - inter) <= iou]
    return np.array(keep, dtype=np.int64)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """torchvision's NMS (what Ultralytics calls) when installed, else the same greedy NMS in numpy."""
    try:
        import torch
        import torchvision
    except ImportError:
        return nms_numpy(boxes, scores, iou)
    return torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).numpy()


def postprocess(out: np.ndarray, conf: float, iou: float, max_det: int,
                single_label: bool = False, agnostic: bool = False):
    """[1, 4 + nc, N] -> boxes (x1, y1, x2, y2 in input pixels), scores, class indices; Ultralytics val's NMS."""
    p = out[0].T.astype(np.float32, copy=False)
    cls = p[:, 4:]
    p = p[cls.max(1) > conf]
    cxcywh, cls = p[:, :4], p[:, 4:]
    boxes = np.concatenate([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], axis=1)
    if single_label:
        j = cls.argmax(1)
        scores = cls[np.arange(len(cls)), j]
        i = np.nonzero(scores > conf)[0]
        boxes, scores, j = boxes[i], scores[i], j[i]
    else:
        i, j = np.nonzero(cls > conf)
        boxes, scores = boxes[i], cls[i, j]
    if len(scores) > MAX_NMS:
        top = np.argsort(-scores, kind="stable")[:MAX_NMS]
        boxes, scores, j = boxes[top], scores[top], j[top]
    if not len(scores):
        return boxes, scores, j
    offset = (j[:, None] * (0 if agnostic else MAX_WH)).astype(np.float32)
    keep = nms(np.ascontiguousarray(boxes + offset), np.ascontiguousarray(scores), iou)[:max_det]
    return boxes[keep], scores[keep], j[keep]


class Detector:
    """An ONNX Runtime session plus the input size and class names it declares."""

    def __init__(self, model: Path):
        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2:4]
        self.size = (h if isinstance(h, int) else 640, w if isinstance(w, int) else 640)
        names = self.sess.get_modelmeta().custom_metadata_map.get("names")
        self.names = list(ast.literal_eval(names).values()) if names else None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return self.sess.run(None, {self.input_name: x})[0]


def detect(det: Detector, path: Path, orig: tuple[int, int], prep: str, conf: float, iou: float,
           max_det: int, single_label: bool = False, agnostic: bool = False):
    """Boxes (x1, y1, x2, y2 in original-image pixels, clipped), scores and class indices for one image."""
    x, unmap = PREPROCESS[prep](read_rgb(path), det.size, orig)
    boxes, scores, cls = postprocess(det(x), conf, iou, max_det, single_label, agnostic)
    boxes = unmap(boxes)
    h0, w0 = orig
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
    return boxes, scores, cls


def coco_results(image_id: int, boxes, scores, cls, cat_ids: list[int]) -> list[dict]:
    """COCO results-format entries, rounded the way Ultralytics' predictions.json is."""
    return [{"image_id": image_id, "category_id": cat_ids[int(c)],
             "bbox": [round(float(v), 3) for v in (b[0], b[1], b[2] - b[0], b[3] - b[1])],
             "score": round(float(s), 5)}
            for b, s, c in zip(boxes, scores, cls)]


def score(coco, results: list[dict], image_ids: list[int], verbose: bool = False) -> dict:
    """COCOeval bbox AP over exactly image_ids: map50_95, map50, map75 (-1 where COCO has no ground truth)."""
    from pycocotools.cocoeval import COCOeval
    if not results:
        return {"map50_95": 0.0, "map50": 0.0, "map75": 0.0}
    quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
    with quiet:
        # loadRes adds area, id, iscrowd and a box polygon to each dict it is given; copies keep
        # results (and --save-json) in plain COCO results format
        ev = COCOeval(coco, coco.loadRes([dict(r) for r in results]), "bbox")
        ev.params.imgIds = sorted(set(image_ids))
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"map50_95": float(ev.stats[0]), "map50": float(ev.stats[1]), "map75": float(ev.stats[2])}


def evaluate(model: Path, paths: list[Path], coco, prep: str, conf: float = 0.001, iou: float = 0.7,
             max_det: int = 300, single_label: bool = False, agnostic: bool = False,
             verbose: bool = False) -> tuple[dict, list[dict]]:
    """Scores and COCO results for one model over the given images."""
    det = Detector(model)
    cats = coco.dataset["categories"]  # file order: index k is the k-th category
    cat_ids = [c["id"] for c in cats]
    if det.names and len(det.names) != len(cat_ids):
        raise ValueError(f"model has {len(det.names)} classes, annotations have {len(cat_ids)} categories")
    if det.names and det.names != [c["name"] for c in cats]:
        diff = [(a, c["name"]) for a, c in zip(det.names, cats) if a != c["name"]]
        print(f"warning: {len(diff)} model class names differ from the annotation's, e.g. {diff[:3]}")
    results, ids = [], []
    for path in paths:
        image_id = coco_id(path)
        info = coco.imgs[image_id]
        boxes, scores, cls = detect(det, path, (info["height"], info["width"]), prep, conf, iou, max_det,
                                    single_label, agnostic)
        results += coco_results(image_id, boxes, scores, cls, cat_ids)
        ids.append(image_id)
    return score(coco, results, ids, verbose), results


def images_label(src: Path, n: int, limit: int | None) -> str:
    return f"{src.name}[:{n}]" if limit else src.name


def settings_tag(args, defaults: dict) -> str:
    """'' at the default settings, else e.g. ' conf0.25 maxdet20': the CSV key has no settings columns."""
    tags = [f"conf{args.conf:g}" if args.conf != defaults["conf"] else "",
            f"iou{args.iou:g}" if args.iou != defaults["iou"] else "",
            f"maxdet{args.max_det}" if args.max_det != defaults["max_det"] else "",
            "single" if args.single_label else "", "agnostic" if args.agnostic else ""]
    return "".join(f" {t}" for t in tags if t)


def write_csv(path: Path, row: dict):
    """Append row, replacing an earlier row with the same device, images, prep and variant."""
    key = lambda r: (r["device"], r["images"], r["prep"], r["variant"])
    old = list(csv.DictReader(path.open())) if path.exists() else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows([r for r in old if key(r) != key(row)] + [row])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True, help="ONNX detector, output [1, 4 + 80, N]")
    parser.add_argument("--images", type=Path, required=True, help="list file or folder of images")
    parser.add_argument("--ann", type=Path, default=None,
                        help="COCO annotations (default: <Ultralytics datasets>/coco/annotations/instances_val2017.json)")
    parser.add_argument("--prep", choices=PREPS, required=True)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, help="first N images only")
    parser.add_argument("--single-label", action="store_true", help="best class per box only (Ultralytics predict)")
    parser.add_argument("--agnostic", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--variant", help="name for the CSV row (default: the model file's stem plus any non-default settings)")
    parser.add_argument("--csv", type=Path, help="append the result here")
    parser.add_argument("--save-json", type=Path, help="write the detections in COCO results format")
    parser.add_argument("--verbose", action="store_true", help="print COCOeval's full summary")
    args = parser.parse_args()

    paths = image_paths(args.images, args.limit)
    if not paths:
        raise SystemExit(f"no images in {args.images}")
    coco = load_coco(args.ann or default_ann())
    t = time.perf_counter()
    try:
        m, results = evaluate(args.model, paths, coco, args.prep, args.conf, args.iou, args.max_det,
                              args.single_label, args.agnostic, args.verbose)
    except ValueError as e:  # wrong image size for --prep, file name not a COCO id, class count mismatch
        raise SystemExit(f"error: {e}")
    defaults = {k: parser.get_default(k) for k in ("conf", "iou", "max_det")}
    variant = args.variant or args.model.stem + settings_tag(args, defaults)
    print(f"{variant:24} {args.prep:9} {len(paths):5} images  conf {args.conf:g}  "
          f"AP {m['map50_95']:.6f}  AP50 {m['map50']:.6f}  AP75 {m['map75']:.6f}  "
          f"({len(results)} detections, {time.perf_counter() - t:.1f} s)")
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(results))
    if args.csv:
        write_csv(args.csv, {"device": device_name(), "images": images_label(args.images, len(paths), args.limit),
                             "prep": args.prep, "variant": variant,
                             **{k: round(v, 4) for k, v in m.items()}})
