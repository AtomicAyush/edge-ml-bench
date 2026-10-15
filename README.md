# edge-ml-bench

How much speed does a real-time object detector gain from lower precision and dedicated hardware, and how much accuracy does it give up?

This repo measures YOLOv8n (COCO, 640×640) across export formats, numeric precisions and compute units:

- **Apple silicon:** CPU, GPU and Neural Engine, through Core ML.
- **CPU:** ONNX Runtime.
- **Raspberry Pi:** ONNX Runtime and TFLite (LiteRT).

Every configuration gets the same measurements: latency, throughput, model size, memory and mAP. All results go into one table and one chart.

<!-- Uncomment once results/tradeoff.png is committed:
![speed vs. accuracy](results/tradeoff.png)
-->

## Results

> In progress. See [PLAN.md](PLAN.md) for the experiment matrix and status.

| Variant | Backend | Device | p50 (ms) | FPS | Size (MB) | mAP50-95 |
|---|---|---|---|---|---|---|
| | | | | | | |

Raw numbers will be in `results/latency.csv` and `results/accuracy.csv`.

## Findings

_To be written once the full runs are in._

## What gets compared

| Variant | Format | Precision | How it's made |
|---|---|---|---|
| `pytorch_fp32` | PyTorch | FP32 | Ultralytics baseline (accuracy reference) |
| `onnx_fp32` | ONNX | FP32 | Ultralytics export, opset 17, simplified |
| `onnx_int8_dynamic` | ONNX | INT8 weights | ONNX Runtime dynamic quantization |
| `onnx_int8_static` | ONNX | INT8 | ONNX Runtime static QDQ, calibrated on coco128 |
| `onnx_int8_static_head_fp32` | ONNX | INT8 + FP32 head | Same, with the detect-head decode kept in FP32 |
| `coreml_fp32` / `coreml_fp16` | Core ML | FP32 / FP16 | Ultralytics export |
| `coreml_int8` | Core ML | INT8 weights | Ultralytics export (weight quantization) |
| `tflite_fp32` / `tflite_int8` | TFLite | FP32 / INT8 | Ultralytics export (Raspberry Pi path) |

Backends:

- Core ML models run on each of `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE` and `ALL`, so the same file is compared across the CPU, GPU and Neural Engine.
- ONNX models run on ONNX Runtime's CPU execution provider.
- TFLite models run on the LiteRT interpreter.

## Method

- **Latency:**
  - Each configuration runs 20 warm-up inferences first, so one-time compilation and caching don't count.
  - It then runs 200 timed inferences on a fixed 640×640 input.
  - Reported: median (p50), p95, mean and FPS.
  - Inference only: preprocessing and NMS are excluded, so the numbers isolate the model and the hardware.
- **Accuracy:** mAP50-95 and mAP50 from `ultralytics val`.
  - The quick runs use coco128 (128 images).
  - Final numbers use COCO val2017 (5,000 images): `python -m bench.accuracy --data coco.yaml`.
  - Accuracy depends only on the model file, so it's measured once on the Mac and reused for other devices.
- **Size and memory:** size of the model on disk, and the process's resident memory after the run.
- **Results files:** every latency row records the device name, so Mac and Pi results share one CSV.

## Running it

```bash
make setup      # Python 3.12 virtualenv + dependencies
make export     # ONNX + Core ML variants into models/
make quantize   # ONNX Runtime INT8 variants (downloads coco128 for calibration)
make latency    # -> results/latency.csv
make accuracy   # -> results/accuracy.csv
make plot       # -> results/tradeoff.png
```

Useful flags:

- `python -m bench.latency --only coreml_fp16 --runs 500`
- `python -m bench.export --only tflite_fp32 tflite_int8` (needs TensorFlow; Ultralytics installs it on first use)

Raspberry Pi setup is in [PLAN.md](PLAN.md#raspberry-pi).

## Layout

```
bench/
  export.py         YOLOv8n -> ONNX / Core ML / TFLite variants
  quantize_onnx.py  ONNX Runtime INT8 (dynamic, static, static with FP32 head)
  latency.py        timing harness, one runner per backend / compute unit
  accuracy.py       mAP for every variant
  plot.py           latency vs. mAP chart
models/             exported models (git-ignored, rebuilt by make export / quantize)
results/            CSVs and the chart
```

## Hardware

| Device | Details |
|---|---|
| MacBook Pro | Apple M4 Pro, 24 GB, macOS 27 |
| Raspberry Pi | _planned_ |
