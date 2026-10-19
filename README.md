# edge-ml-bench

How much faster does a real-time object detector get from lower precision and dedicated AI hardware, and what does each choice cost in accuracy?

This repo measures YOLOv8n (COCO, 640×640) on an M4 Pro MacBook and an iPhone 18 Pro Max:

- **Formats and runtimes:** ONNX Runtime and Core ML.
- **Precisions:** FP32, FP16 and INT8.
- **Compute units:** CPU, GPU and Neural Engine.

It also makes INT8 accurate, through calibration and per-layer mixed precision. And it times the whole frame-to-boxes pipeline (letterbox, model, decode, NMS) in an iPhone app ([`ios/`](ios/)): on a fixed photo for the published numbers, and live on camera frames in the app's Live tab. Camera capture itself is not timed.

![latency and accuracy for every configuration](results/tradeoff.png)

## Results at a glance

- **The Neural Engine is the biggest win.**
  - FP16 on the Neural Engine runs the model in **1.69 ms on the M4 Pro**, 8.1× faster than ONNX Runtime FP32 on the CPU (13.61 ms).
  - On the **iPhone 18 Pro Max** it takes **1.01 ms**, 12.1× faster than FP32 on the phone's CPU (12.18 ms).
  - Accuracy is identical in both cases: mAP50-95 0.368.
- **The Neural Engine does not run FP32 models.**
  - An FP32 Core ML model allowed to use the Neural Engine runs at CPU speed: 12.71 ms vs 12.69 ms on the Mac, 12.28 ms vs 12.18 ms on the iPhone.
  - Core ML raises no error; the model simply runs at CPU speed. Converting it to FP16 is what unlocks the hardware.
- **End to end, the iPhone takes 1.61 ms per image on the Neural Engine** (FP16, an 810×1080 photo).
  - That covers letterboxing, the model, decoding 8,400 candidate boxes and NMS. The work around the model adds 0.6 ms. The Live tab runs the same pipeline on camera frames.
  - The first load of the INT8 model on the Neural Engine after installing took 2.07 s (on-device compile). Cached loads take about 13 ms on the Neural Engine and under 3 ms on the CPU and GPU.
- **Naive static INT8 detects nothing (mAP 0.000).**
  - YOLOv8's last step joins box coordinates (0–640 px) and class scores (0–1) into one tensor. A single INT8 range for that tensor rounds every score to zero.
  - Keeping just that decode step in FP32 brings mAP back to 0.344.
- **Per-channel weights and percentile calibration together close 70% of the remaining INT8 gap.**
  - Percentile activation ranges with per-channel weights raise INT8 from 0.344 to **0.3605**. The FP32 model scores 0.3678.
  - Per-channel weights give most of that gain: 55–80% on the 500-image subset, depending on which change is applied first. Percentile activation ranges give the rest.
  - Entropy (KL) calibration was the worst of the three methods here.
- **No single layer dominates the INT8 error, though the two most sensitive layers are early, high-resolution ones.**
  - Restoring the first convolution (`model.0`, which sees the 640×640 input) to FP32 removes 26% of the INT8 output error on its own. The first convolution of the 160×160 block (`model.2/cv1`) removes 25%.
  - Restoring any one of a dozen layers removes 10–26% of the INT8 output error on its own. These shares overlap rather than add up, which is why keeping more layers in FP32 recovers mAP only slowly.
  - Keeping one layer in FP32 gives **0.3625 mAP at 11.44 ms**: 1.19× faster than FP32, 1.4% below it in mAP.
  - Keeping more layers recovers a little more accuracy but gives back most of the speedup. Restoring `model.0` costs about 0.13 ms, which is within run-to-run noise. Restoring the next three ranked layers costs another 1.34 ms, out of the 2.30 ms that INT8 saves.
- **Weight-only INT8 (`coreml_int8`) halves the model size** (6.5 → 3.4 MB) at nearly the same accuracy (0.367 vs 0.368). On the Neural Engine it runs at the same speed as FP16 on both devices. Its weights are 8-bit indices into a lookup table, which Core ML expands to dense FP16 (one `constexpr_lut_to_dense` op per convolution), so the arithmetic is still 16-bit.

## Results

**Accuracy and timing.**

- mAP50-95 and mAP50 are measured on COCO val2017 (5,000 images). The PyTorch FP32 baseline scores 0.368.
- Every latency is the median of 3 passes. Each pass is 20 warm-up runs followed by 300 timed runs on the Mac, or 10 warm-up and 100 timed runs on the iPhone. The range column shows the fastest and slowest pass.
- Raw numbers: [`results/latency.csv`](results/latency.csv), [`results/accuracy.csv`](results/accuracy.csv), [`results/ios.csv`](results/ios.csv).

### MacBook Pro, Apple M4 Pro (model only, Python harness)

| Variant | Backend | p50 (ms) | Range across passes | Size (MB) | mAP50-95 |
|---|---|---|---|---|---|
| `coreml_fp16` | Core ML · ALL | 1.67 | 1.65–1.69 | 6.5 | 0.368 |
| `coreml_fp16` | Core ML · Neural Engine | 1.69 | 1.66–1.69 | 6.5 | 0.368 |
| `coreml_int8` | Core ML · ALL | 1.69 | 1.69–1.76 | 3.4 | 0.367 |
| `coreml_int8` | Core ML · Neural Engine | 1.70 | 1.69–1.74 | 3.4 | 0.367 |
| `coreml_fp16` | Core ML · GPU | 3.46 | 3.45–3.47 | 6.5 | 0.368 |
| `coreml_fp32` | Core ML · GPU | 3.59 | 3.57–3.93 | 12.8 | 0.368 |
| `coreml_fp32` | Core ML · ALL | 3.69 | 3.57–3.80 | 12.8 | 0.368 |
| `coreml_int8` | Core ML · GPU | 5.23 | 5.19–5.33 | 3.4 | 0.367 |
| `coreml_fp16` | Core ML · CPU | 9.68 | 9.65–9.90 | 6.5 | 0.368 |
| `coreml_int8` | Core ML · CPU | 9.88 | 9.69–9.90 | 3.4 | 0.367 |
| `onnx_int8_static_head_fp32` | ONNX Runtime · CPU | 10.89 | 10.34–11.12 | 3.6 | 0.344 |
| `onnx_int8_static` | ONNX Runtime · CPU | 10.96 | 10.25–10.98 | 3.5 | **0.000** |
| `onnx_int8_tuned` | ONNX Runtime · CPU | 11.31 | 10.79–11.64 | 3.7 | 0.3605 |
| `onnx_int8_mixed1` | ONNX Runtime · CPU | 11.44 | 11.20–12.04 | 3.7 | 0.3625 |
| `coreml_fp32` | Core ML · CPU | 12.69 | 12.65–13.04 | 12.8 | 0.368 |
| `coreml_fp32` | Core ML · Neural Engine | 12.71 | 12.62–13.12 | 12.8 | 0.368 |
| `onnx_int8_mixed4` | ONNX Runtime · CPU | 12.78 | 12.57–13.26 | 3.7 | 0.3630 |
| `onnx_int8_mixed8` | ONNX Runtime · CPU | 12.88 | 12.37–13.10 | 3.9 | 0.3638 |
| `onnx_fp32` | ONNX Runtime · CPU | 13.61 | 13.44–14.68 | 12.9 | 0.368 |
| `onnx_int8_dynamic` | ONNX Runtime · CPU | 16.80 | 16.65–17.42 | 3.5 | 0.352 |

### iPhone 18 Pro Max (EdgeDetect app, Swift harness)

| Variant | Compute unit | Model p50 (ms) | Range across passes | End to end p50 (ms) | mAP50-95 |
|---|---|---|---|---|---|
| `coreml_int8` | Neural Engine | 0.98 | 0.95–1.41 | 1.57 | 0.367 |
| `coreml_fp16` | Neural Engine | 1.01 | 1.00–1.02 | 1.61 | 0.368 |
| `coreml_int8` | ALL | 1.01 | 1.00–1.38 | 1.59 | 0.367 |
| `coreml_fp16` | ALL | 1.03 | 1.03–1.04 | 1.61 | 0.368 |
| `coreml_fp16` | GPU | 2.38 | 2.38–2.38 | 2.97 | 0.368 |
| `coreml_fp32` | GPU | 4.85 | 4.84–4.85 | 5.45 | 0.368 |
| `coreml_fp32` | ALL | 4.96 | 4.95–4.96 | 5.55 | 0.368 |
| `coreml_int8` | GPU | 4.99 | 4.98–5.02 | 5.58 | 0.367 |
| `coreml_fp16` | CPU | 6.97 | 6.97–7.12 | 7.95 | 0.368 |
| `coreml_int8` | CPU | 7.07 | 7.02–7.39 | 8.11 | 0.367 |
| `coreml_fp32` | CPU | 12.18 | 12.14–12.22 | 13.32 | 0.368 |
| `coreml_fp32` | Neural Engine | 12.28 | 12.21–12.33 | 13.32 | 0.368 |

The two harnesses time different amounts of work, so compare phone rows with phone rows and Mac rows with Mac rows:

- **iPhone "Model":** only the Core ML prediction call, on a pooled pixel buffer.
- **Mac Core ML rows:** also include coremltools converting a PIL image in and the output to NumPy on every call. The ONNX Runtime rows pass a NumPy array directly.

The phone's mAP is taken from the Mac run of the same model file. On the test photo, the app's Sample photo view finds five detections, a bus and four people (the `detections` column of `ios.csv`). A hand check against `YOLO("models/coreml_fp16/yolov8n.mlpackage").predict("bus.jpg", iou=0.45)` in Python found the same five, with box corners within a pixel. That comparison is not saved in `results/`.

## Making INT8 accurate

[`bench/int8_sensitivity.py`](bench/int8_sensitivity.py) works in three steps. All runs use ONNX Runtime static QDQ quantization, UINT8 activations and INT8 weights, with the detect-head decode kept in FP32.

**1. Calibration.** Activation ranges come from 128 coco128 images, which are COCO train images. These mAP numbers are on the first 500 val2017 images, where FP32 scores 0.4012 ([`results/int8_calibration.csv`](results/int8_calibration.csv)).

| Method | Per-tensor weights | Per-channel weights |
|---|---|---|
| Min/max | 0.3730 | 0.3892 |
| Entropy (KL, 2048 bins) | 0.3695 | 0.3820 |
| Percentile (99.999th) | 0.3821 | **0.3933** |

**2. Which layers INT8 hurts.**

- **The method:** start from the full INT8 model and put one layer back in FP32, where a layer is a convolution plus its SiLU. Then measure how much closer the raw output gets to FP32's, on 64 other val2017 images. This is called leave-one-out ([`results/int8_sensitivity_loo.csv`](results/int8_sensitivity_loo.csv)).
- **The result:** two early, high-resolution layers dominate. Restoring `model.0` (640×640 input) alone removes 26% of the error, and `model.2/cv1` (160×160) removes 25%. Every other layer removes at most 16% on its own. The shares add up to more than 100% because the errors are not independent.
- **A first attempt failed.** An earlier version of this script ranked layers by quantizing each one alone, with the rest in FP32. It also "kept layers in FP32" by excluding only the convolution, so the SiLU around it stayed INT8. Mixed precision built on that ranking did not beat plain INT8 on the full validation set: 0.3603 with 2 layers in FP32, and 0.3547 with 8, against 0.3605 with none ([`results/int8_first_attempt.csv`](results/int8_first_attempt.csv), copied from that run's logs). The fix was to measure exactly the change mixed precision makes, and to keep the whole layer in FP32. `--stage isolated` still exists, but it now keeps the SiLU in FP32 too.

**3. Mixed precision.** The top-k layers from step 2 stay in FP32 (all 5,000 val2017 images; Mac CPU, ONNX Runtime):

| Model | mAP50-95 | Gap to FP32 | p50 (ms) | Speedup vs FP32 |
|---|---|---|---|---|
| FP32 (`onnx_fp32`) | 0.3678 | — | 13.61 | 1.00× |
| INT8, min/max per-tensor (`onnx_int8_static_head_fp32`) | 0.3437 | −0.0241 | 10.89 | 1.25× |
| INT8, percentile per-channel (`onnx_int8_tuned`) | 0.3605 | −0.0073 | 11.31 | 1.20× |
| + 1 layer in FP32 (`onnx_int8_mixed1`) | 0.3625 | −0.0053 | 11.44 | 1.19× |
| + 4 layers in FP32 (`onnx_int8_mixed4`) | 0.3630 | −0.0048 | 12.78 | 1.06× |
| + 8 layers in FP32 (`onnx_int8_mixed8`) | 0.3638 | −0.0040 | 12.88 | 1.06× |

**Don't trust a 500-image subset.** On 500 images, the first attempt (above) beat plain INT8 with 2 to 6 layers in FP32 (best: 0.3978 vs 0.3933, with 2). On all 5,000 it did not. Even with the final ranking, one FP32 layer gains 0.0058 mAP on the subset ([`results/int8_sweep_loo.csv`](results/int8_sweep_loo.csv)) but only 0.0020 on the full set. Every number in this table uses the full set.

## What gets compared

| Variant | Format | Precision | How it's made |
|---|---|---|---|
| `pytorch_fp32` | PyTorch | FP32 | Ultralytics baseline (accuracy reference) |
| `onnx_fp32` | ONNX | FP32 | Ultralytics export, opset 17, simplified |
| `onnx_int8_dynamic` | ONNX | INT8 | ONNX Runtime dynamic quantization (activation ranges computed at run time) |
| `onnx_int8_static` | ONNX | INT8 | ONNX Runtime static QDQ, min/max calibration on coco128 |
| `onnx_int8_static_head_fp32` | ONNX | INT8 + FP32 decode | Same, with the detect-head decode kept in FP32 |
| `onnx_int8_tuned` | ONNX | INT8 + FP32 decode | Percentile calibration, per-channel weights |
| `onnx_int8_mixed{1,4,8}` | ONNX | Mixed INT8/FP32 | `tuned` with the top 1, 4 or 8 layers in FP32 |
| `coreml_fp32` / `coreml_fp16` | Core ML | FP32 / FP16 | Ultralytics export |
| `coreml_int8` | Core ML | INT8 weights | Ultralytics export (weight-only; compute in FP16) |
| `tflite_fp32` / `tflite_int8` | TFLite | FP32 / INT8 | Ultralytics export (Raspberry Pi path, not yet run) |

Each Core ML model runs with `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE` and `ALL`, so one model file is compared across all three compute units. With `ALL`, Core ML decides how to split the model across them.

## Method

- **Latency:**
  - Warm-up first (20 inferences on the Mac, 10 on the iPhone), so one-time compilation and caching don't count.
  - Then 300 timed inferences on a fixed 640×640 input (100 on the iPhone). Reported: p50, p95, mean, FPS.
  - At least 3 passes per configuration, with no other benchmark jobs running. The chart and tables use the median pass, and the whiskers show the fastest and slowest pass. The three Mac passes ran back to back, and every ONNX row got slower with each pass: 5–9% from the first pass to the third (`onnx_fp32`: 13.44 → 14.68 ms). The Core ML rows did not drift. Speedups computed from a single pass move by a similar amount (`onnx_int8_tuned` is 1.20–1.26× faster than `onnx_fp32`, depending on the pass), so differences under about 10% between ONNX variants are within the noise.
  - **Model only:** preprocessing and NMS are excluded. The iPhone app also reports end-to-end time.
- **Accuracy:**
  - mAP50-95 and mAP50 from `ultralytics val` on COCO val2017 (`make coco-val accuracy-full`). This downloads only the validation images, about 1 GB.
  - Accuracy is measured once per model file, on the Mac, and assumed to carry over to other compute units and devices. Ultralytics runs Core ML detection models with `CPU_AND_NE` for validation, and ONNX models on the CPU.
  - coco128 numbers in `accuracy.csv` are a quick check only. Its images come from COCO's training set, so mAP there reads about 0.08 high.
- **Not measured:**
  - Memory. The process's resident memory depends on what else was loaded earlier, and Neural Engine and GPU memory live outside the process.
  - Power and energy.

## Running it

```bash
make setup         # Python 3.12 virtualenv + dependencies
make export        # ONNX + Core ML variants into models/
make quantize      # ONNX Runtime INT8 variants (downloads coco128 for calibration)
make coco-val      # COCO val2017 images + labels (~1 GB); make int8, w8a8, compress and accuracy-full need them
make int8          # calibration comparison, layer ranking, sweep, tuned + mixed exports (slow: dozens of quantize + validate runs)
make w8a8          # full INT8 (W8A8) Core ML: build, layer ranking, mixed exports, compute plans, mAP (slow)
make compress      # palettized and pruned Core ML models -> results/coreml_compress.csv, then mAP
make latency       # one pass -> results/latency.csv; run it 3 times with nothing else running
make accuracy-full # mAP on COCO val2017 -> results/accuracy.csv
make plot          # -> results/tradeoff.png
make ios-models    # copy the Core ML models into the iPhone app, then open ios/EdgeDetect.xcodeproj
```

iPhone results are added with `python -m bench.merge_ios <exported.csv>` (see [`ios/README.md`](ios/README.md)). Raspberry Pi setup is in [PLAN.md](PLAN.md#raspberry-pi).

## Layout

```
bench/
  export.py           YOLOv8n -> ONNX / Core ML / TFLite variants
  quantize_onnx.py    ONNX Runtime INT8 (dynamic, static, static with FP32 decode)
  int8_sensitivity.py calibration comparison, per-layer sensitivity, mixed-precision sweep
  coreml_quant.py     full INT8 (W8A8) Core ML: build, layer ranking, compute plan
  coreml_compress.py  Core ML palettization and pruning
  latency.py          timing harness, one runner per backend / compute unit
  accuracy.py         mAP for every variant
  coco_val.py         downloads COCO val2017 (images + labels) only
  merge_ios.py        adds an EdgeDetect export to the results
  plot.py             latency and mAP chart, one panel per device
ios/                  EdgeDetect: live detection + on-device benchmark (SwiftUI, Core ML)
data/                 dataset config for COCO val2017
models/               exported models (git-ignored, rebuilt by make export / quantize / int8)
results/              CSVs and the chart
```

## Hardware

| Device | Details |
|---|---|
| MacBook Pro | Apple M4 Pro, 24 GB, macOS 27 |
| iPhone 18 Pro Max | Hardware ID iPhone19,3 (recorded in `results/ios.csv`); iOS 27.0.1, from Xcode's device list |
| Raspberry Pi | _planned_ |

## Limitations

- **One model:** YOLOv8n at 640×640, batch size 1.
- **INT8 on ONNX was only measured on the M4 Pro's CPU**, where ONNX Runtime's INT8 kernels give a modest speedup (1.2×). ARM boards and NPUs with dedicated INT8 hardware may differ.
- **The only INT8 model run on the Neural Engine is weight-only** (`coreml_int8`: 8-bit lookup-table weights, FP16 arithmetic). Core ML's activation quantization (W8A8) is not covered.
