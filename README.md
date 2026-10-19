# edge-ml-bench

How much faster does a real-time object detector get from lower precision and dedicated AI hardware, and what does each choice cost in accuracy?

This repo measures YOLOv8n (COCO, 640×640) on an M4 Pro MacBook and an iPhone 18 Pro Max:

- **Formats and runtimes:** ONNX Runtime and Core ML.
- **Precisions:** FP32, FP16 and INT8.
- **Compute units:** CPU, GPU and Neural Engine.

It also makes INT8 accurate, through calibration and per-layer mixed precision. And it times the whole frame-to-boxes pipeline (letterbox, model, decode, NMS) in an iPhone app ([`ios/`](ios/)): on a fixed photo for the main tables, and live on camera frames in the app's Live tab, from each frame's capture timestamp to the screen refresh that first shows its boxes.

A second round adds full INT8 and weight compression in Core ML, four more detectors, two more input sizes, quantization-aware training and Ultralytics' own INT8 export ([Beyond the main study](#beyond-the-main-study-mac)).

![latency and accuracy for every configuration of the main study](results/tradeoff.png)

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
  - **Under continuous load it was mostly much slower.** In 10-minute runs on 16 Oct, 73% of the 10-second windows on the Neural Engine took about 4 ms for the model and 7.28–7.55 ms end to end (each run's median) instead of 1.0 and 1.6 ms, at thermal Nominal as well as Fair. The cause is not known yet ([details](#iphone-18-pro-max-16-oct-15-models-sustained-load-and-the-live-camera)).
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
- **Going beyond FP16 on the Neural Engine buys little.** On the M4 Pro, full INT8 (W8A8) takes 6.7–8.5% less time than FP16 (1.50–1.53 against 1.64 ms) but loses 0.018–0.108 mAP; keeping 8 layers in FP16 cuts the loss to 0.012 and the speed gain with it (1.62 ms). 8-bit palettization halves the file for 0.0007 mAP at the same Neural Engine speed, though it slows the GPU by 45%. 4-bit and lower, and pruning, break the model without fine-tuning.
- **The INT8 recipe carries over to YOLOv8s, YOLOv8m and YOLO11n** (tuned INT8 within 0.9–1.5% of FP32), **but not to RT-DETR-L** (9% below).

## Results

**Accuracy and timing.**

- mAP50-95 and mAP50 are measured on COCO val2017 (5,000 images). The PyTorch FP32 baseline scores 0.368.
- The two tables below, and the chart, are the 15 Oct passes: each device's first batch in the CSVs. Every latency is the median of that batch's 3 passes. Each pass is 20 warm-up runs followed by 300 timed runs on the Mac, or 10 warm-up and 100 timed runs on the iPhone. The range column shows the fastest and slowest pass.
- Raw numbers: [`results/latency.csv`](results/latency.csv), [`results/accuracy.csv`](results/accuracy.csv), [`results/ios.csv`](results/ios.csv). `latency.csv` and `ios.csv` also hold 16 Oct batches, reported in their own sections: the Mac's in [Beyond the main study](#beyond-the-main-study-mac) and the iPhone's [below](#iphone-18-pro-max-16-oct-15-models-sustained-load-and-the-live-camera); pooling two batches gives medians that match neither.

### MacBook Pro, Apple M4 Pro (model only, Python harness)

The 15 Oct passes (the first batch in `latency.csv`):

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

The 15 Oct passes (the first batch in `ios.csv`):

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

### iPhone 18 Pro Max, 16 Oct: 15 models, sustained load and the live camera

The app as built on 16 Oct bundled 15 models: the three above, `coreml_w8a8` and `coreml_w8a8_head_fp16` from [Full INT8](#full-int8-w8a8-on-the-neural-engine), and the ten models of [Weight compression](#weight-compression). It recorded one benchmark sweep ([`results/ios_raw/`](results/ios_raw/), merged into `ios.csv` and `latency.csv`), five 10-minute sustained runs ([`results/ios_sustained/`](results/ios_sustained/)) and three live camera recordings ([`results/ios_live/`](results/ios_live/)). Read them with these caveats:

- **The sweep is one pass, not three.** 13 of the 15 models were compiled on the phone just before their Neural Engine and ALL rows (first loads of 1.76–3.51 s). Their model-only times there (1.02–1.53 ms) are less trustworthy than their end-to-end times. The chart above keeps the 15 Oct passes.
- **The sustained runs ran back to back,** 16 to 176 s apart, so each started as warm as the previous one left the phone. The FP32 CPU run started at thermal state Fair.
- **The display ran at 60 Hz** (`refresh_ms` 16.67) for all but 13 of the 5,873 live frames, which had 20.83 ms (48 Hz), so on-screen times in the live recordings move in steps of one refresh.
- The phone was charging during every sustained run, and the thermal state is the OS's coarse four-level label.

What they show:

- **The sweep reproduces 15 Oct's FP32 and FP16 rows.** FP16 on the Neural Engine takes 1.03 ms for the model and 1.59 ms end to end (1.01 and 1.61 ms on 15 Oct). Every FP32 and FP16 row is within about 3% on both. INT8 on the Neural Engine reads 1.42 ms model-only, like 15 Oct's first pass, which also compiled (1.41 ms); its end-to-end time is 1.56 ms.
- **No compressed model is meaningfully faster end to end on the Neural Engine.** The 14 models other than FP32 take 1.48–1.62 ms end to end on the Neural Engine and ALL, apart from 6 rows at 2.73–3.08 ms. W8A8 takes 1.55 ms, against FP16's 1.59.
- **Palettized weights cost time on the GPU, as on the Mac.** INT8, PAL8, PAL4 and PAL2 take 4.96–5.36 ms, against 2.36–2.38 ms for FP16, PAL6 and the pruned models. PRUNE50_PAL4 takes 35.96 ms on the CPU, 5.1× FP16's 7.03 ms.
- **On the photo, 2-bit palettization and pruning find none of the 5 objects** (`detections`), matching their Mac mAP. PAL4 finds 3 (0 on the GPU) and PAL4_TENSOR finds 2.
- **Under continuous load, the Neural Engine path ran about 4× slower than in the sweep.** Of the 180 ten-second windows on the Neural Engine (FP16, INT8 and W8A8), 131 took 3.83–4.58 ms for the model and 6.45–7.73 ms end to end. Only 13 matched the sweep (0.89–1.00 and 1.46–1.59 ms). The slow level appeared at thermal Nominal (108 windows) as well as Fair (23), and the work around the model slowed too: end to end minus model was 3.40 ms, against 0.59 ms in the fast windows (medians). The cause is not known. In the four sweep passes (three on 15 Oct, one on 16 Oct), none of the 40 end-to-end loops on the Neural Engine and ALL (FP32 aside) went above 3.08 ms.
- **The GPU slowed as the phone warmed.** FP16 on the GPU went from 2.39–2.41 ms for the model to 3.17 ms in its first window at Fair (60 s), and its last minute was 25% slower than its first (3.01 against 2.40 ms). FP32 on the CPU ranged from 12.31 to 15.65 ms per window with no clear link to the thermal label (its slowest window came at Nominal); its last minute was 6.5% slower than its first (14.35 against 13.48 ms).
- **Live, every recording ran at the camera's 30 fps with no dropped frames,** so live fps does not tell the models apart. Capture to boxes on screen took 55.67 ms (p50) for FP16 on the Neural Engine, 70.15 ms for INT8 on the Neural Engine and 72.92 ms for FP32 on the CPU. Most of that gap is in detection to screen (handing the boxes to the UI and waiting for a 60 Hz refresh), not in the detector: the detector's work per frame differs by only 0.6 and 6.5 ms (medians of 8.92, 9.55 and 15.45 ms), while detection to screen differs by 12.7 and 13.2 ms. The model stage of FP16 on the Neural Engine took 4.74 ms live, close to the sustained runs' slow level (3.83–4.58 ms) and far from the sweep's 1.03 ms.

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

## Beyond the main study (Mac)

A second batch of passes on 16 Oct covers full INT8 and weight compression in Core ML, four more detectors, two more input sizes, quantization-aware training and Ultralytics' own INT8 export. Every configuration again has 3 passes of 300 timed runs, and mAP is on all 5,000 val2017 images.

That batch ran faster than the 15 Oct one behind the tables above: 2.3–3.4% on Core ML (`coreml_fp16` on the Neural Engine: 1.64 against 1.69 ms) and 7.0–7.9% on ONNX Runtime (`onnx_fp32`: 12.53 against 13.61 ms), and the two batches' pass ranges do not overlap. Each table below therefore re-times its reference rows in the same batch. Compare rows within a table, not with the tables above.

### Full INT8 (W8A8) on the Neural Engine

[`bench/coreml_quant.py`](bench/coreml_quant.py) quantizes activations as well as weights: per-channel INT8 weights, and activation ranges from the running min/max over 128 coco128 images. The convolutions can then use INT8 arithmetic, which coremltools' documentation says is faster on the A17 Pro and M4 Neural Engines.

| Variant | Kept in FP16 | mAP50-95 | Neural Engine (ms) | GPU (ms) | CPU (ms) |
|---|---|---|---|---|---|
| `coreml_fp16` | everything | 0.3678 | 1.64 | 3.38 | 9.35 |
| `coreml_w8a8` | nothing | 0.2599 | 1.50 | 3.82 | 14.29 |
| `coreml_w8a8_head_fp16` | the decode | 0.3387 | 1.53 | 3.81 | 14.18 |
| `coreml_w8a8_mixed1` | decode + 1 layer | 0.3476 | 1.52 | 3.80 | 14.35 |
| `coreml_w8a8_mixed4` | decode + 4 layers | 0.3494 | 1.51 | 3.79 | 13.87 |
| `coreml_w8a8_mixed8` | decode + 8 layers | 0.3561 | 1.62 | 3.75 | 13.51 |

- **W8A8 takes 8.5% less time than FP16 on the Neural Engine, but costs accuracy.** 1.50 against 1.64 ms, with pass ranges of 1.49–1.50 and 1.63–1.65 ms. It loses 0.108 mAP with the decode in INT8 and 0.029 with the decode in FP16. With 8 layers in FP16 it still loses 0.012, and the speed gain is gone (1.62 ms). FP16 remains the format for the Neural Engine.
- **On the GPU and CPU, W8A8 is slower than FP16:** by 13% and 53%.
- **The small gain is not explained by ops falling back to the CPU.** Core ML plans all but 2 ops of every W8A8 model on the Neural Engine ([`results/coreml_compute_plan.csv`](results/coreml_compute_plan.csv)). In `coreml_w8a8`, 290 of those 466 ops are quantize or dequantize ops.
- **The layer ranking is weak beyond its first layer** ([`results/coreml_w8a8_loo.csv`](results/coreml_w8a8_loo.csv), first 500 val2017 images). Restoring `model.2.cv1.conv` to FP16 gains 0.0097 mAP and the next seven layers 0.0014–0.0035, while 39 of the 63 layers lose mAP when restored. Better layers for `mixed4` and `mixed8` may exist.

### Weight compression

[`bench/coreml_compress.py`](bench/coreml_compress.py) palettizes the convolution weights (k-means lookup tables), prunes them (zeroes the smallest), or both, with no fine-tuning ([`results/coreml_compress.csv`](results/coreml_compress.csv)). The reference is the same FP16 export with an iOS 18 target, which grouped tables and sparse weights need. Weight error is ‖W′ − W‖ / ‖W‖ over the compressed weights.

| Variant | Method | Size (MB) | Weight error | mAP50-95 | Neural Engine (ms) | GPU (ms) | CPU (ms) |
|---|---|---|---|---|---|---|---|
| `coreml_fp16_ios18` | none | 6.48 | 0 | 0.3678 | 1.64 | 3.52 | 9.69 |
| `coreml_pal8` | 8-bit lookup table | 3.38 | 0.0078 | 0.3671 | 1.65 | 5.12 | 9.64 |
| `coreml_pal6` | 6-bit | 2.57 | 0.0362 | 0.3587 | 1.65 | 3.54 | 9.61 |
| `coreml_pal4` | 4-bit, one table per 16 channels | 1.79 | 0.1402 | 0.1154 | 1.69 | 5.28 | 9.63 |
| `coreml_pal4_tensor` | 4-bit, one table per tensor | 1.78 | 0.1563 | 0.0449 | 1.63 | 5.32 | 9.64 |
| `coreml_pal2` | 2-bit, one table per 16 channels | 0.99 | 0.4753 | 0.0000 | 1.70 | 5.30 | 9.65 |
| `coreml_pal2_tensor` | 2-bit, one table per tensor | 0.98 | 0.4995 | 0.0000 | 1.65 | 5.31 | 9.69 |
| `coreml_prune50` | 50% of weights zeroed | 3.74 | 0.1734 | 0.0015 | 1.56 | 3.51 | 9.65 |
| `coreml_prune75` | 75% zeroed | 2.17 | 0.3661 | 0.0000 | 1.45 | 3.52 | 9.72 |
| `coreml_prune50_pal4` | 50% zeroed, then 4-bit | 1.44 | 0.2077 | 0.0002 | 1.61 | 3.53 | 20.29 |

- **Only 8-bit and 6-bit palettization keep the model working.** 8-bit halves the file for 0.0007 mAP, and 6-bit cuts it to 40% for 0.009. At 4 bits or fewer, and with pruning, detection breaks; these would need fine-tuning.
- **Neural Engine speed barely moves.** Every variant is within 0.06 ms of FP16 except the pruned ones: `coreml_prune75` takes 12% less time and `coreml_prune50` 5% less, outside the pass ranges. Both detect nothing, and a fine-tuned model with the same sparsity was not tested.
- **On the GPU, palettized weights cost time.** 8-, 4- and 2-bit take 5.12–5.32 ms against 3.52 ms, 45–51% more; 6-bit and pruning do not. Weight-only `coreml_int8` in the main table shows the same GPU penalty.
- **`coreml_prune50_pal4` is 2.1× slower than FP16 on the CPU** (20.29 against 9.69 ms).

### Other detectors and input sizes

[`bench/generalize.py`](bench/generalize.py) repeats the INT8 recipe on YOLOv8s, YOLOv8m, YOLO11n and RT-DETR-L at 640, and on YOLOv8n at 320 and 480 px ([`results/generalize.csv`](results/generalize.csv)). [`bench/heads.py`](bench/heads.py) finds each model's decode nodes from its graph. "Naive" is `onnx_int8_static`, "decode FP32" is `onnx_int8_static_head_fp32` and "tuned" is `onnx_int8_tuned`. The YOLOv8n mAPs come from the main study.

| Model | ONNX FP32 | Naive INT8 | INT8, decode FP32 | Tuned INT8 | Tuned vs FP32 | Gap closed by tuning |
|---|---|---|---|---|---|---|
| YOLOv8n | 0.3678 | 0.0000 | 0.3437 | 0.3605 | −2.0% | 70% |
| YOLOv8s | 0.4443 | 0.0000 | 0.4312 | 0.4395 | −1.1% | 63% |
| YOLO11n | 0.3878 | 0.0000 | 0.3445 | 0.3819 | −1.5% | 86% |
| YOLOv8m | 0.4971 | 0.0000 | 0.4869 | 0.4927 | −0.9% | 57% |
| RT-DETR-L | 0.5232 | 0.0000 | 0.4615 | 0.4761 | −9.0% | 24% |

| Model | ONNX FP32 (ms) | Tuned INT8 (ms) | INT8 speedup | Core ML FP16, Neural Engine (ms) | Core ML FP16, ALL (ms) | Neural Engine vs ONNX FP32 |
|---|---|---|---|---|---|---|
| YOLOv8n | 12.53 | 10.52 | 1.19× | 1.64 | 1.62 | 7.6× |
| YOLOv8s | 25.82 | 21.63 | 1.19× | 3.16 | 3.15 | 8.2× |
| YOLO11n | 11.53 | 10.50 | 1.10× | 1.68 | 1.67 | 6.9× |
| YOLOv8m | 57.25 | 48.22 | 1.19× | 6.60 | 6.59 | 8.7× |
| RT-DETR-L | 96.18 | 83.83 | 1.15× | 49.16 | 16.59 | 2.0× |

- **The YOLO family follows YOLOv8n.** Naive INT8 detects nothing on every model, keeping the decode in FP32 fixes it, and tuned INT8 lands within 0.9–2.0% of FP32. Under one shared UINT8 range, the class scores of YOLOv8s, YOLO11n and YOLOv8m span only 0.37–0.38 steps (`score_levels`), so every score rounds to the same value.
- **RT-DETR-L only partly follows it.** Naive INT8 also scores 0, and keeping its 86 decoder-tail nodes in FP32 brings it back. But tuning closes only 24% of the remaining gap, which leaves it 9% below FP32. Its output mixes 0–1 boxes and scores with a class index up to 79, so under one range the scores and the boxes get only about 3 steps each.
- **INT8 on the ONNX Runtime CPU is a small speedup for every model** (1.10–1.19×). **Core ML FP16 on the Neural Engine is 6.9–8.7× faster than ONNX FP32 for every YOLO model**, and scores within 0.0006 mAP of it.
- **RT-DETR-L gains little from the Neural Engine.** `CPU_AND_NE` takes 49.16 ms, only 8% less than the CPU alone (53.33 ms). ALL, which adds the GPU, takes 16.59 ms.

YOLOv8n at other input sizes:

| Input | ONNX FP32 mAP | Tuned INT8 mAP | ONNX FP32 (ms) | Tuned INT8 (ms) | Core ML FP16, Neural Engine (ms) | Core ML FP16, GPU (ms) |
|---|---|---|---|---|---|---|
| 320 | 0.2812 | 0.2762 | 3.89 | 3.45 | 0.66 | 1.62 |
| 480 | 0.3430 | 0.3387 | 7.60 | 6.54 | 2.37 | 2.29 |
| 640 | 0.3678 | 0.3605 | 12.53 | 10.52 | 1.64 | 3.38 |

- **On the CPU, smaller inputs trade smoothly.** 480 saves 39% of the FP32 time for 0.025 mAP, and 320 saves 69% for 0.087. Tuned INT8 loses 0.004–0.007 at every size.
- **On the Neural Engine, 320 is 2.5× faster than 640, but 480 is slower than 640** (2.37 ms, passes 2.33–2.39), although it is faster on the GPU and CPU. That is unexplained and was not checked on the iPhone.

### Quantization-aware training

[`bench/qat.py`](bench/qat.py) fine-tunes YOLOv8n through the same INT8 rounding `onnx_int8_tuned` applies (fake quantization, starting from its ranges), then exports with the same recipe ([`results/qat.csv`](results/qat.csv)). One training run: 3 epochs over 5,000 COCO train2017 images ([`bench/coco_train_subset.py`](bench/coco_train_subset.py)), 936 iterations of batch 16, learning rate 1e-4.

| Model | mAP50-95 | Gap to FP32 | ONNX Runtime CPU (ms) |
|---|---|---|---|
| `onnx_fp32` | 0.3678 | — | 12.53 |
| `onnx_int8_tuned` (post-training INT8) | 0.3605 | −0.0073 | 10.52 |
| `onnx_int8_qat` | 0.3623 | −0.0055 | 10.60 |
| QAT-trained weights, not quantized (`qat_weights_fp32`) | 0.3618 | −0.0060 | not timed |

QAT gains 0.0018 mAP over post-training INT8 at the same speed, about what keeping one layer in FP32 gives (`onnx_int8_mixed1`: 0.3625). From one training run, that is too small to call a win. The fine-tune also cost the FP32 weights 0.006 mAP, so a longer or gentler schedule is the next thing to try.

### Ultralytics' own INT8 export

Since Ultralytics 8.4.91, `export(format="onnx", quantize=8)` keeps every node except `Conv`, `Gemm` and `MatMul` in float, which avoids the 0-mAP failure above ([ultralytics/ultralytics#25046](https://github.com/ultralytics/ultralytics/pull/25046)). Otherwise it uses ONNX Runtime's defaults: per-tensor weights, min/max calibration, and an INT8 DFL convolution. [`bench/upstream_onnx_int8.py`](bench/upstream_onnx_int8.py) measures the export as installed, and with two changes:

| Variant | Change | mAP50-95 | ONNX Runtime CPU (ms) | Speedup vs FP32 |
|---|---|---|---|---|
| `onnx_fp32` | — | 0.3678 | 12.53 | 1.00× |
| `onnx_int8_ultralytics` | as shipped | 0.3409 | 11.50 | 1.09× |
| `onnx_int8_ultralytics_pc` | + per-channel weights | 0.3532 | 11.55 | 1.08× |
| `onnx_int8_ultralytics_pc_dfl` | + DFL convolution kept in float | 0.3574 | 11.79 | 1.06× |
| `onnx_int8_tuned` | this study's recipe | 0.3605 | 10.52 | 1.19× |

- **As shipped, the export loses 0.027 mAP.** Per-channel weights recover 0.0123 and a float DFL another 0.0042: 61% of the gap together.
- **This study's tuned recipe is still 0.0031 mAP better and about 1 ms faster.** The two recipes differ in calibration, in which decode nodes stay in float, and in which non-Conv nodes are quantized. None of these was isolated.
- **Upstream docs:** Ultralytics' ONNX integration page now says what `quantize=8` quantizes, and how to exclude the same nodes when running `quantize_static` on an exported model yourself ([ultralytics/ultralytics#26705](https://github.com/ultralytics/ultralytics/pull/26705), a docs change based on this study, merged on 16 Oct 2026). The per-channel and DFL changes were measured here only.

### Stretched instead of letterboxed input

Ultralytics, and every mAP above, letterboxes the image to 640×640; some deployment pipelines stretch it to the square instead. [`bench/stretch_eval.py`](bench/stretch_eval.py) scores an ONNX model both ways with pycocotools, on the first 300 val2017 images ([`results/stretch_accuracy.csv`](results/stretch_accuracy.csv), where `yolov8n_fp32`, `yolov8n_int8` and `yolov8n_int8_mixed` are the `onnx_fp32`, `onnx_int8_tuned` and `onnx_int8_mixed1` files). Stretching costs 0.028 mAP50-95 in FP32 (0.4370 to 0.4092), 0.032 for tuned INT8 (0.4288 to 0.3966) and 0.040 with one layer in FP32 (0.4350 to 0.3947). These are COCO API numbers on a 300-image subset, so compare the two inputs with each other, not with the tables above.

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
  - Power and energy. [`bench/energy.py`](bench/energy.py) (`make energy`) measures energy per inference from `powermetrics`, but it has not been run yet.

## Running it

```bash
make setup         # Python 3.12 virtualenv + dependencies
make export        # ONNX + Core ML variants into models/
make quantize      # ONNX Runtime INT8 variants (downloads coco128 for calibration)
make coco-val      # COCO val2017 images + labels (~1 GB); make int8, w8a8, compress, generalize, qat, upstream and accuracy-full need them
make int8          # calibration comparison, layer ranking, sweep, tuned + mixed exports (slow: dozens of quantize + validate runs)
make w8a8          # full INT8 (W8A8) Core ML: build, layer ranking, mixed exports, compute plans, mAP (slow)
make compress      # palettized and pruned Core ML models -> results/coreml_compress.csv, then mAP
make generalize    # the INT8 recipe on four more detectors and on YOLOv8n at 320 and 480 px (slow)
make qat           # quantization-aware training (downloads 5,000 COCO train images, ~800 MB; slow)
make upstream      # Ultralytics' own ONNX INT8 export, as shipped and with two changes, then mAP
make latency       # one pass -> results/latency.csv; run it 3 times with nothing else running
make accuracy-full # mAP on COCO val2017 -> results/accuracy.csv
make plot          # -> results/tradeoff.png
make test          # unit tests; no models or root needed
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
  heads.py            finds the output-decode nodes of YOLO and RT-DETR ONNX graphs
  generalize.py       the INT8 recipe on other detectors and input sizes
  qat.py              quantization-aware training through onnx_int8_tuned's rounding
  coco_train_subset.py downloads a fixed 5,000-image subset of COCO train2017 for QAT
  upstream_onnx_int8.py Ultralytics' built-in ONNX INT8 export, as shipped and with two changes
  latency.py          timing harness, one runner per backend / compute unit
  accuracy.py         mAP for every variant
  coco_val.py         downloads COCO val2017 (images + labels) only
  merge_ios.py        adds an EdgeDetect export to the results
  plot.py             latency and mAP chart, one panel per device
  energy.py           energy per inference from powermetrics (needs root)
  stretch_eval.py     COCO mAP with the input stretched or letterboxed (pycocotools)
tests/                unit tests (python -m unittest)
ios/                  EdgeDetect: live detection + on-device benchmark (SwiftUI, Core ML)
data/                 dataset configs: COCO val2017, and the COCO train2017 subset for QAT
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

- **Mostly one model:** YOLOv8n at 640×640, batch size 1. Four more detectors and two more input sizes were measured on the Mac only, for the INT8 recipe and Core ML FP16.
- **INT8 on ONNX was only measured on the M4 Pro's CPU**, where ONNX Runtime's INT8 kernels give a modest speedup (1.2×). ARM boards and NPUs with dedicated INT8 hardware may differ.
- **Full INT8 (W8A8) in Core ML used one calibration method** (running min/max over 128 images), and its layer ranking is noisy beyond the first layer, so a better W8A8 model may exist.
- **The 16 Oct iPhone data is one sweep pass and back-to-back sustained runs**, and the sustained Neural Engine slowdown is unexplained.
- **Compression had no fine-tuning, and QAT is one 3-epoch run.** The broken low-bit and pruned models, and QAT's small gain, show what these methods do without more training, not their best case.
