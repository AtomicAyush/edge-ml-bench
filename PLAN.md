# Plan

## Question

For a small real-time detector, which levers move latency the most: format, precision, or compute unit? And what does each lever cost in accuracy?

## Experiment matrix

✅ = run, ☐ = to do, — = not applicable.

| Variant | Mac: ORT CPU | Mac: Core ML CPU / GPU / NE / ALL | Pi: ORT CPU | Pi: LiteRT | mAP (coco128) | mAP (val2017) |
|---|---|---|---|---|---|---|
| pytorch_fp32 | — | — | — | — | ✅ | ✅ |
| onnx_fp32 | ✅ | — | ☐ | — | ✅ | ✅ |
| onnx_int8_dynamic | ✅ | — | ☐ | — | ✅ | ✅ |
| onnx_int8_static | ✅ | — | ☐ | — | ✅ | ✅ |
| onnx_int8_static_head_fp32 | ✅ | — | ☐ | — | ✅ | ✅ |
| coreml_fp32 | — | ✅ | — | — | ✅ | ✅ |
| coreml_fp16 | — | ✅ | — | — | ✅ | ✅ |
| coreml_int8 | — | ✅ | — | — | ✅ | ✅ |
| tflite_fp32 | — | — | — | ☐ | ☐ | ☐ |
| tflite_int8 | — | — | — | ☐ | ☐ | ☐ |

Things to keep fixed or note in every run:

- The machine is plugged in, nothing else heavy is running, and Low Power Mode is off.
- Each configuration runs at least 3 times. Look at the spread between runs before trusting any difference smaller than about 10%.
- On the Pi, note the cooling (fan or no fan) and check `vcgencmd measure_temp` after each run. Thermal throttling will show up as a p95 much higher than p50.

## Milestones

1. **Harness works end to end on the Mac.**
   - [x] export, quantize, latency, accuracy and plot all run.
2. **Full Mac runs.**
   - [x] `make export quantize latency accuracy plot`: 3 passes × 300 timed runs per configuration.
3. **Full COCO accuracy.**
   - [x] `make coco-val accuracy-full`: COCO val2017 only, about 1 GB, roughly 15 minutes on the M4 Pro.
4. **Raspberry Pi runs** (steps below).
5. **Batch size and input size sweep** (optional).
   - [x] Repeat the best two or three configurations at 320 and 480 px.
   - Smaller inputs trade accuracy for speed differently from quantization, which makes a good comparison.
6. **INT8 accuracy.**
   - [x] Calibration comparison, leave-one-out layer ranking, mixed-precision sweep (`make int8`).
7. **iPhone.**
   - [x] EdgeDetect app: live detection with stage timings, on-device benchmark, 3 runs merged.
8. **Write-up.**
   - [x] README results and findings.
   - [ ] Write a short blog post or report following the outline below.
9. **Stretch goals.**
   - [x] End-to-end latency, including preprocessing and NMS (iPhone app).
   - [x] Live camera demo with FPS for each precision and compute unit (iPhone app).
   - [x] Core ML activation quantization (W8A8) on the Neural Engine.
   - [x] Add a second model (YOLOv8s or MobileNet-SSD) to see if the conclusions hold.

## Raspberry Pi

Models are exported and quantized on the Mac, then copied to the Pi. The Pi only runs the latency harness.

**On the Mac:**

```bash
make export quantize
.venv/bin/python -m bench.export --only tflite_fp32 tflite_int8
rsync -av --exclude .venv --exclude '*.pt' --exclude results ./ pi@raspberrypi.local:edge-ml-bench/
```

**On the Pi** (64-bit Raspberry Pi OS):

```bash
cd edge-ml-bench
python3 -m venv .venv && .venv/bin/pip install -r requirements-pi.txt
rm -rf models/coreml_*                 # Core ML is Apple-only
.venv/bin/python -m bench.latency --runs 100
vcgencmd measure_temp
```

**Back on the Mac:** copy `results/latency.csv` back from the Pi and merge it into the main file:

```bash
scp pi@raspberrypi.local:edge-ml-bench/results/latency.csv /tmp/pi.csv
tail -n +2 /tmp/pi.csv >> results/latency.csv
make plot
```

## Write-up outline

1. **Why it matters.** Detection on the device itself means lower latency, offline operation and privacy. But "just quantize it" is not free.
2. **Setup.** The model, the variants, the hardware, and how timing and accuracy were measured, including what's excluded and why.
3. **Results.** The trade-off chart, plus the table.
4. **What moved latency the most.**
   - Compute unit (CPU → GPU → Neural Engine) against precision (FP32 → FP16 → INT8).
   - Which of those mattered more, and on which device.
5. **What it cost.** mAP change for each variant. Where INT8 hurt and why.
   - Look closely at the detect-head result: naive static INT8 against keeping the head in FP32.
6. **Surprises and dead ends.**
7. **Recommendation.** What you'd ship on an iPhone, and what you'd ship on a Pi.

## Résumé bullet

> **Edge ML Inference Benchmark** — Benchmarked YOLOv8n at FP32/FP16/INT8 on the CPU, GPU and Neural Engine (8× faster at equal mAP). Fixed INT8's 0 mAP and closed 78% of its remaining accuracy gap with per-channel calibration and mixed precision. iPhone app: 1.6 ms image-to-boxes.

78% = (0.3625 − 0.3437) / (0.3678 − 0.3437): from INT8 with the FP32 decode (min/max calibration) to `onnx_int8_mixed1`, relative to FP32.
