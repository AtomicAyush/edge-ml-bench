# EdgeDetect (iPhone)

The Python benchmark times the model call alone. An app pays for more than that: getting the camera frame into the model's 640×640 input, the model itself, and turning 8,400 raw candidate boxes into a handful of detections. EdgeDetect runs YOLOv8n live on the iPhone camera and times all three stages, on every Core ML precision and compute unit.

**Live tab:** the camera feed with boxes, two switches, and stage timings.

- **Switches:**
  - Precision: FP32 / FP16 / INT8.
  - Compute unit: CPU / GPU / Neural Engine / All.
- **Stage timings,** each the median of the last 30 frames:
  - **Pre:** letterbox the 1280×720 frame into 640×640 with Core Image.
  - **Model:** the Core ML prediction.
  - **Post:** decode the `[1, 84, 8400]` output and run non-maximum suppression, written in Swift (`Detector.swift`).
  - **FPS:** frames actually processed per second. Late camera frames are dropped, not queued.
- **Model load time and thermal state.** The first Neural Engine load includes an on-device compile. Thermal state shows when the phone starts throttling.

**Benchmark tab:**

- Runs every precision on every compute unit against a fixed photo (`bus.jpg`): 10 warm-up runs, then 100 timed runs.
- Records both the first model load (which may include a one-time Neural Engine compile) and an immediate warm reload.
- Reports model-only latency and end-to-end latency. Model-only covers just the Core ML call. The Mac's Python harness also pays for converting inputs and outputs, so compare phone runs with each other rather than with the Mac's Python rows.
- Exports the table as CSV through the share sheet.
- The **Sample photo** section draws the detections on `bus.jpg` and lists their pixel coordinates. A hand check against the Python Core ML pipeline (`YOLO("models/coreml_fp16/yolov8n.mlpackage").predict("bus.jpg", iou=0.45)`) found the same five detections, one bus and four people, with box corners within a pixel. The app's NMS uses an IoU threshold of 0.45; Ultralytics' default is 0.7.

## Results (iPhone 18 Pro Max, hardware ID iPhone19,3)

Median of 3 benchmark runs:

- **FP16 on the Neural Engine:** 1.01 ms for the model, 1.61 ms end to end (letterbox, model, decode and NMS on the 810×1080 sample photo).
- **FP32 on the CPU:** 12.18 ms for the model.
- **FP32 on the Neural Engine:** 12.28 ms. It runs at CPU speed, with no error from Core ML.
- **Weight-only INT8:** the same speed as FP16 on the Neural Engine (0.98 ms).
- **Loads:** the first Neural Engine load of the INT8 model after installing took 2.07 s (on-device compile). Cached loads take about 13 ms on the Neural Engine and under 3 ms on the CPU and GPU. FP16 has no cold row, because both of its Neural Engine settings are compiled before the run starts. The Live tab, the default tab, loads FP16 on the Neural Engine at launch. The Sample photo section loads FP16 on All when the Benchmark tab opens. The compile is cached separately for each compute-unit setting: INT8 compiled again on All (2.07 s) right after compiling on the Neural Engine.

The full table is in the [main README](../README.md#iphone-18-pro-max-edgedetect-app-swift-harness).

## Running it

```bash
make export ios-models    # from the repo root: Core ML models + sample photo into ios/EdgeDetect/MLModels
open ios/EdgeDetect.xcodeproj
```

Then pick your iPhone as the run destination and set your own team under Signing & Capabilities.

Each benchmark run also saves its CSV in the app's Documents folder. Copy it to the Mac and add it to the results:

```bash
xcrun devicectl device info files --device <UDID> --domain-type appDataContainer \
    --domain-identifier com.atomicayush.EdgeDetect --subdirectory Documents
xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer \
    --domain-identifier com.atomicayush.EdgeDetect --source "Documents/<file>.csv" --destination run.csv
python -m bench.merge_ios run.csv     # from the repo root
```

Get the UDID from `xcrun devicectl list devices`. Run the benchmark at least 3 times: the chart wants 3 passes per configuration, and a phone's first run includes one-time Neural Engine compiles.

The Simulator works for the Benchmark tab, but it has no camera, and Core ML runs on the Mac's hardware rather than the phone's, so its numbers say nothing about a phone.

## Layout

| File | What it does |
|---|---|
| `Detector.swift` | Letterbox → Core ML → decode → NMS, with per-stage timing |
| `Camera.swift` | `AVCaptureSession`, 720p BGRA frames, upright, late frames dropped |
| `LiveDetection.swift` | Runs the detector on each frame; rolling medians, FPS, thermal state |
| `LiveView.swift` | Camera preview, box overlay, controls |
| `BenchmarkView.swift` | The precision × compute-unit sweep and CSV export |
| `SamplePhotoView.swift` | Detections on the sample photo, for checking against Python |
