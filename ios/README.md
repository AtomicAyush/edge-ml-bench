# EdgeDetect (iPhone)

The Python benchmark times the model call alone. An app pays for more than that: getting the camera frame into the model's 640×640 input, the model itself, and turning 8,400 raw candidate boxes into a handful of detections. EdgeDetect runs YOLOv8n live on the iPhone camera and times all three stages, on every bundled Core ML model and compute unit.

**Models:** every `.mlmodelc` in the app bundle appears in the pickers, so a new export (W8A8, palettized, mixed precision, another input size) only needs copying into `EdgeDetect/MLModels` and a rebuild. Xcode compiles `MLModels/YOLOv8n_<NAME>.mlpackage` to `YOLOv8n_<NAME>.mlmodelc`; the app shows it as `<NAME>` and writes it to CSVs as variant `coreml_<name>`, the folder it came from under `models/`. FP32, FP16 and INT8 come first, then the rest by name. The input size is read from the model, so a 320 px export letterboxes to 320. The decoder reads Float32 outputs (every Ultralytics export) and Float16 outputs (the iOS 18 exports from `bench.coreml_compress`: palettized, pruned and `fp16_ios18`); any other output type is refused with an error rather than misread. For a Float16 model, Post includes widening the output to Float32 as it is read.

**Live tab:** the camera feed with boxes, two switches, and stage timings.

- **Switches:**
  - Model: every bundled model (a segmented control for up to four, a menu beyond that).
  - Compute unit: CPU / GPU / ANE (the Neural Engine) / All.
- **Stage timings,** each the median of the last 30 frames:
  - **Pre:** letterbox the 1280×720 frame into 640×640 with Core Image.
  - **Model:** the Core ML prediction.
  - **Post:** decode the `[1, 84, 8400]` output and run non-maximum suppression, written in Swift (`Detector.swift`).
  - **FPS:** frames actually processed per second. Late camera frames are dropped, not queued.
- **Camera to boxes:** median and p95 over the last 120 frames of the time from a frame's capture to the screen refresh that first shows its boxes, plus how many frames the camera dropped (`captureOutput(_:didDrop:from:)`) out of all it produced. These restart when the model or compute unit changes, when the tab is reopened and when the app returns from the background, and frames that waited out a model load are not counted.
  - All three timestamps are on the host clock (`mach_absolute_time`): the frame's presentation timestamp, converted from the capture session's synchronization clock with `CMSyncConvertTime`; detector start and end from `CACurrentMediaTime()`; and the on-screen time, which is `CADisplayLink.targetTimestamp` of the first refresh after the main thread hands the boxes to SwiftUI. That is when the refresh is due on the display, so the figure is good to one refresh interval (saved per frame as `refresh_ms`).
  - The red button records every timed frame. Tap it again, leave the tab, or leave the app, and the recording is saved to Documents as `live-<unix time>.csv` and offered through the share button.
- **Model load time and thermal state.** The first Neural Engine load includes an on-device compile. Thermal state shows when the phone starts throttling.

**Benchmark tab:**

- Runs every bundled model on every compute unit against a fixed photo (`bus.jpg`): 10 warm-up runs, then 100 timed runs.
- Records both the first model load (which may include a one-time Neural Engine compile) and an immediate warm reload.
- Reports model-only latency and end-to-end latency. Model-only covers just the Core ML call. The Mac's Python harness also pays for converting inputs and outputs, so compare phone runs with each other rather than with the Mac's Python rows.
- Exports the table as CSV through the share sheet.
- **Sustained** runs one model on one compute unit back to back on the photo (letterbox, model, decode, NMS) for a chosen time, 10 minutes by default, with the screen kept on. After 10 warm-up runs it closes a window every 10 s and logs:
  - the median Core ML time and the median and p95 end-to-end time over the window;
  - `ProcessInfo.thermalState`, battery level and charging state (`UIDevice` battery monitoring), and Low Power Mode.
  - Windows are contiguous: the measuring loop never waits for the UI. Each window is written to Documents as `sustained-<unix time>.csv` as it completes, so a run that is stopped, interrupted or killed keeps what it measured. Leaving the app stops the run at once and drops any window not yet recorded. A chart shows the medians over the run, with the first and latest window and when the thermal state changed.
  - Only one of the sweep and the sustained run can run at a time, and the Live tab and sample photo stay idle while either runs. Stay on the Benchmark tab: the Live tab's camera would add load.
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

`make ios-models` copies `models/coreml_<name>/yolov8n.mlpackage` to `MLModels/YOLOv8n_<NAME>.mlpackage`, where `<NAME>` is the folder name without `coreml_`, upper-cased: `coreml_fp16` becomes `YOLOv8n_FP16`, `coreml_w8a8` becomes `YOLOv8n_W8A8`. It takes every such folder except the 320 and 480 px exports and the W8A8 mixed-precision ones (`coreml_w8a8_mixed*`). After `make export compress w8a8` that is the 15 models measured on 16 Oct; after `make export` alone it is FP32, FP16 and INT8. `make ios-models IOS_MODELS="coreml_fp16 coreml_w8a8_mixed8"` copies a list of your own. It replaces only the models it copies, so a model dropped into `MLModels` by hand stays: to reproduce the 16 Oct bundle exactly, start from an empty `MLModels` folder.

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

`sustained-*.csv` and `live-*.csv` come off the phone the same way. `bench.merge_ios` takes only the benchmark CSVs. It refuses the other two, because they lack the latency columns. Their columns:

| File | One row per | Columns |
|---|---|---|
| `sustained-<time>.csv` | 10 s window | `timestamp` (window end), `device`, `variant`, `backend`, `window`, `elapsed_s`, `window_s`, `iterations`, `model_p50_ms`, `end_to_end_p50_ms`, `end_to_end_p95_ms`, `fps`, `thermal`, `battery_level` (0–1, empty if unknown), `battery_state`, `low_power` |
| `live-<time>.csv` | processed camera frame | `timestamp` (capture), `device`, `variant`, `backend`, `frame`, `t_s` (since the first recorded frame), `capture_to_detect_ms`, `pre_ms`, `model_ms`, `post_ms`, `detect_to_screen_ms`, `capture_to_boxes_ms`, `refresh_ms`, `dropped_before` (camera drops since the previous processed frame), `thermal` |

`model_p50_ms` in the sustained file and `model_ms` in the live file time the detector's Core ML stage, as the Live tab's Model column does: the prediction plus wrapping the input and reading the output. The benchmark sweep's `p50_ms` times the prediction alone, so it covers slightly less work. `capture_to_detect_ms` should be positive (camera pipeline plus waiting for the detector queue); a negative value would mean the clocks disagree. Empty on-screen columns in the live file mean a newer frame's boxes replaced these before any refresh showed them.

The Simulator works for the Benchmark tab, but it has no camera, and Core ML runs on the Mac's hardware rather than the phone's, so its numbers say nothing about a phone. For a quick check of the sustained mode, launch arguments shorten the run and its windows:

```bash
xcrun simctl launch booted com.atomicayush.EdgeDetect -sustainedSeconds 0.3 -sustainedWindowSeconds 0.1
```

## Layout

| File | What it does |
|---|---|
| `EdgeDetectApp.swift` | The app: the Live and Benchmark tabs, Live first |
| `Detector.swift` | The bundled-model list; letterbox → Core ML → decode → NMS, with per-stage timing |
| `Camera.swift` | `AVCaptureSession`, 720p BGRA frames, upright, late frames dropped; capture times on the host clock and drop reports |
| `LiveDetection.swift` | Runs the detector on each frame; rolling medians, FPS, thermal state, camera-to-boxes latency and recording |
| `LiveView.swift` | Camera preview, box overlay, controls |
| `BenchmarkView.swift` | The model × compute-unit sweep and CSV export |
| `Sustained.swift` | Sustained mode: one model for N minutes, 10 s windows with thermal state and battery |
| `SamplePhotoView.swift` | Detections on the sample photo, for checking against Python |
| `CocoNames.swift` | The 80 COCO class names, in the model's output order |
