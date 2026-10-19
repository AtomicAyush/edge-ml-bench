import Charts
import CoreVideo
import SwiftUI
import UIKit

/// One 10-second window of a sustained run.
struct SustainedWindow: Identifiable {
    var id: Int { index }
    let index: Int
    let timestamp: Date             // end of the window
    let elapsedSeconds: Double      // from the start of the timed loop to the end of this window
    let seconds: Double             // length of this window
    let iterations: Int
    let modelP50: Double            // the Core ML stage of detect(), as in the Live tab's Model column
    let endToEndP50: Double         // letterbox + model + decode + NMS
    let endToEndP95: Double
    let thermal: ProcessInfo.ThermalState
    let batteryLevel: Float         // 0...1, or -1 when unknown (the Simulator)
    let batteryState: UIDevice.BatteryState
    let lowPower: Bool
    var fps: Double { Double(iterations) / seconds }
}

/// Raw milliseconds from one window, stamped when the window closed, handed from the
/// measuring task to the main actor.
struct WindowSamples: Sendable {
    let closed: Date
    let elapsedSeconds: Double, seconds: Double
    let model: [Double], endToEnd: [Double]
    let thermal: ProcessInfo.ThermalState
}

/// Runs one model on one compute unit back to back on bus.jpg for a fixed time, logging
/// latency, thermal state and battery every 10 s, to show whether the sweep's numbers
/// hold once the phone warms up. Each completed window is written to
/// Documents/sustained-<unix time>.csv as it finishes, so a run cut short keeps its data.
@MainActor
final class SustainedRun: ObservableObject {
    @Published var model: BundledModel = .preferred
    @Published var unit: ComputeUnit = .neuralEngine
    @Published var minutes = 10
    @Published private(set) var windows: [SustainedWindow] = []
    @Published private(set) var isRunning = false
    @Published private(set) var elapsedSeconds = 0.0
    @Published private(set) var plannedSeconds = 600.0
    @Published private(set) var windowSeconds = 10.0
    @Published private(set) var status: String?
    @Published private(set) var statusIsError = false
    @Published private(set) var fileName: String?

    private var worker: Task<String?, Never>?
    private var stopRequested = false
    private var interrupted = false
    // The configuration of the current or last run, for the CSV.
    private var runModel = BundledModel.preferred
    private var runUnit = ComputeUnit.neuralEngine

    var csv: String {
        let header = ["timestamp", "device", "variant", "backend", "window", "elapsed_s", "window_s", "iterations",
                      "model_p50_ms", "end_to_end_p50_ms", "end_to_end_p95_ms", "fps", "thermal",
                      "battery_level", "battery_state", "low_power"]
        let iso = ISO8601DateFormatter()
        let device = Benchmark.deviceModel, backend = "coreml-swift-\(runUnit.pythonName)"
        let lines = windows.map { w in
            [iso.string(from: w.timestamp), device, runModel.variant, backend, "\(w.index)",
             String(format: "%.1f", w.elapsedSeconds), String(format: "%.2f", w.seconds), "\(w.iterations)",
             fmt(w.modelP50), fmt(w.endToEndP50), fmt(w.endToEndP95), String(format: "%.1f", w.fps),
             w.thermal.label, w.batteryLevel < 0 ? "" : String(format: "%.2f", w.batteryLevel),
             w.batteryState.label, w.lowPower ? "1" : "0"]
                .map(Benchmark.csvField).joined(separator: ",")
        }
        return ([header.joined(separator: ",")] + lines).joined(separator: "\n") + "\n"
    }

    func run() async {
        guard !isRunning, !BenchmarkActivity.isRunning else { return }
        windows = []
        status = nil
        statusIsError = false
        fileName = nil
        elapsedSeconds = 0
        stopRequested = false
        interrupted = false
        guard let frame = Benchmark.sampleFrame() else {
            fail("bus.jpg is missing from the bundle.")
            return
        }
        let model = model, unit = unit
        let total = Self.override("sustainedSeconds") ?? Double(minutes) * 60
        let window = Self.override("sustainedWindowSeconds") ?? 10
        runModel = model
        runUnit = unit
        plannedSeconds = total
        windowSeconds = window
        fileName = "sustained-\(Int(Date().timeIntervalSince1970)).csv"

        // As in the sweep: leaving the foreground would put suspended time into a sample, and
        // the GPU is off limits in the background. Stop at once; completed windows stay saved.
        let observer = NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.interrupted = true
                self?.worker?.cancel()
            }
        }
        let wasMonitoringBattery = UIDevice.current.isBatteryMonitoringEnabled
        UIDevice.current.isBatteryMonitoringEnabled = true
        BenchmarkActivity.isRunning = true
        UIApplication.shared.isIdleTimerDisabled = true
        isRunning = true
        defer {
            NotificationCenter.default.removeObserver(observer)
            UIDevice.current.isBatteryMonitoringEnabled = wasMonitoringBattery
            BenchmarkActivity.isRunning = false
            UIApplication.shared.isIdleTimerDisabled = false
            isRunning = false
            worker = nil
        }

        // Windows go through a stream, so the measuring loop never waits for the main thread
        // and runs back to back; this loop records them in order as the main thread frees up.
        let (stream, windowsOut) = AsyncStream.makeStream(of: WindowSamples.self)
        let worker = Task.detached(priority: .userInitiated) { () -> String? in
            defer { windowsOut.finish() }
            do {
                try Self.measure(model, unit, frame: frame, seconds: total, window: window) { windowsOut.yield($0) }
                return nil
            } catch is CancellationError {
                return nil
            } catch {
                return error.localizedDescription
            }
        }
        self.worker = worker
        if interrupted { worker.cancel() }
        for await samples in stream { record(samples) }
        let failure = await worker.value

        let saved = windows.isEmpty ? "Nothing was saved." : "\(windows.count) windows saved to Documents/\(fileName ?? "")."
        if windows.isEmpty { fileName = nil }
        if interrupted {
            fail("Stopped: the app left the foreground. Windows not yet recorded by then were dropped. " + saved)
        } else if let failure {
            fail("Stopped by an error: \(failure) " + saved)
        } else if stopRequested {
            status = "Stopped early. " + saved
        } else {
            status = "Done. " + saved
        }
    }

    func stop() {
        stopRequested = true
        worker?.cancel()
    }

    /// Smoke-test overrides from launch arguments, e.g. a run of a few dozen iterations:
    /// `xcrun simctl launch booted com.atomicayush.EdgeDetect -sustainedSeconds 0.3 -sustainedWindowSeconds 0.1`
    private static func override(_ key: String) -> Double? {
        let value = UserDefaults.standard.double(forKey: key)
        return value > 0 ? value : nil
    }

    private func fail(_ message: String) {
        status = message
        statusIsError = true
    }

    /// Back-to-back detect() calls, closing a window every `window` seconds; windows are
    /// contiguous, so the run has no idle gaps. A window cut short by cancellation is dropped.
    nonisolated private static func measure(_ bundled: BundledModel, _ unit: ComputeUnit, frame: CVPixelBuffer,
                                            seconds total: Double, window windowSeconds: Double,
                                            emit: (WindowSamples) -> Void) throws {
        let detector = try Detector(model: bundled, unit: unit)
        for _ in 0..<10 { _ = try detector.detect(frame) }   // warm-up, as in the sweep
        let start = Clock.now
        var windowStart = start
        var model: [Double] = [], endToEnd: [Double] = []
        while true {
            try Task.checkCancellation()
            let (_, t) = try detector.detect(frame)
            model.append(t.inference)
            endToEnd.append(t.total)
            let now = Clock.now
            let seconds = Double(now - windowStart) / 1e9
            guard seconds >= windowSeconds else { continue }
            let elapsed = Double(now - start) / 1e9
            emit(WindowSamples(closed: Date(), elapsedSeconds: elapsed, seconds: seconds, model: model,
                               endToEnd: endToEnd, thermal: ProcessInfo.processInfo.thermalState))
            if elapsed >= total { return }
            model.removeAll(keepingCapacity: true)
            endToEnd.removeAll(keepingCapacity: true)
            windowStart = now
        }
    }

    private func record(_ samples: WindowSamples) {
        // Battery is read here, normally within milliseconds of the window closing. After
        // the app has left the foreground it would be read late, so such windows are dropped.
        guard !interrupted else { return }
        let model = samples.model.sorted(), endToEnd = samples.endToEnd.sorted()
        let device = UIDevice.current
        windows.append(SustainedWindow(
            index: windows.count + 1, timestamp: samples.closed, elapsedSeconds: samples.elapsedSeconds,
            seconds: samples.seconds, iterations: endToEnd.count,
            modelP50: Benchmark.median(model), endToEndP50: Benchmark.median(endToEnd),
            endToEndP95: endToEnd[Int(Double(endToEnd.count - 1) * 0.95)],
            thermal: samples.thermal, batteryLevel: device.batteryLevel,
            batteryState: device.batteryState, lowPower: ProcessInfo.processInfo.isLowPowerModeEnabled))
        elapsedSeconds = samples.elapsedSeconds
        // Rewritten after every window, so the file is complete up to the last window
        // even if the run is stopped, interrupted or the app is killed.
        if let fileName, let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            try? csv.write(to: dir.appendingPathComponent(fileName), atomically: true, encoding: .utf8)
        }
    }

    private func fmt(_ v: Double) -> String { String(format: "%.2f", v) }
}

extension UIDevice.BatteryState {
    var label: String {
        switch self {
        case .unplugged: "unplugged"
        case .charging: "charging"
        case .full: "full"
        case .unknown: "unknown"
        @unknown default: "unknown"
        }
    }
}

/// The Sustained section of the Benchmark tab.
struct SustainedSection: View {
    @ObservedObject var run: SustainedRun
    /// The sweep is running; both use the Neural Engine, so only one runs at a time.
    let sweepRunning: Bool

    var body: some View {
        Section {
            Group {
                Picker("Model", selection: $run.model) {
                    ForEach(BundledModel.all) { Text($0.name).tag($0) }
                }
                Picker("Compute unit", selection: $run.unit) {
                    ForEach(ComputeUnit.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                Stepper("Duration \(run.minutes) min", value: $run.minutes, in: 1...60)
            }
            .disabled(run.isRunning)
            Button(run.isRunning ? "Stop" : "Run sustained") {
                if run.isRunning { run.stop() } else { Task { await run.run() } }
            }
            .disabled(!run.isRunning && sweepRunning)
            if run.isRunning {
                ProgressView(value: min(run.elapsedSeconds, run.plannedSeconds), total: run.plannedSeconds) {
                    Text("Window \(run.windows.count) of \(Int(run.plannedSeconds / run.windowSeconds))")
                        .font(.caption)
                }
            }
            if !run.windows.isEmpty {
                SustainedChart(windows: run.windows)
                SustainedSummary(windows: run.windows)
            }
            if let status = run.status {
                Text(status).font(.caption).foregroundStyle(run.statusIsError ? .red : .secondary)
            }
            if !run.isRunning, !run.windows.isEmpty {
                ShareLink(item: run.csv, preview: SharePreview(run.fileName ?? "sustained.csv")) {
                    Label("Share CSV", systemImage: "square.and.arrow.up")
                }
            }
        } header: {
            Text("Sustained")
        } footer: {
            Text("One model on one compute unit, back to back on the photo (letterbox, model, decode, NMS) "
                 + "with the screen kept on. Every 10 s it logs the median latency, thermal state and battery, "
                 + "and saves them to Documents/sustained-<time>.csv. Stay on this tab: leaving the app stops the run.")
        }
    }
}

/// Median latency per window over the run.
struct SustainedChart: View {
    let windows: [SustainedWindow]

    var body: some View {
        Chart(windows) { w in
            LineMark(x: .value("Minutes", w.elapsedSeconds / 60), y: .value("ms", w.endToEndP50),
                     series: .value("Stage", "End to end"))
                .foregroundStyle(by: .value("Stage", "End to end"))
            LineMark(x: .value("Minutes", w.elapsedSeconds / 60), y: .value("ms", w.modelP50),
                     series: .value("Stage", "Model"))
                .foregroundStyle(by: .value("Stage", "Model"))
        }
        .chartXAxisLabel("minutes")
        .chartYAxisLabel("median ms")
        .frame(height: 160)
    }
}

/// First and latest window, and when the thermal state changed.
struct SustainedSummary: View {
    let windows: [SustainedWindow]

    var body: some View {
        if let first = windows.first, let last = windows.last {
            VStack(alignment: .leading, spacing: 4) {
                Text(String(format: "End to end %.2f → %.2f ms (%+.0f%%) · model %.2f → %.2f ms",
                            first.endToEndP50, last.endToEndP50, 100 * (last.endToEndP50 / first.endToEndP50 - 1),
                            first.modelP50, last.modelP50))
                Text("Thermal: " + thermalChanges)
                    .foregroundStyle(last.thermal.rawValue >= 2 ? .red : .secondary)
                if last.batteryLevel >= 0 {
                    Text(String(format: "Battery %d%% → %d%% · %@", Int(first.batteryLevel * 100),
                                Int(last.batteryLevel * 100), last.batteryState.label))
                        .foregroundStyle(.secondary)
                }
            }
            .font(.caption.monospacedDigit())
        }
    }

    /// "Nominal, Fair at 4:10, Serious at 7:30".
    private var thermalChanges: String {
        var parts: [String] = []
        var previous: ProcessInfo.ThermalState?
        for w in windows where w.thermal != previous {
            let t = Int(w.elapsedSeconds)
            parts.append(previous == nil ? w.thermal.label : "\(w.thermal.label) at \(t / 60):\(String(format: "%02d", t % 60))")
            previous = w.thermal
        }
        return parts.joined(separator: ", ")
    }
}
