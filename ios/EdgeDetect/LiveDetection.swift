import AVFoundation
import QuartzCore
import SwiftUI
import UIKit

/// Runs the detector on every camera frame it can keep up with and publishes the
/// boxes plus rolling per-stage timings. All detector work happens on `queue`.
///
/// It also times each frame from capture to the screen refresh that first shows its boxes.
/// Three timestamps, all on the host clock (mach_absolute_time, in seconds):
/// - captured: the sample buffer's presentation time, converted to the host clock (Camera).
/// - detect start and end: CACurrentMediaTime() on the detector queue.
/// - on screen: CADisplayLink.targetTimestamp of the first refresh after the main thread
///   handed the boxes to SwiftUI, which is when that refresh is due on the display.
/// Its resolution is one refresh interval, saved per frame as `refresh_ms`.
@MainActor
final class LiveDetection: ObservableObject {
    @Published var model: BundledModel = .preferred { didSet { reload() } }
    @Published var unit: ComputeUnit = .neuralEngine { didSet { reload() } }
    @Published private(set) var detections: [Detection] = []
    @Published private(set) var frameSize = CGSize(width: 720, height: 1280)
    @Published private(set) var stats = Timings()
    @Published private(set) var fps = 0.0
    @Published private(set) var latency = CameraLatency()
    @Published private(set) var loadMilliseconds = 0.0
    @Published private(set) var thermal = ProcessInfo.processInfo.thermalState
    @Published private(set) var cameraMessage: String?
    @Published private(set) var modelError: String?
    @Published private(set) var isRecording = false
    /// The last recording, also saved to Documents, for the share sheet.
    @Published private(set) var export: LiveExport?
    var message: String? { modelError ?? cameraMessage }

    let queue = DispatchQueue(label: "detector", qos: .userInitiated)
    private(set) lazy var camera = Camera(
        queue: queue,
        onFrame: { [weak self] frame, captured in self?.process(frame, capturedAt: captured) },
        onDrop: { [weak self] captured in self?.dropped(capturedAt: captured) })

    private var requestedModel = false
    private var startToken = 0

    // Camera-to-boxes timing, on the main thread.
    private var displayLink: CADisplayLink?
    private var awaitingDisplay: FrameLatency?
    private var latencyWindow = LatencySamples(capacity: 120)
    /// Frames captured before this host time are not timed: they waited out a model load,
    /// or were captured before the app went to the background.
    private var countFrom = 0.0
    private var recorded: [FrameLatency] = []
    private var recordingStarted = (wall: Date(), host: 0.0)

    // Touched only on `queue`.
    nonisolated(unsafe) private var detector: Detector?
    nonisolated(unsafe) private var window = RollingWindow(capacity: 30)
    nonisolated(unsafe) private var drops = DropCount()

    func start() async {
        if !requestedModel { reload() }
        if displayLink == nil {
            let link = CADisplayLink(target: DisplayTarget { [weak self] in self?.refreshed($0) },
                                     selector: #selector(DisplayTarget.tick(_:)))
            link.isPaused = true
            link.add(to: .main, forMode: .common)
            displayLink = link
        }
        restartCounting(at: CACurrentMediaTime())
        startToken += 1
        let token = startToken
        let running = await camera.start()
        // The tab may have been left while the camera was starting; stopping wins, unless
        // the tab has since come back and a newer start() owns the camera.
        if Task.isCancelled {
            if token == startToken { camera.stop() }
            return
        }
        cameraMessage = running ? nil : "No camera here. Use the Benchmark tab, or run on an iPhone."
    }

    func stop() {
        camera.stop()
        stopRecording()
        displayLink?.invalidate()
        displayLink = nil
        awaitingDisplay = nil
    }

    /// The camera is interrupted in the background and screen refreshes stop, so a frame
    /// in flight would be timed across the suspension. Save any recording and start the
    /// statistics afresh on return.
    func enteredBackground() {
        stopRecording()
        awaitingDisplay = nil
        displayLink?.isPaused = true
    }

    func enteredForeground() {
        restartCounting(at: CACurrentMediaTime())
    }

    func toggleRecording() {
        if isRecording {
            stopRecording()
        } else {
            recorded = []
            recordingStarted = (Date(), CACurrentMediaTime())
            isRecording = true
        }
    }

    /// Saves the recording to Documents as live-<unix time>.csv, like the benchmark CSVs.
    func stopRecording() {
        guard isRecording else { return }
        isRecording = false
        guard !recorded.isEmpty else { return }
        let csv = LiveExport.csv(recorded, started: recordingStarted)
        let name = "live-\(Int(recordingStarted.wall.timeIntervalSince1970)).csv"
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            try? csv.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        export = LiveExport(name: name, csv: csv, frames: recorded.count)
        recorded = []
    }

    private func reload() {
        requestedModel = true
        let model = model, unit = unit
        queue.async { [weak self] in
            guard let self else { return }
            self.detector = nil
            do {
                let detector = try Detector(model: model, unit: unit)
                self.detector = detector
                self.window = RollingWindow(capacity: 30)
                let ready = CACurrentMediaTime()
                self.drops = DropCount(since: ready)
                Task { @MainActor in
                    self.loadMilliseconds = detector.loadMilliseconds
                    self.modelError = nil
                    self.restartCounting(at: ready)
                }
            } catch {
                Task { @MainActor in self.modelError = error.localizedDescription }
            }
        }
    }

    /// Clears the camera-to-boxes statistics and drop counts from host time `t` on.
    private func restartCounting(at t: Double) {
        countFrom = t
        latencyWindow = LatencySamples(capacity: 120)
        awaitingDisplay = nil
        latency = CameraLatency()
        queue.async { [weak self] in
            guard let self, self.drops.since < t else { return }
            self.drops = DropCount(since: t)
        }
    }

    nonisolated private func process(_ frame: CVPixelBuffer, capturedAt captured: Double) {
        // Leave the Neural Engine, GPU and CPU to the benchmark while one is running.
        guard !BenchmarkActivity.isRunning else { return }
        guard let detector else { return }
        let started = CACurrentMediaTime()
        guard let result = try? detector.detect(frame) else { return }
        let finished = CACurrentMediaTime()
        let (boxes, timings) = result
        window.add(timings, at: Clock.now)
        let stats = window.median, fps = window.fps
        let size = CGSize(width: CVPixelBufferGetWidth(frame), height: CVPixelBufferGetHeight(frame))
        let thermal = ProcessInfo.processInfo.thermalState

        if captured >= drops.since { drops.processed += 1 }
        let sample = FrameLatency(model: detector.bundled, unit: detector.unit, captured: captured,
                                  detectStarted: started, detectFinished: finished, timings: timings,
                                  droppedBefore: drops.sinceLastFrame, thermal: thermal)
        drops.sinceLastFrame = 0
        let counts = (dropped: drops.dropped, processed: drops.processed)
        Task { @MainActor in
            self.detections = boxes
            self.frameSize = size
            self.stats = stats
            self.fps = fps
            self.thermal = thermal
            self.latency = CameraLatency(p50: self.latencyWindow.median, p95: self.latencyWindow.p95,
                                         dropped: counts.dropped, processed: counts.processed)
            self.handedToSwiftUI(sample)
        }
    }

    nonisolated private func dropped(capturedAt captured: Double) {
        guard captured >= drops.since else { return }
        drops.dropped += 1
        drops.sinceLastFrame += 1
    }

    /// The frame's boxes were just published; the next display refresh is the first that
    /// can show them.
    private func handedToSwiftUI(_ sample: FrameLatency) {
        if let earlier = awaitingDisplay {
            finish(earlier, link: nil)   // replaced before any refresh: its boxes were never shown
        }
        awaitingDisplay = sample
        displayLink?.isPaused = false
    }

    private func refreshed(_ link: CADisplayLink) {
        link.isPaused = true
        guard let sample = awaitingDisplay else { return }
        awaitingDisplay = nil
        finish(sample, link: link)
    }

    private func finish(_ sample: FrameLatency, link: CADisplayLink?) {
        // Frames from a model that has since been switched away from, or captured before
        // the statistics were restarted, are not counted.
        guard sample.captured >= countFrom, sample.model == model, sample.unit == unit else { return }
        var sample = sample
        if let link {
            sample.shown = link.targetTimestamp
            sample.refreshInterval = link.targetTimestamp - link.timestamp
            latencyWindow.add((link.targetTimestamp - sample.captured) * 1000)
        }
        if isRecording { recorded.append(sample) }
    }
}

/// Lets a CADisplayLink call back without retaining LiveDetection.
@MainActor
private final class DisplayTarget: NSObject {
    let action: @MainActor (CADisplayLink) -> Void
    init(_ action: @escaping @MainActor (CADisplayLink) -> Void) { self.action = action }
    @objc func tick(_ link: CADisplayLink) { action(link) }
}

/// One processed camera frame, timed from capture to the screen. Times are host seconds.
struct FrameLatency {
    let model: BundledModel
    let unit: ComputeUnit
    let captured: Double
    let detectStarted: Double
    let detectFinished: Double
    let timings: Timings
    /// Frames the camera dropped between the previous processed frame and this one.
    let droppedBefore: Int
    let thermal: ProcessInfo.ThermalState
    /// Target time of the refresh that first showed the boxes; nil if a newer frame's boxes
    /// replaced them before any refresh.
    var shown: Double?
    var refreshInterval: Double?
}

/// Rolling camera-to-boxes statistics for the Live tab.
struct CameraLatency {
    var p50 = 0.0, p95 = 0.0
    /// Since the current model loaded (or the tab or app came back): frames the camera
    /// dropped, and frames the detector processed.
    var dropped = 0, processed = 0
}

/// Counts on the detector queue, from host time `since`.
struct DropCount {
    var since = 0.0
    var dropped = 0, processed = 0
    var sinceLastFrame = 0
}

/// The last `capacity` camera-to-boxes latencies, in milliseconds.
struct LatencySamples {
    let capacity: Int
    private var values: [Double] = []

    init(capacity: Int) { self.capacity = capacity }

    mutating func add(_ value: Double) {
        values.append(value)
        if values.count > capacity { values.removeFirst() }
    }

    var median: Double {
        values.isEmpty ? 0 : Benchmark.median(values.sorted())
    }

    /// Same index as the benchmark's p95.
    var p95: Double {
        let sorted = values.sorted()
        return sorted.isEmpty ? 0 : sorted[Int(Double(sorted.count - 1) * 0.95)]
    }
}

/// A finished Live-tab recording: one row per processed frame.
struct LiveExport {
    let name: String
    let csv: String
    let frames: Int

    /// Each row: when the frame was captured (wall clock, and seconds since recording
    /// started), the model and compute unit, then where the time went. capture_to_detect
    /// is camera pipeline plus waiting for the detector queue; detect_to_screen is
    /// publishing the boxes plus waiting for a display refresh. Empty on-screen columns
    /// mean a newer frame's boxes replaced these before any refresh showed them.
    @MainActor
    static func csv(_ samples: [FrameLatency], started: (wall: Date, host: Double)) -> String {
        let header = ["timestamp", "device", "variant", "backend", "frame", "t_s", "capture_to_detect_ms",
                      "pre_ms", "model_ms", "post_ms", "detect_to_screen_ms", "capture_to_boxes_ms",
                      "refresh_ms", "dropped_before", "thermal"]
        let iso = ISO8601DateFormatter()
        let device = Benchmark.deviceModel
        let first = samples.first?.captured ?? started.host
        func ms(_ seconds: Double?) -> String { seconds.map { String(format: "%.2f", $0 * 1000) } ?? "" }
        let lines = samples.enumerated().map { i, s in
            [iso.string(from: started.wall.addingTimeInterval(s.captured - started.host)), device,
             s.model.variant, "coreml-swift-\(s.unit.pythonName)", "\(i)", String(format: "%.3f", s.captured - first),
             ms(s.detectStarted - s.captured), String(format: "%.2f", s.timings.preprocess),
             String(format: "%.2f", s.timings.inference), String(format: "%.2f", s.timings.postprocess),
             ms(s.shown.map { $0 - s.detectFinished }), ms(s.shown.map { $0 - s.captured }),
             ms(s.refreshInterval), "\(s.droppedBefore)", s.thermal.label]
                .map(Benchmark.csvField).joined(separator: ",")
        }
        return ([header.joined(separator: ",")] + lines).joined(separator: "\n") + "\n"
    }
}

/// Median of each stage over the last `capacity` frames, plus frames per second.
struct RollingWindow {
    let capacity: Int
    private var samples: [Timings] = []
    private var stamps: [UInt64] = []

    init(capacity: Int) { self.capacity = capacity }

    mutating func add(_ t: Timings, at time: UInt64) {
        samples.append(t)
        stamps.append(time)
        if samples.count > capacity {
            samples.removeFirst()
            stamps.removeFirst()
        }
    }

    var median: Timings {
        func med(_ f: (Timings) -> Double) -> Double {
            let v = samples.map(f).sorted()
            return v.isEmpty ? 0 : v[v.count / 2]
        }
        return Timings(preprocess: med(\.preprocess), inference: med(\.inference), postprocess: med(\.postprocess))
    }

    var fps: Double {
        guard let first = stamps.first, let last = stamps.last, stamps.count > 1 else { return 0 }
        return Double(stamps.count - 1) / (Double(last - first) / 1e9)
    }
}

extension ProcessInfo.ThermalState {
    var label: String {
        switch self {
        case .nominal: "Nominal"
        case .fair: "Fair"
        case .serious: "Serious (throttling)"
        case .critical: "Critical (throttling)"
        @unknown default: "Unknown"
        }
    }
}
