import AVFoundation
import SwiftUI

/// Runs the detector on every camera frame it can keep up with and publishes the
/// boxes plus rolling per-stage timings. All detector work happens on `queue`.
@MainActor
final class LiveDetection: ObservableObject {
    @Published var precision: Precision = .fp16 { didSet { reload() } }
    @Published var unit: ComputeUnit = .neuralEngine { didSet { reload() } }
    @Published private(set) var detections: [Detection] = []
    @Published private(set) var frameSize = CGSize(width: 720, height: 1280)
    @Published private(set) var stats = Timings()
    @Published private(set) var fps = 0.0
    @Published private(set) var loadMilliseconds = 0.0
    @Published private(set) var thermal = ProcessInfo.processInfo.thermalState
    @Published private(set) var cameraMessage: String?
    @Published private(set) var modelError: String?
    var message: String? { modelError ?? cameraMessage }

    let queue = DispatchQueue(label: "detector", qos: .userInitiated)
    private(set) lazy var camera = Camera(queue: queue) { [weak self] frame in self?.process(frame) }

    private var requestedModel = false
    private var startToken = 0

    // Touched only on `queue`.
    nonisolated(unsafe) private var detector: Detector?
    nonisolated(unsafe) private var window = RollingWindow(capacity: 30)

    func start() async {
        if !requestedModel { reload() }
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

    func stop() { camera.stop() }

    private func reload() {
        requestedModel = true
        let precision = precision, unit = unit
        queue.async { [weak self] in
            guard let self else { return }
            self.detector = nil
            do {
                let detector = try Detector(precision: precision, unit: unit)
                self.detector = detector
                self.window = RollingWindow(capacity: 30)
                Task { @MainActor in
                    self.loadMilliseconds = detector.loadMilliseconds
                    self.modelError = nil
                }
            } catch {
                Task { @MainActor in self.modelError = error.localizedDescription }
            }
        }
    }

    nonisolated private func process(_ frame: CVPixelBuffer) {
        // Leave the Neural Engine, GPU and CPU to the benchmark while one is running.
        guard !BenchmarkActivity.isRunning else { return }
        guard let detector, let result = try? detector.detect(frame) else { return }
        let (boxes, timings) = result
        window.add(timings, at: Clock.now)
        let stats = window.median, fps = window.fps
        let size = CGSize(width: CVPixelBufferGetWidth(frame), height: CVPixelBufferGetHeight(frame))
        let thermal = ProcessInfo.processInfo.thermalState
        Task { @MainActor in
            self.detections = boxes
            self.frameSize = size
            self.stats = stats
            self.fps = fps
            self.thermal = thermal
        }
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
