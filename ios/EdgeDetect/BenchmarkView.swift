import CoreImage
import os
import SwiftUI
import UIKit

/// Set for the duration of a benchmark run so the Live tab and the sample photo stay
/// off the CPU, GPU and Neural Engine while timings are being taken.
enum BenchmarkActivity {
    private static let flag = OSAllocatedUnfairLock(initialState: false)
    static var isRunning: Bool {
        get { flag.withLock { $0 } }
        set { flag.withLock { $0 = newValue } }
    }
}

/// One measured (precision, compute unit) pair on a fixed image.
struct BenchmarkRow: Identifiable {
    var id: String { "\(precision.rawValue)-\(unit.rawValue)" }
    let timestamp: Date
    let precision: Precision
    let unit: ComputeUnit
    /// First load in this process. Includes the on-device Neural Engine / GPU compile
    /// unless Core ML already cached it (an earlier run, or the Live tab / sample photo).
    let firstLoadMilliseconds: Double
    /// An immediate second load: the cached, steady-state cost.
    let warmLoadMilliseconds: Double
    let modelP50: Double        // MLModel.prediction only; see the note on timeInference
    let modelP95: Double
    let modelMean: Double
    let endToEndP50: Double     // letterbox + model + decode + NMS
    let detections: Int
    let thermal: ProcessInfo.ThermalState
}

/// Times every precision on every compute unit against the bundled bus.jpg.
@MainActor
final class Benchmark: ObservableObject {
    @Published private(set) var rows: [BenchmarkRow] = []
    @Published private(set) var progress: String?
    /// Why the last run was not saved, if it was not.
    @Published private(set) var status: String?
    var runs = 100

    /// Same leading columns as results/latency.csv, then the iOS-only ones;
    /// `python -m bench.merge_ios <file>` adds it to the results. The backend is
    /// "coreml-swift-*", not the Mac's "coreml-*", because the two harnesses time
    /// different amounts of work (see Detector.timeInference).
    var csv: String {
        let header = ["timestamp", "device", "variant", "backend", "p50_ms", "p95_ms", "mean_ms", "fps",
                      "size_mb", "first_load_ms", "warm_load_ms", "end_to_end_p50_ms", "detections", "thermal"]
        let iso = ISO8601DateFormatter()
        let lines = rows.map { r in
            [iso.string(from: r.timestamp), Self.deviceModel, "coreml_\(r.precision.rawValue.lowercased())",
             "coreml-swift-\(r.unit.pythonName)", fmt(r.modelP50), fmt(r.modelP95), fmt(r.modelMean),
             String(format: "%.1f", 1000 / r.modelMean), "", fmt(r.firstLoadMilliseconds),
             fmt(r.warmLoadMilliseconds), fmt(r.endToEndP50), "\(r.detections)", r.thermal.label]
                .map(Self.csvField).joined(separator: ",")
        }
        return ([header.joined(separator: ",")] + lines).joined(separator: "\n") + "\n"
    }

    func run() async {
        rows = []
        status = nil
        guard let frame = Self.sampleFrame() else {
            status = "bus.jpg is missing from the bundle."
            return
        }
        // A run takes about half a minute. If the app is backgrounded or the screen locks
        // mid-sample, that sample would include the suspended time, so such runs are discarded.
        var interrupted = false
        let observer = NotificationCenter.default.addObserver(
            forName: UIApplication.didEnterBackgroundNotification, object: nil, queue: .main
        ) { _ in interrupted = true }
        BenchmarkActivity.isRunning = true
        UIApplication.shared.isIdleTimerDisabled = true
        defer {
            NotificationCenter.default.removeObserver(observer)
            BenchmarkActivity.isRunning = false
            UIApplication.shared.isIdleTimerDisabled = false
            progress = nil
        }

        let runs = runs
        var failures: [String] = []
        for precision in Precision.allCases {
            for unit in ComputeUnit.allCases {
                progress = "\(precision.rawValue) on \(unit.rawValue)…"
                let result = await Task.detached(priority: .userInitiated) {
                    Result { try Self.measure(precision, unit, frame: frame, runs: runs) }
                }.value
                switch result {
                case .success(let row): rows.append(row)
                case .failure(let error): failures.append("\(precision.rawValue)/\(unit.rawValue): \(error.localizedDescription)")
                }
                if interrupted {
                    rows = []
                    status = "Run discarded: the app left the foreground. Keep it open and run again."
                    return
                }
            }
        }
        if !failures.isEmpty {
            status = "Not saved. Failed: " + failures.joined(separator: "; ")
            return
        }
        // Also keep a copy in the app's Documents folder, so it can be pulled to the Mac with
        // `xcrun devicectl device copy from --domain-type appDataContainer ...` (see ios/README.md).
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let name = "benchmark-\(Self.deviceModel)-\(Int(Date().timeIntervalSince1970)).csv"
            try? csv.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
    }

    nonisolated private static func measure(_ precision: Precision, _ unit: ComputeUnit,
                                            frame: CVPixelBuffer, runs: Int) throws -> BenchmarkRow {
        let first = try Detector(precision: precision, unit: unit)
        let detector = try Detector(precision: precision, unit: unit)
        let (input, _) = try detector.preprocess(frame)
        for _ in 0..<10 { _ = try detector.detect(frame) }   // warm-up: first calls pay one-time setup

        var model: [Double] = [], endToEnd: [Double] = []
        var count = 0
        for _ in 0..<runs { model.append(try detector.timeInference(on: input)) }
        for _ in 0..<runs {
            let (boxes, t) = try detector.detect(frame)
            endToEnd.append(t.total)
            count = boxes.count
        }
        let mean = model.reduce(0, +) / Double(runs)
        model.sort()
        endToEnd.sort()
        return BenchmarkRow(timestamp: Date(), precision: precision, unit: unit,
                            firstLoadMilliseconds: first.loadMilliseconds,
                            warmLoadMilliseconds: detector.loadMilliseconds,
                            modelP50: Self.median(model), modelP95: model[Int(Double(runs - 1) * 0.95)],
                            modelMean: mean, endToEndP50: Self.median(endToEnd), detections: count,
                            thermal: ProcessInfo.processInfo.thermalState)
    }

    /// Median of sorted values, averaging the middle two for an even count like
    /// Python's statistics.median, which bench/latency.py uses.
    nonisolated static func median(_ sorted: [Double]) -> Double {
        let n = sorted.count
        return n % 2 == 1 ? sorted[n / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2
    }

    /// RFC 4180 quoting: device identifiers such as "iPhone18,2" contain a comma.
    nonisolated static func csvField(_ s: String) -> String {
        s.contains(where: { $0 == "," || $0 == "\"" || $0 == "\n" })
            ? "\"" + s.replacingOccurrences(of: "\"", with: "\"\"") + "\"" : s
    }

    nonisolated static func sampleFrame() -> CVPixelBuffer? {
        guard let url = Bundle.main.url(forResource: "bus", withExtension: "jpg"),
              let image = CIImage(contentsOf: url) else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, Int(image.extent.width), Int(image.extent.height), kCVPixelFormatType_32BGRA,
                            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &buffer)
        guard let buffer else { return nil }
        CIContext().render(image, to: buffer)
        return buffer
    }

    /// Hardware identifier such as "iPhone18,2" (the marketing name is not available to apps).
    /// The Simulator reports the Mac's "arm64", so name it as a simulator instead.
    static var deviceModel: String {
        if let simulated = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] {
            return "\(simulated) Simulator"
        }
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }

    private func fmt(_ v: Double) -> String { String(format: "%.2f", v) }
}

struct BenchmarkView: View {
    @StateObject private var bench = Benchmark()

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button(bench.progress ?? "Run benchmark (\(bench.runs) runs each)") {
                        Task { await bench.run() }
                    }
                    .disabled(bench.progress != nil)
                    if let status = bench.status {
                        Text(status).foregroundStyle(.red)
                    }
                } footer: {
                    Text("Every precision on every compute unit, on a fixed photo. \"Model\" times the "
                         + "Core ML call only; \"End to end\" adds letterboxing, decoding and NMS. "
                         + "Keep the phone cool and plugged in; watch the thermal column.")
                }
                Section("Sample photo") { SamplePhotoView() }
                ForEach(Precision.allCases) { precision in
                    let rows = bench.rows.filter { $0.precision == precision }
                    if !rows.isEmpty {
                        Section(precision.rawValue) {
                            ForEach(rows) { row in BenchmarkRowView(row: row) }
                        }
                    }
                }
            }
            .navigationTitle(Benchmark.deviceModel)
            .toolbar {
                if !bench.rows.isEmpty && bench.progress == nil {
                    ShareLink(item: bench.csv, preview: SharePreview("edge-ml-bench results.csv"))
                }
            }
        }
    }
}

struct BenchmarkRowView: View {
    let row: BenchmarkRow

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(row.unit.rawValue).font(.headline)
                Spacer()
                Text("\(row.modelP50, specifier: "%.2f") ms model")
                    .font(.headline.monospacedDigit())
            }
            HStack {
                Text("End to end \(row.endToEndP50, specifier: "%.2f") ms")
                Text("· p95 \(row.modelP95, specifier: "%.2f")")
                Text("· load \(row.firstLoadMilliseconds, specifier: "%.0f")/\(row.warmLoadMilliseconds, specifier: "%.0f") ms")
                Spacer()
                Text("\(row.detections) boxes")
            }
            .font(.caption.monospacedDigit())
            .foregroundStyle(.secondary)
            if row.thermal.rawValue >= 2 {
                Text("Thermal: \(row.thermal.label)").font(.caption).foregroundStyle(.red)
            }
        }
    }
}
