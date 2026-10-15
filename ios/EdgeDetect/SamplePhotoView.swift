import SwiftUI
import UIKit

/// Runs the detector once on the bundled bus.jpg and draws the result, with each box's
/// pixel coordinates listed so they can be checked against the Python pipeline
/// (`YOLO(...).predict("bus.jpg")` finds a bus and four people).
struct SamplePhotoView: View {
    @State private var detections: [Detection] = []
    @State private var timings = Timings()
    @State private var error: String?
    private let image = Bundle.main.url(forResource: "bus", withExtension: "jpg")
        .flatMap { UIImage(contentsOfFile: $0.path) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .overlay {
                        GeometryReader { geo in
                            BoxOverlay(detections: detections, frameSize: image.size, viewSize: geo.size, fill: false)
                        }
                    }
                    .frame(maxHeight: 360)
            }
            if let error { Text(error).foregroundStyle(.red) }
            Text("FP16 · All compute units · end to end \(timings.total, specifier: "%.1f") ms")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(detections) { d in
                Text(String(format: "%@ %.2f  [%d, %d, %d, %d]", d.name, d.score,
                            Int(d.rect.minX), Int(d.rect.minY), Int(d.rect.maxX), Int(d.rect.maxY)))
                    .font(.caption.monospaced())
            }
        }
        .task {
            // Computed once, and never while a benchmark run is taking timings.
            guard detections.isEmpty, !BenchmarkActivity.isRunning else { return }
            do {
                guard let frame = Benchmark.sampleFrame() else { return }
                let detector = try Detector(precision: .fp16, unit: .all)
                (detections, timings) = try detector.detect(frame)
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}
