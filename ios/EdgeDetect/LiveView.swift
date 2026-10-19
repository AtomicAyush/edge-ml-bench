import AVFoundation
import SwiftUI
import UIKit

struct LiveView: View {
    @StateObject private var live = LiveDetection()

    var body: some View {
        ZStack(alignment: .bottom) {
            GeometryReader { geo in
                ZStack {
                    CameraPreview(session: live.camera.session)
                    BoxOverlay(detections: live.detections, frameSize: live.frameSize, viewSize: geo.size)
                }
            }
            .ignoresSafeArea()

            VStack(spacing: 10) {
                if let message = live.message {
                    Text(message).font(.callout).multilineTextAlignment(.center)
                }
                ModelPicker(selection: $live.model)
                Picker("Compute unit", selection: $live.unit) {
                    ForEach(ComputeUnit.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                StatsRow(stats: live.stats, fps: live.fps)
                CameraLatencyRow(live: live)
                HStack {
                    Text("Model load \(live.loadMilliseconds, specifier: "%.0f") ms")
                    Spacer()
                    Text("Thermal: \(live.thermal.label)")
                        .foregroundStyle(live.thermal.rawValue >= 2 ? .red : .secondary)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .padding()
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
            .padding()
        }
        .task { await live.start() }
        .onDisappear { live.stop() }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.didEnterBackgroundNotification)) { _ in
            live.enteredBackground()
        }
        .onReceive(NotificationCenter.default.publisher(for: UIApplication.willEnterForegroundNotification)) { _ in
            live.enteredForeground()
        }
    }
}

/// Every model bundled in the app: a segmented control for a few, a menu for more.
struct ModelPicker: View {
    @Binding var selection: BundledModel

    var body: some View {
        let picker = Picker("Model", selection: $selection) {
            ForEach(BundledModel.all) { Text($0.name).tag($0) }
        }
        if BundledModel.all.count <= 4 {
            picker.pickerStyle(.segmented)
        } else {
            HStack {
                Text("Model").font(.callout)
                Spacer()
                picker.pickerStyle(.menu)
            }
        }
    }
}

/// Capture to boxes on screen over the last 120 timed frames, frames the camera dropped,
/// and the record / share controls for live-<time>.csv.
struct CameraLatencyRow: View {
    @ObservedObject var live: LiveDetection

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Camera to boxes \(live.latency.p50, specifier: "%.1f") ms · p95 \(live.latency.p95, specifier: "%.1f")")
                    .font(.subheadline.monospacedDigit())
                Text("Dropped \(live.latency.dropped) of \(live.latency.dropped + live.latency.processed) frames")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if let export = live.export, !live.isRecording {
                ShareLink(item: export.csv, preview: SharePreview(export.name)) {
                    Image(systemName: "square.and.arrow.up")
                }
                .accessibilityLabel("Share \(export.name)")
            }
            Button {
                live.toggleRecording()
            } label: {
                Image(systemName: live.isRecording ? "stop.circle.fill" : "record.circle")
                    .font(.title2)
                    .foregroundStyle(.red)
            }
            .accessibilityLabel(live.isRecording ? "Stop recording and save CSV" : "Record per-frame timings")
        }
    }
}

/// Per-stage medians over the last 30 frames.
struct StatsRow: View {
    let stats: Timings
    let fps: Double

    var body: some View {
        HStack {
            stat("FPS", fps, "%.0f")
            stat("Pre", stats.preprocess)
            stat("Model", stats.inference)
            stat("Post", stats.postprocess)
            stat("Total", stats.total)
        }
    }

    private func stat(_ title: String, _ value: Double, _ format: String = "%.1f") -> some View {
        VStack(spacing: 2) {
            Text(String(format: format, value)).font(.headline.monospacedDigit())
            Text(title == "FPS" ? title : "\(title) ms").font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

/// Draws boxes given in frame pixels over an image shown at `viewSize`, either filling
/// the view (the camera preview) or fitting inside it (a still photo).
struct BoxOverlay: View {
    let detections: [Detection]
    let frameSize: CGSize
    let viewSize: CGSize
    var fill = true

    var body: some View {
        let sx = viewSize.width / frameSize.width, sy = viewSize.height / frameSize.height
        let scale = fill ? max(sx, sy) : min(sx, sy)
        let dx = (viewSize.width - frameSize.width * scale) / 2
        let dy = (viewSize.height - frameSize.height * scale) / 2
        ZStack(alignment: .topLeading) {
            ForEach(detections) { d in
                let rect = CGRect(x: d.rect.minX * scale + dx, y: d.rect.minY * scale + dy,
                                  width: d.rect.width * scale, height: d.rect.height * scale)
                Rectangle()
                    .stroke(color(d.label), lineWidth: 2)
                    .frame(width: rect.width, height: rect.height)
                    .overlay(alignment: .topLeading) {
                        Text("\(d.name) \(Int(d.score * 100))%")
                            .font(.caption2.bold())
                            .lineLimit(1)
                            .fixedSize()
                            .padding(.horizontal, 3)
                            .background(color(d.label))
                            .foregroundStyle(.white)
                            .offset(y: -16)
                    }
                    .offset(x: rect.minX, y: rect.minY)
            }
        }
        .frame(width: viewSize.width, height: viewSize.height, alignment: .topLeading)
        .allowsHitTesting(false)
    }

    private func color(_ label: Int) -> Color {
        Color(hue: Double(label * 37 % 80) / 80, saturation: 0.8, brightness: 0.9)
    }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.backgroundColor = .black
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {}
}
