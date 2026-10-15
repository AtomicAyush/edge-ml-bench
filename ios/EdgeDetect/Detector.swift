import CoreImage
import CoreML
import CoreVideo
import Foundation

/// The three Core ML exports from `bench/export.py`, bundled as YOLOv8n_<rawValue>.mlpackage.
enum Precision: String, CaseIterable, Identifiable {
    case fp32 = "FP32", fp16 = "FP16", int8 = "INT8"
    var id: String { rawValue }
    var resource: String { "YOLOv8n_\(rawValue)" }
}

/// Where Core ML is allowed to run the model. `.all` lets Core ML choose per layer.
enum ComputeUnit: String, CaseIterable, Identifiable {
    case cpu = "CPU", gpu = "GPU", neuralEngine = "ANE", all = "All"
    var id: String { rawValue }
    /// The name bench/latency.py uses for the same setting (coremltools ComputeUnit).
    var pythonName: String {
        switch self {
        case .cpu: "cpu_only"
        case .gpu: "cpu_and_gpu"
        case .neuralEngine: "cpu_and_ne"
        case .all: "all"
        }
    }

    var mlComputeUnits: MLComputeUnits {
        switch self {
        case .cpu: .cpuOnly
        case .gpu: .cpuAndGPU
        case .neuralEngine: .cpuAndNeuralEngine
        case .all: .all
        }
    }
}

struct Detection: Identifiable {
    let id = UUID()
    /// In the input frame's pixel coordinates, origin top-left.
    let rect: CGRect
    let label: Int
    let score: Float
    var name: String { cocoNames[label] }
}

/// Milliseconds spent in each stage of one frame.
struct Timings {
    var preprocess = 0.0, inference = 0.0, postprocess = 0.0
    var total: Double { preprocess + inference + postprocess }
}

enum DetectorError: LocalizedError {
    case missingModel(String), badOutput
    var errorDescription: String? {
        switch self {
        case .missingModel(let name):
            "\(name).mlmodelc is not in the app bundle. Run `make ios-models` and rebuild."
        case .badOutput: "The model returned an unexpected output."
        }
    }
}

/// YOLOv8n end to end: letterbox a camera frame to 640×640, run Core ML, decode the
/// raw [1, 84, 8400] output and apply non-maximum suppression. The Python benchmark
/// times only the middle step; this times all three, because that is what an app pays.
final class Detector {
    static let side = 640
    let precision: Precision
    let unit: ComputeUnit
    /// Milliseconds to load the compiled model for this compute unit. The first load on
    /// the Neural Engine includes an on-device compile, so it is much slower than later ones.
    let loadMilliseconds: Double

    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let pool: CVPixelBufferPool
    private let background: CIImage

    init(precision: Precision, unit: ComputeUnit) throws {
        guard let url = Bundle.main.url(forResource: precision.resource, withExtension: "mlmodelc") else {
            throw DetectorError.missingModel(precision.resource)
        }
        let config = MLModelConfiguration()
        config.computeUnits = unit.mlComputeUnits
        let start = Clock.now
        model = try MLModel(contentsOf: url, configuration: config)
        loadMilliseconds = Clock.milliseconds(since: start)
        self.precision = precision
        self.unit = unit
        inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? ""

        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: Self.side,
            kCVPixelBufferHeightKey as String: Self.side,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
        self.pool = pool!
        // Ultralytics pads letterboxed images with gray 114.
        background = CIImage(color: CIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255))
            .cropped(to: CGRect(x: 0, y: 0, width: Self.side, height: Self.side))
    }

    func detect(_ frame: CVPixelBuffer, confidence: Float = 0.25, iou: Float = 0.45) throws -> ([Detection], Timings) {
        var timings = Timings()

        var start = Clock.now
        let (input, letterbox) = try preprocess(frame)
        timings.preprocess = Clock.milliseconds(since: start)

        start = Clock.now
        let features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: input)])
        guard let output = try model.prediction(from: features).featureValue(for: outputName)?.multiArrayValue else {
            throw DetectorError.badOutput
        }
        timings.inference = Clock.milliseconds(since: start)

        start = Clock.now
        let boxes = nonMaxSuppression(decode(output, confidence: confidence), iou: iou)
            .map { $0.mapped(by: letterbox) }
        timings.postprocess = Clock.milliseconds(since: start)
        return (boxes, timings)
    }

    /// Model-only timing on an already-letterboxed, pooled IOSurface buffer: the
    /// prediction call and nothing else. bench/latency.py's Core ML rows also pay for
    /// Python converting a PIL image into a pixel buffer and the output into NumPy on
    /// every call, so on the same hardware this number is somewhat lower. Compare
    /// phones with phones, and the Mac's Python rows with each other.
    func timeInference(on input: CVPixelBuffer) throws -> Double {
        let features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: input)])
        let start = Clock.now
        _ = try model.prediction(from: features)
        return Clock.milliseconds(since: start)
    }

    // MARK: - Pre-processing

    /// Scale and offset that map the frame into the 640×640 model input.
    struct Letterbox { let scale: CGFloat, dx: CGFloat, dy: CGFloat }

    func preprocess(_ frame: CVPixelBuffer) throws -> (CVPixelBuffer, Letterbox) {
        let width = CGFloat(CVPixelBufferGetWidth(frame)), height = CGFloat(CVPixelBufferGetHeight(frame))
        let side = CGFloat(Self.side)
        let scale = side / max(width, height)
        let box = Letterbox(scale: scale, dx: (side - width * scale) / 2, dy: (side - height * scale) / 2)
        // Padding is symmetric, so Core Image's bottom-left origin needs no extra flip.
        let image = CIImage(cvPixelBuffer: frame)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale).translatedBy(x: box.dx / scale, y: box.dy / scale))
            .composited(over: background)
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        guard let buffer else { throw DetectorError.badOutput }
        context.render(image, to: buffer)
        return (buffer, box)
    }

    // MARK: - Post-processing

    /// The output holds, for each of 8,400 anchors, a box (cx, cy, w, h in input pixels)
    /// followed by 80 class probabilities. Keep each anchor's best class if it clears
    /// `confidence`. Loops run class-major so memory is read in order.
    private func decode(_ output: MLMultiArray, confidence: Float) -> [Detection] {
        let channels = output.shape[1].intValue, anchors = output.shape[2].intValue
        let rowStride = output.strides[1].intValue, colStride = output.strides[2].intValue
        var best = [Float](repeating: 0, count: anchors)
        var bestClass = [Int](repeating: 0, count: anchors)
        var detections: [Detection] = []

        output.withUnsafeBufferPointer(ofType: Float.self) { p in
            for c in 4..<channels {
                let row = c * rowStride
                for j in 0..<anchors where p[row + j * colStride] > best[j] {
                    best[j] = p[row + j * colStride]
                    bestClass[j] = c - 4
                }
            }
            for j in 0..<anchors where best[j] >= confidence {
                let cx = p[j * colStride], cy = p[rowStride + j * colStride]
                let w = p[2 * rowStride + j * colStride], h = p[3 * rowStride + j * colStride]
                detections.append(Detection(
                    rect: CGRect(x: CGFloat(cx - w / 2), y: CGFloat(cy - h / 2), width: CGFloat(w), height: CGFloat(h)),
                    label: bestClass[j], score: best[j]))
            }
        }
        return detections
    }

    /// Greedy per-class NMS: keep the highest-scoring box, drop same-class boxes that
    /// overlap it by more than `iou`, repeat.
    private func nonMaxSuppression(_ detections: [Detection], iou: Float, limit: Int = 100) -> [Detection] {
        var kept: [Detection] = []
        for d in detections.sorted(by: { $0.score > $1.score }) {
            if kept.count == limit { break }
            if !kept.contains(where: { $0.label == d.label && $0.rect.iou(d.rect) > CGFloat(iou) }) {
                kept.append(d)
            }
        }
        return kept
    }
}

private extension Detection {
    func mapped(by box: Detector.Letterbox) -> Detection {
        Detection(rect: CGRect(x: (rect.minX - box.dx) / box.scale, y: (rect.minY - box.dy) / box.scale,
                               width: rect.width / box.scale, height: rect.height / box.scale),
                  label: label, score: score)
    }
}

private extension CGRect {
    func iou(_ other: CGRect) -> CGFloat {
        let overlap = intersection(other)
        guard !overlap.isNull else { return 0 }
        let inter = overlap.width * overlap.height
        return inter / (width * height + other.width * other.height - inter)
    }
}

enum Clock {
    static var now: UInt64 { DispatchTime.now().uptimeNanoseconds }
    static func milliseconds(since start: UInt64) -> Double { Double(now - start) / 1_000_000 }
}
