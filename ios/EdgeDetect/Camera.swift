import AVFoundation

/// Back camera at 1280×720, delivered upright (portrait) as BGRA frames on `queue`.
/// Late frames are dropped rather than queued, so measured FPS is what the
/// detector can actually sustain. The session is configured, started and stopped
/// only on `queue`, which is what makes the unchecked Sendable conformance safe.
final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue: DispatchQueue
    private let onFrame: (CVPixelBuffer) -> Void
    private var configured = false   // touched only on `queue`

    init(queue: DispatchQueue, onFrame: @escaping (CVPixelBuffer) -> Void) {
        self.queue = queue
        self.onFrame = onFrame
    }

    /// Safe to call again after `stop()` (e.g. on returning to the Live tab): the session
    /// is configured once and only restarted afterwards. Returns false when there is no
    /// camera (the Simulator) or access was denied.
    func start() async -> Bool {
        guard await AVCaptureDevice.requestAccess(for: .video) else { return false }
        return await withCheckedContinuation { continuation in
            queue.async {
                if !self.configured {
                    self.configured = self.configure()
                }
                if self.configured && !self.session.isRunning {
                    self.session.startRunning()
                }
                continuation.resume(returning: self.configured)
            }
        }
    }

    func stop() {
        queue.async { self.session.stopRunning() }
    }

    private func configure() -> Bool {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
              let input = try? AVCaptureDeviceInput(device: device)
        else { return false }

        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1280x720
        guard session.canAddInput(input), session.canAddOutput(output) else { return false }
        session.addInput(input)
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        session.addOutput(output)
        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(90) {
            connection.videoRotationAngle = 90
        }
        return true
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if let frame = CMSampleBufferGetImageBuffer(sampleBuffer) {
            onFrame(frame)
        }
    }
}
