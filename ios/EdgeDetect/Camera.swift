import AVFoundation

/// Back camera at 1280×720, delivered upright (portrait) as BGRA frames on `queue`.
/// Late frames are dropped rather than queued, so measured FPS is what the
/// detector can actually sustain. The session is configured, started and stopped
/// only on `queue`, which is what makes the unchecked Sendable conformance safe.
///
/// Each frame comes with its capture time in host-clock seconds: the sample buffer's
/// presentation timestamp, converted from the session's synchronization clock to
/// CMClockGetHostTimeClock(). The host clock is mach_absolute_time, the same timebase as
/// CACurrentMediaTime() and CADisplayLink, so capture-to-screen latency can be taken
/// by subtraction. Dropped frames are reported with their capture times too.
final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue: DispatchQueue
    private let onFrame: (CVPixelBuffer, Double) -> Void
    private let onDrop: (Double) -> Void
    private var configured = false   // touched only on `queue`

    init(queue: DispatchQueue, onFrame: @escaping (CVPixelBuffer, Double) -> Void,
         onDrop: @escaping (Double) -> Void) {
        self.queue = queue
        self.onFrame = onFrame
        self.onDrop = onDrop
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

    /// The frame's capture time in host-clock seconds (see the type's note).
    private func captureTime(_ sampleBuffer: CMSampleBuffer) -> Double {
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        // For camera capture the session clock is normally the host clock already, in which
        // case this conversion is the identity; converting makes that an outcome, not an
        // assumption. The clock exists once the session is running, which it is here.
        guard let clock = session.synchronizationClock else { return pts.seconds }
        return CMSyncConvertTime(pts, from: clock, to: CMClockGetHostTimeClock()).seconds
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        if let frame = CMSampleBufferGetImageBuffer(sampleBuffer) {
            onFrame(frame, captureTime(sampleBuffer))
        }
    }

    /// Called for each frame discarded because the delegate queue was still busy with an
    /// earlier one (alwaysDiscardsLateVideoFrames), or for lack of buffers. The sample
    /// buffer carries timing but no image.
    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        onDrop(captureTime(sampleBuffer))
    }
}
