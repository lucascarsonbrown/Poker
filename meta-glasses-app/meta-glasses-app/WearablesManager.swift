import Foundation
import SwiftUI
import Combine
import MWDATCore
import MWDATCamera

// connection state for the glasses to see if its working
enum GlassesConnectionState: Equatable {
    case notConfigured
    case disconnected
    case registering
    case waitingForDevice
    case connecting
    case connected
    case streaming
    case error(String)

    var displayText: String {
        switch self {
        case .notConfigured: return "SDK Not Configured"
        case .disconnected: return "Disconnected"
        case .registering: return "Registering..."
        case .waitingForDevice: return "Waiting for Glasses..."
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .streaming: return "Streaming"
        case .error(let msg): return "Error: \(msg)"
        }
    }

    var isConnected: Bool {
        switch self {
        case .connected, .streaming: return true
        default: return false
        }
    }
}

// Manages Meta Wearables SDK lifecycle and video streaming
@MainActor
class WearablesManager: ObservableObject {
    static let shared = WearablesManager()

    // MARK: - Published State
    @Published var connectionState: GlassesConnectionState = .notConfigured
    @Published var currentFrame: UIImage?
    @Published var availableDevices: [String] = []
    @Published var cameraPermissionGranted: Bool = false
    @Published var frameRate: Int = 0
    @Published var resolution: String = "Unknown"

    // Frame callback for CV processing
    var onFrameReceived: ((UIImage) -> Void)?

    // MARK: - Private Properties
    private var streamSession: StreamSession?
    private var stateToken: (any AnyListenerToken)?
    private var frameToken: (any AnyListenerToken)?
    private var photoToken: (any AnyListenerToken)?

    // MARK: - Initialization
    private init() {}

    // MARK: - SDK Configuration
    func configure() {
        do {
            try Wearables.configure()
            connectionState = .disconnected
            startObservingState()
            print("[WearablesManager] SDK configured successfully")
        } catch {
            connectionState = .error("Configuration failed: \(error)")
            print("[WearablesManager] Configuration error: \(error)")
        }
    }

    // MARK: - Registration
    func startRegistration() {
        do {
            connectionState = .registering
            try Wearables.shared.startRegistration()
            print("[WearablesManager] Registration started")
        } catch {
            connectionState = .error("Registration failed: \(error)")
            print("[WearablesManager] Registration error: \(error)")
        }
    }

    func startUnregistration() {
        do {
            try Wearables.shared.startUnregistration()
            stopStreaming()
            connectionState = .disconnected
            availableDevices = []
            print("[WearablesManager] Unregistration started")
        } catch {
            print("[WearablesManager] Unregistration error: \(error)")
        }
    }

    // MARK: - URL Handling (for Meta AI callbacks)
    func handleURL(_ url: URL) async {
        do {
            _ = try await Wearables.shared.handleUrl(url)
            print("[WearablesManager] URL handled: \(url)")
        } catch {
            print("[WearablesManager] URL handling error: \(error)")
        }
    }

    // MARK: - Permissions
    func checkCameraPermission() async {
        do {
            let status = try await Wearables.shared.checkPermissionStatus(.camera)
            cameraPermissionGranted = (status == .granted)
        } catch {
            print("[WearablesManager] Permission check error: \(error)")
            cameraPermissionGranted = false
        }
    }

    func requestCameraPermission() async {
        do {
            let status = try await Wearables.shared.requestPermission(.camera)
            cameraPermissionGranted = (status == .granted)
            print("[WearablesManager] Camera permission: \(status)")
        } catch {
            print("[WearablesManager] Permission request error: \(error)")
            cameraPermissionGranted = false
        }
    }

    // MARK: - State Observation
    private func startObservingState() {
        let wearables = Wearables.shared

        // Observe registration state
        Task {
            for await state in wearables.registrationStateStream() {
                await MainActor.run {
                    self.updateConnectionState(from: state)
                }
            }
        }

        // Observe available devices
        Task {
            for await deviceIds in wearables.devicesStream() {
                await MainActor.run {
                    self.availableDevices = deviceIds.compactMap { id in
                        wearables.deviceForIdentifier(id)?.name ?? id
                    }
                    print("[WearablesManager] Devices updated: \(self.availableDevices)")
                }
            }
        }
    }

    private func updateConnectionState(from registrationState: RegistrationState) {
        switch registrationState {
        case .unavailable:
            connectionState = .notConfigured
        case .available:
            connectionState = .disconnected
        case .registering:
            connectionState = .registering
        case .registered:
            connectionState = .connected
            // Check camera permission when registered
            Task {
                await checkCameraPermission()
            }
        @unknown default:
            connectionState = .disconnected
        }
        print("[WearablesManager] Registration state: \(registrationState)")
    }

    // MARK: - Video Streaming
    func startStreaming(resolution: StreamResolution = .medium, fps: Int = 24) {
        guard connectionState.isConnected else {
            print("[WearablesManager] Cannot stream - not connected")
            return
        }

        stopStreaming() // Clean up any existing session

        let wearables = Wearables.shared
        let deviceSelector = AutoDeviceSelector(wearables: wearables)

        let streamResolution: StreamingResolution
        switch resolution {
        case .low: streamResolution = .low
        case .medium: streamResolution = .medium
        case .high: streamResolution = .high
        }

        let config = StreamSessionConfig(
            videoCodec: .raw,
            resolution: streamResolution,
            frameRate: UInt(fps)
        )

        streamSession = StreamSession(
            streamSessionConfig: config,
            deviceSelector: deviceSelector
        )

        setupStreamObservers()

        Task {
            await streamSession?.start()
        }

        self.resolution = resolution.displayName
        self.frameRate = fps
        print("[WearablesManager] Starting stream: \(resolution.displayName) @ \(fps)fps")
    }

    func stopStreaming() {
        Task {
            await stateToken?.cancel()
            await frameToken?.cancel()
            await photoToken?.cancel()
        }
        stateToken = nil
        frameToken = nil
        photoToken = nil

        Task {
            await streamSession?.stop()
            streamSession = nil
        }

        if connectionState == .streaming {
            connectionState = .connected
        }
        currentFrame = nil
        print("[WearablesManager] Streaming stopped")
    }

    private func setupStreamObservers() {
        guard let session = streamSession else { return }

        // Observe stream state
        stateToken = session.statePublisher.listen { [weak self] state in
            Task { @MainActor in
                self?.handleStreamState(state)
            }
        }

        // Observe video frames
        frameToken = session.videoFramePublisher.listen { [weak self] frame in
            guard let image = frame.makeUIImage() else { return }
            Task { @MainActor in
                self?.currentFrame = image
                self?.onFrameReceived?(image)
            }
        }

        // Observe errors
        _ = session.errorPublisher.listen { error in
            print("[WearablesManager] Stream error: \(error)")
        }
    }

    private func handleStreamState(_ state: StreamSessionState) {
        switch state {
        case .stopped:
            if connectionState == .streaming {
                connectionState = .connected
            }
        case .waitingForDevice:
            connectionState = .waitingForDevice
        case .starting:
            connectionState = .connecting
        case .streaming:
            connectionState = .streaming
        case .paused:
            connectionState = .connected
        case .stopping:
            break
        }
        print("[WearablesManager] Stream state: \(state)")
    }

    // MARK: - Photo Capture
    func capturePhoto() {
        guard let session = streamSession else {
            print("[WearablesManager] No active session for photo capture")
            return
        }

        photoToken = session.photoDataPublisher.listen { [weak self] photoData in
            if let image = UIImage(data: photoData.data) {
                Task { @MainActor in
                    self?.currentFrame = image
                    self?.onFrameReceived?(image)
                }
            }
        }

        session.capturePhoto(format: .jpeg)
        print("[WearablesManager] Photo capture requested")
    }
}

// MARK: - Supporting Types
enum StreamResolution {
    case low, medium, high

    var displayName: String {
        switch self {
        case .low: return "360p"
        case .medium: return "504p"
        case .high: return "720p"
        }
    }
}
