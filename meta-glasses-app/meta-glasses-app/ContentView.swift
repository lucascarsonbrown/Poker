import SwiftUI

struct ContentView: View {
    @StateObject private var wearables = WearablesManager.shared
    @StateObject private var engine = PokerEngineClient.shared
    @State private var selectedResolution: StreamResolution = .medium
    @State private var selectedFPS: Int = 24
    @State private var serverURL: String = "ws://localhost:8000"

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Video preview area
                VideoPreviewView(image: wearables.currentFrame)
                    .frame(maxWidth: .infinity)
                    .frame(height: UIScreen.main.bounds.height * 0.4)

                // Controls area
                ScrollView {
                    VStack(spacing: 16) {
                        // Glasses connection status
                        ConnectionStatusCard(
                            state: wearables.connectionState,
                            devices: wearables.availableDevices
                        )

                        // Stream info (when streaming)
                        if wearables.connectionState == .streaming {
                            StreamInfoCard(
                                resolution: wearables.resolution,
                                frameRate: wearables.frameRate
                            )
                        }

                        // Glasses controls
                        ControlsCard(
                            connectionState: wearables.connectionState,
                            selectedResolution: $selectedResolution,
                            selectedFPS: $selectedFPS,
                            onConnect: { wearables.startRegistration() },
                            onDisconnect: { wearables.startUnregistration() },
                            onStartStream: {
                                wearables.startStreaming(
                                    resolution: selectedResolution,
                                    fps: selectedFPS
                                )
                            },
                            onStopStream: { wearables.stopStreaming() },
                            onCapturePhoto: { wearables.capturePhoto() }
                        )

                        Divider().padding(.vertical, 8)

                        // Server connection card
                        ServerConnectionCard(
                            isConnected: engine.isConnected,
                            serverURL: $serverURL,
                            gameState: engine.gameState,
                            analysis: engine.latestAnalysis,
                            error: engine.connectionError,
                            onConnect: {
                                engine.serverURL = serverURL
                                engine.connect()
                            },
                            onDisconnect: { engine.disconnect() },
                            onRequestAnalysis: { engine.requestAnalysis() },
                            onTestHand: { sendTestHand() }
                        )
                    }
                    .padding()
                }
            }
            .navigationTitle("Poker Glasses")
            .navigationBarTitleDisplayMode(.inline)
        }
        .onAppear {
            wearables.configure()
        }
        .onOpenURL { url in
            Task {
                await wearables.handleURL(url)
            }
        }
    }

    // Test function to simulate a hand
    private func sendTestHand() {
        engine.startHand(heroStack: 1000, villainStack: 1000, smallBlind: 5, bigBlind: 10, heroIsButton: true)

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
            engine.setHoleCards(["Ah", "Kd"])
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.6) {
            engine.requestAnalysis()
        }
    }
}

// MARK: - Video Preview
struct VideoPreviewView: View {
    let image: UIImage?

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Background
                Color.black

                if let image = image {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } else {
                    // Placeholder
                    VStack(spacing: 16) {
                        Image(systemName: "video.slash")
                            .font(.system(size: 48))
                            .foregroundColor(.gray)
                        Text("No Video Feed")
                            .font(.headline)
                            .foregroundColor(.gray)
                        Text("Connect glasses and start streaming")
                            .font(.caption)
                            .foregroundColor(.gray.opacity(0.7))
                    }
                }
            }
        }
    }
}

// MARK: - Connection Status Card
struct ConnectionStatusCard: View {
    let state: GlassesConnectionState
    let devices: [String]

    var statusColor: Color {
        switch state {
        case .connected, .streaming: return .green
        case .error: return .red
        case .registering, .connecting, .waitingForDevice: return .orange
        default: return .gray
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 12, height: 12)
                Text(state.displayText)
                    .font(.headline)
                Spacer()
            }

            if !devices.isEmpty {
                Divider()
                VStack(alignment: .leading, spacing: 4) {
                    Text("Available Devices")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    ForEach(devices, id: \.self) { device in
                        HStack {
                            Image(systemName: "eyeglasses")
                                .foregroundColor(.blue)
                            Text(device)
                                .font(.subheadline)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Stream Info Card
struct StreamInfoCard: View {
    let resolution: String
    let frameRate: Int

    var body: some View {
        HStack {
            Label(resolution, systemImage: "rectangle.on.rectangle")
            Spacer()
            Label("\(frameRate) FPS", systemImage: "speedometer")
        }
        .font(.subheadline)
        .foregroundColor(.secondary)
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Controls Card
struct ControlsCard: View {
    let connectionState: GlassesConnectionState
    @Binding var selectedResolution: StreamResolution
    @Binding var selectedFPS: Int

    let onConnect: () -> Void
    let onDisconnect: () -> Void
    let onStartStream: () -> Void
    let onStopStream: () -> Void
    let onCapturePhoto: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            // Connection controls
            if !connectionState.isConnected {
                Button(action: onConnect) {
                    Label("Connect to Glasses", systemImage: "link")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .disabled(connectionState == .registering)
            } else {
                // Stream settings
                VStack(alignment: .leading, spacing: 8) {
                    Text("Stream Settings")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack {
                        Text("Resolution")
                            .font(.subheadline)
                        Spacer()
                        Picker("Resolution", selection: $selectedResolution) {
                            Text("Low").tag(StreamResolution.low)
                            Text("Medium").tag(StreamResolution.medium)
                            Text("High").tag(StreamResolution.high)
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 200)
                    }

                    HStack {
                        Text("Frame Rate")
                            .font(.subheadline)
                        Spacer()
                        Picker("FPS", selection: $selectedFPS) {
                            Text("15").tag(15)
                            Text("24").tag(24)
                            Text("30").tag(30)
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 150)
                    }
                }

                // Stream controls
                HStack(spacing: 12) {
                    if connectionState == .streaming {
                        Button(action: onStopStream) {
                            Label("Stop", systemImage: "stop.fill")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.red)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }

                        Button(action: onCapturePhoto) {
                            Image(systemName: "camera.fill")
                                .padding()
                                .background(Color.orange)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    } else {
                        Button(action: onStartStream) {
                            Label("Start Stream", systemImage: "play.fill")
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.green)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                    }
                }

                // Disconnect button
                Button(action: onDisconnect) {
                    Label("Disconnect", systemImage: "link.badge.plus")
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(.systemGray5))
                        .foregroundColor(.primary)
                        .cornerRadius(10)
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

// MARK: - Server Connection Card
struct ServerConnectionCard: View {
    let isConnected: Bool
    @Binding var serverURL: String
    let gameState: GameState?
    let analysis: AnalysisResult?
    let error: String?
    let onConnect: () -> Void
    let onDisconnect: () -> Void
    let onRequestAnalysis: () -> Void
    let onTestHand: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: "server.rack")
                    .foregroundColor(isConnected ? .green : .gray)
                Text("AI Server")
                    .font(.headline)
                Spacer()
                Circle()
                    .fill(isConnected ? Color.green : Color.gray)
                    .frame(width: 10, height: 10)
            }

            // Server URL input
            if !isConnected {
                TextField("Server URL", text: $serverURL)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                    .autocapitalization(.none)
            }

            // Error message
            if let error = error {
                Text(error)
                    .font(.caption)
                    .foregroundColor(.red)
            }

            // Game state display
            if let state = gameState {
                VStack(alignment: .leading, spacing: 4) {
                    Divider()
                    Text("Hand #\(state.handNumber) - \(state.street.rawValue.capitalized)")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    if !state.heroCards.isEmpty {
                        HStack {
                            Text("Cards:")
                                .font(.caption)
                            Text(state.heroCards.joined(separator: " "))
                                .font(.caption.monospaced())
                                .fontWeight(.bold)
                        }
                    }

                    if !state.boardCards.isEmpty {
                        HStack {
                            Text("Board:")
                                .font(.caption)
                            Text(state.boardCards.joined(separator: " "))
                                .font(.caption.monospaced())
                        }
                    }

                    HStack {
                        Text("Pot: \(state.pot)")
                        Spacer()
                        Text("To call: \(state.toCall)")
                    }
                    .font(.caption)
                }
            }

            // Analysis display
            if let analysis = analysis {
                VStack(alignment: .leading, spacing: 4) {
                    Divider()
                    HStack {
                        Text("Recommendation:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(analysis.action.uppercased())
                            .font(.caption.bold())
                            .foregroundColor(.blue)
                        if let amount = analysis.amount, amount > 0 {
                            Text("(\(amount))")
                                .font(.caption)
                        }
                    }
                    Text(String(format: "Equity: %.1f%%", analysis.equity * 100))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

            // Buttons
            HStack(spacing: 8) {
                if isConnected {
                    Button(action: onTestHand) {
                        Text("Test Hand")
                            .font(.caption)
                            .frame(maxWidth: .infinity)
                            .padding(8)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(6)
                    }

                    Button(action: onRequestAnalysis) {
                        Text("Get Analysis")
                            .font(.caption)
                            .frame(maxWidth: .infinity)
                            .padding(8)
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(6)
                    }

                    Button(action: onDisconnect) {
                        Text("Disconnect")
                            .font(.caption)
                            .frame(maxWidth: .infinity)
                            .padding(8)
                            .background(Color(.systemGray5))
                            .foregroundColor(.primary)
                            .cornerRadius(6)
                    }
                } else {
                    Button(action: onConnect) {
                        Label("Connect to Server", systemImage: "bolt.fill")
                            .font(.subheadline)
                            .frame(maxWidth: .infinity)
                            .padding(10)
                            .background(Color.purple)
                            .foregroundColor(.white)
                            .cornerRadius(8)
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

#Preview {
    ContentView()
}
