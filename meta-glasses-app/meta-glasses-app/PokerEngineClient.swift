import Foundation
import Combine

// MARK: - Event Types (iOS → Server)

enum Street: String, Codable {
    case preflop, flop, turn, river
}

enum ActionType: String, Codable {
    case fold = "f"
    case check = "k"
    case call = "c"
    case betMin = "bMIN"
    case betMid = "bMID"
    case betMax = "bMAX"
}

enum Player: String, Codable {
    case hero, villain
}

struct HandStartEvent: Codable {
    let eventType: String = "hand_start"
    let heroStack: Int
    let villainStack: Int
    let smallBlind: Int
    let bigBlind: Int
    let heroIsButton: Bool

    enum CodingKeys: String, CodingKey {
        case eventType = "event_type"
        case heroStack = "hero_stack"
        case villainStack = "villain_stack"
        case smallBlind = "small_blind"
        case bigBlind = "big_blind"
        case heroIsButton = "hero_is_button"
    }
}

struct HoleCardsEvent: Codable {
    let eventType: String = "hole_cards"
    let cards: [String]
    let confidence: Double

    enum CodingKeys: String, CodingKey {
        case eventType = "event_type"
        case cards, confidence
    }
}

struct BoardUpdateEvent: Codable {
    let eventType: String = "board_update"
    let cards: [String]
    let street: Street
    let confidence: Double

    enum CodingKeys: String, CodingKey {
        case eventType = "event_type"
        case cards, street, confidence
    }
}

struct ActionEvent: Codable {
    let eventType: String = "action"
    let player: Player
    let actionType: ActionType
    let amount: Int?
    let street: Street
    let confidence: Double

    enum CodingKeys: String, CodingKey {
        case eventType = "event_type"
        case player
        case actionType = "action_type"
        case amount, street, confidence
    }
}

struct HandEndEvent: Codable {
    let eventType: String = "hand_end"
    let winner: Player?
    let potWon: Int?
    let showdown: Bool
    let villainCards: [String]?

    enum CodingKeys: String, CodingKey {
        case eventType = "event_type"
        case winner
        case potWon = "pot_won"
        case showdown
        case villainCards = "villain_cards"
    }
}

struct RequestAnalysisEvent: Codable {
    let eventType: String = "request_analysis"

    enum CodingKeys: String, CodingKey {
        case eventType = "event_type"
    }
}

// MARK: - Response Types (Server → iOS)

struct GameState: Codable {
    let handNumber: Int
    let street: Street
    let heroCards: [String]
    let boardCards: [String]
    let pot: Int
    let heroStack: Int
    let villainStack: Int
    let heroToAct: Bool
    let toCall: Int
    let actionHistory: [[String: AnyCodable]]

    enum CodingKeys: String, CodingKey {
        case handNumber = "hand_number"
        case street
        case heroCards = "hero_cards"
        case boardCards = "board_cards"
        case pot
        case heroStack = "hero_stack"
        case villainStack = "villain_stack"
        case heroToAct = "hero_to_act"
        case toCall = "to_call"
        case actionHistory = "action_history"
    }
}

struct AnalysisResult: Codable {
    let action: String
    let amount: Int?
    let equity: Double
    let strategy: [String: Double]
}

struct ServerMessage: Codable {
    let msgType: String
    let data: [String: AnyCodable]

    enum CodingKeys: String, CodingKey {
        case msgType = "msg_type"
        case data
    }
}

// Helper for decoding arbitrary JSON
struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let int = try? container.decode(Int.self) {
            value = int
        } else if let double = try? container.decode(Double.self) {
            value = double
        } else if let string = try? container.decode(String.self) {
            value = string
        } else if let bool = try? container.decode(Bool.self) {
            value = bool
        } else if let array = try? container.decode([AnyCodable].self) {
            value = array.map { $0.value }
        } else if let dict = try? container.decode([String: AnyCodable].self) {
            value = dict.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let int as Int: try container.encode(int)
        case let double as Double: try container.encode(double)
        case let string as String: try container.encode(string)
        case let bool as Bool: try container.encode(bool)
        case let array as [Any]: try container.encode(array.map { AnyCodable($0) })
        case let dict as [String: Any]: try container.encode(dict.mapValues { AnyCodable($0) })
        default: try container.encodeNil()
        }
    }
}

// MARK: - WebSocket Client

@MainActor
class PokerEngineClient: ObservableObject {
    static let shared = PokerEngineClient()

    @Published var isConnected = false
    @Published var gameState: GameState?
    @Published var latestAnalysis: AnalysisResult?
    @Published var connectionError: String?

    private var webSocket: URLSessionWebSocketTask?
    private var session: URLSession?
    private let clientId = UUID().uuidString

    var serverURL: String = "ws://localhost:8000"

    private init() {}

    // MARK: - Connection

    func connect() {
        guard !isConnected else { return }

        let urlString = "\(serverURL)/ws/\(clientId)"
        guard let url = URL(string: urlString) else {
            connectionError = "Invalid URL"
            return
        }

        session = URLSession(configuration: .default)
        webSocket = session?.webSocketTask(with: url)
        webSocket?.resume()

        isConnected = true
        connectionError = nil
        print("[PokerEngine] Connected to \(urlString)")

        receiveMessage()
    }

    func disconnect() {
        webSocket?.cancel(with: .goingAway, reason: nil)
        webSocket = nil
        isConnected = false
        gameState = nil
        print("[PokerEngine] Disconnected")
    }

    // MARK: - Receive Messages

    private func receiveMessage() {
        webSocket?.receive { [weak self] result in
            Task { @MainActor in
                switch result {
                case .success(let message):
                    self?.handleMessage(message)
                    self?.receiveMessage() // Continue listening
                case .failure(let error):
                    self?.connectionError = error.localizedDescription
                    self?.isConnected = false
                    print("[PokerEngine] Receive error: \(error)")
                }
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            parseServerMessage(text)
        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                parseServerMessage(text)
            }
        @unknown default:
            break
        }
    }

    private func parseServerMessage(_ text: String) {
        guard let data = text.data(using: .utf8) else { return }

        do {
            let message = try JSONDecoder().decode(ServerMessage.self, from: data)

            switch message.msgType {
            case "state":
                let stateData = try JSONSerialization.data(withJSONObject: message.data.mapValues { $0.value })
                gameState = try JSONDecoder().decode(GameState.self, from: stateData)
                print("[PokerEngine] State updated: hand #\(gameState?.handNumber ?? 0)")

            case "analysis":
                let analysisData = try JSONSerialization.data(withJSONObject: message.data.mapValues { $0.value })
                latestAnalysis = try JSONDecoder().decode(AnalysisResult.self, from: analysisData)
                print("[PokerEngine] Analysis: \(latestAnalysis?.action ?? "?")")

            case "error":
                if let error = message.data["error"]?.value as? String {
                    connectionError = error
                    print("[PokerEngine] Error: \(error)")
                }

            default:
                print("[PokerEngine] Unknown message type: \(message.msgType)")
            }
        } catch {
            print("[PokerEngine] Parse error: \(error)")
        }
    }

    // MARK: - Send Events

    private func send<T: Encodable>(_ event: T) {
        guard isConnected else {
            print("[PokerEngine] Not connected")
            return
        }

        do {
            let data = try JSONEncoder().encode(event)
            if let json = String(data: data, encoding: .utf8) {
                webSocket?.send(.string(json)) { error in
                    if let error = error {
                        print("[PokerEngine] Send error: \(error)")
                    }
                }
            }
        } catch {
            print("[PokerEngine] Encode error: \(error)")
        }
    }

    // MARK: - Public API

    func startHand(heroStack: Int, villainStack: Int, smallBlind: Int = 1, bigBlind: Int = 2, heroIsButton: Bool = true) {
        let event = HandStartEvent(
            heroStack: heroStack,
            villainStack: villainStack,
            smallBlind: smallBlind,
            bigBlind: bigBlind,
            heroIsButton: heroIsButton
        )
        send(event)
    }

    func setHoleCards(_ cards: [String], confidence: Double = 1.0) {
        let event = HoleCardsEvent(cards: cards, confidence: confidence)
        send(event)
    }

    func updateBoard(_ cards: [String], street: Street, confidence: Double = 1.0) {
        let event = BoardUpdateEvent(cards: cards, street: street, confidence: confidence)
        send(event)
    }

    func recordAction(player: Player, action: ActionType, amount: Int? = nil, street: Street, confidence: Double = 1.0) {
        let event = ActionEvent(
            player: player,
            actionType: action,
            amount: amount,
            street: street,
            confidence: confidence
        )
        send(event)
    }

    func endHand(winner: Player? = nil, potWon: Int? = nil, showdown: Bool = false, villainCards: [String]? = nil) {
        let event = HandEndEvent(
            winner: winner,
            potWon: potWon,
            showdown: showdown,
            villainCards: villainCards
        )
        send(event)
    }

    func requestAnalysis() {
        let event = RequestAnalysisEvent()
        send(event)
        print("[PokerEngine] Analysis requested")
    }
}
