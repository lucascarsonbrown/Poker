"""
FastAPI + WebSocket server for poker AI.
Wraps the existing src.calculator.PokerCalculator.
"""
import sys
import os
import json
import asyncio
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calculator import PokerCalculator
from schemas import *
from hand_state import HandState


app = FastAPI(title="Poker AI Server")

# Allow iOS app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global calculator instance (models loaded once)
calculator = PokerCalculator(model_path=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
))

# Track connected clients and their hand states
clients: Dict[str, HandState] = {}


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        clients[client_id] = HandState()
        print(f"[Server] Client connected: {client_id}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in clients:
            del clients[client_id]
        print(f"[Server] Client disconnected: {client_id}")

    async def send_message(self, client_id: str, msg_type: str, data: dict):
        if client_id in self.active_connections:
            message = ServerMessage(msg_type=msg_type, data=data)
            await self.active_connections[client_id].send_json(message.model_dump())


manager = ConnectionManager()


@app.get("/")
async def root():
    return {"status": "ok", "service": "Poker AI Server"}


@app.get("/health")
async def health():
    return {"status": "healthy", "clients": len(clients)}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)

    try:
        # Send initial state
        state = clients[client_id].get_state_response()
        await manager.send_message(client_id, "state", state.model_dump())

        while True:
            # Receive event from iOS
            data = await websocket.receive_json()
            print(f"[Server] Received from {client_id}: {data.get('event_type')}")

            try:
                event = parse_event(data)
                response = await process_event(client_id, event)

                # Send updated state
                state = clients[client_id].get_state_response()
                await manager.send_message(client_id, "state", state.model_dump())

                # If analysis was requested, send that too
                if response:
                    await manager.send_message(client_id, "analysis", response)

            except ValueError as e:
                await manager.send_message(client_id, "error", {"error": str(e)})

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        print(f"[Server] Error: {e}")
        manager.disconnect(client_id)


async def process_event(client_id: str, event) -> dict | None:
    """Process an event and update hand state."""
    hand = clients[client_id]

    if isinstance(event, HandStartEvent):
        hand.start_hand(event)

    elif isinstance(event, HoleCardsEvent):
        hand.set_hole_cards(event)

    elif isinstance(event, BoardUpdateEvent):
        hand.update_board(event)

    elif isinstance(event, ActionEvent):
        hand.process_action(event)

    elif isinstance(event, HandEndEvent):
        hand.end_hand(event)

    elif isinstance(event, RequestAnalysisEvent):
        return get_analysis(hand)

    return None


def get_analysis(hand: HandState) -> dict:
    """Get AI recommendation using PokerCalculator."""
    if not hand.hero_cards:
        return {"error": "No hole cards set"}

    try:
        params = hand.get_calculator_params()
        result = calculator.get_ai_action(**params)

        return AnalysisResponse(
            action=result["action"],
            amount=result.get("amount"),
            equity=result["equity"],
            strategy=result.get("strategy", {}),
        ).model_dump()

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("Starting Poker AI Server...")
    print(f"Models path: {calculator.model_path}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
