"""
This will house the event scemas for the iOS->Server communication
"""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class ActionType(str, Enum):
    FOLD = "f"
    CHECK = "k"
    CALL = "c"
    BET_MIN = "bMIN"
    BET_MID = "bMID"
    BET_MAX = "bMAX"


class Player(str, Enum):
    HERO = "hero"
    VILLAIN = "villain"


# Events: iOS -> Server

class HandStartEvent(BaseModel):
    event_type: str = "hand_start"
    hero_stack: int
    villain_stack: int
    small_blind: int = 1
    big_blind: int = 2
    hero_is_button: bool = True


class HoleCardsEvent(BaseModel):
    event_type: str = "hole_cards"
    cards: List[str]  # e.g. ["Ah", "Kd"]
    confidence: float = 1.0


class BoardUpdateEvent(BaseModel):
    event_type: str = "board_update"
    cards: List[str]  # e.g. ["Qh", "Jd", "Ts"]
    street: Street
    confidence: float = 1.0


class ActionEvent(BaseModel):
    event_type: str = "action"
    player: Player
    action_type: ActionType
    amount: Optional[int] = None
    street: Street
    confidence: float = 1.0


class HandEndEvent(BaseModel):
    event_type: str = "hand_end"
    winner: Optional[Player] = None
    pot_won: Optional[int] = None
    showdown: bool = False
    villain_cards: Optional[List[str]] = None


class RequestAnalysisEvent(BaseModel):
    event_type: str = "request_analysis"


# Responses: Server -> iOS

class GameStateResponse(BaseModel):
    """Current game state for the mirror UI."""
    hand_number: int = 0
    street: Street = Street.PREFLOP
    hero_cards: List[str] = []
    board_cards: List[str] = []
    pot: int = 0
    hero_stack: int = 0
    villain_stack: int = 0
    hero_to_act: bool = False
    to_call: int = 0
    action_history: List[Dict[str, Any]] = []


class AnalysisResponse(BaseModel):
    """AI recommendation from PokerCalculator."""
    action: str
    amount: Optional[int] = None
    equity: float
    strategy: Dict[str, float] = {}


class ServerMessage(BaseModel):
    """All server responses wrapped in this."""
    msg_type: str  # "state", "analysis", "error", "ack"
    data: Dict[str, Any]


# Event parsing

EVENT_MAP = {
    "hand_start": HandStartEvent,
    "hole_cards": HoleCardsEvent,
    "board_update": BoardUpdateEvent,
    "action": ActionEvent,
    "hand_end": HandEndEvent,
    "request_analysis": RequestAnalysisEvent,
}


def parse_event(data: dict) -> BaseModel:
    event_type = data.get("event_type")
    if event_type not in EVENT_MAP:
        raise ValueError(f"Unknown event: {event_type}")
    return EVENT_MAP[event_type](**data)
