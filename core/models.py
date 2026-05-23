from __future__ import annotations
from pydantic import BaseModel, Field, computed_field
from typing import Optional, List
from enum import Enum


class Market(str, Enum):
    CRYPTO = "crypto"
    INDIA = "india"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class SignalOutcome(str, Enum):
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"


class AgentVote(BaseModel):
    agent_name: str
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Position(BaseModel):
    signal_id: str
    ticker: str
    market: Market
    direction: Direction
    entry_price: float = Field(gt=0.0)
    stop_price: float = Field(gt=0.0)
    target_price: float = Field(gt=0.0)
    size_inr: float = Field(gt=0.0)
    opened_at: str
    broker_order_id: str
    instrument_token: Optional[str] = None
    option_type: Optional[str] = None  # "CE" or "PE"
    strike: Optional[float] = None
    expiry: Optional[str] = None


class Signal(BaseModel):
    id: str
    market: Market
    ticker: str
    direction: Direction
    entry_price: float = Field(gt=0.0)
    target_price: float = Field(gt=0.0)
    stop_price: float = Field(gt=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    position_size_inr: float = Field(gt=0.0)
    generated_at: str
    executed: bool = False
    outcome: SignalOutcome = SignalOutcome.PENDING
    outcome_price: Optional[float] = None
    outcome_at: Optional[str] = None
    hypothetical_pnl_pct: Optional[float] = None
    hypothetical_pnl_inr: Optional[float] = None
    agent_votes: List[AgentVote] = Field(default_factory=list)
    debate_transcript: str = ""

    @computed_field
    @property
    def rr_ratio(self) -> float:
        reward = abs(self.target_price - self.entry_price)
        risk = abs(self.entry_price - self.stop_price)
        if risk == 0:
            return 0.0
        return reward / risk


class HarnessState(BaseModel):
    session_count: int = 0
    portfolio_value_inr: float = 100000.0
    open_positions: List[Position] = Field(default_factory=list)
    signal_history: List[str] = Field(default_factory=list)
    agent_learnings: str = ""
    last_run: Optional[str] = None
