from core.models import (
    Signal,
    HarnessState,
    AgentVote,
    Market,
    Direction,
    SignalOutcome,
)


def test_signal_creation():
    s = Signal(
        id="sig_001",
        market=Market.INDIA,
        ticker="NIFTY",
        direction=Direction.LONG,
        entry_price=62400.0,
        target_price=65100.0,
        stop_price=61000.0,
        confidence=0.74,
        position_size_inr=2000.0,
        generated_at="2026-05-23T08:00:00Z",
    )
    assert s.executed == False
    assert s.outcome == SignalOutcome.PENDING
    assert s.hypothetical_pnl_inr is None


def test_signal_rr_ratio():
    s = Signal(
        id="sig_002",
        market=Market.INDIA,
        ticker="BANKNIFTY",
        direction=Direction.LONG,
        entry_price=3000.0,
        target_price=3300.0,
        stop_price=2850.0,
        confidence=0.70,
        position_size_inr=2000.0,
        generated_at="2026-05-23T08:00:00Z",
    )
    assert round(s.rr_ratio, 2) == 2.0


def test_harness_state_serialization():
    state = HarnessState()
    d = state.model_dump()
    restored = HarnessState(**d)
    assert restored.portfolio_value_inr == 100000.0
    assert restored.open_positions == []


def test_agent_vote():
    vote = AgentVote(
        agent_name="KronosTechnical",
        direction=Direction.LONG,
        confidence=0.80,
        reasoning="Pattern matches 8/10 historical bullish setups",
    )
    assert vote.confidence == 0.80
