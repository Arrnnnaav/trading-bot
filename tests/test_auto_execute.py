import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from core.models import Direction, Market, Signal


def _make_signal(ticker="NIFTY", direction=Direction.LONG):
    return Signal(
        id="sig_test_auto_001",
        market=Market.INDIA,
        ticker=ticker,
        direction=direction,
        entry_price=23000.0,
        target_price=46000.0,
        stop_price=14950.0,
        confidence=0.75,
        position_size_inr=2000.0,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def auto_harness(tmp_path):
    # Use MagicMock() instances (not classes) so __init__ args don't become specs.
    mock_debate_cls = MagicMock(return_value=MagicMock())
    mock_risk_cls = MagicMock(return_value=MagicMock())
    mock_tracker_cls = MagicMock(return_value=MagicMock())
    mock_pattern_cls = MagicMock(return_value=MagicMock())
    mock_llm_cls = MagicMock(return_value=MagicMock())

    patches = {
        "DebateEngine": mock_debate_cls,
        "RiskManager": mock_risk_cls,
        "AgentPerformanceTracker": mock_tracker_cls,
        "PatternReputationTracker": mock_pattern_cls,
        "ClaudeCodeClient": mock_llm_cls,
    }
    with patch.multiple("harnesses.base_harness", **patches):
        from harnesses.base_harness import BaseHarness

        class ConcreteHarness(BaseHarness):
            MARKET = Market.INDIA

        broker = MagicMock()
        broker.place_options_order.return_value = {
            "order_id": "paper-opt-test123",
            "status": "paper",
            "quantity": 65,
            "expiry": "2026-06-26",
            "strike": 23000.0,
        }
        broker.resolve_options_expiry.return_value = "2026-06-26"

        aggregator = MagicMock()
        aggregator.is_duplicate.return_value = False
        aggregator.get_all_signals.return_value = []

        harness = ConcreteHarness(
            state_path=str(tmp_path / "state.json"),
            broker=broker,
            agents=[],
            telegram_bot=None,
            signal_aggregator=aggregator,
        )
        return harness, broker, aggregator


@pytest.mark.asyncio
async def test_run_session_calls_place_options_order(auto_harness):
    harness, broker, aggregator = auto_harness
    signal = _make_signal()
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.LONG, "confidence": 0.75, "transcript": ""}
    )
    harness.risk_manager.approve.return_value = (True, "ok")
    harness.risk_manager.build_signal.return_value = signal

    await harness.run_session("NIFTY", [])

    broker.place_options_order.assert_called_once()
    call_kwargs = broker.place_options_order.call_args[1]
    assert call_kwargs["index"] == "NIFTY"
    assert call_kwargs["direction"] == "LONG"
    assert call_kwargs["size_inr"] == 2000.0


@pytest.mark.asyncio
async def test_run_session_adds_position_to_state(auto_harness):
    harness, broker, aggregator = auto_harness
    signal = _make_signal()
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.LONG, "confidence": 0.75, "transcript": ""}
    )
    harness.risk_manager.approve.return_value = (True, "ok")
    harness.risk_manager.build_signal.return_value = signal

    await harness.run_session("NIFTY", [])

    assert len(harness.state.open_positions) == 1
    pos = harness.state.open_positions[0]
    assert pos.ticker == "NIFTY"
    assert pos.direction == Direction.LONG
    assert pos.signal_id == "sig_test_auto_001"
    assert pos.broker_order_id == "paper-opt-test123"


@pytest.mark.asyncio
async def test_run_session_marks_signal_executed(auto_harness):
    harness, broker, aggregator = auto_harness
    signal = _make_signal()
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.LONG, "confidence": 0.75, "transcript": ""}
    )
    harness.risk_manager.approve.return_value = (True, "ok")
    harness.risk_manager.build_signal.return_value = signal

    await harness.run_session("NIFTY", [])

    aggregator.mark_signal_executed.assert_called_once_with("sig_test_auto_001")


@pytest.mark.asyncio
async def test_run_session_survives_broker_exception(auto_harness):
    """Broker crash must not propagate — signal is still returned."""
    harness, broker, aggregator = auto_harness
    broker.place_options_order.side_effect = RuntimeError("API down")
    signal = _make_signal()
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.LONG, "confidence": 0.75, "transcript": ""}
    )
    harness.risk_manager.approve.return_value = (True, "ok")
    harness.risk_manager.build_signal.return_value = signal

    result = await harness.run_session("NIFTY", [])

    assert result is not None  # signal returned even though broker failed
    assert len(harness.state.open_positions) == 0  # position NOT added on failure


@pytest.mark.asyncio
async def test_run_session_hold_skips_execution(auto_harness):
    harness, broker, aggregator = auto_harness
    harness.debate_engine.reach_consensus = AsyncMock(
        return_value={"direction": Direction.HOLD, "confidence": 0.5, "transcript": ""}
    )

    result = await harness.run_session("NIFTY", [])

    assert result is None
    broker.place_options_order.assert_not_called()
