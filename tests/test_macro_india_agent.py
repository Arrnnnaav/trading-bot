from agents.macro_india import MacroIndiaAgent
from core.models import Direction, Market


def _klines():
    return [
        {"open": 100, "high": 105, "low": 95, "close": 100, "volume": 1000}
        for _ in range(10)
    ]


def test_macro_sgx_positive_gap_is_long():
    agent = MacroIndiaAgent()
    vote = agent.analyze(
        "NIFTY",
        _klines(),
        Market.INDIA,
        vix=15.0,
        sgx_gap_pct=1.5,
        usdinr_change_pct=0.0,
        is_rbi_day=False,
    )
    assert vote.direction == Direction.LONG


def test_macro_sgx_negative_gap_is_short():
    agent = MacroIndiaAgent()
    vote = agent.analyze(
        "NIFTY",
        _klines(),
        Market.INDIA,
        vix=15.0,
        sgx_gap_pct=-1.5,
        usdinr_change_pct=0.0,
        is_rbi_day=False,
    )
    assert vote.direction == Direction.SHORT


def test_macro_rbi_day_is_hold():
    agent = MacroIndiaAgent()
    vote = agent.analyze(
        "NIFTY",
        _klines(),
        Market.INDIA,
        vix=15.0,
        sgx_gap_pct=2.0,
        usdinr_change_pct=0.0,
        is_rbi_day=True,
    )
    assert vote.direction == Direction.HOLD


def test_macro_high_vix_reduces_confidence():
    agent = MacroIndiaAgent()
    normal = agent.analyze(
        "NIFTY",
        _klines(),
        Market.INDIA,
        vix=15.0,
        sgx_gap_pct=1.5,
        usdinr_change_pct=0.0,
        is_rbi_day=False,
    )
    elevated = agent.analyze(
        "NIFTY",
        _klines(),
        Market.INDIA,
        vix=22.0,
        sgx_gap_pct=1.5,
        usdinr_change_pct=0.0,
        is_rbi_day=False,
    )
    # Both should be LONG, but elevated VIX should have lower confidence
    assert normal.direction == Direction.LONG
    assert elevated.direction == Direction.LONG
    assert elevated.confidence < normal.confidence


def test_macro_usdinr_surge_is_short():
    agent = MacroIndiaAgent()
    vote = agent.analyze(
        "NIFTY",
        _klines(),
        Market.INDIA,
        vix=15.0,
        sgx_gap_pct=0.0,
        usdinr_change_pct=0.6,
        is_rbi_day=False,
    )
    assert vote.direction == Direction.SHORT


def test_macro_agent_never_raises():
    agent = MacroIndiaAgent()
    vote = agent.analyze("NIFTY", [], Market.INDIA)
    assert vote.direction in (Direction.LONG, Direction.SHORT, Direction.HOLD)
