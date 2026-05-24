from unittest.mock import patch
from core.debate_engine import DebateEngine
from core.models import AgentVote, Direction


def make_vote(direction, confidence, name="Agent"):
    return AgentVote(
        agent_name=name, direction=direction, confidence=confidence, reasoning="test"
    )


def _mock_debate(direction, confidence):
    """Patch _adversarial_debate to avoid real API calls."""
    return patch.object(
        DebateEngine,
        "_adversarial_debate",
        return_value={
            "direction": direction,
            "confidence": confidence,
            "transcript": "mocked debate",
        },
    )


def test_consensus_long_majority():
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.80, "A"),
        make_vote(Direction.LONG, 0.70, "B"),
        make_vote(Direction.SHORT, 0.60, "C"),
    ]
    with _mock_debate(Direction.LONG, 0.75):
        result = engine.reach_consensus(votes, ticker="BTC/USDT")
    assert result["direction"] == Direction.LONG
    assert result["confidence"] >= 0.65


def test_no_consensus_below_threshold():
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.55, "A"),
        make_vote(Direction.SHORT, 0.55, "B"),
        make_vote(Direction.HOLD, 0.0, "C"),
    ]
    result = engine.reach_consensus(votes, ticker="BTC/USDT")
    assert result["direction"] == Direction.HOLD
    assert result["confidence"] < 0.65


def test_consensus_requires_two_agreeing():
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.90, "A"),
        make_vote(Direction.SHORT, 0.80, "B"),
    ]
    result = engine.reach_consensus(votes, ticker="BTC/USDT")
    assert result["direction"] == Direction.HOLD


def test_adjudicator_can_override_to_hold():
    """Opus adjudicator returning HOLD should override initial LONG consensus."""
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.80, "A"),
        make_vote(Direction.LONG, 0.75, "B"),
    ]
    with _mock_debate(Direction.HOLD, 0.0):
        result = engine.reach_consensus(votes, ticker="ETH/USDT")
    assert result["direction"] == Direction.HOLD


def test_parse_adjudication_fallback():
    engine = DebateEngine()
    direction, confidence = engine._parse_adjudication(
        "gibberish", Direction.LONG, 0.70
    )
    assert direction == Direction.LONG
    assert confidence == 0.70


def test_parse_adjudication_parses_correctly():
    engine = DebateEngine()
    text = "DECISION: SHORT\nCONFIDENCE: 0.82\nREASON: Bearish momentum confirmed."
    direction, confidence = engine._parse_adjudication(text, Direction.LONG, 0.70)
    assert direction == Direction.SHORT
    assert confidence == 0.82
