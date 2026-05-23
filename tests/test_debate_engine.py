from core.debate_engine import DebateEngine
from core.models import AgentVote, Direction


def make_vote(direction, confidence, name="Agent"):
    return AgentVote(
        agent_name=name, direction=direction, confidence=confidence, reasoning="test"
    )


def test_consensus_long_majority():
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.80, "A"),
        make_vote(Direction.LONG, 0.70, "B"),
        make_vote(Direction.SHORT, 0.60, "C"),
    ]
    result = engine.reach_consensus(votes)
    assert result["direction"] == Direction.LONG
    assert result["confidence"] >= 0.65


def test_no_consensus_below_threshold():
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.55, "A"),
        make_vote(Direction.SHORT, 0.55, "B"),
        make_vote(Direction.HOLD, 0.0, "C"),
    ]
    result = engine.reach_consensus(votes)
    assert result["direction"] == Direction.HOLD
    assert result["confidence"] < 0.65


def test_consensus_requires_two_agreeing():
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.90, "A"),
        make_vote(Direction.SHORT, 0.80, "B"),
    ]
    result = engine.reach_consensus(votes)
    assert result["direction"] == Direction.HOLD  # only 1 agrees on each side
