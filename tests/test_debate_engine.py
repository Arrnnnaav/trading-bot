import pytest
from unittest.mock import AsyncMock, patch
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
        new=AsyncMock(
            return_value={
                "direction": direction,
                "confidence": confidence,
                "transcript": "mocked debate",
            }
        ),
    )


@pytest.mark.asyncio
async def test_consensus_long_majority():
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.80, "A"),
        make_vote(Direction.LONG, 0.70, "B"),
        make_vote(Direction.SHORT, 0.60, "C"),
    ]
    with _mock_debate(Direction.LONG, 0.75):
        result = await engine.reach_consensus(votes, ticker="BTC/USDT")
    assert result["direction"] == Direction.LONG
    assert result["confidence"] >= 0.65


@pytest.mark.asyncio
async def test_no_consensus_below_threshold():
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.55, "A"),
        make_vote(Direction.SHORT, 0.55, "B"),
        make_vote(Direction.HOLD, 0.0, "C"),
    ]
    result = await engine.reach_consensus(votes, ticker="BTC/USDT")
    assert result["direction"] == Direction.HOLD
    assert result["confidence"] < 0.65


@pytest.mark.asyncio
async def test_consensus_requires_two_agreeing():
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.90, "A"),
        make_vote(Direction.SHORT, 0.80, "B"),
    ]
    result = await engine.reach_consensus(votes, ticker="BTC/USDT")
    assert result["direction"] == Direction.HOLD


@pytest.mark.asyncio
async def test_adjudicator_can_override_to_hold(monkeypatch):
    """Opus adjudicator returning HOLD should override initial LONG consensus."""
    from config import config

    monkeypatch.setattr(config, "enable_llm_adjudication", True)
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.80, "A"),
        make_vote(Direction.LONG, 0.75, "B"),
    ]
    with _mock_debate(Direction.HOLD, 0.0):
        result = await engine.reach_consensus(votes, ticker="ETH/USDT")
    assert result["direction"] == Direction.HOLD


@pytest.mark.asyncio
async def test_llm_adjudication_defaults_off(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "enable_llm_adjudication", False)
    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.80, "A"),
        make_vote(Direction.LONG, 0.75, "B"),
    ]
    with _mock_debate(Direction.HOLD, 0.0) as debate:
        result = await engine.reach_consensus(votes, ticker="ETH/USDT")
    assert result["direction"] == Direction.LONG
    debate.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_adversarial_debate_makes_three_sequential_haiku_calls():
    """Bull R1 → Bear R2 → Bull R3 → opus. Three haiku calls, one opus call."""
    from unittest.mock import patch

    engine = DebateEngine()
    votes = [
        make_vote(Direction.LONG, 0.80, "A"),
        make_vote(Direction.LONG, 0.75, "B"),
    ]
    haiku_responses = ["bull_opening", "bear_attack", "bull_rebuttal"]
    opus_response = "DECISION: LONG\nCONFIDENCE: 0.82\nREASON: Bull case held."

    call_count = 0

    async def fake_call(prompt, system, model):
        nonlocal call_count
        call_count += 1
        if model == "claude-haiku-4-5-20251001":
            return haiku_responses.pop(0)
        return opus_response

    with patch.object(engine._client, "call", side_effect=fake_call):
        result = await engine._adversarial_debate(votes, "NSEI", Direction.LONG, 0.77)

    assert call_count == 4  # 3 haiku + 1 opus
    assert result["direction"] == Direction.LONG
    assert result["confidence"] == 0.82


@pytest.mark.asyncio
async def test_bear_receives_bull_r1_in_prompt():
    """Bear R2 prompt must contain Bull R1 text so bear can specifically attack it."""
    engine = DebateEngine()
    votes = [
        make_vote(Direction.SHORT, 0.80, "A"),
        make_vote(Direction.SHORT, 0.75, "B"),
    ]
    captured_prompts = []

    async def capture_call(prompt, system, model):
        captured_prompts.append((model, prompt))
        if model == "claude-haiku-4-5-20251001":
            return f"response_{len(captured_prompts)}"
        return "DECISION: SHORT\nCONFIDENCE: 0.78\nREASON: ok"

    with patch.object(engine._client, "call", side_effect=capture_call):
        await engine._adversarial_debate(votes, "NSEBANK", Direction.SHORT, 0.77)

    haiku_calls = [
        (m, p) for m, p in captured_prompts if m == "claude-haiku-4-5-20251001"
    ]
    assert len(haiku_calls) == 3
    # Bear R2 (2nd haiku call) prompt must reference bull's opening
    bear_prompt = haiku_calls[1][1]
    assert "response_1" in bear_prompt  # bull R1 content appears in bear R2 prompt


@pytest.mark.asyncio
async def test_bull_rebuttal_receives_bear_attack_in_prompt():
    """Bull R3 prompt must contain Bear R2 text so bull can rebut specifically."""
    engine = DebateEngine()
    votes = [make_vote(Direction.LONG, 0.80, "A"), make_vote(Direction.LONG, 0.75, "B")]
    captured_prompts = []

    async def capture_call(prompt, system, model):
        captured_prompts.append((model, prompt))
        if model == "claude-haiku-4-5-20251001":
            return f"round_{len([p for p in captured_prompts if p[0] == 'claude-haiku-4-5-20251001'])}"
        return "DECISION: LONG\nCONFIDENCE: 0.80\nREASON: ok"

    with patch.object(engine._client, "call", side_effect=capture_call):
        await engine._adversarial_debate(votes, "NSEI", Direction.LONG, 0.77)

    haiku_calls = [
        (m, p) for m, p in captured_prompts if m == "claude-haiku-4-5-20251001"
    ]
    # Bull R3 (3rd haiku) prompt must contain bear's attack text
    bull_r3_prompt = haiku_calls[2][1]
    assert "round_2" in bull_r3_prompt  # bear R2 content in bull R3 prompt


@pytest.mark.asyncio
async def test_adjudicator_receives_all_three_rounds():
    """Opus prompt must contain bull_r1, bear_r2, and bull_r3 text."""
    engine = DebateEngine()
    votes = [make_vote(Direction.LONG, 0.80, "A"), make_vote(Direction.LONG, 0.75, "B")]
    haiku_responses = ["BULL_OPENING", "BEAR_ATTACK", "BULL_REBUTTAL"]
    opus_prompt_captured = []

    async def capture_call(prompt, system, model):
        if model == "claude-haiku-4-5-20251001":
            return haiku_responses.pop(0)
        opus_prompt_captured.append(prompt)
        return "DECISION: LONG\nCONFIDENCE: 0.80\nREASON: ok"

    with patch.object(engine._client, "call", side_effect=capture_call):
        await engine._adversarial_debate(votes, "NSEI", Direction.LONG, 0.77)

    assert len(opus_prompt_captured) == 1
    opus_prompt = opus_prompt_captured[0]
    assert "BULL_OPENING" in opus_prompt
    assert "BEAR_ATTACK" in opus_prompt
    assert "BULL_REBUTTAL" in opus_prompt


@pytest.mark.asyncio
async def test_transcript_includes_all_three_round_labels():
    """Output transcript must have [Bull R1], [Bear R2], [Bull R3] labels."""
    engine = DebateEngine()
    votes = [make_vote(Direction.LONG, 0.80, "A"), make_vote(Direction.LONG, 0.75, "B")]
    haiku_responses = ["opening", "attack", "rebuttal"]

    async def fake_call(prompt, system, model):
        if model == "claude-haiku-4-5-20251001":
            return haiku_responses.pop(0)
        return "DECISION: LONG\nCONFIDENCE: 0.80\nREASON: ok"

    with patch.object(engine._client, "call", side_effect=fake_call):
        result = await engine._adversarial_debate(votes, "NSEI", Direction.LONG, 0.77)

    assert "[Bull R1]" in result["transcript"]
    assert "[Bear R2]" in result["transcript"]
    assert "[Bull R3]" in result["transcript"]
