"""
DebateEngine — multi-agent vote consensus + adversarial bull/bear debate.

Flow:
  1. reach_consensus(votes, ticker) — vote counting + threshold check
  2. If direction != HOLD: _adversarial_debate() runs
     a. Bull researcher (haiku)  — argues FOR the trade
     b. Bear researcher (haiku)  — argues AGAINST the trade
     c. Opus adjudicator         — makes final LONG/SHORT/HOLD + confidence
  3. generate_reasoning()        — Telegram-facing 2-sentence rationale (haiku)
"""

import re
from anthropic import Anthropic
from core.models import AgentVote, Direction
from config import config


class DebateEngine:
    def __init__(self, learnings: str = ""):
        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.learnings = learnings

    def update_learnings(self, learnings: str):
        self.learnings = learnings

    # ── Public API ────────────────────────────────────────────────────────────

    def reach_consensus(self, votes: list[AgentVote], ticker: str = "") -> dict:
        if not votes:
            return {
                "direction": Direction.HOLD,
                "confidence": 0.0,
                "transcript": "No votes",
            }

        active = [
            v for v in votes if v.direction != Direction.HOLD and v.confidence >= 0.50
        ]
        long_v = [v for v in active if v.direction == Direction.LONG]
        short_v = [v for v in active if v.direction == Direction.SHORT]

        if len(long_v) >= 2 and len(long_v) > len(short_v):
            direction, agreeing = Direction.LONG, long_v
        elif len(short_v) >= 2 and len(short_v) > len(long_v):
            direction, agreeing = Direction.SHORT, short_v
        else:
            return {
                "direction": Direction.HOLD,
                "confidence": 0.0,
                "transcript": self._build_transcript(votes),
            }

        confidence = sum(v.confidence for v in agreeing) / len(agreeing)
        if confidence < config.min_confidence:
            return {
                "direction": Direction.HOLD,
                "confidence": confidence,
                "transcript": self._build_transcript(votes),
            }

        # Adversarial debate — may override direction or confidence
        return self._adversarial_debate(votes, ticker, direction, round(confidence, 3))

    def generate_reasoning(
        self, votes: list[AgentVote], direction: Direction, ticker: str
    ) -> str:
        transcript = self._build_transcript(votes)
        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=self._build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"Agent debate for {ticker}:\n{transcript}\n\nConsensus: {direction.value}\n"
                    "Write a 2-sentence trading rationale. Be specific about key signals.",
                }
            ],
        )
        return resp.content[0].text.strip()

    # ── Adversarial debate ────────────────────────────────────────────────────

    def _adversarial_debate(
        self,
        votes: list[AgentVote],
        ticker: str,
        initial_direction: Direction,
        initial_confidence: float,
    ) -> dict:
        transcript = self._build_transcript(votes)
        bull_arg = self._run_researcher("bull", transcript, ticker, initial_direction)
        bear_arg = self._run_researcher("bear", transcript, ticker, initial_direction)
        return self._adjudicate(
            bull_arg,
            bear_arg,
            transcript,
            ticker,
            initial_direction,
            initial_confidence,
        )

    def _run_researcher(
        self, role: str, transcript: str, ticker: str, direction: Direction
    ) -> str:
        stance = "FOR" if role == "bull" else "AGAINST"
        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            system=self._build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"You are a {role}ish researcher. Make the strongest 2-sentence case "
                    f"{stance} a {direction.value} trade on {ticker}.\n\n"
                    f"Agent signals:\n{transcript}\n\nBe specific, not generic.",
                }
            ],
        )
        return resp.content[0].text.strip()

    def _adjudicate(
        self,
        bull_arg: str,
        bear_arg: str,
        transcript: str,
        ticker: str,
        initial_direction: Direction,
        initial_confidence: float,
    ) -> dict:
        prompt = (
            f"You are the senior portfolio manager making the final trading decision for {ticker}.\n\n"
            f"Initial consensus: {initial_direction.value} (confidence: {initial_confidence:.2f})\n\n"
            f"Bull argument:\n{bull_arg}\n\n"
            f"Bear argument:\n{bear_arg}\n\n"
            f"Agent votes:\n{transcript}\n\n"
            "Respond in EXACTLY this format (no extra text):\n"
            "DECISION: LONG|SHORT|HOLD\n"
            "CONFIDENCE: 0.50-0.95\n"
            "REASON: one sentence"
        )
        resp = self.client.messages.create(
            model="claude-opus-4-7",
            max_tokens=80,
            system=self._build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        direction, confidence = self._parse_adjudication(
            text, initial_direction, initial_confidence
        )
        full_transcript = (
            f"{transcript}\n\n"
            f"[Bull] {bull_arg}\n"
            f"[Bear] {bear_arg}\n"
            f"[Adjudicator] {text}"
        )
        return {
            "direction": direction,
            "confidence": confidence,
            "transcript": full_transcript,
        }

    @staticmethod
    def _parse_adjudication(
        text: str, fallback_dir: Direction, fallback_conf: float
    ) -> tuple[Direction, float]:
        dir_match = re.search(r"DECISION:\s*(LONG|SHORT|HOLD)", text, re.IGNORECASE)
        conf_match = re.search(r"CONFIDENCE:\s*(0\.\d+)", text)
        direction = Direction(dir_match.group(1).upper()) if dir_match else fallback_dir
        confidence = float(conf_match.group(1)) if conf_match else fallback_conf
        return direction, round(min(max(confidence, 0.50), 0.95), 3)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_transcript(self, votes: list[AgentVote]) -> str:
        return "\n".join(
            f"{v.agent_name}: {v.direction.value} (conf={v.confidence:.2f}) — {v.reasoning}"
            for v in votes
        )

    def _build_system_prompt(self) -> str:
        base = "You are a crypto trading analyst."
        if self.learnings:
            return f"{base}\n\nRecent learnings from signal outcomes:\n{self.learnings}"
        return base
