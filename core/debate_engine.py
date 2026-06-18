"""
DebateEngine — multi-agent vote consensus + 3-round adversarial bull/bear debate.

Flow:
  1. reach_consensus(votes, ticker) — vote counting + threshold check
  2. If direction != HOLD: _adversarial_debate() runs
     a. Bull R1 (haiku)   — opening case FOR the trade
     b. Bear R2 (haiku)   — sees Bull R1, specifically attacks its weakest points
     c. Bull R3 (haiku)   — sees Bear R2, rebuts strongest criticism
     d. Opus adjudicator  — sees all 3 rounds → LONG/SHORT/HOLD + confidence
  3. generate_reasoning() — Telegram-facing 2-sentence rationale (haiku)
"""

import re
from core.llm_client import ClaudeCodeClient
from core.models import AgentVote, Direction
from config import config


class DebateEngine:
    def __init__(self, learnings: str = ""):
        self._client = ClaudeCodeClient()
        self.learnings = learnings

    def update_learnings(self, learnings: str):
        self.learnings = learnings

    # ── Public API ────────────────────────────────────────────────────────────

    async def reach_consensus(
        self,
        votes: list[AgentVote],
        ticker: str = "",
        agent_weights: dict[str, float] | None = None,
    ) -> dict:
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

        if agent_weights:
            weights = [agent_weights.get(v.agent_name, 1.0) for v in agreeing]
            confidence = sum(w * v.confidence for w, v in zip(weights, agreeing)) / sum(
                weights
            )
        else:
            confidence = sum(v.confidence for v in agreeing) / len(agreeing)
        if confidence < config.min_confidence:
            return {
                "direction": Direction.HOLD,
                "confidence": confidence,
                "transcript": self._build_transcript(votes),
            }

        transcript = self._build_transcript(votes)
        if not config.enable_llm_adjudication:
            return {
                "direction": direction,
                "confidence": round(confidence, 3),
                "transcript": transcript,
            }

        return await self._adversarial_debate(
            votes, ticker, direction, round(confidence, 3)
        )

    async def generate_reasoning(
        self, votes: list[AgentVote], direction: Direction, ticker: str
    ) -> str:
        transcript = self._build_transcript(votes)
        prompt = (
            f"Agent debate for {ticker}:\n{transcript}\n\nConsensus: {direction.value}\n"
            "Write a 2-sentence trading rationale. Be specific about key signals."
        )
        result = await self._client.call(
            prompt=prompt,
            system=self._build_system_prompt(),
            model="claude-haiku-4-5-20251001",
        )
        return result or "Signal generated from multi-agent consensus."

    # ── Adversarial debate ────────────────────────────────────────────────────

    async def _adversarial_debate(
        self,
        votes: list[AgentVote],
        ticker: str,
        initial_direction: Direction,
        initial_confidence: float,
    ) -> dict:
        transcript = self._build_transcript(votes)

        # Round 1: Bull opens
        bull_r1 = await self._run_researcher(
            role="bull",
            transcript=transcript,
            ticker=ticker,
            direction=initial_direction,
            prior_arguments="",
        )

        # Round 2: Bear specifically attacks Bull R1
        bear_r2 = await self._run_researcher(
            role="bear",
            transcript=transcript,
            ticker=ticker,
            direction=initial_direction,
            prior_arguments=f"Bull's opening argument:\n{bull_r1}",
        )

        # Round 3: Bull rebuts Bear R2
        bull_r3 = await self._run_researcher(
            role="bull_rebuttal",
            transcript=transcript,
            ticker=ticker,
            direction=initial_direction,
            prior_arguments=(
                f"Bull's opening argument:\n{bull_r1}\n\nBear's attack:\n{bear_r2}"
            ),
        )

        return await self._adjudicate(
            bull_r1=bull_r1,
            bear_r2=bear_r2,
            bull_r3=bull_r3,
            transcript=transcript,
            ticker=ticker,
            initial_direction=initial_direction,
            initial_confidence=initial_confidence,
        )

    async def _run_researcher(
        self,
        role: str,
        transcript: str,
        ticker: str,
        direction: Direction,
        prior_arguments: str = "",
    ) -> str:
        if role == "bull":
            instruction = (
                f"You are a bullish researcher. Make the strongest 3-sentence opening case "
                f"FOR a {direction.value} trade on {ticker}. Use the specific agent signals below. "
                "Be concrete — cite exact indicators, levels, or flows."
            )
        elif role == "bear":
            instruction = (
                "You are a bearish researcher. You have read the bull's opening argument below. "
                "Specifically attack its weakest points in 3 sentences. "
                "Do not make generic bearish statements — respond to what bull actually said."
            )
        else:  # bull_rebuttal
            instruction = (
                "You are the bull researcher responding to the bear's attack. "
                "In 2 sentences: address the bear's strongest criticism directly, "
                "then reinforce the most compelling part of your original case."
            )

        context_block = (
            f"\n\nPrior arguments:\n{prior_arguments}" if prior_arguments else ""
        )
        prompt = (
            f"{instruction}\n\nAgent signals for {ticker}:\n{transcript}{context_block}"
        )
        return (
            await self._client.call(
                prompt=prompt,
                system=self._build_system_prompt(),
                model="claude-haiku-4-5-20251001",
            )
            or ""
        )

    async def _adjudicate(
        self,
        bull_r1: str,
        bear_r2: str,
        bull_r3: str,
        transcript: str,
        ticker: str,
        initial_direction: Direction,
        initial_confidence: float,
    ) -> dict:
        prompt = (
            f"You are the senior portfolio manager making the final trading decision for {ticker}.\n\n"
            f"Initial consensus: {initial_direction.value} (confidence: {initial_confidence:.2f})\n\n"
            f"=== 3-Round Debate ===\n\n"
            f"[Bull R1 — Opening]\n{bull_r1}\n\n"
            f"[Bear R2 — Attack]\n{bear_r2}\n\n"
            f"[Bull R3 — Rebuttal]\n{bull_r3}\n\n"
            f"=== Agent Votes ===\n{transcript}\n\n"
            "Assess: did the bear expose fatal flaws, or did the bull defend successfully?\n"
            "Respond in EXACTLY this format (no extra text):\n"
            "DECISION: LONG|SHORT|HOLD\n"
            "CONFIDENCE: 0.50-0.95\n"
            "REASON: one sentence"
        )
        text = await self._client.call(
            prompt=prompt,
            system=self._build_system_prompt(),
            model="claude-opus-4-7",
        )
        direction, confidence = self._parse_adjudication(
            text, initial_direction, initial_confidence
        )
        full_transcript = (
            f"{transcript}\n\n"
            f"[Bull R1] {bull_r1}\n"
            f"[Bear R2] {bear_r2}\n"
            f"[Bull R3] {bull_r3}\n"
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
        base = "You are an India index options trading analyst."
        if self.learnings:
            return f"{base}\n\nRecent learnings from signal outcomes:\n{self.learnings}"
        return base
