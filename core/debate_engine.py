from anthropic import Anthropic
from core.models import AgentVote, Direction
from config import config


class DebateEngine:
    def __init__(self, learnings: str = ""):
        self.client = Anthropic(api_key=config.anthropic_api_key)
        self.learnings = learnings

    def update_learnings(self, learnings: str):
        self.learnings = learnings

    def reach_consensus(self, votes: list[AgentVote]) -> dict:
        if not votes:
            return {
                "direction": Direction.HOLD,
                "confidence": 0.0,
                "transcript": "No votes",
            }

        active_votes = [
            v for v in votes if v.direction != Direction.HOLD and v.confidence >= 0.50
        ]

        long_votes = [v for v in active_votes if v.direction == Direction.LONG]
        short_votes = [v for v in active_votes if v.direction == Direction.SHORT]

        if len(long_votes) >= 2 and len(long_votes) > len(short_votes):
            direction = Direction.LONG
            agreeing = long_votes
        elif len(short_votes) >= 2 and len(short_votes) > len(long_votes):
            direction = Direction.SHORT
            agreeing = short_votes
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

        return {
            "direction": direction,
            "confidence": round(confidence, 3),
            "transcript": self._build_transcript(votes),
        }

    def _build_transcript(self, votes: list[AgentVote]) -> str:
        lines = []
        for v in votes:
            lines.append(
                f"{v.agent_name}: {v.direction.value} (conf={v.confidence:.2f}) — {v.reasoning}"
            )
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        base = "You are a crypto trading analyst."
        if self.learnings:
            return f"{base}\n\nRecent learnings from signal outcomes:\n{self.learnings}"
        return base

    def generate_reasoning(
        self, votes: list[AgentVote], direction: Direction, ticker: str
    ) -> str:
        transcript = self._build_transcript(votes)
        prompt = f"""Agent debate for {ticker}:
{transcript}

Consensus: {direction.value}
Write a 2-sentence trading rationale based on the above debate. Be specific about the key signals."""

        resp = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            system=self._build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
