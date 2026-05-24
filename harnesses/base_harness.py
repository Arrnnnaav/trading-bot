import json
import uuid
from datetime import datetime, timezone
from anthropic import Anthropic
from core.models import HarnessState, Direction
from core.debate_engine import DebateEngine
from core.risk_manager import RiskManager
from core.signal_aggregator import SignalAggregator
from config import config


class BaseHarness:
    MARKET = None  # set by subclass

    def __init__(
        self,
        state_path: str,
        broker,
        agents: list,
        telegram_bot,
        signal_aggregator: SignalAggregator,
    ):
        self.state_path = state_path
        self.broker = broker
        self.agents = agents
        self.telegram_bot = telegram_bot
        self.aggregator = signal_aggregator
        self.state = self._load_state()
        self.debate_engine = DebateEngine(learnings=self.state.agent_learnings)
        self.risk_manager = RiskManager(self.MARKET)

    def _load_state(self) -> HarnessState:
        try:
            with open(self.state_path) as f:
                return HarnessState(**json.load(f))
        except (FileNotFoundError, Exception):
            return HarnessState()

    def _save_state(self):
        with open(self.state_path, "w") as f:
            f.write(self.state.model_dump_json(indent=2))

    def _make_signal_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"sig_{ts}_{uuid.uuid4().hex[:6]}"

    async def run_session(self, ticker: str, klines: list[dict], **agent_kwargs):
        votes = [
            agent.analyze(ticker, klines, self.MARKET, **agent_kwargs)
            for agent in self.agents
        ]

        consensus = self.debate_engine.reach_consensus(votes, ticker=ticker)
        direction = consensus["direction"]
        confidence = consensus["confidence"]

        if direction == Direction.HOLD:
            return None

        approved, reason = self.risk_manager.approve(
            self.state, ticker, direction, confidence, klines
        )
        if not approved:
            return None

        signal_id = self._make_signal_id()
        reasoning = self.debate_engine.generate_reasoning(votes, direction, ticker)
        signal = self.risk_manager.build_signal(
            signal_id,
            ticker,
            direction,
            confidence,
            klines,
            self.state,
            votes,
            consensus["transcript"],
        )

        if self.aggregator.is_duplicate(signal):
            return None

        self.aggregator.log_signal(signal)
        self.state.signal_history.append(signal_id)
        self.state.last_run = datetime.now(timezone.utc).isoformat()
        self.state.session_count += 1
        self._save_state()

        await self.telegram_bot.send_signal(signal)
        return signal

    def update_learnings(self):
        if self.state.session_count % 10 != 0:
            return
        recent_ids = self.state.signal_history[-50:]
        all_signals = {s["id"]: s for s in self.aggregator.get_all_signals()}
        outcomes = [all_signals[id] for id in recent_ids if id in all_signals]
        if not outcomes:
            return

        client = Anthropic(api_key=config.anthropic_api_key)
        summary_prompt = f"""Analyze these {len(outcomes)} trading signal outcomes:
{json.dumps([{"ticker": o["ticker"], "direction": o["direction"], "outcome": o["outcome"], "pnl_pct": o.get("hypothetical_pnl_pct")} for o in outcomes], indent=2)}

Write 3 concise sentences about: (1) which setups worked, (2) which failed, (3) one rule to apply next session."""

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": summary_prompt}],
        )
        learnings = resp.content[0].text.strip()
        self.state.agent_learnings = learnings
        self._save_state()

        # Route learnings to DebateEngine (where LLM actually runs)
        self.debate_engine.update_learnings(learnings)

        # Write human-readable log
        from pathlib import Path

        log_path = Path("data") / f"{self.MARKET.value}_learnings.md"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            ts = datetime.now(timezone.utc).isoformat()
            f.write(f"\n## {ts}\n{learnings}\n")
