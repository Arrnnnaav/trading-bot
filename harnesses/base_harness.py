import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from core.llm_client import ClaudeCodeClient
from core.models import HarnessState, Direction
from core.debate_engine import DebateEngine
from core.risk_manager import RiskManager
from core.signal_aggregator import SignalAggregator
from core.agent_tracker import AgentPerformanceTracker
from core.pattern_reputation import PatternReputationTracker
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
        self._agent_tracker = AgentPerformanceTracker(config.signal_log_path)
        self._pattern_reputation = PatternReputationTracker(config.signal_log_path)
        self._llm_client = ClaudeCodeClient()
        self.risk_manager = RiskManager(
            self.MARKET, pattern_reputation=self._pattern_reputation
        )

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
        self._agent_tracker.refresh()
        self._pattern_reputation.refresh()
        agent_weights = self._agent_tracker.get_weights()

        votes = [
            agent.analyze(ticker, klines, self.MARKET, **agent_kwargs)
            for agent in self.agents
        ]

        consensus = await self.debate_engine.reach_consensus(
            votes, ticker=ticker, agent_weights=agent_weights
        )
        direction = consensus["direction"]
        confidence = consensus["confidence"]

        if direction == Direction.HOLD:
            return None

        approved, reason = self.risk_manager.approve(
            self.state, ticker, direction, confidence, klines, votes=votes
        )
        if not approved:
            return None

        signal_id = self._make_signal_id()
        reasoning = await self.debate_engine.generate_reasoning(
            votes, direction, ticker
        )
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

    async def update_learnings(self):
        if self.state.session_count % 10 != 0:
            return
        recent_ids = self.state.signal_history[-50:]
        all_signals = {s["id"]: s for s in self.aggregator.get_all_signals()}
        outcomes = [all_signals[id] for id in recent_ids if id in all_signals]
        if not outcomes:
            return

        summary_prompt = (
            f"Analyze these {len(outcomes)} trading signal outcomes:\n"
            + json.dumps(
                [
                    {
                        "ticker": o["ticker"],
                        "direction": o["direction"],
                        "outcome": o["outcome"],
                        "pnl_pct": o.get("hypothetical_pnl_pct"),
                    }
                    for o in outcomes
                ],
                indent=2,
            )
            + "\n\nWrite 3 concise sentences about: (1) which setups worked, "
            "(2) which failed, (3) one rule to apply next session."
        )

        learnings = await self._llm_client.call(
            prompt=summary_prompt,
            model="claude-haiku-4-5-20251001",
        )
        if not learnings:
            return

        self.state.agent_learnings = learnings
        self._save_state()
        self.debate_engine.update_learnings(learnings)

        log_path = Path("data") / f"{self.MARKET.value}_learnings.md"
        log_path.parent.mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            ts = datetime.now(timezone.utc).isoformat()
            f.write(f"\n## {ts}\n{learnings}\n")
