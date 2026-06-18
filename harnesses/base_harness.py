import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pydantic import ValidationError
from core.llm_client import ClaudeCodeClient
from core.json_store import read_json, write_json_atomic
from core.models import HarnessState, Direction
from core.debate_engine import DebateEngine
from core.risk_manager import RiskManager
from core.signal_aggregator import SignalAggregator
from core.agent_tracker import AgentPerformanceTracker
from core.pattern_reputation import PatternReputationTracker
from config import config

_LOG = logging.getLogger(__name__)


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
            return HarnessState(**read_json(self.state_path, {}))
        except FileNotFoundError:
            return HarnessState()
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            _LOG.warning("Invalid harness state at %s: %s", self.state_path, exc)
            return HarnessState()

    def _save_state(self):
        write_json_atomic(self.state_path, self.state.model_dump(mode="json"))

    def _make_signal_id(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"sig_{ts}_{uuid.uuid4().hex[:6]}"

    async def run_session(self, ticker: str, klines: list[dict], **agent_kwargs):
        self._agent_tracker.refresh()
        self._pattern_reputation.refresh()
        agent_weights = self._agent_tracker.get_weights()

        votes = []
        for agent in self.agents:
            try:
                votes.append(agent.analyze(ticker, klines, self.MARKET, **agent_kwargs))
            except Exception as exc:
                _LOG.warning("%s failed for %s: %s", agent.name, ticker, exc)

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

        try:
            learnings = await self._llm_client.call(
                prompt=summary_prompt,
                model="claude-haiku-4-5-20251001",
            )
        except Exception as exc:
            _LOG.warning("LLM learnings call failed: %s", exc)
            return
        if not learnings:
            return

        self.state.agent_learnings = learnings
        self._save_state()
        self.debate_engine.update_learnings(learnings)

        log_path = Path("data") / f"{self.MARKET.value}_learnings.md"
        log_path.parent.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        tmp = log_path.with_suffix(".tmp")
        tmp.write_text(existing + f"\n## {ts}\n{learnings}\n", encoding="utf-8")
        tmp.replace(log_path)
