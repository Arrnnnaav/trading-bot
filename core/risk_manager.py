import numpy as np
from core.models import HarnessState, Signal, Direction, Market, AgentVote
from core.pattern_reputation import PatternReputationTracker
from config import config


class RiskManager:
    def __init__(
        self, market: Market, pattern_reputation: PatternReputationTracker | None = None
    ):
        self.market = market
        self.pattern_reputation = pattern_reputation

    def _calc_atr(self, klines: list[dict], period: int = 14) -> float:
        if len(klines) < period:
            return abs(klines[-1]["high"] - klines[-1]["low"])
        trs = []
        for i in range(1, len(klines)):
            tr = max(
                klines[i]["high"] - klines[i]["low"],
                abs(klines[i]["high"] - klines[i - 1]["close"]),
                abs(klines[i]["low"] - klines[i - 1]["close"]),
            )
            trs.append(tr)
        return float(np.mean(trs[-period:]))

    def size_position(
        self,
        state: HarnessState,
        entry: float | None = None,
        stop: float | None = None,
    ) -> float:
        max_notional = state.portfolio_value_inr * config.max_portfolio_pct_per_trade
        if entry is None or stop is None:
            return round(max_notional, 2)

        stop_distance_pct = abs(entry - stop) / entry if entry > 0 else 0.0
        if stop_distance_pct <= 0:
            return 0.0

        risk_budget = state.portfolio_value_inr * config.max_risk_pct_per_trade
        risk_sized_notional = risk_budget / stop_distance_pct
        return round(min(max_notional, risk_sized_notional), 2)

    def calc_levels(
        self, entry: float, direction: Direction, atr: float
    ) -> tuple[float, float]:
        atr_mult = 1.5
        if direction == Direction.LONG:
            stop = round(entry - atr_mult * atr, 2)
            target = round(entry + config.min_rr_ratio * atr_mult * atr, 2)
        else:
            stop = round(entry + atr_mult * atr, 2)
            target = round(entry - config.min_rr_ratio * atr_mult * atr, 2)
        return target, stop

    def approve(
        self,
        state: HarnessState,
        ticker: str,
        direction: Direction,
        confidence: float,
        klines: list[dict],
        votes: list[AgentVote] | None = None,
    ) -> tuple[bool, str]:
        if len(state.open_positions) >= config.max_open_positions_per_market:
            return (
                False,
                f"Max {config.max_open_positions_per_market} open positions reached",
            )

        if state.portfolio_value_inr > 0:
            daily_loss_pct = (
                abs(min(state.daily_realized_pnl_inr, 0.0)) / state.portfolio_value_inr
            )
            weekly_loss_pct = (
                abs(min(state.weekly_realized_pnl_inr, 0.0)) / state.portfolio_value_inr
            )
            if daily_loss_pct >= config.max_daily_loss_pct:
                return False, f"Daily loss limit reached ({daily_loss_pct:.2%})"
            if weekly_loss_pct >= config.max_weekly_loss_pct:
                return False, f"Weekly loss limit reached ({weekly_loss_pct:.2%})"

        if state.consecutive_losses >= config.max_consecutive_losses:
            return False, f"Consecutive loss limit reached ({state.consecutive_losses})"

        existing = [p for p in state.open_positions if p.ticker == ticker]
        if existing:
            return False, f"Already have open position in {ticker}"

        if confidence < config.min_confidence:
            return (
                False,
                f"Confidence {confidence:.2f} below threshold {config.min_confidence}",
            )

        if direction == Direction.HOLD:
            return False, "Direction is HOLD"

        if self.pattern_reputation and votes:
            votes_as_dicts = [
                {
                    "agent_name": v.agent_name,
                    "direction": v.direction.value,
                    "confidence": v.confidence,
                }
                for v in votes
            ]
            allowed, reason = self.pattern_reputation.is_pattern_allowed(
                direction.value, ticker, votes_as_dicts
            )
            if not allowed:
                return False, reason

        entry = klines[-1]["close"]
        atr = self._calc_atr(klines)
        target, stop = self.calc_levels(entry, direction, atr)
        reward = abs(target - entry)
        risk = abs(entry - stop)
        rr = reward / risk if risk > 0 else 0.0

        if rr < config.min_rr_ratio:
            return False, f"R:R {rr:.2f} below minimum {config.min_rr_ratio}"

        if self.size_position(state, entry, stop) <= 0:
            return False, "Position size is zero"

        return True, "approved"

    def build_signal(
        self,
        signal_id: str,
        ticker: str,
        direction: Direction,
        confidence: float,
        klines: list[dict],
        state: HarnessState,
        votes: list[AgentVote],
        transcript: str,
    ) -> Signal:
        from datetime import datetime, timezone

        entry = klines[-1]["close"]
        atr = self._calc_atr(klines)
        target, stop = self.calc_levels(entry, direction, atr)
        size = self.size_position(state, entry, stop)

        return Signal(
            id=signal_id,
            market=self.market,
            ticker=ticker,
            direction=direction,
            entry_price=entry,
            target_price=target,
            stop_price=stop,
            confidence=confidence,
            position_size_inr=size,
            generated_at=datetime.now(timezone.utc).isoformat(),
            agent_votes=votes,
            debate_transcript=transcript,
        )
