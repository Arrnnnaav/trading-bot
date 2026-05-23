import numpy as np
from core.models import HarnessState, Signal, Direction, Market, AgentVote
from config import config


class RiskManager:
    def __init__(self, market: Market):
        self.market = market

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

    def size_position(self, state: HarnessState) -> float:
        return round(state.portfolio_value_inr * config.max_portfolio_pct_per_trade, 2)

    def calc_levels(
        self, entry: float, direction: Direction, atr: float
    ) -> tuple[float, float]:
        atr_mult = 2.0 if self.market == Market.CRYPTO else 1.5
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
    ) -> tuple[bool, str]:
        if len(state.open_positions) >= config.max_open_positions_per_market:
            return (
                False,
                f"Max {config.max_open_positions_per_market} open positions reached",
            )

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

        entry = klines[-1]["close"]
        atr = self._calc_atr(klines)
        target, stop = self.calc_levels(entry, direction, atr)
        reward = abs(target - entry)
        risk = abs(entry - stop)
        rr = reward / risk if risk > 0 else 0.0

        if rr < config.min_rr_ratio:
            return False, f"R:R {rr:.2f} below minimum {config.min_rr_ratio}"

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
        size = self.size_position(state)

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
