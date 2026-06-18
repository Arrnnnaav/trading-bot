import asyncio
import logging
from datetime import datetime, timezone, timedelta

from config import config
from core.json_store import write_json_atomic
from core.models import Position, Market, Direction, SignalOutcome

_LOG = logging.getLogger(__name__)


class StopMonitor:
    def __init__(
        self,
        india_broker,
        harness_states: dict,
        telegram_bot,
        signal_aggregator=None,
    ):
        self.brokers = {
            Market.INDIA: india_broker,
        }
        self.harness_states = (
            harness_states  # market -> HarnessState (shared reference)
        )
        self.telegram_bot = telegram_bot
        self.signal_aggregator = signal_aggregator

    @staticmethod
    def _state_path(market: Market) -> str:
        return config.india_progress_path

    def _save_state(self, market: Market) -> None:
        state = self.harness_states.get(market)
        if state is None:
            return
        write_json_atomic(self._state_path(market), state.model_dump(mode="json"))

    def _should_close(self, position: Position, current_price: float) -> bool:
        return self._close_reason(position, current_price) is not None

    def _close_reason(
        self, position: Position, current_price: float
    ) -> SignalOutcome | None:
        if position.direction == Direction.LONG:
            if current_price <= position.stop_price:
                return SignalOutcome.STOP_HIT
            if current_price >= position.target_price:
                return SignalOutcome.TARGET_HIT
        else:
            if current_price >= position.stop_price:
                return SignalOutcome.STOP_HIT
            if current_price <= position.target_price:
                return SignalOutcome.TARGET_HIT

        if self._is_expired(position):
            return SignalOutcome.EXPIRED
        return None

    def _is_expired(self, position: Position) -> bool:
        try:
            if position.expires_at:
                expires_at = datetime.fromisoformat(position.expires_at)
            else:
                opened_at = datetime.fromisoformat(position.opened_at)
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
                expires_at = opened_at + timedelta(days=config.india_signal_expiry_days)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) >= expires_at
        except Exception as exc:
            _LOG.warning("Failed to parse expiry for %s: %s", position.ticker, exc)
            return False

    @staticmethod
    def _calc_pnl(position: Position, current_price: float) -> tuple[float, float]:
        pnl = (current_price - position.entry_price) / position.entry_price
        if position.direction == Direction.SHORT:
            pnl = -pnl
        return pnl, position.size_inr * pnl

    async def _check_position(self, position: Position, market: Market):
        broker = self.brokers[market]
        try:
            current_price = broker.get_price(position.ticker)
        except Exception as exc:
            _LOG.warning("Failed to fetch price for %s: %s", position.ticker, exc)
            return

        reason = self._close_reason(position, current_price)
        if reason is None:
            return

        close_side = "sell" if position.direction == Direction.LONG else "buy"
        quantity = position.quantity or round(position.size_inr / current_price, 6)
        close_ticker = position.instrument_token or position.ticker.replace("/", "")

        # Step 1: broker close — if this fails, leave position in state and abort.
        try:
            broker.close_position(
                position.broker_order_id, close_ticker, close_side, quantity
            )
        except Exception as exc:
            _LOG.error(
                "Broker close failed for %s — position left open: %s",
                position.ticker,
                exc,
            )
            await self.telegram_bot.send_error(
                f"Stop close failed for {position.ticker}: {exc}"
            )
            return

        # Step 2: broker succeeded — update in-memory state and persist.
        pnl, pnl_inr = self._calc_pnl(position, current_price)
        state = self.harness_states.get(market)
        if state:
            state.open_positions = [
                p
                for p in state.open_positions
                if p.broker_order_id != position.broker_order_id
            ]
            state.portfolio_value_inr = max(0.0, state.portfolio_value_inr + pnl_inr)
            state.daily_realized_pnl_inr += pnl_inr
            state.weekly_realized_pnl_inr += pnl_inr
            state.consecutive_losses = (
                state.consecutive_losses + 1 if pnl_inr < 0 else 0
            )
            try:
                self._save_state(market)
            except Exception as exc:
                _LOG.error(
                    "State save failed after closing %s — position removed from "
                    "memory but disk may be stale: %s",
                    position.ticker,
                    exc,
                )

        # Step 3: update signal outcome.
        if self.signal_aggregator is not None:
            try:
                self.signal_aggregator.update_signal_outcome(
                    position.signal_id, reason.value, current_price, pnl, pnl_inr
                )
            except Exception as exc:
                _LOG.warning(
                    "Failed to update outcome for signal %s: %s",
                    position.signal_id,
                    exc,
                )

        # Step 4: notify Telegram.
        try:
            if hasattr(self.telegram_bot, "send_position_closed"):
                await self.telegram_bot.send_position_closed(
                    ticker=position.ticker,
                    close_price=current_price,
                    pnl_pct=pnl,
                    pnl_inr=pnl_inr,
                    market=market,
                    reason=reason,
                )
            else:
                await self.telegram_bot.send_stop_hit(
                    ticker=position.ticker,
                    close_price=current_price,
                    pnl_pct=pnl,
                    pnl_inr=pnl_inr,
                    market=market,
                )
        except Exception as exc:
            _LOG.warning("Telegram notify failed for %s: %s", position.ticker, exc)

    async def run_once(self):
        for market, state in self.harness_states.items():
            for position in list(state.open_positions):
                await self._check_position(position, market)

    async def run_forever(self, interval_seconds: int = 300):
        while True:
            try:
                await self.run_once()
            except Exception:
                _LOG.exception("Stop monitor iteration failed")
            await asyncio.sleep(interval_seconds)
