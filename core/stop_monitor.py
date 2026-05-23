import asyncio
from core.models import Position, Market, Direction


class StopMonitor:
    def __init__(self, crypto_broker, india_broker, harness_states: dict, telegram_bot):
        self.brokers = {Market.CRYPTO: crypto_broker, Market.INDIA: india_broker}
        self.harness_states = (
            harness_states  # market -> HarnessState (shared reference)
        )
        self.telegram_bot = telegram_bot

    def _should_close(self, position: Position, current_price: float) -> bool:
        if position.direction == Direction.LONG:
            return current_price <= position.stop_price
        else:
            return current_price >= position.stop_price

    async def _check_position(self, position: Position, market: Market):
        broker = self.brokers[market]
        try:
            current_price = broker.get_price(position.ticker)
        except Exception:
            return

        if not self._should_close(position, current_price):
            return

        # Close the position
        try:
            close_side = "sell" if position.direction == Direction.LONG else "buy"
            broker.close_position(position.broker_order_id, position.ticker, close_side)

            # Remove from harness state
            state = self.harness_states.get(market)
            if state:
                state.open_positions = [
                    p
                    for p in state.open_positions
                    if p.broker_order_id != position.broker_order_id
                ]

            pnl = (current_price - position.entry_price) / position.entry_price
            if position.direction == Direction.SHORT:
                pnl = -pnl
            pnl_inr = position.size_inr * pnl

            await self.telegram_bot.send_stop_hit(
                ticker=position.ticker,
                close_price=current_price,
                pnl_pct=pnl,
                pnl_inr=pnl_inr,
                market=market,
            )
        except Exception as e:
            await self.telegram_bot.send_error(
                f"Stop close failed for {position.ticker}: {e}"
            )

    async def run_once(self):
        for market, state in self.harness_states.items():
            for position in list(state.open_positions):
                await self._check_position(position, market)

    async def run_forever(self, interval_seconds: int = 300):
        while True:
            await self.run_once()
            await asyncio.sleep(interval_seconds)
