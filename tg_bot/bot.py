from __future__ import annotations
import importlib
from datetime import datetime, timezone
from pathlib import Path

# Avoid naming conflict: load telegram library explicitly before local telegram/ package is imported
_telegram_lib = importlib.import_module("telegram")
Update = _telegram_lib.Update
InlineKeyboardButton = _telegram_lib.InlineKeyboardButton
InlineKeyboardMarkup = _telegram_lib.InlineKeyboardMarkup

_telegram_ext = importlib.import_module("telegram.ext")
Application = _telegram_ext.Application
CallbackQueryHandler = _telegram_ext.CallbackQueryHandler
ContextTypes = _telegram_ext.ContextTypes

from core.models import Signal, Market, Direction, Position, SignalOutcome
from core.json_store import write_json_atomic
from config import config

_TICKER_TO_INDEX = {
    "NIFTY": "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "SENSEX": "SENSEX",
    "NIFTYIT": "NIFTYIT",
}


class TelegramBot:
    def __init__(self, harness_states: dict, upstox_broker, signal_aggregator):
        self.harness_states = harness_states
        self.brokers = {
            Market.INDIA: upstox_broker,
        }
        self.aggregator = signal_aggregator
        self.app = Application.builder().token(config.telegram_token).build()
        self.app.add_handler(CallbackQueryHandler(self._handle_callback))

    def _chat_id(self, market: Market) -> str:
        return config.telegram_india_chat_id

    def _format_signal(self, signal: Signal) -> str:
        emoji = "🟢" if signal.direction == Direction.LONG else "🔴"
        action = "LONG" if signal.direction == Direction.LONG else "SHORT"
        paper_prefix = "[PAPER] " if config.paper_trading else ""
        state = self.harness_states.get(signal.market)
        portfolio_pct = (
            (signal.position_size_inr / state.portfolio_value_inr * 100)
            if state
            else 2.0
        )
        return (
            f"{paper_prefix}{emoji} {action} {signal.ticker}\n"
            f"{'━' * 20}\n"
            f"Entry:   ₹{signal.entry_price:,.1f}\n"
            f"Target:  ₹{signal.target_price:,.1f}  ({((signal.target_price - signal.entry_price) / signal.entry_price * 100):+.1f}%)\n"
            f"Stop:    ₹{signal.stop_price:,.1f}  ({((signal.stop_price - signal.entry_price) / signal.entry_price * 100):+.1f}%)\n"
            f"R:R      1 : {signal.rr_ratio:.1f}\n"
            f"Size:    ₹{signal.position_size_inr:,.0f}  ({portfolio_pct:.1f}% portfolio)\n"
            f"Confidence: {signal.confidence * 100:.0f}%"
        )

    async def send_signal(self, signal: Signal):
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(
                        "✅ EXECUTE", callback_data=f"exec:{signal.id}"
                    ),
                    InlineKeyboardButton("❌ SKIP", callback_data=f"skip:{signal.id}"),
                    InlineKeyboardButton(
                        "📊 DEBATE", callback_data=f"debate:{signal.id}"
                    ),
                ]
            ]
        )
        await self.app.bot.send_message(
            chat_id=self._chat_id(signal.market),
            text=self._format_signal(signal),
            reply_markup=keyboard,
        )

    async def send_stop_hit(
        self,
        ticker: str,
        close_price: float,
        pnl_pct: float,
        pnl_inr: float,
        market: Market,
    ):
        await self.send_position_closed(
            ticker=ticker,
            close_price=close_price,
            pnl_pct=pnl_pct,
            pnl_inr=pnl_inr,
            market=market,
            reason=SignalOutcome.STOP_HIT,
        )

    async def send_position_closed(
        self,
        ticker: str,
        close_price: float,
        pnl_pct: float,
        pnl_inr: float,
        market: Market,
        reason: SignalOutcome,
    ):
        emoji = "🟢" if pnl_inr >= 0 else "🔴"
        label = {
            SignalOutcome.TARGET_HIT: "Target hit",
            SignalOutcome.STOP_HIT: "Stop hit",
            SignalOutcome.EXPIRED: "Position expired",
        }.get(reason, "Position closed")
        text = (
            f"{label}: {ticker}\n"
            f"Closed at ₹{close_price:,.1f}\n"
            f"{emoji} P&L: {pnl_pct * 100:+.1f}%  (₹{pnl_inr:+,.0f})"
        )
        await self.app.bot.send_message(chat_id=self._chat_id(market), text=text)

    async def send_error(self, message: str):
        chat_id = config.telegram_india_chat_id
        if chat_id:
            await self.app.bot.send_message(chat_id=chat_id, text=f"⚠️ {message}")

    async def _handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        query = update.callback_query
        allowed_ids = config.telegram_allowed_user_ids
        if allowed_ids:
            user_id = str(update.effective_user.id) if update.effective_user else ""
            if user_id not in [uid.strip() for uid in allowed_ids.split(",")]:
                await query.answer("Unauthorized.", show_alert=True)
                return
        await query.answer()
        parts = query.data.split(":", 1)
        if len(parts) != 2:
            await query.edit_message_text("Invalid callback data.")
            return
        action, signal_id = parts

        all_signals = {s["id"]: s for s in self.aggregator.get_all_signals()}
        entry = all_signals.get(signal_id)
        if not entry:
            await query.edit_message_text("Signal not found.")
            return

        if action == "exec":
            await self._execute_signal(query, entry)
        elif action == "skip":
            await query.edit_message_text(query.message.text + "\n\n❌ Skipped")
        elif action == "debate":
            await query.message.reply_text(
                f"📊 Agent debate:\n\n{entry.get('debate_transcript', 'No transcript')[:4000]}"
            )

    async def _execute_signal(self, query, entry: dict):
        if entry.get("executed"):
            await query.edit_message_text(query.message.text + "\n\nAlready executed.")
            return

        market = Market(entry["market"])
        broker = self.brokers[market]
        ticker = entry["ticker"]
        direction = Direction(entry["direction"])
        size_inr = entry["position_size_inr"]

        try:
            if market == Market.INDIA:
                expiry = entry.get("expiry")
                if not expiry and hasattr(broker, "resolve_options_expiry"):
                    expiry = broker.resolve_options_expiry()
                index = _TICKER_TO_INDEX.get(ticker.upper(), ticker.upper())
                result = broker.place_options_order(
                    index=index,
                    direction=direction.value,
                    expiry=expiry,
                    size_inr=size_inr,
                )

            fill_price = result.get("fill_price", entry["entry_price"])
            quantity = result.get("quantity")
            await query.edit_message_text(
                query.message.text + f"\n\n✅ Filled at ₹{fill_price:,.1f}"
            )

            position = Position(
                signal_id=entry["id"],
                ticker=ticker,
                market=market,
                direction=direction,
                entry_price=float(fill_price),
                stop_price=float(entry["stop_price"]),
                target_price=float(entry["target_price"]),
                size_inr=float(size_inr),
                quantity=float(quantity) if quantity is not None else None,
                opened_at=datetime.now(timezone.utc).isoformat(),
                broker_order_id=result["order_id"],
                instrument_token=result.get("instrument_token"),
                option_type=result.get("option_type"),
                strike=result.get("strike"),
                expiry=result.get("expiry"),
                expires_at=entry.get("expires_at"),
            )
            state = self.harness_states.get(market)
            already_open = (
                any(p.signal_id == position.signal_id for p in state.open_positions)
                if state
                else False
            )
            if state and not already_open:
                state.open_positions.append(position)
                self._save_harness_state(market)

            self.aggregator.mark_signal_executed(entry["id"])

        except Exception as e:
            await query.edit_message_text(
                query.message.text + f"\n\n⚠️ Order failed: {e}\nPlace manually."
            )

    def _save_harness_state(self, market: Market) -> None:
        state = self.harness_states.get(market)
        if not state:
            return
        path = config.india_progress_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, state.model_dump(mode="json"))

    def run_polling(self):
        self.app.run_polling()
