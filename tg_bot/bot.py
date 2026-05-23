from __future__ import annotations
import importlib

# Avoid naming conflict: load telegram library explicitly before local telegram/ package is imported
_telegram_lib = importlib.import_module("telegram")
Update = _telegram_lib.Update
InlineKeyboardButton = _telegram_lib.InlineKeyboardButton
InlineKeyboardMarkup = _telegram_lib.InlineKeyboardMarkup

_telegram_ext = importlib.import_module("telegram.ext")
Application = _telegram_ext.Application
CallbackQueryHandler = _telegram_ext.CallbackQueryHandler
ContextTypes = _telegram_ext.ContextTypes

from core.models import Signal, Market, Direction
from config import config


class TelegramBot:
    def __init__(
        self, harness_states: dict, coindcx_broker, upstox_broker, signal_aggregator
    ):
        self.harness_states = harness_states
        self.brokers = {Market.CRYPTO: coindcx_broker, Market.INDIA: upstox_broker}
        self.aggregator = signal_aggregator
        self.app = Application.builder().token(config.telegram_token).build()
        self.app.add_handler(CallbackQueryHandler(self._handle_callback))

    def _chat_id(self, market: Market) -> str:
        return (
            config.telegram_crypto_chat_id
            if market == Market.CRYPTO
            else config.telegram_india_chat_id
        )

    def _format_signal(self, signal: Signal) -> str:
        emoji = "🟢" if signal.direction == Direction.LONG else "🔴"
        action = "LONG" if signal.direction == Direction.LONG else "SHORT"
        state = self.harness_states.get(signal.market)
        portfolio_pct = (
            (signal.position_size_inr / state.portfolio_value_inr * 100)
            if state
            else 2.0
        )
        return (
            f"{emoji} {action} {signal.ticker}\n"
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
        emoji = "🟢" if pnl_inr >= 0 else "🔴"
        text = (
            f"🛑 Stop hit: {ticker}\n"
            f"Closed at ₹{close_price:,.1f}\n"
            f"{emoji} P&L: {pnl_pct * 100:+.1f}%  (₹{pnl_inr:+,.0f})"
        )
        await self.app.bot.send_message(chat_id=self._chat_id(market), text=text)

    async def send_error(self, message: str):
        for chat_id in [config.telegram_crypto_chat_id, config.telegram_india_chat_id]:
            if chat_id:
                await self.app.bot.send_message(chat_id=chat_id, text=f"⚠️ {message}")

    async def _handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        query = update.callback_query
        await query.answer()
        action, signal_id = query.data.split(":", 1)

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
        market = Market(entry["market"])
        broker = self.brokers[market]
        ticker = entry["ticker"]
        direction = entry["direction"]
        size_inr = entry["position_size_inr"]

        try:
            if market == Market.INDIA:
                result = broker.place_options_order(
                    index="NIFTY"
                    if "NIFTY" in ticker.upper() and "BANK" not in ticker.upper()
                    else "BANKNIFTY",
                    direction=direction,
                    expiry="2026-05-29",
                    size_inr=size_inr,
                )
            else:
                side = "buy" if direction == "LONG" else "sell"
                result = broker.place_order(ticker.replace("/", ""), side, size_inr)

            fill_price = result.get("fill_price", entry["entry_price"])
            await query.edit_message_text(
                query.message.text + f"\n\n✅ Filled at ₹{fill_price:,.1f}"
            )

            # Mark signal as executed in log
            signals = self.aggregator.get_all_signals()
            for s in signals:
                if s["id"] == entry["id"]:
                    s["executed"] = True
                    break
            self.aggregator._save_log(signals)

        except Exception as e:
            await query.edit_message_text(
                query.message.text + f"\n\n⚠️ Order failed: {e}\nPlace manually."
            )

    def run_polling(self):
        self.app.run_polling()
