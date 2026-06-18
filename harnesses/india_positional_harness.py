import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from agents.technical_india import TechnicalIndiaAgent
from agents.options_chain import OptionsChainAgent
from agents.macro_india import MacroIndiaAgent
from agents.news_analogue import NewsAnalogueAgent
from agents.fii_dii import FIIDIIAgent
from agents.kronos_india import KronosAgent
from brokers.paper_broker import PaperBroker
from config import config
from core.models import Market
from core.signal_aggregator import SignalAggregator
from harnesses.base_harness import BaseHarness
from harnesses.india_intraday_harness import _build_raw_broker

IST = pytz.timezone("Asia/Kolkata")

TICKERS = ["NIFTY", "BANKNIFTY", "SENSEX", "NIFTYIT"]

# Zerodha integer instrument tokens for daily OHLCV
TICKER_TOKENS = {
    "NIFTY": 256265,
    "BANKNIFTY": 260105,
    "SENSEX": 1,
    "NIFTYIT": 259849,
}


class IndiaPositionalHarness(BaseHarness):
    MARKET = Market.INDIA

    def __init__(self, telegram_bot, signal_aggregator: SignalAggregator):
        _broker = _build_raw_broker()
        broker = (
            PaperBroker(_broker, config.paper_trades_path)
            if config.paper_trading
            else _broker
        )
        agents = [
            TechnicalIndiaAgent(),
            OptionsChainAgent(broker),
            MacroIndiaAgent(),
            NewsAnalogueAgent(),
            FIIDIIAgent(),
            KronosAgent(),
        ]
        super().__init__(
            state_path="data/india_positional_progress.json",
            broker=broker,
            agents=agents,
            telegram_bot=telegram_bot,
            signal_aggregator=signal_aggregator,
        )
        self.scheduler = AsyncIOScheduler(timezone=IST)
        # 9:20 IST weekdays only
        self.scheduler.add_job(
            self._run_all_tickers,
            CronTrigger(day_of_week="mon-fri", hour=9, minute=20, timezone=IST),
        )

    async def _run_all_tickers(self):
        expiry = (
            self.broker.resolve_options_expiry()
            if hasattr(self.broker, "resolve_options_expiry")
            else None
        )
        for ticker in TICKERS:
            token = TICKER_TOKENS[ticker]
            try:
                klines = self.broker.get_ohlcv(token, interval="1d", limit=60)
                if not klines:
                    continue
                await self.run_session(ticker, klines, expiry=expiry)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Positional session failed for %s: %s", ticker, exc
                )

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
