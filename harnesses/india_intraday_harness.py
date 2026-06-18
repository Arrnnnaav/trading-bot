from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from agents.fundamentals import FundamentalsAgent
from agents.fii_dii import FIIDIIAgent
from agents.options_oi import OptionsOIAgent
from brokers.upstox import UpstoxBroker
from brokers.paper_broker import PaperBroker
from config import config
from core.models import Market
from core.signal_aggregator import SignalAggregator
from harnesses.base_harness import BaseHarness

IST = pytz.timezone("Asia/Kolkata")

TICKERS = ["NIFTY", "BANKNIFTY", "SENSEX", "NIFTYIT"]

# Instrument tokens for live price + OHLCV fetching
TICKER_TOKENS = {
    "NIFTY": "NSE_INDEX|Nifty 50",
    "BANKNIFTY": "NSE_INDEX|Nifty Bank",
    "SENSEX": "BSE_INDEX|SENSEX",
    "NIFTYIT": "NSE_INDEX|Nifty IT",
}


class IndiaIntradayHarness(BaseHarness):
    MARKET = Market.INDIA

    def __init__(self, telegram_bot, signal_aggregator: SignalAggregator):
        _broker = UpstoxBroker(
            api_key=config.upstox_api_key,
            access_token=config.upstox_access_token,
        )
        broker = (
            PaperBroker(_broker, config.paper_trades_path)
            if config.paper_trading
            else _broker
        )
        # Phase 1 placeholder agents — replaced in Phase 2
        agents = [
            FundamentalsAgent(),
            FIIDIIAgent(),
            OptionsOIAgent(broker),
        ]
        super().__init__(
            state_path=config.india_progress_path,
            broker=broker,
            agents=agents,
            telegram_bot=telegram_bot,
            signal_aggregator=signal_aggregator,
        )
        self.scheduler = AsyncIOScheduler(timezone=IST)
        # 9:20, 10:15, 11:30, 13:00, 14:00 IST weekdays
        for hour, minute in [(9, 20), (10, 15), (11, 30), (13, 0), (14, 0)]:
            self.scheduler.add_job(
                self._run_all_tickers,
                CronTrigger(
                    day_of_week="mon-fri", hour=hour, minute=minute, timezone=IST
                ),
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
                klines = self.broker.get_ohlcv_intraday(
                    token, interval="15minute", days_back=5
                )
                if not klines:
                    continue
                await self.run_session(ticker, klines, expiry=expiry)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Intraday session failed for %s: %s", ticker, exc
                )

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
