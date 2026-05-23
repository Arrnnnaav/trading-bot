from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from harnesses.base_harness import BaseHarness
from core.models import Market
from agents.kronos_technical import KronosTechnicalAgent
from agents.fundamentals import FundamentalsAgent
from agents.fii_dii import FIIDIIAgent
from agents.options_oi import OptionsOIAgent
from brokers.upstox import UpstoxBroker
from config import config

IST = pytz.timezone("Asia/Kolkata")

INDIA_TICKERS = [
    "NSE_INDEX|Nifty 50",
    "NSE_INDEX|Nifty Bank",
    "NSE_EQ|INE040A01034",  # HDFC Bank
    "NSE_EQ|INE009A01021",  # Infosys
    "NSE_EQ|INE030A01027",  # Reliance
]

NEXT_WEEKLY_EXPIRY = "2026-05-29"  # update weekly or fetch dynamically


class IndiaHarness(BaseHarness):
    MARKET = Market.INDIA

    def __init__(self, telegram_bot, signal_aggregator):
        broker = UpstoxBroker(config.upstox_api_key, config.upstox_access_token)
        agents = [
            KronosTechnicalAgent(),
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

    async def _run_all_tickers(self):
        for token in INDIA_TICKERS:
            try:
                klines = self.broker.get_ohlcv(token, interval="1d", limit=60)
                ticker_display = token.split("|")[-1]
                await self.run_session(
                    ticker_display, klines, expiry=NEXT_WEEKLY_EXPIRY
                )
            except Exception:
                continue
        self.update_learnings()

    def start(self):
        # 09:00, 11:30, 14:45 IST on weekdays
        for hour, minute in [(9, 0), (11, 30), (14, 45)]:
            self.scheduler.add_job(
                self._run_all_tickers,
                CronTrigger(
                    day_of_week="mon-fri", hour=hour, minute=minute, timezone=IST
                ),
            )
        self.scheduler.start()
