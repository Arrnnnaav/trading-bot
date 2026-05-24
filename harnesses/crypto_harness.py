from apscheduler.schedulers.asyncio import AsyncIOScheduler
from harnesses.base_harness import BaseHarness
from core.models import Market
from agents.chronos_technical import ChronosTechnicalAgent
from agents.news_sentiment import NewsSentimentAgent
from agents.onchain import OnChainAgent
from brokers.coindcx import CoinDCXBroker
from config import config

CRYPTO_TICKERS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]


class CryptoHarness(BaseHarness):
    MARKET = Market.CRYPTO

    def __init__(self, telegram_bot, signal_aggregator):
        broker = CoinDCXBroker(config.coindcx_api_key, config.coindcx_api_secret)
        agents = [
            ChronosTechnicalAgent(),
            NewsSentimentAgent(config.cryptopanic_api_key),
            OnChainAgent(config.coinglass_api_key),
        ]
        super().__init__(
            state_path=config.crypto_progress_path,
            broker=broker,
            agents=agents,
            telegram_bot=telegram_bot,
            signal_aggregator=signal_aggregator,
        )
        self.scheduler = AsyncIOScheduler()

    async def _run_all_tickers(self):
        for ticker_raw in CRYPTO_TICKERS:
            try:
                klines = self.broker.get_ohlcv(ticker_raw, interval="15m", limit=60)
                ticker_display = f"{ticker_raw[:3]}/{ticker_raw[3:]}"
                await self.run_session(ticker_display, klines)
            except Exception:
                continue
        self.update_learnings()

    def start(self):
        self.scheduler.add_job(self._run_all_tickers, "interval", minutes=15)
        self.scheduler.start()
