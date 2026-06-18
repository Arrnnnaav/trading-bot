"""Trading Bot v2 — India Index Options — main entry point.

Wires all components together and runs the async event loop.
"""

import asyncio
import logging
import sys

import uvicorn

from config import config
from core.models import Market
from core.signal_aggregator import SignalAggregator
from core.signal_tracker import SignalTracker
from core.stop_monitor import StopMonitor
from tg_bot.bot import TelegramBot
from dashboard.server import app as dashboard_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_LOG = logging.getLogger(__name__)


def build_components():
    errors = config.validate_for_runtime()
    if errors:
        raise RuntimeError("Invalid runtime configuration:\n- " + "\n- ".join(errors))

    if config.run_startup_calibration:
        from training.calibrate_thresholds import calibrate

        calibrate()

    aggregator = SignalAggregator(signal_log_path=config.signal_log_path)
    harness_states: dict = {}

    india_intraday_harness = None
    india_positional_harness = None
    india_broker = None

    if config.enable_india:
        if config.enable_intraday:
            from harnesses.india_intraday_harness import IndiaIntradayHarness

            india_intraday_harness = IndiaIntradayHarness(
                telegram_bot=None, signal_aggregator=aggregator
            )
            harness_states[Market.INDIA] = india_intraday_harness.state
            india_broker = india_intraday_harness.broker

        if config.enable_positional:
            from harnesses.india_positional_harness import IndiaPositionalHarness

            india_positional_harness = IndiaPositionalHarness(
                telegram_bot=None, signal_aggregator=aggregator
            )
            if Market.INDIA not in harness_states:
                harness_states[Market.INDIA] = india_positional_harness.state
            if india_broker is None:
                india_broker = india_positional_harness.broker

    if india_broker is None and config.enable_india:
        raise RuntimeError(
            "ENABLE_INDIA=true but both ENABLE_INTRADAY and ENABLE_POSITIONAL are false. "
            "Set at least one to true."
        )

    telegram_bot = TelegramBot(
        harness_states=harness_states,
        upstox_broker=india_broker,
        signal_aggregator=aggregator,
    )

    if india_intraday_harness:
        india_intraday_harness.telegram_bot = telegram_bot
    if india_positional_harness:
        india_positional_harness.telegram_bot = telegram_bot

    return (
        india_intraday_harness,
        india_positional_harness,
        telegram_bot,
        aggregator,
        harness_states,
        india_broker,
    )


def _build_data_scheduler():
    """Build APScheduler for data ingestion jobs (FII/DII + live news).

    FII/DII:   19:00 IST weekdays  (NSE publishes ~18:30 IST)
    Live news: every 30min, 09:00–15:30 IST weekdays
    """
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

    # FII/DII: 19:00 IST, Monday–Friday
    def _run_fii_dii():
        from scripts.fetch_fii_dii import fetch_and_save

        result = fetch_and_save(
            output_dir="data/fii_dii",
        )
        if result:
            _LOG.info("[data-scheduler] FII/DII saved: %s", result)
        else:
            _LOG.warning("[data-scheduler] FII/DII fetch returned None")

    scheduler.add_job(
        _run_fii_dii,
        trigger=CronTrigger(
            day_of_week="mon-fri", hour=19, minute=0, timezone="Asia/Kolkata"
        ),
        id="fii_dii_daily",
        name="FII/DII daily fetch",
        replace_existing=True,
        misfire_grace_time=600,
    )

    # Live news: every 30min, 09:00–15:30 IST, Monday–Friday
    def _run_news_fetch():
        from scripts.fetch_news_live import fetch_and_ingest

        result = fetch_and_ingest(
            api_key=config.marketaux_api_key,
            chroma_path=config.chroma_db_path,
            output_dir="data/news",
        )
        _LOG.info("[data-scheduler] News ingestion: %s", result)

    scheduler.add_job(
        _run_news_fetch,
        trigger=CronTrigger(
            day_of_week="mon-fri",
            hour="9-15",
            minute="0,30",
            timezone="Asia/Kolkata",
        ),
        id="news_live_30min",
        name="Live news fetch (30min)",
        replace_existing=True,
        misfire_grace_time=300,
    )

    return scheduler


async def main():
    (
        india_intraday_harness,
        india_positional_harness,
        telegram_bot,
        aggregator,
        harness_states,
        india_broker,
    ) = build_components()

    data_scheduler = _build_data_scheduler()

    print("=" * 50)
    print("Trading Bot v2 — India Index Options")
    print(f"  Dashboard:  http://{config.dashboard_host}:{config.dashboard_port}")
    print("  Telegram:   polling active")
    print(f"  Mode:       {'PAPER TRADING' if config.paper_trading else 'LIVE'}")
    print(f"  Strategy:   {config.strategy_profile}")
    print(
        f"  Harnesses:  "
        f"{'intraday ' if config.enable_intraday else ''}"
        f"{'positional' if config.enable_positional else ''}"
    )
    print("=" * 50)

    if india_intraday_harness:
        india_intraday_harness.start()
    if india_positional_harness:
        india_positional_harness.start()

    data_scheduler.start()
    _LOG.info(
        "Data scheduler started: FII/DII@19:00 IST, news@every 30min 09:00-15:30 IST"
    )

    tracker = SignalTracker(aggregator, india_broker)
    monitor = StopMonitor(
        india_broker=india_broker,
        harness_states=harness_states,
        telegram_bot=telegram_bot,
        signal_aggregator=aggregator,
        intraday_force_exit_time=config.intraday_force_exit,
    )

    server = uvicorn.Server(
        uvicorn.Config(
            app=dashboard_app,
            host=config.dashboard_host,
            port=config.dashboard_port,
            log_level="warning",
        )
    )

    tasks = [
        asyncio.create_task(tracker.run_forever(900)),
        asyncio.create_task(monitor.run_forever(300)),
        asyncio.create_task(asyncio.to_thread(telegram_bot.run_polling)),
    ]

    try:
        await server.serve()
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        server.should_exit = True
        if data_scheduler.running:
            data_scheduler.shutdown(wait=False)
        if india_intraday_harness and india_intraday_harness.scheduler.running:
            india_intraday_harness.scheduler.shutdown(wait=False)
        if india_positional_harness and india_positional_harness.scheduler.running:
            india_positional_harness.scheduler.shutdown(wait=False)
        print("Trading Bot shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except RuntimeError as exc:
        _LOG.error("%s", exc)
        sys.exit(2)
