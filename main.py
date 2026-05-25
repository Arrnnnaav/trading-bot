"""Trading Bot v1 — main entry point.

Wires all components together and runs the async event loop.
"""

import asyncio
import sys

import uvicorn

from config import config
from core.models import Market
from core.signal_aggregator import SignalAggregator
from core.signal_tracker import SignalTracker
from core.stop_monitor import StopMonitor
from harnesses.crypto_harness import CryptoHarness
from tg_bot.bot import TelegramBot
from dashboard.server import app as dashboard_app


def build_components():
    from training.calibrate_thresholds import calibrate

    calibrate()

    aggregator = SignalAggregator(signal_log_path=config.signal_log_path)
    harness_states: dict = {}

    crypto_harness = CryptoHarness(telegram_bot=None, signal_aggregator=aggregator)
    harness_states[Market.CRYPTO] = crypto_harness.state

    india_harness = None
    india_broker = None
    if config.enable_india:
        from harnesses.india_harness import IndiaHarness

        india_harness = IndiaHarness(telegram_bot=None, signal_aggregator=aggregator)
        harness_states[Market.INDIA] = india_harness.state
        india_broker = india_harness.broker

    telegram_bot = TelegramBot(
        harness_states=harness_states,
        coindcx_broker=crypto_harness.broker,
        upstox_broker=india_broker,
        signal_aggregator=aggregator,
    )

    crypto_harness.telegram_bot = telegram_bot
    if india_harness:
        india_harness.telegram_bot = telegram_bot

    return crypto_harness, india_harness, telegram_bot, aggregator, harness_states


async def main():
    crypto_harness, india_harness, telegram_bot, aggregator, harness_states = (
        build_components()
    )

    print("=" * 50)
    print("Trading Bot v1 starting...")
    print(f"  Dashboard:  http://{config.dashboard_host}:{config.dashboard_port}")
    print("  Telegram:   polling active")
    print(f"  Mode:       {'PAPER TRADING' if config.paper_trading else 'LIVE'}")
    print(f"  Strategy:   {config.strategy_profile}")
    print(f"  Markets:    CRYPTO{' + INDIA' if config.enable_india else ''}")
    print("=" * 50)

    crypto_harness.start()
    if india_harness:
        india_harness.start()

    tracker = SignalTracker(
        aggregator,
        crypto_harness.broker,
        india_harness.broker if india_harness else None,
    )
    monitor = StopMonitor(
        crypto_harness.broker,
        india_harness.broker if india_harness else None,
        harness_states,
        telegram_bot,
    )

    server = uvicorn.Server(
        uvicorn.Config(
            app=dashboard_app,
            host=config.dashboard_host,
            port=config.dashboard_port,
            log_level="warning",
        )
    )

    tracker_task = asyncio.create_task(tracker.run_forever(900))
    monitor_task = asyncio.create_task(monitor.run_forever(300))
    tg_task = asyncio.create_task(asyncio.to_thread(telegram_bot.run_polling))

    try:
        await server.serve()
    finally:
        for task in (tracker_task, monitor_task, tg_task):
            task.cancel()
        await asyncio.gather(
            tracker_task, monitor_task, tg_task, return_exceptions=True
        )
        server.should_exit = True
        if crypto_harness.scheduler.running:
            crypto_harness.scheduler.shutdown(wait=False)
        if india_harness and india_harness.scheduler.running:
            india_harness.scheduler.shutdown(wait=False)
        print("Trading Bot shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
