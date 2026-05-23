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
from harnesses.india_harness import IndiaHarness
from tg_bot.bot import TelegramBot
from dashboard.server import app as dashboard_app


def build_components():
    """Instantiate and wire all components. Returns (crypto_harness, india_harness, telegram_bot, aggregator)."""

    # Shared signal aggregator
    aggregator = SignalAggregator(signal_log_path=config.signal_log_path)

    # Create TelegramBot with an empty harness_states dict — we fill it after harnesses exist
    harness_states: dict = {}

    # Harnesses create their own brokers and agents internally
    crypto_harness = CryptoHarness(telegram_bot=None, signal_aggregator=aggregator)
    india_harness = IndiaHarness(telegram_bot=None, signal_aggregator=aggregator)

    # Populate harness_states so StopMonitor and TelegramBot share live references
    harness_states[Market.CRYPTO] = crypto_harness.state
    harness_states[Market.INDIA] = india_harness.state

    # Now create TelegramBot — brokers are stored on the harness objects
    telegram_bot = TelegramBot(
        harness_states=harness_states,
        coindcx_broker=crypto_harness.broker,
        upstox_broker=india_harness.broker,
        signal_aggregator=aggregator,
    )

    # Inject telegram_bot back into harnesses
    crypto_harness.telegram_bot = telegram_bot
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
    print("=" * 50)

    # Start APScheduler jobs for both harnesses
    crypto_harness.start()
    india_harness.start()

    # Background tasks
    tracker = SignalTracker(aggregator, crypto_harness.broker, india_harness.broker)
    monitor = StopMonitor(
        crypto_harness.broker, india_harness.broker, harness_states, telegram_bot
    )

    server = uvicorn.Server(
        uvicorn.Config(
            app=dashboard_app,
            host=config.dashboard_host,
            port=config.dashboard_port,
            log_level="warning",
        )
    )

    # Create explicit tasks so we can cancel them on shutdown
    tracker_task = asyncio.create_task(tracker.run_forever(900))
    monitor_task = asyncio.create_task(monitor.run_forever(300))
    tg_task = asyncio.create_task(asyncio.to_thread(telegram_bot.run_polling))

    try:
        await server.serve()  # blocks until server exits
    finally:
        # Cancel background tasks to prevent hanging infinite loops
        for task in (tracker_task, monitor_task, tg_task):
            task.cancel()

        # Wait for cancellation to propagate with exception handling
        await asyncio.gather(
            tracker_task, monitor_task, tg_task, return_exceptions=True
        )

        # Signal uvicorn to exit cleanly
        server.should_exit = True

        # NOTE: asyncio.to_thread(telegram_bot.run_polling) wraps a blocking
        # forever-loop (python-telegram-bot v20 Application.run_polling starts
        # its own event loop).  There is no safe cross-thread stop handle exposed
        # here, so the thread will finish on its own once the process exits.
        # This is a known limitation of mixing sync polling with asyncio.

        # Graceful shutdown of APScheduler
        if crypto_harness.scheduler.running:
            crypto_harness.scheduler.shutdown(wait=False)
        if india_harness.scheduler.running:
            india_harness.scheduler.shutdown(wait=False)
        print("Trading Bot shut down.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
