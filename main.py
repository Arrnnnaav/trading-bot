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


async def main():
    (
        india_intraday_harness,
        india_positional_harness,
        telegram_bot,
        aggregator,
        harness_states,
        india_broker,
    ) = build_components()

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
