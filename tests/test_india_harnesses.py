from unittest.mock import MagicMock, patch


def test_intraday_harness_has_5_scheduled_times():
    from harnesses.india_intraday_harness import IndiaIntradayHarness

    with patch("harnesses.india_intraday_harness.UpstoxBroker"):
        h = IndiaIntradayHarness(telegram_bot=None, signal_aggregator=MagicMock())
    jobs = h.scheduler.get_jobs()
    assert len(jobs) == 5  # 9:20, 10:15, 11:30, 13:00, 14:00


def test_positional_harness_has_1_scheduled_time():
    from harnesses.india_positional_harness import IndiaPositionalHarness

    with patch("harnesses.india_positional_harness.UpstoxBroker"):
        h = IndiaPositionalHarness(telegram_bot=None, signal_aggregator=MagicMock())
    jobs = h.scheduler.get_jobs()
    assert len(jobs) == 1  # 9:20 only


def test_intraday_harness_tickers():
    from harnesses.india_intraday_harness import TICKERS

    assert "NIFTY" in TICKERS
    assert "BANKNIFTY" in TICKERS
    assert "SENSEX" in TICKERS
    assert "NIFTYIT" in TICKERS


def test_positional_harness_tickers():
    from harnesses.india_positional_harness import TICKERS

    assert len(TICKERS) == 4


def test_both_harnesses_use_market_india():
    from core.models import Market
    from harnesses.india_intraday_harness import IndiaIntradayHarness
    from harnesses.india_positional_harness import IndiaPositionalHarness

    assert IndiaIntradayHarness.MARKET == Market.INDIA
    assert IndiaPositionalHarness.MARKET == Market.INDIA
