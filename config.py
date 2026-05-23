import os
from dataclasses import dataclass


@dataclass
class Config:
    # Telegram
    telegram_token: str = os.environ.get("TELEGRAM_TOKEN", "")
    telegram_crypto_chat_id: str = os.environ.get("TELEGRAM_CRYPTO_CHAT_ID", "")
    telegram_india_chat_id: str = os.environ.get("TELEGRAM_INDIA_CHAT_ID", "")

    # Brokers
    coindcx_api_key: str = os.environ.get("COINDCX_API_KEY", "")
    coindcx_api_secret: str = os.environ.get("COINDCX_API_SECRET", "")
    upstox_api_key: str = os.environ.get("UPSTOX_API_KEY", "")
    upstox_access_token: str = os.environ.get("UPSTOX_ACCESS_TOKEN", "")

    # Claude API
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # Data APIs
    cryptopanic_api_key: str = os.environ.get("CRYPTOPANIC_API_KEY", "")
    coinglass_api_key: str = os.environ.get("COINGLASS_API_KEY", "")

    # Trading thresholds
    min_confidence: float = 0.65
    max_portfolio_pct_per_trade: float = 0.02  # 2%
    max_open_positions_per_market: int = 3
    min_rr_ratio: float = 1.8
    options_stop_pct: float = 0.40  # exit if premium drops 40%
    options_target_pct: float = 1.00  # exit if premium doubles
    crypto_signal_expiry_hours: int = 24
    india_signal_expiry_days: int = 3

    # Paths
    crypto_progress_path: str = "data/crypto_progress.json"
    india_progress_path: str = "data/india_progress.json"
    signal_log_path: str = "data/signal_log.json"

    # Dashboard
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 5000


config = Config()
