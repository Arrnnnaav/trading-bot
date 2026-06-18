import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Telegram
    telegram_token: str = os.environ.get("TELEGRAM_TOKEN", "")
    telegram_india_chat_id: str = os.environ.get("TELEGRAM_INDIA_CHAT_ID", "")
    telegram_allowed_user_ids: str = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "")

    # Brokers
    upstox_api_key: str = os.environ.get("UPSTOX_API_KEY", "")
    upstox_access_token: str = os.environ.get("UPSTOX_ACCESS_TOKEN", "")

    # Data APIs
    marketaux_api_key: str = os.environ.get("MARKETAUX_API_KEY", "")

    # Trading thresholds
    min_confidence: float = 0.65
    max_portfolio_pct_per_trade: float = 0.02
    max_risk_pct_per_trade: float = 0.005
    max_open_positions_per_market: int = 3
    max_daily_loss_pct: float = 0.02
    max_weekly_loss_pct: float = 0.05
    max_consecutive_losses: int = 3
    min_rr_ratio: float = 1.8
    max_slippage_bps: float = 15.0
    backtest_fee_bps: float = 8.0
    backtest_slippage_bps: float = 5.0
    options_stop_pct: float = 0.35
    options_target_pct: float = 1.00
    india_signal_expiry_days: int = 3

    # India-specific
    india_options_expiry: str = os.environ.get("INDIA_OPTIONS_EXPIRY", "")
    india_vix_gate: float = float(os.environ.get("INDIA_VIX_GATE", "25.0"))
    intraday_force_exit: str = os.environ.get("INTRADAY_FORCE_EXIT", "15:15")
    positional_max_days: int = int(os.environ.get("POSITIONAL_MAX_DAYS", "5"))

    # Paths
    india_progress_path: str = "data/india_progress.json"
    signal_log_path: str = "data/signal_log.json"

    # Paper trading
    paper_trading: bool = os.environ.get("PAPER_TRADING", "false").lower() == "true"
    paper_trades_path: str = "data/paper_trades.json"

    # Market toggles
    enable_india: bool = os.environ.get("ENABLE_INDIA", "true").lower() == "true"
    enable_intraday: bool = os.environ.get("ENABLE_INTRADAY", "true").lower() == "true"
    enable_positional: bool = (
        os.environ.get("ENABLE_POSITIONAL", "true").lower() == "true"
    )

    # Strategy
    strategy_profile: str = os.environ.get("STRATEGY_PROFILE", "balanced")
    enable_llm_adjudication: bool = (
        os.environ.get("ENABLE_LLM_ADJUDICATION", "false").lower() == "true"
    )
    execution_mode: str = os.environ.get("EXECUTION_MODE", "semi_auto")

    # Model paths
    kronos_model_path: str = "models/kronos_india/best.pt"
    xgb_model_path: str = "models/xgb_india/model.pkl"

    # News / vector store
    news_db_path: str = "data/news.db"
    chroma_db_path: str = "data/chroma"

    # Dashboard
    dashboard_host: str = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    dashboard_port: int = int(os.environ.get("DASHBOARD_PORT", "5000"))

    # Runtime
    run_startup_calibration: bool = (
        os.environ.get("RUN_STARTUP_CALIBRATION", "false").lower() == "true"
    )

    def __repr__(self) -> str:
        _MASKED = {"key", "secret", "token", "password"}
        fields = []
        for f in self.__dataclass_fields__:
            val = getattr(self, f)
            if any(m in f.lower() for m in _MASKED) and val:
                val = "***"
            fields.append(f"{f}={val!r}")
        return f"Config({', '.join(fields)})"

    def validate_for_runtime(self) -> list[str]:
        errors: list[str] = []
        if not self.telegram_token:
            errors.append("TELEGRAM_TOKEN is required")
        if self.enable_india and not self.telegram_india_chat_id:
            errors.append("TELEGRAM_INDIA_CHAT_ID is required when ENABLE_INDIA=true")
        if self.enable_india and not self.telegram_allowed_user_ids:
            errors.append(
                "TELEGRAM_ALLOWED_USER_IDS is required when ENABLE_INDIA=true"
            )
        if self.enable_india and not self.paper_trading and not self.upstox_api_key:
            errors.append(
                "UPSTOX_API_KEY is required when ENABLE_INDIA=true and PAPER_TRADING=false"
            )
        if not self.paper_trading and not self.upstox_access_token:
            errors.append("UPSTOX_ACCESS_TOKEN is required when PAPER_TRADING=false")
        if self.dashboard_host == "0.0.0.0" and not self.paper_trading:
            errors.append("Refuse to bind dashboard to 0.0.0.0 in live mode")
        return errors


config = Config()
