# src/config/environment.py

import os
from typing import Any, Literal

try:
    from dotenv import load_dotenv
except Exception:  # soft-fallback for smoke tests without dotenv
    def load_dotenv(*args, **kwargs):
        return False

try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
except Exception:  # minimal fallback types for smoke test
    class BaseSettings:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def Field(default=None, env: str | None = None):  # type: ignore
        return default

from src.utils.logger import system_logger

# --- Environment Loading ---
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    pass


class EnvironmentSettings(BaseSettings):
    """Manages all environment-specific settings using Pydantic."""

    # --- Basic Trading Settings ---
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    trading_environment: Literal["LIVE", "TESTNET", "PAPER"] = Field(
        default="PAPER",
        env="TRADING_ENVIRONMENT",
    )
    initial_equity: float = Field(default=100.0, env="INITIAL_EQUITY")
    trade_symbol: str = Field(default="ETHUSDT", env="TRADE_SYMBOL")
    exchange_name: str = Field(default="BINANCE", env="EXCHANGE_NAME")
    timeframe: str = Field(default="15m", env="TIMEFRAME")

    # --- Exchange Credentials ---
    # Gate.io
    gateio_api_key: str | None = Field(default=None, env="GATEIO_API_KEY")
    gateio_api_secret: str | None = Field(default=None, env="GATEIO_API_SECRET")

    # MEXC
    mexc_api_key: str | None = Field(default=None, env="MEXC_API_KEY")
    mexc_api_secret: str | None = Field(default=None, env="MEXC_API_SECRET")

    # OKX
    okx_api_key: str | None = Field(default=None, env="OKX_API_KEY")
    okx_api_secret: str | None = Field(default=None, env="OKX_API_SECRET")
    okx_password: str | None = Field(default=None, env="OKX_PASSWORD")

    # Binance
    binance_api_key: str | None = Field(default=None, env="BINANCE_API_KEY")
    binance_api_secret: str | None = Field(default=None, env="BINANCE_API_SECRET")

    # --- Database Credentials ---
    # Firestore
    google_application_credentials: str | None = Field(
        default=None,
        env="GOOGLE_APPLICATION_CREDENTIALS",
    )
    firestore_project_id: str | None = Field(
        default=None,
        env="FIRESTORE_PROJECT_ID",
    )

    # InfluxDB
    influxdb_url: str | None = Field(
        default="http://localhost:8086",
        env="INFLUXDB_URL",
    )
    influxdb_token: str | None = Field(
        default="your_influxdb_token",
        env="INFLUXDB_TOKEN",
    )
    influxdb_org: str | None = Field(
        default="your_org",
        env="INFLUXDB_ORG",
    )
    influxdb_bucket: str | None = Field(
        default="ares_market_data",
        env="INFLUXDB_BUCKET",
    )

    # --- Email Credentials ---
    email_sender_address: str | None = Field(
        default=None,
        env="EMAIL_SENDER_ADDRESS",
    )
    email_sender_password: str | None = Field(
        default=None,
        env="EMAIL_SENDER_PASSWORD",
    )
    email_recipient_address: str | None = Field(
        default=None,
        env="EMAIL_RECIPIENT_ADDRESS",
    )

    # --- MLflow Configuration ---
    mlflow_tracking_uri: str | None = Field(
        default="file:./mlruns",
        env="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str | None = Field(
        default="Ares_Trading_Models",
        env="MLFLOW_EXPERIMENT_NAME",
    )

    # --- Derived Properties ---
    @property
    @property
    def is_testnet_mode(self) -> bool:
        """Check if running in testnet mode."""
        return self.trading_environment == "TESTNET"

    @property
    def validate_credentials(self, exchange_name: str) -> bool:
        """Validate that credentials are available for the specified exchange.

        Args:
            exchange_name: Name of the exchange

        Returns:
            bool: True if credentials are available

        """
        credentials = self.get_exchange_credentials(exchange_name)
        return (
            credentials["api_key"] is not None and credentials["api_secret"] is not None
        )

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"

