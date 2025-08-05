# src/config/environment.py

import os
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

from src.utils.logger import system_logger

# --- Environment Loading ---
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    print(".env file loaded.")
else:
    print(".env file not found. Using environment variables or defaults.")


class EnvironmentSettings(BaseSettings):
    """
    Manages all environment-specific settings using Pydantic.
    """

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
    def is_live_mode(self) -> bool:
        """Check if running in live trading mode."""
        return self.trading_environment == "LIVE"

    @property
    def is_testnet_mode(self) -> bool:
        """Check if running in testnet mode."""
        return self.trading_environment == "TESTNET"

    @property
    def is_paper_mode(self) -> bool:
        """Check if running in paper trading mode."""
        return self.trading_environment == "PAPER"

    @property
    def current_exchange_credentials(self) -> dict[str, Any]:
        """Get credentials for the current exchange."""
        exchange = self.exchange_name.lower()
        
        if exchange == "gateio":
            return {
                "api_key": self.gateio_api_key,
                "api_secret": self.gateio_api_secret,
            }
        elif exchange == "mexc":
            return {
                "api_key": self.mexc_api_key,
                "api_secret": self.mexc_api_secret,
            }
        elif exchange == "okx":
            return {
                "api_key": self.okx_api_key,
                "api_secret": self.okx_api_secret,
                "password": self.okx_password,
            }
        elif exchange == "binance":
            return {
                "api_key": self.binance_api_key,
                "api_secret": self.binance_api_secret,
            }
        else:
            return {}

    # --- Validators ---
    class Config:
        case_sensitive = True


# --- Global Settings Instance ---
try:
    settings = EnvironmentSettings()
    system_logger.info(
        f"Environment configuration loaded successfully. "
        f"Trading Environment: {settings.trading_environment}, "
        f"Exchange: {settings.exchange_name}"
    )
except Exception as e:
    system_logger.error(f"Failed to load or validate environment configuration: {e}")
    raise


def get_environment_settings() -> EnvironmentSettings:
    """
    Get the global environment settings instance.
    
    Returns:
        EnvironmentSettings: The global settings instance
    """
    return settings


def get_trading_environment() -> str:
    """
    Get the current trading environment.
    
    Returns:
        str: The trading environment (LIVE, TESTNET, or PAPER)
    """
    return settings.trading_environment


def get_exchange_name() -> str:
    """
    Get the current exchange name.
    
    Returns:
        str: The exchange name
    """
    return settings.exchange_name


def get_trade_symbol() -> str:
    """
    Get the current trade symbol.
    
    Returns:
        str: The trade symbol
    """
    return settings.trade_symbol


def get_timeframe() -> str:
    """
    Get the current timeframe.
    
    Returns:
        str: The timeframe
    """
    return settings.timeframe


def get_initial_equity() -> float:
    """
    Get the initial equity.
    
    Returns:
        float: The initial equity
    """
    return settings.initial_equity


def is_live_mode() -> bool:
    """
    Check if running in live mode.
    
    Returns:
        bool: True if in live mode, False otherwise
    """
    return settings.is_live_mode


def get_exchange_credentials() -> dict[str, Any]:
    """
    Get credentials for the current exchange.
    
    Returns:
        dict: Exchange credentials
    """
    return settings.current_exchange_credentials


def get_database_config() -> dict[str, Any]:
    """
    Get database configuration.
    
    Returns:
        dict: Database configuration
    """
    return {
        "influxdb": {
            "url": settings.influxdb_url,
            "token": settings.influxdb_token,
            "org": settings.influxdb_org,
            "bucket": settings.influxdb_bucket,
        },
        "firestore": {
            "credentials": settings.google_application_credentials,
            "project_id": settings.firestore_project_id,
        },
    }


def get_mlflow_config() -> dict[str, Any]:
    """
    Get MLflow configuration.
    
    Returns:
        dict: MLflow configuration
    """
    return {
        "tracking_uri": settings.mlflow_tracking_uri,
        "experiment_name": settings.mlflow_experiment_name,
    }


def get_email_config() -> dict[str, Any]:
    """
    Get email configuration.
    
    Returns:
        dict: Email configuration
    """
    return {
        "sender_address": settings.email_sender_address,
        "sender_password": settings.email_sender_password,
        "recipient_address": settings.email_recipient_address,
    } 