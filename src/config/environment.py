from __future__ import annotations
import os
from typing import Any, Literal
from typing import Dict, List, Optional, Union, Any, Tuple
try:
    from dotenv import load_dotenv
except Exception:

    def load_dotenv(*args, **kwargs) -> Any:
        return False
try:
    from pydantic import Field
    from pydantic_settings import BaseSettings
except Exception:

    class BaseSettings:

        def __init__(self, **kwargs) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def Field(default: Any=None, env: str | None=None) -> None:
        return default
import os.path
from src.utils.logger import system_logger
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
else:
    pass

class EnvironmentSettings(BaseSettings):
    """Manages all environment-specific settings using Pydantic."""
    log_level: str = Field(default='INFO', env='LOG_LEVEL')
    trading_environment: Literal['LIVE', 'TESTNET', 'PAPER'] = Field(default='PAPER', env='TRADING_ENVIRONMENT')
    initial_equity: float = Field(default=100.0, env='INITIAL_EQUITY')
    trade_symbol: str = Field(default='ETHUSDT', env='TRADE_SYMBOL')
    exchange_name: str = Field(default='BINANCE', env='EXCHANGE_NAME')
    timeframe: str = Field(default='15m', env='TIMEFRAME')
    gateio_api_key: str | None = Field(default=None, env='GATEIO_API_KEY')
    gateio_api_secret: str | None = Field(default=None, env='GATEIO_API_SECRET')
    mexc_api_key: str | None = Field(default=None, env='MEXC_API_KEY')
    mexc_api_secret: str | None = Field(default=None, env='MEXC_API_SECRET')
    okx_api_key: str | None = Field(default=None, env='OKX_API_KEY')
    okx_api_secret: str | None = Field(default=None, env='OKX_API_SECRET')
    okx_password: str | None = Field(default=None, env='OKX_PASSWORD')
    binance_api_key: str | None = Field(default=None, env='BINANCE_API_KEY')
    binance_api_secret: str | None = Field(default=None, env='BINANCE_API_SECRET')
    google_application_credentials: str | None = Field(default=None, env='GOOGLE_APPLICATION_CREDENTIALS')
    firestore_project_id: str | None = Field(default=None, env='FIRESTORE_PROJECT_ID')
    influxdb_url: str | None = Field(default='http://localhost:8086', env='INFLUXDB_URL')
    influxdb_token: str | None = Field(default='your_influxdb_token', env='INFLUXDB_TOKEN')
    influxdb_org: str | None = Field(default='your_org', env='INFLUXDB_ORG')
    influxdb_bucket: str | None = Field(default='ares_market_data', env='INFLUXDB_BUCKET')
    email_sender_address: str | None = Field(default=None, env='EMAIL_SENDER_ADDRESS')
    email_sender_password: str | None = Field(default=None, env='EMAIL_SENDER_PASSWORD')
    email_recipient_address: str | None = Field(default=None, env='EMAIL_RECIPIENT_ADDRESS')
    mlflow_tracking_uri: str | None = Field(default='file:./mlruns', env='MLFLOW_TRACKING_URI')
    mlflow_experiment_name: str | None = Field(default='Ares_Trading_Models', env='MLFLOW_EXPERIMENT_NAME')

    @property
    def is_live_mode(self) -> bool:
        """Check if running in live mode."""
        return self.trading_environment == 'LIVE'

    @property
    def is_testnet_mode(self) -> bool:
        """Check if running in testnet mode."""
        return self.trading_environment == 'TESTNET'

    @property
    def is_paper_mode(self) -> bool:
        """Check if running in paper mode."""
        return self.trading_environment == 'PAPER'

    def get_exchange_credentials(self, exchange_name: str) -> dict[str, str | None]:
        """Get credentials for a specific exchange.

        Args:
            exchange_name: Name of the exchange

        Returns:
            dict: Exchange credentials

        """
        exchange_name_lower = exchange_name.lower()
        if exchange_name_lower == 'binance':
            return {'api_key': self.binance_api_key, 'api_secret': self.binance_api_secret}
        if exchange_name_lower == 'gateio':
            return {'api_key': self.gateio_api_key, 'api_secret': self.gateio_api_secret}
        if exchange_name_lower == 'mexc':
            return {'api_key': self.mexc_api_key, 'api_secret': self.mexc_api_secret}
        if exchange_name_lower == 'okx':
            return {'api_key': self.okx_api_key, 'api_secret': self.okx_api_secret, 'password': self.okx_password}
        return {'api_key': None, 'api_secret': None}

    def validate_credentials(self, exchange_name: str) -> bool:
        """Validate that credentials are available for the specified exchange.

        Args:
            exchange_name: Name of the exchange

        Returns:
            bool: True if credentials are available

        """
        credentials = self.get_exchange_credentials(exchange_name)
        return credentials['api_key'] is not None and credentials['api_secret'] is not None

    def get_database_config(self, database_type: str) -> dict[str, Any]:
        """Get database configuration for a specific database type.

        Args:
            database_type: Type of database (firestore, influxdb, etc.)

        Returns:
            dict: Database configuration

        """
        if database_type.lower() == 'firestore':
            return {'project_id': self.firestore_project_id, 'credentials_path': self.google_application_credentials}
        if database_type.lower() == 'influxdb':
            return {'url': self.influxdb_url, 'token': self.influxdb_token, 'org': self.influxdb_org, 'bucket': self.influxdb_bucket}
        return {}

    def get_email_config(self) -> dict[str, str | None]:
        """Get email configuration.

        Returns:
            dict: Email configuration

        """
        return {'sender_address': self.email_sender_address, 'sender_password': self.email_sender_password, 'recipient_address': self.email_recipient_address}

    def get_mlflow_config(self) -> dict[str, str | None]:
        """Get MLflow configuration.

        Returns:
            dict: MLflow configuration

        """
        return {'tracking_uri': self.mlflow_tracking_uri, 'experiment_name': self.mlflow_experiment_name}

    class Config:
        """Pydantic configuration."""
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
        extra = 'ignore'

def get_environment_settings() -> EnvironmentSettings:
    """Get environment settings instance.

    Returns:
        EnvironmentSettings: Environment settings instance

    """
    try:
        return EnvironmentSettings()
    except Exception as e:
        system_logger.error(f'Error loading environment settings: {e}')
        return EnvironmentSettings(trading_environment='PAPER', trade_symbol='ETHUSDT', exchange_name='BINANCE', timeframe='15m', initial_equity=100.0)