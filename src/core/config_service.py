from __future__ import annotations
import asyncio
import importlib
import json
import os
import os.path
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml
from src.utils.logger import system_logger
from src.utils.warning_symbols import failed, warning
from copy import copy
from typing import Dict, List, Optional, Union, Any, Tuple
from src.core.decorators.errors import handles_errors
try:
    _watchdog_events = importlib.import_module('watchdog.events')
    _watchdog_observers = importlib.import_module('watchdog.observers')
    FileSystemEventHandler = _watchdog_events.FileSystemEventHandler
    Observer = _watchdog_observers.Observer
    WATCHDOG_AVAILABLE = True
except Exception:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

@dataclass
class DatabaseConfig:
    """Database configuration dataclass."""
    database_path: str = 'data/ares.db'
    auto_backup: bool = True
    backup_interval: int = 3600
    max_connections: int = 10
    enable_foreign_keys: bool = True
    journal_mode: str = 'WAL'
    max_recovery_attempts: int = 3
    recovery_cooldown: int = 60

@dataclass
class ExchangeConfig:
    """Exchange configuration dataclass."""
    exchange_name: str = 'BINANCE'
    api_key: str = ''
    api_secret: str = ''
    testnet: bool = True
    rate_limit: int = 1200
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1

@dataclass
class ModelTrainingConfig:
    """Model training configuration dataclass."""
    enable_advanced_training: bool = True
    enable_ensemble_training: bool = True
    enable_multi_timeframe_training: bool = True
    enable_adaptive_training: bool = True
    training_interval: int = 3600
    max_training_history: int = 100
    lookback_days: int = 730
    min_data_points: int = 100000

@dataclass
class RiskConfig:
    """Risk management configuration dataclass."""
    max_position_size: float = 0.1
    max_portfolio_risk: float = 0.02
    stop_loss_percentage: float = 0.05
    take_profit_percentage: float = 0.15
    max_drawdown: float = 0.2
    risk_free_rate: float = 0.02
if WATCHDOG_AVAILABLE:

    class ConfigurationWatcher(FileSystemEventHandler):
        """Watchdog-based configuration file watcher."""

        def __init__(self, config_service: 'ConfigurationService') -> None:
            self.config_service = config_service
            self.logger = system_logger.getChild('ConfigurationWatcher')

        def on_modified(self, event: Any) -> None:
            """Handle file modification events."""
            if event.src_path.endswith(('.yaml', '.yml', '.json')):
                self.logger.info(f'Configuration file changed: {event.src_path}')
                try:
                    loop = self.config_service.loop
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(self.config_service._reload_configuration(), loop)
                    else:
                        asyncio.run(self.config_service._reload_configuration())
                except Exception:
                    self.logger.exception('Failed to schedule configuration reload')
else:

    class ConfigurationWatcher:
        """Dummy configuration watcher when watchdog is not available."""

        def __init__(self, config_service: 'ConfigurationService') -> None:
            self.config_service = config_service
            self.logger = system_logger.getChild('ConfigurationWatcher')

        def on_modified(self, event: Any) -> None:
            """Handle file modification events."""

class ConfigurationService:
    """
    Enhanced Configuration Service with hot-reload, environment-specific configs,
    and dynamic configuration management.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild('ConfigurationService')
        self.is_initialized: bool = False
        self.config_data: dict[str, Any] = {}
        self.config_sections: dict[str, Any] = {}
        self.config_history: list[dict[str, Any]] = []
        self.max_history: int = 100
        self.environment: str = os.getenv('TRADING_ENV', 'development')
        self.config_files: list[str] = []
        self.config_directories: list[str] = ['config']
        self.enable_hot_reload: bool = self.config.get('enable_hot_reload', True)
        self.watcher: Any | None = None
        self.watched_files: set[str] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.validation_rules: dict[str, Any] = {}
        self.validation_errors: list[str] = []
        self.encryption_enabled: bool = self.config.get('encryption_enabled', False)
        self.encryption_key: str | None = None
        self.load_times: list[float] = []
        self.last_load_time: float = 0

    def get_value(self, dotted_key: str, default: Any=None) -> Any:
        """Retrieve a configuration value using a dotted path from config_data.

        Falls back to the initial raw config (self.config) if not present in
        the merged config_data.
        """
        try:

            def _get(dct: dict, path: list[str]) -> Any:
                cur = dct
                for part in path:
                    if not isinstance(cur, dict) or part not in cur:
                        return None
                    cur = cur[part]
                return cur
            parts = dotted_key.split('.') if dotted_key else []
            val = _get(self.config_data, parts) if parts else None
            if val is None:
                val = _get(self.config, parts) if parts else None
            return default if val is None else val
        except Exception:
            self.logger.exception(f'Error reading config value for key: {dotted_key}')
            return default

    def print(self, message: str) -> None:
        """Proxy print to logger to keep output consistent in terminal."""
        try:
            self.logger.info(message)
        except Exception:
            self.print(message)

    @handles_errors(error_handlers={ValueError: (False, 'Invalid configuration service setup'), AttributeError: (False, 'Missing required configuration parameters'), KeyError: (False, 'Missing configuration keys')}, default_return=False, context='configuration service initialization')
    async def initialize(self) -> bool:
        """Initialize configuration service with enhanced capabilities."""
        try:
            self.logger.info('Initializing Configuration Service...')
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = None
            await self._load_configuration()
            if not await self._validate_configuration():
                self.print(failed('Configuration validation failed'))
                return False
            await self._setup_configuration_sections()
            if self.enable_hot_reload:
                await self._setup_hot_reload()
            if self.encryption_enabled:
                await self._setup_encryption()
            self.is_initialized = True
            self.logger.info('✅ Configuration Service initialization completed successfully')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Configuration Service initialization failed: {e}')
            return False

    @handles_errors(fallback=None)
    async def _load_configuration(self) -> None:
        """Load configuration from multiple sources."""
        try:
            start_time = time.time()
            self.config_files = ['config/base.yaml', f'config/{self.environment}.yaml', f'config/{self.environment}_local.yaml']
            for config_file in self.config_files:
                if os.path.exists(config_file):
                    await self._load_config_file(config_file)
            await self._load_from_environment()
            await self._load_from_arguments()
            load_time = time.time() - start_time
            self.load_times.append(load_time)
            self.last_load_time = load_time
            if len(self.load_times) > 10:
                self.load_times = self.load_times[-10:]
            self.logger.info(f'Configuration loaded successfully in {load_time:.3f}s')
        except Exception as e:
            self.print(error(f'Error loading configuration: {e}'))

    @handles_errors(default_return=None, context='config file loading')
    async def _load_config_file(self, config_file: str) -> None:
        """Load configuration from a specific file."""
        try:
            file_path = Path(config_file)
            if not file_path.exists():
                self.logger.warning(f'Configuration file not found: {config_file}')
                return
            with open(file_path, encoding='utf-8') as f:
                if config_file.endswith(('.yaml', '.yml')):
                    file_config = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    file_config = json.load(f)
                else:
                    self.logger.warning(f'Unsupported config file format: {config_file}')
                    return
            if file_config:
                self._merge_configuration(file_config)
                self.logger.info(f'Loaded configuration from: {config_file}')
        except Exception as e:
            self.logger.exception(f'Error loading config file {config_file}: {e}')

    async def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        try:
            env_config = {}
            for key, value in os.environ.items():
                if key.startswith('TRADING_'):
                    config_key = key[8:].lower()
                    keys = config_key.split('.')
                    current = env_config
                    for k in keys[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[keys[-1]] = value
            if env_config:
                self._merge_configuration(env_config)
                self.logger.info('Loaded configuration from environment variables')
        except Exception as e:
            self.logger.exception(f'Error loading from environment: {e}')

    async def _load_from_arguments(self) -> None:
        """Load configuration from command line arguments."""
        try:
            arg_config = {}
            if arg_config:
                self._merge_configuration(arg_config)
                self.logger.info('Loaded configuration from command line arguments')
        except Exception as e:
            self.logger.exception(f'Error loading from arguments: {e}')

    def _merge_configuration(self, new_config: dict[str, Any]) -> None:
        """Merge new configuration with existing configuration."""
        try:

            def deep_merge(base: dict, update: dict) -> dict:
                """Deep merge two dictionaries."""
                result = base.copy()
                for key, value in update.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result
            self.config_data = deep_merge(self.config_data, new_config)
            self.config_history.append({'timestamp': datetime.now().isoformat(), 'config': new_config.copy()})
            if len(self.config_history) > self.max_history:
                self.config_history = self.config_history[-self.max_history:]
        except Exception as e:
            self.logger.exception(f'Error merging configuration: {e}')

    async def _validate_configuration(self) -> bool:
        """Validate configuration using defined rules."""
        try:
            self.validation_errors.clear()
            required_keys = ['database', 'exchange', 'risk']
            for key in required_keys:
                if key not in self.config_data:
                    self.validation_errors.append(f'Missing required configuration section: {key}')
            if 'database' in self.config_data:
                db_config = self.config_data['database']
                if not isinstance(db_config.get('database_path'), str):
                    self.validation_errors.append('Database path must be a string')
            if 'exchange' in self.config_data:
                exchange_config = self.config_data['exchange']
                if not exchange_config.get('api_key'):
                    self.validation_errors.append('Exchange API key is required')
            if self.validation_errors:
                for error_msg in self.validation_errors:
                    self.print(error(f'Configuration validation error: {error_msg}'))
                return False
            return True
        except Exception as e:
            self.logger.exception(f'Error validating configuration: {e}')
            return False

    async def _setup_configuration_sections(self) -> None:
        """Setup typed configuration sections."""
        try:
            db_config_data = self.config_data.get('database', {})
            self.config_sections['database'] = DatabaseConfig(**db_config_data)
            exchange_config_data = self.config_data.get('exchange', {})
            self.config_sections['exchange'] = ExchangeConfig(**exchange_config_data)
            training_config_data = self.config_data.get('training', {})
            self.config_sections['training'] = ModelTrainingConfig(**training_config_data)
            risk_config_data = self.config_data.get('risk', {})
            self.config_sections['risk'] = RiskConfig(**risk_config_data)
            self.logger.info('Configuration sections setup completed')
        except Exception as e:
            self.logger.exception(f'Error setting up configuration sections: {e}')

    async def _setup_hot_reload(self) -> None:
        """Setup hot-reload for configuration files."""
        try:
            if not WATCHDOG_AVAILABLE:
                self.print(warning('Watchdog not available, hot-reload disabled'))
                return
            if not self.watcher:
                self.watcher = Observer()
                self.watcher.start()
            for config_dir in self.config_directories:
                if os.path.exists(config_dir):
                    event_handler = ConfigurationWatcher(self)
                    self.watcher.schedule(event_handler, config_dir, recursive=True)
                    self.watched_files.add(config_dir)
                    self.logger.info(f'Watching configuration directory: {config_dir}')
        except Exception as e:
            self.logger.exception(f'Error setting up hot-reload: {e}')

    async def _setup_encryption(self) -> None:
        """Setup configuration encryption."""
        try:
            self.encryption_key = os.getenv('CONFIG_ENCRYPTION_KEY')
            if not self.encryption_key:
                self.print(warning('No encryption key provided, encryption disabled'))
                self.encryption_enabled = False
        except Exception as e:
            self.logger.exception(f'Error setting up encryption: {e}')

    async def _reload_configuration(self) -> None:
        """Reload configuration from files."""
        try:
            self.logger.info('🔄 Reloading configuration...')
            self.config_data.clear()
            self.config_sections.clear()
            await self._load_configuration()
            if await self._validate_configuration():
                await self._setup_configuration_sections()
                self.logger.info('✅ Configuration reloaded successfully')
            else:
                self.logger.error('❌ Configuration reload failed validation')
        except Exception as e:
            self.logger.exception(f'Error reloading configuration: {e}')

    def get_config(self, section: str | None=None) -> Any:
        """Get configuration data."""
        try:
            if section:
                return self.config_sections.get(section)
            return self.config_data
        except Exception as e:
            self.logger.exception(f'Error getting configuration: {e}')
            return None

    def update_config(self, section: str, updates: dict[str, Any]) -> bool:
        """Update configuration dynamically."""
        try:
            if section not in self.config_sections:
                self.print(error(f'Unknown configuration section: {section}'))
                return False
            current_config = asdict(self.config_sections[section])
            current_config.update(updates)
            if section == 'database':
                self.config_sections[section] = DatabaseConfig(**current_config)
            elif section == 'exchange':
                self.config_sections[section] = ExchangeConfig(**current_config)
            elif section == 'training':
                self.config_sections[section] = ModelTrainingConfig(**current_config)
            elif section == 'risk':
                self.config_sections[section] = RiskConfig(**current_config)
            self.logger.info(f'Updated configuration section: {section}')
            return True
        except Exception as e:
            self.logger.exception(f'Error updating configuration: {e}')
            return False

    def get_status(self) -> dict[str, Any]:
        """Get configuration service status."""
        try:
            return {'is_initialized': self.is_initialized, 'environment': self.environment, 'config_files': self.config_files, 'watched_files': list(self.watched_files), 'validation_errors': self.validation_errors, 'load_times': self.load_times, 'last_load_time': self.last_load_time}
        except Exception as e:
            self.logger.exception(f'Error getting status: {e}')
            return {}

    def get_history(self, limit: int | None=None) -> list[dict[str, Any]]:
        """Get configuration history."""
        try:
            history = self.config_history.copy()
            if limit:
                history = history[-limit:]
            return history
        except Exception as e:
            self.logger.exception(f'Error getting history: {e}')
            return []

    async def shutdown(self) -> None:
        """Shutdown the configuration service."""
        try:
            if self.watcher:
                self.watcher.stop()
                self.watcher.join()
            self.is_initialized = False
            self.logger.info('Configuration service shutdown completed')
        except Exception as e:
            self.logger.exception(f'Error during shutdown: {e}')
config_service: ConfigurationService | None = None

def get_config_service() -> ConfigurationService:
    """Get the global configuration service instance."""
    global config_service
    if config_service is None:
        default_config = {'enable_hot_reload': True, 'encryption_enabled': False}
        config_service = ConfigurationService(default_config)
    return config_service