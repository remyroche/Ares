from datetime import datetime
from pathlib import Path
from src.utils.logger import system_logger
from typing import Any, Dict, List, Optional
import asyncio
import json
import os
import time
import importlib
from dataclasses import asdict, dataclass
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.warning_symbols import error, failed, warning
import yaml

# Try to import watchdog for file watching using dynamic import to avoid linter warnings
try:
    _watchdog_events = importlib.import_module("watchdog.events")
    _watchdog_observers = importlib.import_module("watchdog.observers")
    
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
    database_path: str = "data/ares.db"
    auto_backup: bool = True
    backup_interval: int = 3600
    max_connections: int = 10
    enable_foreign_keys: bool = True
    journal_mode: str = "WAL"
    max_recovery_attempts: int = 3
    recovery_cooldown: int = 60


@dataclass
class ExchangeConfig:
    """Exchange configuration dataclass."""
    exchange_name: str = "BINANCE"
    api_key: str = ""
    api_secret: str = ""
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
    max_drawdown: float = 0.20
    risk_free_rate: float = 0.02


class ConfigurationWatcher:
    """Watchdog event handler for configuration file changes."""
    
    def __init__(self, config_service):
        self.config_service = config_service
        self.logger = config_service.logger
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            self.logger.info(f"Configuration file modified: {event.src_path}")
            asyncio.create_task(self.config_service._reload_configuration())


class ConfigurationService:
    """Configuration service for managing application settings."""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = system_logger
        self.config = config
        self.is_initialized = False
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Configuration data
        self.config_data: Dict[str, Any] = {}
        self.config_sections: Dict[str, Any] = {}
        self.config_files: List[str] = []
        self.config_directories: List[str] = []
        self.watched_files: set = set()
        
        # Hot reload
        self.enable_hot_reload = config.get("enable_hot_reload", True)
        self.watcher: Optional[Observer] = None
        
        # Encryption
        self.encryption_enabled = config.get("encryption_enabled", False)
        self.encryption_key: Optional[str] = None
        
        # History and validation
        self.config_history: List[Dict[str, Any]] = []
        self.validation_errors: List[str] = []
        self.load_times: Dict[str, float] = {}
        self.last_load_time: Optional[datetime] = None
    
    async def initialize(self) -> bool:
        """Initialize the configuration service."""
        try:
            self.logger.info("Initializing configuration service...")
            
            # Setup configuration directories
            await self._setup_configuration_directories()
            
            # Load configuration
            if not await self._load_configuration():
                return False
            
            # Validate configuration
            if not await self._validate_configuration():
                return False
            
            # Setup configuration sections
            await self._setup_configuration_sections()
            
            # Setup hot reload if enabled
            if self.enable_hot_reload:
                await self._setup_hot_reload()
            
            # Setup encryption if enabled
            if self.encryption_enabled:
                await self._setup_encryption()
            
            self.is_initialized = True
            self.logger.info("Configuration service initialized successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error initializing configuration service: {e}")
            return False
    
    async def _setup_configuration_directories(self):
        """Setup configuration directories."""
        try:
            base_dir = Path(__file__).parent.parent.parent
            self.config_directories = [
                str(base_dir / "config"),
                str(base_dir / "config" / self.environment),
                str(Path.home() / ".ares" / "config"),
            ]
            
            # Create directories if they don't exist
            for config_dir in self.config_directories:
                Path(config_dir).mkdir(parents=True, exist_ok=True)
                
        except Exception as e:
            self.logger.exception(f"Error setting up configuration directories: {e}")
    
    async def _load_configuration(self) -> bool:
        """Load configuration from files."""
        try:
            self.logger.info("Loading configuration...")
            
            # Load from each configuration directory
            for config_dir in self.config_directories:
                if os.path.exists(config_dir):
                    await self._load_from_directory(config_dir)
            
            # Load environment variables
            self._load_environment_variables()
            
            self.last_load_time = datetime.now()
            self.logger.info(f"Configuration loaded successfully from {len(self.config_files)} files")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error loading configuration: {e}")
            return False
    
    async def _load_from_directory(self, directory: str):
        """Load configuration from a specific directory."""
        try:
            for file_path in Path(directory).glob("*.yaml"):
                await self._load_yaml_file(str(file_path))
            
            for file_path in Path(directory).glob("*.yml"):
                await self._load_yaml_file(str(file_path))
            
            for file_path in Path(directory).glob("*.json"):
                await self._load_json_file(str(file_path))
                
        except Exception as e:
            self.logger.error(f"Error loading from directory {directory}: {e}")
    
    async def _load_yaml_file(self, file_path: str):
        """Load configuration from a YAML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data:
                    self.config_data.update(data)
                    self.config_files.append(file_path)
                    self.load_times[file_path] = time.time()
                    
        except Exception as e:
            self.logger.error(f"Error loading YAML file {file_path}: {e}")
    
    async def _load_json_file(self, file_path: str):
        """Load configuration from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data:
                    self.config_data.update(data)
                    self.config_files.append(file_path)
                    self.load_times[file_path] = time.time()
                    
        except Exception as e:
            self.logger.error(f"Error loading JSON file {file_path}: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables."""
        try:
            # Database configuration
            if os.getenv("ARES_DB_PATH"):
                self.config_data.setdefault("database", {})["database_path"] = os.getenv("ARES_DB_PATH")
            
            if os.getenv("ARES_DB_MAX_CONNECTIONS"):
                self.config_data.setdefault("database", {})["max_connections"] = int(os.getenv("ARES_DB_MAX_CONNECTIONS"))
            
            # Exchange configuration
            if os.getenv("ARES_EXCHANGE_NAME"):
                self.config_data.setdefault("exchange", {})["exchange_name"] = os.getenv("ARES_EXCHANGE_NAME")
            
            if os.getenv("ARES_API_KEY"):
                self.config_data.setdefault("exchange", {})["api_key"] = os.getenv("ARES_API_KEY")
            
            if os.getenv("ARES_API_SECRET"):
                self.config_data.setdefault("exchange", {})["api_secret"] = os.getenv("ARES_API_SECRET")
                
        except Exception as e:
            self.logger.error(f"Error loading environment variables: {e}")
    
    async def _validate_configuration(self) -> bool:
        """Validate the loaded configuration."""
        try:
            self.logger.info("Validating configuration...")
            self.validation_errors.clear()
            
            # Validate required sections
            required_sections = ["database", "exchange", "training", "risk"]
            for section in required_sections:
                if section not in self.config_data:
                    self.validation_errors.append(f"Missing required configuration section: {section}")
            
            # Validate database configuration
            if "database" in self.config_data:
                db_config = self.config_data["database"]
                if not isinstance(db_config.get("database_path"), str):
                    self.validation_errors.append("Database path must be a string")
            
            # Validate exchange configuration
            if "exchange" in self.config_data:
                exchange_config = self.config_data["exchange"]
                if not exchange_config.get("api_key") and not exchange_config.get("testnet", True):
                    self.validation_errors.append("API key required for non-testnet mode")
            
            is_valid = len(self.validation_errors) == 0
            if is_valid:
                self.logger.info("Configuration validation passed")
            else:
                self.logger.warning(f"Configuration validation failed: {self.validation_errors}")
            
            return is_valid
            
        except Exception as e:
            self.logger.exception(f"Error validating configuration: {e}")
            return False
    
    async def _setup_configuration_sections(self):
        """Setup configuration section objects."""
        try:
            self.logger.info("Setting up configuration sections...")
            
            # Setup database configuration
            database_config_data = self.config_data.get("database", {})
            self.config_sections["database"] = DatabaseConfig(**database_config_data)
            
            # Setup exchange configuration
            exchange_config_data = self.config_data.get("exchange", {})
            self.config_sections["exchange"] = ExchangeConfig(**exchange_config_data)
            
            # Setup training configuration
            training_config_data = self.config_data.get("training", {})
            self.config_sections["training"] = ModelTrainingConfig(**training_config_data)
            
            # Setup risk configuration
            risk_config_data = self.config_data.get("risk", {})
            self.config_sections["risk"] = RiskConfig(**risk_config_data)
            
            self.logger.info("Configuration sections setup completed")
            
        except Exception as e:
            self.logger.exception(f"Error setting up configuration sections: {e}")
    
    async def _setup_hot_reload(self):
        """Setup hot reload functionality."""
        try:
            if not WATCHDOG_AVAILABLE:
                self.logger.warning("Watchdog not available, hot-reload disabled")
                return
            
            if not self.watcher:
                self.watcher = Observer()
                self.watcher.start()
            
            # Watch configuration directories
            for config_dir in self.config_directories:
                if os.path.exists(config_dir):
                    event_handler = ConfigurationWatcher(self)
                    self.watcher.schedule(event_handler, config_dir, recursive=True)
                    self.watched_files.add(config_dir)
                    self.logger.info(f"Watching configuration directory: {config_dir}")
                    
        except Exception as e:
            self.logger.exception(f"Error setting up hot-reload: {e}")
    
    async def _setup_encryption(self):
        """Setup encryption for sensitive configuration."""
        try:
            # In a real implementation, you would setup encryption keys here
            self.encryption_key = os.getenv("CONFIG_ENCRYPTION_KEY")
            if not self.encryption_key:
                self.logger.warning("No encryption key provided, encryption disabled")
                self.encryption_enabled = False
                
        except Exception as e:
            self.logger.exception(f"Error setting up encryption: {e}")
    
    async def _reload_configuration(self):
        """Reload configuration from files."""
        try:
            self.logger.info("🔄 Reloading configuration...")
            
            # Clear current configuration
            self.config_data.clear()
            self.config_sections.clear()
            
            # Reload configuration
            await self._load_configuration()
            
            # Re-validate and setup sections
            if await self._validate_configuration():
                await self._setup_configuration_sections()
                self.logger.info("✅ Configuration reloaded successfully")
            else:
                self.logger.error("❌ Configuration reload failed validation")
                
        except Exception as e:
            self.logger.exception(f"Error reloading configuration: {e}")
    
    def get_config(self, section: Optional[str] = None) -> Any:
        """Get configuration data."""
        try:
            if section:
                return self.config_sections.get(section)
            return self.config_data
            
        except Exception as e:
            self.logger.exception(f"Error getting configuration: {e}")
            return None
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update configuration section."""
        try:
            if section not in self.config_sections:
                self.logger.error(f"Unknown configuration section: {section}")
                return False
            
            # Update the section
            current_config = asdict(self.config_sections[section])
            current_config.update(updates)
            
            # Recreate the section with updated values
            if section == "database":
                self.config_sections[section] = DatabaseConfig(**current_config)
            elif section == "exchange":
                self.config_sections[section] = ExchangeConfig(**current_config)
            elif section == "training":
                self.config_sections[section] = ModelTrainingConfig(**current_config)
            elif section == "risk":
                self.config_sections[section] = RiskConfig(**current_config)
            
            self.logger.info(f"Updated configuration section: {section}")
            return True
            
        except Exception as e:
            self.logger.exception(f"Error updating configuration: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get configuration service status."""
        try:
            return {
                "is_initialized": self.is_initialized,
                "environment": self.environment,
                "config_files": self.config_files,
                "watched_files": list(self.watched_files),
                "validation_errors": self.validation_errors,
                "load_times": self.load_times,
                "last_load_time": self.last_load_time,
            }
            
        except Exception as e:
            self.logger.exception(f"Error getting status: {e}")
            return {}
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get configuration history."""
        try:
            history = self.config_history.copy()
            if limit:
                history = history[-limit:]
            return history
            
        except Exception as e:
            self.logger.exception(f"Error getting history: {e}")
            return []
    
    async def shutdown(self):
        """Shutdown the configuration service."""
        try:
            # Stop hot-reload watcher
            if self.watcher:
                self.watcher.stop()
                self.watcher.join()
            
            self.is_initialized = False
            self.logger.info("Configuration service shutdown completed")
            
        except Exception as e:
            self.logger.exception(f"Error during shutdown: {e}")


# Global configuration service instance
config_service: Optional[ConfigurationService] = None


def get_config_service() -> ConfigurationService:
    """Get the global configuration service instance."""
    global config_service
    if config_service is None:
        # Initialize with default configuration
        default_config = {
            "enable_hot_reload": True,
            "encryption_enabled": False,
        }
        config_service = ConfigurationService(default_config)
    return config_service

