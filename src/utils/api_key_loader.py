"""
API Key Loader

Utility module for loading API keys from secret/api_keys file.
Supports live and testnet keys for different exchanges.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Any
from src.utils.logger import system_logger

logger = system_logger.getChild("APIKeyLoader")

# Default path to API keys file
DEFAULT_API_KEYS_PATH = Path(__file__).parent.parent.parent / "secret" / "api_keys.json"


class APIKeyLoader:
    """Loads API keys from secret/api_keys.json file."""
    
    def __init__(self, api_keys_path: Optional[Path] = None):
        """
        Initialize API key loader.
        
        Args:
            api_keys_path: Optional path to API keys file. Defaults to secret/api_keys.json
        """
        self.api_keys_path = api_keys_path or DEFAULT_API_KEYS_PATH
        self._keys_cache: Optional[Dict[str, Any]] = None
        logger.info(f"API key loader initialized with path: {self.api_keys_path}")
    
    def _load_keys_file(self) -> Dict[str, Any]:
        """Load API keys from JSON file."""
        if self._keys_cache is not None:
            return self._keys_cache
        
        if not self.api_keys_path.exists():
            logger.warning(f"API keys file not found at {self.api_keys_path}")
            logger.warning("Please create secret/api_keys.json with your API keys")
            return {}
        
        try:
            with open(self.api_keys_path, 'r') as f:
                keys = json.load(f)
                self._keys_cache = keys
                logger.info(f"✅ Loaded API keys from {self.api_keys_path}")
                return keys
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in API keys file: {e}")
            return {}
        except Exception as e:
            logger.error(f"❌ Error loading API keys file: {e}")
            return {}
    
    def get_keys(
        self,
        exchange: str,
        use_live: bool = False
    ) -> Dict[str, Optional[str]]:
        """
        Get API keys for an exchange.
        
        Args:
            exchange: Exchange name (e.g., 'binance', 'okx', 'gateio')
            use_live: If True, return live keys. If False, return testnet keys.
        
        Returns:
            Dictionary with 'api_key', 'api_secret', and 'password' (if available)
        """
        keys_data = self._load_keys_file()
        
        # Normalize exchange name
        exchange = exchange.lower()
        
        # Determine which key set to use
        key_set = "live" if use_live else "testnet"
        
        # Get exchange-specific keys
        exchange_keys = keys_data.get(exchange, {})
        key_set_data = exchange_keys.get(key_set, {})
        
        result = {
            "api_key": key_set_data.get("api_key"),
            "api_secret": key_set_data.get("api_secret"),
            "password": key_set_data.get("password"),  # Optional, for exchanges like OKX
        }
        
        if not result["api_key"] or not result["api_secret"]:
            logger.warning(
                f"⚠️ Missing API keys for {exchange} ({key_set}). "
                f"Please ensure secret/api_keys.json contains {exchange}.{key_set}.api_key and api_secret"
            )
        
        return result
    
    def get_keys_for_exchange(
        self,
        exchange: str,
        use_live: bool = False
    ) -> tuple[str, str, Optional[str]]:
        """
        Get API keys for an exchange as a tuple.
        
        Args:
            exchange: Exchange name (e.g., 'binance', 'okx', 'gateio')
            use_live: If True, return live keys. If False, return testnet keys.
        
        Returns:
            Tuple of (api_key, api_secret, password)
        """
        keys = self.get_keys(exchange, use_live)
        return (
            keys.get("api_key") or "",
            keys.get("api_secret") or "",
            keys.get("password")
        )
    
    def reload(self) -> None:
        """Reload API keys from file (clear cache)."""
        self._keys_cache = None
        logger.info("API keys cache cleared, will reload on next access")


# Global instance
_key_loader: Optional[APIKeyLoader] = None


def get_key_loader(api_keys_path: Optional[Path] = None) -> APIKeyLoader:
    """Get or create global API key loader instance."""
    global _key_loader
    if _key_loader is None:
        _key_loader = APIKeyLoader(api_keys_path)
    return _key_loader


def get_api_keys(exchange: str, use_live: bool = False) -> Dict[str, Optional[str]]:
    """
    Convenience function to get API keys for an exchange.
    
    Args:
        exchange: Exchange name (e.g., 'binance', 'okx', 'gateio')
        use_live: If True, return live keys. If False, return testnet keys.
    
    Returns:
        Dictionary with 'api_key', 'api_secret', and 'password' (if available)
    """
    return get_key_loader().get_keys(exchange, use_live)


def get_api_keys_tuple(exchange: str, use_live: bool = False) -> tuple[str, str, Optional[str]]:
    """
    Convenience function to get API keys for an exchange as a tuple.
    
    Args:
        exchange: Exchange name (e.g., 'binance', 'okx', 'gateio')
        use_live: If True, return live keys. If False, return testnet keys.
    
    Returns:
        Tuple of (api_key, api_secret, password)
    """
    return get_key_loader().get_keys_for_exchange(exchange, use_live)
