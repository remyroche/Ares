"""
Comprehensive S/R Configuration Loader

This module loads and merges S/R configurations from multiple sources.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from dataclasses import dataclass, asdict

from src.config.config_sr import SRConfig, get_sr_config
from src.utils.logger import system_logger

logger = system_logger.getChild('SRComprehensiveConfigLoader')


@dataclass
class SRComprehensiveConfig:
    """Comprehensive S/R configuration combining all sources."""
    
    # From config_sr.py (optimizable parameters)
    sr_optimization: SRConfig
    
    # From sr_levels_config.yaml
    sr_levels_manager: Dict[str, Any]
    sr_trading_intelligence: Dict[str, Any]
    sr_breakout_predictor: Dict[str, Any]
    data_integration: Dict[str, Any]
    performance: Dict[str, Any]
    risk_management: Dict[str, Any]
    reporting: Dict[str, Any]
    integrations: Dict[str, Any]
    testing: Dict[str, Any]
    
    # Additional comprehensive integration config
    sr_comprehensive_integration: Dict[str, Any]


class SRComprehensiveConfigLoader:
    """Loads and manages comprehensive S/R configuration."""
    
    def __init__(self):
        self.logger = logger
        self.yaml_config_path = Path("config/sr_levels_config.yaml")
        self.config: Optional[SRComprehensiveConfig] = None
        
    def load_config(self) -> SRComprehensiveConfig:
        """Load comprehensive S/R configuration from all sources."""
        try:
            # Load optimizable parameters
            sr_optimization = get_sr_config()
            
            # Load YAML configuration
            yaml_config = self._load_yaml_config()
            
            # Create comprehensive configuration
            self.config = SRComprehensiveConfig(
                sr_optimization=sr_optimization,
                sr_levels_manager=yaml_config.get('sr_levels_manager', {}),
                sr_trading_intelligence=yaml_config.get('sr_trading_intelligence', {}),
                sr_breakout_predictor=yaml_config.get('sr_breakout_predictor', {}),
                data_integration=yaml_config.get('data_integration', {}),
                performance=yaml_config.get('performance', {}),
                risk_management=yaml_config.get('risk_management', {}),
                reporting=yaml_config.get('reporting', {}),
                integrations=yaml_config.get('integrations', {}),
                testing=yaml_config.get('testing', {}),
                sr_comprehensive_integration=self._get_comprehensive_integration_config()
            )
            
            self.logger.info("✅ Comprehensive S/R configuration loaded successfully")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load comprehensive S/R configuration: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load S/R configuration from YAML file."""
        try:
            if self.yaml_config_path.exists():
                with open(self.yaml_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self.logger.info(f"Loaded S/R configuration from {self.yaml_config_path}")
                    return config
            else:
                self.logger.warning(f"S/R configuration file not found: {self.yaml_config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading YAML configuration: {e}")
            return {}
    
    def _get_comprehensive_integration_config(self) -> Dict[str, Any]:
        """Get configuration for comprehensive S/R integration."""
        return {
            'enable_all_components': True,
            'use_ensemble_methods': True,
            'cache_ttl_seconds': 300,
            'parallel_processing': True,
            'max_workers': 4,
            'component_timeout_seconds': 30,
            'fallback_on_error': True,
            'log_performance_metrics': True,
            'save_intermediate_results': False
        }
    
    def _get_default_config(self) -> SRComprehensiveConfig:
        """Get default configuration as fallback."""
        return SRComprehensiveConfig(
            sr_optimization=get_sr_config(),
            sr_levels_manager={
                'storage_path': 'data/sr_levels',
                'max_levels': 50,
                'min_strength': 0.3,
                'proximity_threshold': 0.005
            },
            sr_trading_intelligence={
                'enable_real_time_updates': False,
                'update_interval_seconds': 60
            },
            sr_breakout_predictor={
                'enable_detailed_reporting': True,
                'report_directory': 'reports/sr_optimization'
            },
            data_integration={
                'symbol': 'BTCUSDT',
                'exchange': 'BINANCE',
                'timeframes': ['1m', '5m', '15m']
            },
            performance={
                'enable_level_caching': True,
                'cache_ttl_seconds': 300
            },
            risk_management={
                'max_risk_per_trade': 0.02,
                'default_stop_loss_pct': 0.02
            },
            reporting={
                'log_level': 'INFO',
                'enable_structured_logging': True
            },
            integrations={
                'exchange_api': {
                    'enable_rate_limiting': True,
                    'max_requests_per_second': 10
                }
            },
            testing={
                'enable_mock_data': False,
                'enable_backtesting': True
            },
            sr_comprehensive_integration=self._get_comprehensive_integration_config()
        )
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        if not self.config:
            self.load_config()
        
        config_dict = {
            'sr_optimization': asdict(self.config.sr_optimization),
            'sr_levels_manager': self.config.sr_levels_manager,
            'sr_trading_intelligence': self.config.sr_trading_intelligence,
            'sr_breakout_predictor': self.config.sr_breakout_predictor,
            'data_integration': self.config.data_integration,
            'performance': self.config.performance,
            'risk_management': self.config.risk_management,
            'reporting': self.config.reporting,
            'integrations': self.config.integrations,
            'testing': self.config.testing,
            'sr_comprehensive_integration': self.config.sr_comprehensive_integration
        }
        
        return config_dict
    
    def get_component_config(self, component: str) -> Dict[str, Any]:
        """Get configuration for a specific S/R component."""
        if not self.config:
            self.load_config()
        
        # Map component names to config attributes
        component_map = {
            'levels_manager': self.config.sr_levels_manager,
            'trading_intelligence': self.config.sr_trading_intelligence,
            'breakout_predictor': self.config.sr_breakout_predictor,
            'data_integration': self.config.data_integration,
            'performance': self.config.performance,
            'risk_management': self.config.risk_management,
            'reporting': self.config.reporting,
            'integrations': self.config.integrations,
            'testing': self.config.testing,
            'optimization': asdict(self.config.sr_optimization),
            'comprehensive_integration': self.config.sr_comprehensive_integration
        }
        
        return component_map.get(component, {})
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with new values."""
        try:
            if not self.config:
                self.load_config()
            
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                elif key in ['sr_optimization'] and hasattr(self.config.sr_optimization, key):
                    # Update optimizable parameters
                    for param_key, param_value in value.items():
                        if hasattr(self.config.sr_optimization, param_key):
                            setattr(self.config.sr_optimization, param_key, param_value)
            
            self.logger.info("S/R configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update S/R configuration: {e}")
            return False


# Singleton instance
_sr_config_loader: Optional[SRComprehensiveConfigLoader] = None


def get_sr_comprehensive_config_loader() -> SRComprehensiveConfigLoader:
    """Get singleton instance of S/R comprehensive config loader."""
    global _sr_config_loader
    if _sr_config_loader is None:
        _sr_config_loader = SRComprehensiveConfigLoader()
    return _sr_config_loader


def get_sr_comprehensive_config() -> SRComprehensiveConfig:
    """Get comprehensive S/R configuration."""
    loader = get_sr_comprehensive_config_loader()
    return loader.load_config()


def get_sr_comprehensive_config_dict() -> Dict[str, Any]:
    """Get comprehensive S/R configuration as dictionary."""
    loader = get_sr_comprehensive_config_loader()
    return loader.get_config_dict()