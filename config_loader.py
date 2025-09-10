#!/usr/bin/env python3
"""
Configuration Loader

This module provides utilities for loading and managing configuration templates.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(environment: str = 'development') -> Dict[str, Any]:
    """Load configuration for specified environment."""
    config_file = Path(f'configs/{environment}_config.json')
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

def get_available_configs() -> list:
    """Get list of available configuration environments."""
    configs_dir = Path('configs')
    if not configs_dir.exists():
        return []
    
    config_files = list(configs_dir.glob('*_config.json'))
    return [f.stem.replace('_config', '') for f in config_files]

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure."""
    required_keys = [
        'environment', 'symbol', 'exchange', 'timeframe',
        'model_training_config', 'evaluation_config',
        'feature_engineering_config', 'data_quality_config',
        'performance_config'
    ]
    
    for key in required_keys:
        if key not in config:
            print(f"❌ Missing required configuration key: {key}")
            return False
    
    print("✅ Configuration validation passed")
    return True

if __name__ == "__main__":
    # Example usage
    print("Available configurations:", get_available_configs())
    
    for env in ['development', 'testing', 'production']:
        try:
            config = load_config(env)
            print(f"\n{env.upper()} Configuration:")
            print(f"  Environment: {config['environment']}")
            print(f"  Symbol: {config['symbol']}")
            print(f"  Max Workers: {config['max_workers']}")
            print(f"  Memory Limit: {config['memory_limit']}")
            validate_config(config)
        except FileNotFoundError:
            print(f"❌ Configuration not found for environment: {env}")
