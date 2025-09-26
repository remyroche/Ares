"""
SR Levels Mock Data Configuration

This module handles the configuration and initialization of mock data for the SR levels system.
It integrates with the existing configuration system and provides proper mock data functionality.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from ..utils.sr_mock_data_generator import SRMockDataGenerator, create_mock_data_from_config


class SRMockDataConfig:
    """Configuration handler for SR levels mock data."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the mock data configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
    def _find_config_file(self) -> str:
        """Find the appropriate configuration file."""
        possible_paths = [
            "config/sr_levels_config.yaml",
            "config/features/sr_levels_config.yaml",
            "sr_levels_config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find SR levels configuration file")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def is_mock_data_enabled(self) -> bool:
        """Check if mock data is enabled in configuration."""
        return self.config.get('testing', {}).get('enable_mock_data', False)
    
    def get_mock_data_points(self) -> int:
        """Get the number of mock data points to generate."""
        return self.config.get('testing', {}).get('mock_data_points', 1000)
    
    def get_mock_data_seed(self) -> int:
        """Get the mock data seed for reproducibility."""
        return self.config.get('testing', {}).get('mock_data_seed', 42)
    
    def get_mock_data_output_dir(self) -> str:
        """Get the output directory for mock data."""
        return self.config.get('testing', {}).get('mock_data_output_dir', 'data/mock_sr_data')
    
    def get_sr_levels_config(self) -> Dict[str, Any]:
        """Get SR levels specific configuration."""
        return self.config.get('sr_levels_manager', {})
    
    def get_data_integration_config(self) -> Dict[str, Any]:
        """Get data integration configuration."""
        return self.config.get('data_integration', {})
    
    def create_mock_data_generator(self) -> SRMockDataGenerator:
        """Create a mock data generator with configured settings."""
        if not self.is_mock_data_enabled():
            raise ValueError("Mock data is disabled in configuration")
        
        seed = self.get_mock_data_seed()
        return SRMockDataGenerator(seed=seed)
    
    def generate_mock_data(self) -> Dict[str, Any]:
        """
        Generate mock data based on current configuration.
        
        Returns:
            Dictionary containing generated mock data
        """
        if not self.is_mock_data_enabled():
            self.logger.warning("Mock data is disabled, returning empty dataset")
            return {}
        
        try:
            # Create generator with configured settings
            generator = self.create_mock_data_generator()
            
            # Get configuration parameters
            data_points = self.get_mock_data_points()
            output_dir = self.get_mock_data_output_dir()
            
            # Get SR levels configuration
            sr_config = self.get_sr_levels_config()
            num_sr_levels = sr_config.get('max_levels', 20)
            
            # Generate mock data
            self.logger.info(f"Generating mock data with {data_points} points and {num_sr_levels} SR levels")
            mock_data = generator.generate_complete_mock_dataset(
                data_points=data_points,
                num_sr_levels=num_sr_levels,
                num_scenarios=50,
                output_dir=output_dir
            )
            
            self.logger.info("Mock data generation completed successfully")
            return mock_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate mock data: {e}")
            raise
    
    def validate_mock_data_config(self) -> bool:
        """
        Validate the mock data configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check if mock data is enabled
            if not self.is_mock_data_enabled():
                self.logger.info("Mock data is disabled, validation skipped")
                return True
            
            # Validate data points
            data_points = self.get_mock_data_points()
            if data_points < 100:
                self.logger.warning(f"Mock data points ({data_points}) is very low, consider increasing")
            
            if data_points > 10000:
                self.logger.warning(f"Mock data points ({data_points}) is very high, may impact performance")
            
            # Validate seed
            seed = self.get_mock_data_seed()
            if not isinstance(seed, int):
                self.logger.error("Mock data seed must be an integer")
                return False
            
            # Validate output directory
            output_dir = self.get_mock_data_output_dir()
            if not output_dir:
                self.logger.error("Mock data output directory is not specified")
                return False
            
            self.logger.info("Mock data configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Mock data configuration validation failed: {e}")
            return False
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        try:
            # Update configuration
            for key, value in updates.items():
                if key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            
            # Save updated configuration
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            raise
    
    def get_mock_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of mock data configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'enabled': self.is_mock_data_enabled(),
            'data_points': self.get_mock_data_points(),
            'seed': self.get_mock_data_seed(),
            'output_dir': self.get_mock_data_output_dir(),
            'config_file': self.config_path,
            'valid': self.validate_mock_data_config()
        }


def load_sr_mock_data_config(config_path: Optional[str] = None) -> SRMockDataConfig:
    """
    Load SR mock data configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SRMockDataConfig instance
    """
    return SRMockDataConfig(config_path)


def create_mock_data_from_sr_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create mock data from SR configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing generated mock data
    """
    config = load_sr_mock_data_config(config_path)
    return config.generate_mock_data()


if __name__ == "__main__":
    # Example usage
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load configuration
        config = load_sr_mock_data_config()
        
        # Print configuration summary
        summary = config.get_mock_data_summary()
        print("Mock Data Configuration Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Generate mock data if enabled
        if config.is_mock_data_enabled():
            print("\nGenerating mock data...")
            mock_data = config.generate_mock_data()
            print(f"Generated mock data with {len(mock_data.get('market_data', []))} market data points")
        else:
            print("\nMock data is disabled in configuration")
            
    except Exception as e:
        print(f"Error: {e}")