"""
SR Levels Mock Data Integration

This module integrates the mock data generator with the existing SR levels system,
providing seamless mock data functionality for testing and development.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from ..config.sr_mock_data_config import SRMockDataConfig
from ..utils.sr_mock_data_generator import SRMockDataGenerator, SRLevel


class SRMockDataIntegration:
    """Integration class for SR levels mock data functionality."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the mock data integration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = SRMockDataConfig(config_path)
        self.logger = logging.getLogger(__name__)
        self.mock_data = None
        
    def initialize_mock_data(self) -> bool:
        """
        Initialize mock data if enabled in configuration.
        
        Returns:
            True if mock data was initialized successfully, False otherwise
        """
        try:
            if not self.config.is_mock_data_enabled():
                self.logger.info("Mock data is disabled in configuration")
                return False
            
            # Validate configuration
            if not self.config.validate_mock_data_config():
                self.logger.error("Mock data configuration validation failed")
                return False
            
            # Generate mock data
            self.logger.info("Generating mock data...")
            self.mock_data = self.config.generate_mock_data()
            
            self.logger.info("Mock data initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize mock data: {e}")
            return False
    
    def get_market_data(self) -> Optional[pd.DataFrame]:
        """
        Get mock market data.
        
        Returns:
            DataFrame with market data or None if not available
        """
        if self.mock_data is None:
            self.logger.warning("Mock data not initialized")
            return None
        
        return self.mock_data.get('market_data')
    
    def get_sr_levels(self) -> Optional[List[SRLevel]]:
        """
        Get mock SR levels.
        
        Returns:
            List of SRLevel objects or None if not available
        """
        if self.mock_data is None:
            self.logger.warning("Mock data not initialized")
            return None
        
        return self.mock_data.get('sr_levels')
    
    def get_trading_scenarios(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get mock trading scenarios.
        
        Returns:
            List of trading scenario dictionaries or None if not available
        """
        if self.mock_data is None:
            self.logger.warning("Mock data not initialized")
            return None
        
        return self.mock_data.get('scenarios')
    
    def get_performance_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get mock performance metrics.
        
        Returns:
            Dictionary of performance metrics or None if not available
        """
        if self.mock_data is None:
            self.logger.warning("Mock data not initialized")
            return None
        
        return self.mock_data.get('metrics')
    
    def is_mock_data_available(self) -> bool:
        """
        Check if mock data is available.
        
        Returns:
            True if mock data is available, False otherwise
        """
        return self.mock_data is not None
    
    def get_mock_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available mock data.
        
        Returns:
            Dictionary with mock data summary
        """
        if not self.is_mock_data_available():
            return {'available': False, 'message': 'Mock data not initialized'}
        
        summary = {
            'available': True,
            'market_data_points': len(self.mock_data.get('market_data', [])),
            'sr_levels_count': len(self.mock_data.get('sr_levels', [])),
            'scenarios_count': len(self.mock_data.get('scenarios', [])),
            'has_metrics': 'metrics' in self.mock_data,
            'generated_at': datetime.now().isoformat()
        }
        
        return summary
    
    def export_mock_data(self, output_dir: str) -> bool:
        """
        Export mock data to files.
        
        Args:
            output_dir: Directory to export mock data
            
        Returns:
            True if export was successful, False otherwise
        """
        try:
            if not self.is_mock_data_available():
                self.logger.error("No mock data available to export")
                return False
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Export market data
            market_data = self.get_market_data()
            if market_data is not None:
                market_data.to_csv(os.path.join(output_dir, "market_data.csv"))
                self.logger.info(f"Market data exported to {output_dir}/market_data.csv")
            
            # Export SR levels
            sr_levels = self.get_sr_levels()
            if sr_levels is not None:
                sr_levels_data = []
                for level in sr_levels:
                    sr_levels_data.append({
                        'price': level.price,
                        'level_type': level.level_type,
                        'strength': level.strength,
                        'touch_count': level.touch_count,
                        'first_touch': level.first_touch.isoformat(),
                        'last_touch': level.last_touch.isoformat(),
                        'bounce_rate': level.bounce_rate,
                        'isolation_score': level.isolation_score,
                        'volume_at_level': level.volume_at_level,
                        'age_days': level.age_days
                    })
                
                with open(os.path.join(output_dir, "sr_levels.json"), 'w') as f:
                    json.dump(sr_levels_data, f, indent=2)
                self.logger.info(f"SR levels exported to {output_dir}/sr_levels.json")
            
            # Export scenarios
            scenarios = self.get_trading_scenarios()
            if scenarios is not None:
                with open(os.path.join(output_dir, "trading_scenarios.json"), 'w') as f:
                    json.dump(scenarios, f, indent=2, default=str)
                self.logger.info(f"Trading scenarios exported to {output_dir}/trading_scenarios.json")
            
            # Export metrics
            metrics = self.get_performance_metrics()
            if metrics is not None:
                with open(os.path.join(output_dir, "performance_metrics.json"), 'w') as f:
                    json.dump(metrics, f, indent=2)
                self.logger.info(f"Performance metrics exported to {output_dir}/performance_metrics.json")
            
            self.logger.info(f"Mock data exported successfully to {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export mock data: {e}")
            return False
    
    def reload_mock_data(self) -> bool:
        """
        Reload mock data from configuration.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            self.logger.info("Reloading mock data...")
            self.mock_data = None
            return self.initialize_mock_data()
            
        except Exception as e:
            self.logger.error(f"Failed to reload mock data: {e}")
            return False
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """
        Get information about the mock data configuration.
        
        Returns:
            Dictionary with configuration information
        """
        return self.config.get_mock_data_summary()


class SRMockDataManager:
    """Manager class for SR levels mock data operations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the mock data manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.integration = SRMockDataIntegration(config_path)
        self.logger = logging.getLogger(__name__)
    
    def start_mock_data_service(self) -> bool:
        """
        Start the mock data service.
        
        Returns:
            True if service started successfully, False otherwise
        """
        try:
            self.logger.info("Starting mock data service...")
            
            # Initialize mock data
            if not self.integration.initialize_mock_data():
                self.logger.error("Failed to initialize mock data")
                return False
            
            self.logger.info("Mock data service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start mock data service: {e}")
            return False
    
    def stop_mock_data_service(self) -> bool:
        """
        Stop the mock data service.
        
        Returns:
            True if service stopped successfully, False otherwise
        """
        try:
            self.logger.info("Stopping mock data service...")
            self.integration.mock_data = None
            self.logger.info("Mock data service stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop mock data service: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get the status of the mock data service.
        
        Returns:
            Dictionary with service status
        """
        return {
            'running': self.integration.is_mock_data_available(),
            'config_summary': self.integration.get_configuration_info(),
            'data_summary': self.integration.get_mock_data_summary()
        }
    
    def export_all_mock_data(self, output_dir: str) -> bool:
        """
        Export all mock data to files.
        
        Args:
            output_dir: Directory to export mock data
            
        Returns:
            True if export was successful, False otherwise
        """
        return self.integration.export_mock_data(output_dir)


def create_sr_mock_data_manager(config_path: Optional[str] = None) -> SRMockDataManager:
    """
    Create a mock data manager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SRMockDataManager instance
    """
    return SRMockDataManager(config_path)


if __name__ == "__main__":
    # Example usage
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create mock data manager
        manager = create_sr_mock_data_manager()
        
        # Start service
        if manager.start_mock_data_service():
            print("Mock data service started successfully")
            
            # Get service status
            status = manager.get_service_status()
            print(f"Service status: {status}")
            
            # Export mock data
            if manager.export_all_mock_data("data/exported_mock_data"):
                print("Mock data exported successfully")
            
            # Stop service
            manager.stop_mock_data_service()
            print("Mock data service stopped")
        else:
            print("Failed to start mock data service")
            
    except Exception as e:
        print(f"Error: {e}")