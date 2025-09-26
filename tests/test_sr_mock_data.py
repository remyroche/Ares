"""
Test suite for SR Levels Mock Data functionality.

This module tests the mock data generator, configuration, and integration components.
"""

import unittest
import tempfile
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import the modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.sr_mock_data_generator import SRMockDataGenerator, SRLevel, create_mock_data_from_config
from config.sr_mock_data_config import SRMockDataConfig, load_sr_mock_data_config
from integration.sr_mock_data_integration import SRMockDataIntegration, SRMockDataManager


class TestSRMockDataGenerator(unittest.TestCase):
    """Test cases for SRMockDataGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = SRMockDataGenerator(seed=42)
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.seed, 42)
        self.assertIsNotNone(self.generator)
    
    def test_generate_market_data(self):
        """Test market data generation."""
        market_data = self.generator.generate_market_data(
            symbol="ETHUSDT",
            data_points=100,
            start_price=3000.0
        )
        
        # Check basic properties
        self.assertIsInstance(market_data, pd.DataFrame)
        self.assertEqual(len(market_data), 100)
        self.assertIn('open', market_data.columns)
        self.assertIn('high', market_data.columns)
        self.assertIn('low', market_data.columns)
        self.assertIn('close', market_data.columns)
        self.assertIn('volume', market_data.columns)
        self.assertIn('vwap', market_data.columns)
        
        # Check OHLC consistency
        for _, row in market_data.iterrows():
            self.assertGreaterEqual(row['high'], row['open'])
            self.assertGreaterEqual(row['high'], row['close'])
            self.assertLessEqual(row['low'], row['open'])
            self.assertLessEqual(row['low'], row['close'])
    
    def test_generate_sr_levels(self):
        """Test SR levels generation."""
        market_data = self.generator.generate_market_data(data_points=100)
        sr_levels = self.generator.generate_sr_levels(market_data, num_levels=10)
        
        # Check basic properties
        self.assertIsInstance(sr_levels, list)
        self.assertEqual(len(sr_levels), 10)
        
        for level in sr_levels:
            self.assertIsInstance(level, SRLevel)
            self.assertIn(level.level_type, ['support', 'resistance'])
            self.assertGreater(level.strength, 0)
            self.assertLessEqual(level.strength, 1)
            self.assertGreaterEqual(level.touch_count, 2)
    
    def test_generate_trading_scenarios(self):
        """Test trading scenarios generation."""
        market_data = self.generator.generate_market_data(data_points=100)
        sr_levels = self.generator.generate_sr_levels(market_data, num_levels=5)
        scenarios = self.generator.generate_trading_scenarios(market_data, sr_levels, num_scenarios=10)
        
        # Check basic properties
        self.assertIsInstance(scenarios, list)
        self.assertEqual(len(scenarios), 10)
        
        for scenario in scenarios:
            self.assertIn('scenario_id', scenario)
            self.assertIn('scenario_type', scenario)
            self.assertIn('confidence', scenario)
            self.assertIn('risk_reward_ratio', scenario)
            self.assertIn('sr_level', scenario)
    
    def test_generate_performance_metrics(self):
        """Test performance metrics generation."""
        market_data = self.generator.generate_market_data(data_points=100)
        sr_levels = self.generator.generate_sr_levels(market_data, num_levels=5)
        scenarios = self.generator.generate_trading_scenarios(market_data, sr_levels, num_scenarios=10)
        metrics = self.generator.generate_performance_metrics(scenarios)
        
        # Check basic properties
        self.assertIsInstance(metrics, dict)
        self.assertIn('success_rate', metrics)
        self.assertIn('total_pnl', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
    
    def test_generate_complete_mock_dataset(self):
        """Test complete mock dataset generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_data = self.generator.generate_complete_mock_dataset(
                data_points=100,
                num_sr_levels=5,
                num_scenarios=10,
                output_dir=temp_dir
            )
            
            # Check return structure
            self.assertIsInstance(mock_data, dict)
            self.assertIn('market_data', mock_data)
            self.assertIn('sr_levels', mock_data)
            self.assertIn('scenarios', mock_data)
            self.assertIn('metrics', mock_data)
            
            # Check files were created
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "market_data.csv")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "sr_levels.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "trading_scenarios.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "performance_metrics.json")))


class TestSRMockDataConfig(unittest.TestCase):
    """Test cases for SRMockDataConfig."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.config_content = """
testing:
  enable_mock_data: true
  mock_data_points: 1000
  mock_data_seed: 42
  mock_data_output_dir: "data/mock_sr_data"
  mock_data_validation: true
  mock_data_export_format: "json"
  mock_data_retention_days: 30

sr_levels_manager:
  max_levels: 20
  min_strength: 0.3
  proximity_threshold: 0.005

data_integration:
  symbol: "ETHUSDT"
  exchange: "BINANCE"
  timeframes: ["1m", "5m", "15m"]
"""
        self.temp_config.write(self.config_content)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config.name)
    
    def test_config_loading(self):
        """Test configuration loading."""
        config = SRMockDataConfig(self.temp_config.name)
        
        self.assertTrue(config.is_mock_data_enabled())
        self.assertEqual(config.get_mock_data_points(), 1000)
        self.assertEqual(config.get_mock_data_seed(), 42)
        self.assertEqual(config.get_mock_data_output_dir(), "data/mock_sr_data")
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = SRMockDataConfig(self.temp_config.name)
        self.assertTrue(config.validate_mock_data_config())
    
    def test_mock_data_generator_creation(self):
        """Test mock data generator creation."""
        config = SRMockDataConfig(self.temp_config.name)
        generator = config.create_mock_data_generator()
        
        self.assertIsInstance(generator, SRMockDataGenerator)
        self.assertEqual(generator.seed, 42)
    
    def test_mock_data_generation(self):
        """Test mock data generation from config."""
        config = SRMockDataConfig(self.temp_config.name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update output directory
            config.config['testing']['mock_data_output_dir'] = temp_dir
            
            mock_data = config.generate_mock_data()
            
            self.assertIsInstance(mock_data, dict)
            self.assertIn('market_data', mock_data)
            self.assertIn('sr_levels', mock_data)
            self.assertIn('scenarios', mock_data)
            self.assertIn('metrics', mock_data)


class TestSRMockDataIntegration(unittest.TestCase):
    """Test cases for SRMockDataIntegration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.config_content = """
testing:
  enable_mock_data: true
  mock_data_points: 100
  mock_data_seed: 42
  mock_data_output_dir: "data/mock_sr_data"

sr_levels_manager:
  max_levels: 5

data_integration:
  symbol: "ETHUSDT"
  exchange: "BINANCE"
"""
        self.temp_config.write(self.config_content)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config.name)
    
    def test_integration_initialization(self):
        """Test integration initialization."""
        integration = SRMockDataIntegration(self.temp_config.name)
        
        self.assertIsNotNone(integration.config)
        self.assertIsNotNone(integration.logger)
    
    def test_mock_data_initialization(self):
        """Test mock data initialization."""
        integration = SRMockDataIntegration(self.temp_config.name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update output directory
            integration.config.config['testing']['mock_data_output_dir'] = temp_dir
            
            result = integration.initialize_mock_data()
            self.assertTrue(result)
            self.assertTrue(integration.is_mock_data_available())
    
    def test_data_access_methods(self):
        """Test data access methods."""
        integration = SRMockDataIntegration(self.temp_config.name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update output directory
            integration.config.config['testing']['mock_data_output_dir'] = temp_dir
            
            integration.initialize_mock_data()
            
            # Test data access
            market_data = integration.get_market_data()
            self.assertIsInstance(market_data, pd.DataFrame)
            
            sr_levels = integration.get_sr_levels()
            self.assertIsInstance(sr_levels, list)
            
            scenarios = integration.get_trading_scenarios()
            self.assertIsInstance(scenarios, list)
            
            metrics = integration.get_performance_metrics()
            self.assertIsInstance(metrics, dict)
    
    def test_data_summary(self):
        """Test data summary generation."""
        integration = SRMockDataIntegration(self.temp_config.name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update output directory
            integration.config.config['testing']['mock_data_output_dir'] = temp_dir
            
            integration.initialize_mock_data()
            summary = integration.get_mock_data_summary()
            
            self.assertIn('available', summary)
            self.assertIn('market_data_points', summary)
            self.assertIn('sr_levels_count', summary)
            self.assertIn('scenarios_count', summary)
    
    def test_data_export(self):
        """Test data export functionality."""
        integration = SRMockDataIntegration(self.temp_config.name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update output directory
            integration.config.config['testing']['mock_data_output_dir'] = temp_dir
            
            integration.initialize_mock_data()
            
            export_dir = os.path.join(temp_dir, "export")
            result = integration.export_mock_data(export_dir)
            
            self.assertTrue(result)
            self.assertTrue(os.path.exists(os.path.join(export_dir, "market_data.csv")))
            self.assertTrue(os.path.exists(os.path.join(export_dir, "sr_levels.json")))
            self.assertTrue(os.path.exists(os.path.join(export_dir, "trading_scenarios.json")))
            self.assertTrue(os.path.exists(os.path.join(export_dir, "performance_metrics.json")))


class TestSRMockDataManager(unittest.TestCase):
    """Test cases for SRMockDataManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        self.config_content = """
testing:
  enable_mock_data: true
  mock_data_points: 100
  mock_data_seed: 42
  mock_data_output_dir: "data/mock_sr_data"

sr_levels_manager:
  max_levels: 5

data_integration:
  symbol: "ETHUSDT"
  exchange: "BINANCE"
"""
        self.temp_config.write(self.config_content)
        self.temp_config.close()
    
    def tearDown(self):
        """Clean up test fixtures."""
        os.unlink(self.temp_config.name)
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = SRMockDataManager(self.temp_config.name)
        
        self.assertIsNotNone(manager.integration)
        self.assertIsNotNone(manager.logger)
    
    def test_service_lifecycle(self):
        """Test service lifecycle."""
        manager = SRMockDataManager(self.temp_config.name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update output directory
            manager.integration.config.config['testing']['mock_data_output_dir'] = temp_dir
            
            # Start service
            start_result = manager.start_mock_data_service()
            self.assertTrue(start_result)
            
            # Check status
            status = manager.get_service_status()
            self.assertTrue(status['running'])
            
            # Stop service
            stop_result = manager.stop_mock_data_service()
            self.assertTrue(stop_result)
    
    def test_data_export(self):
        """Test data export through manager."""
        manager = SRMockDataManager(self.temp_config.name)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update output directory
            manager.integration.config.config['testing']['mock_data_output_dir'] = temp_dir
            
            manager.start_mock_data_service()
            
            export_dir = os.path.join(temp_dir, "export")
            result = manager.export_all_mock_data(export_dir)
            
            self.assertTrue(result)
            self.assertTrue(os.path.exists(os.path.join(export_dir, "market_data.csv")))
            
            manager.stop_mock_data_service()


class TestMockDataFromConfig(unittest.TestCase):
    """Test cases for create_mock_data_from_config function."""
    
    def test_create_mock_data_from_config(self):
        """Test creating mock data from configuration."""
        # Create a temporary config file
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        config_content = """
testing:
  enable_mock_data: true
  mock_data_points: 100
  mock_data_seed: 42
  mock_data_output_dir: "data/mock_sr_data"

sr_levels_manager:
  max_levels: 5

data_integration:
  symbol: "ETHUSDT"
  exchange: "BINANCE"
"""
        temp_config.write(config_content)
        temp_config.close()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Update config to use temp directory
                config = SRMockDataConfig(temp_config.name)
                config.config['testing']['mock_data_output_dir'] = temp_dir
                
                # Save updated config
                with open(temp_config.name, 'w') as f:
                    import yaml
                    yaml.dump(config.config, f, default_flow_style=False, indent=2)
                
                mock_data = create_mock_data_from_config(config.config)
                
                self.assertIsInstance(mock_data, dict)
                self.assertIn('market_data', mock_data)
                self.assertIn('sr_levels', mock_data)
                self.assertIn('scenarios', mock_data)
                self.assertIn('metrics', mock_data)
        
        finally:
            os.unlink(temp_config.name)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)