"""
Example: How to Access SR Levels Dictionary in Training Scripts

This example demonstrates how training scripts in pre_training and models_training
directories can access the SR levels dictionary with scores and metadata that was
saved by the SR clustering component.
"""

import asyncio
import logging
from typing import Any, Dict, List
from datetime import datetime

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger

logger = logging.getLogger(__name__)


class ExampleSRAccessStep(BaseStep):
    """
    Example step showing how to access SR levels dictionary in training scripts.
    """

    def __init__(self, step_name: str = "example_sr_access"):
        """Initialize the example step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('ExampleSRAccess')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the example step showing SR levels access.
        
        Args:
            config: Configuration dictionary containing symbol, exchange, etc.
            
        Returns:
            Execution result with SR levels information
        """
        self.logger.info('🔍 Starting SR levels access example')
        
        try:
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            
            # Method 1: Access via BaseStep._get_sr_levels() (recommended for training scripts)
            self.logger.info('📊 Method 1: Accessing SR levels via BaseStep._get_sr_levels()')
            sr_levels = self._get_sr_levels(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction
            )
            
            # Method 2: Access via Feature Bank (alternative method)
            self.logger.info('📊 Method 2: Accessing SR levels via Feature Bank')
            from src.feature_generation.core.feature_bank import get_global_feature_bank
            feature_bank = get_global_feature_bank()
            sr_levels_fb = feature_bank.get_sr_levels(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction
            )
            
            # Process SR levels data
            levels = sr_levels.get('levels', [])
            summary = sr_levels.get('summary', {})
            
            self.logger.info(f'📈 Found {len(levels)} SR levels')
            self.logger.info(f'📊 Summary: {summary}')
            
            # Example: Process support levels
            support_levels = [level for level in levels if level.get('type', '').lower() == 'support']
            resistance_levels = [level for level in levels if level.get('type', '').lower() == 'resistance']
            
            self.logger.info(f'🟢 Support levels: {len(support_levels)}')
            self.logger.info(f'🔴 Resistance levels: {len(resistance_levels)}')
            
            # Example: Access level details
            if levels:
                example_level = levels[0]
                self.logger.info(f'📋 Example level details:')
                self.logger.info(f'   - Price: {example_level.get("price", "N/A")}')
                self.logger.info(f'   - Type: {example_level.get("type", "N/A")}')
                self.logger.info(f'   - Strength: {example_level.get("strength", "N/A")}')
                self.logger.info(f'   - Confidence: {example_level.get("confidence", "N/A")}')
                self.logger.info(f'   - Touches: {example_level.get("touches", "N/A")}')
                self.logger.info(f'   - Cluster ID: {example_level.get("cluster_id", "N/A")}')
                
                # Access cluster information
                cluster_info = example_level.get('cluster_info', {})
                if cluster_info:
                    self.logger.info(f'   - Cluster size: {cluster_info.get("cluster_size", "N/A")}')
                    self.logger.info(f'   - Cluster type: {cluster_info.get("cluster_type", "N/A")}')
            
            # Example: Use SR levels for feature generation
            features = self._generate_sr_features(levels, config)
            
            # Save results
            result_artifact_path = self._save_artifact(
                {
                    'sr_levels': sr_levels,
                    'features': features,
                    'summary': summary,
                    'access_methods_tested': ['BaseStep._get_sr_levels()', 'FeatureBank.get_sr_levels()']
                },
                'sr_levels_access_example',
                'data',
                metadata={
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'levels_processed': len(levels),
                    'created_at': datetime.now().isoformat()
                }
            )
            
            return {
                'success': True,
                'artifacts': [result_artifact_path],
                'metrics': {
                    'total_levels': len(levels),
                    'support_levels': len(support_levels),
                    'resistance_levels': len(resistance_levels),
                    'features_generated': len(features) if features else 0
                },
                'sr_levels_summary': summary
            }
            
        except Exception as e:
            self.logger.error(f'❌ SR levels access example failed: {e}')
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }

    def _generate_sr_features(self, sr_levels: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example: Generate features from SR levels.
        
        This shows how training scripts can use SR levels data to create features.
        """
        try:
            if not sr_levels:
                return {}
            
            features = {}
            
            # Feature 1: Count of levels by type
            support_count = len([l for l in sr_levels if l.get('type', '').lower() == 'support'])
            resistance_count = len([l for l in sr_levels if l.get('type', '').lower() == 'resistance'])
            
            features['sr_support_count'] = support_count
            features['sr_resistance_count'] = resistance_count
            features['sr_total_count'] = len(sr_levels)
            
            # Feature 2: Average strength by type
            support_strengths = [l.get('strength', 0) for l in sr_levels if l.get('type', '').lower() == 'support']
            resistance_strengths = [l.get('strength', 0) for l in sr_levels if l.get('type', '').lower() == 'resistance']
            
            features['sr_avg_support_strength'] = sum(support_strengths) / len(support_strengths) if support_strengths else 0
            features['sr_avg_resistance_strength'] = sum(resistance_strengths) / len(resistance_strengths) if resistance_strengths else 0
            
            # Feature 3: Cluster statistics
            cluster_sizes = [l.get('cluster_info', {}).get('cluster_size', 0) for l in sr_levels]
            features['sr_avg_cluster_size'] = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
            features['sr_max_cluster_size'] = max(cluster_sizes) if cluster_sizes else 0
            
            # Feature 4: Price range of levels
            prices = [l.get('price', 0) for l in sr_levels if l.get('price', 0) > 0]
            if prices:
                features['sr_min_price'] = min(prices)
                features['sr_max_price'] = max(prices)
                features['sr_price_range'] = max(prices) - min(prices)
            
            self.logger.info(f'Generated {len(features)} SR-based features')
            return features
            
        except Exception as e:
            self.logger.error(f'Failed to generate SR features: {e}')
            return {}


# Example usage in a training script
async def example_usage():
    """Example of how to use SR levels in a training script."""
    
    # Create step instance
    step = ExampleSRAccessStep()
    
    # Configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'longs',
        'execution_mode': 'light'
    }
    
    # Execute step
    result = await step.run(config)
    
    if result['success']:
        print("✅ SR levels access example completed successfully")
        print(f"📊 Processed {result['metrics']['total_levels']} SR levels")
        print(f"🟢 Support levels: {result['metrics']['support_levels']}")
        print(f"🔴 Resistance levels: {result['metrics']['resistance_levels']}")
    else:
        print(f"❌ SR levels access example failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(example_usage())