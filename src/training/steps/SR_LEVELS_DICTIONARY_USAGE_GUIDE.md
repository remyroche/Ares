# SR Levels Dictionary Usage Guide

## Overview

The SR levels dictionary is a comprehensive data structure that contains Support/Resistance levels with their scores, metadata, and cluster information. It is automatically created and saved by the SR clustering component and made available to all training scripts in `steps/training/pre_training` and `models_training/` directories.

## Access Methods

### 1. Via BaseStep (Recommended for Training Scripts)

All training scripts that inherit from `BaseStep` can access SR levels using the `_get_sr_levels()` method:

```python
class MyTrainingStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Get SR levels for specific symbol/exchange/timeframe
        sr_levels = self._get_sr_levels(
            symbol=config.get('symbol'),
            exchange=config.get('exchange'),
            timeframe=config.get('timeframe'),
            direction=config.get('direction')
        )
        
        # Process the levels
        levels = sr_levels.get('levels', [])
        summary = sr_levels.get('summary', {})
```

### 2. Via Feature Bank

Access SR levels through the global feature bank:

```python
from src.feature_generation.core.feature_bank import get_global_feature_bank

feature_bank = get_global_feature_bank()
sr_levels = feature_bank.get_sr_levels(
    symbol='ETHUSDT',
    exchange='binance',
    timeframe='15m',
    direction='longs'
)
```

### 3. Via Artifact Manager (Direct Access)

Direct access through the artifact manager:

```python
from src.utils.artifact_manager import ArtifactManager

artifact_manager = ArtifactManager(config={})
sr_levels = artifact_manager.get_artifact(
    artifact_name='sr_levels_dictionary',
    artifact_type='data'
)
```

## Dictionary Structure

The SR levels dictionary has the following structure:

```python
{
    'levels': [
        {
            'id': 0,                    # Unique level ID
            'cluster_id': 1,            # Cluster this level belongs to
            'price': 1.2000,            # Price level
            'type': 'support',          # 'support', 'resistance', or 'mixed'
            'strength': 0.85,           # Level strength (0.0-1.0)
            'confidence': 0.78,         # Level confidence (0.0-1.0)
            'touches': 3,               # Number of times price touched this level
            'first_touch': '2024-01-01T10:00:00Z',  # First touch timestamp
            'last_touch': '2024-01-15T14:30:00Z',   # Last touch timestamp
            'features': {               # Additional features
                'volume_profile': 0.7,
                'price_action': 0.8,
                'technical_indicators': 0.6
            },
            'cluster_info': {           # Cluster information
                'cluster_id': 1,
                'cluster_type': 'support',
                'cluster_size': 5,
                'cluster_representative': {...}
            },
            'metadata': {               # Level metadata
                'symbol': 'ETHUSDT',
                'timeframe': '15m',
                'direction': 'longs',
                'execution_mode': 'light',
                'enhancement_version': '2.0',
                'created_at': '2024-01-15T15:00:00Z'
            }
        },
        # ... more levels
    ],
    'summary': {
        'total_levels': 50,
        'total_clusters': 12,
        'clustering_efficiency': 0.75,
        'support_levels': 25,
        'resistance_levels': 20,
        'mixed_levels': 5
    },
    'clustering_metrics': {...},        # Clustering performance metrics
    'quality_metrics': {...},           # Clustering quality metrics
    'performance_metrics': {...},       # Performance metrics
    'hardware_metrics': {...},          # Hardware optimization metrics
    'metadata': {...},                  # Overall metadata
    'access_info': {                    # Access information
        'purpose': 'feature_bank_and_training_access',
        'format_version': '2.0',
        'created_at': '2024-01-15T15:00:00Z',
        'access_methods': [
            'feature_bank.get_sr_levels()',
            'artifact_manager.get_artifact("sr_levels_dictionary")',
            'BaseStep._get_artifact("sr_levels_dictionary")'
        ]
    }
}
```

## Usage Examples

### Example 1: Basic Level Access

```python
class FeatureGenerationStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Get SR levels
        sr_levels = self._get_sr_levels(
            symbol=config.get('symbol'),
            exchange=config.get('exchange'),
            timeframe=config.get('timeframe')
        )
        
        levels = sr_levels.get('levels', [])
        
        # Filter by type
        support_levels = [l for l in levels if l.get('type') == 'support']
        resistance_levels = [l for l in levels if l.get('type') == 'resistance']
        
        # Process levels
        for level in levels:
            price = level.get('price', 0)
            strength = level.get('strength', 0)
            confidence = level.get('confidence', 0)
            touches = level.get('touches', 0)
            
            # Use level data for feature generation
            # ...
```

### Example 2: Feature Generation from SR Levels

```python
def generate_sr_features(self, sr_levels: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate features from SR levels."""
    features = {}
    
    if not sr_levels:
        return features
    
    # Count features
    support_count = len([l for l in sr_levels if l.get('type') == 'support'])
    resistance_count = len([l for l in sr_levels if l.get('type') == 'resistance'])
    
    features['sr_support_count'] = support_count
    features['sr_resistance_count'] = resistance_count
    features['sr_total_count'] = len(sr_levels)
    
    # Strength features
    strengths = [l.get('strength', 0) for l in sr_levels]
    features['sr_avg_strength'] = sum(strengths) / len(strengths) if strengths else 0
    features['sr_max_strength'] = max(strengths) if strengths else 0
    
    # Price range features
    prices = [l.get('price', 0) for l in sr_levels if l.get('price', 0) > 0]
    if prices:
        features['sr_price_min'] = min(prices)
        features['sr_price_max'] = max(prices)
        features['sr_price_range'] = max(prices) - min(prices)
    
    # Cluster features
    cluster_sizes = [l.get('cluster_info', {}).get('cluster_size', 0) for l in sr_levels]
    features['sr_avg_cluster_size'] = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0
    
    return features
```

### Example 3: Model Training with SR Levels

```python
class ModelTrainingStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Get SR levels for model training
        sr_levels = self._get_sr_levels(
            symbol=config.get('symbol'),
            exchange=config.get('exchange'),
            timeframe=config.get('timeframe')
        )
        
        # Extract features for model training
        sr_features = self._extract_sr_features(sr_levels.get('levels', []))
        
        # Use SR features in model training
        model = self._train_model_with_sr_features(sr_features, config)
        
        return {
            'success': True,
            'model': model,
            'sr_features_used': len(sr_features),
            'sr_levels_processed': len(sr_levels.get('levels', []))
        }
```

## Filtering and Querying

The SR levels dictionary supports filtering by various parameters:

```python
# Filter by symbol
sr_levels = self._get_sr_levels(symbol='ETHUSDT')

# Filter by exchange
sr_levels = self._get_sr_levels(exchange='binance')

# Filter by timeframe
sr_levels = self._get_sr_levels(timeframe='15m')

# Filter by direction
sr_levels = self._get_sr_levels(direction='longs')

# Multiple filters
sr_levels = self._get_sr_levels(
    symbol='ETHUSDT',
    exchange='binance',
    timeframe='15m',
    direction='longs'
)
```

## Error Handling

Always handle cases where SR levels might not be available:

```python
sr_levels = self._get_sr_levels(symbol=config.get('symbol'))

if sr_levels.get('error'):
    self.logger.warning(f"SR levels not available: {sr_levels['error']}")
    # Use fallback or skip SR-based features
    return self._execute_without_sr_features(config)

levels = sr_levels.get('levels', [])
if not levels:
    self.logger.warning("No SR levels found")
    # Handle empty levels case
```

## Performance Considerations

1. **Caching**: The SR levels dictionary is cached by the artifact manager for efficient access
2. **Filtering**: Use filtering parameters to reduce data size when possible
3. **Lazy Loading**: The dictionary is loaded only when accessed
4. **Memory**: Large datasets are handled efficiently with compression

## Integration with Feature Bank

The SR levels dictionary is automatically integrated with the feature bank system:

```python
from src.feature_generation.core.feature_bank import get_global_feature_bank

feature_bank = get_global_feature_bank()

# Get SR levels summary
summary = feature_bank.get_sr_levels_summary()

# Get full SR levels
sr_levels = feature_bank.get_sr_levels()
```

## Troubleshooting

### Common Issues

1. **SR levels not found**: Ensure the SR clustering step has been executed before accessing levels
2. **Empty levels**: Check if the clustering step completed successfully
3. **Filtering issues**: Verify that filter parameters match the saved metadata

### Debug Information

Enable debug logging to see detailed access information:

```python
import logging
logging.getLogger('ares.step').setLevel(logging.DEBUG)
```

## Best Practices

1. **Always check for errors**: Handle cases where SR levels might not be available
2. **Use appropriate filters**: Filter by symbol/exchange/timeframe to get relevant data
3. **Cache results**: Store processed SR features to avoid reprocessing
4. **Validate data**: Check that levels have required fields before processing
5. **Log access**: Log when SR levels are accessed for debugging purposes

## Example Scripts

See the following example scripts for complete usage examples:

- `src/training/steps/pre_training/example_sr_levels_access.py` - Complete example
- `src/training/steps/pre_training/feature_generation_feature_generation_step.py` - Feature generation
- `src/training/steps/models_training/core/base_trainer.py` - Model training

## Support

For questions or issues with SR levels dictionary access, check:

1. Logs for error messages
2. Artifact manager for saved dictionaries
3. SR clustering step execution status
4. Configuration parameters