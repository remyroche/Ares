# Artifact Management Guide

## Overview

The Ares trading system uses a centralized artifact management system that automatically handles data persistence, compression, versioning, and format conversion. All steps use this system to store and retrieve intermediate results.

## Key Features

### Automatic Format Generation
- **Parquet Files**: Primary format for all data artifacts (always generated)
- **CSV Files**: Automatically generated for DataFrames with < 2000 rows
- **Compression**: Automatic compression for large datasets
- **Metadata**: Automatic metadata tracking and versioning

### Smart Storage
- **Context-Aware**: Automatically organizes artifacts by symbol, exchange, direction, model
- **Versioning**: Tracks artifact versions and dependencies
- **Caching**: Intelligent caching for frequently accessed artifacts
- **Cleanup**: Automatic cleanup of old artifacts

## Usage

### Basic Artifact Operations

```python
class MyStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Set context for artifact organization
        self.artifact_manager.set_context(
            symbol='ETHUSDT',
            exchange='binance',
            direction='longs',
            model='Analyst'
        )
        
        # Save data as artifact (auto-generates Parquet + CSV if applicable)
        artifact_path = self._save_artifact(
            data,                    # Your data (DataFrame, dict, etc.)
            'my_artifact',          # Artifact name
            'data'                  # Artifact type
        )
        
        # Retrieve artifact
        retrieved_data = self._get_artifact('my_artifact', 'data')
        
        return {
            'success': True,
            'artifacts': [artifact_path],
            'data': retrieved_data
        }
```

### Artifact Types

#### Data Artifacts
```python
# DataFrame (auto-generates CSV if < 2000 rows)
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
artifact_path = self._save_artifact(df, 'dataframe_artifact', 'data')

# Dictionary
data_dict = {'key1': 'value1', 'key2': 'value2'}
artifact_path = self._save_artifact(data_dict, 'dict_artifact', 'data')

# List
data_list = [1, 2, 3, 4, 5]
artifact_path = self._save_artifact(data_list, 'list_artifact', 'data')
```

#### Model Artifacts
```python
# Trained model
model = train_model(data)
artifact_path = self._save_artifact(model, 'trained_model', 'model')

# Model metadata
metadata = {
    'model_type': 'XGBoost',
    'accuracy': 0.85,
    'training_time': 120.5
}
artifact_path = self._save_artifact(metadata, 'model_metadata', 'metadata')
```

#### Configuration Artifacts
```python
# Configuration dictionary
config = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'epochs': 100
}
artifact_path = self._save_artifact(config, 'training_config', 'config')
```

### Advanced Features

#### Compression Options
```python
# Automatic compression (default)
artifact_path = self._save_artifact(data, 'artifact', 'data')

# Specific compression
artifact_path = self._save_artifact(data, 'artifact', 'data', compression='gzip')

# No compression
artifact_path = self._save_artifact(data, 'artifact', 'data', compression='none')
```

#### Metadata Tracking
```python
# Save with custom metadata
metadata = {
    'created_by': 'my_step',
    'version': '1.0',
    'description': 'Processed market data'
}
artifact_path = self._save_artifact(
    data, 
    'artifact', 
    'data', 
    metadata=metadata
)
```

#### Context Management
```python
# Set context for automatic organization
self.artifact_manager.set_context(
    symbol='ETHUSDT',
    exchange='binance', 
    direction='longs',
    model='Analyst',
    information='processed_data'
)

# All subsequent saves will be organized under this context
artifact_path = self._save_artifact(data, 'artifact', 'data')
```

## File Organization

### Directory Structure
```
artifacts/
├── ETHUSDT/
│   ├── binance/
│   │   ├── longs/
│   │   │   ├── Analyst/
│   │   │   │   ├── data_download/
│   │   │   │   │   ├── raw_data.parquet
│   │   │   │   │   └── raw_data.csv
│   │   │   │   ├── sr_detection/
│   │   │   │   │   ├── sr_levels.parquet
│   │   │   │   │   └── sr_levels.csv
│   │   │   │   └── model_training/
│   │   │   │       ├── trained_model.pkl
│   │   │   │       └── model_metadata.json
│   │   │   └── Tactician/
│   │   │       └── ...
│   │   └── shorts/
│   │       └── ...
│   └── other_exchanges/
└── other_symbols/
```

### Naming Conventions
- **Symbols**: Uppercase (e.g., ETHUSDT, BTCUSDT)
- **Exchanges**: Lowercase (e.g., binance, coinbase)
- **Directions**: Lowercase (e.g., longs, shorts)
- **Models**: PascalCase (e.g., Analyst, Tactician)
- **Artifacts**: snake_case (e.g., sr_levels, trained_model)

## Automatic CSV Generation

### When CSV is Generated
- **Always**: Parquet files are generated for all data
- **Conditionally**: CSV files are generated for DataFrames with < 2000 rows
- **Logging**: Automatic logging of CSV generation decisions

### Example
```python
# Small DataFrame (< 2000 rows) - generates both Parquet and CSV
small_df = pd.DataFrame({'col1': range(100), 'col2': range(100)})
artifact_path = self._save_artifact(small_df, 'small_data', 'data')
# Logs: "📊 Auto-saved CSV artifact (rows < 2000): small_data -> path/to/small_data.csv"

# Large DataFrame (>= 2000 rows) - generates only Parquet
large_df = pd.DataFrame({'col1': range(5000), 'col2': range(5000)})
artifact_path = self._save_artifact(large_df, 'large_data', 'data')
# Logs: "📊 Skipping CSV auto-save for large_data (rows >= 2000: 5000)"
```

## Performance Optimization

### Caching
```python
# Enable caching for frequently accessed artifacts
cached_data = self._get_artifact('frequent_artifact', 'data', use_cache=True)
```

### Batch Operations
```python
# Save multiple artifacts efficiently
artifacts = []
for i, data_chunk in enumerate(data_chunks):
    artifact_path = self._save_artifact(data_chunk, f'chunk_{i}', 'data')
    artifacts.append(artifact_path)
```

### Memory Management
```python
# For large datasets, use streaming/chunked processing
for chunk in data_stream:
    processed_chunk = process_chunk(chunk)
    artifact_path = self._save_artifact(processed_chunk, f'chunk_{chunk.id}', 'data')
```

## Error Handling

### Common Errors and Solutions

#### Artifact Not Found
```python
try:
    data = self._get_artifact('missing_artifact', 'data')
except FileNotFoundError:
    self.logger.warning("Artifact not found, generating default data")
    data = generate_default_data()
```

#### Insufficient Disk Space
```python
try:
    artifact_path = self._save_artifact(large_data, 'artifact', 'data')
except OSError as e:
    if "No space left" in str(e):
        self.logger.error("Insufficient disk space, cleaning up old artifacts")
        self.artifact_manager.cleanup_old_artifacts()
        # Retry save
        artifact_path = self._save_artifact(large_data, 'artifact', 'data')
```

#### Corrupted Artifact
```python
try:
    data = self._get_artifact('artifact', 'data')
except Exception as e:
    self.logger.error(f"Failed to load artifact: {e}")
    # Regenerate artifact
    data = regenerate_data()
    artifact_path = self._save_artifact(data, 'artifact', 'data')
```

## Monitoring and Maintenance

### Artifact Statistics
```python
# Get artifact statistics
stats = self.artifact_manager.get_artifact_stats()
print(f"Total artifacts: {stats['total_count']}")
print(f"Total size: {stats['total_size_mb']} MB")
print(f"Oldest artifact: {stats['oldest_artifact']}")
```

### Cleanup Operations
```python
# Clean up artifacts older than 30 days
self.artifact_manager.cleanup_old_artifacts(days=30)

# Clean up artifacts for specific symbol
self.artifact_manager.cleanup_symbol_artifacts('OLD_SYMBOL')

# Clean up temporary artifacts
self.artifact_manager.cleanup_temp_artifacts()
```

### Health Checks
```python
# Check artifact manager health
health = self.artifact_manager.health_check()
if not health['healthy']:
    self.logger.error(f"Artifact manager issues: {health['issues']}")
```

## Best Practices

### 1. Naming
- Use descriptive, consistent artifact names
- Include step name or processing stage in artifact name
- Use snake_case for artifact names

### 2. Organization
- Set appropriate context before saving artifacts
- Group related artifacts logically
- Use consistent artifact types

### 3. Performance
- Use appropriate compression for data size
- Consider memory usage for large datasets
- Clean up temporary artifacts regularly

### 4. Error Handling
- Always handle artifact loading errors gracefully
- Implement fallback strategies for missing artifacts
- Log artifact operations for debugging

### 5. Versioning
- Include version information in metadata
- Track artifact dependencies
- Maintain backward compatibility when possible

## Examples

### Data Pipeline with Artifacts
```python
class DataPipelineStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Set context
        self.artifact_manager.set_context(
            symbol=config['symbol'],
            exchange=config['exchange'],
            direction=config['direction'],
            model='DataPipeline'
        )
        
        artifacts = []
        
        # Step 1: Load raw data
        raw_data = await self._load_raw_data(config)
        raw_path = self._save_artifact(raw_data, 'raw_data', 'data')
        artifacts.append(raw_path)
        
        # Step 2: Clean data
        cleaned_data = self._clean_data(raw_data)
        cleaned_path = self._save_artifact(cleaned_data, 'cleaned_data', 'data')
        artifacts.append(cleaned_path)
        
        # Step 3: Process data
        processed_data = self._process_data(cleaned_data)
        processed_path = self._save_artifact(processed_data, 'processed_data', 'data')
        artifacts.append(processed_path)
        
        return {
            'success': True,
            'artifacts': artifacts,
            'metrics': {
                'raw_rows': len(raw_data),
                'cleaned_rows': len(cleaned_data),
                'processed_rows': len(processed_data)
            }
        }
```

### Model Training with Artifacts
```python
class ModelTrainingStep(BaseStep):
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # Load training data
        training_data = self._get_artifact('training_data', 'data')
        
        # Train model
        model = await self._train_model(training_data, config)
        
        # Save model and metadata
        model_path = self._save_artifact(model, 'trained_model', 'model')
        
        metadata = {
            'model_type': 'XGBoost',
            'accuracy': model.accuracy,
            'training_time': model.training_time,
            'features': model.feature_names
        }
        metadata_path = self._save_artifact(metadata, 'model_metadata', 'metadata')
        
        return {
            'success': True,
            'artifacts': [model_path, metadata_path],
            'metrics': metadata
        }
```

This guide provides comprehensive information about using the artifact management system effectively in the Ares trading system.
