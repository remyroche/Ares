# Artifact Versioning and Pickup System

This guide explains how to use the enhanced artifact management system with version and timestamp support in the Ares pipeline.

## Overview

The enhanced artifact management system provides:

- **Versioned Artifacts**: All artifacts are automatically named with version and timestamp
- **Automatic Pickup**: Easy discovery and loading of the most recent artifacts
- **Version Management**: Configuration-driven version handling
- **Cleanup**: Automatic cleanup of old artifacts
- **Pipeline Integration**: Seamless integration with existing sub-pipelines

## Key Components

### 1. Version Manager (`src/utils/version_manager.py`)

Manages the Ares version and provides timestamp generation.

```python
from src.utils.version_manager import get_version_manager, set_ares_version

# Get current version
vm = get_version_manager()
version = vm.get_ares_version()  # Returns "v1" by default

# Set new version
set_ares_version("v2")

# Generate timestamp
timestamp = vm.generate_timestamp()  # Returns "20250127_143022"
```

### 2. Enhanced Artifact Manager (`src/utils/enhanced_artifact_manager.py`)

Handles saving and loading artifacts with versioned filenames.

```python
from src.utils.enhanced_artifact_manager import get_artifact_manager

# Initialize manager
am = get_artifact_manager()

# Save artifact with versioned filename
data = {"key": "value"}
file_path = am.save_artifact(
    data, 
    "my_artifact", 
    ".json", 
    "artifacts"
)
# Creates: my_artifact_v1_20250127_143022.json

# Load most recent artifact
loaded_data, metadata = am.load_most_recent_artifact(
    "my_artifact", 
    "artifacts"
)
```

### 3. Artifact Pickup Utils (`src/utils/artifact_pickup_utils.py`)

Provides utilities for finding and loading the most recent artifacts.

```python
from src.utils.artifact_pickup_utils import get_artifact_pickup_utils

# Get pickup utilities
pickup_utils = get_artifact_pickup_utils()

# Find most recent artifact
recent_path = pickup_utils.find_most_recent_artifact("my_artifact", "artifacts")

# Load most recent artifact
data, metadata = pickup_utils.load_most_recent_artifact("my_artifact", "artifacts")
```

## Filename Format

All artifacts follow this naming convention:

```
{base_name}_{version}_{timestamp}.{extension}
```

Examples:
- `market_data_v1_20250127_143022.parquet`
- `trained_model_v2_20250127_143045.pkl`
- `features_v1_20250127_143100.json`

## Configuration

### Version Configuration (`config/version_config.json`)

```json
{
  "ares_version": "v1",
  "version_history": [],
  "created_at": "2025-01-27T10:00:00.000000",
  "description": "Ares pipeline version configuration",
  "artifact_naming": {
    "include_version": true,
    "include_timestamp": true,
    "timestamp_format": "%Y%m%d_%H%M%S",
    "separator": "_"
  },
  "cleanup_policy": {
    "keep_recent_count": 5,
    "auto_cleanup": false,
    "cleanup_older_than_days": 30
  }
}
```

### Pipeline Configuration Updates

Add to your pipeline configuration files:

```json
{
  "artifact_management": {
    "enabled": true,
    "versioning": true,
    "auto_cleanup": false,
    "keep_recent_count": 5
  },
  "ares_version": "v1"
}
```

## Usage Examples

### 1. Basic Artifact Creation

```python
from src.utils.enhanced_artifact_manager import get_artifact_manager

# Initialize manager
am = get_artifact_manager()

# Save different types of artifacts
import pandas as pd
import pickle

# Save DataFrame
df = pd.DataFrame({"price": [100, 101, 102]})
df_path = am.save_artifact(df, "market_data", ".parquet", "artifacts")

# Save Python object
model = {"type": "linear", "params": {"alpha": 0.1}}
model_path = am.save_artifact(model, "trained_model", ".pkl", "artifacts")

# Save JSON data
config = {"symbol": "BTCUSDT", "timeframe": "1m"}
config_path = am.save_artifact(config, "pipeline_config", ".json", "artifacts")
```

### 2. Artifact Discovery and Loading

```python
from src.utils.artifact_pickup_utils import get_artifact_pickup_utils

pickup_utils = get_artifact_pickup_utils()

# Find most recent market data
recent_data_path = pickup_utils.find_most_recent_artifact("market_data", "artifacts")
if recent_data_path:
    print(f"Found recent data: {recent_data_path}")

# Load most recent model
model_data, metadata = pickup_utils.load_most_recent_artifact("trained_model", "artifacts")
if model_data:
    print(f"Loaded model version: {metadata.version}")
    print(f"Created at: {metadata.timestamp}")
```

### 3. Pipeline Integration

```python
class MySubPipeline:
    def __init__(self):
        self.artifact_manager = get_artifact_manager()
        self.pickup_utils = get_artifact_pickup_utils()
    
    async def execute_pipeline(self, config):
        # Load input artifacts from previous stage
        input_data, _ = self.pickup_utils.load_most_recent_artifact(
            "previous_stage_output", "artifacts"
        )
        
        # Process data
        processed_data = self.process_data(input_data)
        
        # Save output artifacts with versioned names
        output_path = self.artifact_manager.save_artifact(
            processed_data,
            "my_stage_output",
            ".parquet",
            "artifacts"
        )
        
        return {"output_file": output_path}
```

### 4. Version Management

```python
from src.utils.version_manager import set_ares_version, get_version_manager

# Upgrade to new version
set_ares_version("v2")

# Create artifacts with new version
vm = get_version_manager()
print(f"Current version: {vm.get_ares_version()}")  # "v2"

# All new artifacts will use v2
am = get_artifact_manager()
new_artifact = am.save_artifact(data, "new_artifact", ".json", "artifacts")
# Creates: new_artifact_v2_20250127_143022.json
```

### 5. Artifact Cleanup

```python
from src.utils.artifact_pickup_utils import get_artifact_pickup_utils

pickup_utils = get_artifact_pickup_utils()

# Clean up old artifacts (keep 5 most recent)
deleted_files = pickup_utils.cleanup_old_artifacts(
    "my_artifact", 
    "artifacts", 
    keep_count=5
)
print(f"Deleted {len(deleted_files)} old artifacts")
```

## Migration from Existing System

### 1. Run Migration Script

```bash
python3 scripts/migrate_to_versioned_artifacts.py
```

This script will:
- Scan for existing artifacts
- Rename them to versioned format
- Update configuration files
- Generate a migration report

### 2. Update Pipeline Code

Replace existing artifact handling:

```python
# Old way
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# New way
from src.utils.enhanced_artifact_manager import get_artifact_manager
am = get_artifact_manager()
am.save_artifact(model, "model", ".pkl", "artifacts")
```

### 3. Update Artifact Loading

```python
# Old way
import pickle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# New way
from src.utils.artifact_pickup_utils import get_artifact_pickup_utils
pickup_utils = get_artifact_pickup_utils()
model, metadata = pickup_utils.load_most_recent_artifact("model", "artifacts")
```

## Best Practices

### 1. Consistent Naming

Use descriptive base names for artifacts:
- `market_data` instead of `data`
- `trained_model` instead of `model`
- `feature_engineering_output` instead of `features`

### 2. Version Management

- Use semantic versioning (v1, v2, v3)
- Document version changes
- Test with new versions before production

### 3. Artifact Organization

- Group related artifacts by pipeline stage
- Use consistent directory structure
- Document artifact dependencies

### 4. Cleanup Strategy

- Set appropriate `keep_recent_count` values
- Monitor disk usage
- Implement automated cleanup for production

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Permission Errors**: Check file system permissions
3. **Version Conflicts**: Verify version configuration
4. **Missing Artifacts**: Check directory paths and naming

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("EnhancedArtifactManager").setLevel(logging.DEBUG)
logging.getLogger("ArtifactPickupUtils").setLevel(logging.DEBUG)
```

## API Reference

### EnhancedArtifactManager

- `save_artifact(data, base_name, extension, directory, **kwargs)`: Save artifact
- `load_most_recent_artifact(base_name, directory, version, extension)`: Load most recent
- `find_artifacts(base_name, directory, version, extension)`: Find all matching artifacts
- `cleanup_old_artifacts(base_name, directory, keep_count, version)`: Cleanup old artifacts

### ArtifactPickupUtils

- `find_most_recent_artifact(base_name, directory, version, extension)`: Find most recent
- `load_most_recent_artifact(base_name, directory, version, extension)`: Load most recent
- `list_available_artifacts(directory, base_name_filter)`: List all artifacts
- `get_pipeline_artifacts(pipeline_stage, artifact_types, directory)`: Get pipeline artifacts

### VersionManager

- `get_ares_version()`: Get current version
- `set_ares_version(version)`: Set new version
- `generate_timestamp()`: Generate timestamp string
- `get_version_info()`: Get version information

## Examples

See `examples/artifact_versioning_example.py` for a complete working example demonstrating all features of the system.