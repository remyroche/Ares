# Unified Configuration Directory

This directory contains all configuration files for the Ares trading system, consolidating both the previous `config/` and `configs/` directories.

## Directory Structure

```
config/
├── environments/          # Environment-specific configurations
│   ├── development.json   # Development environment settings
│   ├── production.json    # Production environment settings
│   └── testing.json       # Testing environment settings
├── features/              # Feature-specific configurations
│   ├── enhanced_reporting_config.yaml
│   ├── explainability_config.yaml
│   ├── probabilistic_optimization.yaml
│   ├── sr_levels_config.yaml
│   ├── training_config.json
│   └── training_modes.yaml
├── version_config.json    # Version and artifact configuration
└── README.md             # This file
```

## Backward Compatibility

The system maintains backward compatibility with the old directory structure:

- **Old paths still work**: `configs/development_config.json` → `config/environments/development.json`
- **Automatic path resolution**: Configuration loaders automatically resolve old paths to new locations
- **Gradual migration**: You can migrate at your own pace without breaking existing functionality

## Usage

### Environment Configurations

```python
from src.core.config_service import ConfigService

# Load environment-specific configuration
config_service = ConfigService(environment='production')
config = config_service.load_config()

# Or load directly
config = ConfigService.load_environment_config('development')
```

### Feature Configurations

```python
from src.core.config_service import ConfigService

# Load feature-specific configuration
config = ConfigService.load_feature_config('enhanced_reporting')
```

### Legacy Support

```python
# These still work for backward compatibility
config = ConfigService.load_config('configs/production_config.json')
config = ConfigService.load_config('config/enhanced_reporting_config.yaml')
```

## Migration Guide

### For New Code
- Use the new unified structure: `config/environments/` and `config/features/`
- Use the `ConfigService` class for loading configurations

### For Existing Code
- No immediate changes required - backward compatibility is maintained
- Gradually migrate to new paths when convenient
- Update imports to use `ConfigService` instead of direct file loading

## Configuration Types

### Environment Configurations
- **development.json**: Development environment with debugging enabled
- **production.json**: Production environment with full features and optimization
- **testing.json**: Testing environment with minimal resource usage

### Feature Configurations
- **enhanced_reporting_config.yaml**: Enhanced training manager reporting
- **explainability_config.yaml**: Model explainability settings
- **probabilistic_optimization.yaml**: Probabilistic optimization parameters
- **sr_levels_config.yaml**: Support/resistance level configuration
- **training_config.json**: Training pipeline configuration
- **training_modes.yaml**: Training mode definitions

### System Configurations
- **version_config.json**: Version management and artifact naming

## Best Practices

1. **Environment-specific**: Use environment configurations for deployment-specific settings
2. **Feature-specific**: Use feature configurations for component-specific settings
3. **Validation**: Always validate configurations before use
4. **Security**: Keep sensitive data in environment variables, not config files
5. **Versioning**: Use version_config.json for artifact versioning and naming