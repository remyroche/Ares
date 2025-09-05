# Directory Organization Plan

## Current Status
Based on the script integration analysis, we have:
- **181 total scripts** across the code_quality directory
- **56.9% fully integrated** (103 scripts)
- **34.8% partially integrated** (63 scripts)
- **2.8% not integrated** (5 scripts)
- **5.5% need review** (10 scripts)

## Proposed Directory Structure

### 1. Core Pipelines (`/pipelines/`)
**Purpose**: Main pipeline execution scripts
```
pipelines/
├── core/
│   ├── unified_enhanced_pipeline.py          # Main comprehensive pipeline
│   ├── sequential_code_fixer.py              # Sequential fixing pipeline
│   ├── code_interaction_mapper.py            # Code interaction mapping
│   ├── dead_code_analyzer.py                 # Dead code analysis
│   └── complexity_cli.py                     # Complexity analysis CLI
├── specialized/
│   ├── validation_pipeline.py                # Validation and testing
│   ├── reporting_pipeline.py                 # Report generation
│   └── utility_pipeline.py                   # Utility scripts
├── plugins/
│   ├── enhanced_import_analyzer_plugin.py    # Enhanced import analysis
│   └── script_integration_manager.py         # Integration management
└── enhanced_import_analysis.py               # Enhanced import analysis runner
```

### 2. Analyzers (`/analyzers/`)
**Purpose**: Code analysis components
```
analyzers/
├── core/
│   ├── import_analyzer.py
│   ├── syntax_validator.py
│   ├── linter_analyzer.py
│   └── static_analysis_analyzer.py
├── advanced/
│   ├── architecture_analyzer.py
│   ├── call_graph_analyzer.py
│   ├── complexity_analyzer.py
│   └── dependency_analyzer.py
├── quality/
│   ├── code_smell_detector.py
│   ├── metrics_analyzer.py
│   ├── performance_analyzer.py
│   └── test_coverage_analyzer.py
└── dead_code/
    ├── dead_code_analyzer.py
    ├── enhanced_dead_code_analyzer.py
    └── improved_dead_code_analyzer.py
```

### 3. Fixers (`/fixers/`)
**Purpose**: Code fixing and repair components
```
fixers/
├── core/
│   ├── auto_fixer.py
│   ├── conservative_auto_fixer.py
│   └── sequential_fixer.py
├── import_fixers/
│   ├── comprehensive_import_fixer.py
│   ├── targeted_import_fixer.py
│   └── fix_import_issues.py
├── syntax_fixers/
│   ├── fix_undefined_names.py
│   ├── fix_common_undefined_names.py
│   └── fix_simple_undefined_names.py
└── specialized/
    ├── auto_fix_dead_code.py
    └── fix_top_undefined_names.py
```

### 4. Validators (`/validators/`)
**Purpose**: Code validation and checking components
```
validators/
├── enhanced_validator.py
├── integrated_validator.py
├── function_validator.py
├── simple_import_undefined_checker.py
├── import_and_undefined_checker.py
└── check_undefined_names.py
```

### 5. Scripts (`/scripts/`)
**Purpose**: Utility and helper scripts
```
scripts/
├── syntax/
│   ├── advanced_syntax_fixer.py
│   ├── bulk_syntax_cleanup.py
│   └── fix_common_syntax_patterns.py
├── imports/
│   ├── fix_missing_imports.py
│   ├── detect_circular_imports.py
│   └── add_type_hints.py
├── interactions/
│   ├── extract_interactions.py
│   ├── simple_interaction_mapper.py
│   └── interaction_summary.py
├── async/
│   ├── fix_async_await.py
│   └── robust_async_fixer.py
└── master/
    ├── master_code_quality.py
    ├── final_code_fixes.py
    └── apply_all_fixes.py
```

### 6. Visualizers (`/visualizers/`)
**Purpose**: Code visualization components
```
visualizers/
├── dashboard_generator.py
├── complexity_heatmap.py
├── dependency_graph.py
├── interaction_network.py
├── code_visualizer.py
└── visualize_interactions.py
```

### 7. Reporters (`/reporters/`)
**Purpose**: Report generation components
```
reporters/
├── html_reporter.py
├── quality_reporter.py
├── trend_reporter.py
├── error_reporter.py
└── report_aggregator.py
```

### 8. Plugins (`/plugins/`)
**Purpose**: Plugin system components
```
plugins/
├── base_plugin.py
├── plugin_manager.py
├── plugin_registry.py
├── production/
│   ├── syntax_fixer.py
│   ├── import_fixer.py
│   ├── dead_code_fixer.py
│   ├── linter_runner.py
│   └── security_scanner.py
├── code_quality/
│   ├── black_fixer.py
│   ├── isort_fixer.py
│   ├── flake8_analyzer.py
│   └── ruff_analyzer.py
└── examples/
    ├── syntax_plugin.py
    ├── import_plugin.py
    └── security_plugin.py
```

### 9. Core (`/core/`)
**Purpose**: Core configuration and utilities
```
core/
├── config.py
├── plugins.py
└── __init__.py
```

### 10. Utils (`/utils/`)
**Purpose**: Utility functions and helpers
```
utils/
├── file_utils.py
├── dependency_manager.py
├── progress.py
└── report_aggregator.py
```

### 11. Code Complexity (`/code_complexity/`)
**Purpose**: Code complexity analysis (keep existing structure)
```
code_complexity/
├── cli.py
├── complexity_pipeline.py
├── analyzers/
├── config/
├── utils/
└── reports/
```

## Migration Plan

### Phase 1: Core Pipeline Organization
1. ✅ Move specified files to pipelines/ with descriptive names
2. ✅ Update import statements
3. ✅ Create enhanced import analyzer plugin
4. ✅ Integrate enhanced import analysis into pipelines

### Phase 2: Script Categorization
1. Move analyzers to organized subdirectories
2. Move fixers to organized subdirectories
3. Move validators to dedicated directory
4. Organize scripts by functionality

### Phase 3: Plugin System Enhancement
1. Organize plugins by category
2. Complete plugin registration system
3. Add plugin discovery mechanisms

### Phase 4: Documentation and Testing
1. Update all documentation
2. Create comprehensive examples
3. Add integration tests

## Benefits of This Organization

1. **Clear Separation of Concerns**: Each directory has a specific purpose
2. **Easy Navigation**: Related components are grouped together
3. **Scalable Structure**: Easy to add new components
4. **Plugin System**: Extensible architecture
5. **Pipeline Integration**: All components can be easily integrated into pipelines

## Implementation Status

- ✅ **Pipelines Structure**: Created and populated
- ✅ **Import Updates**: Updated for moved files
- ✅ **Enhanced Import Analysis**: Integrated into pipelines
- ✅ **Script Integration Analysis**: Completed
- 🔄 **Directory Organization**: In progress
- ⏳ **Plugin System Enhancement**: Pending
- ⏳ **Documentation Updates**: Pending

## Next Steps

1. **Complete Directory Reorganization**: Move remaining files to appropriate directories
2. **Update All Import Statements**: Ensure all imports work with new structure
3. **Create Pipeline Orchestrator**: Master script to run all pipelines
4. **Add Configuration Management**: Centralized configuration system
5. **Implement Monitoring**: Pipeline execution monitoring and reporting