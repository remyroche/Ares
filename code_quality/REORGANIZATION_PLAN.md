# Code Quality Directory Reorganization Plan

## Current Issues Identified

### 1. Redundancy Issues
- **Dead Code Analyzers**: 4 different implementations with overlapping functionality
- **Import/Undefined Checkers**: 4 different implementations 
- **Pipeline Versions**: 6+ different pipeline implementations with similar functionality
- **Backup Files**: 20+ backup files cluttering the directory

### 2. Integration Issues
- **221 scripts NOT integrated** into any pipeline
- **623 scripts PARTIALLY integrated** 
- **97 scripts need manual review**
- Many standalone scripts that should be integrated

## Reorganization Strategy

### Phase 1: Remove Redundancy

#### 1.1 Consolidate Dead Code Analyzers
**Keep**: `analyzers/enhanced_dead_code_analyzer.py` (most comprehensive)
**Remove**: 
- `improved_dead_code_analyzer.py` (root level)
- `analyzers/improved_dead_code_analyzer.py` (duplicate)
- Backup versions

#### 1.2 Consolidate Import/Undefined Checkers
**Keep**: `analyzers/import_analyzer.py` + `analyzers/undefined_names_analyzer.py`
**Remove**:
- `simple_import_undefined_checker.py`
- `import_and_undefined_checker.py` 
- `check_undefined_names.py`
- `check_undefined_names_standalone.py`

#### 1.3 Consolidate Pipelines
**Keep**: `pipelines/pipeline_unified_enhanced.py` (most comprehensive)
**Remove**:
- `pipeline_unified_enhanced_fixed.py`
- `pipeline_unified_enhanced_standalone.py`
- `unified_enhanced_pipeline.py`
- `unified_standalone_pipeline.py`
- All backup versions

### Phase 2: Integrate Missing Scripts

#### 2.1 Scripts to Integrate into Main Pipeline
- All analyzers in `analyzers/` directory
- All fixers in `fixers/` directory
- All reporters in `reporters/` directory
- All visualizers in `visualizers/` directory

#### 2.2 Scripts to Create Specialized Pipelines For
- Testing scripts → `pipelines/testing_pipeline.py`
- Validation scripts → `pipelines/validation_pipeline.py`
- Utility scripts → `pipelines/utility_pipeline.py`

### Phase 3: Directory Structure Cleanup

#### 3.1 Remove Backup Files
- All `.backup_*` files
- All duplicate versions

#### 3.2 Organize by Function
```
code_quality/
├── pipelines/           # Main pipeline orchestrators
├── analyzers/          # Analysis modules
├── fixers/            # Code fixing modules  
├── reporters/         # Reporting modules
├── visualizers/       # Visualization modules
├── utils/            # Utility modules
├── core/             # Core configuration
├── tests/            # Test files
└── examples/         # Example usage
```

## Implementation Steps

1. **Backup Current State** (already done via git)
2. **Remove Redundant Files**
3. **Update Pipeline Imports**
4. **Integrate Missing Scripts**
5. **Update Documentation**
6. **Test Integration**

## Expected Outcomes

- **Reduce file count by ~40%** (remove 50+ redundant files)
- **100% script integration** into appropriate pipelines
- **Cleaner directory structure** with clear separation of concerns
- **Improved maintainability** with single source of truth for each functionality