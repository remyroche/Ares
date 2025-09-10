# Files to Delete After Migration

## ⚠️ **IMPORTANT**: Backup Before Deletion

Before deleting any files, create a backup:
```bash
# Create backup directory
mkdir -p backup/monolithic_architecture_$(date +%Y%m%d)

# Backup the entire training directory
cp -r src/training backup/monolithic_architecture_$(date +%Y%m%d)/
```

## Files to Delete (High Priority)

### **1. Monolithic Training Manager**
```bash
# Main monolithic file (3,079 lines)
rm src/training/enhanced_training_manager.py
rm src/training/enhanced_training_manager.py.syntax_backup

# Backup versions
rm src/training/enhanced_training_manager_optimized.py
rm src/training/enhanced_training_manager_optimized.py.syntax_backup
```

### **2. Old Training Managers**
```bash
# Old training manager implementations
rm src/training/training_manager.py
rm src/training/training_manager.py.syntax_backup
rm src/training/simplified_training_manager.py
rm src/training/di_training_manager.py
rm src/training/di_training_manager.py.syntax_backup
```

### **3. Old Orchestration Files**
```bash
# Old orchestration components
rm src/training/training_orchestrator.py
rm src/training/training_orchestrator.py.syntax_backup
rm src/training/step_orchestrator.py
```

## Files to Delete (Medium Priority)

### **4. Redundant Step Files**
```bash
# Individual step files that are now replaced by modular components
rm src/training/steps/step01_data_collection.py
rm src/training/steps/step01_5_data_converter.py
rm src/training/steps/step02_data_reading.py
rm src/training/steps/step06_feature_engineering.py
rm src/training/steps/step09_hmm_based_training.py

# And their validators
rm src/training/steps/step01_data_collection_validator.py
rm src/training/steps/step01_5_data_converter_validator.py
rm src/training/steps/step02_data_reading_validator.py
rm src/training/steps/step06_feature_engineering_validator.py
rm src/training/steps/step09_hmm_based_training_validator.py
```

### **5. Old Configuration Files**
```bash
# Old configuration system
rm src/training/step_config.py
```

### **6. Migration and Documentation Files (After Review)**
```bash
# These can be deleted after confirming migration is complete
rm src/training/migration_script.py
rm src/training/migration_quickstart.py
rm src/training/IMPORT_MAPPING_GUIDE.md
rm src/training/MIGRATION_DASHBOARD.md
rm src/training/MIGRATION_GUIDE.md
rm src/training/MIGRATION_NEXT_STEPS.md
rm src/training/MIGRATION_SUMMARY.md
rm src/training/MIGRATION_TASKS.md
rm src/training/MODULE_STRUCTURE.md
rm src/training/REFACTORING_SUMMARY.md
```

## Files to Keep (Do NOT Delete)

### **Core Architecture Files**
```bash
# Keep these - they are the new architecture
src/training/simplified_architecture/
├── dependency_injection.py
├── enhanced_interfaces.py
├── enhanced_config_system.py
├── enhanced_pipeline_orchestrator.py
├── modular_components.py
├── migrated_components/
├── config/
└── tests/
```

### **Utility Files**
```bash
# Keep utility files that are still useful
src/training/validator.py
src/training/progress_manager.py
src/training/ensemble_manager.py
src/training/optimization_manager.py
```

### **Data and Model Files**
```bash
# Keep data and model related files
src/training/data_manager.py
src/training/model_trainer.py
src/training/multi_output_model_trainer.py
```

## Deletion Script

Create a safe deletion script:

```bash
#!/bin/bash
# safe_delete_old_files.sh

echo "⚠️  WARNING: This will delete old monolithic architecture files!"
echo "Make sure you have backed up the files and tested the new architecture."
read -p "Are you sure you want to proceed? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Deletion cancelled."
    exit 1
fi

# Create backup first
BACKUP_DIR="backup/monolithic_architecture_$(date +%Y%m%d_%H%M%S)"
echo "Creating backup in $BACKUP_DIR..."
mkdir -p "$BACKUP_DIR"
cp -r src/training "$BACKUP_DIR/"

echo "Backup created. Proceeding with deletion..."

# Delete monolithic files
echo "Deleting monolithic training manager..."
rm -f src/training/enhanced_training_manager.py
rm -f src/training/enhanced_training_manager.py.syntax_backup
rm -f src/training/enhanced_training_manager_optimized.py
rm -f src/training/enhanced_training_manager_optimized.py.syntax_backup

# Delete old training managers
echo "Deleting old training managers..."
rm -f src/training/training_manager.py
rm -f src/training/training_manager.py.syntax_backup
rm -f src/training/simplified_training_manager.py
rm -f src/training/di_training_manager.py
rm -f src/training/di_training_manager.py.syntax_backup

# Delete old orchestration
echo "Deleting old orchestration files..."
rm -f src/training/training_orchestrator.py
rm -f src/training/training_orchestrator.py.syntax_backup
rm -f src/training/step_orchestrator.py

# Delete old step files
echo "Deleting old step files..."
rm -f src/training/steps/step01_data_collection.py
rm -f src/training/steps/step01_5_data_converter.py
rm -f src/training/steps/step02_data_reading.py
rm -f src/training/steps/step06_feature_engineering.py
rm -f src/training/steps/step09_hmm_based_training.py

# Delete validators
rm -f src/training/steps/step01_data_collection_validator.py
rm -f src/training/steps/step01_5_data_converter_validator.py
rm -f src/training/steps/step02_data_reading_validator.py
rm -f src/training/steps/step06_feature_engineering_validator.py
rm -f src/training/steps/step09_hmm_based_training_validator.py

# Delete old configuration
rm -f src/training/step_config.py

echo "✅ Deletion complete!"
echo "📁 Backup available at: $BACKUP_DIR"
echo "🧪 Please test the new architecture before proceeding with production use."
```

## Verification After Deletion

After deleting files, verify the new architecture works:

```bash
# Test imports
python -c "
from src.training.simplified_architecture.enhanced_pipeline_orchestrator import create_pipeline
print('✅ New architecture imports working!')
"

# Test configuration
python -c "
from src.training.simplified_architecture.enhanced_config_system import ConfigurationManager
config_manager = ConfigurationManager()
print('✅ Configuration system working!')
"

# Run tests
cd src/training/simplified_architecture
python -m pytest tests/ -v
```

## Space Savings

After deletion, you should see significant space savings:

- **enhanced_training_manager.py**: ~200KB (3,079 lines)
- **training_manager.py**: ~37KB (1,076 lines)
- **step_orchestrator.py**: ~26KB (728 lines)
- **Individual step files**: ~500KB+ (21+ files)
- **Total estimated savings**: ~800KB+ of code

## Rollback Plan

If you need to rollback:

```bash
# Restore from backup
cp -r backup/monolithic_architecture_YYYYMMDD_HHMMSS/training/* src/training/

# Or restore specific files
cp backup/monolithic_architecture_YYYYMMDD_HHMMSS/training/enhanced_training_manager.py src/training/
```

## Final Checklist

Before deleting files, ensure:

- [ ] ✅ New architecture is tested and working
- [ ] ✅ All imports have been updated
- [ ] ✅ Configuration files are created
- [ ] ✅ Backup has been created
- [ ] ✅ Team has been notified
- [ ] ✅ Rollback plan is ready

After deletion:

- [ ] ✅ Verify new architecture still works
- [ ] ✅ Update any remaining references
- [ ] ✅ Clean up any broken imports
- [ ] ✅ Update documentation
- [ ] ✅ Notify team of completion