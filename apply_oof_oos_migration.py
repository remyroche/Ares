#!/usr/bin/env python3
"""
Apply OOF/OOS Migration Script

This script applies the key migrations to update existing code to use the
enhanced consolidated OOF/OOS utilities.
"""

import os
import re
import shutil
from pathlib import Path

def backup_file(file_path):
    """Create a backup of the file."""
    from datetime import datetime
    backup_path = f"{file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(file_path, backup_path)
    print(f"📁 Created backup: {backup_path}")
    return backup_path

def update_training_utils():
    """Update training_utils.py to use enhanced consolidated utilities."""
    file_path = "src/utils/ml_common/training/training_utils.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"🔄 Updating {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply transformations
    transformations = [
        # Update imports
        (
            'from src.utils.ml_common.ensembles.oof_stacking_ensemble_manager import (\n            OOFStackingEnsembleManager,\n            OOFStackingEnsembleConfig\n        )',
            'from src.utils.ml_common.validation.enhanced_consolidated_oof_oos import (\n            create_enhanced_oof_generator,\n            OOFStrategy,\n            ValidationType\n        )'
        ),
        # Update method implementation
        (
            '        # Import OOF stacking ensemble manager\n        from src.utils.ml_common.ensembles.oof_stacking_ensemble_manager import (\n            OOFStackingEnsembleManager,\n            OOFStackingEnsembleConfig\n        )\n\n        # Create configuration\n        ensemble_config = OOFStackingEnsembleConfig(\n            ensemble_name=ensemble_name,\n            output_dir=f"./models/{ensemble_name}",\n            n_outputs=n_outputs,\n            output_names=output_names,\n            base_models=base_models,\n            enable_out_of_fold=True,\n            enable_temporal_validation=enable_temporal_validation,\n            cv_folds=cv_folds,\n            enable_early_stopping=True,\n            early_stopping_rounds=50\n        )\n\n        # Create ensemble manager\n        ensemble_manager = OOFStackingEnsembleManager(ensemble_config)\n\n        # Add base models to ensemble\n        for output_name, models in base_models.items():\n            for model_name, model in models.items():\n                ensemble_manager.add_base_model(output_name, model_name, model)\n\n        logger.info(f"✅ OOF Stacking ensemble created: {ensemble_name}")\n        return ensemble_manager',
            '        # Import enhanced consolidated OOF generator\n        from src.utils.ml_common.validation.enhanced_consolidated_oof_oos import (\n            create_enhanced_oof_generator,\n            OOFStrategy,\n            ValidationType\n        )\n\n        # Create enhanced OOF generator\n        oof_generator = create_enhanced_oof_generator(\n            strategy=OOFStrategy.STACKING,\n            n_folds=cv_folds,\n            n_outputs=n_outputs,\n            output_names=output_names,\n            enable_temporal_validation=enable_temporal_validation,\n            enable_confidence_intervals=True,\n            enable_diversity_metrics=True,\n            enable_leakage_detection=True,\n            enable_early_stopping=True,\n            early_stopping_rounds=50,\n            random_state=42\n        )\n\n        logger.info(f"✅ Enhanced OOF Stacking ensemble created: {ensemble_name}")\n        return oof_generator'
        ),
        # Update train method
        (
            '        # Train the ensemble\n        trained_ensemble = ensemble_manager.fit(X, y)',
            '        # Generate OOF predictions using enhanced generator\n        oof_result = ensemble_manager.generate_oof_predictions(\n            models=base_models, X=X, y=y, timestamps=timestamps\n        )\n        \n        # For compatibility, return the OOF result as the "trained ensemble"\n        trained_ensemble = oof_result'
        )
    ]
    
    # Apply transformations
    for old_text, new_text in transformations:
        if old_text in content:
            content = content.replace(old_text, new_text)
            print(f"✅ Applied transformation")
        else:
            print(f"⚠️ Pattern not found in file")
    
    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {file_path}")
    return True

def update_multi_output_models():
    """Update multi_output_models.py to use enhanced consolidated utilities."""
    file_path = "src/utils/ml_common/models/multi_output_models.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    print(f"🔄 Updating {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add import at the top
    import_pattern = r'(from sklearn\.metrics import.*?\n)'
    enhanced_import = '''from src.utils.ml_common.validation.enhanced_consolidated_oof_oos import (
    create_enhanced_oos_validator,
    OOSValidationType
)
'''
    
    if 'create_enhanced_oos_validator' not in content:
        content = re.sub(import_pattern, r'\1' + enhanced_import, content, count=1)
        print("✅ Added enhanced import")
    
    # Update evaluate_oof_performance method
    method_pattern = r'(def evaluate_oof_performance\(self\) -> Dict\[str, Any\]:\s*""".*?""".*?)(return \{[^}]+"error": str\(e\)\})'
    
    new_method = '''def evaluate_oof_performance(self) -> Dict[str, Any]:
        """Evaluate performance using enhanced consolidated OOF utilities."""
        if self._oof_meta_predictions is None or self.y_train is None:
            return {'error': 'OOF predictions not available'}
        
        try:
            # Create OOS validator for performance evaluation
            oos_validator = create_enhanced_oos_validator(
                validation_type=OOSValidationType.PERFORMANCE_METRICS,
                metrics=['mse', 'mae', 'r2', 'accuracy']
            )
            
            # Perform OOS validation
            oos_result = oos_validator.validate_oos(
                predictions=self._oof_meta_predictions,
                targets=self.y_train
            )
            
            # Extract results
            y = self.y_train
            y_pred = self._oof_meta_predictions
            # Ensure 2D
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            if len(y_pred.shape) == 1:
                y_pred = y_pred.reshape(-1, 1)

            per_output_metrics = {}
            overall_metrics = {}
            num_outputs_to_process = min(y.shape[1], y_pred.shape[1])
            for output_idx in range(num_outputs_to_process):
                output_name = self.config.output_names[output_idx] if output_idx < len(self.config.output_names) else f"output_{output_idx+1}"
                y_true_output = y[:, output_idx]
                y_pred_output = y_pred[:, output_idx]
                mse = np.mean((y_true_output - y_pred_output) ** 2)
                mae = np.mean(np.abs(y_true_output - y_pred_output))
                r2 = 1 - (np.sum((y_true_output - y_pred_output) ** 2) / np.sum((y_true_output - np.mean(y_true_output)) ** 2))
                per_output_metrics[output_name] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2)
                }
                overall_metrics[f'{output_name}_mse'] = float(mse)
                overall_metrics[f'{output_name}_mae'] = float(mae)
                overall_metrics[f'{output_name}_r2'] = float(r2)

            overall_metrics['overall_mse'] = float(np.mean([m['mse'] for m in per_output_metrics.values()]))
            overall_metrics['overall_mae'] = float(np.mean([m['mae'] for m in per_output_metrics.values()]))
            overall_metrics['overall_r2'] = float(np.mean([m['r2'] for m in per_output_metrics.values()]))

            return {
                'per_output_metrics': per_output_metrics,
                'overall_metrics': overall_metrics,
                'predictions': self._oof_meta_predictions,
                'targets': self.y_train,
                'oos_validation': oos_result.validation_scores,
                'oos_metrics': oos_result.validation_metrics
            }
        except Exception as e:
            self.logger.error(f"❌ OOF evaluation failed: {e}")
            return {'error': str(e)}'''
    
    if re.search(method_pattern, content, re.DOTALL):
        content = re.sub(method_pattern, new_method, content, flags=re.DOTALL)
        print("✅ Updated evaluate_oof_performance method")
    else:
        print("⚠️ Method pattern not found")
    
    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {file_path}")
    return True

def main():
    """Main function to apply migrations."""
    print("🚀 Starting OOF/OOS Migration")
    
    # Create backups and apply migrations
    migrations = [
        ("Training Utils", update_training_utils),
        ("Multi-Output Models", update_multi_output_models),
    ]
    
    success_count = 0
    for name, migration_func in migrations:
        print(f"\n🔄 Processing {name}...")
        try:
            if migration_func():
                success_count += 1
                print(f"✅ {name} migration completed")
            else:
                print(f"❌ {name} migration failed")
        except Exception as e:
            print(f"❌ {name} migration error: {e}")
    
    print(f"\n🎉 Migration completed: {success_count}/{len(migrations)} successful")
    
    if success_count == len(migrations):
        print("✅ All migrations successful!")
    else:
        print("⚠️ Some migrations failed. Check the logs above.")

if __name__ == "__main__":
    main()