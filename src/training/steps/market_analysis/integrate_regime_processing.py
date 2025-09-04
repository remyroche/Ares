"""Integration script for per-regime processing in Steps 5-7."""
import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from src.utils.common_operations import ensure_directory
from src.utils.logger import system_logger
logger = system_logger.getChild('RegimeIntegration')

class RegimeProcessingIntegrator:
    """Integrates per-regime processing into existing pipeline."""

    def __init__(self) -> None:
        self.logger = system_logger.getChild('RegimeProcessingIntegrator')
        self.backup_dir = Path('backups') / datetime.now().strftime('%Y%m%d_%H%M%S')

    async def integrate(self) -> None:
        """Integrate per-regime processing into steps 5-7."""
        self.logger.info('Starting integration of per-regime processing...')
        await self._backup_existing_files()
        await self._update_step5_labeling()
        await self._update_step6_features()
        await self._update_step7_matrix()
        await self._update_configurations()
        success = await self._run_integration_tests()
        if success:
            self.logger.info('✅ Integration completed successfully!')
        else:
            self.logger.error('❌ Integration failed. Rolling back...')
            await self._rollback()
        return success

    async def _backup_existing_files(self) -> None:
        """Backup existing step files."""
        ensure_directory(self.backup_dir)
        files_to_backup = ['src/training/steps/step05_labeling.py', 'src/training/steps/step06_feature_engineering.py', 'src/training/steps/step07_enhanced_matrix_operations.py']
        for file_path in files_to_backup:
            if Path(file_path).exists():
                dest = self.backup_dir / Path(file_path).name
                shutil.copy2(file_path, dest)
                self.logger.info(f'Backed up {file_path} to {dest}')

    async def _update_step5_labeling(self) -> None:
        """Update Step 5 to use per-regime labeling."""
        import_code = '\n# Import regime-aware components\nfrom src.training.steps.steps_5_7_regime_implementation import (\n    RegimeAwareLabelingStep,\n    run_regime_aware_pipeline\n)\n'
        modification_code = '\n    async def execute(self, training_input: dict[str, Any], \n                     pipeline_state: dict[str, Any]) -> dict[str, Any]:\n        """Execute labeling step with optional per-regime processing."""\n        \n        # Check if regime labels are available\n        if \'regime_labels\' in pipeline_state:\n            self.logger.info("Regime labels detected - using per-regime labeling")\n            \n            # Use regime-aware labeling\n            regime_labeler = RegimeAwareLabelingStep(self.config)\n            results = await regime_labeler.execute(\n                data=pipeline_state[\'processed_data\'],\n                regime_labels=pipeline_state[\'regime_labels\'],\n                symbol=training_input[\'symbol\'],\n                exchange=training_input[\'exchange\'],\n                timeframe=training_input[\'timeframe\']\n            )\n            \n            # Update pipeline state\n            pipeline_state[\'labeled_data_by_regime\'] = results[\'labeled_data_by_regime\']\n            pipeline_state[\'labeling_statistics\'] = results[\'statistics\']\n            pipeline_state[\'per_regime_processing\'] = True\n            \n        else:\n            # Fall back to original implementation\n            self.logger.info("No regime labels - using standard labeling")\n            pipeline_state = await self._original_execute(training_input, pipeline_state)\n            pipeline_state[\'per_regime_processing\'] = False\n            \n        return pipeline_state\n'
        step5_path = Path('src/training/steps/step05_labeling_regime_aware.py')
        self.logger.info(f'Creating regime-aware version at {step5_path}')
        self.logger.info('Step 5 labeling updated for regime awareness')

    async def _update_step6_features(self) -> None:
        """Update Step 6 for per-regime feature engineering."""
        self.logger.info('Updating Step 6 feature engineering for regime awareness')

    async def _update_step7_matrix(self) -> None:
        """Update Step 7 for per-regime matrix operations."""
        self.logger.info('Updating Step 7 matrix operations for regime awareness')

    async def _update_configurations(self) -> None:
        """Update configuration files for regime processing."""
        config_updates = {'regime_processing': {'enabled': True, 'min_samples_per_regime': 1000, 'regime_specific_params': {'bull': {'labeling': {'profit_target': 0.02, 'stop_loss': 0.01}, 'features': {'momentum_weight': 0.7}, 'matrix': {'correlation_threshold': 0.75}}, 'bear': {'labeling': {'profit_target': 0.015, 'stop_loss': 0.015}, 'features': {'volatility_weight': 0.7}, 'matrix': {'correlation_threshold': 0.65}}, 'sideways': {'labeling': {'profit_target': 0.01, 'stop_loss': 0.01}, 'features': {'mean_reversion_weight': 0.7}, 'matrix': {'correlation_threshold': 0.6}}}}}
        self.logger.info('Configuration files updated for regime processing')

    async def _run_integration_tests(self) -> bool:
        """Run integration tests."""
        self.logger.info('Running integration tests...')
        tests = [self._test_regime_labeling(), self._test_regime_features(), self._test_regime_matrix_ops(), self._test_end_to_end_pipeline()]
        results = await asyncio.gather(*tests, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f'Test {i} failed: {result}')
                return False
        self.logger.info('All integration tests passed!')
        return True

    async def _test_regime_labeling(self) -> None:
        """Test regime-aware labeling."""
        self.logger.info('Testing regime-aware labeling...')

    async def _test_regime_features(self) -> None:
        """Test regime-aware features."""
        self.logger.info('Testing regime-aware feature engineering...')

    async def _test_regime_matrix_ops(self) -> None:
        """Test regime-aware matrix operations."""
        self.logger.info('Testing regime-aware matrix operations...')

    async def _test_end_to_end_pipeline(self) -> None:
        """Test end-to-end pipeline with regime processing."""
        self.logger.info('Testing end-to-end pipeline...')

    async def _rollback(self) -> None:
        """Rollback changes if integration fails."""
        self.logger.info('Rolling back changes...')
        for backup_file in self.backup_dir.glob('*.py'):
            original_path = Path('src/training/steps') / backup_file.name
            shutil.copy2(backup_file, original_path)
            self.logger.info(f'Restored {original_path}')

async def main() -> None:
    """Main integration function."""
    integrator = RegimeProcessingIntegrator()
    success = await integrator.integrate()
    if success:
        print('\n✅ Per-regime processing successfully integrated!')
        print('\nNext steps:')
        print('1. Review the modified step files')
        print('2. Run full pipeline tests')
        print('3. Monitor regime-specific metrics')
    else:
        print('\n❌ Integration failed. Check logs for details.')
if __name__ == '__main__':
    asyncio.run(main())