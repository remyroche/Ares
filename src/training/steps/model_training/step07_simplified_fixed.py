"""
Step07 Enhanced Matrix Operations - Simplified with Fixed Imports

This is a simplified version that addresses the import issues
identified in the audit while maintaining core functionality.
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Use the import fix module
try:
    from src.utils.step07_import_fix import (
        numpy as np, pandas as pd, torch, numba, psutil,
        system_logger, handles_errors, BaseStep, check_dependencies
    )
except ImportError:
    # Fallback imports
    import logging
    system_logger = logging.getLogger('step07_simplified')
    logging.basicConfig(level=logging.INFO)
    
    def handles_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class BaseStep:
        def __init__(self, config, step_id, step_name):
            self.config = config
            self.step_id = step_id
            self.step_name = step_name

class SimplifiedMatrixOperationsStep(BaseStep):
    """Simplified Step07 with fixed imports and reduced complexity."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, '07', 'simplified_matrix_operations')
        self.logger = system_logger.getChild('SimplifiedMatrixOperations')
        
        # Check dependencies
        if not check_dependencies():
            self.logger.warning("⚠️ Some dependencies missing, using fallback implementations")
        
        # Configuration
        self.matrix_config = config.get('matrix_operations_config', {
            'use_gpu': False,  # Disable GPU by default to avoid torch issues
            'use_numba': False,  # Disable numba by default
            'batch_size': 1000,
            'max_memory_mb': 1024
        })
    
    @handles_errors(exceptions=(Exception,), default_return={'success': False})
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simplified matrix operations."""
        self.logger.info("🔢 Starting simplified matrix operations...")
        
        try:
            # Get data
            data_dict = self._get_data_to_process(pipeline_state)
            if not data_dict:
                self.logger.error("❌ No data available for processing")
                return pipeline_state
            
            # Process each split
            matrix_results = {}
            for split_name, data in data_dict.items():
                self.logger.info(f"🧮 Processing {split_name} split...")
                matrices = await self._compute_matrices_simple(data)
                matrix_results[split_name] = matrices
            
            # Update pipeline state
            pipeline_state.update({
                'matrix_results': matrix_results,
                'step07_simplified_completed': True
            })
            
            self.logger.info("✅ Simplified matrix operations completed")
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f"❌ Error in simplified matrix operations: {e}")
            return pipeline_state
    
    def _get_data_to_process(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get data to process with fallback handling."""
        # Try to get engineered data
        if 'engineered_data' in pipeline_state:
            return pipeline_state['engineered_data']
        
        # Try to get advanced features
        if 'advanced_features' in pipeline_state:
            advanced_features = pipeline_state['advanced_features']
            data_dict = {}
            
            for split in ['train', 'val', 'test']:
                if split in advanced_features:
                    path = advanced_features[split]
                    if isinstance(path, str) and Path(path).exists():
                        try:
                            if pd is not None:
                                data_dict[split] = pd.read_parquet(path)
                            else:
                                self.logger.warning(f"⚠️ pandas not available, cannot load {split} data")
                        except Exception as e:
                            self.logger.warning(f"⚠️ Failed to load {split} data: {e}")
            
            if data_dict:
                return data_dict
        
        # Fallback to individual data keys
        data_dict = {}
        for split in ['train', 'val', 'test']:
            if f'{split}_data' in pipeline_state:
                data_dict[split] = pipeline_state[f'{split}_data']
        
        return data_dict
    
    async def _compute_matrices_simple(self, data: Any) -> Dict[str, Any]:
        """Compute matrices with simplified approach."""
        matrices = {}
        
        try:
            if pd is not None and isinstance(data, pd.DataFrame):
                # Get numeric columns
                numeric_cols = data.select_dtypes(include=['number']).columns
                if len(numeric_cols) == 0:
                    self.logger.warning("⚠️ No numeric columns found")
                    return matrices
                
                numeric_data = data[numeric_cols]
                
                # Compute correlation matrix
                try:
                    corr_matrix = numeric_data.corr()
                    matrices['correlation_matrix'] = corr_matrix.values if hasattr(corr_matrix, 'values') else corr_matrix
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to compute correlation matrix: {e}")
                
                # Compute covariance matrix
                try:
                    cov_matrix = numeric_data.cov()
                    matrices['covariance_matrix'] = cov_matrix.values if hasattr(cov_matrix, 'values') else cov_matrix
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to compute covariance matrix: {e}")
                
                # Compute basic statistics
                matrices['feature_stats'] = {
                    'mean': numeric_data.mean().to_dict(),
                    'std': numeric_data.std().to_dict(),
                    'count': numeric_data.count().to_dict()
                }
                
            else:
                self.logger.warning("⚠️ Data is not a pandas DataFrame, skipping matrix computations")
                
        except Exception as e:
            self.logger.error(f"❌ Error in matrix computation: {e}")
        
        return matrices
    
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate inputs with simplified checks."""
        errors = []
        
        # Check for data
        has_data = (
            'engineered_data' in pipeline_state or
            'advanced_features' in pipeline_state or
            any(f'{split}_data' in pipeline_state for split in ['train', 'val', 'test'])
        )
        
        if not has_data:
            errors.append('No data available for processing')
        
        return len(errors) == 0, errors
    
    def get_required_inputs(self) -> List[str]:
        """Get required inputs."""
        return ['engineered_data or split data']
    
    def get_produced_outputs(self) -> List[str]:
        """Get produced outputs."""
        return ['matrix_results']

# Factory function for creating the step
def create_step07_step(config: Dict[str, Any]) -> SimplifiedMatrixOperationsStep:
    """Create a Step07 step instance."""
    return SimplifiedMatrixOperationsStep(config)

if __name__ == "__main__":
    # Test the simplified step
    print("🧪 Testing Simplified Step07")
    print("=" * 40)
    
    config = {
        'matrix_operations_config': {
            'use_gpu': False,
            'use_numba': False,
            'batch_size': 1000
        }
    }
    
    step = create_step07_step(config)
    print(f"✅ Created step: {step.step_name}")
    print(f"📊 Required inputs: {step.get_required_inputs()}")
    print(f"📤 Produced outputs: {step.get_produced_outputs()}")
