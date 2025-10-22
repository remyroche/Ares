"""
Example Enhanced Training Step

This example demonstrates how to use the enhanced BaseStep with all the new utility integrations.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from src.training.steps.base_step import BaseStep


class ExampleEnhancedStep(BaseStep):
    """
    Example step demonstrating the enhanced BaseStep capabilities.
    
    This step shows how to use:
    - Direct utility access
    - Convenience methods
    - Hardware optimization
    - Comprehensive logging
    - Error handling
    """
    
    def __init__(self, step_name: str = "example_enhanced_step", config: Optional[Dict[str, Any]] = None):
        """Initialize the example enhanced step."""
        super().__init__(step_name, config)
        
        # Print utility help to show what's available
        self._print_utility_help()
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the example enhanced step.
        
        This demonstrates various utility usage patterns.
        """
        tprint_step_start("Example Enhanced Step", config)
        
        try:
            # 1. Use convenience methods for common operations
            tprint_info("🔧 Using convenience methods...")
            
            # Safe JSON operations
            test_data = {"test": "value", "number": 42}
            json_saved = self._safe_json_save(test_data, "test_data.json")
            tprint_success(f"JSON saved: {json_saved}")
            
            loaded_data = self._safe_json_load("test_data.json")
            tprint_info(f"JSON loaded: {loaded_data}")
            
            # Safe math operations
            result = self._safe_divide(10, 2, default=0)
            tprint_info(f"Safe division result: {result}")
            
            # Validation
            finite_value = self._validate_finite(3.14, default=0)
            positive_value = self._validate_positive(-5, default=1)
            tprint_info(f"Finite validation: {finite_value}, Positive validation: {positive_value}")
            
            # 2. Use direct utility access
            tprint_info("📦 Using direct utility access...")
            
            if self.common_ops:
                tprint_info("✅ Common operations available")
                # Use common operations directly
                current_time = self.common_ops['get_current_datetime']()
                tprint_info(f"Current time: {current_time}")
            
            if self.math_validation:
                tprint_info("✅ Math validation available")
                # Use math validation directly
                safe_result = self.math_validation['safe_divide'](100, 3, default=0)
                tprint_info(f"Math validation result: {safe_result}")
            
            # 3. Use hardware optimization
            tprint_info("⚡ Using hardware optimization...")
            
            if self.hardware_utils:
                tprint_info("✅ Hardware utilities available")
                # Create sample data
                sample_data = pd.DataFrame({
                    'col1': np.random.randn(1000),
                    'col2': np.random.randn(1000),
                    'col3': np.random.randn(1000)
                })
                
                # Optimize DataFrame
                optimized_data = self.hardware_utils['optimize_dataframe'](sample_data)
                tprint_success(f"DataFrame optimized: {optimized_data.shape}")
            
            # 4. Use ML utilities
            tprint_info("🤖 Using ML utilities...")
            
            if self.ml_common:
                tprint_info("✅ ML common utilities available")
                optimizer = self._get_ml_optimizer("bayesian")
                cv_validator = self._get_cv_validator("time_series")
                tprint_info(f"ML optimizer: {optimizer is not None}, CV validator: {cv_validator is not None}")
            
            # 5. Use data quality utilities
            tprint_info("🧹 Using data quality utilities...")
            
            if self.data_quality:
                tprint_info("✅ Data quality utilities available")
                cleaner = self._get_data_cleaner()
                tprint_info(f"Data cleaner: {cleaner is not None}")
            
            # 6. Use model persistence utilities
            tprint_info("💾 Using model persistence utilities...")
            
            if self.model_persistence:
                tprint_info("✅ Model persistence utilities available")
                cache = self._get_model_cache()
                tprint_info(f"Model cache: {cache is not None}")
            
            # 7. Demonstrate DataFrame operations
            tprint_info("📊 Demonstrating DataFrame operations...")
            
            # Create sample DataFrame
            df = pd.DataFrame({
                'feature1': np.random.randn(100),
                'feature2': np.random.randn(100),
                'target': np.random.randint(0, 2, 100)
            })
            
            # Validate columns
            has_required_cols = self._validate_dataframe_columns(df, ['feature1', 'feature2'])
            tprint_info(f"DataFrame has required columns: {has_required_cols}")
            
            # Safe DataFrame operation
            cleaned_df = self._safe_dataframe_operation(df, "fillna")
            tprint_info(f"DataFrame cleaned: {cleaned_df.shape}")
            
            # 8. Use comprehensive logging
            tprint_info("📝 Using comprehensive logging...")
            
            # Data preview
            tprint_data_preview(df, "sample_dataframe", max_rows=5, level="INFO")
            
            # Performance metrics
            metrics = {
                'rows_processed': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'execution_time': 1.23
            }
            tprint_performance_summary(metrics)
            
            # 9. Check utility availability
            tprint_info("📋 Checking utility availability...")
            availability = self._get_availability_status()
            available_count = sum(availability.values())
            total_count = len(availability)
            tprint_info(f"Utilities available: {available_count}/{total_count}")
            
            # 10. Create artifacts
            tprint_info("💾 Creating artifacts...")
            
            # Save processed data
            self._save_dataframe(cleaned_df, "processed_data")
            
            # Save metrics
            self._save_metadata(metrics, "execution_metrics")
            
            # Create outcome
            outcome = {
                'success': True,
                'artifacts': ['processed_data', 'execution_metrics'],
                'metrics': metrics,
                'utility_availability': availability,
                'execution_time': 1.23
            }
            
            tprint_step_end("Example Enhanced Step", True, 1.23)
            tprint_success("✅ Example enhanced step completed successfully!")
            
            return outcome
            
        except Exception as e:
            tprint_error(f"❌ Example enhanced step failed: {e}")
            tprint_exception(e, "Example Enhanced Step Error")
            
            return {
                'success': False,
                'error': str(e),
                'artifacts': [],
                'metrics': {}
            }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Create step instance
        step = ExampleEnhancedStep()
        
        # Example configuration
        config = {
            'symbol': 'ETHUSDT',
            'exchange': 'binance',
            'timeframe': '15m',
            'execution_mode': 'light'
        }
        
        # Execute step
        result = await step.run(config)
        
        # Print result
        tprint_structured(result, "Step Execution Result")
    
    # Run example
    asyncio.run(main())