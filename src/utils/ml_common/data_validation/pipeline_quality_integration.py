"""
Pipeline Quality Integration Module

This module provides integration hooks and decorators for automatic quality verification
at the end of data collection and at the beginning of each pipeline stage.

Key Features:
- Automatic quality verification hooks
- Decorator-based integration
- Stage beginning quality checks
- Data collection completion quality checks
- Configuration management
- Integration with existing pipeline steps

Built on existing utilities:
- Uses unified_quality_verification.py for quality verification
- Leverages validation_utils.py for validation framework
- Integrates with structured_logging.py for comprehensive logging
"""

import functools
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datetime import datetime
import pandas as pd

from .unified_quality_verification import (
    UnifiedQualityVerifier,
    VerificationStage,
    DataType,
    create_unified_quality_verifier,
    create_pipeline_quality_config
)
from ...validation_utils import ValidationError
from ...structured_logging import StructuredLogger

logger = logging.getLogger(__name__)


class PipelineQualityIntegration:
    """Pipeline quality integration manager."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize pipeline quality integration.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or create_pipeline_quality_config()
        self.logger = logger or logging.getLogger(f"{__name__}.PipelineQualityIntegration")
        self.structured_logger = StructuredLogger(self.logger)

        # Initialize quality verifier
        self.quality_verifier = create_unified_quality_verifier(self.config, self.logger)

        # Integration settings
        self.enable_auto_verification = self.config.get('enable_auto_verification', True)
        self.verification_history: List[Dict[str, Any]] = []

    def data_collection_completion_hook(self, exchange: str, symbol: str, data_type: Optional[DataType] = None):
        """
        Decorator for data collection completion quality verification.

        Args:
            exchange: Exchange name
            symbol: Symbol name
            data_type: Data type (auto-detected if None)

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Extract data from result
                data = None
                if isinstance(result, pd.DataFrame):
                    data = result
                elif isinstance(result, dict) and 'data' in result:
                    data = result['data']
                elif isinstance(result, tuple) and len(result) > 0:
                    data = result[0] if isinstance(result[0], pd.DataFrame) else None
                
                if data is not None and self.enable_auto_verification:
                    try:
                        self.logger.info(f"🔍 Auto-verifying data collection completion for {exchange}_{symbol}")
                        
                        # Verify quality
                        cleaned_data, quality_report = self.quality_verifier.verify_data_collection_completion(
                            data, exchange, symbol, data_type
                        )
                        
                        # Store verification history
                        self.verification_history.append({
                            'timestamp': datetime.now(),
                            'stage': 'data_collection_completion',
                            'exchange': exchange,
                            'symbol': symbol,
                            'data_type': quality_report.data_type.value,
                            'quality_score': quality_report.quality_score,
                            'issues_count': len(quality_report.issues),
                            'total_rows': quality_report.total_rows
                        })
                        
                        # Log results
                        self.logger.info(f"✅ Data collection quality verification completed")
                        self.logger.info(f"   Quality score: {quality_report.quality_score:.3f}")
                        self.logger.info(f"   Issues found: {len(quality_report.issues)}")
                        
                        # Return cleaned data if original result was just data
                        if isinstance(result, pd.DataFrame):
                            return cleaned_data
                        elif isinstance(result, dict):
                            result['data'] = cleaned_data
                            result['quality_report'] = quality_report
                            return result
                        elif isinstance(result, tuple):
                            return (cleaned_data,) + result[1:]
                        
                    except Exception as e:
                        self.logger.error(f"❌ Data collection quality verification failed: {e}")
                        # Continue with original result if verification fails
                
                return result
            return wrapper
        return decorator

    def stage_beginning_hook(self, stage_name: str, data_type: Optional[DataType] = None):
        """
        Decorator for stage beginning quality verification.

        Args:
            stage_name: Name of the stage
            data_type: Data type (auto-detected if None)

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract data from arguments
                data = None
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        data = arg
                        break
                
                if 'data' in kwargs and isinstance(kwargs['data'], pd.DataFrame):
                    data = kwargs['data']
                
                if data is not None and self.enable_auto_verification:
                    try:
                        self.logger.info(f"🔍 Auto-verifying stage beginning for: {stage_name}")
                        
                        # Verify quality
                        cleaned_data, quality_report = self.quality_verifier.verify_stage_beginning(
                            data, stage_name, data_type
                        )
                        
                        # Store verification history
                        self.verification_history.append({
                            'timestamp': datetime.now(),
                            'stage': 'stage_beginning',
                            'stage_name': stage_name,
                            'data_type': quality_report.data_type.value,
                            'quality_score': quality_report.quality_score,
                            'issues_count': len(quality_report.issues),
                            'total_rows': quality_report.total_rows
                        })
                        
                        # Log results
                        self.logger.info(f"✅ Stage beginning quality verification completed")
                        self.logger.info(f"   Quality score: {quality_report.quality_score:.3f}")
                        self.logger.info(f"   Issues found: {len(quality_report.issues)}")
                        
                        # Update data in arguments
                        if 'data' in kwargs:
                            kwargs['data'] = cleaned_data
                        else:
                            # Replace first DataFrame argument
                            for i, arg in enumerate(args):
                                if isinstance(arg, pd.DataFrame):
                                    args = list(args)
                                    args[i] = cleaned_data
                                    args = tuple(args)
                                    break
                        
                    except Exception as e:
                        self.logger.error(f"❌ Stage beginning quality verification failed: {e}")
                        # Continue with original data if verification fails
                
                # Execute the original function with potentially cleaned data
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def quality_gate(self, min_quality_score: float = 0.8, stage_name: str = "quality_gate"):
        """
        Decorator for quality gate enforcement.

        Args:
            min_quality_score: Minimum required quality score
            stage_name: Name of the stage

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract data from arguments
                data = None
                for arg in args:
                    if isinstance(arg, pd.DataFrame):
                        data = arg
                        break
                
                if 'data' in kwargs and isinstance(kwargs['data'], pd.DataFrame):
                    data = kwargs['data']
                
                if data is not None and self.enable_auto_verification:
                    try:
                        self.logger.info(f"🚪 Quality gate check for {stage_name} (min score: {min_quality_score})")
                        
                        # Verify quality
                        cleaned_data, quality_report = self.quality_verifier.verify_data_quality(
                            data, VerificationStage.CUSTOM, None, {'stage_name': stage_name}
                        )
                        
                        # Check quality gate
                        if quality_report.quality_score < min_quality_score:
                            error_msg = f"Quality gate failed for {stage_name}: {quality_report.quality_score:.3f} < {min_quality_score}"
                            self.logger.error(f"❌ {error_msg}")
                            raise ValidationError(error_msg, "quality_gate_failed", {
                                'stage_name': stage_name,
                                'quality_score': quality_report.quality_score,
                                'min_required': min_quality_score,
                                'issues_count': len(quality_report.issues)
                            })
                        
                        self.logger.info(f"✅ Quality gate passed for {stage_name}: {quality_report.quality_score:.3f}")
                        
                        # Update data in arguments
                        if 'data' in kwargs:
                            kwargs['data'] = cleaned_data
                        else:
                            # Replace first DataFrame argument
                            for i, arg in enumerate(args):
                                if isinstance(arg, pd.DataFrame):
                                    args = list(args)
                                    args[i] = cleaned_data
                                    args = tuple(args)
                                    break
                        
                    except ValidationError:
                        raise
                    except Exception as e:
                        self.logger.error(f"❌ Quality gate verification failed: {e}")
                        # Continue with original data if verification fails
                
                # Execute the original function
                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def get_verification_summary(self) -> Dict[str, Any]:
        """Get summary of all quality verifications performed."""
        if not self.verification_history:
            return {'message': 'No verifications performed yet'}
        
        # Calculate statistics
        total_verifications = len(self.verification_history)
        avg_quality_score = sum(v['quality_score'] for v in self.verification_history) / total_verifications
        min_quality_score = min(v['quality_score'] for v in self.verification_history)
        max_quality_score = max(v['quality_score'] for v in self.verification_history)
        
        # Count by stage
        stage_counts = {}
        for verification in self.verification_history:
            stage = verification.get('stage_name', verification['stage'])
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        # Count by data type
        data_type_counts = {}
        for verification in self.verification_history:
            data_type = verification['data_type']
            data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1
        
        return {
            'total_verifications': total_verifications,
            'quality_scores': {
                'average': avg_quality_score,
                'minimum': min_quality_score,
                'maximum': max_quality_score
            },
            'stage_counts': stage_counts,
            'data_type_counts': data_type_counts,
            'verification_history': self.verification_history
        }

    def export_verification_summary(self, filepath: str) -> None:
        """Export verification summary to JSON file."""
        import json
        
        summary = self.get_verification_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"📄 Verification summary exported to: {filepath}")


# Global instance for easy access
_global_quality_integration = None


def get_quality_integration(config: Optional[Dict[str, Any]] = None) -> PipelineQualityIntegration:
    """Get global quality integration instance."""
    global _global_quality_integration
    if _global_quality_integration is None:
        _global_quality_integration = PipelineQualityIntegration(config)
    return _global_quality_integration


# Convenience decorators
def verify_data_collection_quality(exchange: str, symbol: str, data_type: Optional[DataType] = None):
    """Convenience decorator for data collection quality verification."""
    integration = get_quality_integration()
    return integration.data_collection_completion_hook(exchange, symbol, data_type)


def verify_stage_beginning_quality(stage_name: str, data_type: Optional[DataType] = None):
    """Convenience decorator for stage beginning quality verification."""
    integration = get_quality_integration()
    return integration.stage_beginning_hook(stage_name, data_type)


def enforce_quality_gate(min_quality_score: float = 0.8, stage_name: str = "quality_gate"):
    """Convenience decorator for quality gate enforcement."""
    integration = get_quality_integration()
    return integration.quality_gate(min_quality_score, stage_name)


# Example usage functions
def example_data_collection_integration():
    """Example of data collection integration."""
    
    @verify_data_collection_quality("binance", "BTCUSDT", DataType.AGGRADES)
    async def collect_aggtrades_data(exchange: str, symbol: str) -> pd.DataFrame:
        """Collect aggtrades data with automatic quality verification."""
        # Your data collection logic here
        data = pd.DataFrame()  # Placeholder
        return data
    
    @verify_data_collection_quality("binance", "BTCUSDT", DataType.KLINES)
    async def collect_klines_data(exchange: str, symbol: str) -> pd.DataFrame:
        """Collect klines data with automatic quality verification."""
        # Your data collection logic here
        data = pd.DataFrame()  # Placeholder
        return data


def example_stage_integration():
    """Example of stage integration."""
    
    @verify_stage_beginning_quality("preprocessing", DataType.AGGRADES)
    async def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with automatic quality verification."""
        # Your preprocessing logic here
        return data
    
    @verify_stage_beginning_quality("feature_engineering", DataType.KLINES)
    async def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features with automatic quality verification."""
        # Your feature engineering logic here
        return data
    
    @enforce_quality_gate(0.9, "model_training")
    async def train_model(data: pd.DataFrame) -> Any:
        """Train model with quality gate enforcement."""
        # Your model training logic here
        return None


def example_pipeline_integration():
    """Example of complete pipeline integration."""
    
    # Data collection with quality verification
    @verify_data_collection_quality("binance", "BTCUSDT")
    async def collect_data() -> pd.DataFrame:
        """Collect data with quality verification."""
        return pd.DataFrame()
    
    # Preprocessing with quality verification
    @verify_stage_beginning_quality("preprocessing")
    async def preprocess(data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess with quality verification."""
        return data
    
    # Feature engineering with quality verification
    @verify_stage_beginning_quality("feature_engineering")
    async def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features with quality verification."""
        return data
    
    # Model training with quality gate
    @enforce_quality_gate(0.85, "model_training")
    async def train_model(data: pd.DataFrame) -> Any:
        """Train model with quality gate."""
        return None
    
    # Complete pipeline
    async def run_pipeline():
        """Run complete pipeline with quality verification."""
        data = await collect_data()
        processed_data = await preprocess(data)
        features = await engineer_features(processed_data)
        model = await train_model(features)
        
        # Get verification summary
        integration = get_quality_integration()
        summary = integration.get_verification_summary()
        print(f"Pipeline completed with {summary['total_verifications']} quality verifications")
        
        return model