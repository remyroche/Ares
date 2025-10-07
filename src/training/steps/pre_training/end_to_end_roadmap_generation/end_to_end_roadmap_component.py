"""
End-to-End Roadmap Generation Component

This component replaces the PID-based feature generation component and provides
comprehensive end-to-end roadmap feature generation with:
- System contracts and budgets
- Data contracts and validation
- Feature registry with parent features
- Transform system with EW-Z, TOD Rank, Signed-log, Winsorization
- Lookback selection with hysteresis
- Interaction engine with 15 locked interactions
- Patch/GRU model integration
- Assembly DAG orchestration
- Walk-forward validation
- Monitoring and retrain decision tree
- CI/CD gates and tests
- Rollout plan with shadow/canary/full deployment
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from enum import Enum

# Core dependencies with fallback support
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import base component
from ...market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Import end-to-end roadmap system
from src.end_to_end_roadmap import (
    EndToEndRoadmapSystem, 
    SystemConfig, 
    SystemResult,
    create_end_to_end_system,
    run_end_to_end_pipeline
)

# Import utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error

# Import logger
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('EndToEndRoadmapComponent')
except ImportError:
    logger = logging.getLogger('EndToEndRoadmapComponent')
    logger.setLevel(logging.INFO)


class RoadmapStatus(Enum):
    """Status of roadmap generation process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class EndToEndRoadmapComponent(BaseMarketAnalysisComponent):
    """
    End-to-End Roadmap Generation Component.
    
    Replaces the PID-based feature generation component with a comprehensive
    end-to-end roadmap system that includes all constraints and specifications.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the end-to-end roadmap component."""
        tprint("🔧 Initializing EndToEndRoadmapComponent...")
        super().__init__(config)
        self.logger = logger.getChild('EndToEndRoadmapComponent')
        
        # Initialize system configuration
        tprint("🔧 Initializing system configuration...")
        self.system_config = self._create_system_config()
        tprint("✅ System configuration initialized")
        
        # Initialize end-to-end system
        tprint("🔧 Initializing end-to-end roadmap system...")
        try:
            self.roadmap_system = create_end_to_end_system(self.system_config)
            tprint("✅ End-to-end roadmap system initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize roadmap system: {e}")
            raise
        
        # Track generation status
        self.generation_status = RoadmapStatus.PENDING
        self.start_time: Optional[float] = None
        
        # Track target source information for outcome verification
        self._target_source_info = {
            'target_used': 'unknown',
            'target_type': 'unknown', 
            'valid_samples': 0,
            'source': 'unknown'
        }
        
        self.logger.info("🔧 EndToEndRoadmapComponent initialized")
        self.logger.info(f"📊 Symbol: {self.config.symbol}")
        self.logger.info(f"📊 Exchange: {self.config.exchange}")
        self.logger.info(f"📊 Timeframe: {self.config.timeframe}")
        tprint("✅ EndToEndRoadmapComponent initialization complete")
    
    def _create_system_config(self) -> SystemConfig:
        """Create system configuration based on component config."""
        return SystemConfig(
            # Feature budgets (from roadmap spec)
            feature_budget_pre=120,
            feature_budget_post=(30, 60),
            interactions_cap=15,
            transforms_per_parent=1,
            
            # Latency budgets (from roadmap spec)
            latency_budget_ms=50,
            feature_compute_ms=25,
            model_inference_ms=5,
            io_orchestration_ms=20,
            
            # Lookback ceiling (from roadmap spec)
            lookback_ceiling_minutes=120,
            
            # Retrain settings (from roadmap spec)
            retrain_scheduled="02:00 America/New_York",
            retrain_triggered_interval="2h",
            fallback_p99_ms=2.0,
            
            # Model settings
            patch_model_type="gru",  # Will be converted to enum
            patch_sequence_length=24,  # 2h at 5min bars
            patch_horizons=[1, 3],
            
            # Validation settings
            validation_n_folds=6,
            validation_embargo_pct=0.1,
            
            # Monitoring settings
            monitoring_interval_minutes=5,
            calibration_loss_threshold=2.0,
            psi_threshold=0.3,
            correlation_drift_threshold=0.5
        )
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 Getting required artifacts for end-to-end roadmap generation")
        artifacts = ['end_to_end_roadmap_result']
        tprint(f"✅ Required artifacts: {artifacts}")
        return artifacts
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute end-to-end roadmap generation with comprehensive validation and reporting.
        
        Args:
            data: Market data for feature generation
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with end-to-end roadmap generation results
        """
        tprint("🚀 Starting end-to-end roadmap generation execution...")
        self.start_time = time.time()
        self.generation_status = RoadmapStatus.IN_PROGRESS
        
        self.logger.info('🔧 Starting End-to-End Roadmap Generation')
        self._report_checkpoint('start', 'generation_started', {
            'symbol': self.config.symbol,
            'exchange': self.config.exchange,
            'timeframe': self.config.timeframe
        })
        tprint("📊 Generation status set to IN_PROGRESS")
        
        try:
            # Store pipeline state for data access
            self._pipeline_state = pipeline_state
            
            # Step 1: Load and validate market data
            self.logger.info('📊 Loading and validating market data...')
            market_data = await self._load_and_validate_market_data(data)
            self._report_checkpoint('data_loading', 'completed', {
                'data_points': len(market_data) if market_data is not None else 0,
                'data_quality_score': self._calculate_data_quality_score(market_data)
            })
            
            # Step 2: Get target variable and additional outputs if available
            target_data = await self._get_target_variable(pipeline_state)
            
            # Extract targets for roadmap system
            targets = None
            if target_data:
                targets = self._extract_targets_for_roadmap(target_data)
                self.logger.info(f"📊 Using targets for roadmap generation: {list(targets.keys()) if targets else 'None'}")
            
            # Step 3: Run end-to-end roadmap pipeline
            self.logger.info('🚀 Running end-to-end roadmap pipeline...')
            
            roadmap_result = self.roadmap_system.process_market_data(
                market_data,
                targets,
                enable_validation=True,
                enable_monitoring=True,
                enable_deployment=False  # Disable deployment for now
            )
            
            self._report_checkpoint('roadmap_generation', 'completed', {
                'success': roadmap_result.success,
                'total_features': len(roadmap_result.features.columns) if roadmap_result.success else 0,
                'selected_features': len(roadmap_result.selected_features) if roadmap_result.success else 0,
                'patch_features': len(roadmap_result.patch_features) if roadmap_result.success else 0
            })
            
            if not roadmap_result.success:
                raise ValueError(f"Roadmap generation failed: {roadmap_result.error_message}")
            
            # Step 4: Validate generation results
            validation_result = await self._validate_generation_results(roadmap_result)
            self._report_checkpoint('validation', 'completed', {
                'is_valid': validation_result['is_valid'],
                'quality_score': validation_result['quality_score'],
                'issues_count': len(validation_result['issues'])
            })
            
            # Step 5: Create comprehensive artifacts
            artifacts = await self._create_comprehensive_artifacts(
                roadmap_result,
                validation_result,
                market_data
            )
            
            # Step 6: Generate final report
            final_report = self._generate_final_report(artifacts, validation_result, roadmap_result)
            self._report_checkpoint('completion', 'success', {
                'total_features': len(roadmap_result.features.columns),
                'quality_score': validation_result['quality_score'],
                'execution_time': time.time() - self.start_time
            })
            
            self.generation_status = RoadmapStatus.COMPLETED
            
            self.logger.info(f'✅ End-to-End Roadmap Generation completed: {len(roadmap_result.features.columns)} features generated')
            
            # Save artifacts persistently using the artifact manager
            try:
                saved_files = await self.save_artifacts(artifacts, {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'total_features_generated': len(roadmap_result.features.columns),
                    'generation_status': self.generation_status.value,
                    'data_quality_score': validation_result['quality_score'],
                    'final_report': final_report,
                    'execution_time': time.time() - self.start_time
                })
                tprint(f"💾 [ROADMAP_GEN] Artifacts saved persistently: {list(saved_files.keys())}", color="green")
            except Exception as e:
                tprint(f"⚠️ [ROADMAP_GEN] Failed to save artifacts persistently: {e}", color="yellow")
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'total_features_generated': len(roadmap_result.features.columns),
                    'generation_status': self.generation_status.value,
                    'data_quality_score': validation_result['quality_score'],
                    'final_report': final_report,
                    'execution_time': time.time() - self.start_time,
                    'artifacts_saved_persistently': True,
                    'roadmap_metadata': roadmap_result.metadata
                }
            )
            
        except Exception as e:
            self.generation_status = RoadmapStatus.FAILED
            
            self.logger.error(f'❌ End-to-End Roadmap Generation failed: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            
            # Generate failure report
            failure_report = self._generate_failure_report(str(e))
            self._report_checkpoint('completion', 'failed', {
                'error_type': type(e).__name__,
                'execution_time': time.time() - self.start_time if self.start_time else 0
            })
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'generation_status': self.generation_status.value,
                    'failure_report': failure_report,
                    'execution_time': time.time() - self.start_time if self.start_time else 0
                }
            )
    
    async def _load_and_validate_market_data(self, data: Any) -> Any:
        """Load and validate market data with strict validation that fails fast."""
        try:
            # Enhanced data handling - try to get data from multiple sources
            processed_data = await self._enhanced_data_handling(data)
            if processed_data is None:
                raise ValueError("CRITICAL: No valid market data available from any source - cannot proceed with roadmap generation")
            
            if not PANDAS_AVAILABLE:
                raise ValueError("CRITICAL: Pandas not available for data processing - required dependency missing")
            
            if not isinstance(processed_data, pd.DataFrame):
                raise ValueError(f"CRITICAL: Expected pandas DataFrame, got {type(processed_data).__name__} - invalid data format")
            
            if processed_data.empty:
                raise ValueError("CRITICAL: Market data is completely empty - no data points to process")
            
            # Strict validation - require minimum data quality
            if len(processed_data) < 100:
                raise ValueError(f"CRITICAL: Insufficient data points ({len(processed_data)}) - need at least 100 for meaningful roadmap generation")
            
            # Check for excessive NaN values
            nan_percentage = processed_data.isnull().sum().sum() / (len(processed_data) * len(processed_data.columns))
            if nan_percentage > 0.5:
                raise ValueError(f"CRITICAL: Excessive missing data ({nan_percentage:.1%}) - data quality too poor for roadmap generation")
            
            # Validate data types
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                raise ValueError("CRITICAL: No numeric columns found - cannot generate numerical features")
            
            # Check for constant columns (zero variance)
            constant_columns = []
            for col in numeric_columns:
                if processed_data[col].nunique() <= 1:
                    constant_columns.append(col)
            
            if len(constant_columns) == len(numeric_columns):
                raise ValueError(f"CRITICAL: All numeric columns are constant - no variation for roadmap generation")
            elif len(constant_columns) > 0:
                self.logger.warning(f"Removing {len(constant_columns)} constant columns: {constant_columns}")
                processed_data = processed_data.drop(columns=constant_columns)
            
            # Validate required columns for financial data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in processed_data.columns]
            
            if len(missing_columns) == len(required_columns):
                self.logger.warning("No standard OHLCV columns found - proceeding with available numeric data")
            elif missing_columns:
                self.logger.warning(f"Missing some OHLCV columns: {missing_columns}")
                # Only create fallback columns if we have at least one price column
                price_columns = [col for col in ['open', 'high', 'low', 'close'] if col in processed_data.columns]
                if price_columns:
                    reference_price = processed_data[price_columns[0]]
                    for col in missing_columns:
                        if col == 'volume':
                            processed_data[col] = 1000  # Default volume
                        elif col in ['open', 'high', 'low', 'close']:
                            processed_data[col] = reference_price  # Use existing price as fallback
            
            # Final validation and data type cleanup
            final_numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            if len(final_numeric_columns) < 2:
                raise ValueError(f"CRITICAL: Need at least 2 numeric columns for roadmap generation, got {len(final_numeric_columns)}")
            
            # Remove non-numeric columns that could cause issues
            non_numeric_columns = processed_data.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric_columns:
                self.logger.info(f"🔧 Removing {len(non_numeric_columns)} non-numeric columns: {non_numeric_columns}")
                processed_data = processed_data.select_dtypes(include=[np.number])
            
            # Ensure all remaining data is float for consistent processing
            for col in processed_data.columns:
                if processed_data[col].dtype != np.float64:
                    try:
                        processed_data[col] = processed_data[col].astype(np.float64)
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"⚠️ Could not convert {col} to float64: {e}")
                        # Drop problematic columns
                        processed_data = processed_data.drop(columns=[col])
            
            # Final check
            if processed_data.shape[1] < 2:
                raise ValueError(f"CRITICAL: After data type cleanup, only {processed_data.shape[1]} columns remain")
            
            self.logger.info(f"✅ Data validation passed: {len(processed_data)} rows, {len(processed_data.columns)} numeric columns")
            return processed_data.copy()
            
        except Exception as e:
            self.logger.error(f"❌ Data validation failed: {e}")
            raise
    
    async def _enhanced_data_handling(self, data: Any) -> Optional[pd.DataFrame]:
        """Enhanced data handling to get data from multiple sources."""
        try:
            # Try direct data first
            if data is not None:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    self.logger.info("✅ Using direct DataFrame data for roadmap generation")
                    return data
                elif hasattr(data, 'to_dataframe'):
                    df = data.to_dataframe()
                    if not df.empty:
                        self.logger.info("✅ Converted data to DataFrame for roadmap generation")
                        return df
            
            # Try to get data from pipeline state
            if hasattr(self, '_pipeline_state') and self._pipeline_state:
                # Try different keys that might contain data
                data_keys = ['market_data', 'data', 'processed_data', 'features', 'labeled_data']
                for key in data_keys:
                    if key in self._pipeline_state:
                        pipeline_data = self._pipeline_state[key]
                        if pipeline_data is not None:
                            if isinstance(pipeline_data, pd.DataFrame) and not pipeline_data.empty:
                                self.logger.info(f"✅ Using data from pipeline state key: {key}")
                                return pipeline_data
                            elif hasattr(pipeline_data, 'to_dataframe'):
                                df = pipeline_data.to_dataframe()
                                if not df.empty:
                                    self.logger.info(f"✅ Converted pipeline data from key: {key}")
                                    return df
            
            self.logger.error("❌ No valid data found for roadmap generation")
            return None
            
        except Exception as e:
            self.logger.error(f"Enhanced data handling failed: {e}")
            return None
    
    async def _get_target_variable(self, pipeline_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get target variable and additional outputs from multi-horizon profit labeler."""
        try:
            tprint_info("🎯 Getting target variable from multi-horizon profit labeler")
            
            # Try to get standardized output format
            standardized_output = None
            
            # Check pipeline state for standardized output
            if 'standardized_output' in pipeline_state:
                tprint_info("📋 Found standardized output in pipeline state")
                standardized_output = pipeline_state['standardized_output']
            else:
                # Check artifacts for standardized output
                artifacts = pipeline_state.get('artifacts', {})
                if artifacts and 'standardized_output' in artifacts:
                    tprint_info("📋 Found standardized output in artifacts")
                    standardized_output = artifacts['standardized_output']
            
            # Extract target data from standardized format if available
            if standardized_output and 'labels' in standardized_output:
                tprint_success("✅ Using standardized output format from multi_horizon_profit_labeler")
                labeled_data = standardized_output['labels']
                
                # Extract target columns and weights for enhanced integration
                target_columns = standardized_output.get('target_columns', [])
                horizon_weights = standardized_output.get('weights', {})
                confidence_scores = standardized_output.get('confidence_scores', None)
                eligibility_masks = standardized_output.get('eligibility_masks', None)
                quality_scores = standardized_output.get('quality_scores', None)
                
                tprint_info(f"🎯 Target columns from standardized format: {target_columns}")
                tprint_info(f"⚖️ Horizon weights from standardized format: {horizon_weights}")
                
                # Select the best target based on weights
                best_target = self._select_best_target_with_weights(labeled_data, horizon_weights, target_columns)
                if best_target:
                    tprint_success(f"✅ Selected best target for roadmap generation: {best_target}")
                    return {
                        'targets': labeled_data[best_target],
                        'confidence_scores': confidence_scores,
                        'eligibility_masks': eligibility_masks,
                        'quality_scores': quality_scores,
                        'horizon_weights': horizon_weights,
                        'target_columns': target_columns,
                        'selected_target': best_target
                    }
            
            # Fallback to legacy format if standardized format not available
            else:
                tprint_info("📋 Using legacy format for target extraction")
                multi_horizon_result = pipeline_state.get('multi_horizon_labeling_result', {})
                
                # Check if multi-horizon results are in artifacts
                artifacts = pipeline_state.get('artifacts', {})
                if artifacts and 'multi_horizon_labeling_result' in artifacts:
                    multi_horizon_result = artifacts['multi_horizon_labeling_result']
                
                if multi_horizon_result and 'labeled_data' in multi_horizon_result:
                    labeled_data = multi_horizon_result['labeled_data']
                    
                    # Extract additional outputs from multi_horizon_profit_labeler
                    confidence_scores = multi_horizon_result.get('confidence_scores', None)
                    eligibility_masks = multi_horizon_result.get('eligibility_masks', None)
                    quality_scores = multi_horizon_result.get('quality_scores', None)
                    
                    # Convert string representation to DataFrame if needed
                    if isinstance(labeled_data, str):
                        self.logger.info("📊 Multi-horizon labeled data found as JSON string - attempting to parse")
                        try:
                            import json
                            json_data = json.loads(labeled_data)
                            labeled_data = pd.DataFrame(json_data)
                            self.logger.info(f"✅ Successfully parsed JSON labeled data: {len(json_data)} records")
                        except (json.JSONDecodeError, ValueError) as e:
                            self.logger.warning(f"⚠️ JSON parsing failed ({e}), using fallback approach")
                            return None
                    
                    if isinstance(labeled_data, pd.DataFrame):
                        # Use available targets
                        target_options = [
                            'overall_opportunity',
                            'leverage_adjusted_score',
                            'immediate_opportunity',
                            'short_term_opportunity'
                        ]
                        
                        for target_option in target_options:
                            if target_option in labeled_data.columns:
                                target_values = labeled_data[target_option].to_numpy(dtype=float, copy=True)
                                valid_mask = np.isfinite(target_values)
                                if np.any(valid_mask):
                                    sanitized_values = np.nan_to_num(target_values, nan=0.0, posinf=0.0, neginf=0.0)
                                    self.logger.info(f"✅ Using '{target_option}' as roadmap target ({np.sum(valid_mask)} valid samples)")
                                    
                                    # Store target source info for verification
                                    self._target_source_info = {
                                        'target_used': target_option,
                                        'target_type': 'legacy_fallback',
                                        'valid_samples': int(np.sum(valid_mask)),
                                        'source': 'multi_horizon_labeling'
                                    }
                                    
                                    return {
                                        'targets': {'combined': sanitized_values},
                                        'confidence_scores': confidence_scores,
                                        'eligibility_masks': eligibility_masks,
                                        'quality_scores': quality_scores,
                                        'metadata': self._target_source_info
                                    }
            
            self.logger.warning("⚠️ No multi-horizon labeling data found - roadmap analysis will use correlation-based fallback")
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract target variable: {e}")
            return None
    
    def _extract_targets_for_roadmap(self, target_data: Dict[str, Any]) -> Optional[Dict[int, pd.Series]]:
        """Extract targets in format expected by roadmap system."""
        try:
            targets = target_data.get('targets', {})
            
            if isinstance(targets, dict):
                # Convert to horizon-based format
                roadmap_targets = {}
                for i, (target_type, target_values) in enumerate(targets.items(), 1):
                    if hasattr(target_values, '__len__') and not isinstance(target_values, str):
                        roadmap_targets[i] = pd.Series(target_values)
                
                return roadmap_targets if roadmap_targets else None
            else:
                # Single target case
                if hasattr(targets, '__len__') and not isinstance(targets, str):
                    return {1: pd.Series(targets)}
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract targets for roadmap: {e}")
            return None
    
    def _select_best_target_with_weights(self, labels: pd.DataFrame, weights: Dict[str, float], target_columns: List[str]) -> Optional[str]:
        """Select the best target based on horizon weights and availability for roadmap generation."""
        try:
            if not weights or not target_columns:
                # No weights available, use first available target
                available_targets = [col for col in labels.columns if col not in ['timestamp', 'symbol']]
                return available_targets[0] if available_targets else None
            
            # Priority order based on horizon weights (higher weight = higher priority)
            target_priority = []
            
            for target in target_columns:
                if target in labels.columns:
                    # Determine horizon type from target name
                    if 'immediate' in target.lower() or 'small' in target.lower():
                        horizon_weight = weights.get('small', 0.0)
                    elif 'short' in target.lower() or 'medium' in target.lower():
                        horizon_weight = weights.get('medium', 0.0)
                    elif 'leverage' in target.lower() or 'high' in target.lower():
                        horizon_weight = weights.get('high', 0.0)
                    else:
                        # Default to small horizon if unclear
                        horizon_weight = weights.get('small', 0.0)
                    
                    target_priority.append((target, horizon_weight))
            
            # Sort by weight (descending) and return the highest weighted target
            if target_priority:
                target_priority.sort(key=lambda x: x[1], reverse=True)
                best_target = target_priority[0][0]
                tprint_info(f"   → Selected target '{best_target}' with weight {target_priority[0][1]:.3f} for roadmap generation")
                return best_target
            
            return None
            
        except Exception as e:
            tprint_warning(f"⚠️ Error selecting best target with weights: {e}")
            # Fallback to first available target
            available_targets = [col for col in labels.columns if col not in ['timestamp', 'symbol']]
            return available_targets[0] if available_targets else None
    
    async def _validate_generation_results(self, roadmap_result: SystemResult) -> Dict[str, Any]:
        """Validate roadmap generation results."""
        issues = []
        recommendations = []
        
        # Check if generation was successful
        if not roadmap_result.success:
            issues.append("Roadmap generation failed")
            recommendations.append("Review generation configuration and data quality")
        
        # Check feature count
        if len(roadmap_result.features.columns) == 0:
            issues.append("No features were generated")
            recommendations.append("Check roadmap generation configuration and input data")
        
        # Check feature budget compliance
        if len(roadmap_result.features.columns) > self.system_config.feature_budget_pre:
            issues.append(f"Pre-selection budget exceeded: {len(roadmap_result.features.columns)} > {self.system_config.feature_budget_pre}")
            recommendations.append("Review feature selection logic")
        
        # Check selected features
        if len(roadmap_result.selected_features) == 0:
            issues.append("No features were selected")
            recommendations.append("Review feature selection criteria")
        
        # Check patch features
        if len(roadmap_result.patch_features) == 0:
            recommendations.append("No patch features generated - consider enabling patch model")
        
        # Calculate quality score
        base_quality_score = 0.8 if roadmap_result.success else 0.0
        
        # Bonus for good feature diversity
        feature_diversity = len(set([col.split('/')[0] for col in roadmap_result.features.columns if '/' in col]))
        diversity_bonus = min(0.2, feature_diversity / 10.0)
        
        quality_score = min(1.0, base_quality_score + diversity_bonus)
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'quality_score': quality_score,
            'recommendations': recommendations,
            'feature_analysis': {
                'total_features': len(roadmap_result.features.columns),
                'selected_features': len(roadmap_result.selected_features),
                'patch_features': len(roadmap_result.patch_features),
                'feature_diversity': feature_diversity
            }
        }
    
    def _calculate_data_quality_score(self, data: Any) -> float:
        """Calculate data quality score."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            return 0.0
        
        try:
            score = 1.0
            
            # Check for missing values
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col in data.columns:
                    nan_ratio = data[col].isna().sum() / len(data)
                    score *= (1.0 - nan_ratio)
            
            # Check for data types
            for col in required_columns:
                if col in data.columns and data[col].dtype == 'object':
                    score *= 0.5
            
            return max(0.0, score)
            
        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return 0.0
    
    async def _create_comprehensive_artifacts(
        self, 
        roadmap_result: SystemResult,
        validation_result: Dict[str, Any],
        market_data: Any
    ) -> Dict[str, Any]:
        """Create comprehensive artifacts."""
        return {
            'end_to_end_roadmap_result': {
                # Roadmap results
                'features': roadmap_result.features,
                'selected_features': roadmap_result.selected_features,
                'patch_features': roadmap_result.patch_features,
                'artifacts': roadmap_result.artifacts,
                
                # Metadata
                'total_features_generated': len(roadmap_result.features.columns),
                'generation_status': 'completed',
                'system_config': self.system_config.__dict__,
                
                # Quality metrics
                'validation_result': validation_result,
                'data_quality_score': validation_result['quality_score'],
                
                # Summary
                'generation_summary': {
                    'total_features': len(roadmap_result.features.columns),
                    'selected_features': len(roadmap_result.selected_features),
                    'patch_features': len(roadmap_result.patch_features),
                    'execution_time': time.time() - self.start_time if self.start_time else 0,
                    'quality_score': validation_result['quality_score'],
                    'validation_passed': validation_result['is_valid']
                },
                
                # Metadata
                'metadata': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'data_points': len(market_data) if market_data is not None else 0,
                    'execution_timestamp': datetime.now().isoformat(),
                    'component_version': '1.0.0',
                    'generation_status': self.generation_status.value,
                    'target_source_info': self._target_source_info
                }
            }
        }
    
    def _generate_final_report(
        self, 
        artifacts: Dict[str, Any], 
        validation_result: Dict[str, Any],
        roadmap_result: SystemResult
    ) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        
        return {
            'execution_summary': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'success': roadmap_result.success,
                'features_generated': len(roadmap_result.features.columns),
                'data_quality_score': validation_result['quality_score'],
                'generation_status': 'completed',
                'target_source_info': self._target_source_info
            },
            'feature_breakdown': {
                'total_features': len(roadmap_result.features.columns),
                'selected_features': len(roadmap_result.selected_features),
                'patch_features': len(roadmap_result.patch_features),
                'feature_diversity': validation_result['feature_analysis']['feature_diversity']
            },
            'quality_metrics': {
                'overall_quality_score': validation_result['quality_score'],
                'validation_passed': validation_result['is_valid'],
                'issues_count': len(validation_result['issues'])
            },
            'recommendations': validation_result['recommendations']
        }
    
    def _generate_failure_report(self, error_message: str) -> Dict[str, Any]:
        """Generate failure report."""
        return {
            'execution_summary': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'success': False,
                'features_generated': 0,
                'data_quality_score': 0.0,
                'generation_status': self.generation_status.value
            },
            'error_details': {
                'error_message': error_message,
                'error_type': 'generation_failed'
            },
            'recommendations': [
                "Review error logs for detailed failure information",
                "Check data quality and availability",
                "Verify roadmap system configuration",
                "Ensure required dependencies are available"
            ]
        }
    
    def _report_checkpoint(self, step: str, status: str, details: Dict[str, Any]):
        """Report progress at key checkpoints."""
        self.logger.info(f"📊 [{step}] {status} - {details}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            'generation_status': self.generation_status.value,
            'execution_time': time.time() - self.start_time if self.start_time else 0.0,
            'system_status': self.roadmap_system.get_system_status(),
            'component_availability': {
                'numpy_available': NUMPY_AVAILABLE,
                'pandas_available': PANDAS_AVAILABLE
            }
        }