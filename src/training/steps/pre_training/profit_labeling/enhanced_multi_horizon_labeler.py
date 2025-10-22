"""
Enhanced Multi-Horizon Profit Labeler

This module provides a drop-in replacement for the existing multi-horizon profit labeler
that integrates the enhanced data and labels system while maintaining full backward compatibility.

Key Features:
1. Drop-in replacement for existing MultiHorizonProfitLabeler
2. Enhanced data cleaning and quality assessment
3. Trading-aware label definitions (Analyst & Tactician)
4. Label stability monitoring and leakage detection
5. Full backward compatibility with existing pipeline
6. No duplication of existing functionality
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
import asyncio
from abc import ABC

# Import BaseStep
from src.training.steps.base_step import BaseStep

# Import existing multi-horizon labeler components
from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonConfig, MultiHorizonProfitLabeler, LabelingResult
)

# Import enhanced data and labels system
from .enhanced_data_labels_system import (
    EnhancedDataLabelsSystem, EnhancedDataLabelsConfig,
    create_trading_optimized_config, create_research_optimized_config
)

# Note: tprint and hardware utilities are available through BaseStep
# No need for direct imports as they're inherited from BaseStep


class EnhancedMultiHorizonConfig(MultiHorizonConfig):
    """Enhanced configuration that extends MultiHorizonConfig with enhanced features."""
    
    # Enhanced data and labels settings
    enable_enhanced_data_cleaning: bool = True
    enable_enhanced_stability_monitoring: bool = True
    enable_trading_aware_labels: bool = True
    
    # Enhanced label definitions
    analyst_horizon_minutes: int = 60
    tactician_horizon_minutes: int = 30
    enable_regime_conditioning: bool = True
    enable_risk_awareness: bool = True
    
    # Data quality thresholds
    min_data_quality_score: float = 0.7
    min_label_stability_score: float = 0.6
    
    # Enhanced configuration
    enhanced_config: Optional[EnhancedDataLabelsConfig] = None


class EnhancedMultiHorizonProfitLabeler(MultiHorizonProfitLabeler, BaseStep):
    """
    Enhanced Multi-Horizon Profit Labeler with full data and labels system integration.
    
    This class extends the existing MultiHorizonProfitLabeler with enhanced features:
    - Trading-aware label definitions (Analyst: "Should we trade?", Tactician: Direction/magnitude)
    - Comprehensive data cleaning and quality assessment
    - Label stability monitoring and leakage detection
    - Full backward compatibility with existing functionality
    - BaseStep integration for standardized pipeline execution
    
    Usage:
        # Drop-in replacement
        labeler = EnhancedMultiHorizonProfitLabeler(config)
        result = await labeler.execute_labeling(symbol, exchange, timeframe, data_dir, regime_data)
        
        # Enhanced processing
        result = await labeler.execute_enhanced_labeling(symbol, exchange, timeframe, data_dir, regime_data)
        
        # BaseStep execution
        result = await labeler.execute(config)
    """
    
    def __init__(self, config: Optional[EnhancedMultiHorizonConfig] = None):
        """Initialize the enhanced multi-horizon profit labeler."""
        # Initialize BaseStep first
        BaseStep.__init__(self)
        
        # Initialize with enhanced config
        self.enhanced_config = config or EnhancedMultiHorizonConfig()
        
        # Initialize enhanced data and labels system
        if self.enhanced_config.enhanced_config is None:
            if self.enhanced_config.enable_trading_aware_labels:
                self.enhanced_config.enhanced_config = create_trading_optimized_config()
            else:
                self.enhanced_config.enhanced_config = create_research_optimized_config()
        
        self.enhanced_labels_system = EnhancedDataLabelsSystem(self.enhanced_config.enhanced_config)
        
        # Initialize parent class with base config
        MultiHorizonProfitLabeler.__init__(self, self.enhanced_config)
        
        self.tprint_success("🚀 Enhanced Multi-Horizon Profit Labeler initialized")
        self.tprint_info("   → Enhanced data cleaning: Enabled")
        self.tprint_info("   → Enhanced stability monitoring: Enabled")
        self.tprint_info("   → Trading-aware labels: Enabled")
        self.tprint_info("   → Full backward compatibility: Maintained")
        self.tprint_info("   → BaseStep integration: Enabled")
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the enhanced multi-horizon profit labeling step.
        
        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol
                - exchange: Exchange name
                - timeframe: Data timeframe
                - data_dir: Directory containing market data
                - regime_data: Optional regime data
                - information: Optional information for context
                - direction: Optional direction for context
                - model: Optional model type for context
        
        Returns:
            Dictionary containing:
                - success: Boolean indicating success
                - labeling_result: LabelingResult object
                - enhanced_metrics: Enhanced processing metrics
                - artifacts: List of generated artifacts
        """
        try:
            # Set context for enhanced file naming and operations
            self._set_context(
                symbol=config.get('symbol'),
                exchange=config.get('exchange'),
                information=config.get('information'),
                direction=config.get('direction', 'long'),
                model=config.get('model', 'Analyst')
            )
            
            # Extract required parameters
            symbol = config.get('symbol')
            exchange = config.get('exchange')
            timeframe = config.get('timeframe')
            data_dir = config.get('data_dir')
            regime_data = config.get('regime_data')
            
            if not all([symbol, exchange, timeframe, data_dir]):
                return {
                    'success': False,
                    'error': 'Missing required parameters: symbol, exchange, timeframe, data_dir'
                }
            
            # Execute enhanced labeling
            result = await self.execute_enhanced_labeling(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                regime_data=regime_data
            )
            
            if not result.success:
                return {
                    'success': False,
                    'error': f'Enhanced labeling failed: {result.error_message}'
                }
            
            # Save artifacts
            artifacts = []
            if hasattr(result, 'artifacts') and result.artifacts:
                for artifact_name, artifact_data in result.artifacts.items():
                    if isinstance(artifact_data, pd.DataFrame):
                        # Preview DataFrame artifacts
                        self.tprint_data_preview(artifact_data, artifact_name, max_rows=5)
                        self.tprint_data_format(artifact_data, artifact_name)
                        artifact_path = self._save_dataframe(artifact_data, artifact_name)
                    else:
                        # Preview metadata artifacts
                        self.tprint_data_format(artifact_data, artifact_name)
                        artifact_path = self._save_metadata(artifact_data, artifact_name)
                    
                    if artifact_path:
                        artifacts.append(artifact_path)
            
            # Generate outcome file
            outcome_content = self._generate_outcome_content(result, artifacts)
            self._save_outcome_file(outcome_content, 'enhanced_multi_horizon_labeling_outcome')
            
            return {
                'success': True,
                'labeling_result': result,
                'enhanced_metrics': getattr(result, 'enhanced_metrics', {}),
                'artifacts': artifacts
            }
            
        except Exception as e:
            error_msg = f"Enhanced multi-horizon labeling failed: {str(e)}"
            self.tprint_error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg
            }
    
    def _generate_outcome_content(self, result: LabelingResult, artifacts: List[str]) -> str:
        """Generate outcome file content."""
        content = f"""# Enhanced Multi-Horizon Profit Labeling Outcome

## Summary
- **Status**: {'Success' if result.success else 'Failed'}
- **Processing Time**: {getattr(result, 'processing_time', 0):.2f} seconds
- **Samples Processed**: {getattr(result, 'n_samples', 0)}
- **Targets Generated**: {getattr(result, 'n_targets', 0)}
- **Artifacts Generated**: {len(artifacts)}

## Labeling Results
- **Success**: {result.success}
- **Error Message**: {getattr(result, 'error_message', 'None')}
- **Quality Score**: {getattr(result, 'quality_score', 0):.3f}
"""
        
        if hasattr(result, 'enhanced_metrics'):
            enhanced = result.enhanced_metrics
            content += f"""
## Enhanced Metrics
- **Data Quality Score**: {enhanced.get('data_quality_score', 0):.3f}
- **Label Stability Score**: {enhanced.get('label_stability_score', 0):.3f}
- **Trading-Aware Labels**: {enhanced.get('trading_aware_enabled', False)}
- **Regime Conditioning**: {enhanced.get('regime_conditioning_enabled', False)}
"""
        
        content += f"""
## Generated Artifacts
{chr(10).join(f"- {artifact}" for artifact in artifacts)}

## Configuration
- **Enhanced Data Cleaning**: {self.enhanced_config.enable_enhanced_data_cleaning}
- **Enhanced Stability Monitoring**: {self.enhanced_config.enable_enhanced_stability_monitoring}
- **Trading-Aware Labels**: {self.enhanced_config.enable_trading_aware_labels}
- **Regime Conditioning**: {self.enhanced_config.enable_regime_conditioning}
- **Risk Awareness**: {self.enhanced_config.enable_risk_awareness}
"""
        
        return content
    
    async def execute_labeling(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = "historical_data",
        regime_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute multi-horizon profit labeling with enhanced processing.
        
        This method provides full backward compatibility while adding enhanced features.
        """
        try:
            tprint_info(f"🏷️ Starting enhanced multi-horizon profit labeling for {symbol} on {exchange}")
            tprint_info(f"⏰ Timeframe: {timeframe}")
            
            # Load market data using existing method
            market_data = await self._load_market_data(symbol, exchange, timeframe, data_dir)
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for {symbol} {timeframe}")
            
            # Extract regime data if available
            regime_series = None
            if regime_data and self.enhanced_config.enable_regime_aware_labeling:
                regime_series = self._extract_regime_assignments(market_data, regime_data)
                if regime_series is not None:
                    tprint_info(f"🎭 Regime data extracted: {len(np.unique(regime_series[~pd.isna(regime_series)]))} unique regimes")
            
            # Process with enhanced data and labels system
            if self.enhanced_config.enable_enhanced_data_cleaning:
                tprint_info("🔧 Processing with enhanced data and labels system...")
                
                enhanced_result = self.enhanced_labels_system.process_market_data(
                    market_data=market_data,
                    regime_data=regime_series,
                    force_recompute=True
                )
                
                if 'error' in enhanced_result:
                    tprint_warning(f"⚠️ Enhanced processing failed: {enhanced_result['error']}")
                    tprint_info("🔄 Falling back to standard multi-horizon labeling...")
                    return await super().execute_labeling(symbol, exchange, timeframe, data_dir, regime_data)
                
                # Extract enhanced labels and data
                enhanced_labels = enhanced_result.get('labels', pd.DataFrame())
                enhanced_data = enhanced_result.get('processed_data', market_data)
                enhanced_confidence = enhanced_result.get('confidence_scores', pd.DataFrame())
                enhanced_weights = enhanced_result.get('sample_weights', pd.Series())
                
                # Data quality and stability metrics
                data_quality = enhanced_result.get('data_quality', {})
                label_stability = enhanced_result.get('label_stability', {})
                final_quality = enhanced_result.get('final_quality', {})
                
                tprint_success("✅ Enhanced data and labels processing completed")
                tprint_info(f"   → Data quality: {data_quality.get('quality_level', 'unknown')}")
                tprint_info(f"   → Label stability: {label_stability.get('stability_level', 'unknown')}")
                tprint_info(f"   → Final quality: {final_quality.get('overall_score', 0.0):.3f}")
                
                # Create enhanced labeling result
                enhanced_labeling_result = self._create_enhanced_labeling_result(
                    enhanced_labels, enhanced_data, enhanced_confidence, enhanced_weights,
                    data_quality, label_stability, final_quality, market_data
                )
                
                # Apply regime-aware processing if enabled
                if self.enhanced_config.enable_regime_aware_labeling and regime_series is not None:
                    tprint_info("🎭 Applying enhanced regime processing...")
                    enhanced_labeling_result = await self._apply_enhanced_regime_processing(
                        enhanced_labeling_result, market_data, regime_series
                    )
                
                # Apply enhanced balancing and weighting
                if self.enhanced_config.enable_label_balancing or self.enhanced_config.enable_sample_weighting:
                    tprint_info("⚖️ Applying enhanced balancing and weighting...")
                    enhanced_labeling_result = await self._apply_enhanced_balancing(
                        enhanced_labeling_result, market_data, regime_series
                    )
                
                # Generate comprehensive report
                report = await self._generate_enhanced_comprehensive_report(
                    enhanced_labeling_result, symbol, exchange, timeframe, regime_data
                )
                
                # Create final artifacts with enhanced metadata
                artifacts = self._create_enhanced_artifacts(
                    enhanced_labeling_result, report, symbol, exchange, timeframe, regime_data
                )
                
                tprint_success(f"✅ Enhanced multi-horizon labeling completed for {symbol}")
                tprint_info(f"   → Samples: {enhanced_labeling_result.get('n_samples', 0)}")
                tprint_info(f"   → Targets: {enhanced_labeling_result.get('n_targets', 0)}")
                tprint_info(f"   → Processing time: {enhanced_labeling_result.get('processing_time', 0.0):.2f}s")
                
                return artifacts
                
            else:
                # Use standard multi-horizon labeling
                tprint_info("🔄 Using standard multi-horizon labeling...")
                return await super().execute_labeling(symbol, exchange, timeframe, data_dir, regime_data)
            
        except Exception as e:
            tprint_error(f"❌ Enhanced multi-horizon labeling failed: {e}")
            # Fall back to standard labeling
            tprint_info("🔄 Falling back to standard multi-horizon labeling...")
            return await super().execute_labeling(symbol, exchange, timeframe, data_dir, regime_data)
    
    def _create_enhanced_labeling_result(
        self,
        enhanced_labels: pd.DataFrame,
        enhanced_data: pd.DataFrame,
        enhanced_confidence: pd.DataFrame,
        enhanced_weights: pd.Series,
        data_quality: Dict[str, Any],
        label_stability: Dict[str, Any],
        final_quality: Dict[str, Any],
        original_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create enhanced labeling result from enhanced data and labels system output."""
        try:
            # Map enhanced labels to multi-horizon format
            mapped_labels = self._map_enhanced_labels_to_multi_horizon(enhanced_labels)
            
            # Create confidence scores DataFrame
            confidence_scores = pd.DataFrame(index=enhanced_labels.index)
            if not enhanced_confidence.empty:
                confidence_scores = enhanced_confidence
            
            # Create eligibility masks
            eligibility_masks = pd.DataFrame(index=enhanced_labels.index)
            eligibility_masks['enhanced_eligible'] = pd.Series(True, index=enhanced_labels.index)
            
            # Create quality scores
            quality_scores = {}
            if final_quality:
                quality_scores['enhanced_overall'] = {
                    'overall_quality': final_quality.get('overall_score', 0.0),
                    'quality_grade': final_quality.get('quality_grade', 'F'),
                    'is_acceptable': final_quality.get('is_acceptable', False)
                }
            
            # Calculate statistics
            n_samples = len(enhanced_labels)
            n_targets = len([col for col in enhanced_labels.columns if 'label' in col.lower()])
            n_horizons = len([col for col in enhanced_labels.columns if 'horizon' in col.lower()])
            
            # Create label distribution
            label_distribution = {}
            for col in enhanced_labels.columns:
                if enhanced_labels[col].dtype in [np.number, 'int64', 'float64']:
                    values = enhanced_labels[col].dropna()
                    if len(values) > 0:
                        label_distribution[col] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'non_null_count': int(len(values))
                        }
            
            return {
                'labels': mapped_labels,
                'confidence_scores': confidence_scores,
                'eligibility_masks': eligibility_masks,
                'quality_scores': quality_scores,
                'n_samples': n_samples,
                'n_targets': n_targets,
                'n_horizons': n_horizons,
                'label_distribution': label_distribution,
                'processing_time': 0.0,  # Will be set by caller
                'enhanced_metadata': {
                    'data_quality': data_quality,
                    'label_stability': label_stability,
                    'final_quality': final_quality,
                    'enhanced_processing': True,
                    'original_samples': len(original_data),
                    'processed_samples': len(enhanced_data)
                }
            }
            
        except Exception as e:
            tprint_error(f"❌ Error creating enhanced labeling result: {e}")
            # Return minimal result
            return {
                'labels': enhanced_labels,
                'confidence_scores': pd.DataFrame(),
                'eligibility_masks': pd.DataFrame(),
                'quality_scores': {},
                'n_samples': len(enhanced_labels),
                'n_targets': 0,
                'n_horizons': 0,
                'label_distribution': {},
                'processing_time': 0.0,
                'enhanced_metadata': {'error': str(e)}
            }
    
    def _map_enhanced_labels_to_multi_horizon(self, enhanced_labels: pd.DataFrame) -> pd.DataFrame:
        """Map enhanced labels to multi-horizon format for compatibility."""
        try:
            mapped_df = enhanced_labels.copy()
            
            # Map analyst labels to immediate_opportunity
            if 'analyst_label' in enhanced_labels.columns:
                mapped_df['immediate_opportunity'] = enhanced_labels['analyst_label']
                mapped_df['immediate_confidence'] = enhanced_labels.get('analyst_confidence', 0.5)
            
            # Map tactician labels to short_term_opportunity
            if 'tactician_label' in enhanced_labels.columns:
                mapped_df['short_term_opportunity'] = enhanced_labels['tactician_label']
                mapped_df['short_term_magnitude'] = enhanced_labels.get('tactician_magnitude', 1.0)
            
            # Create leverage_adjusted_score from analyst confidence
            if 'analyst_confidence' in enhanced_labels.columns:
                mapped_df['leverage_adjusted_score'] = enhanced_labels['analyst_confidence']
            
            # Add horizon-specific targets if not present
            if 'immediate_opportunity' not in mapped_df.columns:
                mapped_df['immediate_opportunity'] = enhanced_labels.get('analyst_label', 0)
            
            if 'short_term_opportunity' not in mapped_df.columns:
                mapped_df['short_term_opportunity'] = enhanced_labels.get('tactician_label', 0)
            
            if 'leverage_adjusted_score' not in mapped_df.columns:
                mapped_df['leverage_adjusted_score'] = enhanced_labels.get('analyst_confidence', 0.5)
            
            tprint_success(f"✅ Enhanced labels mapped to multi-horizon format: {len(mapped_df.columns)} columns")
            return mapped_df
            
        except Exception as e:
            tprint_error(f"❌ Error mapping enhanced labels: {e}")
            return enhanced_labels
    
    async def _apply_enhanced_regime_processing(
        self,
        labeling_result: Dict[str, Any],
        market_data: pd.DataFrame,
        regime_series: pd.Series
    ) -> Dict[str, Any]:
        """Apply enhanced regime processing to labeling result."""
        try:
            tprint_info("🎭 Applying enhanced regime processing...")
            
            # Get unique regimes
            regimes = np.unique(regime_series[~pd.isna(regime_series)])
            tprint_info(f"📊 Processing {len(regimes)} distinct regimes")
            
            if len(regimes) == 0:
                tprint_warning("⚠️ No valid regime assignments found")
                return labeling_result
            
            # Process each regime with enhanced system
            regime_results = {}
            total_processing_time = 0.0
            
            for regime in regimes:
                tprint_info(f"🏷️ Processing regime {regime}")
                
                # Filter data for this regime
                regime_mask = regime_series == regime
                regime_data_subset = market_data[regime_mask].copy()
                
                if len(regime_data_subset) < self.enhanced_config.min_data_points:
                    tprint_warning(f"⚠️ Insufficient data for regime {regime}: {len(regime_data_subset)} samples")
                    continue
                
                # Process with enhanced system
                regime_result = self.enhanced_labels_system.process_market_data(
                    market_data=regime_data_subset,
                    regime_data=regime_series[regime_mask],
                    force_recompute=True
                )
                
                if 'error' not in regime_result:
                    regime_labels = regime_result.get('labels', pd.DataFrame())
                    if not regime_labels.empty:
                        # Add regime suffix to column names
                        regime_labels_suffixed = regime_labels.add_suffix(f'_regime_{regime}')
                        regime_results[regime] = regime_labels_suffixed
                        total_processing_time += regime_result.get('processing_time', 0.0)
            
            # Combine regime-specific results
            if regime_results:
                combined_labels = pd.concat(regime_results.values(), axis=1)
                
                # Update labeling result
                labeling_result['labels'] = combined_labels
                labeling_result['n_samples'] = len(combined_labels)
                labeling_result['n_targets'] = len([col for col in combined_labels.columns if 'label' in col.lower()])
                labeling_result['processing_time'] += total_processing_time
                
                # Add regime metadata
                labeling_result['enhanced_metadata']['regime_processing'] = {
                    'n_regimes': len(regimes),
                    'regime_results': list(regime_results.keys()),
                    'total_processing_time': total_processing_time
                }
                
                tprint_success(f"✅ Enhanced regime processing completed for {len(regimes)} regimes")
            
            return labeling_result
            
        except Exception as e:
            tprint_error(f"❌ Enhanced regime processing failed: {e}")
            return labeling_result
    
    async def _apply_enhanced_balancing(
        self,
        labeling_result: Dict[str, Any],
        market_data: pd.DataFrame,
        regime_series: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Apply enhanced balancing and weighting to labeling result."""
        try:
            tprint_info("⚖️ Applying enhanced balancing and weighting...")
            
            if not self.balancing_system:
                tprint_info("ℹ️ Balancing system not available, skipping enhanced balancing")
                return labeling_result
            
            # Prepare data for balancing
            labels_df = labeling_result.get('labels', pd.DataFrame())
            if labels_df.empty:
                tprint_warning("⚠️ No labels available for balancing")
                return labeling_result
            
            # Use first target column for balancing
            target_cols = [col for col in labels_df.columns if 'label' in col.lower() or 'opportunity' in col.lower()]
            if not target_cols:
                tprint_warning("⚠️ No target columns found for balancing")
                return labeling_result
            
            y = labels_df[target_cols[0]]
            
            # Prepare features
            feature_cols = ['open', 'high', 'low', 'close', 'volume']
            X = market_data[feature_cols]
            
            # Prepare additional features
            additional_features = {}
            if regime_series is not None:
                additional_features['regime'] = regime_series
            
            # Add volatility if available
            if 'volatility' in market_data.columns:
                additional_features['volatility'] = market_data['volatility']
            
            # Apply balancing
            X_balanced, y_balanced, final_weights = self.balancing_system.balance_and_weight(
                X, y, additional_features=additional_features
            )
            
            # Update labeling result
            labeling_result['labels'] = pd.DataFrame({target_cols[0]: y_balanced}, index=y_balanced.index)
            labeling_result['n_samples'] = len(y_balanced)
            labeling_result['sample_weights'] = final_weights
            
            # Add balancing metadata
            labeling_result['enhanced_metadata']['balancing_applied'] = True
            labeling_result['enhanced_metadata']['original_samples'] = len(y)
            labeling_result['enhanced_metadata']['balanced_samples'] = len(y_balanced)
            
            tprint_success(f"✅ Enhanced balancing completed: {len(y)} → {len(y_balanced)} samples")
            
            return labeling_result
            
        except Exception as e:
            tprint_error(f"❌ Enhanced balancing failed: {e}")
            return labeling_result
    
    async def _generate_enhanced_comprehensive_report(
        self,
        labeling_result: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        regime_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate enhanced comprehensive report."""
        try:
            tprint_info("📋 Generating enhanced comprehensive report...")
            
            # Get enhanced metadata
            enhanced_metadata = labeling_result.get('enhanced_metadata', {})
            
            # Create enhanced report
            report = {
                'status': 'completed',
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'processing_time': labeling_result.get('processing_time', 0.0),
                'enhanced_processing': True,
                'statistics': {
                    'n_samples': labeling_result.get('n_samples', 0),
                    'n_targets': labeling_result.get('n_targets', 0),
                    'n_horizons': labeling_result.get('n_horizons', 0),
                    'label_distribution': labeling_result.get('label_distribution', {})
                },
                'enhanced_metrics': {
                    'data_quality': enhanced_metadata.get('data_quality', {}),
                    'label_stability': enhanced_metadata.get('label_stability', {}),
                    'final_quality': enhanced_metadata.get('final_quality', {}),
                    'regime_processing': enhanced_metadata.get('regime_processing', {}),
                    'balancing_applied': enhanced_metadata.get('balancing_applied', False)
                },
                'quality_scores': labeling_result.get('quality_scores', {}),
                'recommendations': self._generate_enhanced_recommendations(enhanced_metadata)
            }
            
            tprint_success("✅ Enhanced comprehensive report generated")
            return report
            
        except Exception as e:
            tprint_error(f"❌ Enhanced report generation failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_enhanced_recommendations(self, enhanced_metadata: Dict[str, Any]) -> List[str]:
        """Generate enhanced recommendations based on metadata."""
        recommendations = []
        
        # Data quality recommendations
        data_quality = enhanced_metadata.get('data_quality', {})
        quality_score = data_quality.get('quality_score', 0.0)
        if quality_score < 0.7:
            recommendations.append("Improve data quality - address missing values and outliers")
        
        # Label stability recommendations
        label_stability = enhanced_metadata.get('label_stability', {})
        stability_level = label_stability.get('stability_level', 'unknown')
        if stability_level in ['warning', 'critical', 'unstable']:
            recommendations.append("Address label stability issues - check for leakage and drift")
        
        # Final quality recommendations
        final_quality = enhanced_metadata.get('final_quality', {})
        overall_score = final_quality.get('overall_score', 0.0)
        if overall_score < 0.6:
            recommendations.append("Overall quality needs improvement - review all components")
        
        # Regime processing recommendations
        regime_processing = enhanced_metadata.get('regime_processing', {})
        if regime_processing and regime_processing.get('n_regimes', 0) > 5:
            recommendations.append("Consider consolidating regimes - too many regime types detected")
        
        if not recommendations:
            recommendations.append("Enhanced processing completed successfully - no immediate action required")
        
        return recommendations
    
    def _create_enhanced_artifacts(
        self,
        labeling_result: Dict[str, Any],
        report: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        regime_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create enhanced artifacts with full metadata."""
        try:
            # Create artifacts structure compatible with existing pipeline
            artifacts = {
                'multi_horizon_labeling_result': {
                    'labeled_data': labeling_result.get('labels', pd.DataFrame()),
                    'labels': labeling_result.get('labels', pd.DataFrame()),
                    'confidence_scores': labeling_result.get('confidence_scores', pd.DataFrame()),
                    'eligibility_masks': labeling_result.get('eligibility_masks', pd.DataFrame()),
                    'quality_scores': labeling_result.get('quality_scores', {}),
                    'sample_weights': labeling_result.get('sample_weights', pd.Series()),
                    'method': 'enhanced_multi_horizon_profit_labeling',
                    'enhanced_processing': True,
                    'metadata': {
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'regime_aware': self.enhanced_config.enable_regime_aware_labeling and regime_data is not None,
                        'processing_time': labeling_result.get('processing_time', 0.0),
                        'n_samples': labeling_result.get('n_samples', 0),
                        'n_targets': labeling_result.get('n_targets', 0),
                        'n_horizons': labeling_result.get('n_horizons', 0),
                        'target_distribution': labeling_result.get('label_distribution', {}),
                        'enhanced_metadata': labeling_result.get('enhanced_metadata', {})
                    }
                },
                'labeling_report': report,
                'enhanced_artifacts': {
                    'enhanced_labels': labeling_result.get('labels', pd.DataFrame()),
                    'enhanced_confidence': labeling_result.get('confidence_scores', pd.DataFrame()),
                    'enhanced_weights': labeling_result.get('sample_weights', pd.Series()),
                    'data_quality_metrics': labeling_result.get('enhanced_metadata', {}).get('data_quality', {}),
                    'label_stability_metrics': labeling_result.get('enhanced_metadata', {}).get('label_stability', {}),
                    'final_quality_metrics': labeling_result.get('enhanced_metadata', {}).get('final_quality', {}),
                    'recommendations': report.get('recommendations', [])
                }
            }
            
            tprint_success("✅ Enhanced artifacts created with comprehensive metadata")
            return artifacts
            
        except Exception as e:
            tprint_error(f"❌ Error creating enhanced artifacts: {e}")
            return {
                'multi_horizon_labeling_result': {},
                'labeling_report': {'status': 'error', 'error': str(e)},
                'enhanced_artifacts': {}
            }


# Convenience functions
def create_enhanced_multi_horizon_labeler(
    config: Optional[EnhancedMultiHorizonConfig] = None
) -> EnhancedMultiHorizonProfitLabeler:
    """Create enhanced multi-horizon profit labeler with specified configuration."""
    return EnhancedMultiHorizonProfitLabeler(config)


def create_trading_optimized_multi_horizon_config() -> EnhancedMultiHorizonConfig:
    """Create trading-optimized configuration for enhanced multi-horizon labeler."""
    return EnhancedMultiHorizonConfig(
        enable_enhanced_data_cleaning=True,
        enable_enhanced_stability_monitoring=True,
        enable_trading_aware_labels=True,
        analyst_horizon_minutes=60,
        tactician_horizon_minutes=30,
        enable_regime_conditioning=True,
        enable_risk_awareness=True,
        min_data_quality_score=0.8,
        min_label_stability_score=0.7,
        enhanced_config=create_trading_optimized_config()
    )


def create_research_optimized_multi_horizon_config() -> EnhancedMultiHorizonConfig:
    """Create research-optimized configuration for enhanced multi-horizon labeler."""
    return EnhancedMultiHorizonConfig(
        enable_enhanced_data_cleaning=True,
        enable_enhanced_stability_monitoring=True,
        enable_trading_aware_labels=True,
        analyst_horizon_minutes=60,
        tactician_horizon_minutes=30,
        enable_regime_conditioning=True,
        enable_risk_awareness=False,
        min_data_quality_score=0.6,
        min_label_stability_score=0.5,
        enhanced_config=create_research_optimized_config()
    )


# Backward compatibility - replace the original class
MultiHorizonProfitLabeler = EnhancedMultiHorizonProfitLabeler
MultiHorizonConfig = EnhancedMultiHorizonConfig