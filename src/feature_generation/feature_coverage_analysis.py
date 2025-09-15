"""
Feature Coverage Analysis

This module analyzes all existing features to ensure complete coverage
in the new unified feature generation system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureCoverageAnalyzer:
    """
    Analyzes feature coverage to ensure all existing features are covered
    in the new unified system.
    """
    
    def __init__(self):
        self.logger = logger.getChild('FeatureCoverageAnalyzer')
        
        # Existing features from old system
        self.existing_features = self._analyze_existing_features()
        
        # New features from unified system
        self.new_features = self._analyze_new_features()
        
        # Coverage analysis
        self.coverage_report = self._analyze_coverage()
    
    def _analyze_existing_features(self) -> Dict[str, List[str]]:
        """Analyze all existing features from the old system."""
        
        existing_features = {
            # From feature_generators.py
            'technical_indicators': [
                'sma', 'simple_moving_average',
                'ema', 'exponential_moving_average', 
                'volatility', 'rolling_volatility',
                'momentum', 'price_momentum',
                'rsi', 'relative_strength_index',
                'macd',
                'bollinger_bands', 'bbands',
                'stochastic',
                'volume_sma',
                'body_size',
                'taker_buy_ratio'
            ],
            
            # From step06_enhanced_feature_engineering_step.py
            'regime_features': [
                'regime_volatility',
                'regime_momentum', 
                'regime_volume',
                'regime_correlation'
            ],
            
            'sr_features': [
                'distance_to_sr',
                'sr_strength',
                'sr_breakout_probability',
                'sr_volume_profile'
            ],
            
            'time_features': [
                'hour_of_day',
                'day_of_week',
                'month_of_year',
                'quarter',
                'is_weekend',
                'is_month_end',
                'is_quarter_end'
            ],
            
            # From cross_timeframe_interaction_features.py
            'cross_timeframe_features': [
                'momentum_cross_timeframe',
                'volatility_cross_timeframe',
                'volume_cross_timeframe',
                'correlation_cross_timeframe',
                'microstructure_cross_timeframe'
            ],
            
            # From feature_engineering_orchestrator.py
            'advanced_features': [
                'autoencoder_features',
                'microstructure_features',
                'entropy_features',
                'legacy_features'
            ],
            
            # From limited_microstructure_features.py
            'microstructure_features': [
                'order_flow_imbalance',
                'volume_weighted_price',
                'trade_size_distribution',
                'bid_ask_spread_proxy',
                'market_impact_proxy'
            ],
            
            # From autoencoder_feature_generator.py
            'autoencoder_features': [
                'encoded_features',
                'reconstruction_error',
                'anomaly_score'
            ],
            
            # From entropy_feature_engine.py
            'entropy_features': [
                'shannon_entropy',
                'renyi_entropy',
                'tsallis_entropy',
                'permutation_entropy'
            ]
        }
        
        return existing_features
    
    def _analyze_new_features(self) -> Dict[str, List[str]]:
        """Analyze all new features from the unified system."""
        
        new_features = {
            # Base calculations
            'base_calculations': [
                'price_returns',
                'returns_vwap',
                'price_levels',
                'volume_weighted'
            ],
            
            # Enhanced technical indicators
            'enhanced_technical_indicators': [
                'rsi_price_returns',
                'rsi_returns_vwap',
                'rsi_price_levels',
                'macd_price_returns',
                'macd_returns_vwap',
                'macd_price_levels',
                'bollinger_bands_price_returns',
                'bollinger_bands_returns_vwap',
                'bollinger_bands_price_levels',
                'sma_price_returns',
                'sma_returns_vwap',
                'sma_price_levels',
                'ema_price_returns',
                'ema_returns_vwap',
                'ema_price_levels'
            ],
            
            # Interaction features
            'interaction_features': [
                'cross_timeframe_ratio',
                'cross_timeframe_difference',
                'cross_timeframe_product',
                'feature_ratio_sma',
                'feature_ratio_ema',
                'feature_ratio_volatility',
                'polynomial_returns',
                'polynomial_volatility',
                'correlation_returns_volume',
                'correlation_volatility_returns'
            ],
            
            # Category-based features
            'returns_features': [
                'simple_return_1period',
                'simple_return_5period',
                'simple_return_10period',
                'simple_return_20period',
                'log_return',
                'ewma_return_1period',
                'ewma_return_5period',
                'ewma_return_10period',
                'ewma_return_20period'
            ],
            
            'momentum_features': [
                'rsi_10', 'rsi_20', 'rsi_50',
                'macd', 'macd_signal', 'macd_hist',
                'momentum_10', 'momentum_20', 'momentum_50',
                'roc_10', 'roc_20', 'roc_50'
            ],
            
            'volume_features': [
                'obv',
                'chaikin_ad',
                'volume_sma_10', 'volume_sma_20', 'volume_sma_50',
                'volume_ema_10', 'volume_ema_20', 'volume_ema_50',
                'volume_roc_10', 'volume_roc_20', 'volume_roc_50'
            ],
            
            'volatility_features': [
                'atr_14', 'atr_20',
                'bb_upper_20_2', 'bb_middle_20_2', 'bb_lower_20_2', 'bb_width_20_2',
                'stddev_14', 'stddev_20'
            ],
            
            'trend_features': [
                'adx_14', 'adx_20', 'adx_50',
                'plus_di_14', 'plus_di_20', 'plus_di_50',
                'minus_di_14', 'minus_di_20', 'minus_di_50',
                'aroon_down_14', 'aroon_down_20', 'aroon_down_50',
                'aroon_up_14', 'aroon_up_20', 'aroon_up_50',
                'aroon_oscillator_14', 'aroon_oscillator_20', 'aroon_oscillator_50',
                'sar'
            ],
            
            'oscillator_features': [
                'stoch_slowk_14_3', 'stoch_slowd_14_3',
                'cci_14', 'cci_20',
                'williams_r_14', 'williams_r_20'
            ],
            
            'support_resistance_features': [
                'dist_to_sr',
                'is_near_sr',
                'sr_strength',
                'sr_breakout_probability'
            ],
            
            'candlestick_pattern_features': [
                'cdldoji',
                'cdlhammer',
                'cdlinvertedhammer',
                'cdlengulfing',
                'cdlharami',
                'cdlpiercing',
                'cdlmorningstar',
                'cdlevenstar',
                'cdl3whitesoldiers',
                'cdl3blackcrows',
                'cdldragonflydoji',
                'cdlgravestonedoji',
                'cdlmarubozu',
                'cdlspinningtop'
            ],
            
            'hmm_regime_features': [
                'hmm_current_regime',
                'hmm_regime_proba_0',
                'hmm_regime_proba_1',
                'hmm_regime_proba_2'
            ]
        }
        
        return new_features
    
    def _analyze_coverage(self) -> Dict[str, Any]:
        """Analyze coverage between existing and new features."""
        
        coverage_report = {
            'covered_features': {},
            'missing_features': {},
            'enhanced_features': {},
            'new_features': {},
            'coverage_percentage': {}
        }
        
        # Analyze coverage for each category
        for category, existing_feature_list in self.existing_features.items():
            covered = []
            missing = []
            enhanced = []
            
            for feature in existing_feature_list:
                # Check if feature is covered in new system
                if self._is_feature_covered(feature, category):
                    covered.append(feature)
                    
                    # Check if it's enhanced
                    if self._is_feature_enhanced(feature, category):
                        enhanced.append(feature)
                else:
                    missing.append(feature)
            
            coverage_report['covered_features'][category] = covered
            coverage_report['missing_features'][category] = missing
            coverage_report['enhanced_features'][category] = enhanced
            
            # Calculate coverage percentage
            total_features = len(existing_feature_list)
            covered_count = len(covered)
            coverage_percentage = (covered_count / total_features * 100) if total_features > 0 else 0
            coverage_report['coverage_percentage'][category] = coverage_percentage
        
        # Identify completely new features
        all_existing = set()
        for feature_list in self.existing_features.values():
            all_existing.update(feature_list)
        
        all_new = set()
        for feature_list in self.new_features.values():
            all_new.update(feature_list)
        
        coverage_report['new_features'] = list(all_new - all_existing)
        
        return coverage_report
    
    def _is_feature_covered(self, feature: str, category: str) -> bool:
        """Check if a feature is covered in the new system."""
        
        # Direct mapping
        feature_mappings = {
            'sma': ['sma_price_returns', 'sma_returns_vwap', 'sma_price_levels'],
            'ema': ['ema_price_returns', 'ema_returns_vwap', 'ema_price_levels'],
            'rsi': ['rsi_price_returns', 'rsi_returns_vwap', 'rsi_price_levels', 'rsi_10', 'rsi_20', 'rsi_50'],
            'macd': ['macd_price_returns', 'macd_returns_vwap', 'macd_price_levels', 'macd', 'macd_signal', 'macd_hist'],
            'bollinger_bands': ['bollinger_bands_price_returns', 'bollinger_bands_returns_vwap', 'bollinger_bands_price_levels', 'bb_upper_20_2', 'bb_middle_20_2', 'bb_lower_20_2', 'bb_width_20_2'],
            'volatility': ['atr_14', 'atr_20', 'stddev_14', 'stddev_20'],
            'stochastic': ['stoch_slowk_14_3', 'stoch_slowd_14_3'],
            'momentum': ['momentum_10', 'momentum_20', 'momentum_50', 'roc_10', 'roc_20', 'roc_50'],
            'volume_sma': ['volume_sma_10', 'volume_sma_20', 'volume_sma_50'],
            'cross_timeframe_features': ['cross_timeframe_ratio', 'cross_timeframe_difference', 'cross_timeframe_product'],
            'regime_features': ['hmm_current_regime', 'hmm_regime_proba_0', 'hmm_regime_proba_1', 'hmm_regime_proba_2'],
            'sr_features': ['dist_to_sr', 'is_near_sr', 'sr_strength', 'sr_breakout_probability'],
            'time_features': ['hour_of_day', 'day_of_week', 'month_of_year', 'quarter', 'is_weekend', 'is_month_end', 'is_quarter_end'],
            'microstructure_features': ['order_flow_imbalance', 'volume_weighted_price', 'trade_size_distribution', 'bid_ask_spread_proxy', 'market_impact_proxy'],
            'autoencoder_features': ['encoded_features', 'reconstruction_error', 'anomaly_score'],
            'entropy_features': ['shannon_entropy', 'renyi_entropy', 'tsallis_entropy', 'permutation_entropy']
        }
        
        if feature in feature_mappings:
            return True
        
        # Check if feature exists in new system
        for new_category, new_features in self.new_features.items():
            if feature in new_features:
                return True
        
        return False
    
    def _is_feature_enhanced(self, feature: str, category: str) -> bool:
        """Check if a feature is enhanced in the new system."""
        
        enhanced_features = [
            'sma', 'ema', 'rsi', 'macd', 'bollinger_bands'
        ]
        
        return feature in enhanced_features
    
    def generate_coverage_report(self) -> str:
        """Generate a comprehensive coverage report."""
        
        report = []
        report.append("=" * 80)
        report.append("FEATURE COVERAGE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall coverage
        total_existing = sum(len(features) for features in self.existing_features.values())
        total_covered = sum(len(features) for features in self.coverage_report['covered_features'].values())
        overall_coverage = (total_covered / total_existing * 100) if total_existing > 0 else 0
        
        report.append(f"OVERALL COVERAGE: {overall_coverage:.1f}% ({total_covered}/{total_existing})")
        report.append("")
        
        # Category-wise coverage
        report.append("CATEGORY-WISE COVERAGE:")
        report.append("-" * 40)
        
        for category, percentage in self.coverage_report['coverage_percentage'].items():
            covered_count = len(self.coverage_report['covered_features'][category])
            total_count = len(self.existing_features[category])
            report.append(f"{category:25} {percentage:6.1f}% ({covered_count}/{total_count})")
        
        report.append("")
        
        # Enhanced features
        report.append("ENHANCED FEATURES:")
        report.append("-" * 40)
        
        for category, enhanced_features in self.coverage_report['enhanced_features'].items():
            if enhanced_features:
                report.append(f"{category}:")
                for feature in enhanced_features:
                    report.append(f"  ✓ {feature} (enhanced with base calculations)")
        
        report.append("")
        
        # Missing features
        report.append("MISSING FEATURES:")
        report.append("-" * 40)
        
        missing_found = False
        for category, missing_features in self.coverage_report['missing_features'].items():
            if missing_features:
                missing_found = True
                report.append(f"{category}:")
                for feature in missing_features:
                    report.append(f"  ✗ {feature}")
        
        if not missing_found:
            report.append("No missing features found!")
        
        report.append("")
        
        # New features
        report.append("NEW FEATURES:")
        report.append("-" * 40)
        
        if self.coverage_report['new_features']:
            for feature in self.coverage_report['new_features']:
                report.append(f"  + {feature}")
        else:
            report.append("No completely new features added.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def get_missing_features(self) -> List[str]:
        """Get list of missing features that need to be implemented."""
        
        missing_features = []
        for category, features in self.coverage_report['missing_features'].items():
            missing_features.extend(features)
        
        return missing_features
    
    def get_enhancement_recommendations(self) -> List[str]:
        """Get recommendations for feature enhancements."""
        
        recommendations = []
        
        # Check for features that could be enhanced with base calculations
        enhanceable_features = ['atr', 'stochastic', 'williams_r', 'cci', 'adx', 'aroon']
        
        for feature in enhanceable_features:
            if feature in str(self.existing_features.values()):
                recommendations.append(f"Enhance {feature} with base calculation support (price_returns, returns_vwap)")
        
        # Check for missing interaction features
        if 'cross_timeframe_features' in self.coverage_report['missing_features']:
            recommendations.append("Implement comprehensive cross-timeframe interaction features")
        
        # Check for missing microstructure features
        if 'microstructure_features' in self.coverage_report['missing_features']:
            recommendations.append("Implement microstructure features (order flow, market impact)")
        
        # Check for missing autoencoder features
        if 'autoencoder_features' in self.coverage_report['missing_features']:
            recommendations.append("Implement autoencoder-based feature generation")
        
        return recommendations

def run_coverage_analysis():
    """Run the complete coverage analysis."""
    
    print("🔍 Running Feature Coverage Analysis...")
    print("=" * 60)
    
    analyzer = FeatureCoverageAnalyzer()
    
    # Generate and print report
    report = analyzer.generate_coverage_report()
    print(report)
    
    # Get missing features
    missing_features = analyzer.get_missing_features()
    if missing_features:
        print("\n🚨 MISSING FEATURES TO IMPLEMENT:")
        print("-" * 40)
        for feature in missing_features:
            print(f"  - {feature}")
    
    # Get enhancement recommendations
    recommendations = analyzer.get_enhancement_recommendations()
    if recommendations:
        print("\n💡 ENHANCEMENT RECOMMENDATIONS:")
        print("-" * 40)
        for rec in recommendations:
            print(f"  - {rec}")
    
    return analyzer

if __name__ == "__main__":
    analyzer = run_coverage_analysis()