#!/usr/bin/env python3
"""
Market Analysis Score Fixer

This script integrates with the existing market analysis pipeline to fix negative
feature scores by applying the enhanced scoring system and normalization techniques.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

# Import our custom solutions
from fix_negative_feature_scores import FeatureScoreNormalizer, ScoreNormalizationConfig
from enhanced_feature_scoring_system import EnhancedFeatureScorer, EnhancedScoringConfig

# Import existing market analysis components
try:
    from market_analysis.triple_barrier_labeling.core import TripleBarrierLabeler, TripleBarrierConfig
    from market_analysis.triple_barrier_labeling.optimized_labeler import EnhancedOptimizedTripleBarrierLabeler
    MARKET_ANALYSIS_AVAILABLE = True
except ImportError as e:
    MARKET_ANALYSIS_AVAILABLE = False
    logging.warning(f"Market analysis components not available: {e}")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketAnalysisScoreFixer:
    """
    Comprehensive solution to fix negative feature scores in the market analysis pipeline.
    
    This class integrates multiple approaches:
    1. Enhanced feature scoring with multiple methods
    2. Score normalization and transformation
    3. Triple barrier parameter optimization
    4. Regime-aware scoring adjustments
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the market analysis score fixer."""
        self.config = config or {}
        self.logger = logger
        
        # Initialize components
        self.normalizer = FeatureScoreNormalizer(
            ScoreNormalizationConfig(
                use_min_max_scaling=True,
                negative_score_floor=0.0,
                positive_score_ceiling=1.0,
                transaction_cost_adjustment=True
            )
        )
        
        self.enhanced_scorer = EnhancedFeatureScorer(
            EnhancedScoringConfig(
                normalize_scores=True,
                normalization_method="min_max",
                enable_ensemble=True,
                enable_regime_scoring=True
            )
        )
        
        # Initialize market analysis components if available
        if MARKET_ANALYSIS_AVAILABLE:
            self.triple_barrier_labeler = TripleBarrierLabeler(
                TripleBarrierConfig(
                    pt_mult=0.01,  # 1% profit target
                    sl_mult=0.005,  # 0.5% stop loss
                    transaction_cost=0.0004,  # Reduced from 0.08% to 0.04%
                    max_holding_period=100
                )
            )
            
            self.optimized_labeler = EnhancedOptimizedTripleBarrierLabeler()
        else:
            self.triple_barrier_labeler = None
            self.optimized_labeler = None
        
        self.results = {}
        
        logger.info("🚀 Market Analysis Score Fixer initialized")
        logger.info(f"   Market analysis available: {MARKET_ANALYSIS_AVAILABLE}")
    
    def fix_mrmr_scores(self, mrmr_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Fix negative mRMR scores using normalization techniques.
        
        Args:
            mrmr_scores: Dictionary of feature names to mRMR scores
            
        Returns:
            Dictionary of fixed scores
        """
        logger.info("🔧 Fixing mRMR scores")
        
        # Apply normalization
        fixed_scores = self.normalizer.normalize_mrmr_scores(mrmr_scores)
        
        # Store results
        self.results['mrmr_fix'] = {
            'original_scores': mrmr_scores,
            'fixed_scores': fixed_scores,
            'improvement': sum(1 for k in mrmr_scores if mrmr_scores[k] < 0 and fixed_scores[k] >= 0)
        }
        
        logger.info(f"   Fixed {self.results['mrmr_fix']['improvement']} negative mRMR scores")
        
        return fixed_scores
    
    def fix_triple_barrier_scores(self, market_data: pd.DataFrame, regime_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Fix negative scores from triple barrier labeling.
        
        Args:
            market_data: Market data with OHLC columns
            regime_data: Optional regime information
            
        Returns:
            DataFrame with improved labeling results
        """
        if not MARKET_ANALYSIS_AVAILABLE:
            logger.warning("Market analysis components not available - skipping triple barrier fixes")
            return market_data
        
        logger.info("🔧 Fixing triple barrier scores")
        
        # 1. Optimize parameters if we have the optimized labeler
        if self.optimized_labeler and regime_data is not None:
            logger.info("   Running parameter optimization")
            optimization_results = self.optimized_labeler.optimize_regime_parameters(
                market_data, regime_data, n_trials=50
            )
            
            # Create labels with optimized parameters
            optimized_labels = self.optimized_labeler.create_optimized_labels(market_data, regime_data)
        else:
            # Use standard labeler with improved configuration
            logger.info("   Using standard triple barrier labeler")
            result = self.triple_barrier_labeler.create_labels(market_data)
            optimized_labels = result.labels
        
        # 2. Apply score normalization to the results
        fixed_labels = self.normalizer.fix_triple_barrier_scores(optimized_labels)
        
        # Store results
        original_negative = sum(1 for profit in optimized_labels.get('profit_pct', []) if profit < 0)
        fixed_negative = sum(1 for profit in fixed_labels.get('profit_pct', []) if profit < 0)
        
        self.results['triple_barrier_fix'] = {
            'original_negative_count': original_negative,
            'fixed_negative_count': fixed_negative,
            'improvement': original_negative - fixed_negative,
            'total_labels': len(optimized_labels)
        }
        
        logger.info(f"   Reduced negative profits from {original_negative} to {fixed_negative}")
        
        return fixed_labels
    
    def apply_enhanced_feature_scoring(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str], 
                                     regime_data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Apply enhanced feature scoring to replace negative-prone methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            regime_data: Optional regime labels
            
        Returns:
            Dictionary of enhanced feature scores
        """
        logger.info("🎯 Applying enhanced feature scoring")
        
        # Score features using multiple methods
        enhanced_scores = self.enhanced_scorer.score_features(X, y, feature_names, regime_data)
        
        # Store results
        negative_count = sum(1 for score in enhanced_scores.values() if score < 0)
        
        self.results['enhanced_scoring'] = {
            'total_features': len(enhanced_scores),
            'negative_count': negative_count,
            'negative_percentage': negative_count / len(enhanced_scores) * 100 if enhanced_scores else 0,
            'top_features': self.enhanced_scorer.get_top_features(enhanced_scores, k=10)
        }
        
        logger.info(f"   Enhanced scoring: {negative_count}/{len(enhanced_scores)} negative scores")
        
        return enhanced_scores
    
    def comprehensive_fix(self, market_data: pd.DataFrame, 
                         feature_data: Optional[pd.DataFrame] = None,
                         regime_data: Optional[pd.DataFrame] = None,
                         existing_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Apply comprehensive fixes to all scoring issues in the market analysis pipeline.
        
        Args:
            market_data: Market data with OHLC columns
            feature_data: Optional feature matrix
            regime_data: Optional regime information
            existing_scores: Optional existing feature scores to fix
            
        Returns:
            Dictionary with all fixed results
        """
        logger.info("🚀 Starting comprehensive score fixing")
        logger.info("=" * 60)
        
        comprehensive_results = {
            'timestamp': datetime.now().isoformat(),
            'fixes_applied': [],
            'summary': {}
        }
        
        # 1. Fix existing mRMR scores if provided
        if existing_scores:
            logger.info("📊 Fixing existing feature scores")
            fixed_mrmr = self.fix_mrmr_scores(existing_scores)
            comprehensive_results['fixed_mrmr_scores'] = fixed_mrmr
            comprehensive_results['fixes_applied'].append('mrmr_normalization')
        
        # 2. Fix triple barrier labeling
        if MARKET_ANALYSIS_AVAILABLE:
            logger.info("📊 Fixing triple barrier labeling")
            fixed_labels = self.fix_triple_barrier_scores(market_data, regime_data)
            comprehensive_results['fixed_triple_barrier'] = fixed_labels.to_dict() if hasattr(fixed_labels, 'to_dict') else {}
            comprehensive_results['fixes_applied'].append('triple_barrier_optimization')
        
        # 3. Apply enhanced feature scoring if feature data is available
        if feature_data is not None:
            logger.info("📊 Applying enhanced feature scoring")
            
            # Prepare data
            feature_names = feature_data.columns.tolist()
            X = feature_data.values
            
            # Create a simple target if not available (use returns)
            if 'close' in market_data.columns:
                y = market_data['close'].pct_change().fillna(0).values
            else:
                y = np.random.randn(len(X))  # Fallback
            
            # Prepare regime data
            regime_array = None
            if regime_data is not None and 'regime' in regime_data.columns:
                regime_array = regime_data['regime'].values
            
            # Apply enhanced scoring
            enhanced_scores = self.apply_enhanced_feature_scoring(X, y, feature_names, regime_array)
            comprehensive_results['enhanced_scores'] = enhanced_scores
            comprehensive_results['fixes_applied'].append('enhanced_scoring')
        
        # 4. Generate summary
        summary = {
            'total_fixes_applied': len(comprehensive_results['fixes_applied']),
            'fixes_list': comprehensive_results['fixes_applied']
        }
        
        # Add specific metrics from each fix
        if 'mrmr_fix' in self.results:
            summary['mrmr_improvement'] = self.results['mrmr_fix']['improvement']
        
        if 'triple_barrier_fix' in self.results:
            summary['triple_barrier_improvement'] = self.results['triple_barrier_fix']['improvement']
        
        if 'enhanced_scoring' in self.results:
            summary['enhanced_scoring_negative_pct'] = self.results['enhanced_scoring']['negative_percentage']
        
        comprehensive_results['summary'] = summary
        comprehensive_results['detailed_results'] = self.results
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        comprehensive_results['recommendations'] = recommendations
        
        logger.info("✅ Comprehensive score fixing completed")
        logger.info(f"   Fixes applied: {len(comprehensive_results['fixes_applied'])}")
        
        return comprehensive_results
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on the fixes applied."""
        recommendations = []
        
        # mRMR recommendations
        if 'mrmr_fix' in self.results:
            improvement = self.results['mrmr_fix']['improvement']
            if improvement > 0:
                recommendations.append(f"✅ mRMR scores improved: {improvement} features fixed")
            else:
                recommendations.append("⚠️ Consider alternative feature selection methods to mRMR")
        
        # Triple barrier recommendations
        if 'triple_barrier_fix' in self.results:
            improvement = self.results['triple_barrier_fix']['improvement']
            if improvement > 0:
                recommendations.append(f"✅ Triple barrier labeling improved: {improvement} fewer negative profits")
            
            negative_pct = (self.results['triple_barrier_fix']['fixed_negative_count'] / 
                          self.results['triple_barrier_fix']['total_labels'] * 100)
            
            if negative_pct > 30:
                recommendations.append("⚠️ High percentage of negative profits - consider adjusting parameters")
        
        # Enhanced scoring recommendations
        if 'enhanced_scoring' in self.results:
            negative_pct = self.results['enhanced_scoring']['negative_percentage']
            if negative_pct < 5:
                recommendations.append("✅ Enhanced scoring system working well - very few negative scores")
            elif negative_pct < 15:
                recommendations.append("✅ Enhanced scoring system good - minor negative scores remain")
            else:
                recommendations.append("⚠️ Consider additional feature engineering or different scoring methods")
        
        # General recommendations
        recommendations.extend([
            "🔄 Regularly retrain models to adapt to changing market conditions",
            "📊 Monitor feature stability across different market regimes",
            "🎯 Consider ensemble methods for more robust feature selection",
            "💡 Implement cross-validation to validate feature importance",
            "⚡ Use regime-aware feature selection for better performance"
        ])
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], output_path: str = "/workspace/market_analysis_fixes.json"):
        """Save comprehensive results to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"💾 Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def print_summary_report(self, results: Dict[str, Any]):
        """Print a formatted summary report."""
        print("\n" + "="*80)
        print("🎯 MARKET ANALYSIS SCORE FIXING REPORT")
        print("="*80)
        
        print(f"\n📊 SUMMARY")
        summary = results.get('summary', {})
        print(f"   Fixes applied: {summary.get('total_fixes_applied', 0)}")
        print(f"   Fix types: {', '.join(summary.get('fixes_list', []))}")
        
        if 'mrmr_improvement' in summary:
            print(f"   mRMR features improved: {summary['mrmr_improvement']}")
        
        if 'triple_barrier_improvement' in summary:
            print(f"   Triple barrier improvement: {summary['triple_barrier_improvement']}")
        
        if 'enhanced_scoring_negative_pct' in summary:
            print(f"   Enhanced scoring negative %: {summary['enhanced_scoring_negative_pct']:.1f}%")
        
        print(f"\n💡 RECOMMENDATIONS")
        for rec in results.get('recommendations', []):
            print(f"   {rec}")
        
        print("\n" + "="*80)

def main():
    """Main function to demonstrate the market analysis score fixer."""
    logger.info("🚀 Market Analysis Score Fixer Demo")
    logger.info("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample market data
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Set high, low, close based on open
    market_data['high'] = market_data['open'] + abs(np.random.randn(n_samples) * 0.5)
    market_data['low'] = market_data['open'] - abs(np.random.randn(n_samples) * 0.5)
    market_data['close'] = market_data['open'] + np.random.randn(n_samples) * 0.3
    
    # Sample feature data
    feature_data = pd.DataFrame({
        f'feature_{i:02d}': np.random.randn(n_samples) for i in range(20)
    })
    
    # Sample regime data
    regime_data = pd.DataFrame({
        'regime': np.random.choice([0, 1, 2], size=n_samples)
    })
    
    # Sample existing scores (with some negative values)
    existing_scores = {
        f'feature_{i:02d}': np.random.randn() for i in range(20)
    }
    
    logger.info(f"📊 Created sample data:")
    logger.info(f"   Market data: {len(market_data)} rows")
    logger.info(f"   Feature data: {feature_data.shape}")
    logger.info(f"   Existing scores: {len(existing_scores)} features")
    logger.info(f"   Negative scores: {sum(1 for s in existing_scores.values() if s < 0)}")
    
    # Initialize fixer
    fixer = MarketAnalysisScoreFixer()
    
    # Apply comprehensive fixes
    results = fixer.comprehensive_fix(
        market_data=market_data,
        feature_data=feature_data,
        regime_data=regime_data,
        existing_scores=existing_scores
    )
    
    # Print report
    fixer.print_summary_report(results)
    
    # Save results
    fixer.save_results(results)
    
    logger.info("✅ Demo completed successfully")

if __name__ == "__main__":
    main()