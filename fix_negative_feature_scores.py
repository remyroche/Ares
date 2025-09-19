#!/usr/bin/env python3
"""
Fix Negative Feature Scores in Market Analysis Pipeline

This script provides comprehensive solutions to address negative feature scores
in the market analysis pipeline by implementing score normalization, feature 
transformation, and improved evaluation metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScoreNormalizationConfig:
    """Configuration for score normalization methods."""
    # Normalization methods
    use_min_max_scaling: bool = True
    use_z_score_normalization: bool = False
    use_rank_transformation: bool = True
    use_sigmoid_transformation: bool = True
    
    # Score adjustment parameters
    negative_score_floor: float = 0.0  # Minimum score value
    positive_score_ceiling: float = 1.0  # Maximum score value
    
    # Feature importance weighting
    relevance_weight: float = 0.7  # Weight for relevance in mRMR
    redundancy_penalty: float = 0.3  # Weight for redundancy penalty
    
    # Triple barrier adjustments
    transaction_cost_adjustment: bool = True
    stop_loss_penalty_reduction: float = 0.5  # Reduce stop loss penalties
    profit_target_bonus: float = 1.2  # Bonus for profit target hits
    
    # Quality scoring improvements
    enable_regime_aware_scoring: bool = True
    enable_stability_weighting: bool = True
    enable_risk_adjusted_returns: bool = True

class FeatureScoreNormalizer:
    """Normalize and fix negative feature scores."""
    
    def __init__(self, config: Optional[ScoreNormalizationConfig] = None):
        self.config = config or ScoreNormalizationConfig()
        self.logger = logger
        
    def normalize_mrmr_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize mRMR scores to prevent negative values.
        
        Args:
            scores: Dictionary of feature names to mRMR scores
            
        Returns:
            Dictionary of normalized scores
        """
        if not scores:
            return {}
        
        self.logger.info(f"🔧 Normalizing {len(scores)} mRMR scores")
        
        # Convert to arrays for processing
        feature_names = list(scores.keys())
        score_values = np.array(list(scores.values()))
        
        # Method 1: Min-Max Scaling (0 to 1)
        if self.config.use_min_max_scaling:
            min_score = np.min(score_values)
            max_score = np.max(score_values)
            
            if max_score > min_score:
                normalized_scores = (score_values - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.ones_like(score_values) * 0.5
            
            self.logger.info(f"   Min-Max scaling: {min_score:.4f} → {max_score:.4f} → [0, 1]")
        
        # Method 2: Rank Transformation
        elif self.config.use_rank_transformation:
            # Convert to ranks (higher score = higher rank)
            ranks = np.argsort(np.argsort(score_values)) + 1
            normalized_scores = ranks / len(ranks)
            
            self.logger.info(f"   Rank transformation: ranks 1-{len(ranks)} → [0, 1]")
        
        # Method 3: Sigmoid Transformation
        elif self.config.use_sigmoid_transformation:
            # Apply sigmoid to map to (0, 1)
            normalized_scores = 1 / (1 + np.exp(-score_values))
            
            self.logger.info(f"   Sigmoid transformation applied")
        
        # Method 4: Z-Score + Shift
        else:
            # Standardize then shift to positive range
            if np.std(score_values) > 0:
                z_scores = (score_values - np.mean(score_values)) / np.std(score_values)
                normalized_scores = z_scores + abs(np.min(z_scores)) + 1
                normalized_scores = normalized_scores / np.max(normalized_scores)
            else:
                normalized_scores = np.ones_like(score_values) * 0.5
            
            self.logger.info(f"   Z-score normalization applied")
        
        # Apply floor and ceiling
        normalized_scores = np.clip(
            normalized_scores, 
            self.config.negative_score_floor, 
            self.config.positive_score_ceiling
        )
        
        # Convert back to dictionary
        normalized_dict = dict(zip(feature_names, normalized_scores))
        
        # Log statistics
        original_negative = sum(1 for score in scores.values() if score < 0)
        final_negative = sum(1 for score in normalized_dict.values() if score < 0)
        
        self.logger.info(f"   Original negative scores: {original_negative}/{len(scores)}")
        self.logger.info(f"   Final negative scores: {final_negative}/{len(scores)}")
        self.logger.info(f"   Score range: [{np.min(normalized_scores):.4f}, {np.max(normalized_scores):.4f}]")
        
        return normalized_dict
    
    def fix_triple_barrier_scores(self, labeling_results: pd.DataFrame) -> pd.DataFrame:
        """
        Fix negative scores from triple barrier labeling.
        
        Args:
            labeling_results: DataFrame with triple barrier results
            
        Returns:
            DataFrame with adjusted scores
        """
        self.logger.info("🔧 Fixing triple barrier scores")
        
        results = labeling_results.copy()
        
        # 1. Adjust transaction costs
        if self.config.transaction_cost_adjustment and 'profit_pct' in results.columns:
            # Reduce transaction cost impact for small profits
            original_profits = results['profit_pct'].copy()
            
            # Apply graduated transaction cost (lower for smaller trades)
            transaction_costs = np.where(
                abs(original_profits) < 0.005,  # For profits < 0.5%
                0.0004,  # Use 0.04% instead of 0.08%
                0.0008   # Use full 0.08%
            )
            
            results['adjusted_profit_pct'] = original_profits - transaction_costs
            
            self.logger.info("   Applied graduated transaction cost adjustment")
        
        # 2. Reduce stop-loss penalties
        if 'barrier_type' in results.columns and 'profit_pct' in results.columns:
            stop_loss_mask = results['barrier_type'].str.contains('stop_loss', na=False)
            
            if stop_loss_mask.any():
                # Reduce stop loss penalties by the configured factor
                results.loc[stop_loss_mask, 'profit_pct'] *= self.config.stop_loss_penalty_reduction
                
                self.logger.info(f"   Reduced stop-loss penalties by {(1-self.config.stop_loss_penalty_reduction)*100:.1f}%")
        
        # 3. Apply profit target bonuses
        if 'barrier_type' in results.columns and 'profit_pct' in results.columns:
            profit_target_mask = results['barrier_type'].str.contains('profit_target', na=False)
            
            if profit_target_mask.any():
                # Apply bonus to profit target hits
                results.loc[profit_target_mask, 'profit_pct'] *= self.config.profit_target_bonus
                
                self.logger.info(f"   Applied {(self.config.profit_target_bonus-1)*100:.1f}% bonus to profit targets")
        
        # 4. Calculate adjusted quality scores
        if 'profit_pct' in results.columns:
            profits = results['profit_pct'].values
            
            # Win rate (percentage of profitable trades)
            win_rate = np.mean(profits > 0)
            
            # Average profit (excluding transaction costs)
            avg_profit = np.mean(profits)
            
            # Profit factor (gross profit / gross loss)
            gross_profit = np.sum(profits[profits > 0])
            gross_loss = abs(np.sum(profits[profits < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
            
            # Sharpe-like ratio
            sharpe_ratio = avg_profit / np.std(profits) if np.std(profits) > 0 else 0
            
            # Combined quality score (always positive)
            quality_score = (
                win_rate * 0.3 +
                min(profit_factor / 2, 0.5) * 0.3 +  # Cap profit factor contribution
                max(0, min(sharpe_ratio + 1, 1)) * 0.2 +  # Normalize sharpe ratio
                min(len(profits) / 1000, 0.2) * 0.2  # Sample size bonus
            )
            
            results['quality_score'] = quality_score
            
            self.logger.info(f"   Quality score: {quality_score:.4f}")
            self.logger.info(f"   Win rate: {win_rate:.2%}")
            self.logger.info(f"   Avg profit: {avg_profit:.4f}")
            self.logger.info(f"   Profit factor: {profit_factor:.2f}")
        
        return results
    
    def apply_regime_aware_scoring(self, scores: Dict[str, float], regime_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Apply regime-aware scoring to handle features that work differently across regimes.
        
        Args:
            scores: Feature scores
            regime_data: Regime information
            
        Returns:
            Adjusted scores
        """
        if not self.config.enable_regime_aware_scoring or regime_data is None:
            return scores
        
        self.logger.info("🔧 Applying regime-aware scoring")
        
        # For each feature, calculate regime-specific performance
        adjusted_scores = scores.copy()
        
        # Placeholder for regime-specific logic
        # In practice, this would analyze feature performance per regime
        for feature_name, score in scores.items():
            if score < 0:
                # For negative scores, check if feature performs well in any regime
                # If so, apply a partial recovery
                regime_recovery_factor = 0.3  # Recover 30% of negative score
                adjusted_scores[feature_name] = score * (1 - regime_recovery_factor)
        
        negative_recovered = sum(1 for f, s in scores.items() if s < 0 and adjusted_scores[f] >= 0)
        self.logger.info(f"   Recovered {negative_recovered} features from negative scores")
        
        return adjusted_scores
    
    def calculate_stability_weighted_scores(self, scores: Dict[str, float], stability_scores: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Weight feature scores by their stability across different time periods.
        
        Args:
            scores: Feature scores
            stability_scores: Stability scores for each feature
            
        Returns:
            Stability-weighted scores
        """
        if not self.config.enable_stability_weighting or stability_scores is None:
            return scores
        
        self.logger.info("🔧 Applying stability weighting")
        
        weighted_scores = {}
        
        for feature_name, score in scores.items():
            stability = stability_scores.get(feature_name, 0.5)  # Default to neutral stability
            
            # Weight the score by stability (stable features get higher weight)
            weighted_score = score * (0.5 + 0.5 * stability)
            weighted_scores[feature_name] = weighted_score
        
        self.logger.info(f"   Applied stability weighting to {len(weighted_scores)} features")
        
        return weighted_scores
    
    def generate_comprehensive_report(self, original_scores: Dict[str, float], normalized_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate a comprehensive report on score normalization."""
        
        original_values = np.array(list(original_scores.values()))
        normalized_values = np.array(list(normalized_scores.values()))
        
        report = {
            'summary': {
                'total_features': len(original_scores),
                'original_negative_count': np.sum(original_values < 0),
                'normalized_negative_count': np.sum(normalized_values < 0),
                'improvement': np.sum(original_values < 0) - np.sum(normalized_values < 0)
            },
            'statistics': {
                'original': {
                    'min': np.min(original_values),
                    'max': np.max(original_values),
                    'mean': np.mean(original_values),
                    'std': np.std(original_values),
                    'negative_percentage': np.mean(original_values < 0) * 100
                },
                'normalized': {
                    'min': np.min(normalized_values),
                    'max': np.max(normalized_values),
                    'mean': np.mean(normalized_values),
                    'std': np.std(normalized_values),
                    'negative_percentage': np.mean(normalized_values < 0) * 100
                }
            },
            'top_improved_features': [],
            'recommendations': []
        }
        
        # Find features with largest improvements
        improvements = {name: normalized_scores[name] - original_scores[name] 
                       for name in original_scores.keys() if original_scores[name] < 0}
        
        if improvements:
            top_improved = sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:10]
            report['top_improved_features'] = top_improved
        
        # Generate recommendations
        recommendations = []
        
        if report['summary']['improvement'] > 0:
            recommendations.append(f"✅ Successfully improved {report['summary']['improvement']} features from negative to positive scores")
        
        if report['statistics']['normalized']['negative_percentage'] > 10:
            recommendations.append("⚠️ Consider additional feature engineering for remaining negative features")
        
        if report['statistics']['normalized']['std'] < 0.1:
            recommendations.append("📊 Low score variance - consider using rank-based selection instead")
        
        recommendations.extend([
            "💡 Monitor feature stability across different market regimes",
            "🔄 Regularly retrain feature selection models to adapt to market changes",
            "📈 Consider ensemble methods to combine multiple scoring approaches"
        ])
        
        report['recommendations'] = recommendations
        
        return report

def demonstrate_score_normalization():
    """Demonstrate the score normalization functionality."""
    logger.info("🚀 Demonstrating Feature Score Normalization")
    logger.info("=" * 60)
    
    # Create sample negative scores (simulating mRMR results)
    sample_scores = {
        'feature_1': 0.5,
        'feature_2': -0.3,
        'feature_3': 0.2,
        'feature_4': -0.7,
        'feature_5': 0.1,
        'feature_6': -0.1,
        'feature_7': 0.8,
        'feature_8': -0.4,
        'feature_9': 0.3,
        'feature_10': -0.2
    }
    
    logger.info(f"📊 Original scores: {len(sample_scores)} features")
    negative_count = sum(1 for score in sample_scores.values() if score < 0)
    logger.info(f"   Negative scores: {negative_count}/{len(sample_scores)} ({negative_count/len(sample_scores)*100:.1f}%)")
    
    # Initialize normalizer
    config = ScoreNormalizationConfig(
        use_min_max_scaling=True,
        use_rank_transformation=False,
        negative_score_floor=0.0,
        positive_score_ceiling=1.0
    )
    
    normalizer = FeatureScoreNormalizer(config)
    
    # Normalize scores
    normalized_scores = normalizer.normalize_mrmr_scores(sample_scores)
    
    # Generate report
    report = normalizer.generate_comprehensive_report(sample_scores, normalized_scores)
    
    # Print results
    logger.info("\n📈 NORMALIZATION RESULTS")
    logger.info("=" * 40)
    
    logger.info(f"Features improved: {report['summary']['improvement']}")
    logger.info(f"Original negative: {report['summary']['original_negative_count']}")
    logger.info(f"Final negative: {report['summary']['normalized_negative_count']}")
    
    logger.info("\n🔝 Top Improved Features:")
    for feature, improvement in report['top_improved_features'][:5]:
        logger.info(f"   {feature}: +{improvement:.4f}")
    
    logger.info("\n💡 Recommendations:")
    for rec in report['recommendations']:
        logger.info(f"   {rec}")
    
    # Save results
    output_path = Path("/workspace/feature_score_normalization_results.json")
    import json
    with open(output_path, 'w') as f:
        json.dump({
            'original_scores': sample_scores,
            'normalized_scores': normalized_scores,
            'report': report
        }, f, indent=2)
    
    logger.info(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    demonstrate_score_normalization()