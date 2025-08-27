#!/usr/bin/env python3
"""
TPSL (Take Profit/Stop Loss) Integration with Profit Threshold Optimization.

This module shows how to integrate profit tracking with TPSL optimization
in the existing project structure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.utils.logger import get_logger

@dataclass
class TPSLConfig:
    """Configuration for TPSL with profit threshold optimization."""
    profit_take_multiplier: float = 0.002  # 0.2%
    stop_loss_multiplier: float = 0.001    # 0.1%
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    
    # Profit threshold optimization parameters
    enable_profit_thresholds: bool = True
    min_profit_threshold: float = -0.03    # -3%
    max_profit_threshold: float = 0.06     # +6%
    threshold_step: float = 0.005          # 0.5% steps
    optimization_metric: str = "total_profit"  # "total_profit" or "win_rate"
    
    # TPSL-specific parameters
    dynamic_tpsl: bool = True
    profit_based_position_sizing: bool = True
    risk_reward_ratio_target: float = 2.0

class TPSLProfitOptimizer:
    """Optimizes TPSL parameters based on profit tracking data."""
    
    def __init__(self, config: TPSLConfig):
        self.config = config
        self.logger = get_logger("TPSLProfitOptimizer")
        
    def optimize_tpsl_thresholds(self, data: pd.DataFrame) -> Dict:
        """
        Optimize TPSL thresholds based on profit tracking data.
        
        Args:
            data: DataFrame with 'potential_profit_pct' column from triple barrier
            
        Returns:
            Dictionary with optimal thresholds and performance metrics
        """
        self.logger.info("🔧 Optimizing TPSL thresholds with profit tracking...")
        
        # Filter signal data (non-zero labels)
        signal_data = data[data['label'] != 0].copy()
        
        if len(signal_data) == 0:
            self.logger.warning("❌ No signals found for TPSL optimization")
            return {}
        
        # Test different profit thresholds
        thresholds = np.arange(
            self.config.min_profit_threshold,
            self.config.max_profit_threshold,
            self.config.threshold_step
        )
        
        results = []
        
        for threshold in thresholds:
            # Filter trades above threshold
            above_threshold = signal_data['potential_profit_pct'] > threshold
            
            if above_threshold.sum() > 0:
                threshold_data = signal_data[above_threshold]
                
                # Calculate TPSL performance metrics
                avg_profit = threshold_data['potential_profit_pct'].mean()
                trade_count = len(threshold_data)
                win_rate = (threshold_data['potential_profit_pct'] > 0).mean()
                total_profit = avg_profit * trade_count
                
                # Calculate risk-reward ratio
                positive_profits = threshold_data[threshold_data['potential_profit_pct'] > 0]['potential_profit_pct']
                negative_profits = threshold_data[threshold_data['potential_profit_pct'] < 0]['potential_profit_pct']
                
                avg_win = positive_profits.mean() if len(positive_profits) > 0 else 0
                avg_loss = abs(negative_profits.mean()) if len(negative_profits) > 0 else 0
                risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
                
                results.append({
                    'threshold': threshold,
                    'avg_profit': avg_profit,
                    'trade_count': trade_count,
                    'win_rate': win_rate,
                    'total_profit': total_profit,
                    'risk_reward_ratio': risk_reward_ratio,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss
                })
        
        if not results:
            self.logger.error("❌ No valid results found for TPSL optimization")
            return {}
        
        results_df = pd.DataFrame(results)
        
        # Find optimal threshold based on optimization metric
        if self.config.optimization_metric == "total_profit":
            optimal_idx = results_df['total_profit'].idxmax()
        else:  # win_rate
            optimal_idx = results_df['win_rate'].idxmax()
        
        optimal_result = results_df.loc[optimal_idx]
        
        self.logger.info(f"✅ TPSL optimization completed")
        self.logger.info(f"   Optimal threshold: {optimal_result['threshold']:.3f}")
        self.logger.info(f"   Average profit: {optimal_result['avg_profit']:.4f}")
        self.logger.info(f"   Trade count: {optimal_result['trade_count']}")
        self.logger.info(f"   Win rate: {optimal_result['win_rate']:.4f}")
        self.logger.info(f"   Risk-reward ratio: {optimal_result['risk_reward_ratio']:.2f}")
        
        return {
            'optimal_threshold': optimal_result['threshold'],
            'optimal_metrics': optimal_result.to_dict(),
            'all_results': results_df,
            'config': self.config
        }
    
    def calculate_dynamic_tpsl(self, profit_prediction: float, base_price: float) -> Tuple[float, float]:
        """
        Calculate dynamic TPSL levels based on profit prediction.
        
        Args:
            profit_prediction: Predicted profit percentage
            base_price: Entry price
            
        Returns:
            Tuple of (take_profit_price, stop_loss_price)
        """
        if not self.config.dynamic_tpsl:
            # Use fixed TPSL
            take_profit = base_price * (1 + self.config.profit_take_multiplier)
            stop_loss = base_price * (1 - self.config.stop_loss_multiplier)
            return take_profit, stop_loss
        
        # Dynamic TPSL based on profit prediction
        if profit_prediction > 0.02:  # High profit potential
            # More aggressive take profit, tighter stop loss
            take_profit_mult = self.config.profit_take_multiplier * 1.5
            stop_loss_mult = self.config.stop_loss_multiplier * 0.8
        elif profit_prediction > 0.01:  # Medium profit potential
            # Standard TPSL
            take_profit_mult = self.config.profit_take_multiplier
            stop_loss_mult = self.config.stop_loss_multiplier
        else:  # Low profit potential
            # Conservative TPSL
            take_profit_mult = self.config.profit_take_multiplier * 0.8
            stop_loss_mult = self.config.stop_loss_multiplier * 1.2
        
        take_profit = base_price * (1 + take_profit_mult)
        stop_loss = base_price * (1 - stop_loss_mult)
        
        return take_profit, stop_loss
    
    def calculate_position_size(self, profit_prediction: float, base_size: float = 1.0) -> float:
        """
        Calculate position size based on profit prediction.
        
        Args:
            profit_prediction: Predicted profit percentage
            base_size: Base position size
            
        Returns:
            Adjusted position size
        """
        if not self.config.profit_based_position_sizing:
            return base_size
        
        # Scale position size with profit potential
        profit_factor = np.clip(profit_prediction * 20, 0.5, 3.0)
        position_size = base_size * profit_factor
        
        return position_size

class TPSLIntegrationManager:
    """Manages TPSL integration with profit tracking in the existing pipeline."""
    
    def __init__(self, config: TPSLConfig):
        self.config = config
        self.optimizer = TPSLProfitOptimizer(config)
        self.logger = get_logger("TPSLIntegrationManager")
        
    def integrate_with_triple_barrier(self, labeled_data: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate TPSL optimization with triple barrier labeled data.
        
        Args:
            labeled_data: DataFrame with triple barrier labels and profit tracking
            
        Returns:
            Enhanced DataFrame with TPSL recommendations
        """
        self.logger.info("🔗 Integrating TPSL with triple barrier profit tracking...")
        
        # Optimize TPSL thresholds
        optimization_results = self.optimizer.optimize_tpsl_thresholds(labeled_data)
        
        if not optimization_results:
            self.logger.warning("⚠️ TPSL optimization failed, using default values")
            return labeled_data
        
        optimal_threshold = optimization_results['optimal_threshold']
        
        # Add TPSL recommendations to the data
        enhanced_data = labeled_data.copy()
        
        # Filter trades that meet the optimal profit threshold
        meets_threshold = enhanced_data['potential_profit_pct'] > optimal_threshold
        enhanced_data['tpsl_recommended'] = meets_threshold
        
        # Calculate dynamic TPSL levels for recommended trades
        enhanced_data['tpsl_take_profit'] = np.nan
        enhanced_data['tpsl_stop_loss'] = np.nan
        enhanced_data['tpsl_position_size'] = np.nan
        
        for idx in enhanced_data[meets_threshold].index:
            profit_pred = enhanced_data.loc[idx, 'potential_profit_pct']
            close_price = enhanced_data.loc[idx, 'close']
            
            take_profit, stop_loss = self.optimizer.calculate_dynamic_tpsl(
                profit_pred, close_price
            )
            position_size = self.optimizer.calculate_position_size(profit_pred)
            
            enhanced_data.loc[idx, 'tpsl_take_profit'] = take_profit
            enhanced_data.loc[idx, 'tpsl_stop_loss'] = stop_loss
            enhanced_data.loc[idx, 'tpsl_position_size'] = position_size
        
        # Log integration results
        recommended_trades = enhanced_data['tpsl_recommended'].sum()
        total_signals = (enhanced_data['label'] != 0).sum()
        
        self.logger.info(f"✅ TPSL integration completed")
        self.logger.info(f"   Recommended trades: {recommended_trades}/{total_signals}")
        self.logger.info(f"   Recommendation rate: {recommended_trades/total_signals:.2%}")
        
        return enhanced_data
    
    def generate_tpsl_report(self, enhanced_data: pd.DataFrame) -> Dict:
        """
        Generate comprehensive TPSL report with profit tracking analysis.
        
        Args:
            enhanced_data: DataFrame with TPSL recommendations
            
        Returns:
            Dictionary with TPSL analysis report
        """
        self.logger.info("📊 Generating TPSL report...")
        
        # Filter recommended trades
        recommended = enhanced_data[enhanced_data['tpsl_recommended']].copy()
        
        if len(recommended) == 0:
            return {"error": "No recommended trades found"}
        
        # Calculate TPSL performance metrics
        avg_take_profit_pct = ((recommended['tpsl_take_profit'] - recommended['close']) / recommended['close']).mean()
        avg_stop_loss_pct = ((recommended['close'] - recommended['tpsl_stop_loss']) / recommended['close']).mean()
        avg_position_size = recommended['tpsl_position_size'].mean()
        
        # Risk-reward analysis
        risk_reward_ratio = avg_take_profit_pct / avg_stop_loss_pct if avg_stop_loss_pct > 0 else float('inf')
        
        # Profit distribution analysis
        profit_distribution = {
            'high_profit': (recommended['potential_profit_pct'] > 0.02).sum(),
            'medium_profit': ((recommended['potential_profit_pct'] > 0.01) & (recommended['potential_profit_pct'] <= 0.02)).sum(),
            'low_profit': ((recommended['potential_profit_pct'] > 0) & (recommended['potential_profit_pct'] <= 0.01)).sum(),
            'losses': (recommended['potential_profit_pct'] <= 0).sum()
        }
        
        report = {
            'total_recommended_trades': len(recommended),
            'avg_take_profit_pct': avg_take_profit_pct,
            'avg_stop_loss_pct': avg_stop_loss_pct,
            'risk_reward_ratio': risk_reward_ratio,
            'avg_position_size': avg_position_size,
            'profit_distribution': profit_distribution,
            'avg_expected_profit': recommended['potential_profit_pct'].mean(),
            'win_rate': (recommended['potential_profit_pct'] > 0).mean()
        }
        
        self.logger.info(f"📈 TPSL Report Summary:")
        self.logger.info(f"   Total recommended trades: {report['total_recommended_trades']}")
        self.logger.info(f"   Average take profit: {report['avg_take_profit_pct']:.4f}")
        self.logger.info(f"   Average stop loss: {report['avg_stop_loss_pct']:.4f}")
        self.logger.info(f"   Risk-reward ratio: {report['risk_reward_ratio']:.2f}")
        self.logger.info(f"   Expected win rate: {report['win_rate']:.2%}")
        
        return report

# Example usage and integration with existing pipeline
def integrate_tpsl_with_existing_pipeline():
    """Example of how to integrate TPSL with the existing triple barrier pipeline."""
    
    # Configuration
    config = TPSLConfig(
        profit_take_multiplier=0.002,
        stop_loss_multiplier=0.001,
        enable_profit_thresholds=True,
        dynamic_tpsl=True,
        profit_based_position_sizing=True
    )
    
    # Create integration manager
    manager = TPSLIntegrationManager(config)
    
    # This would be called after triple barrier labeling
    # labeled_data = load_triple_barrier_results()  # Your existing data loading
    
    # Integrate TPSL
    # enhanced_data = manager.integrate_with_triple_barrier(labeled_data)
    
    # Generate report
    # report = manager.generate_tpsl_report(enhanced_data)
    
    print("✅ TPSL integration ready for use with existing pipeline")

if __name__ == "__main__":
    integrate_tpsl_with_existing_pipeline()