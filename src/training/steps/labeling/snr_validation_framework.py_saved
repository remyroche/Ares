"""
Signal-to-Noise Ratio (SNR) Validation Framework for Enhanced Layer2 Filtering

This module provides comprehensive testing to validate that enhanced filtering
actually improves signal-to-noise ratio compared to baseline methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = logging.getLogger(__name__)

@dataclass
class SNRTestResults:
    """Results from SNR validation testing."""
    baseline_snr: float
    enhanced_snr: float
    snr_improvement: float
    baseline_noise_level: float
    enhanced_noise_level: float
    baseline_signal_level: float
    enhanced_signal_level: float
    statistical_significance: float
    test_duration: str
    sample_size: int

class SNRValidator:
    """
    Comprehensive SNR validation for enhanced Layer2 filtering.
    
    Tests signal quality across multiple dimensions to ensure
    enhanced filtering actually improves signal-to-noise ratio.
    """
    
    def __init__(self, symbol: str = "ETHUSDT"):
        self.symbol = symbol
        self.test_results = {}
        
    def generate_test_data(self, days: int = 30, noise_level: float = 0.5) -> pd.DataFrame:
        """
        Generate synthetic test data with known signal and noise characteristics.
        
        Args:
            days: Number of days of test data
            noise_level: Level of noise to inject (0-1)
            
        Returns:
            DataFrame with OHLCV data
        """
        # Generate timestamps
        timestamps = pd.date_range(start='2024-01-01', periods=days*24*4, freq='15min')
        
        # Generate base signal (trend + cycles + regime changes)
        t = np.arange(len(timestamps)) / (24*4)  # Convert to days
        
        # Trend component
        trend = 100 * np.exp(0.001 * t)
        
        # Cyclical component (daily + weekly patterns)
        daily_cycle = 2 * np.sin(2 * np.pi * t / 1)  # Daily
        weekly_cycle = 5 * np.sin(2 * np.pi * t / 7)  # Weekly
        
        # Regime changes (volatility shifts)
        regime_volatility = np.ones(len(t))
        regime_changes = [len(t)//4, len(t)//2, 3*len(t)//4]
        for change_point in regime_changes:
            regime_volatility[change_point:] *= np.random.choice([0.5, 2.0])
        
        # Combine signal components
        signal = trend + daily_cycle + weekly_cycle
        
        # Add noise with regime-dependent volatility
        noise = np.random.normal(0, noise_level * regime_volatility, len(t))
        
        # Generate OHLC from signal + noise
        price_signal = signal + noise
        
        # Create realistic OHLC
        high_low_range = 0.002 * price_signal  # 0.2% typical range
        high = price_signal + np.abs(np.random.normal(0, high_low_range/4, len(t)))
        low = price_signal - np.abs(np.random.normal(0, high_low_range/4, len(t)))
        close = price_signal
        open_price = close.shift(1).fillna(close.iloc[0]) + np.random.normal(0, high_low_range/8, len(t))
        
        # Generate volume correlated with price movements
        volume_base = 1000000
        volume_volatility = np.abs(np.diff(close, prepend=close.iloc[0])) * 1000000
        volume = volume_base + volume_volatility + np.random.normal(0, volume_base*0.2, len(t))
        volume = np.abs(volume)
        
        df = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=timestamps)
        
        return df
    
    def calculate_snr(self, signal: pd.Series, noise: pd.Series = None) -> float:
        """
        Calculate signal-to-noise ratio.
        
        Args:
            signal: Signal series
            noise: Noise series (if None, calculated from signal)
            
        Returns:
            SNR value (higher is better)
        """
        if noise is None:
            # Estimate noise as high-frequency component
            noise = signal.diff().rolling(5).std()
        
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Avoid division by zero
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def extract_signal_components(self, price_series: pd.Series) -> Dict[str, pd.Series]:
        """
        Extract signal and noise components from price series.
        
        Args:
            price_series: Input price series
            
        Returns:
            Dictionary with signal and noise components
        """
        # Signal: low-frequency component (trend + cycles)
        signal = price_series.rolling(50, center=True).mean()
        
        # Noise: high-frequency component
        noise = price_series - signal
        
        # Additional noise estimation using Hodrick-Prescott filter
        try:
            from statsmodels.tsa.filters.hp_filter import hpfilter
            trend, cycle = hpfilter(price_series, lamb=1600)
            signal_hp = trend + cycle
            noise_hp = price_series - signal_hp
            
            return {
                'signal_rolling': signal,
                'noise_rolling': noise,
                'signal_hp': signal_hp,
                'noise_hp': noise_hp,
                'signal_combined': (signal + signal_hp) / 2,
                'noise_combined': (noise + noise_hp) / 2
            }
        except ImportError:
            # Fallback if statsmodels not available
            return {
                'signal_rolling': signal,
                'noise_rolling': noise,
                'signal_combined': signal,
                'noise_combined': noise
            }
    
    def test_enhanced_filtering(self, df: pd.DataFrame, 
                              enhanced_params: dict = None) -> SNRTestResults:
        """
        Test enhanced filtering vs baseline for SNR improvement.
        
        Args:
            df: Test data
            enhanced_params: Enhanced filtering parameters
            
        Returns:
            SNR test results
        """
        if enhanced_params is None:
            enhanced_params = {
                'kalman_Q': 1e-4,
                'kalman_R': 0.01,
                'vwap_weight': 0.4,
                'vwap_lookback': 50,
                'median_filter_enabled': True,
                'median_window': 5,
                'adaptive_kalman_enabled': True,
                'robust_vwap_enabled': True
            }
        
        logger.info(f"Starting SNR validation for {self.symbol}")
        
        # Baseline: Simple Kalman + VWAP
        baseline_params = {
            'kalman_Q': 1e-4,
            'kalman_R': 0.01,
            'vwap_weight': 0.4,
            'vwap_lookback': 50,
            'median_filter_enabled': False,
            'adaptive_kalman_enabled': False,
            'robust_vwap_enabled': False
        }
        
        try:
            from .unified_price_layer2 import generate_unified_layer2_price
            
            # Generate baseline price
            baseline_price = generate_unified_layer2_price(df, baseline_params)
            
            # Generate enhanced price
            enhanced_price = generate_unified_layer2_price(df, enhanced_params)
            
            # Extract signal components
            baseline_components = self.extract_signal_components(baseline_price)
            enhanced_components = self.extract_signal_components(enhanced_price)
            
            # Calculate SNR for both
            baseline_snr = self.calculate_snr(
                baseline_components['signal_combined'],
                baseline_components['noise_combined']
            )
            
            enhanced_snr = self.calculate_snr(
                enhanced_components['signal_combined'],
                enhanced_components['noise_combined']
            )
            
            # Calculate noise levels
            baseline_noise_level = np.std(baseline_components['noise_combined'])
            enhanced_noise_level = np.std(enhanced_components['noise_combined'])
            
            # Calculate signal levels
            baseline_signal_level = np.std(baseline_components['signal_combined'])
            enhanced_signal_level = np.std(enhanced_components['signal_combined'])
            
            # Statistical significance test
            statistical_significance = self._test_statistical_significance(
                baseline_components['noise_combined'],
                enhanced_components['noise_combined']
            )
            
            # Create results
            results = SNRTestResults(
                baseline_snr=baseline_snr,
                enhanced_snr=enhanced_snr,
                snr_improvement=enhanced_snr - baseline_snr,
                baseline_noise_level=baseline_noise_level,
                enhanced_noise_level=enhanced_noise_level,
                baseline_signal_level=baseline_signal_level,
                enhanced_signal_level=enhanced_signal_level,
                statistical_significance=statistical_significance,
                test_duration=f"{len(df)} periods",
                sample_size=len(df)
            )
            
            logger.info(f"SNR Test Results:")
            logger.info(f"  Baseline SNR: {baseline_snr:.2f} dB")
            logger.info(f"  Enhanced SNR: {enhanced_snr:.2f} dB")
            logger.info(f"  SNR Improvement: {results.snr_improvement:.2f} dB")
            logger.info(f"  Noise Reduction: {(1 - enhanced_noise_level/baseline_noise_level)*100:.1f}%")
            logger.info(f"  Statistical Significance: {statistical_significance:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"SNR testing failed: {e}")
            raise
    
    def _test_statistical_significance(self, baseline_noise: pd.Series, 
                                     enhanced_noise: pd.Series) -> float:
        """Test statistical significance of noise reduction."""
        try:
            # Paired t-test on noise levels
            t_stat, p_value = stats.ttest_rel(baseline_noise.dropna(), enhanced_noise.dropna())
            return p_value
        except Exception as e:
            logger.warning(f"Statistical significance test failed: {e}")
            return 1.0
    
    def comprehensive_snr_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive SNR analysis across multiple filter combinations.
        
        Args:
            df: Test data
            
        Returns:
            Comprehensive analysis results
        """
        logger.info("Starting comprehensive SNR analysis...")
        
        # Test different filter combinations
        filter_configs = [
            {
                'name': 'baseline',
                'params': {
                    'kalman_Q': 1e-4, 'kalman_R': 0.01, 'vwap_weight': 0.4,
                    'median_filter_enabled': False, 'adaptive_kalman_enabled': False,
                    'robust_vwap_enabled': False
                }
            },
            {
                'name': 'median_only',
                'params': {
                    'kalman_Q': 1e-4, 'kalman_R': 0.01, 'vwap_weight': 0.4,
                    'median_filter_enabled': True, 'median_window': 5,
                    'adaptive_kalman_enabled': False, 'robust_vwap_enabled': False
                }
            },
            {
                'name': 'adaptive_kalman_only',
                'params': {
                    'kalman_Q': 1e-4, 'kalman_R': 0.01, 'vwap_weight': 0.4,
                    'median_filter_enabled': False, 'adaptive_kalman_enabled': True,
                    'robust_vwap_enabled': False
                }
            },
            {
                'name': 'robust_vwap_only',
                'params': {
                    'kalman_Q': 1e-4, 'kalman_R': 0.01, 'vwap_weight': 0.4,
                    'median_filter_enabled': False, 'adaptive_kalman_enabled': False,
                    'robust_vwap_enabled': True
                }
            },
            {
                'name': 'all_enhanced',
                'params': {
                    'kalman_Q': 1e-4, 'kalman_R': 0.01, 'vwap_weight': 0.4,
                    'median_filter_enabled': True, 'median_window': 5,
                    'adaptive_kalman_enabled': True, 'robust_vwap_enabled': True
                }
            }
        ]
        
        results = {}
        
        for config in filter_configs:
            try:
                logger.info(f"Testing {config['name']} configuration...")
                test_result = self.test_enhanced_filtering(df, config['params'])
                results[config['name']] = test_result
            except Exception as e:
                logger.error(f"Failed to test {config['name']}: {e}")
                continue
        
        # Generate comparison summary
        summary = self._generate_comparison_summary(results)
        
        return {
            'detailed_results': results,
            'summary': summary,
            'test_metadata': {
                'symbol': self.symbol,
                'test_date': datetime.now().isoformat(),
                'data_points': len(df),
                'configurations_tested': len(filter_configs)
            }
        }
    
    def _generate_comparison_summary(self, results: Dict[str, SNRTestResults]) -> Dict[str, Any]:
        """Generate summary comparison of all test results."""
        if not results:
            return {'error': 'No valid results to compare'}
        
        # Find best configuration
        best_config = max(results.keys(), key=lambda k: results[k].snr_improvement)
        best_result = results[best_config]
        
        # Calculate improvements
        improvements = {name: result.snr_improvement for name, result in results.items()}
        
        # Noise reduction percentages
        noise_reductions = {}
        for name, result in results.items():
            if result.baseline_noise_level > 0:
                noise_reductions[name] = (1 - result.enhanced_noise_level/result.baseline_noise_level) * 100
        
        return {
            'best_configuration': best_config,
            'best_snr_improvement': best_result.snr_improvement,
            'best_noise_reduction': noise_reductions.get(best_config, 0),
            'all_improvements': improvements,
            'all_noise_reductions': noise_reductions,
            'statistical_significance': {
                name: result.statistical_significance 
                for name, result in results.items()
            }
        }
    
    def visualize_snr_comparison(self, results: Dict[str, SNRTestResults], 
                              save_path: str = None) -> str:
        """
        Visualize SNR comparison across different configurations.
        
        Args:
            results: Test results
            save_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'SNR Validation Results - {self.symbol}', fontsize=16)
            
            configs = list(results.keys())
            snr_improvements = [results[c].snr_improvement for c in configs]
            noise_reductions = [(1 - results[c].enhanced_noise_level/results[c].baseline_noise_level) * 100 
                              for c in configs]
            baseline_snrs = [results[c].baseline_snr for c in configs]
            enhanced_snrs = [results[c].enhanced_snr for c in configs]
            
            # 1. SNR Improvement Comparison
            axes[0, 0].bar(configs, snr_improvements, color='skyblue')
            axes[0, 0].set_title('SNR Improvement (dB)')
            axes[0, 0].set_ylabel('Improvement (dB)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Noise Reduction Percentage
            axes[0, 1].bar(configs, noise_reductions, color='lightgreen')
            axes[0, 1].set_title('Noise Reduction (%)')
            axes[0, 1].set_ylabel('Reduction (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Baseline vs Enhanced SNR
            x = np.arange(len(configs))
            width = 0.35
            axes[1, 0].bar(x - width/2, baseline_snrs, width, label='Baseline', color='orange')
            axes[1, 0].bar(x + width/2, enhanced_snrs, width, label='Enhanced', color='blue')
            axes[1, 0].set_title('Baseline vs Enhanced SNR')
            axes[1, 0].set_ylabel('SNR (dB)')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(configs, rotation=45)
            axes[1, 0].legend()
            
            # 4. Statistical Significance
            p_values = [results[c].statistical_significance for c in configs]
            colors = ['red' if p < 0.05 else 'gray' for p in p_values]
            axes[1, 1].bar(configs, -np.log10(p_values), color=colors)
            axes[1, 1].set_title('Statistical Significance (-log10(p-value))')
            axes[1, 1].set_ylabel('-log10(p-value)')
            axes[1, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            if save_path is None:
                save_path = f"outcomes/snr_validation_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"SNR visualization saved to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return ""
    
    def run_real_data_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run SNR validation on real market data.
        
        Args:
            df: Real market data
            
        Returns:
            Validation results
        """
        logger.info(f"Running SNR validation on real {self.symbol} data...")
        
        # Ensure we have enough data
        if len(df) < 1000:
            raise ValueError(f"Insufficient data: need at least 1000 points, got {len(df)}")
        
        # Run comprehensive analysis
        results = self.comprehensive_snr_analysis(df)
        
        # Generate visualization
        viz_path = self.visualize_snr_comparison(results['detailed_results'])
        
        # Save results
        results_path = f"outcomes/snr_validation_real_{self.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Convert results to JSON-serializable format
            json_results = {
                'summary': results['summary'],
                'test_metadata': results['test_metadata'],
                'visualization_path': viz_path
            }
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"Real data SNR validation saved to {results_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
        
        return results

def run_snr_validation_suite(symbol: str = "ETHUSDT", 
                           use_real_data: bool = True,
                           real_data_path: str = None) -> Dict[str, Any]:
    """
    Run complete SNR validation suite.
    
    Args:
        symbol: Trading symbol
        use_real_data: Whether to use real market data
        real_data_path: Path to real data file
        
    Returns:
        Complete validation results
    """
    validator = SNRValidator(symbol)
    
    if use_real_data:
        # Load real data
        if real_data_path is None:
            # Try to find data in standard locations
            import glob
            data_files = glob.glob(f"historical_data/**/{symbol.lower()}/**/*.parquet", recursive=True)
            if not data_files:
                raise FileNotFoundError(f"No data found for {symbol}")
            real_data_path = data_files[0]
        
        try:
            df = pd.read_parquet(real_data_path)
            # Ensure we have required columns
            required_cols = ['close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")
            
            # Use recent data for testing
            df = df.tail(10000)  # Last 10k points
            
            return validator.run_real_data_validation(df)
            
        except Exception as e:
            logger.error(f"Failed to load real data: {e}")
            logger.info("Falling back to synthetic data...")
            use_real_data = False
    
    if not use_real_data:
        # Generate synthetic test data
        df = validator.generate_test_data(days=30, noise_level=0.5)
        return validator.comprehensive_snr_analysis(df)

# Example usage
if __name__ == "__main__":
    # Run SNR validation
    results = run_snr_validation_suite("ETHUSDT", use_real_data=True)
    
    print("\n=== SNR Validation Results ===")
    print(f"Best configuration: {results['summary']['best_configuration']}")
    print(f"SNR improvement: {results['summary']['best_snr_improvement']:.2f} dB")
    print(f"Noise reduction: {results['summary']['best_noise_reduction']:.1f}%")
    
    # Check if improvement is statistically significant
    best_config = results['summary']['best_configuration']
    p_value = results['summary']['statistical_significance'][best_config]
    print(f"Statistical significance: p={p_value:.4f} {'✅' if p_value < 0.05 else '❌'}")
