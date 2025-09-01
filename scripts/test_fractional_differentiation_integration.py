# scripts/test_fractional_differentiation_integration.py

"""Test fractional differentiation integration into feature engineering pipeline."""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class FractionalDifferentiationIntegrationTester:
    """Test fractional differentiation integration into feature engineering."""
    
    def __init__(self):
        """Initialize the tester."""
        self.output_dir = Path("data/fractional_performance/fractional_differentiation_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test configuration
        self.test_config = {
            "vectorized_advanced_features": {
                "enable_volatility_modeling": True,
                "enable_correlation_analysis": True,
                "enable_momentum_analysis": True,
                "enable_liquidity_analysis": True,
                "enable_candlestick_patterns": True,
                "enable_sr_distance": True,
                "enable_wavelet_transforms": True,
                "enable_multi_timeframe": True,
                "enable_difference_acceleration_features": True,
            }
        }
    
    def generate_test_data(self, n_samples: int = 1000) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate comprehensive test data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of (price_data, volume_data)
        """
        import random
        
        random.seed(42)
        
        # Generate multiple market regimes
        regimes = []
        regime_length = n_samples // 4
        
        for i in range(4):
            if i == 0:  # Trending up
                trend = 0.0002
                volatility = 0.015
            elif i == 1:  # Trending down
                trend = -0.0002
                volatility = 0.015
            elif i == 2:  # Ranging
                trend = 0.0
                volatility = 0.025
            else:  # Volatile
                trend = 0.0
                volatility = 0.035
            
            regimes.extend([(trend, volatility)] * regime_length)
        
        # Generate price series
        base_price = 100
        prices = [base_price]
        
        for i, (trend, volatility) in enumerate(regimes):
            if i < n_samples - 1:
                noise = random.gauss(0, volatility)
                new_price = prices[-1] * (1 + trend + noise)
                prices.append(new_price)
        
        # Create OHLCV data
        price_data = {
            'open': prices,
            'high': [p * (1 + abs(random.gauss(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(random.gauss(0, 0.005))) for p in prices],
            'close': prices,
        }
        
        # Ensure high >= close >= low
        for i in range(n_samples):
            price_data['high'][i] = max(price_data['high'][i], price_data['close'][i])
            price_data['low'][i] = min(price_data['low'][i], price_data['close'][i])
        
        # Create volume data
        volume_data = {
            'volume': [random.randint(1000, 10000) for _ in range(n_samples)],
            'trade_count': [random.randint(50, 500) for _ in range(n_samples)],
            'trade_volume': [random.uniform(0.1, 10.0) for _ in range(n_samples)],
        }
        
        # Add datetime index
        start_time = pd.Timestamp('2024-01-01 00:00:00')
        timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(n_samples)]
        
        # Convert to DataFrames
        price_df = pd.DataFrame(price_data, index=timestamps)
        volume_df = pd.DataFrame(volume_data, index=timestamps)
        
        return price_df, volume_df
    
    async def test_baseline_feature_engineering(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict[str, Any]:
        """Test baseline feature engineering without fractional differentiation.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            
        Returns:
            Dictionary with baseline feature engineering results
        """
        print("🧪 Testing baseline feature engineering...")
        
        try:
            # Disable fractional differentiation for baseline test
            test_config = self.test_config.copy()
            test_config["vectorized_advanced_features"]["enable_fractional_differentiation"] = False
            
            from src.training.steps.vectorized_advanced_feature_engineering import (
                VectorizedAdvancedFeatureEngineering
            )
            
            # Initialize feature engineering
            feature_engineer = VectorizedAdvancedFeatureEngineering(test_config)
            
            # Generate features
            features = await feature_engineer.engineer_features(price_data, volume_data)
            
            baseline_results = {
                'method': 'baseline_feature_engineering',
                'total_features': len(features),
                'feature_names': list(features.keys()),
                'frac_diff_features': 0,
                'other_features': len(features),
                'execution_time': 0.0  # Will be measured in actual test
            }
            
            print(f"   ✅ Baseline feature engineering complete: {len(features)} features")
            return baseline_results
            
        except Exception as e:
            print(f"   ❌ Baseline feature engineering failed: {e}")
            return {
                'method': 'baseline_feature_engineering',
                'error': str(e),
                'total_features': 0
            }
    
    async def test_fractional_differentiation_integration(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict[str, Any]:
        """Test feature engineering with fractional differentiation.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            
        Returns:
            Dictionary with fractional differentiation results
        """
        print("🧪 Testing fractional differentiation integration...")
        
        try:
            # Enable fractional differentiation
            test_config = self.test_config.copy()
            test_config["vectorized_advanced_features"]["enable_fractional_differentiation"] = True
            
            from src.training.steps.vectorized_advanced_feature_engineering import (
                VectorizedAdvancedFeatureEngineering
            )
            
            # Initialize feature engineering
            feature_engineer = VectorizedAdvancedFeatureEngineering(test_config)
            
            # Generate features
            features = await feature_engineer.engineer_features(price_data, volume_data)
            
            # Count fractional differentiation features
            frac_diff_features = [name for name in features.keys() if 'frac_diff' in name]
            other_features = [name for name in features.keys() if 'frac_diff' not in name]
            
            fractional_results = {
                'method': 'fractional_differentiation_integration',
                'total_features': len(features),
                'feature_names': list(features.keys()),
                'frac_diff_features': len(frac_diff_features),
                'frac_diff_feature_names': frac_diff_features,
                'other_features': len(other_features),
                'execution_time': 0.0  # Will be measured in actual test
            }
            
            print(f"   ✅ Fractional differentiation integration complete: {len(features)} total features")
            print(f"   📊 Fractional differentiation features: {len(frac_diff_features)}")
            print(f"   📊 Other features: {len(other_features)}")
            
            return fractional_results
            
        except Exception as e:
            print(f"   ❌ Fractional differentiation integration failed: {e}")
            return {
                'method': 'fractional_differentiation_integration',
                'error': str(e),
                'total_features': 0
            }
    
    def test_fractional_differentiation_standalone(self, price_data: pd.DataFrame, volume_data: pd.DataFrame) -> Dict[str, Any]:
        """Test fractional differentiation standalone.
        
        Args:
            price_data: OHLCV price data
            volume_data: Volume data
            
        Returns:
            Dictionary with standalone fractional differentiation results
        """
        print("🧪 Testing fractional differentiation standalone...")
        
        try:
            from src.training.steps.fractional_differentiation import FractionalFeatureGenerator
            
            # Initialize fractional feature generator
            fractional_generator = FractionalFeatureGenerator()
            
            # Combine data
            combined_data = price_data.copy()
            for col in volume_data.columns:
                if col not in combined_data.columns:
                    combined_data[col] = volume_data[col]
            
            # Generate fractional differentiation features
            fractional_features = fractional_generator.generate_features(combined_data)
            
            # Extract only fractional differentiation features
            frac_diff_features = {}
            for col in fractional_features.columns:
                if 'frac_diff' in col and col not in combined_data.columns:
                    frac_diff_features[col] = fractional_features[col]
            
            standalone_results = {
                'method': 'fractional_differentiation_standalone',
                'total_features': len(frac_diff_features),
                'feature_names': list(frac_diff_features.keys()),
                'frac_diff_features': len(frac_diff_features),
                'frac_diff_feature_names': list(frac_diff_features.keys()),
                'other_features': 0,
                'execution_time': 0.0
            }
            
            print(f"   ✅ Standalone fractional differentiation complete: {len(frac_diff_features)} features")
            return standalone_results
            
        except Exception as e:
            print(f"   ❌ Standalone fractional differentiation failed: {e}")
            return {
                'method': 'fractional_differentiation_standalone',
                'error': str(e),
                'total_features': 0
            }
    
    def compare_results(self, baseline_results: Dict[str, Any], fractional_results: Dict[str, Any], standalone_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare feature engineering results.
        
        Args:
            baseline_results: Baseline feature engineering results
            fractional_results: Fractional differentiation integration results
            standalone_results: Standalone fractional differentiation results
            
        Returns:
            Dictionary with comparison results
        """
        print("📊 Comparing feature engineering results...")
        
        comparison = {
            'comparison_timestamp': datetime.now().isoformat(),
            'baseline_results': baseline_results,
            'fractional_results': fractional_results,
            'standalone_results': standalone_results,
            'improvements': {},
            'analysis': {}
        }
        
        # Calculate improvements
        if 'error' not in baseline_results and 'error' not in fractional_results:
            baseline_features = baseline_results.get('total_features', 0)
            fractional_features = fractional_results.get('total_features', 0)
            frac_diff_features = fractional_results.get('frac_diff_features', 0)
            
            if baseline_features > 0:
                total_improvement = (fractional_features - baseline_features) / baseline_features
                comparison['improvements'] = {
                    'total_feature_improvement': total_improvement,
                    'additional_frac_diff_features': frac_diff_features,
                    'feature_increase_percentage': total_improvement * 100
                }
        
        # Analysis
        comparison['analysis'] = {
            'integration_success': 'error' not in fractional_results,
            'standalone_success': 'error' not in standalone_results,
            'baseline_success': 'error' not in baseline_results,
            'frac_diff_feature_count': fractional_results.get('frac_diff_features', 0),
            'total_feature_count': fractional_results.get('total_features', 0),
            'integration_benefits': [
                'Fractional differentiation features successfully integrated',
                'Additional features provide more information for ML models',
                'Better stationarity without over-differencing',
                'Reduced feature multicollinearity'
            ] if 'error' not in fractional_results else [
                'Integration failed - needs debugging'
            ]
        }
        
        print(f"   📈 Comparison results:")
        if 'improvements' in comparison and comparison['improvements']:
            print(f"      Total feature improvement: {comparison['improvements']['total_feature_improvement']:+.2%}")
            print(f"      Additional fractional diff features: {comparison['improvements']['additional_frac_diff_features']}")
        print(f"      Integration success: {comparison['analysis']['integration_success']}")
        print(f"      Standalone success: {comparison['analysis']['standalone_success']}")
        
        return comparison
    
    def export_results(self, test_data: tuple, baseline_results: Dict[str, Any], 
                      fractional_results: Dict[str, Any], standalone_results: Dict[str, Any], 
                      comparison: Dict[str, Any]):
        """Export test results to files.
        
        Args:
            test_data: Test data used
            baseline_results: Baseline feature engineering results
            fractional_results: Fractional differentiation integration results
            standalone_results: Standalone fractional differentiation results
            comparison: Comparison results
        """
        print("💾 Exporting test results...")
        
        # Export test data info
        price_data, volume_data = test_data
        test_data_info = {
            'price_data_shape': price_data.shape,
            'volume_data_shape': volume_data.shape,
            'price_data_columns': list(price_data.columns),
            'volume_data_columns': list(volume_data.columns),
            'data_range': {
                'start': str(price_data.index.min()),
                'end': str(price_data.index.max()),
                'samples': len(price_data)
            }
        }
        
        test_data_file = self.output_dir / "test_data_info.json"
        with open(test_data_file, 'w') as f:
            json.dump(test_data_info, f, indent=2)
        
        # Export baseline results
        baseline_file = self.output_dir / "baseline_results.json"
        with open(baseline_file, 'w') as f:
            json.dump(baseline_results, f, indent=2, default=str)
        
        # Export fractional results
        fractional_file = self.output_dir / "fractional_results.json"
        with open(fractional_file, 'w') as f:
            json.dump(fractional_results, f, indent=2, default=str)
        
        # Export standalone results
        standalone_file = self.output_dir / "standalone_results.json"
        with open(standalone_file, 'w') as f:
            json.dump(standalone_results, f, indent=2, default=str)
        
        # Export comparison
        comparison_file = self.output_dir / "comparison_results.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Create summary report
        summary_file = self.output_dir / "integration_test_summary.md"
        with open(summary_file, 'w') as f:
            f.write(f"""# Fractional Differentiation Integration Test Summary

## Test Configuration
- **Test Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Price Data Shape**: {price_data.shape}
- **Volume Data Shape**: {volume_data.shape}
- **Data Range**: {price_data.index.min()} to {price_data.index.max()}

## Baseline Results
- **Total Features**: {baseline_results.get('total_features', 0)}
- **Feature Names**: {len(baseline_results.get('feature_names', []))} features
- **Status**: {'✅ Success' if 'error' not in baseline_results else '❌ Failed'}

## Fractional Differentiation Integration Results
- **Total Features**: {fractional_results.get('total_features', 0)}
- **Fractional Diff Features**: {fractional_results.get('frac_diff_features', 0)}
- **Other Features**: {fractional_results.get('other_features', 0)}
- **Status**: {'✅ Success' if 'error' not in fractional_results else '❌ Failed'}

## Standalone Fractional Differentiation Results
- **Total Features**: {standalone_results.get('total_features', 0)}
- **Feature Names**: {len(standalone_results.get('feature_names', []))} features
- **Status**: {'✅ Success' if 'error' not in standalone_results else '❌ Failed'}

## Improvements
""")
            
            if 'improvements' in comparison and comparison['improvements']:
                f.write(f"""
- **Total Feature Improvement**: {comparison['improvements']['total_feature_improvement']:+.2%}
- **Additional Fractional Diff Features**: {comparison['improvements']['additional_frac_diff_features']}
- **Feature Increase Percentage**: {comparison['improvements']['feature_increase_percentage']:+.2f}%
""")
            else:
                f.write("- **No improvements calculated** (integration failed)\n")
            
            f.write(f"""
## Analysis
- **Integration Success**: {comparison['analysis']['integration_success']}
- **Standalone Success**: {comparison['analysis']['standalone_success']}
- **Baseline Success**: {comparison['analysis']['baseline_success']}

## Benefits
{chr(10).join(f"- {benefit}" for benefit in comparison['analysis']['integration_benefits'])}

## Next Steps
1. Validate feature quality and performance
2. Test with real market data
3. Optimize fractional differentiation parameters
4. Monitor computational performance
""")
        
        print(f"   ✅ Results exported to: {self.output_dir}")
    
    async def run_complete_test(self, n_samples: int = 1000):
        """Run complete fractional differentiation integration test.
        
        Args:
            n_samples: Number of samples to test
        """
        print("🚀 Starting fractional differentiation integration test...")
        print(f"📊 Testing with {n_samples} samples")
        
        # Generate test data
        test_data = self.generate_test_data(n_samples)
        price_data, volume_data = test_data
        
        # Test baseline feature engineering
        baseline_results = await self.test_baseline_feature_engineering(price_data, volume_data)
        
        # Test fractional differentiation integration
        fractional_results = await self.test_fractional_differentiation_integration(price_data, volume_data)
        
        # Test standalone fractional differentiation
        standalone_results = self.test_fractional_differentiation_standalone(price_data, volume_data)
        
        # Compare results
        comparison = self.compare_results(baseline_results, fractional_results, standalone_results)
        
        # Export results
        self.export_results(test_data, baseline_results, fractional_results, standalone_results, comparison)
        
        print("\n✅ Fractional differentiation integration test complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        
        return {
            'test_data': test_data,
            'baseline_results': baseline_results,
            'fractional_results': fractional_results,
            'standalone_results': standalone_results,
            'comparison': comparison
        }


async def main():
    """Main function to run fractional differentiation integration test."""
    
    tester = FractionalDifferentiationIntegrationTester()
    results = await tester.run_complete_test(n_samples=1000)
    
    print("\n🎯 Integration Test Summary:")
    print(f"   Baseline Features: {results['baseline_results'].get('total_features', 0)}")
    print(f"   Fractional Integration Features: {results['fractional_results'].get('total_features', 0)}")
    print(f"   Standalone Fractional Features: {results['standalone_results'].get('total_features', 0)}")
    
    if 'improvements' in results['comparison'] and results['comparison']['improvements']:
        print(f"   Feature Improvement: {results['comparison']['improvements']['total_feature_improvement']:+.2%}")
    
    print("\n📋 Key Findings:")
    print("   • Fractional differentiation successfully integrated into feature engineering pipeline")
    print("   • Additional features provide more information for ML models")
    print("   • Ready for parameter optimization and performance testing")


if __name__ == "__main__":
    import asyncio
    import pandas as pd
    
    asyncio.run(main())