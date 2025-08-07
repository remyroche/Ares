"""
Wavelet Feature Selection Demo

This script demonstrates the complete wavelet feature selection workflow:
1. Run full wavelet analysis
2. Build ML models
3. Perform feature selection
4. Identify winner features
5. Create optimized live configurations
"""

import asyncio
import pandas as pd
import numpy as np
import yaml
import time
from pathlib import Path
from typing import Dict, Any

from src.training.wavelet_feature_selection_workflow import WaveletFeatureSelectionWorkflow
from src.utils.logger import system_logger


class WaveletFeatureSelectionDemo:
    """
    Demo class for the wavelet feature selection workflow.
    
    Demonstrates the complete process from full analysis to optimized live configurations.
    """
    
    def __init__(self, config_path: str = "src/config/trading.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = system_logger.getChild("WaveletFeatureSelectionDemo")
        
        # Initialize workflow
        self.workflow = WaveletFeatureSelectionWorkflow(self.config)
        
        # Demo data
        self.price_data = None
        self.volume_data = None
        self.labels = None
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return {}
    
    async def initialize(self) -> bool:
        """Initialize the demo."""
        try:
            self.logger.info("🚀 Initializing Wavelet Feature Selection Demo...")
            
            # Initialize workflow
            success = await self.workflow.initialize()
            if not success:
                self.logger.error("Failed to initialize workflow")
                return False
            
            # Generate demo data
            self._generate_demo_data()
            
            self.logger.info("✅ Wavelet Feature Selection Demo initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error initializing demo: {e}")
            return False
    
    def _generate_demo_data(self) -> None:
        """Generate realistic demo data for the workflow."""
        try:
            # Generate 5000 data points of realistic price data
            np.random.seed(42)
            n_points = 5000
            
            # Base price with trend and volatility
            base_price = 50000
            trend = np.linspace(0, 0.2, n_points)  # 20% upward trend
            volatility = np.random.gamma(2, 0.02, n_points)
            
            # Generate prices with realistic movements
            returns = np.random.normal(0, volatility) + trend
            
            # Add market events
            # Market crash at 1000
            returns[1000:1050] -= 0.1
            # Recovery rally at 2000
            returns[2000:2050] += 0.15
            # Volatility spike at 3000
            returns[3000:3100] += np.random.normal(0, 0.05, 100)
            # Sideways market at 4000
            returns[4000:4500] = np.random.normal(0, 0.01, 500)
            
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Create OHLCV data
            ohlcv_data = []
            for i in range(n_points):
                price = prices[i]
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = price * (1 + np.random.normal(0, 0.005))
                volume = np.random.uniform(1000, 10000)
                
                ohlcv_data.append({
                    "timestamp": pd.Timestamp.now() + pd.Timedelta(minutes=i),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": volume
                })
            
            self.price_data = pd.DataFrame(ohlcv_data)
            self.volume_data = pd.DataFrame({
                "timestamp": self.price_data["timestamp"],
                "volume": self.price_data["volume"]
            })
            
            # Generate labels (simple trend following)
            self.labels = self._generate_labels(prices)
            
            self.logger.info(f"📊 Generated {len(self.price_data)} demo data points")
            self.logger.info(f"📊 Price range: {prices.min():.2f} - {prices.max():.2f}")
            self.logger.info(f"📊 Label distribution: {self.labels.value_counts().to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Error generating demo data: {e}")
    
    def _generate_labels(self, prices: np.ndarray) -> pd.Series:
        """Generate trading labels based on price movements."""
        try:
            # Calculate returns
            returns = np.diff(prices) / prices[:-1]
            
            # Create labels based on future returns (next 10 periods)
            labels = []
            for i in range(len(returns) - 10):
                future_return = np.sum(returns[i:i+10])
                
                if future_return > 0.02:  # 2% gain
                    labels.append(1)  # Buy
                elif future_return < -0.02:  # 2% loss
                    labels.append(-1)  # Sell
                else:
                    labels.append(0)  # Hold
            
            # Pad with holds for the last 10 periods
            labels.extend([0] * 10)
            
            return pd.Series(labels, index=self.price_data.index)
            
        except Exception as e:
            self.logger.error(f"Error generating labels: {e}")
            return pd.Series([0] * len(prices), index=self.price_data.index)
    
    async def run_complete_workflow(self) -> None:
        """Run the complete wavelet feature selection workflow."""
        try:
            self.logger.info("🎬 Starting complete wavelet feature selection workflow...")
            start_time = time.time()
            
            # Run complete workflow
            results = await self.workflow.run_complete_workflow(
                self.price_data, 
                self.volume_data, 
                self.labels
            )
            
            if not results:
                self.logger.error("❌ Workflow failed")
                return
            
            # Display results
            self._display_results(results)
            
            total_time = time.time() - start_time
            self.logger.info(f"✅ Complete workflow finished in {total_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error running complete workflow: {e}")
    
    def _display_results(self, results: Dict[str, Any]) -> None:
        """Display comprehensive workflow results."""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 WAVELET FEATURE SELECTION RESULTS")
            self.logger.info("="*80)
            
            # Step 1: Full Analysis Results
            analysis_results = results["analysis_results"]
            self.logger.info("\n📊 STEP 1: FULL WAVELET ANALYSIS")
            self.logger.info(f"  Total features generated: {analysis_results['feature_count']}")
            self.logger.info(f"  Wavelet features generated: {analysis_results['wavelet_feature_count']}")
            self.logger.info(f"  Computation time: {analysis_results['computation_time']:.2f}s")
            
            # Step 2: ML Model Results
            model_results = results["model_results"]
            self.logger.info("\n🤖 STEP 2: ML MODEL PERFORMANCE")
            for name, perf in model_results["performance"].items():
                self.logger.info(f"  {name.upper()}:")
                self.logger.info(f"    CV Score: {perf['cv_mean']:.3f} ± {perf['cv_std']:.3f}")
                self.logger.info(f"    Test Accuracy: {perf['test_accuracy']:.3f}")
            
            # Step 3: Feature Selection Results
            feature_results = results["feature_results"]
            self.logger.info("\n🔍 STEP 3: FEATURE SELECTION")
            self.logger.info(f"  Features analyzed: {len(feature_results)}")
            
            # Top 10 features
            self.logger.info("  Top 10 features by importance:")
            for i, result in enumerate(feature_results[:10]):
                self.logger.info(f"    {i+1:2d}. {result.feature_name}")
                self.logger.info(f"        Importance: {result.combined_score:.4f}")
                self.logger.info(f"        Type: {result.feature_type}")
                self.logger.info(f"        Cost: {result.computation_cost:.1f}ms")
            
            # Step 4: Winner Features
            winner_features = results["winner_features"]
            self.logger.info("\n🏆 STEP 4: WINNER FEATURES")
            self.logger.info(f"  Winner features selected: {len(winner_features)}")
            
            # Group by type
            wavelet_winners = [f for f in winner_features if f.feature_type == 'wavelet']
            technical_winners = [f for f in winner_features if f.feature_type == 'technical']
            other_winners = [f for f in winner_features if f.feature_type == 'other']
            
            self.logger.info(f"  Wavelet features: {len(wavelet_winners)}")
            self.logger.info(f"  Technical features: {len(technical_winners)}")
            self.logger.info(f"  Other features: {len(other_winners)}")
            
            total_cost = sum(f.computation_cost for f in winner_features)
            self.logger.info(f"  Total computation cost: {total_cost:.1f}ms")
            
            # Step 5: Live Configurations
            live_configs = results["live_configs"]
            self.logger.info("\n⚡ STEP 5: LIVE CONFIGURATIONS")
            self.logger.info("  Generated configurations:")
            for config_name in live_configs.keys():
                self.logger.info(f"    - {config_name}_config.yaml")
            
            # Summary
            summary = results["summary"]
            self.logger.info("\n📈 PERFORMANCE SUMMARY")
            self.logger.info(f"  Computation time reduction: {summary['performance_improvement']['computation_time_reduction']:.1%}")
            self.logger.info(f"  Feature count reduction: {summary['performance_improvement']['feature_count_reduction']:.1%}")
            self.logger.info(f"  Best model accuracy: {summary['workflow_summary']['best_model_accuracy']:.3f}")
            
            # Winner features details
            self.logger.info("\n🏆 WINNER FEATURES DETAILS:")
            for i, winner in enumerate(summary["winner_features"]):
                self.logger.info(f"  {i+1:2d}. {winner['name']}")
                self.logger.info(f"      Score: {winner['importance_score']:.4f}")
                self.logger.info(f"      Cost: {winner['computation_cost_ms']:.1f}ms")
                self.logger.info(f"      Type: {winner['feature_type']}")
            
            self.logger.info("\n" + "="*80)
            self.logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY!")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"Error displaying results: {e}")
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save workflow results to files."""
        try:
            # Save summary report
            summary_path = self.workflow.results_dir / "workflow_summary.yaml"
            with open(summary_path, 'w') as f:
                yaml.dump(results["summary"], f, default_flow_style=False)
            
            # Save feature importance results
            importance_path = self.workflow.results_dir / "feature_importance.csv"
            importance_df = pd.DataFrame([
                {
                    "feature_name": f.feature_name,
                    "permutation_importance": f.permutation_importance,
                    "shap_importance": f.shap_importance,
                    "combined_score": f.combined_score,
                    "feature_type": f.feature_type,
                    "computation_cost_ms": f.computation_cost
                }
                for f in results["feature_results"]
            ])
            importance_df.to_csv(importance_path, index=False)
            
            # Save winner features
            winners_path = self.workflow.results_dir / "winner_features.csv"
            winners_df = pd.DataFrame([
                {
                    "feature_name": f.feature_name,
                    "importance_score": f.combined_score,
                    "computation_cost_ms": f.computation_cost,
                    "feature_type": f.feature_type
                }
                for f in results["winner_features"]
            ])
            winners_df.to_csv(winners_path, index=False)
            
            self.logger.info(f"💾 Results saved to {self.workflow.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


async def main():
    """Main demo function."""
    try:
        # Create and initialize demo
        demo = WaveletFeatureSelectionDemo()
        
        success = await demo.initialize()
        if not success:
            print("❌ Failed to initialize demo")
            return
        
        # Run complete workflow
        await demo.run_complete_workflow()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error in demo: {e}")


if __name__ == "__main__":
    asyncio.run(main())