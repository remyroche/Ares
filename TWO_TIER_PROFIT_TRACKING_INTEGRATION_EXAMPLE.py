#!/usr/bin/env python3
"""
Two-Tier Profit Tracking Integration Example

This example demonstrates how to use the implemented two-tier profit tracking system
with Analyst and Tactician coordination.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

# Import the implemented components
from src.analyst.analyst import Analyst
from src.tactician.tactician import Tactician
from src.training.steps.step4_analyst_labeling_feature_engineering_components.profit_tracking_ml_integration import ProfitTrackingMLIntegrator


class TwoTierProfitTrackingExample:
    """
    Example class demonstrating the two-tier profit tracking integration.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the two-tier profit tracking example.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.analyst = None
        self.tactician = None
        self.profit_integrator = None
        
    async def initialize_system(self):
        """Initialize the two-tier system with profit tracking."""
        print("🚀 Initializing Two-Tier Profit Tracking System...")
        
        # Initialize Analyst with profit tracking
        self.analyst = Analyst(self.config)
        await self.analyst.initialize()
        print("✅ Analyst initialized with profit tracking")
        
        # Initialize Tactician with profit coordination
        self.tactician = Tactician(self.config)
        await self.tactician.initialize()
        print("✅ Tactician initialized with profit coordination")
        
        # Initialize profit tracking integrator
        self.profit_integrator = ProfitTrackingMLIntegrator(self.config)
        print("✅ Profit tracking integrator initialized")
        
        print("🎯 Two-Tier Profit Tracking System ready!")
    
    async def run_complete_analysis_and_execution(self, market_data: pd.DataFrame, current_price: float, account_balance: float = 10000.0):
        """
        Run complete analysis and execution with profit tracking.
        
        Args:
            market_data: Market data for analysis
            current_price: Current market price
            account_balance: Account balance for position sizing
            
        Returns:
            dict: Complete results from both tiers
        """
        print("\n🔄 Starting complete two-tier analysis and execution...")
        
        # Step 1: Analyst Analysis with Profit Predictions
        print("\n📊 Step 1: Analyst Analysis with Profit Predictions")
        analyst_results = await self._run_analyst_analysis(market_data, current_price)
        
        if not analyst_results:
            print("❌ Analyst analysis failed")
            return {}
        
        print(f"✅ Analyst analysis completed")
        print(f"   - Direction: {analyst_results.get('trading_decision', {}).get('direction', 'Unknown')}")
        print(f"   - Enhanced Confidence: {analyst_results.get('enhanced_confidence', 0.0):.3f}")
        print(f"   - Profit Prediction: {analyst_results.get('profit_predictions', {}).get('profit', 0.0):.4f}")
        
        # Step 2: Tactician Execution with Enhanced Data
        print("\n⚡ Step 2: Tactician Execution with Enhanced Data")
        tactician_results = await self._run_tactician_execution(analyst_results, account_balance)
        
        if not tactician_results:
            print("❌ Tactician execution failed")
            return {}
        
        print(f"✅ Tactician execution completed")
        print(f"   - Position Size: {tactician_results.get('position_sizing', {}).get('final_position_size', 0.0):.4f}")
        print(f"   - Leverage: {tactician_results.get('leverage_sizing', {}).get('final_leverage', 0.0):.1f}x")
        
        # Step 3: Two-Tier Coordination
        print("\n🔄 Step 3: Two-Tier Profit Coordination")
        coordinated_results = await self._run_two_tier_coordination(analyst_results, tactician_results)
        
        if not coordinated_results:
            print("❌ Two-tier coordination failed")
            return {}
        
        print(f"✅ Two-tier coordination completed")
        print(f"   - Combined Profit: {coordinated_results.get('combined_profit', {}).get('profit', 0.0):.4f}")
        print(f"   - Combined Confidence: {coordinated_results.get('combined_confidence', 0.0):.3f}")
        
        return coordinated_results
    
    async def _run_analyst_analysis(self, market_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """Run Analyst analysis with profit predictions."""
        try:
            # Prepare analysis input
            analysis_input = {
                "market_data": market_data,
                "current_price": current_price,
                "current_position": None,  # No current position
                "target_direction": "long"  # Default target direction
            }
            
            # Execute comprehensive analysis
            success = await self.analyst.execute_analysis(analysis_input)
            
            if success:
                return self.analyst.analysis_results
            else:
                print("❌ Analyst analysis failed")
                return {}
                
        except Exception as e:
            print(f"❌ Error in Analyst analysis: {e}")
            return {}
    
    async def _run_tactician_execution(self, analyst_results: Dict[str, Any], account_balance: float) -> Dict[str, Any]:
        """Run Tactician execution with enhanced Analyst data."""
        try:
            # Execute tactics with enhanced Analyst data
            tactician_results = await self.tactician.execute_tactics_with_analyst_results(
                analyst_results=analyst_results,
                account_balance=account_balance
            )
            
            return tactician_results
            
        except Exception as e:
            print(f"❌ Error in Tactician execution: {e}")
            return {}
    
    async def _run_two_tier_coordination(self, analyst_results: Dict[str, Any], tactician_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run two-tier coordination between Analyst and Tactician."""
        try:
            # Coordinate profit predictions between tiers
            coordinated_results = await self.tactician.coordinate_with_analyst(
                analyst_results=analyst_results,
                account_balance=10000.0  # Default account balance
            )
            
            return coordinated_results
            
        except Exception as e:
            print(f"❌ Error in two-tier coordination: {e}")
            return {}
    
    async def demonstrate_profit_tracking_features(self, market_data: pd.DataFrame):
        """Demonstrate individual profit tracking features."""
        print("\n🔍 Demonstrating Profit Tracking Features...")
        
        # 1. Profit-based feature engineering
        print("\n1️⃣ Profit-Based Feature Engineering")
        enhanced_data = self.profit_integrator.integrate_profit_features_into_pipeline(market_data)
        print(f"   - Original features: {market_data.shape[1]}")
        print(f"   - Enhanced features: {enhanced_data.shape[1]}")
        print(f"   - Added {enhanced_data.shape[1] - market_data.shape[1]} profit-based features")
        
        # 2. Multi-output prediction
        print("\n2️⃣ Multi-Output Prediction")
        # Create sample data for demonstration
        sample_features = enhanced_data.iloc[:100]  # Use first 100 samples
        
        # Adapt existing model with profit tracking
        adapted_model = self.profit_integrator.adapt_existing_model(
            model=None,  # Will create new model
            data=enhanced_data,
            target_column="label",
            model_name="demo_model"
        )
        
        # Make predictions with profit tracking
        predictions = self.profit_integrator.predict_with_profit_tracking(
            model_name="demo_model",
            X=sample_features
        )
        
        print(f"   - Direction predictions: {len(predictions['direction'])}")
        print(f"   - Profit predictions: {len(predictions['profit']) if predictions['profit'] is not None else 0}")
        print(f"   - High-value factors: {len(predictions['high_value_trades'])}")
        print(f"   - Confidence scores: {len(predictions['confidence'])}")
        
        # 3. Position sizing with profit tracking
        print("\n3️⃣ Position Sizing with Profit Tracking")
        position_sizing = predictions['position_sizing']
        print(f"   - Base position size: {position_sizing['base_position_size'][0]:.4f}")
        print(f"   - Leverage: {position_sizing['leverage'][0]:.1f}x")
        print(f"   - Risk-adjusted size: {position_sizing['risk_adjusted_size'][0]:.4f}")
        
        return predictions
    
    async def demonstrate_performance_feedback(self, actual_profit: float, predicted_profit: float):
        """Demonstrate performance feedback and weight adjustment."""
        print("\n📊 Demonstrating Performance Feedback...")
        
        # Get coordination summary
        if self.tactician.profit_coordinator:
            summary = self.tactician.profit_coordinator.get_coordination_summary()
            print(f"   - Current Analyst weight: {summary['current_weights']['analyst_weight']:.3f}")
            print(f"   - Current Tactician weight: {summary['current_weights']['tactician_weight']:.3f}")
            print(f"   - Recent performance: {summary['recent_performance']:.3f}")
            
            # Update performance feedback
            feedback = await self.tactician.profit_coordinator.update_performance_feedback(
                actual_profit=actual_profit,
                predicted_profit=predicted_profit,
                execution_quality=0.8  # High execution quality
            )
            
            print(f"   - Prediction accuracy: {feedback['prediction_accuracy']:.3f}")
            print(f"   - Updated Analyst weight: {feedback['updated_weights']['analyst_weight']:.3f}")
            print(f"   - Updated Tactician weight: {feedback['updated_weights']['tactician_weight']:.3f}")
        
        return feedback if 'feedback' in locals() else {}


async def main():
    """Main function demonstrating the two-tier profit tracking system."""
    print("🎯 Two-Tier Profit Tracking Integration Example")
    print("=" * 60)
    
    # Configuration
    config = {
        "analyst": {
            "enable_dual_model_system": True,
            "enable_market_health_analysis": True,
            "enable_liquidation_risk_analysis": True,
            "enable_feature_engineering": True
        },
        "tactician": {
            "tactics_interval": 30,
            "max_history": 100
        },
        "two_tier_profit_coordinator": {
            "analyst_weight": 0.7,
            "tactician_weight": 0.3,
            "confidence_threshold": 0.6,
            "max_history": 100
        },
        "profit_tracking_ml_integration": {
            "enable_profit_features": True,
            "enable_multi_output": True,
            "enable_sample_weighting": True,
            "min_samples_for_profit": 100,
            "time_series_splits": 5,
            "model_save_path": "models/profit_tracking"
        }
    }
    
    # Initialize the example system
    example = TwoTierProfitTrackingExample(config)
    await example.initialize_system()
    
    # Create sample market data (replace with real data)
    print("\n📈 Creating sample market data...")
    np.random.seed(42)
    n_samples = 1000
    
    market_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'open': np.random.normal(100, 5, n_samples),
        'high': np.random.normal(102, 5, n_samples),
        'low': np.random.normal(98, 5, n_samples),
        'close': np.random.normal(101, 5, n_samples),
        'volume': np.random.normal(1000, 200, n_samples),
        'label': np.random.choice([0, 1], n_samples),
        'potential_profit_pct': np.random.normal(0.02, 0.05, n_samples)  # Profit information
    })
    
    # Add some technical indicators
    market_data['sma_20'] = market_data['close'].rolling(20).mean()
    market_data['rsi'] = 50 + np.random.normal(0, 15, n_samples)
    market_data['volatility'] = market_data['close'].rolling(20).std()
    
    current_price = market_data['close'].iloc[-1]
    account_balance = 10000.0
    
    print(f"   - Market data shape: {market_data.shape}")
    print(f"   - Current price: ${current_price:.2f}")
    print(f"   - Account balance: ${account_balance:.2f}")
    
    # Demonstrate profit tracking features
    await example.demonstrate_profit_tracking_features(market_data)
    
    # Run complete analysis and execution
    results = await example.run_complete_analysis_and_execution(
        market_data=market_data,
        current_price=current_price,
        account_balance=account_balance
    )
    
    if results:
        print("\n🎉 Complete Two-Tier Profit Tracking Results:")
        print("=" * 50)
        
        # Display key results
        combined_profit = results.get('combined_profit', {})
        print(f"📊 Combined Profit Prediction: {combined_profit.get('profit', 0.0):.4f}")
        print(f"🎯 Combined Confidence: {results.get('combined_confidence', 0.0):.3f}")
        print(f"📈 Direction: {combined_profit.get('direction', 'Unknown')}")
        print(f"💎 High-Value Factor: {combined_profit.get('high_value_trades', 0.0):.3f}")
        
        # Display position sizing
        tactician_results = results.get('tactician_results', {})
        position_sizing = tactician_results.get('position_sizing', {})
        leverage_sizing = tactician_results.get('leverage_sizing', {})
        
        print(f"\n💰 Position Sizing:")
        print(f"   - Position Size: {position_sizing.get('final_position_size', 0.0):.4f}")
        print(f"   - Leverage: {leverage_sizing.get('final_leverage', 0.0):.1f}x")
        print(f"   - Kelly Position: {position_sizing.get('kelly_position_size', 0.0):.4f}")
        
        # Demonstrate performance feedback
        await example.demonstrate_performance_feedback(
            actual_profit=0.025,  # 2.5% actual profit
            predicted_profit=combined_profit.get('profit', 0.0)
        )
        
        print("\n✅ Two-Tier Profit Tracking Integration Complete!")
        
    else:
        print("\n❌ Two-tier profit tracking failed")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())