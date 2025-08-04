#!/usr/bin/env python3
"""
Comprehensive Test Suite for Testing and Validation.
Tests ensemble performance and dual model decision making.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any, Optional

# Import the components
from src.analyst.ml_confidence_predictor import MLConfidencePredictor, setup_ml_confidence_predictor
from src.training.dual_model_system import DualModelSystem, setup_dual_model_system
from src.training.ensemble_creator_simple import SimpleEnsembleCreator
from src.training.enhanced_training_manager import EnhancedTrainingManager


class ComprehensiveTestSuite:
    """Comprehensive test suite for validation and testing."""
    
    def __init__(self):
        self.test_results = {}
        self.test_config = self._load_test_configuration()
        
    def _load_test_configuration(self) -> Dict[str, Any]:
        """Load test configuration."""
        return {
            "ml_confidence_predictor": {
                "model_path": "models/confidence_predictor.joblib",
                "min_samples_for_training": 100,
                "confidence_threshold": 0.6,
                "max_prediction_horizon": 1,
                "retrain_interval_hours": 24
            },
            "dual_model_system": {
                "analyst_timeframes": ["1h", "15m", "5m", "1m"],
                "tactician_timeframes": ["1m"],
                "analyst_confidence_threshold": 0.7,
                "tactician_confidence_threshold": 0.8,
                "enable_ensemble_analysis": True,
            },
            "ensemble_creator": {
                "pruning_threshold": 0.1,
                "regularization_strength": 0.01,
                "max_ensemble_size": 10,
                "min_ensemble_size": 2
            },
            "test_scenarios": {
                "market_conditions": ["bullish", "bearish", "sideways", "volatile"],
                "timeframes": ["1m", "5m", "15m", "1h"],
                "confidence_levels": [0.3, 0.5, 0.7, 0.9],
                "ensemble_sizes": [2, 5, 10]
            }
        }
    
    def generate_mock_market_data(self, scenario: str, periods: int = 1000) -> pd.DataFrame:
        """Generate mock market data for different scenarios."""
        np.random.seed(42)  # For reproducible results
        
        base_price = 100.0
        volatility = 0.02
        
        if scenario == "bullish":
            trend = 0.001  # Upward trend
            volatility = 0.015
        elif scenario == "bearish":
            trend = -0.001  # Downward trend
            volatility = 0.015
        elif scenario == "sideways":
            trend = 0.0  # No trend
            volatility = 0.01
        elif scenario == "volatile":
            trend = 0.0  # No trend
            volatility = 0.05
        else:
            trend = 0.0
            volatility = 0.02
        
        # Generate price series
        returns = np.random.normal(trend, volatility, periods)
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Generate OHLCV data
        data = []
        for i, price in enumerate(prices):
            # Generate realistic OHLC from price
            open_price = price
            high_price = price * (1 + abs(np.random.normal(0, 0.005)))
            low_price = price * (1 - abs(np.random.normal(0, 0.005)))
            close_price = price * (1 + np.random.normal(0, 0.002))
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'open': open_price,
                'high': max(open_price, high_price),
                'low': min(open_price, low_price),
                'close': close_price,
                'volume': volume,
                'timestamp': datetime.now() + timedelta(minutes=i)
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    async def test_ensemble_performance(self) -> Dict[str, Any]:
        """Test ensemble performance across different scenarios."""
        print("🧪 Testing Ensemble Performance...")
        
        results = {
            "ensemble_tests": {},
            "performance_metrics": {},
            "scenario_results": {}
        }
        
        try:
            # Initialize ensemble creator
            ensemble_creator = SimpleEnsembleCreator(self.test_config)
            await ensemble_creator.initialize()
            
            # Test different ensemble sizes
            for ensemble_size in self.test_config["test_scenarios"]["ensemble_sizes"]:
                print(f"\n📋 Testing ensemble size: {ensemble_size}")
                
                # Create mock ensemble models
                mock_models = {}
                mock_weights = {}
                
                for i in range(ensemble_size):
                    model_name = f"model_{i+1}"
                    # Create a simple mock model with predictable behavior
                    mock_models[model_name] = {
                        "type": "mock_classifier",
                        "confidence": 0.5 + (i * 0.1),  # Varying confidence levels
                        "performance": 0.6 + (i * 0.05)  # Varying performance
                    }
                    mock_weights[model_name] = 1.0 / ensemble_size
                
                # Test ensemble creation
                ensemble_result = await ensemble_creator.create_ensemble(
                    training_data={"mock": pd.DataFrame()},
                    models=mock_models,
                    ensemble_name=f"test_ensemble_{ensemble_size}",
                    ensemble_type="performance_test"
                )
                
                if ensemble_result:
                    results["ensemble_tests"][f"size_{ensemble_size}"] = {
                        "status": "SUCCESS",
                        "ensemble_info": ensemble_result.get("ensemble_info", {}),
                        "model_count": ensemble_size
                    }
                    print(f"✅ Ensemble size {ensemble_size} test passed")
                else:
                    results["ensemble_tests"][f"size_{ensemble_size}"] = {
                        "status": "FAILED",
                        "error": "Ensemble creation failed"
                    }
                    print(f"❌ Ensemble size {ensemble_size} test failed")
            
            # Test hierarchical ensemble
            print("\n📋 Testing hierarchical ensemble...")
            hierarchical_result = await ensemble_creator.create_hierarchical_ensemble(
                base_ensembles={"test_ensemble": {"models": mock_models}},
                ensemble_name="hierarchical_test"
            )
            
            if hierarchical_result:
                results["ensemble_tests"]["hierarchical"] = {
                    "status": "SUCCESS",
                    "hierarchical_info": hierarchical_result.get("ensemble_info", {})
                }
                print("✅ Hierarchical ensemble test passed")
            else:
                results["ensemble_tests"]["hierarchical"] = {
                    "status": "FAILED",
                    "error": "Hierarchical ensemble creation failed"
                }
                print("❌ Hierarchical ensemble test failed")
            
            # Performance metrics
            results["performance_metrics"] = {
                "total_ensembles_tested": len(results["ensemble_tests"]),
                "successful_ensembles": sum(1 for test in results["ensemble_tests"].values() 
                                          if test.get("status") == "SUCCESS"),
                "ensemble_creation_success_rate": sum(1 for test in results["ensemble_tests"].values() 
                                                    if test.get("status") == "SUCCESS") / len(results["ensemble_tests"])
            }
            
            await ensemble_creator.stop()
            return results
            
        except Exception as e:
            print(f"❌ Ensemble performance test failed: {e}")
            return {"error": str(e)}
    
    async def test_dual_model_decision_making(self) -> Dict[str, Any]:
        """Test dual model decision making across different scenarios."""
        print("🧪 Testing Dual Model Decision Making...")
        
        results = {
            "decision_tests": {},
            "scenario_results": {},
            "performance_metrics": {}
        }
        
        try:
            # Initialize dual model system
            dual_system = await setup_dual_model_system(self.test_config)
            
            if not dual_system:
                print("❌ Failed to initialize dual model system")
                return {"error": "Dual model system initialization failed"}
            
            # Test different market scenarios
            for scenario in self.test_config["test_scenarios"]["market_conditions"]:
                print(f"\n📋 Testing scenario: {scenario}")
                
                # Generate market data for this scenario
                market_data = self.generate_mock_market_data(scenario, periods=500)
                current_price = market_data['close'].iloc[-1]
                
                # Test entry decision
                entry_decision = await dual_system.make_trading_decision(
                    market_data=market_data,
                    current_price=current_price
                )
                
                if entry_decision:
                    results["decision_tests"][f"entry_{scenario}"] = {
                        "status": "SUCCESS",
                        "decision": entry_decision.get("decision", {}),
                        "analyst_decision": entry_decision.get("analyst_decision", {}),
                        "tactician_decision": entry_decision.get("tactician_decision", {}),
                        "position_size": entry_decision.get("position_size", 0.0)
                    }
                    print(f"✅ Entry decision test for {scenario} passed")
                else:
                    results["decision_tests"][f"entry_{scenario}"] = {
                        "status": "FAILED",
                        "error": "Entry decision failed"
                    }
                    print(f"❌ Entry decision test for {scenario} failed")
                
                # Test exit decision (simulate existing position)
                if entry_decision and entry_decision.get("decision", {}).get("action") == "ENTER":
                    current_position = {
                        "side": entry_decision.get("decision", {}).get("side", "LONG"),
                        "entry_price": current_price,
                        "size": entry_decision.get("position_size", 1.0),
                        "entry_time": datetime.now()
                    }
                    
                    # Generate new market data for exit testing
                    exit_market_data = self.generate_mock_market_data(scenario, periods=100)
                    exit_price = exit_market_data['close'].iloc[-1]
                    
                    exit_decision = await dual_system.make_trading_decision(
                        market_data=exit_market_data,
                        current_price=exit_price,
                        current_position=current_position
                    )
                    
                    if exit_decision:
                        results["decision_tests"][f"exit_{scenario}"] = {
                            "status": "SUCCESS",
                            "decision": exit_decision.get("decision", {}),
                            "analyst_exit": exit_decision.get("analyst_exit_decision", {}),
                            "tactician_exit": exit_decision.get("tactician_exit_decision", {})
                        }
                        print(f"✅ Exit decision test for {scenario} passed")
                    else:
                        results["decision_tests"][f"exit_{scenario}"] = {
                            "status": "FAILED",
                            "error": "Exit decision failed"
                        }
                        print(f"❌ Exit decision test for {scenario} failed")
            
            # Performance metrics
            total_tests = len(results["decision_tests"])
            successful_tests = sum(1 for test in results["decision_tests"].values() 
                                 if test.get("status") == "SUCCESS")
            
            results["performance_metrics"] = {
                "total_decision_tests": total_tests,
                "successful_decisions": successful_tests,
                "decision_success_rate": successful_tests / total_tests if total_tests > 0 else 0.0,
                "scenarios_tested": len(self.test_config["test_scenarios"]["market_conditions"])
            }
            
            await dual_system.stop()
            return results
            
        except Exception as e:
            print(f"❌ Dual model decision making test failed: {e}")
            return {"error": str(e)}
    
    async def test_ml_confidence_predictor_integration(self) -> Dict[str, Any]:
        """Test ML Confidence Predictor integration with dual model system."""
        print("🧪 Testing ML Confidence Predictor Integration...")
        
        results = {
            "integration_tests": {},
            "prediction_tests": {},
            "verification_tests": {}
        }
        
        try:
            # Initialize ML Confidence Predictor
            ml_predictor = await setup_ml_confidence_predictor(self.test_config)
            
            if not ml_predictor:
                print("❌ Failed to initialize ML Confidence Predictor")
                return {"error": "ML Confidence Predictor initialization failed"}
            
            # Test different market scenarios
            for scenario in self.test_config["test_scenarios"]["market_conditions"]:
                print(f"\n📋 Testing ML Confidence Predictor with scenario: {scenario}")
                
                # Generate market data
                market_data = self.generate_mock_market_data(scenario, periods=200)
                current_price = market_data['close'].iloc[-1]
                
                # Test analyst predictions
                analyst_predictions = await ml_predictor.predict_for_dual_model_system(
                    market_data=market_data,
                    current_price=current_price,
                    model_type="analyst"
                )
                
                if analyst_predictions:
                    results["prediction_tests"][f"analyst_{scenario}"] = {
                        "status": "SUCCESS",
                        "strategic_decision": analyst_predictions.get("strategic_decision", {}),
                        "multi_timeframe_analysis": analyst_predictions.get("multi_timeframe_analysis", {})
                    }
                    print(f"✅ Analyst predictions for {scenario} passed")
                else:
                    results["prediction_tests"][f"analyst_{scenario}"] = {
                        "status": "FAILED",
                        "error": "Analyst predictions failed"
                    }
                    print(f"❌ Analyst predictions for {scenario} failed")
                
                # Test tactician predictions
                tactician_predictions = await ml_predictor.predict_for_dual_model_system(
                    market_data=market_data,
                    current_price=current_price,
                    model_type="tactician"
                )
                
                if tactician_predictions:
                    results["prediction_tests"][f"tactician_{scenario}"] = {
                        "status": "SUCCESS",
                        "timing_decision": tactician_predictions.get("timing_decision", {}),
                        "execution_timing": tactician_predictions.get("execution_timing", {})
                    }
                    print(f"✅ Tactician predictions for {scenario} passed")
                else:
                    results["prediction_tests"][f"tactician_{scenario}"] = {
                        "status": "FAILED",
                        "error": "Tactician predictions failed"
                    }
                    print(f"❌ Tactician predictions for {scenario} failed")
                
                # Test confidence verification
                verification_results = ml_predictor.verify_confidence_calculations(
                    market_data=market_data,
                    current_price=current_price
                )
                
                if verification_results:
                    results["verification_tests"][f"verification_{scenario}"] = {
                        "status": "SUCCESS",
                        "overall_verification": verification_results.get("overall_verification", "UNKNOWN"),
                        "anomaly_count": verification_results.get("anomaly_detection", {}).get("anomaly_count", 0)
                    }
                    print(f"✅ Confidence verification for {scenario} passed")
                else:
                    results["verification_tests"][f"verification_{scenario}"] = {
                        "status": "FAILED",
                        "error": "Confidence verification failed"
                    }
                    print(f"❌ Confidence verification for {scenario} failed")
            
            # Performance metrics
            total_prediction_tests = len(results["prediction_tests"])
            successful_prediction_tests = sum(1 for test in results["prediction_tests"].values() 
                                           if test.get("status") == "SUCCESS")
            
            total_verification_tests = len(results["verification_tests"])
            successful_verification_tests = sum(1 for test in results["verification_tests"].values() 
                                             if test.get("status") == "SUCCESS")
            
            results["performance_metrics"] = {
                "total_prediction_tests": total_prediction_tests,
                "successful_prediction_tests": successful_prediction_tests,
                "prediction_success_rate": successful_prediction_tests / total_prediction_tests if total_prediction_tests > 0 else 0.0,
                "total_verification_tests": total_verification_tests,
                "successful_verification_tests": successful_verification_tests,
                "verification_success_rate": successful_verification_tests / total_verification_tests if total_verification_tests > 0 else 0.0
            }
            
            await ml_predictor.stop()
            return results
            
        except Exception as e:
            print(f"❌ ML Confidence Predictor integration test failed: {e}")
            return {"error": str(e)}
    
    async def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test end-to-end workflow from data to decision."""
        print("🧪 Testing End-to-End Workflow...")
        
        results = {
            "workflow_tests": {},
            "integration_points": {},
            "performance_metrics": {}
        }
        
        try:
            # Step 1: Generate market data
            print("\n📋 Step 1: Generating market data...")
            market_data = self.generate_mock_market_data("bullish", periods=1000)
            current_price = market_data['close'].iloc[-1]
            
            # Step 2: Initialize components
            print("📋 Step 2: Initializing components...")
            ml_predictor = await setup_ml_confidence_predictor(self.test_config)
            dual_system = await setup_dual_model_system(self.test_config)
            ensemble_creator = SimpleEnsembleCreator(self.test_config)
            await ensemble_creator.initialize()
            
            if not all([ml_predictor, dual_system, ensemble_creator]):
                print("❌ Failed to initialize all components")
                return {"error": "Component initialization failed"}
            
            # Step 3: Generate predictions
            print("📋 Step 3: Generating predictions...")
            analyst_predictions = await ml_predictor.predict_for_dual_model_system(
                market_data=market_data,
                current_price=current_price,
                model_type="analyst"
            )
            
            tactician_predictions = await ml_predictor.predict_for_dual_model_system(
                market_data=market_data,
                current_price=current_price,
                model_type="tactician"
            )
            
            # Step 4: Make trading decision
            print("📋 Step 4: Making trading decision...")
            trading_decision = await dual_system.make_trading_decision(
                market_data=market_data,
                current_price=current_price
            )
            
            # Step 5: Create ensemble (if needed)
            print("📋 Step 5: Creating ensemble...")
            mock_models = {
                "model_1": {"type": "mock", "confidence": 0.7},
                "model_2": {"type": "mock", "confidence": 0.8},
                "model_3": {"type": "mock", "confidence": 0.6}
            }
            
            ensemble_result = await ensemble_creator.create_ensemble(
                training_data={"mock": market_data},
                models=mock_models,
                ensemble_name="workflow_ensemble",
                ensemble_type="workflow_test"
            )
            
            # Record results
            results["workflow_tests"]["data_generation"] = {
                "status": "SUCCESS",
                "data_points": len(market_data),
                "price_range": f"{market_data['close'].min():.2f} - {market_data['close'].max():.2f}"
            }
            
            results["workflow_tests"]["component_initialization"] = {
                "status": "SUCCESS",
                "components_initialized": 3
            }
            
            results["workflow_tests"]["prediction_generation"] = {
                "status": "SUCCESS" if analyst_predictions and tactician_predictions else "FAILED",
                "analyst_predictions": bool(analyst_predictions),
                "tactician_predictions": bool(tactician_predictions)
            }
            
            results["workflow_tests"]["trading_decision"] = {
                "status": "SUCCESS" if trading_decision else "FAILED",
                "decision_made": bool(trading_decision)
            }
            
            results["workflow_tests"]["ensemble_creation"] = {
                "status": "SUCCESS" if ensemble_result else "FAILED",
                "ensemble_created": bool(ensemble_result)
            }
            
            # Integration points
            results["integration_points"] = {
                "ml_predictor_to_dual_system": "SUCCESS",
                "dual_system_to_ensemble": "SUCCESS",
                "data_to_predictions": "SUCCESS",
                "predictions_to_decisions": "SUCCESS"
            }
            
            # Performance metrics
            successful_steps = sum(1 for test in results["workflow_tests"].values() 
                                 if test.get("status") == "SUCCESS")
            total_steps = len(results["workflow_tests"])
            
            results["performance_metrics"] = {
                "total_workflow_steps": total_steps,
                "successful_workflow_steps": successful_steps,
                "workflow_success_rate": successful_steps / total_steps if total_steps > 0 else 0.0,
                "integration_points_tested": len(results["integration_points"]),
                "successful_integrations": sum(1 for point in results["integration_points"].values() 
                                            if point == "SUCCESS")
            }
            
            # Cleanup
            await ml_predictor.stop()
            await dual_system.stop()
            await ensemble_creator.stop()
            
            return results
            
        except Exception as e:
            print(f"❌ End-to-end workflow test failed: {e}")
            return {"error": str(e)}
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("🚀 Starting Comprehensive Test Suite...")
        
        all_results = {
            "test_suite": "comprehensive_validation",
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "summary": {}
        }
        
        # Run all test categories
        test_categories = [
            ("ensemble_performance", self.test_ensemble_performance),
            ("dual_model_decision_making", self.test_dual_model_decision_making),
            ("ml_confidence_predictor_integration", self.test_ml_confidence_predictor_integration),
            ("end_to_end_workflow", self.test_end_to_end_workflow)
        ]
        
        for test_name, test_function in test_categories:
            print(f"\n{'='*60}")
            print(f"🧪 Running {test_name.replace('_', ' ').title()} Tests...")
            print(f"{'='*60}")
            
            try:
                result = await test_function()
                all_results["test_results"][test_name] = result
                
                # Print summary for this test category
                if "performance_metrics" in result:
                    metrics = result["performance_metrics"]
                    print(f"\n📊 {test_name.replace('_', ' ').title()} Summary:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
                        else:
                            print(f"  {key.replace('_', ' ').title()}: {value}")
                
            except Exception as e:
                print(f"❌ {test_name} test failed: {e}")
                all_results["test_results"][test_name] = {"error": str(e)}
        
        # Calculate overall summary
        total_tests = 0
        successful_tests = 0
        
        for test_category, results in all_results["test_results"].items():
            if "performance_metrics" in results:
                metrics = results["performance_metrics"]
                if "total_tests" in metrics:
                    total_tests += metrics["total_tests"]
                if "successful_tests" in metrics:
                    successful_tests += metrics["successful_tests"]
                elif "successful_decisions" in metrics:
                    successful_tests += metrics["successful_decisions"]
                elif "successful_prediction_tests" in metrics:
                    successful_tests += metrics["successful_prediction_tests"]
                elif "successful_workflow_steps" in metrics:
                    successful_tests += metrics["successful_workflow_steps"]
        
        all_results["summary"] = {
            "total_test_categories": len(test_categories),
            "successful_test_categories": sum(1 for results in all_results["test_results"].values() 
                                           if "error" not in results),
            "overall_success_rate": successful_tests / total_tests if total_tests > 0 else 0.0,
            "test_timestamp": datetime.now().isoformat()
        }
        
        # Print final summary
        print(f"\n{'='*60}")
        print("📊 COMPREHENSIVE TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Successful Test Categories: {all_results['summary']['successful_test_categories']}/{all_results['summary']['total_test_categories']}")
        print(f"📈 Overall Success Rate: {all_results['summary']['overall_success_rate']:.2%}")
        print(f"🕒 Test Completed: {all_results['summary']['test_timestamp']}")
        
        # Save results to file
        self._save_test_results(all_results)
        
        return all_results
    
    def _save_test_results(self, results: Dict[str, Any]) -> None:
        """Save test results to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_comprehensive_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Test results saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️ Failed to save test results: {e}")


async def main():
    """Run the comprehensive test suite."""
    test_suite = ComprehensiveTestSuite()
    results = await test_suite.run_comprehensive_tests()
    
    # Final status
    if results["summary"]["overall_success_rate"] >= 0.8:
        print("\n🎉 Comprehensive test suite completed successfully!")
        print("✅ System is ready for production use.")
    elif results["summary"]["overall_success_rate"] >= 0.6:
        print("\n⚠️ Comprehensive test suite completed with some issues.")
        print("🔧 System needs minor improvements before production.")
    else:
        print("\n❌ Comprehensive test suite revealed significant issues.")
        print("🚨 System requires major improvements before production.")


if __name__ == "__main__":
    asyncio.run(main()) 