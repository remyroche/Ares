#!/usr/bin/env python3
"""
Core Validation Test Suite.
Tests core functionality without complex dependencies.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any, Optional

# Import only core components
from src.analyst.ml_confidence_predictor import MLConfidencePredictor, setup_ml_confidence_predictor
from src.training.dual_model_system import DualModelSystem, setup_dual_model_system
from src.training.ensemble_creator_simple import SimpleEnsembleCreator


class CoreValidationTestSuite:
    """Core validation test suite for essential functionality."""
    
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
                "max_ensemble_size": 10,
                "min_ensemble_size": 2
            }
        }
    
    def generate_mock_market_data(self, scenario: str, periods: int = 100) -> pd.DataFrame:
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
    
    async def test_ml_confidence_predictor_core(self) -> Dict[str, Any]:
        """Test core ML Confidence Predictor functionality."""
        print("🧪 Testing ML Confidence Predictor Core Functionality...")
        
        results = {
            "initialization": {},
            "prediction_tests": {},
            "verification_tests": {},
            "performance_metrics": {}
        }
        
        try:
            # Test 1: Initialization
            print("\n📋 Test 1: Initialization...")
            ml_predictor = await setup_ml_confidence_predictor(self.test_config)
            
            if ml_predictor:
                results["initialization"]["status"] = "SUCCESS"
                print("✅ ML Confidence Predictor initialized successfully")
            else:
                results["initialization"]["status"] = "FAILED"
                print("❌ ML Confidence Predictor initialization failed")
                return results
            
            # Test 2: Basic predictions
            print("\n📋 Test 2: Basic Predictions...")
            market_data = self.generate_mock_market_data("bullish", periods=50)
            current_price = market_data['close'].iloc[-1]
            
            # Test confidence table prediction
            confidence_predictions = await ml_predictor.predict_confidence_table(
                market_data, current_price
            )
            
            if confidence_predictions:
                results["prediction_tests"]["confidence_table"] = {
                    "status": "SUCCESS",
                    "has_predictions": True
                }
                print("✅ Confidence table predictions successful")
            else:
                results["prediction_tests"]["confidence_table"] = {
                    "status": "FAILED",
                    "error": "No predictions generated"
                }
                print("❌ Confidence table predictions failed")
            
            # Test 3: Dual model predictions
            print("\n📋 Test 3: Dual Model Predictions...")
            
            # Test analyst predictions
            analyst_predictions = await ml_predictor.predict_for_dual_model_system(
                market_data=market_data,
                current_price=current_price,
                model_type="analyst"
            )
            
            if analyst_predictions:
                results["prediction_tests"]["analyst_predictions"] = {
                    "status": "SUCCESS",
                    "has_strategic_decision": "strategic_decision" in analyst_predictions
                }
                print("✅ Analyst predictions successful")
            else:
                results["prediction_tests"]["analyst_predictions"] = {
                    "status": "FAILED",
                    "error": "Analyst predictions failed"
                }
                print("❌ Analyst predictions failed")
            
            # Test tactician predictions
            tactician_predictions = await ml_predictor.predict_for_dual_model_system(
                market_data=market_data,
                current_price=current_price,
                model_type="tactician"
            )
            
            if tactician_predictions:
                results["prediction_tests"]["tactician_predictions"] = {
                    "status": "SUCCESS",
                    "has_timing_decision": "timing_decision" in tactician_predictions
                }
                print("✅ Tactician predictions successful")
            else:
                results["prediction_tests"]["tactician_predictions"] = {
                    "status": "FAILED",
                    "error": "Tactician predictions failed"
                }
                print("❌ Tactician predictions failed")
            
            # Test 4: Confidence verification
            print("\n📋 Test 4: Confidence Verification...")
            verification_results = ml_predictor.verify_confidence_calculations(
                market_data=market_data,
                current_price=current_price
            )
            
            if verification_results:
                results["verification_tests"]["confidence_verification"] = {
                    "status": "SUCCESS",
                    "overall_verification": verification_results.get("overall_verification", "UNKNOWN"),
                    "has_anomaly_detection": "anomaly_detection" in verification_results
                }
                print("✅ Confidence verification successful")
            else:
                results["verification_tests"]["confidence_verification"] = {
                    "status": "FAILED",
                    "error": "Confidence verification failed"
                }
                print("❌ Confidence verification failed")
            
            # Performance metrics
            successful_tests = sum(1 for test in results["prediction_tests"].values() 
                                 if test.get("status") == "SUCCESS")
            total_tests = len(results["prediction_tests"])
            
            results["performance_metrics"] = {
                "total_prediction_tests": total_tests,
                "successful_prediction_tests": successful_tests,
                "prediction_success_rate": successful_tests / total_tests if total_tests > 0 else 0.0,
                "verification_success": results["verification_tests"]["confidence_verification"]["status"] == "SUCCESS"
            }
            
            await ml_predictor.stop()
            return results
            
        except Exception as e:
            print(f"❌ ML Confidence Predictor core test failed: {e}")
            return {"error": str(e)}
    
    async def test_dual_model_system_core(self) -> Dict[str, Any]:
        """Test core dual model system functionality."""
        print("🧪 Testing Dual Model System Core Functionality...")
        
        results = {
            "initialization": {},
            "decision_tests": {},
            "performance_metrics": {}
        }
        
        try:
            # Test 1: Initialization
            print("\n📋 Test 1: Initialization...")
            dual_system = await setup_dual_model_system(self.test_config)
            
            if dual_system:
                results["initialization"]["status"] = "SUCCESS"
                print("✅ Dual Model System initialized successfully")
            else:
                results["initialization"]["status"] = "FAILED"
                print("❌ Dual Model System initialization failed")
                return results
            
            # Test 2: System info
            print("\n📋 Test 2: System Info...")
            system_info = dual_system.get_system_info()
            
            if system_info:
                results["initialization"]["system_info"] = {
                    "status": "SUCCESS",
                    "has_config": "config" in system_info,
                    "has_status": "status" in system_info
                }
                print("✅ System info retrieved successfully")
            else:
                results["initialization"]["system_info"] = {
                    "status": "FAILED",
                    "error": "Failed to get system info"
                }
                print("❌ System info retrieval failed")
            
            # Test 3: Trading decisions
            print("\n📋 Test 3: Trading Decisions...")
            market_data = self.generate_mock_market_data("bullish", periods=50)
            current_price = market_data['close'].iloc[-1]
            
            # Test entry decision
            entry_decision = await dual_system.make_trading_decision(
                market_data=market_data,
                current_price=current_price
            )
            
            if entry_decision:
                results["decision_tests"]["entry_decision"] = {
                    "status": "SUCCESS",
                    "has_decision": "decision" in entry_decision,
                    "has_analyst_decision": "analyst_decision" in entry_decision,
                    "has_tactician_decision": "tactician_decision" in entry_decision
                }
                print("✅ Entry decision successful")
            else:
                results["decision_tests"]["entry_decision"] = {
                    "status": "FAILED",
                    "error": "Entry decision failed"
                }
                print("❌ Entry decision failed")
            
            # Test exit decision (simulate existing position)
            current_position = {
                "side": "LONG",
                "entry_price": current_price,
                "size": 1.0,
                "entry_time": datetime.now()
            }
            
            exit_decision = await dual_system.make_trading_decision(
                market_data=market_data,
                current_price=current_price,
                current_position=current_position
            )
            
            if exit_decision:
                results["decision_tests"]["exit_decision"] = {
                    "status": "SUCCESS",
                    "has_decision": "decision" in exit_decision,
                    "has_analyst_exit": "analyst_exit_decision" in exit_decision,
                    "has_tactician_exit": "tactician_exit_decision" in exit_decision
                }
                print("✅ Exit decision successful")
            else:
                results["decision_tests"]["exit_decision"] = {
                    "status": "FAILED",
                    "error": "Exit decision failed"
                }
                print("❌ Exit decision failed")
            
            # Performance metrics
            successful_decisions = sum(1 for test in results["decision_tests"].values() 
                                    if test.get("status") == "SUCCESS")
            total_decisions = len(results["decision_tests"])
            
            results["performance_metrics"] = {
                "total_decision_tests": total_decisions,
                "successful_decisions": successful_decisions,
                "decision_success_rate": successful_decisions / total_decisions if total_decisions > 0 else 0.0
            }
            
            await dual_system.stop()
            return results
            
        except Exception as e:
            print(f"❌ Dual model system core test failed: {e}")
            return {"error": str(e)}
    
    async def test_ensemble_creator_core(self) -> Dict[str, Any]:
        """Test core ensemble creator functionality."""
        print("🧪 Testing Ensemble Creator Core Functionality...")
        
        results = {
            "initialization": {},
            "ensemble_tests": {},
            "performance_metrics": {}
        }
        
        try:
            # Test 1: Initialization
            print("\n📋 Test 1: Initialization...")
            ensemble_creator = SimpleEnsembleCreator(self.test_config)
            await ensemble_creator.initialize()
            
            if ensemble_creator:
                results["initialization"]["status"] = "SUCCESS"
                print("✅ Ensemble Creator initialized successfully")
            else:
                results["initialization"]["status"] = "FAILED"
                print("❌ Ensemble Creator initialization failed")
                return results
            
            # Test 2: Get system info
            print("\n📋 Test 2: System Info...")
            system_info = ensemble_creator.get_ensemble_info()
            
            if system_info:
                results["initialization"]["system_info"] = {
                    "status": "SUCCESS",
                    "has_config": "config" in system_info,
                    "has_status": "status" in system_info
                }
                print("✅ Ensemble Creator system info retrieved successfully")
            else:
                results["initialization"]["system_info"] = {
                    "status": "FAILED",
                    "error": "Failed to get system info"
                }
                print("❌ Ensemble Creator system info retrieval failed")
            
            # Test 3: Create ensemble
            print("\n📋 Test 3: Ensemble Creation...")
            mock_models = {
                "model_1": {"type": "mock", "confidence": 0.7},
                "model_2": {"type": "mock", "confidence": 0.8},
                "model_3": {"type": "mock", "confidence": 0.6}
            }
            
            ensemble_result = await ensemble_creator.create_ensemble(
                training_data={"mock": pd.DataFrame()},
                models=mock_models,
                ensemble_name="test_ensemble",
                ensemble_type="core_test"
            )
            
            if ensemble_result:
                results["ensemble_tests"]["ensemble_creation"] = {
                    "status": "SUCCESS",
                    "has_ensemble_info": "ensemble_info" in ensemble_result,
                    "model_count": len(mock_models)
                }
                print("✅ Ensemble creation successful")
            else:
                results["ensemble_tests"]["ensemble_creation"] = {
                    "status": "FAILED",
                    "error": "Ensemble creation failed"
                }
                print("❌ Ensemble creation failed")
            
            # Test 4: Hierarchical ensemble
            print("\n📋 Test 4: Hierarchical Ensemble...")
            hierarchical_result = await ensemble_creator.create_hierarchical_ensemble(
                base_ensembles={"test_ensemble": {"models": mock_models}},
                ensemble_name="hierarchical_test"
            )
            
            if hierarchical_result:
                results["ensemble_tests"]["hierarchical_ensemble"] = {
                    "status": "SUCCESS",
                    "has_ensemble_info": "ensemble_info" in hierarchical_result
                }
                print("✅ Hierarchical ensemble creation successful")
            else:
                results["ensemble_tests"]["hierarchical_ensemble"] = {
                    "status": "FAILED",
                    "error": "Hierarchical ensemble creation failed"
                }
                print("❌ Hierarchical ensemble creation failed")
            
            # Performance metrics
            successful_ensembles = sum(1 for test in results["ensemble_tests"].values() 
                                    if test.get("status") == "SUCCESS")
            total_ensembles = len(results["ensemble_tests"])
            
            results["performance_metrics"] = {
                "total_ensemble_tests": total_ensembles,
                "successful_ensembles": successful_ensembles,
                "ensemble_success_rate": successful_ensembles / total_ensembles if total_ensembles > 0 else 0.0
            }
            
            await ensemble_creator.stop()
            return results
            
        except Exception as e:
            print(f"❌ Ensemble creator core test failed: {e}")
            return {"error": str(e)}
    
    async def run_core_tests(self) -> Dict[str, Any]:
        """Run all core tests."""
        print("🚀 Starting Core Validation Test Suite...")
        
        all_results = {
            "test_suite": "core_validation",
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "summary": {}
        }
        
        # Run all test categories
        test_categories = [
            ("ml_confidence_predictor_core", self.test_ml_confidence_predictor_core),
            ("dual_model_system_core", self.test_dual_model_system_core),
            ("ensemble_creator_core", self.test_ensemble_creator_core)
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
                if "total_prediction_tests" in metrics:
                    total_tests += metrics["total_prediction_tests"]
                    successful_tests += metrics["successful_prediction_tests"]
                elif "total_decision_tests" in metrics:
                    total_tests += metrics["total_decision_tests"]
                    successful_tests += metrics["successful_decisions"]
                elif "total_ensemble_tests" in metrics:
                    total_tests += metrics["total_ensemble_tests"]
                    successful_tests += metrics["successful_ensembles"]
        
        all_results["summary"] = {
            "total_test_categories": len(test_categories),
            "successful_test_categories": sum(1 for results in all_results["test_results"].values() 
                                           if "error" not in results),
            "overall_success_rate": successful_tests / total_tests if total_tests > 0 else 0.0,
            "test_timestamp": datetime.now().isoformat()
        }
        
        # Print final summary
        print(f"\n{'='*60}")
        print("📊 CORE VALIDATION TEST SUITE SUMMARY")
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
            filename = f"test_results_core_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Test results saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️ Failed to save test results: {e}")


async def main():
    """Run the core validation test suite."""
    test_suite = CoreValidationTestSuite()
    results = await test_suite.run_core_tests()
    
    # Final status
    if results["summary"]["overall_success_rate"] >= 0.8:
        print("\n🎉 Core validation test suite completed successfully!")
        print("✅ Core system is ready for production use.")
    elif results["summary"]["overall_success_rate"] >= 0.6:
        print("\n⚠️ Core validation test suite completed with some issues.")
        print("🔧 Core system needs minor improvements before production.")
    else:
        print("\n❌ Core validation test suite revealed significant issues.")
        print("🚨 Core system requires major improvements before production.")


if __name__ == "__main__":
    asyncio.run(main()) 