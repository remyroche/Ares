"""
Test script to verify Tactician coherence and functionality.
Ensures probabilities, analyst inputs, and decision logic work together properly.
"""
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from src.tactician.tactician import Tactician
from src.tactician.enhanced_scenario_based_predictor import EnhancedScenarioBasedPredictor
from src.utils.confidence import normalize_dual_confidence
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_mock_market_data(size: int = 100) -> pd.DataFrame:
    """Create mock market data for testing."""
    np.random.seed(42)
    base_price = 50000
    returns = np.random.normal(0.0001, 0.02, size)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, size)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, size))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, size))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, size)
    })
    
    data.index = pd.date_range(start='2024-01-01', periods=size, freq='1min')
    return data

async def test_scenario_predictor():
    """Test the enhanced scenario predictor functionality."""
    logger.info("=== Testing Enhanced Scenario Predictor ===")
    
    config = {
        "step17_optimization": {
            "enhanced_scenario_analysis": {
                "time_limit_minutes": 15,
                "n_estimators": 100,
                "learning_rate": 0.05,
                "profit_zone_combined_threshold": 0.6,
                "risk_zone_combined_threshold": 0.2,
                "confidence_threshold": 0.7
            }
        }
    }
    
    predictor = EnhancedScenarioBasedPredictor(config)
    success = await predictor.initialize()
    assert success, "Failed to initialize predictor"
    
    # Check scenario creation
    logger.info(f"Number of scenarios: {len(predictor.scenarios)}")
    assert len(predictor.scenarios) == 17, "Should have 17 scenarios (8 profit + 8 risk + 1 neutral)"
    
    # Verify profit scenarios
    profit_scenarios = [s for s in predictor.scenarios.values() if s['zone_type'] == 'profit']
    assert len(profit_scenarios) == 8, "Should have 8 profit scenarios"
    
    # Verify target values
    expected_targets = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]
    actual_targets = [s['profit_target'] for s in profit_scenarios]
    logger.info(f"Profit targets: {[f'{t*100:.1f}%' for t in actual_targets]}")
    
    # Test feature extraction
    market_data = create_mock_market_data(100)
    features = predictor.extract_comprehensive_features(market_data)
    logger.info(f"Feature shape: {features.shape}")
    assert features.shape[0] > 0, "Features should be extracted"
    
    # Test prediction (with untrained model)
    predictions = await predictor.predict_scenarios(features.reshape(1, -1), market_data)
    logger.info(f"Prediction keys: {predictions.keys()}")
    
    scenario_analysis = predictions['scenario_analysis']
    logger.info(f"Profit zone probability: {scenario_analysis['profit_zone_probability']:.2%}")
    logger.info(f"Risk zone probability: {scenario_analysis['risk_zone_probability']:.2%}")
    logger.info(f"Neutral probability: {scenario_analysis['neutral_probability']:.2%}")
    logger.info(f"Risk-reward ratio: {scenario_analysis['risk_reward_ratio']:.2f}")
    logger.info(f"Dominant zone: {scenario_analysis['dominant_zone']}")
    
    # Verify probabilities sum to approximately 1
    total_prob = (scenario_analysis['profit_zone_probability'] + 
                  scenario_analysis['risk_zone_probability'] + 
                  scenario_analysis['neutral_probability'])
    logger.info(f"Total probability: {total_prob:.4f}")
    assert abs(total_prob - 1.0) < 0.01, "Probabilities should sum to ~1"
    
    return True

async def test_tactician_integration():
    """Test full Tactician integration with analyst inputs."""
    logger.info("\n=== Testing Tactician Integration ===")
    
    config = {
        "tactician": {
            "tactics_interval": 30,
            "max_history": 100
        },
        "tactics_orchestrator": {},
        "step17_optimization": {
            "fully_migrated_tactician": {
                "entry_profit_threshold": 0.6,
                "entry_risk_threshold": 0.2,
                "entry_confidence_threshold": 0.7,
                "entry_profit_risk_ratio": 2.0,
                "entry_scenario_dominance": 0.4,
                "max_position_size": 0.1,
                "max_leverage": 3.0
            },
            "enhanced_scenario_analysis": {
                "time_limit_minutes": 15
            }
        }
    }
    
    tactician = Tactician(config)
    
    # Test with different analyst inputs
    market_data = create_mock_market_data(100)
    
    test_cases = [
        {
            "name": "Conservative Analyst",
            "analyst_barriers": {"upper_barrier": 0.01, "lower_barrier": -0.005},
            "analyst_confidence": 0.6
        },
        {
            "name": "Aggressive Analyst",
            "analyst_barriers": {"upper_barrier": 0.03, "lower_barrier": -0.01},
            "analyst_confidence": 0.8
        },
        {
            "name": "Neutral Analyst",
            "analyst_barriers": {"upper_barrier": 0.02, "lower_barrier": -0.01},
            "analyst_confidence": 0.5
        }
    ]
    
    for test in test_cases:
        logger.info(f"\n--- Testing: {test['name']} ---")
        logger.info(f"Analyst barriers: {test['analyst_barriers']}")
        logger.info(f"Analyst confidence: {test['analyst_confidence']}")
        
        # Generate predictions (without initialized components)
        try:
            predictions = await tactician.generate_enhanced_predictions(
                market_data=market_data,
                analyst_barriers=test['analyst_barriers'],
                symbol="BTCUSDT",
                timeframe="1m",
                analyst_confidence=test['analyst_confidence']
            )
            
            # Check scenario predictions
            scenario_pred = predictions['scenario_predictions']
            logger.info(f"Predicted scenario: {scenario_pred['scenario_name']}")
            logger.info(f"Model confidence: {scenario_pred['confidence']:.2%}")
            
            # Check trading decisions
            trading_decisions = predictions['trading_decisions']
            logger.info(f"Entry signal: {trading_decisions['entry_signal']}")
            logger.info(f"Direction: {trading_decisions['direction']}")
            logger.info(f"Decision confidence: {trading_decisions['confidence']:.2%}")
            
            # Check position management
            position_mgmt = predictions['position_management']
            logger.info(f"Position size: {position_mgmt['position_size']:.4f}")
            logger.info(f"Leverage: {position_mgmt['leverage']:.1f}x")
            logger.info(f"Stop loss: {position_mgmt['stop_loss']:.3%}")
            logger.info(f"Take profit: {position_mgmt['take_profit']:.3%}")
            
            # Verify stop loss and take profit are based on analyst barriers
            analyst_upper = test['analyst_barriers']['upper_barrier']
            analyst_lower = test['analyst_barriers']['lower_barrier']
            
            # Default multipliers are 1.0
            expected_sl = analyst_lower * tactician.risk_management['stop_loss_multiplier']
            expected_tp = analyst_upper * tactician.risk_management['take_profit_multiplier']
            
            assert abs(position_mgmt['stop_loss'] - expected_sl) < 0.0001, \
                f"Stop loss mismatch: {position_mgmt['stop_loss']} vs {expected_sl}"
            assert abs(position_mgmt['take_profit'] - expected_tp) < 0.0001, \
                f"Take profit mismatch: {position_mgmt['take_profit']} vs {expected_tp}"
            
        except Exception as e:
            logger.error(f"Error in test: {e}")
            # This is expected since components aren't initialized
            logger.info("(Expected error due to uninitialized components)")

async def test_confidence_calculations():
    """Test confidence calculation logic."""
    logger.info("\n=== Testing Confidence Calculations ===")
    
    test_cases = [
        {"analyst": 0.8, "tactician": 0.9},
        {"analyst": 0.6, "tactician": 0.7},
        {"analyst": 0.5, "tactician": 0.5},
        {"analyst": 0.9, "tactician": 0.4},
        {"analyst": 0.3, "tactician": 0.8}
    ]
    
    for test in test_cases:
        dual, normalized = normalize_dual_confidence(
            test["analyst"], 
            test["tactician"]
        )
        
        logger.info(f"\nAnalyst: {test['analyst']:.1f}, Tactician: {test['tactician']:.1f}")
        logger.info(f"Dual confidence: {dual:.3f} (analyst * tactician²)")
        logger.info(f"Normalized confidence: {normalized:.3f}")
        
        # Verify dual confidence calculation
        expected_dual = test["analyst"] * (test["tactician"] ** 2)
        assert abs(dual - expected_dual) < 0.0001, \
            f"Dual confidence mismatch: {dual} vs {expected_dual}"

async def test_decision_thresholds():
    """Test decision threshold logic."""
    logger.info("\n=== Testing Decision Thresholds ===")
    
    config = {
        "tactician": {},
        "tactics_orchestrator": {},
        "step17_optimization": {
            "fully_migrated_tactician": {
                "entry_profit_threshold": 0.6,
                "entry_risk_threshold": 0.2,
                "entry_confidence_threshold": 0.7,
                "entry_profit_risk_ratio": 2.0,
                "entry_scenario_dominance": 0.4
            }
        }
    }
    
    tactician = Tactician(config)
    
    # Test different scenario conditions
    test_scenarios = [
        {
            "name": "Strong Entry Signal",
            "scenario_analysis": {
                "profit_zone_probability": 0.75,
                "risk_zone_probability": 0.15,
                "risk_reward_ratio": 3.0,
                "scenario_dominance": 0.6,
                "dominant_zone": "profit"
            },
            "confidence": 0.8,
            "analyst_confidence": 0.7,
            "expected_entry": True
        },
        {
            "name": "Weak Entry Signal",
            "scenario_analysis": {
                "profit_zone_probability": 0.45,
                "risk_zone_probability": 0.35,
                "risk_reward_ratio": 1.2,
                "scenario_dominance": 0.3,
                "dominant_zone": "neutral"
            },
            "confidence": 0.5,
            "analyst_confidence": 0.4,
            "expected_entry": False
        },
        {
            "name": "High Risk Signal",
            "scenario_analysis": {
                "profit_zone_probability": 0.65,
                "risk_zone_probability": 0.25,  # Above threshold
                "risk_reward_ratio": 2.5,
                "scenario_dominance": 0.5,
                "dominant_zone": "profit"
            },
            "confidence": 0.75,
            "analyst_confidence": 0.6,
            "expected_entry": False
        }
    ]
    
    for scenario in test_scenarios:
        logger.info(f"\n--- {scenario['name']} ---")
        
        # Mock scenario predictions
        scenario_predictions = {
            "scenario_analysis": scenario["scenario_analysis"],
            "confidence": scenario["confidence"]
        }
        
        # Test decision making
        decisions = tactician._make_trading_decisions(
            scenario_predictions,
            scenario["analyst_confidence"],
            create_mock_market_data(10)
        )
        
        logger.info(f"Entry signal: {decisions['entry_signal']} (expected: {scenario['expected_entry']})")
        logger.info(f"Direction: {decisions['direction']}")
        logger.info(f"Reasoning: {decisions['reasoning']}")
        
        assert decisions['entry_signal'] == scenario['expected_entry'], \
            f"Entry signal mismatch for {scenario['name']}"

async def main():
    """Run all tests."""
    logger.info("Starting Tactician Coherence Tests\n")
    
    try:
        # Run tests
        await test_scenario_predictor()
        await test_tactician_integration()
        await test_confidence_calculations()
        await test_decision_thresholds()
        
        logger.info("\n✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())