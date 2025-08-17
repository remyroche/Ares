#!/usr/bin/env python3
"""Test script for the new state naming function."""

import sys
import os

sys.path.append("/Users/remyroche/Documents/Ares")

from src.training.steps.step1_7_hmm_regime_discovery import _name_states

# Test data - sample feature medians
test_medians = {
    0: {
        "momentum_strength": 0.8,
        "rsi": 75,
        "bb_position": 0.9,
        "momentum_5": 0.7,
        "momentum_10": 0.6,
        "momentum_20": 0.5,
    },
    1: {
        "momentum_strength": -0.7,
        "rsi": 25,
        "bb_position": -0.8,
        "momentum_5": -0.6,
        "momentum_10": -0.5,
        "momentum_20": -0.4,
    },
    2: {
        "momentum_strength": 0.2,
        "rsi": 45,
        "bb_position": 0.1,
        "momentum_5": 0.1,
        "momentum_10": 0.0,
        "momentum_20": -0.1,
    },
    3: {
        "momentum_strength": 0.4,
        "rsi": 60,
        "bb_position": 0.3,
        "momentum_5": 0.4,
        "momentum_10": 0.3,
        "momentum_20": 0.2,
    },
}

# Test volatility states
volatility_medians = {
    0: {
        "1m_price_volatility": 0.9,
        "parkinson_volatility": 0.8,
        "garman_klass_volatility": 0.7,
        "volume_volatility": 1.2,
        "adaptive_atr": 0.8,
        "ewma_volatility": 0.9,
    },
    1: {
        "1m_price_volatility": 0.3,
        "parkinson_volatility": 0.2,
        "garman_klass_volatility": 0.3,
        "volume_volatility": 0.4,
        "adaptive_atr": 0.3,
        "ewma_volatility": 0.2,
    },
    2: {
        "1m_price_volatility": 0.6,
        "parkinson_volatility": 0.7,
        "garman_klass_volatility": 0.6,
        "volume_volatility": 0.5,
        "adaptive_atr": 0.6,
        "ewma_volatility": 0.7,
    },
}

print("🧪 Testing New State Naming Function")
print("=" * 50)

# Test momentum states
print("📈 MOMENTUM STATES:")
momentum_names = _name_states("momentum", test_medians)
for state_id, name in momentum_names.items():
    print(f"  State {state_id}: {name}")

print()

# Test volatility states
print("📊 VOLATILITY STATES:")
volatility_names = _name_states("volatility", volatility_medians)
for state_id, name in volatility_names.items():
    print(f"  State {state_id}: {name}")

print()
print("✅ Test completed!")
