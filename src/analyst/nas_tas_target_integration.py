"""
NAS/TAS Integration with Trading Target Meta Labels

This module integrates Neural Architecture Search (NAS) and Tree Architecture Search (TAS)
with trading target meta labels to create enhanced meta labeling for ML training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from .trading_target_meta_labels import TradingTargetMetaLabeler, TradingTarget, TargetMetaLabel
from .meta_labeling_system import MetaLabelingSystem
from src.research.profit_labeling.ensemble_labeling_system import EnsembleLabelingSystem


@dataclass
class NAS_TAS_TargetMetaLabel:
    """Enhanced meta label combining NAS/TAS with trading targets."""
    target_type: TradingTarget
    signal_strength: float
    confidence: float
    probability: float
    time_horizon: str
    risk_level: str
    setup_quality: float
    
    # NAS-generated meta features
    nas_architecture_score: float
    nas_regime_accuracy: float
    nas_economic_significance: float
    nas_trading_viability: float
    nas_complexity_score: float
    nas_efficiency_score: float
    
    # TAS-generated meta features
    tas_tree_depth: float
    tas_branching_factor: float
    tas_decision_quality: float
    tas_feature_importance: float
    tas_tree_complexity: float
    
    # Traditional meta features
    traditional_analyst_labels: Dict[str, Any]
    traditional_tactician_labels: Dict[str, Any]
    ensemble_labels: Dict[str, Any]
    
    # Combined meta features
    combined_meta_score: float
    nas_tas_synergy_score: float
    overall_quality_score: float
    
    # Entry/Exit conditions
    entry_conditions: Dict[str, Any]
    exit_conditions: Dict[str, Any]
    
    # Metadata
    metadata: Dict[str, Any]
    timestamp: datetime


class NAS_TAS_TargetIntegrationSystem:
    """
    Integration system combining NAS/TAS with trading target meta labels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("NAS_TAS_TargetIntegration")
        
        # Initialize components
        self.target_meta_labeler = TradingTargetMetaLabeler(config.get("target_meta_labeling", {}))
        self.meta_labeling_system = MetaLabelingSystem(config.get("meta_labeling", {}))
        self.ensemble_system = EnsembleLabelingSystem(config.get("ensemble", {}))
        
        # NAS/TAS configuration
        self.nas_config = config.get("nas", {})
        self.tas_config = config.get("tas", {})
        
        self.logger.info("🚀 NAS/TAS Target Integration System initialized")
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="generate_enhanced_target_labels"
    )
    async def generate_enhanced_target_labels(
        self,
        price_data: pd.DataFrame,
        volume_data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None,
        additional_features: Optional[Dict[str, Any]] = None
    ) -> Dict[str, NAS_TAS_TargetMetaLabel]:
        """Generate enhanced target meta labels with NAS/TAS integration."""
        try:
            self.logger.info("🎯 Generating enhanced target meta labels with NAS/TAS integration")
            
            # Generate traditional target labels
            target_labels = await self.target_meta_labeler.generate_all_target_labels(
                price_data, volume_data, additional_features
            )
            
            # Generate traditional meta labels
            analyst_labels = await self.meta_labeling_system.generate_analyst_labels(price_data, volume_data)
            tactician_labels = await self.meta_labeling_system.generate_tactician_labels(price_data, volume_data)
            
            # Generate ensemble labels
            ensemble_result = self.ensemble_system.generate_ensemble_labels(price_data)
            
            # Generate NAS/TAS meta features
            nas_meta_features = await self._generate_nas_meta_features(price_data, regime_data)
            tas_meta_features = await self._generate_tas_meta_features(price_data, regime_data)
            
            # Combine all labels
            enhanced_labels = {}
            for target_name, target_label in target_labels.items():
                enhanced_label = self._create_enhanced_target_label(
                    target_label,
                    nas_meta_features,
                    tas_meta_features,
                    analyst_labels,
                    tactician_labels,
                    ensemble_result
                )
                enhanced_labels[target_name] = enhanced_label
            
            self.logger.info(f"✅ Generated {len(enhanced_labels)} enhanced target meta labels")
            return enhanced_labels
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced target labels: {e}")
            return {}
    
    async def _generate_nas_meta_features(
        self,
        price_data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Generate NAS-based meta features."""
        try:
            # Simulate NAS architecture search results
            # In practice, this would use your actual NAS systems
            
            # Architecture performance scores
            nas_architecture_score = np.random.uniform(0.6, 0.95)
            nas_regime_accuracy = np.random.uniform(0.7, 0.9)
            nas_economic_significance = np.random.uniform(0.5, 0.8)
            nas_trading_viability = np.random.uniform(0.6, 0.85)
            nas_complexity_score = np.random.uniform(0.3, 0.8)
            nas_efficiency_score = 1.0 - nas_complexity_score
            
            return {
                "architecture_score": nas_architecture_score,
                "regime_accuracy": nas_regime_accuracy,
                "economic_significance": nas_economic_significance,
                "trading_viability": nas_trading_viability,
                "complexity_score": nas_complexity_score,
                "efficiency_score": nas_efficiency_score
            }
            
        except Exception as e:
            self.logger.error(f"Error generating NAS meta features: {e}")
            return {
                "architecture_score": 0.5,
                "regime_accuracy": 0.5,
                "economic_significance": 0.5,
                "trading_viability": 0.5,
                "complexity_score": 0.5,
                "efficiency_score": 0.5
            }
    
    async def _generate_tas_meta_features(
        self,
        price_data: pd.DataFrame,
        regime_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Generate TAS-based meta features."""
        try:
            # Simulate TAS tree search results
            # In practice, this would use your actual TAS systems
            
            # Tree structure scores
            tas_tree_depth = np.random.uniform(3, 12)
            tas_branching_factor = np.random.uniform(2, 8)
            tas_decision_quality = np.random.uniform(0.6, 0.9)
            tas_feature_importance = np.random.uniform(0.5, 0.85)
            tas_tree_complexity = min(tas_tree_depth / 10, 1.0)
            
            return {
                "tree_depth": tas_tree_depth,
                "branching_factor": tas_branching_factor,
                "decision_quality": tas_decision_quality,
                "feature_importance": tas_feature_importance,
                "tree_complexity": tas_tree_complexity
            }
            
        except Exception as e:
            self.logger.error(f"Error generating TAS meta features: {e}")
            return {
                "tree_depth": 5.0,
                "branching_factor": 3.0,
                "decision_quality": 0.5,
                "feature_importance": 0.5,
                "tree_complexity": 0.5
            }
    
    def _create_enhanced_target_label(
        self,
        target_label: TargetMetaLabel,
        nas_meta_features: Dict[str, float],
        tas_meta_features: Dict[str, float],
        analyst_labels: Dict[str, Any],
        tactician_labels: Dict[str, Any],
        ensemble_result: Any
    ) -> NAS_TAS_TargetMetaLabel:
        """Create enhanced target label combining all components."""
        try:
            # Calculate combined meta score
            combined_meta_score = (
                target_label.signal_strength * 0.3 +
                target_label.confidence * 0.3 +
                nas_meta_features["architecture_score"] * 0.2 +
                tas_meta_features["decision_quality"] * 0.2
            )
            
            # Calculate NAS/TAS synergy score
            nas_tas_synergy_score = (
                nas_meta_features["regime_accuracy"] * 0.4 +
                tas_meta_features["feature_importance"] * 0.3 +
                nas_meta_features["trading_viability"] * 0.3
            )
            
            # Calculate overall quality score
            overall_quality_score = (
                target_label.setup_quality * 0.4 +
                combined_meta_score * 0.3 +
                nas_tas_synergy_score * 0.3
            )
            
            return NAS_TAS_TargetMetaLabel(
                target_type=target_label.target_type,
                signal_strength=target_label.signal_strength,
                confidence=target_label.confidence,
                probability=target_label.probability,
                time_horizon=target_label.time_horizon,
                risk_level=target_label.risk_level,
                setup_quality=target_label.setup_quality,
                
                # NAS features
                nas_architecture_score=nas_meta_features["architecture_score"],
                nas_regime_accuracy=nas_meta_features["regime_accuracy"],
                nas_economic_significance=nas_meta_features["economic_significance"],
                nas_trading_viability=nas_meta_features["trading_viability"],
                nas_complexity_score=nas_meta_features["complexity_score"],
                nas_efficiency_score=nas_meta_features["efficiency_score"],
                
                # TAS features
                tas_tree_depth=tas_meta_features["tree_depth"],
                tas_branching_factor=tas_meta_features["branching_factor"],
                tas_decision_quality=tas_meta_features["decision_quality"],
                tas_feature_importance=tas_meta_features["feature_importance"],
                tas_tree_complexity=tas_meta_features["tree_complexity"],
                
                # Traditional features
                traditional_analyst_labels=analyst_labels,
                traditional_tactician_labels=tactician_labels,
                ensemble_labels=ensemble_result.ensemble_labels if hasattr(ensemble_result, 'ensemble_labels') else {},
                
                # Combined features
                combined_meta_score=combined_meta_score,
                nas_tas_synergy_score=nas_tas_synergy_score,
                overall_quality_score=overall_quality_score,
                
                # Entry/Exit conditions
                entry_conditions=target_label.entry_conditions,
                exit_conditions=target_label.exit_conditions,
                
                # Metadata
                metadata={
                    **target_label.metadata,
                    "nas_features": nas_meta_features,
                    "tas_features": tas_meta_features
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced target label: {e}")
            return self._create_empty_enhanced_label(target_label.target_type)
    
    def _create_empty_enhanced_label(self, target_type: TradingTarget) -> NAS_TAS_TargetMetaLabel:
        """Create empty enhanced label for error cases."""
        return NAS_TAS_TargetMetaLabel(
            target_type=target_type,
            signal_strength=0.0,
            confidence=0.0,
            probability=0.0,
            time_horizon="short",
            risk_level="high",
            setup_quality=0.0,
            nas_architecture_score=0.0,
            nas_regime_accuracy=0.0,
            nas_economic_significance=0.0,
            nas_trading_viability=0.0,
            nas_complexity_score=0.0,
            nas_efficiency_score=0.0,
            tas_tree_depth=0.0,
            tas_branching_factor=0.0,
            tas_decision_quality=0.0,
            tas_feature_importance=0.0,
            tas_tree_complexity=0.0,
            traditional_analyst_labels={},
            traditional_tactician_labels={},
            ensemble_labels={},
            combined_meta_score=0.0,
            nas_tas_synergy_score=0.0,
            overall_quality_score=0.0,
            entry_conditions={},
            exit_conditions={},
            metadata={},
            timestamp=datetime.now()
        )
    
    def get_target_meta_label_summary(self, enhanced_labels: Dict[str, NAS_TAS_TargetMetaLabel]) -> Dict[str, Any]:
        """Get summary of target meta labels."""
        try:
            summary = {
                "total_targets": len(enhanced_labels),
                "target_types": list(enhanced_labels.keys()),
                "average_quality_score": np.mean([label.overall_quality_score for label in enhanced_labels.values()]),
                "average_confidence": np.mean([label.confidence for label in enhanced_labels.values()]),
                "average_signal_strength": np.mean([label.signal_strength for label in enhanced_labels.values()]),
                "nas_tas_synergy": np.mean([label.nas_tas_synergy_score for label in enhanced_labels.values()]),
                "top_targets": sorted(
                    [(name, label.overall_quality_score) for name, label in enhanced_labels.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating summary: {e}")
            return {}
    
    def save_enhanced_labels(self, enhanced_labels: Dict[str, NAS_TAS_TargetMetaLabel], output_path: str):
        """Save enhanced labels to file."""
        try:
            import json
            
            # Convert to serializable format
            serializable_labels = {}
            for name, label in enhanced_labels.items():
                serializable_labels[name] = {
                    "target_type": label.target_type.value,
                    "signal_strength": label.signal_strength,
                    "confidence": label.confidence,
                    "probability": label.probability,
                    "time_horizon": label.time_horizon,
                    "risk_level": label.risk_level,
                    "setup_quality": label.setup_quality,
                    "nas_architecture_score": label.nas_architecture_score,
                    "nas_regime_accuracy": label.nas_regime_accuracy,
                    "nas_economic_significance": label.nas_economic_significance,
                    "nas_trading_viability": label.nas_trading_viability,
                    "nas_complexity_score": label.nas_complexity_score,
                    "nas_efficiency_score": label.nas_efficiency_score,
                    "tas_tree_depth": label.tas_tree_depth,
                    "tas_branching_factor": label.tas_branching_factor,
                    "tas_decision_quality": label.tas_decision_quality,
                    "tas_feature_importance": label.tas_feature_importance,
                    "tas_tree_complexity": label.tas_tree_complexity,
                    "combined_meta_score": label.combined_meta_score,
                    "nas_tas_synergy_score": label.nas_tas_synergy_score,
                    "overall_quality_score": label.overall_quality_score,
                    "entry_conditions": label.entry_conditions,
                    "exit_conditions": label.exit_conditions,
                    "metadata": label.metadata,
                    "timestamp": label.timestamp.isoformat()
                }
            
            with open(output_path, 'w') as f:
                json.dump(serializable_labels, f, indent=2)
            
            self.logger.info(f"💾 Enhanced labels saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving enhanced labels: {e}")


# Convenience function
def create_nas_tas_target_integration_system(config: Optional[Dict[str, Any]] = None) -> NAS_TAS_TargetIntegrationSystem:
    """Create NAS/TAS target integration system."""
    if config is None:
        config = {
            "target_meta_labeling": {},
            "meta_labeling": {},
            "ensemble": {},
            "nas": {},
            "tas": {}
        }
    
    return NAS_TAS_TargetIntegrationSystem(config)