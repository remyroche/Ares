"""
Feature Flow Tracker for Comprehensive Selection Reporting

This module tracks features through the entire selection pipeline,
providing detailed reports on which features are kept/discarded and why.
"""

import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error

@dataclass
class FeatureDecision:
    """Track a single feature's decision through the pipeline."""
    name: str
    initial_score: Optional[float] = None
    cmi_score: Optional[float] = None
    cmi_decision: Optional[str] = None
    cluster_id: Optional[int] = None
    cluster_decision: Optional[str] = None
    final_decision: Optional[str] = None
    reasoning: Optional[str] = None
    final_score: Optional[float] = None

class FeatureFlowTracker:
    """
    Tracks features through the entire selection pipeline.
    
    This class provides comprehensive reporting on feature decisions,
    showing exactly which features were kept/discarded and why.
    """
    
    def __init__(self):
        """Initialize feature flow tracker."""
        self.feature_decisions: Dict[str, FeatureDecision] = {}
        self.pipeline_stages: List[str] = []
        self.initial_features: List[str] = []
        self.final_features: List[str] = []
        
    def initialize_features(self, feature_names: List[str]) -> None:
        """
        Initialize tracking for all features.
        
        Args:
            feature_names: List of all feature names to track
        """
        self.initial_features = feature_names.copy()
        self.feature_decisions = {
            name: FeatureDecision(name=name) 
            for name in feature_names
        }
        self.pipeline_stages = ["initial"]
        
    def record_cmi_decisions(
        self, 
        cmi_scores: pd.Series, 
        threshold: float,
        selected_features: List[str]
    ) -> None:
        """
        Record CMI selection decisions.
        
        Args:
            cmi_scores: CMI scores for all features
            threshold: CMI threshold used
            selected_features: Features selected by CMI
        """
        self.pipeline_stages.append("cmi_selection")
        
        for feature_name, score in cmi_scores.items():
            if feature_name in self.feature_decisions:
                decision = self.feature_decisions[feature_name]
                decision.cmi_score = score
                
                if feature_name in selected_features:
                    decision.cmi_decision = "kept"
                    decision.reasoning = f"Above CMI threshold ({threshold:.6f} bits)"
                else:
                    decision.cmi_decision = "discarded"
                    decision.reasoning = f"Below CMI threshold ({threshold:.6f} bits)"
    
    def record_cluster_decisions(
        self, 
        cluster_labels: pd.Series,
        feature_stats: pd.DataFrame,
        selected_features: List[str]
    ) -> None:
        """
        Record De Prado clustering decisions.
        
        Args:
            cluster_labels: Cluster assignments for features
            feature_stats: Feature statistics from De Prado engine
            selected_features: Final selected features
        """
        self.pipeline_stages.append("deprado_clustering")
        
        for feature_name in self.feature_decisions.keys():
            if feature_name in cluster_labels.index:
                decision = self.feature_decisions[feature_name]
                decision.cluster_id = cluster_labels[feature_name]
                
                if feature_name in feature_stats.index:
                    decision.final_score = feature_stats.loc[feature_name, 'CompositeScore']
                
                if feature_name in selected_features:
                    decision.cluster_decision = "kept"
                    if decision.reasoning:
                        decision.reasoning += f", selected as cluster king"
                    else:
                        decision.reasoning = "Selected as cluster king"
                else:
                    decision.cluster_decision = "discarded"
                    # Find the king feature in the same cluster
                    cluster_id = cluster_labels[feature_name]
                    cluster_features = feature_stats[feature_stats['Cluster'] == cluster_id]
                    if len(cluster_features) > 0:
                        king_feature = cluster_features.loc[cluster_features['CompositeScore'].idxmax()]
                        if king_feature.name != feature_name:
                            score_diff = king_feature['CompositeScore'] - decision.final_score
                            if decision.reasoning:
                                decision.reasoning += f", lost to cluster king {king_feature.name} by {score_diff:.3f} points"
                            else:
                                decision.reasoning = f"Lost to cluster king {king_feature.name} by {score_diff:.3f} points"
    
    def finalize_decisions(self, final_features: List[str]) -> None:
        """
        Finalize all feature decisions.
        
        Args:
            final_features: List of final selected features
        """
        self.final_features = final_features.copy()
        self.pipeline_stages.append("final")
        
        for feature_name, decision in self.feature_decisions.items():
            if feature_name in final_features:
                decision.final_decision = "kept"
            else:
                decision.final_decision = "discarded"
    
    def print_comprehensive_flow_report(self) -> None:
        """Print comprehensive feature flow report."""
        if not self.feature_decisions:
            tprint_warning("⚠️ No feature decisions to report")
            return
        
        max_display = 50  # Limit output for large feature sets
        
        tprint_info("="*80)
        tprint_info("📈 COMPREHENSIVE FEATURE FLOW REPORT")
        tprint_info("="*80)
        
        # Pipeline overview
        tprint_info(f"🔍 Feature Selection Pipeline:")
        for i, stage in enumerate(self.pipeline_stages):
            if stage == "initial":
                count = len(self.initial_features)
                tprint_info(f"   {i+1}. {stage.upper()}: {count} features")
            elif stage == "cmi_selection":
                kept = len([d for d in self.feature_decisions.values() if d.cmi_decision == "kept"])
                discarded = len([d for d in self.feature_decisions.values() if d.cmi_decision == "discarded"])
                tprint_info(f"   {i+1}. {stage.upper()}: {kept} kept, {discarded} discarded")
            elif stage == "deprado_clustering":
                kept = len([d for d in self.feature_decisions.values() if d.cluster_decision == "kept"])
                discarded = len([d for d in self.feature_decisions.values() if d.cluster_decision == "discarded"])
                tprint_info(f"   {i+1}. {stage.upper()}: {kept} kept, {discarded} discarded")
            elif stage == "final":
                count = len(self.final_features)
                tprint_info(f"   {i+1}. {stage.upper()}: {count} features")
        
        # Overall reduction
        reduction = (1 - len(self.final_features) / len(self.initial_features)) * 100
        tprint_info(f"📊 Overall reduction: {reduction:.1f}% ({len(self.initial_features)} → {len(self.final_features)} features)")
        
        # Final selected features
        tprint_info(f"✅ FINAL SELECTED FEATURES ({len(self.final_features)}):")
        selected_decisions = [d for d in self.feature_decisions.values() if d.final_decision == "kept"]
        
        # Sort by final score if available
        selected_decisions.sort(key=lambda x: x.final_score or 0, reverse=True)
        
        for i, decision in enumerate(selected_decisions[:max_display]):
            score_info = f" (score: {decision.final_score:.3f})" if decision.final_score else ""
            tprint_info(f"   {i+1:2d}. {decision.name}{score_info}")
        
        if len(selected_decisions) > max_display:
            tprint_info(f"   ... and {len(selected_decisions) - max_display} more selected features")
        
        # Discarded features with reasoning
        discarded_decisions = [d for d in self.feature_decisions.values() if d.final_decision == "discarded"]
        
        if discarded_decisions:
            tprint_info(f"❌ DISCARDED FEATURES ({len(discarded_decisions)}):")
            
            # Group by discard reason
            discard_reasons = {}
            for decision in discarded_decisions:
                reason = decision.reasoning or "Unknown reason"
                if reason not in discard_reasons:
                    discard_reasons[reason] = []
                discard_reasons[reason].append(decision)
            
            for reason, features in discard_reasons.items():
                tprint_info(f"   📋 {reason.upper()} ({len(features)} features):")
                for decision in features[:max_display//2]:  # Limit per reason
                    score_info = f" (score: {decision.final_score:.3f})" if decision.final_score else ""
                    tprint_info(f"      ❌ {decision.name}{score_info}")
                
                if len(features) > max_display//2:
                    tprint_info(f"      ... and {len(features) - max_display//2} more features")
        
        # Stage-by-stage breakdown
        tprint_info(f"📊 STAGE-BY-STAGE BREAKDOWN:")
        
        # CMI stage
        cmi_kept = [d for d in self.feature_decisions.values() if d.cmi_decision == "kept"]
        cmi_discarded = [d for d in self.feature_decisions.values() if d.cmi_decision == "discarded"]
        
        if cmi_discarded:
            tprint_info(f"   🔍 CMI SELECTION:")
            tprint_info(f"      ✅ Kept: {len(cmi_kept)} features")
            tprint_info(f"      ❌ Discarded: {len(cmi_discarded)} features (low information)")
            
            # Show worst CMI performers
            worst_cmi = sorted(cmi_discarded, key=lambda x: x.cmi_score or 0)[:5]
            tprint_info(f"      🗑️  Worst CMI performers:")
            for decision in worst_cmi:
                score_info = f" ({decision.cmi_score:.6f} bits)" if decision.cmi_score else ""
                tprint_info(f"         {decision.name}{score_info}")
        
        # De Prado stage
        deprado_kept = [d for d in self.feature_decisions.values() if d.cluster_decision == "kept"]
        deprado_discarded = [d for d in self.feature_decisions.values() if d.cluster_decision == "discarded"]
        
        if deprado_discarded:
            tprint_info(f"   👑 DE PRADO CLUSTERING:")
            tprint_info(f"      ✅ Kept: {len(deprado_kept)} features (cluster kings)")
            tprint_info(f"      ❌ Discarded: {len(deprado_discarded)} features (redundancy)")
            
            # Show features that lost to kings by smallest margin
            close_losses = [d for d in deprado_discarded if d.final_score is not None]
            close_losses.sort(key=lambda x: x.final_score or 0, reverse=True)
            
            if close_losses:
                tprint_info(f"      🥈 Closest to king (highest scores among discarded):")
                for decision in close_losses[:5]:
                    score_info = f" (score: {decision.final_score:.3f})" if decision.final_score else ""
                    tprint_info(f"         {decision.name}{score_info}")
        
        # Quality comparison
        if selected_decisions and discarded_decisions:
            selected_scores = [d.final_score for d in selected_decisions if d.final_score is not None]
            discarded_scores = [d.final_score for d in discarded_decisions if d.final_score is not None]
            
            if selected_scores and discarded_scores:
                avg_selected = sum(selected_scores) / len(selected_scores)
                avg_discarded = sum(discarded_scores) / len(discarded_scores)
                
                tprint_info(f"📊 QUALITY COMPARISON:")
                tprint_info(f"   📈 Selected features avg score: {avg_selected:.3f}")
                tprint_info(f"   📉 Discarded features avg score: {avg_discarded:.3f}")
                
                if avg_discarded > 0:
                    improvement = ((avg_selected - avg_discarded) / avg_discarded * 100)
                    tprint_info(f"   🎯 Quality improvement: {improvement:.1f}% higher score in selected features")
        
        tprint_info("="*80)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the feature flow.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.feature_decisions:
            return {}
        
        selected = [d for d in self.feature_decisions.values() if d.final_decision == "kept"]
        discarded = [d for d in self.feature_decisions.values() if d.final_decision == "discarded"]
        
        return {
            "initial_features": len(self.initial_features),
            "final_features": len(self.final_features),
            "reduction_percentage": (1 - len(self.final_features) / len(self.initial_features)) * 100,
            "selected_features": len(selected),
            "discarded_features": len(discarded),
            "pipeline_stages": self.pipeline_stages,
            "cmi_kept": len([d for d in self.feature_decisions.values() if d.cmi_decision == "kept"]),
            "cmi_discarded": len([d for d in self.feature_decisions.values() if d.cmi_decision == "discarded"]),
            "deprado_kept": len([d for d in self.feature_decisions.values() if d.cluster_decision == "kept"]),
            "deprado_discarded": len([d for d in self.feature_decisions.values() if d.cluster_decision == "discarded"])
        }
