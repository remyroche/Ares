"""
Conservative Ensemble Pruning

This module implements conservative ensemble pruning that maintains diversity
while removing underperforming specialists.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

logger = system_logger.getChild('ConservativeEnsemblePruner')


@dataclass
class PruningConfig:
    """Configuration for conservative ensemble pruning"""
    min_ensemble_size: int = 8
    max_ensemble_size: int = 12
    diversity_threshold: float = 0.1  # Minimum diversity constraint
    performance_weight: float = 0.4
    orthogonality_weight: float = 0.3
    stability_weight: float = 0.2
    speed_weight: float = 0.1


class ConservativeEnsemblePruner:
    """Conservative ensemble pruning that maintains diversity"""
    
    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()
        self.pruning_history = {}
    
    def prune_ensemble_conservative(self, specialists: Dict[str, Any], 
                                  performance_metrics: Dict[str, Dict],
                                  diversity_matrix: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Conservative pruning that preserves ensemble diversity"""
        
        tprint_info("✂️ Starting conservative ensemble pruning")
        initial_size = len(specialists)
        
        # Calculate comprehensive scores
        specialist_scores = self._calculate_comprehensive_scores(
            specialists, performance_metrics, diversity_matrix
        )
        
        # Conservative pruning strategy
        if initial_size <= self.config.min_ensemble_size:
            tprint_info(f"  Ensemble size ({initial_size}) <= minimum ({self.config.min_ensemble_size}), no pruning needed")
            return specialists
        
        # Sort by score but keep diversity
        ranked_specialists = sorted(
            specialist_scores.items(), 
            key=lambda x: x[1]['combined_score'], 
            reverse=True
        )
        
        # Select top performers with diversity constraint
        selected_specialists = self._select_with_diversity_constraint(
            ranked_specialists, diversity_matrix
        )
        
        # Create pruned ensemble
        pruned_ensemble = {name: specialists[name] for name in selected_specialists}
        
        final_size = len(pruned_ensemble)
        reduction = (initial_size - final_size) / initial_size * 100
        
        tprint_success(f"  Pruned {initial_size} → {final_size} specialists ({reduction:.1f}% reduction)")
        
        # Log pruning details
        self._log_pruning_details(specialist_scores, selected_specialists)
        
        return pruned_ensemble
    
    def _calculate_comprehensive_scores(self, specialists: Dict[str, Any], 
                                      performance_metrics: Dict[str, Dict],
                                      diversity_matrix: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate comprehensive scores for each specialist"""
        
        scores = {}
        for specialist_name in specialists.keys():
            metrics = performance_metrics.get(specialist_name, {})
            
            # Performance score (40% weight)
            auc_score = metrics.get('auc', metrics.get('individual_auc', 0.5))
            performance_score = self._normalize_score(auc_score, 0.5, 0.8)
            
            # Orthogonality score (30% weight)
            orthogonality = metrics.get('orthogonality_score', 0.0)
            orthogonality_score = self._normalize_score(orthogonality, 0.0, 0.5)
            
            # Stability score (20% weight)
            stability = metrics.get('stability', 0.5)
            stability_score = self._normalize_score(stability, 0.3, 0.8)
            
            # Speed score (10% weight)
            speed = metrics.get('training_speed', 1.0)
            speed_score = self._normalize_score(1.0/speed, 0.5, 2.0)
            
            # Combined score
            combined_score = (
                self.config.performance_weight * performance_score +
                self.config.orthogonality_weight * orthogonality_score +
                self.config.stability_weight * stability_score +
                self.config.speed_weight * speed_score
            )
            
            scores[specialist_name] = {
                'combined_score': combined_score,
                'performance_score': performance_score,
                'orthogonality_score': orthogonality_score,
                'stability_score': stability_score,
                'speed_score': speed_score,
                'raw_metrics': metrics
            }
        
        return scores
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize score to [0, 1] range"""
        if max_val <= min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def _select_with_diversity_constraint(self, ranked_specialists: List[Tuple], 
                                        diversity_matrix: Optional[pd.DataFrame] = None) -> List[str]:
        """Select specialists with diversity constraint"""
        
        selected = []
        remaining = [name for name, _ in ranked_specialists]
        
        # Always keep top performer
        if remaining:
            selected.append(remaining[0])
            remaining.remove(remaining[0])
            tprint_info(f"  ✓ Kept top performer: {selected[0]}")
        
        # Add others with diversity constraint
        while remaining and len(selected) < self.config.max_ensemble_size:
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # Check diversity constraint
                if diversity_matrix is not None:
                    min_diversity = self._calculate_min_diversity(candidate, selected, diversity_matrix)
                    if min_diversity < self.config.diversity_threshold:
                        continue  # Skip if not diverse enough
                
                # Get candidate score
                candidate_score = next(score for name, score in ranked_specialists if name == candidate)
                
                if candidate_score['combined_score'] > best_score:
                    best_score = candidate_score['combined_score']
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                tprint_info(f"  ✓ Added {best_candidate} (score: {best_score:.3f})")
            else:
                # No more candidates meet diversity constraint, add best remaining
                if remaining:
                    selected.append(remaining[0])
                    remaining.remove(remaining[0])
                    tprint_info(f"  ✓ Added {selected[-1]} (diversity constraint relaxed)")
                else:
                    break
        
        # Ensure minimum ensemble size
        while len(selected) < self.config.min_ensemble_size and remaining:
            selected.append(remaining[0])
            remaining.remove(remaining[0])
            tprint_info(f"  ✓ Added {selected[-1]} (minimum size constraint)")
        
        return selected
    
    def _calculate_min_diversity(self, candidate: str, selected: List[str], 
                                diversity_matrix: pd.DataFrame) -> float:
        """Calculate minimum diversity between candidate and selected specialists"""
        
        if not selected or candidate not in diversity_matrix.index:
            return 1.0
        
        min_diversity = 1.0
        for selected_specialist in selected:
            if selected_specialist in diversity_matrix.columns:
                diversity = diversity_matrix.loc[candidate, selected_specialist]
                min_diversity = min(min_diversity, abs(diversity))
        
        return min_diversity
    
    def _log_pruning_details(self, specialist_scores: Dict[str, Dict], 
                           selected_specialists: List[str]) -> None:
        """Log detailed pruning information"""
        
        tprint_info("  Pruning Details:")
        
        # Show selected specialists with scores
        tprint_info("    Selected Specialists:")
        for specialist in selected_specialists:
            score_info = specialist_scores[specialist]
            tprint_info(f"      {specialist}: {score_info['combined_score']:.3f} "
                       f"(perf: {score_info['performance_score']:.3f}, "
                       f"orth: {score_info['orthogonality_score']:.3f})")
        
        # Show removed specialists
        removed = [s for s in specialist_scores.keys() if s not in selected_specialists]
        if removed:
            tprint_info("    Removed Specialists:")
            for specialist in removed:
                score_info = specialist_scores[specialist]
                tprint_info(f"      {specialist}: {score_info['combined_score']:.3f}")
    
    def calculate_diversity_matrix(self, specialist_predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Calculate diversity matrix from specialist predictions"""
        
        specialists = list(specialist_predictions.keys())
        diversity_matrix = pd.DataFrame(index=specialists, columns=specialists, dtype=float)
        
        for i, spec1 in enumerate(specialists):
            for j, spec2 in enumerate(specialists):
                if i == j:
                    diversity_matrix.loc[spec1, spec2] = 1.0
                else:
                    # Calculate correlation as diversity measure
                    pred1 = specialist_predictions[spec1]
                    pred2 = specialist_predictions[spec2]
                    
                    if len(pred1) > 1 and len(pred2) > 1:
                        correlation = np.corrcoef(pred1, pred2)[0, 1]
                        # Convert correlation to diversity (lower correlation = higher diversity)
                        diversity = 1.0 - abs(correlation)
                        diversity_matrix.loc[spec1, spec2] = diversity
                    else:
                        diversity_matrix.loc[spec1, spec2] = 0.5
        
        return diversity_matrix
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """Get statistics about pruning operations"""
        
        if not self.pruning_history:
            return {'message': 'No pruning history available'}
        
        total_pruned = sum(stats['removed_count'] for stats in self.pruning_history.values())
        avg_reduction = np.mean([stats['reduction_percentage'] for stats in self.pruning_history.values()])
        
        return {
            'total_pruning_operations': len(self.pruning_history),
            'total_specialists_removed': total_pruned,
            'average_reduction_percentage': avg_reduction,
            'recent_operations': list(self.pruning_history.values())[-5:]  # Last 5 operations
        }


def create_conservative_pruner(config: Optional[PruningConfig] = None) -> ConservativeEnsemblePruner:
    """Factory function to create conservative ensemble pruner"""
    return ConservativeEnsemblePruner(config)
