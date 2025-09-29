"""
Regime Alignment Manager for Hybrid NAS-TAS Regime Discovery.

Handles optimal alignment of regimes between NAS and TAS methods using
optimal transport and Hungarian algorithm approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import logging
from dataclasses import dataclass
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize console for tprint
console = Console()

def tprint(*args, **kwargs):
    """Enhanced print function with rich formatting."""
    console.print(*args, **kwargs)

logger = logging.getLogger(__name__)


@dataclass
class AlignmentConfig:
    """Configuration for regime alignment."""
    method: str = 'hungarian'  # 'hungarian', 'optimal_transport', 'greedy'
    min_overlap_threshold: float = 0.1
    max_regime_distance: float = 0.5
    enable_soft_alignment: bool = True
    alignment_confidence_threshold: float = 0.3


class RegimeAlignmentManager:
    """
    Manages alignment of regimes between NAS and TAS methods.
    
    Uses optimal transport and Hungarian algorithm to find the best
    correspondence between regimes from different methods.
    """
    
    def __init__(self, config: Optional[AlignmentConfig] = None):
        """Initialize the regime alignment manager."""
        self.config = config or AlignmentConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def align_regimes(self, nas_predictions: np.ndarray, tas_predictions: np.ndarray,
                     market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Align regimes between NAS and TAS predictions.
        
        Args:
            nas_predictions: NAS regime predictions
            tas_predictions: TAS regime predictions
            market_data: Optional market data for feature-based alignment
            
        Returns:
            Dictionary containing alignment results
        """
        try:
            tprint(Panel.fit(
                "[bold blue]🔄 Regime Alignment Manager[/bold blue]\n"
                f"NAS predictions: {len(nas_predictions)} samples\n"
                f"TAS predictions: {len(tas_predictions)} samples\n"
                f"Method: {self.config.method}",
                title="Alignment Start",
                border_style="blue"
            ))
            
            self.logger.info("🔄 Starting regime alignment between NAS and TAS")
            
            # Ensure same length
            min_length = min(len(nas_predictions), len(tas_predictions))
            nas_predictions = nas_predictions[:min_length]
            tas_predictions = tas_predictions[:min_length]
            
            tprint(f"[yellow]📏 Aligned to {min_length} samples[/yellow]")
            
            # Get unique regimes
            nas_regimes = np.unique(nas_predictions)
            tas_regimes = np.unique(tas_predictions)
            
            tprint(f"[cyan]📊 NAS regimes: {len(nas_regimes)} {list(nas_regimes)}[/cyan]")
            tprint(f"[cyan]📊 TAS regimes: {len(tas_regimes)} {list(tas_regimes)}[/cyan]")
            
            self.logger.info(f"📊 NAS regimes: {len(nas_regimes)}, TAS regimes: {len(tas_regimes)}")
            
            # Calculate alignment based on method
            if self.config.method == 'hungarian':
                alignment_result = self._hungarian_alignment(nas_predictions, tas_predictions, nas_regimes, tas_regimes)
            elif self.config.method == 'optimal_transport':
                alignment_result = self._optimal_transport_alignment(nas_predictions, tas_predictions, nas_regimes, tas_regimes)
            else:
                alignment_result = self._greedy_alignment(nas_predictions, tas_predictions, nas_regimes, tas_regimes)
            
            # Add market data features if available
            if market_data is not None:
                alignment_result['feature_based_alignment'] = self._feature_based_alignment(
                    nas_predictions, tas_predictions, market_data, nas_regimes, tas_regimes
                )
            
            # Calculate alignment quality metrics
            alignment_result['quality_metrics'] = self._calculate_alignment_quality(
                nas_predictions, tas_predictions, alignment_result
            )
            
            self.logger.info("✅ Regime alignment completed successfully")
            return alignment_result
            
        except Exception as e:
            self.logger.error(f"❌ Regime alignment failed: {e}")
            return {'error': str(e), 'alignment_matrix': {}, 'quality_metrics': {}}
    
    def _hungarian_alignment(self, nas_predictions: np.ndarray, tas_predictions: np.ndarray,
                           nas_regimes: np.ndarray, tas_regimes: np.ndarray) -> Dict[str, Any]:
        """Perform alignment using Hungarian algorithm."""
        try:
            # Create cost matrix based on overlap
            cost_matrix = np.zeros((len(nas_regimes), len(tas_regimes)))
            
            for i, nas_regime in enumerate(nas_regimes):
                for j, tas_regime in enumerate(tas_regimes):
                    nas_mask = nas_predictions == nas_regime
                    tas_mask = tas_predictions == tas_regime
                    
                    # Calculate overlap ratio
                    overlap = np.sum(nas_mask & tas_mask)
                    total = np.sum(nas_mask | tas_mask)
                    overlap_ratio = overlap / total if total > 0 else 0
                    
                    # Cost is inverse of overlap (lower cost = better alignment)
                    cost_matrix[i, j] = 1 - overlap_ratio
            
            # Solve assignment problem
            nas_indices, tas_indices = linear_sum_assignment(cost_matrix)
            
            # Create alignment mapping
            alignment_matrix = {}
            nas_to_tas = {}
            tas_to_nas = {}
            
            for nas_idx, tas_idx in zip(nas_indices, tas_indices):
                nas_regime = nas_regimes[nas_idx]
                tas_regime = tas_regimes[tas_idx]
                confidence = 1 - cost_matrix[nas_idx, tas_idx]
                
                if confidence >= self.config.alignment_confidence_threshold:
                    alignment_matrix[f'nas_{nas_regime}_tas_{tas_regime}'] = confidence
                    nas_to_tas[nas_regime] = tas_regime
                    tas_to_nas[tas_regime] = nas_regime
            
            return {
                'alignment_matrix': alignment_matrix,
                'nas_to_tas': nas_to_tas,
                'tas_to_nas': tas_to_nas,
                'method': 'hungarian',
                'total_alignments': len(nas_indices),
                'high_confidence_alignments': len([c for c in alignment_matrix.values() if c >= 0.7])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Hungarian alignment failed: {e}")
            return {'error': str(e), 'alignment_matrix': {}}
    
    def _optimal_transport_alignment(self, nas_predictions: np.ndarray, tas_predictions: np.ndarray,
                                   nas_regimes: np.ndarray, tas_regimes: np.ndarray) -> Dict[str, Any]:
        """Perform alignment using optimal transport."""
        try:
            # Create distance matrix between regime centroids
            nas_centroids = self._calculate_regime_centroids(nas_predictions, nas_regimes)
            tas_centroids = self._calculate_regime_centroids(tas_predictions, tas_regimes)
            
            # Calculate pairwise distances
            distance_matrix = cdist(nas_centroids, tas_centroids, metric='euclidean')
            
            # Normalize distances
            max_distance = np.max(distance_matrix)
            if max_distance > 0:
                distance_matrix = distance_matrix / max_distance
            
            # Solve assignment problem
            nas_indices, tas_indices = linear_sum_assignment(distance_matrix)
            
            # Create alignment mapping
            alignment_matrix = {}
            nas_to_tas = {}
            tas_to_nas = {}
            
            for nas_idx, tas_idx in zip(nas_indices, tas_indices):
                nas_regime = nas_regimes[nas_idx]
                tas_regime = tas_regimes[tas_idx]
                confidence = 1 - distance_matrix[nas_idx, tas_idx]
                
                if confidence >= self.config.alignment_confidence_threshold:
                    alignment_matrix[f'nas_{nas_regime}_tas_{tas_regime}'] = confidence
                    nas_to_tas[nas_regime] = tas_regime
                    tas_to_nas[tas_regime] = nas_regime
            
            return {
                'alignment_matrix': alignment_matrix,
                'nas_to_tas': nas_to_tas,
                'tas_to_nas': tas_to_nas,
                'method': 'optimal_transport',
                'total_alignments': len(nas_indices),
                'high_confidence_alignments': len([c for c in alignment_matrix.values() if c >= 0.7])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Optimal transport alignment failed: {e}")
            return {'error': str(e), 'alignment_matrix': {}}
    
    def _greedy_alignment(self, nas_predictions: np.ndarray, tas_predictions: np.ndarray,
                         nas_regimes: np.ndarray, tas_regimes: np.ndarray) -> Dict[str, Any]:
        """Perform greedy alignment based on overlap."""
        try:
            alignment_matrix = {}
            nas_to_tas = {}
            tas_to_nas = {}
            used_tas_regimes = set()
            
            # Sort NAS regimes by size (largest first)
            nas_regime_sizes = [(regime, np.sum(nas_predictions == regime)) for regime in nas_regimes]
            nas_regime_sizes.sort(key=lambda x: x[1], reverse=True)
            
            for nas_regime, _ in nas_regime_sizes:
                nas_mask = nas_predictions == nas_regime
                best_tas_regime = None
                best_overlap = 0
                
                for tas_regime in tas_regimes:
                    if tas_regime in used_tas_regimes:
                        continue
                    
                    tas_mask = tas_predictions == tas_regime
                    overlap = np.sum(nas_mask & tas_mask)
                    total = np.sum(nas_mask | tas_mask)
                    overlap_ratio = overlap / total if total > 0 else 0
                    
                    if overlap_ratio > best_overlap and overlap_ratio >= self.config.min_overlap_threshold:
                        best_overlap = overlap_ratio
                        best_tas_regime = tas_regime
                
                if best_tas_regime is not None:
                    alignment_matrix[f'nas_{nas_regime}_tas_{best_tas_regime}'] = best_overlap
                    nas_to_tas[nas_regime] = best_tas_regime
                    tas_to_nas[best_tas_regime] = nas_regime
                    used_tas_regimes.add(best_tas_regime)
            
            return {
                'alignment_matrix': alignment_matrix,
                'nas_to_tas': nas_to_tas,
                'tas_to_nas': tas_to_nas,
                'method': 'greedy',
                'total_alignments': len(alignment_matrix),
                'high_confidence_alignments': len([c for c in alignment_matrix.values() if c >= 0.7])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Greedy alignment failed: {e}")
            return {'error': str(e), 'alignment_matrix': {}}
    
    def _calculate_regime_centroids(self, predictions: np.ndarray, regimes: np.ndarray) -> np.ndarray:
        """Calculate centroids for each regime."""
        centroids = []
        for regime in regimes:
            regime_mask = predictions == regime
            regime_indices = np.where(regime_mask)[0]
            if len(regime_indices) > 0:
                # Use time-based centroid (mean index)
                centroid = np.mean(regime_indices)
                centroids.append([centroid])
            else:
                centroids.append([0])
        return np.array(centroids)
    
    def _feature_based_alignment(self, nas_predictions: np.ndarray, tas_predictions: np.ndarray,
                               market_data: pd.DataFrame, nas_regimes: np.ndarray, tas_regimes: np.ndarray) -> Dict[str, Any]:
        """Perform feature-based alignment using market data characteristics."""
        try:
            # Calculate regime characteristics
            nas_characteristics = self._calculate_regime_characteristics(nas_predictions, market_data, nas_regimes)
            tas_characteristics = self._calculate_regime_characteristics(tas_predictions, market_data, tas_regimes)
            
            # Calculate feature-based distances
            feature_alignment = {}
            for nas_regime in nas_regimes:
                nas_char = nas_characteristics.get(nas_regime, {})
                best_tas_regime = None
                best_similarity = 0
                
                for tas_regime in tas_regimes:
                    tas_char = tas_characteristics.get(tas_regime, {})
                    similarity = self._calculate_regime_similarity(nas_char, tas_char)
                    
                    if similarity > best_similarity and similarity >= 0.3:
                        best_similarity = similarity
                        best_tas_regime = tas_regime
                
                if best_tas_regime is not None:
                    feature_alignment[f'nas_{nas_regime}_tas_{best_tas_regime}'] = best_similarity
            
            return {
                'feature_alignment': feature_alignment,
                'nas_characteristics': nas_characteristics,
                'tas_characteristics': tas_characteristics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Feature-based alignment failed: {e}")
            return {'error': str(e)}
    
    def _calculate_regime_characteristics(self, predictions: np.ndarray, market_data: pd.DataFrame, regimes: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Calculate characteristics for each regime."""
        characteristics = {}
        
        for regime in regimes:
            regime_mask = predictions == regime
            regime_data = market_data[regime_mask]
            
            if len(regime_data) > 0:
                char = {
                    'mean_return': regime_data['close'].pct_change().mean() if 'close' in regime_data.columns else 0.0,
                    'volatility': regime_data['close'].pct_change().std() if 'close' in regime_data.columns else 0.0,
                    'mean_volume': regime_data['volume'].mean() if 'volume' in regime_data.columns else 0.0,
                    'volume_volatility': regime_data['volume'].std() if 'volume' in regime_data.columns else 0.0,
                    'duration': len(regime_data)
                }
                characteristics[regime] = char
            else:
                characteristics[regime] = {}
        
        return characteristics
    
    def _calculate_regime_similarity(self, nas_char: Dict[str, float], tas_char: Dict[str, float]) -> float:
        """Calculate similarity between regime characteristics."""
        if not nas_char or not tas_char:
            return 0.0
        
        # Calculate similarity for each characteristic
        similarities = []
        
        for key in ['mean_return', 'volatility', 'mean_volume', 'volume_volatility']:
            if key in nas_char and key in tas_char:
                nas_val = nas_char[key]
                tas_val = tas_char[key]
                
                if nas_val == 0 and tas_val == 0:
                    similarity = 1.0
                elif nas_val == 0 or tas_val == 0:
                    similarity = 0.0
                else:
                    # Use relative difference
                    diff = abs(nas_val - tas_val) / max(abs(nas_val), abs(tas_val))
                    similarity = 1 - diff
                
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_alignment_quality(self, nas_predictions: np.ndarray, tas_predictions: np.ndarray,
                                   alignment_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality metrics for the alignment."""
        try:
            alignment_matrix = alignment_result.get('alignment_matrix', {})
            
            if not alignment_matrix:
                return {'alignment_quality': 0.0, 'coverage': 0.0, 'confidence': 0.0}
            
            # Calculate average confidence
            confidences = list(alignment_matrix.values())
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Calculate coverage (percentage of regimes aligned)
            nas_regimes = np.unique(nas_predictions)
            tas_regimes = np.unique(tas_predictions)
            total_regimes = len(nas_regimes) + len(tas_regimes)
            aligned_regimes = len(alignment_matrix) * 2  # Each alignment covers 2 regimes
            coverage = aligned_regimes / total_regimes if total_regimes > 0 else 0.0
            
            # Calculate alignment quality score
            alignment_quality = (avg_confidence + coverage) / 2
            
            return {
                'alignment_quality': alignment_quality,
                'avg_confidence': avg_confidence,
                'coverage': coverage,
                'high_confidence_ratio': len([c for c in confidences if c >= 0.7]) / len(confidences) if confidences else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Alignment quality calculation failed: {e}")
            return {'alignment_quality': 0.0, 'coverage': 0.0, 'confidence': 0.0}
    
    def get_alignment_summary(self, alignment_result: Dict[str, Any]) -> str:
        """Get a summary of the alignment results."""
        try:
            quality_metrics = alignment_result.get('quality_metrics', {})
            alignment_matrix = alignment_result.get('alignment_matrix', {})
            
            summary = f"""
            🔄 Regime Alignment Summary:
            📊 Total alignments: {len(alignment_matrix)}
            🎯 Alignment quality: {quality_metrics.get('alignment_quality', 0.0):.3f}
            📈 Average confidence: {quality_metrics.get('avg_confidence', 0.0):.3f}
            📊 Coverage: {quality_metrics.get('coverage', 0.0):.3f}
            ⭐ High confidence ratio: {quality_metrics.get('high_confidence_ratio', 0.0):.3f}
            """
            
            return summary.strip()
            
        except Exception as e:
            return f"❌ Failed to generate alignment summary: {e}"