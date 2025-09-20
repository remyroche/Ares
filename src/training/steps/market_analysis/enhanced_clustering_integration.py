"""
Enhanced Clustering Integration

This module integrates the enhanced outcome generator with the HMM clustering process,
ensuring clean, metrics-focused outcomes without raw data.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from src.utils.logger import system_logger
from .enhanced_outcome_generator import EnhancedOutcomeGenerator


class EnhancedClusteringIntegration:
    """
    Integrate enhanced outcome generation with clustering process.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild('EnhancedClusteringIntegration')
        self.outcome_generator = EnhancedOutcomeGenerator()
    
    async def generate_enhanced_clustering_outcome(
        self,
        clustering_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate enhanced clustering outcome with comprehensive metrics.
        
        Args:
            clustering_results: Results from clustering process
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            output_dir: Output directory (optional)
            
        Returns:
            Enhanced outcome dictionary
        """
        try:
            self.logger.info("🎯 Generating enhanced clustering outcome...")
            
            # Extract regime characteristics from clustering results
            regime_characteristics = self._extract_regime_characteristics(clustering_results)
            
            # Generate enhanced outcome
            enhanced_outcome = self.outcome_generator.generate_clustering_outcome(
                cluster_results=clustering_results,
                regime_characteristics=regime_characteristics,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                target_clusters=20,
                coverage_target=0.90
            )
            
            # Save outcome if output directory provided
            if output_dir:
                await self._save_enhanced_outcome(enhanced_outcome, output_dir, symbol, exchange, timeframe)
            
            self.logger.info("✅ Enhanced clustering outcome generated successfully")
            return enhanced_outcome
            
        except Exception as e:
            self.logger.error(f"❌ Error generating enhanced clustering outcome: {e}")
            raise
    
    def _extract_regime_characteristics(self, clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract regime characteristics from clustering results."""
        try:
            regime_characteristics = {}
            
            # Extract from clustering results structure
            clusters_dict = clustering_results.get("clusters_dict", {})
            
            for cluster_id, cluster_data in clusters_dict.items():
                # Extract feature information
                feature_means = cluster_data.get("feature_means", {})
                feature_stds = cluster_data.get("feature_stds", {})
                
                regime_characteristics[f"regime_{cluster_id}"] = {
                    "features": feature_means,
                    "feature_means": feature_means,
                    "feature_stds": feature_stds,
                    "sample_count": cluster_data.get("sample_count", 0),
                    "volatility": feature_stds.get(list(feature_stds.keys())[0], 0.01) if feature_stds else 0.01
                }
            
            return regime_characteristics
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting regime characteristics: {e}")
            return {}
    
    async def _save_enhanced_outcome(
        self,
        enhanced_outcome: Dict[str, Any],
        output_dir: str,
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> None:
        """Save enhanced outcome to file."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"market_analysis_hmm_clustering_enhanced_{timestamp}.json"
            file_path = output_path / filename
            
            # Save enhanced outcome
            self.outcome_generator.save_enhanced_outcome(enhanced_outcome, str(file_path))
            
            self.logger.info(f"✅ Enhanced outcome saved to: {file_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving enhanced outcome: {e}")
            raise
    
    def validate_clustering_quality(
        self,
        enhanced_outcome: Dict[str, Any],
        target_clusters: int = 20,
        coverage_target: float = 0.90
    ) -> Dict[str, Any]:
        """
        Validate clustering quality against targets.
        
        Args:
            enhanced_outcome: Enhanced outcome dictionary
            target_clusters: Target number of clusters
            coverage_target: Target coverage percentage
            
        Returns:
            Validation results
        """
        try:
            clustering_summary = enhanced_outcome.get("clustering_summary", {})
            comprehensive_metrics = enhanced_outcome.get("comprehensive_metrics", {})
            
            # Extract key metrics
            total_clusters = clustering_summary.get("total_clusters", 0)
            coverage_achieved = clustering_summary.get("coverage_achieved", False)
            quality_score = clustering_summary.get("quality_score", 0.0)
            
            coverage_metrics = comprehensive_metrics.get("coverage_metrics", {})
            top_20_coverage = coverage_metrics.get("top_20_coverage", 0.0)
            
            # Calculate validation scores
            cluster_count_score = 1.0 - abs(total_clusters - target_clusters) / target_clusters
            cluster_count_score = max(0.0, min(1.0, cluster_count_score))
            
            coverage_score = min(1.0, top_20_coverage / (coverage_target * 100))
            
            overall_validation_score = (cluster_count_score * 0.4 + coverage_score * 0.4 + quality_score * 0.2)
            
            validation_results = {
                "overall_score": overall_validation_score,
                "cluster_count_validation": {
                    "target": target_clusters,
                    "actual": total_clusters,
                    "score": cluster_count_score,
                    "acceptable": 15 <= total_clusters <= 25
                },
                "coverage_validation": {
                    "target": coverage_target * 100,
                    "actual": top_20_coverage,
                    "score": coverage_score,
                    "achieved": coverage_achieved
                },
                "quality_validation": {
                    "score": quality_score,
                    "acceptable": quality_score >= 0.7
                },
                "recommendations": enhanced_outcome.get("recommendations", []),
                "validation_status": "PASSED" if overall_validation_score >= 0.8 else "NEEDS_IMPROVEMENT"
            }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Error validating clustering quality: {e}")
            return {"validation_status": "ERROR", "error": str(e)}