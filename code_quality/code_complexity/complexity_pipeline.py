#!/usr/bin/env python3
"""
Code Complexity Analysis Pipeline
Combines PyExamine, Radon, and Xenon for comprehensive complexity analysis
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# Add the parent directory to the path to import from code_quality
sys.path.append(str(Path(__file__).parent.parent))

from analyzers.pyexamine_analyzer import PyExamineAnalyzer
from analyzers.radon_analyzer import RadonAnalyzer
from analyzers.xenon_analyzer import XenonAnalyzer
from utils.report_generator import ReportGenerator
from utils.file_utils import FileUtils
from config.complexity_config import ComplexityConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ComplexityMetrics:
    """Data class for storing complexity metrics"""
    file_path: str
    pyexamine_score: Optional[float] = None
    radon_cc: Optional[float] = None
    radon_mi: Optional[float] = None
    xenon_score: Optional[float] = None
    combined_score: Optional[float] = None
    analysis_timestamp: str = None
    
    def __post_init__(self):
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now().isoformat()


@dataclass
class DirectoryMetrics:
    """Data class for storing directory-level complexity metrics"""
    directory_path: str
    file_count: int
    total_files_analyzed: int
    average_complexity: float
    max_complexity: float
    min_complexity: float
    complexity_distribution: Dict[str, int]
    files_metrics: List[ComplexityMetrics]
    analysis_timestamp: str = None
    
    def __post_init__(self):
        if self.analysis_timestamp is None:
            self.analysis_timestamp = datetime.now().isoformat()


class ComplexityPipeline:
    """Main pipeline class for code complexity analysis"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the complexity analysis pipeline"""
        self.config = ComplexityConfig(config_path)
        self.pyexamine = PyExamineAnalyzer(self.config)
        self.radon = RadonAnalyzer(self.config)
        self.xenon = XenonAnalyzer(self.config)
        self.report_generator = ReportGenerator(self.config)
        self.file_utils = FileUtils()
        
        # Create output directories
        self._create_output_directories()
        
    def _create_output_directories(self):
        """Create necessary output directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.reports_dir, exist_ok=True)
        os.makedirs(self.config.logs_dir, exist_ok=True)
        
    def analyze_file(self, file_path: str) -> ComplexityMetrics:
        """Analyze a single Python file for complexity"""
        logger.info(f"Analyzing file: {file_path}")
        
        metrics = ComplexityMetrics(file_path=file_path)
        
        try:
            # Run PyExamine analysis
            if self.config.enable_pyexamine:
                metrics.pyexamine_score = self.pyexamine.analyze_file(file_path)
                
            # Run Radon analysis
            if self.config.enable_radon:
                radon_results = self.radon.analyze_file(file_path)
                metrics.radon_cc = radon_results.get('cyclomatic_complexity')
                metrics.radon_mi = radon_results.get('maintainability_index')
                
            # Run Xenon analysis
            if self.config.enable_xenon:
                metrics.xenon_score = self.xenon.analyze_file(file_path)
                
            # Calculate combined score
            metrics.combined_score = self._calculate_combined_score(metrics)
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            
        return metrics
        
    def analyze_directory(self, directory_path: str) -> DirectoryMetrics:
        """Analyze all Python files in a directory"""
        logger.info(f"Analyzing directory: {directory_path}")
        
        python_files = self.file_utils.get_python_files(directory_path)
        file_metrics = []
        
        for file_path in python_files:
            try:
                metrics = self.analyze_file(file_path)
                file_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error analyzing file {file_path}: {str(e)}")
                
        return self._aggregate_directory_metrics(directory_path, file_metrics)
        
    def _calculate_combined_score(self, metrics: ComplexityMetrics) -> float:
        """Calculate a combined complexity score from all tools"""
        scores = []
        
        if metrics.pyexamine_score is not None:
            scores.append(metrics.pyexamine_score)
            
        if metrics.radon_cc is not None:
            # Normalize Radon CC (higher is worse, so invert)
            normalized_cc = max(0, 10 - metrics.radon_cc) / 10
            scores.append(normalized_cc)
            
        if metrics.radon_mi is not None:
            # Normalize Radon MI (higher is better)
            normalized_mi = metrics.radon_mi / 100
            scores.append(normalized_mi)
            
        if metrics.xenon_score is not None:
            # Normalize Xenon score (lower is better, so invert)
            normalized_xenon = max(0, 10 - metrics.xenon_score) / 10
            scores.append(normalized_xenon)
            
        if not scores:
            return 0.0
            
        return sum(scores) / len(scores)
        
    def _aggregate_directory_metrics(self, directory_path: str, 
                                   file_metrics: List[ComplexityMetrics]) -> DirectoryMetrics:
        """Aggregate file metrics into directory-level metrics"""
        if not file_metrics:
            return DirectoryMetrics(
                directory_path=directory_path,
                file_count=0,
                total_files_analyzed=0,
                average_complexity=0.0,
                max_complexity=0.0,
                min_complexity=0.0,
                complexity_distribution={},
                files_metrics=[]
            )
            
        # Filter out files with no combined score
        valid_metrics = [m for m in file_metrics if m.combined_score is not None]
        
        if not valid_metrics:
            return DirectoryMetrics(
                directory_path=directory_path,
                file_count=len(file_metrics),
                total_files_analyzed=0,
                average_complexity=0.0,
                max_complexity=0.0,
                min_complexity=0.0,
                complexity_distribution={},
                files_metrics=file_metrics
            )
            
        scores = [m.combined_score for m in valid_metrics]
        
        # Calculate complexity distribution
        distribution = {
            'low': len([s for s in scores if s >= 0.7]),
            'medium': len([s for s in scores if 0.4 <= s < 0.7]),
            'high': len([s for s in scores if s < 0.4])
        }
        
        return DirectoryMetrics(
            directory_path=directory_path,
            file_count=len(file_metrics),
            total_files_analyzed=len(valid_metrics),
            average_complexity=sum(scores) / len(scores),
            max_complexity=max(scores),
            min_complexity=min(scores),
            complexity_distribution=distribution,
            files_metrics=file_metrics
        )
        
    def run_full_analysis(self, target_path: str) -> Dict[str, Any]:
        """Run full complexity analysis on target path"""
        logger.info(f"Starting full complexity analysis on: {target_path}")
        
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'target_path': target_path,
            'config': self.config.config,
            'file_analysis': {},
            'directory_analysis': {}
        }
        
        if os.path.isfile(target_path):
            # Single file analysis
            metrics = self.analyze_file(target_path)
            results['file_analysis'][target_path] = asdict(metrics)
            
        elif os.path.isdir(target_path):
            # Directory analysis
            dir_metrics = self.analyze_directory(target_path)
            results['directory_analysis'][target_path] = asdict(dir_metrics)
            
            # Also analyze individual files
            python_files = self.file_utils.get_python_files(target_path)
            for file_path in python_files:
                metrics = self.analyze_file(file_path)
                results['file_analysis'][file_path] = asdict(metrics)
                
        else:
            raise ValueError(f"Target path does not exist: {target_path}")
            
        # Generate reports
        self.report_generator.generate_reports(results)
        
        return results
        
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save analysis results to JSON file"""
        output_path = os.path.join(self.config.output_dir, output_file)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to: {output_path}")
        
    def load_results(self, input_file: str) -> Dict[str, Any]:
        """Load analysis results from JSON file"""
        input_path = os.path.join(self.config.output_dir, input_file)
        
        with open(input_path, 'r') as f:
            results = json.load(f)
            
        return results


def main():
    """Main entry point for the complexity analysis pipeline"""
    parser = argparse.ArgumentParser(description='Code Complexity Analysis Pipeline')
    parser.add_argument('target', help='Target file or directory to analyze')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--output', help='Output file name for results')
    parser.add_argument('--format', choices=['json', 'html', 'markdown'], 
                       default='json', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        # Initialize pipeline
        pipeline = ComplexityPipeline(args.config)
        
        # Run analysis
        results = pipeline.run_full_analysis(args.target)
        
        # Save results
        if args.output:
            pipeline.save_results(results, args.output)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f'complexity_analysis_{timestamp}.json'
            pipeline.save_results(results, output_file)
            
        logger.info("Complexity analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error running complexity analysis: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()