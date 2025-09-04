"""
Xenon Analyzer for Code Complexity Analysis
"""

import os
import subprocess
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class XenonAnalyzer:
    """Analyzer for Xenon complexity metrics"""
    
    def __init__(self, config):
        """Initialize Xenon analyzer"""
        self.config = config
        self.tool_name = "xenon"
        
    def is_available(self) -> bool:
        """Check if Xenon is available"""
        try:
            result = subprocess.run(['xenon', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
            
    def analyze_file(self, file_path: str) -> Optional[float]:
        """Analyze a single file with Xenon"""
        if not self.is_available():
            logger.warning("Xenon is not available")
            return None
            
        try:
            # Run Xenon analysis
            cmd = ['xenon', '--json', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"Xenon failed for {file_path}: {result.stderr}")
                return None
                
            # Parse JSON output
            data = json.loads(result.stdout)
            
            # Extract complexity score
            if 'complexity' in data:
                return float(data['complexity'])
            elif 'score' in data:
                return float(data['score'])
            elif 'average' in data:
                return float(data['average'])
            else:
                # Try to extract from nested structure
                for key, value in data.items():
                    if isinstance(value, dict):
                        if 'complexity' in value:
                            return float(value['complexity'])
                        elif 'score' in value:
                            return float(value['score'])
                            
            logger.warning(f"Could not extract complexity score from Xenon output for {file_path}")
            return None
            
        except subprocess.TimeoutExpired:
            logger.error(f"Xenon analysis timed out for {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Xenon JSON output for {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error running Xenon on {file_path}: {e}")
            return None
            
    def analyze_directory(self, directory_path: str) -> Dict[str, float]:
        """Analyze all Python files in a directory"""
        results = {}
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    score = self.analyze_file(file_path)
                    if score is not None:
                        results[file_path] = score
                        
        return results
        
    def get_detailed_analysis(self, file_path: str) -> Dict[str, Any]:
        """Get detailed analysis results from Xenon"""
        if not self.is_available():
            return {}
            
        try:
            cmd = ['xenon', '--json', '--show-metrics', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"Xenon detailed analysis failed for {file_path}: {result.stderr}")
                return {}
                
            return json.loads(result.stdout)
            
        except Exception as e:
            logger.error(f"Error getting detailed Xenon analysis for {file_path}: {e}")
            return {}
            
    def get_function_complexity(self, file_path: str) -> List[Dict[str, Any]]:
        """Get complexity metrics for individual functions"""
        if not self.is_available():
            return []
            
        try:
            cmd = ['xenon', '--json', '--show-functions', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return []
                
            data = json.loads(result.stdout)
            
            # Extract function-level complexity
            functions = []
            if isinstance(data, list):
                for item in data:
                    if 'functions' in item:
                        functions.extend(item['functions'])
                    elif 'name' in item and 'complexity' in item:
                        functions.append(item)
                        
            return functions
            
        except Exception as e:
            logger.error(f"Error getting function complexity for {file_path}: {e}")
            return []
            
    def get_module_complexity(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get module-level complexity metrics"""
        if not self.is_available():
            return None
            
        try:
            cmd = ['xenon', '--json', '--show-modules', file_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return None
                
            data = json.loads(result.stdout)
            
            if isinstance(data, list) and data:
                return data[0]
            elif isinstance(data, dict):
                return data
                
        except Exception as e:
            logger.error(f"Error getting module complexity for {file_path}: {e}")
            
        return None
        
    def check_complexity_threshold(self, file_path: str, threshold: float = 10.0) -> Dict[str, Any]:
        """Check if file complexity exceeds threshold"""
        score = self.analyze_file(file_path)
        
        if score is None:
            return {
                'file': file_path,
                'score': None,
                'threshold': threshold,
                'exceeds_threshold': False,
                'status': 'analysis_failed'
            }
            
        return {
            'file': file_path,
            'score': score,
            'threshold': threshold,
            'exceeds_threshold': score > threshold,
            'status': 'exceeds_threshold' if score > threshold else 'within_threshold'
        }