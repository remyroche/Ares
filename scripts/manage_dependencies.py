#!/usr/bin/env python3
"""
Dependency Management Script for Ares
Handles installation, compatibility checking, and version management.
"""

import subprocess
import sys
import os
import json
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages project dependencies and compatibility."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.platform = platform.system().lower()
        self.architecture = platform.machine().lower()
        
        # Platform-specific optimizations
        self.platform_configs = {
            'darwin': {
                'm1_optimized': self.architecture in ['arm64', 'aarch64'],
                'mps_available': True,  # Metal Performance Shaders
                'cuda_available': False
            },
            'linux': {
                'm1_optimized': False,
                'mps_available': False,
                'cuda_available': True
            },
            'windows': {
                'm1_optimized': False,
                'mps_available': False,
                'cuda_available': True
            }
        }
    
    def check_poetry_installed(self) -> bool:
        """Check if Poetry is installed."""
        try:
            result = subprocess.run(['poetry', '--version'], 
                                  capture_output=True, text=True, check=True)
            logger.info(f"Poetry found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Poetry not found. Please install Poetry first.")
            return False
    
    def install_poetry(self) -> bool:
        """Install Poetry if not available."""
        try:
            logger.info("Installing Poetry...")
            if self.platform == 'windows':
                subprocess.run([
                    'powershell', '-Command', 
                    '(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -'
                ], check=True)
            else:
                subprocess.run([
                    'curl', '-sSL', 'https://install.python-poetry.org', '|', 'python3', '-'
                ], shell=True, check=True)
            logger.info("Poetry installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Poetry: {e}")
            return False
    
    def check_python_version(self) -> bool:
        """Check if Python version is compatible."""
        required_version = (3, 11)
        current_version = (sys.version_info.major, sys.version_info.minor)
        
        if current_version < required_version:
            logger.error(f"Python {required_version[0]}.{required_version[1]}+ required, "
                        f"found {current_version[0]}.{current_version[1]}")
            return False
        
        logger.info(f"Python version {self.python_version} is compatible")
        return True
    
    def check_platform_optimizations(self) -> Dict[str, bool]:
        """Check for platform-specific optimizations."""
        config = self.platform_configs.get(self.platform, {})
        
        optimizations = {
            'm1_optimized': config.get('m1_optimized', False),
            'mps_available': config.get('mps_available', False),
            'cuda_available': config.get('cuda_available', False)
        }
        
        if optimizations['m1_optimized']:
            logger.info("🍎 M1/M2 Mac detected - optimizations available")
        if optimizations['mps_available']:
            logger.info("⚡ Metal Performance Shaders available")
        if optimizations['cuda_available']:
            logger.info("🚀 CUDA support available")
        
        return optimizations
    
    def install_dependencies(self, group: str = "main", dev: bool = False) -> bool:
        """Install dependencies using Poetry."""
        if not self.check_poetry_installed():
            if not self.install_poetry():
                return False
        
        try:
            cmd = ['poetry', 'install']
            if dev:
                cmd.append('--with=dev')
            if group != "main":
                cmd.extend(['--with', group])
            
            logger.info(f"Installing dependencies: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=self.project_root, check=True)
            logger.info("Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def check_dependency_conflicts(self) -> List[Tuple[str, str, str]]:
        """Check for potential dependency conflicts."""
        conflicts = []
        
        try:
            # Get installed packages
            result = subprocess.run(['poetry', 'show', '--tree'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                logger.warning("Could not check dependency tree")
                return conflicts
            
            # Parse dependencies (simplified conflict detection)
            lines = result.stdout.split('\n')
            package_versions = {}
            
            for line in lines:
                if '─' in line and '==' in line:
                    parts = line.strip().split('==')
                    if len(parts) == 2:
                        package = parts[0].strip().split()[-1]
                        version = parts[1].strip()
                        if package in package_versions:
                            conflicts.append((package, package_versions[package], version))
                        else:
                            package_versions[package] = version
            
            if conflicts:
                logger.warning(f"Found {len(conflicts)} potential conflicts")
            else:
                logger.info("No dependency conflicts detected")
                
        except Exception as e:
            logger.error(f"Error checking conflicts: {e}")
        
        return conflicts
    
    def update_dependencies(self) -> bool:
        """Update dependencies to latest compatible versions."""
        try:
            logger.info("Updating dependencies...")
            subprocess.run(['poetry', 'update'], cwd=self.project_root, check=True)
            logger.info("Dependencies updated successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to update dependencies: {e}")
            return False
    
    def export_requirements(self, output_file: str = "requirements.txt") -> bool:
        """Export current dependencies to requirements.txt."""
        try:
            logger.info(f"Exporting requirements to {output_file}...")
            subprocess.run(['poetry', 'export', '-f', 'requirements.txt', 
                          '--output', output_file, '--without-hashes'], 
                         cwd=self.project_root, check=True)
            logger.info(f"Requirements exported to {output_file}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to export requirements: {e}")
            return False
    
    def check_gpu_support(self) -> Dict[str, bool]:
        """Check GPU support availability."""
        gpu_support = {
            'cuda': False,
            'mps': False,
            'cpu_only': True
        }
        
        try:
            # Check CUDA
            import torch
            gpu_support['cuda'] = torch.cuda.is_available()
            if gpu_support['cuda']:
                gpu_support['cpu_only'] = False
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass
        
        try:
            # Check MPS (Metal Performance Shaders)
            import torch
            gpu_support['mps'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            if gpu_support['mps']:
                gpu_support['cpu_only'] = False
                logger.info("MPS (Metal Performance Shaders) available")
        except ImportError:
            pass
        
        return gpu_support
    
    def run_compatibility_check(self) -> bool:
        """Run comprehensive compatibility check."""
        logger.info("🔍 Running compatibility check...")
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check platform optimizations
        optimizations = self.check_platform_optimizations()
        
        # Check GPU support
        gpu_support = self.check_gpu_support()
        
        # Check dependency conflicts
        conflicts = self.check_dependency_conflicts()
        
        # Summary
        logger.info("📊 Compatibility Summary:")
        logger.info(f"  Python: {self.python_version}")
        logger.info(f"  Platform: {self.platform} ({self.architecture})")
        logger.info(f"  M1 Optimized: {optimizations['m1_optimized']}")
        logger.info(f"  GPU Support: CUDA={gpu_support['cuda']}, MPS={gpu_support['mps']}")
        logger.info(f"  Conflicts: {len(conflicts)}")
        
        return len(conflicts) == 0

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Ares Dependency Manager")
    parser.add_argument('command', choices=[
        'install', 'update', 'check', 'export', 'conflicts'
    ], help='Command to run')
    parser.add_argument('--dev', action='store_true', help='Include development dependencies')
    parser.add_argument('--group', default='main', help='Dependency group to install')
    parser.add_argument('--output', default='requirements.txt', help='Output file for export')
    
    args = parser.parse_args()
    
    manager = DependencyManager()
    
    if args.command == 'install':
        success = manager.install_dependencies(group=args.group, dev=args.dev)
        sys.exit(0 if success else 1)
    
    elif args.command == 'update':
        success = manager.update_dependencies()
        sys.exit(0 if success else 1)
    
    elif args.command == 'check':
        success = manager.run_compatibility_check()
        sys.exit(0 if success else 1)
    
    elif args.command == 'export':
        success = manager.export_requirements(args.output)
        sys.exit(0 if success else 1)
    
    elif args.command == 'conflicts':
        conflicts = manager.check_dependency_conflicts()
        if conflicts:
            print("Dependency conflicts found:")
            for package, version1, version2 in conflicts:
                print(f"  {package}: {version1} vs {version2}")
        else:
            print("No conflicts found")
        sys.exit(0)

if __name__ == "__main__":
    main()
