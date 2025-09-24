#!/usr/bin/env python3
"""
Cleanup Redundant Code Script

This script helps identify and remove redundant code after the unified utilities
enhancement. It provides a safe way to remove redundant components while
ensuring all imports are properly updated.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Dict, Set
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedundantCodeCleanup:
    """Handles cleanup of redundant code after unified utilities enhancement."""
    
    def __init__(self, workspace_root: str = "/workspace"):
        self.workspace_root = Path(workspace_root)
        self.redundant_files = self._identify_redundant_files()
        self.import_mappings = self._create_import_mappings()
        
    def _identify_redundant_files(self) -> List[Path]:
        """Identify redundant files that can be removed."""
        redundant_files = [
            # NAS system redundant files
            self.workspace_root / "src/training/steps/market_analysis/nas_regime/evaluation/economic_evaluator.py",
            self.workspace_root / "src/training/steps/market_analysis/nas_regime/evaluation/trading_viability_evaluator.py",
            self.workspace_root / "src/training/steps/market_analysis/nas_regime/optimization/multi_objective_optimizer.py",
            
            # Hybrid system redundant files
            self.workspace_root / "src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/economic_significance.py",
            self.workspace_root / "src/training/steps/market_analysis/hybrid_nas_tas_regime/shared_utils/trading_viability.py",
            self.workspace_root / "src/training/steps/market_analysis/hybrid_nas_tas_regime/core/multi_objective_optimizer.py",
        ]
        
        # Filter to only existing files
        existing_files = [f for f in redundant_files if f.exists()]
        logger.info(f"Found {len(existing_files)} redundant files to remove")
        return existing_files
    
    def _create_import_mappings(self) -> Dict[str, str]:
        """Create mappings from old imports to new unified imports."""
        return {
            # Economic evaluator mappings
            "from ..evaluation.economic_evaluator import EconomicSignificanceEvaluator": 
                "from ...hybrid_nas_tas_regime.shared_utils import UnifiedEconomicSignificanceEvaluator",
            "from .economic_evaluator import EconomicSignificanceEvaluator":
                "from ..shared_utils import UnifiedEconomicSignificanceEvaluator",
            "EconomicSignificanceEvaluator": "UnifiedEconomicSignificanceEvaluator",
            
            # Trading viability mappings
            "from ..evaluation.trading_viability_evaluator import TradingViabilityEvaluator":
                "from ...hybrid_nas_tas_regime.shared_utils import UnifiedTradingViabilityEvaluator",
            "from .trading_viability_evaluator import TradingViabilityEvaluator":
                "from ..shared_utils import UnifiedTradingViabilityEvaluator",
            "TradingViabilityEvaluator": "UnifiedTradingViabilityEvaluator",
            
            # Multi-objective optimizer mappings
            "from ..optimization.multi_objective_optimizer import PerfectMultiObjectiveOptimizer":
                "from ...hybrid_nas_tas_regime.shared_utils import UnifiedMultiObjectiveOptimizer",
            "from .multi_objective_optimizer import TradingMultiObjectiveOptimizer":
                "from ..shared_utils import UnifiedMultiObjectiveOptimizer",
            "PerfectMultiObjectiveOptimizer": "UnifiedMultiObjectiveOptimizer",
            "TradingMultiObjectiveOptimizer": "UnifiedMultiObjectiveOptimizer",
        }
    
    def find_files_with_imports(self, import_pattern: str) -> List[Path]:
        """Find files that import the given pattern."""
        files_with_imports = []
        
        for root, dirs, files in os.walk(self.workspace_root):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if import_pattern in content:
                                files_with_imports.append(file_path)
                    except Exception as e:
                        logger.warning(f"Could not read {file_path}: {e}")
        
        return files_with_imports
    
    def update_imports_in_file(self, file_path: Path) -> bool:
        """Update imports in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Apply import mappings
            for old_import, new_import in self.import_mappings.items():
                content = content.replace(old_import, new_import)
            
            # Only write if content changed
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Updated imports in {file_path}")
                return True
            else:
                logger.debug(f"No imports to update in {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update imports in {file_path}: {e}")
            return False
    
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of a file before deletion."""
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    
    def remove_redundant_file(self, file_path: Path, create_backup: bool = True) -> bool:
        """Remove a redundant file."""
        try:
            if create_backup:
                self.backup_file(file_path)
            
            file_path.unlink()
            logger.info(f"Removed redundant file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove {file_path}: {e}")
            return False
    
    def analyze_impact(self) -> Dict[str, List[Path]]:
        """Analyze the impact of removing redundant files."""
        impact = {}
        
        for file_path in self.redundant_files:
            if file_path.exists():
                # Find files that import from this redundant file
                import_patterns = [
                    f"from {file_path.stem} import",
                    f"import {file_path.stem}",
                    f"from .{file_path.stem} import",
                    f"from ..{file_path.stem} import",
                ]
                
                affected_files = []
                for pattern in import_patterns:
                    affected_files.extend(self.find_files_with_imports(pattern))
                
                impact[str(file_path)] = affected_files
        
        return impact
    
    def dry_run(self) -> None:
        """Perform a dry run to show what would be removed."""
        logger.info("=== DRY RUN: Redundant Code Cleanup ===")
        
        # Show redundant files
        logger.info(f"\nRedundant files to remove ({len(self.redundant_files)}):")
        for file_path in self.redundant_files:
            logger.info(f"  - {file_path}")
        
        # Show impact analysis
        impact = self.analyze_impact()
        logger.info(f"\nImpact analysis:")
        for redundant_file, affected_files in impact.items():
            logger.info(f"  {redundant_file}:")
            if affected_files:
                for affected_file in affected_files:
                    logger.info(f"    -> {affected_file}")
            else:
                logger.info("    -> No files affected")
        
        # Show import mappings
        logger.info(f"\nImport mappings to apply:")
        for old_import, new_import in self.import_mappings.items():
            logger.info(f"  {old_import}")
            logger.info(f"    -> {new_import}")
    
    def cleanup(self, dry_run: bool = True, create_backups: bool = True) -> None:
        """Perform the cleanup."""
        if dry_run:
            self.dry_run()
            return
        
        logger.info("=== Starting Redundant Code Cleanup ===")
        
        # Step 1: Update imports in affected files
        logger.info("Step 1: Updating imports...")
        updated_files = 0
        for import_pattern in self.import_mappings.keys():
            affected_files = self.find_files_with_imports(import_pattern)
            for file_path in affected_files:
                if self.update_imports_in_file(file_path):
                    updated_files += 1
        
        logger.info(f"Updated imports in {updated_files} files")
        
        # Step 2: Remove redundant files
        logger.info("Step 2: Removing redundant files...")
        removed_files = 0
        for file_path in self.redundant_files:
            if self.remove_redundant_file(file_path, create_backups):
                removed_files += 1
        
        logger.info(f"Removed {removed_files} redundant files")
        
        # Step 3: Clean up empty directories
        logger.info("Step 3: Cleaning up empty directories...")
        self._cleanup_empty_directories()
        
        logger.info("=== Cleanup Complete ===")
    
    def _cleanup_empty_directories(self) -> None:
        """Remove empty directories after file removal."""
        # This is a simplified version - in practice, you'd want to be more careful
        # about which directories to remove
        pass


def main():
    """Main function to run the cleanup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cleanup redundant code after unified utilities enhancement")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without making changes")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backups before deletion")
    parser.add_argument("--workspace", default="/workspace", help="Workspace root directory")
    
    args = parser.parse_args()
    
    cleanup = RedundantCodeCleanup(args.workspace)
    cleanup.cleanup(dry_run=args.dry_run, create_backups=not args.no_backup)


if __name__ == "__main__":
    main()