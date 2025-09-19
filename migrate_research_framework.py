#!/usr/bin/env python3
"""
Research Framework Migration Script

This script migrates the existing research directories into the new 
cluster_analysis framework structure.

Usage:
    python migrate_research_framework.py [--dry-run] [--phase PHASE]
    
Options:
    --dry-run: Show what would be done without executing
    --phase: Run specific migration phase (1-4)
"""

import os
import shutil
import argparse
from pathlib import Path

class ResearchFrameworkMigrator:
    """Migrates existing research framework to new structure."""
    
    def __init__(self, workspace_root="/workspace", dry_run=False):
        self.workspace_root = Path(workspace_root)
        self.dry_run = dry_run
        
        # Source directories
        self.src_price_patterns = self.workspace_root / "src/research/price_patterns"
        self.src_mixed_factor = self.workspace_root / "src/research/mixed_factor_analysis" 
        self.src_clusters = self.workspace_root / "src/research/clusters"
        
        # Target directory
        self.target_base = self.workspace_root / "src/research/cluster_analysis"
        
        # Target subdirectories
        self.target_patterns = self.target_base / "price_patterns"
        self.target_factors = self.target_base / "market_factor_analysis"
        self.target_clustering = self.target_base / "clustering"
        self.target_relevance = self.target_base / "economic_relevance"
    
    def log(self, message):
        """Log migration action."""
        prefix = "[DRY RUN] " if self.dry_run else "[MIGRATE] "
        print(f"{prefix}{message}")
    
    def copy_file(self, src, dst, rename=None):
        """Copy file with optional rename."""
        if not src.exists():
            self.log(f"WARNING: Source file does not exist: {src}")
            return
        
        if rename:
            dst = dst.parent / rename
        
        self.log(f"Copy: {src} -> {dst}")
        
        if not self.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    
    def migrate_phase_1_price_patterns(self):
        """Phase 1: Migrate price patterns components."""
        
        self.log("=== PHASE 1: Price Patterns Migration ===")
        
        # Core pattern files
        self.copy_file(
            self.src_price_patterns / "core_patterns.py",
            self.target_patterns / "mathematical_definitions.py"
        )
        
        self.copy_file(
            self.src_price_patterns / "pure_price_action_patterns.py", 
            self.target_patterns / "pure_price_patterns.py"
        )
        
        # ML discovery files
        ml_discovery_target = self.target_patterns / "ml_discovery"
        
        self.copy_file(
            self.src_price_patterns / "lstm_discovery.py",
            ml_discovery_target / "lstm_discovery.py"
        )
        
        self.copy_file(
            self.src_price_patterns / "matrix_profile_discovery.py",
            ml_discovery_target / "matrix_profile_discovery.py"
        )
        
        self.copy_file(
            self.src_price_patterns / "ml_pure_price_pattern_discovery.py",
            ml_discovery_target / "clustering_discovery.py"
        )
        
        # From mixed_factor_analysis
        self.copy_file(
            self.src_mixed_factor / "ml_pattern_discovery.py",
            ml_discovery_target / "anomaly_discovery.py"
        )
        
        # Create pattern validation (new file)
        validation_file = self.target_patterns / "pattern_validation.py"
        if not self.dry_run:
            validation_file.parent.mkdir(parents=True, exist_ok=True)
            with open(validation_file, 'w') as f:
                f.write('"""Pattern validation module - to be implemented"""\npass\n')
        self.log(f"Create: {validation_file}")
    
    def migrate_phase_2_market_factors(self):
        """Phase 2: Migrate market factor analysis components."""
        
        self.log("=== PHASE 2: Market Factor Analysis Migration ===")
        
        # From clusters
        self.copy_file(
            self.src_clusters / "dimension_analyzer.py",
            self.target_factors / "dimension_discovery.py"
        )
        
        self.copy_file(
            self.src_clusters / "advanced_feature_engineering.py",
            self.target_factors / "factor_extraction.py"
        )
        
        self.copy_file(
            self.src_clusters / "statistical_dimension_analysis.py",
            self.target_factors / "statistical_analysis.py"
        )
        
        # Create feature clustering (new file)
        clustering_file = self.target_factors / "feature_clustering.py"
        if not self.dry_run:
            clustering_file.parent.mkdir(parents=True, exist_ok=True)
            with open(clustering_file, 'w') as f:
                f.write('"""Feature clustering module - to be implemented"""\npass\n')
        self.log(f"Create: {clustering_file}")
    
    def migrate_phase_3_clustering(self):
        """Phase 3: Migrate clustering components."""
        
        self.log("=== PHASE 3: Clustering Migration ===")
        
        # Core clustering files
        self.copy_file(
            self.src_clusters / "regime_clusterer.py",
            self.target_clustering / "regime_discovery.py"
        )
        
        self.copy_file(
            self.src_clusters / "similarity_matrix_clustering.py",
            self.target_clustering / "similarity_clustering.py"
        )
        
        self.copy_file(
            self.src_clusters / "validation_metrics.py",
            self.target_clustering / "validation_metrics.py"
        )
        
        self.copy_file(
            self.src_clusters / "data_driven_clustering_framework.py",
            self.target_clustering / "optimal_cluster_selection.py"
        )
    
    def migrate_phase_4_economic_relevance(self):
        """Phase 4: Migrate economic relevance components."""
        
        self.log("=== PHASE 4: Economic Relevance Migration ===")
        
        # From mixed_factor_analysis
        self.copy_file(
            self.src_mixed_factor / "economic_relevance_research_framework.py",
            self.target_relevance / "causal_analysis.py"
        )
        
        self.copy_file(
            self.src_mixed_factor / "pattern_ml_integration.py",
            self.target_relevance / "pattern_dimension_analysis.py"
        )
        
        # From clusters
        self.copy_file(
            self.src_clusters / "economic_metrics.py",
            self.target_relevance / "trading_significance.py"
        )
        
        # Create market state relevance (new file - merge dimension_economic_relevance.py)
        state_relevance_file = self.target_relevance / "market_state_relevance.py"
        if not self.dry_run:
            state_relevance_file.parent.mkdir(parents=True, exist_ok=True)
            # Copy content from dimension_economic_relevance.py as base
            if (self.src_clusters / "dimension_economic_relevance.py").exists():
                shutil.copy2(
                    self.src_clusters / "dimension_economic_relevance.py",
                    state_relevance_file
                )
        self.log(f"Create: {state_relevance_file} (based on dimension_economic_relevance.py)")
    
    def update_imports_phase_1(self):
        """Update import statements for phase 1 files."""
        self.log("=== Updating imports for Phase 1 ===")
        
        # This would be implemented to update import statements
        # For now, just log what needs to be done
        files_to_update = [
            self.target_patterns / "mathematical_definitions.py",
            self.target_patterns / "pure_price_patterns.py",
            self.target_patterns / "ml_discovery" / "lstm_discovery.py",
            self.target_patterns / "ml_discovery" / "matrix_profile_discovery.py",
            self.target_patterns / "ml_discovery" / "clustering_discovery.py",
            self.target_patterns / "ml_discovery" / "anomaly_discovery.py",
        ]
        
        for file_path in files_to_update:
            self.log(f"TODO: Update imports in {file_path}")
    
    def create_backup(self):
        """Create backup of existing directories."""
        self.log("=== Creating Backup ===")
        
        backup_dir = self.workspace_root / "research_backup"
        
        if not self.dry_run:
            backup_dir.mkdir(exist_ok=True)
        
        for src_dir in [self.src_price_patterns, self.src_mixed_factor, self.src_clusters]:
            if src_dir.exists():
                backup_target = backup_dir / src_dir.name
                self.log(f"Backup: {src_dir} -> {backup_target}")
                
                if not self.dry_run:
                    if backup_target.exists():
                        shutil.rmtree(backup_target)
                    shutil.copytree(src_dir, backup_target)
    
    def run_migration(self, phase=None):
        """Run complete migration or specific phase."""
        
        if phase is None:
            # Run all phases
            self.create_backup()
            self.migrate_phase_1_price_patterns()
            self.migrate_phase_2_market_factors()
            self.migrate_phase_3_clustering()
            self.migrate_phase_4_economic_relevance()
            self.update_imports_phase_1()
        else:
            # Run specific phase
            phase_methods = {
                1: self.migrate_phase_1_price_patterns,
                2: self.migrate_phase_2_market_factors,
                3: self.migrate_phase_3_clustering,
                4: self.migrate_phase_4_economic_relevance
            }
            
            if phase in phase_methods:
                if phase == 1:
                    self.create_backup()
                phase_methods[phase]()
                if phase == 1:
                    self.update_imports_phase_1()
            else:
                self.log(f"ERROR: Invalid phase {phase}. Must be 1-4.")
    
    def generate_migration_report(self):
        """Generate migration report."""
        self.log("=== Migration Report ===")
        
        # Count files in each source directory
        def count_py_files(directory):
            if not directory.exists():
                return 0
            return len(list(directory.glob("*.py")))
        
        self.log(f"Source files:")
        self.log(f"  price_patterns/: {count_py_files(self.src_price_patterns)} Python files")
        self.log(f"  mixed_factor_analysis/: {count_py_files(self.src_mixed_factor)} Python files")
        self.log(f"  clusters/: {count_py_files(self.src_clusters)} Python files")
        
        self.log(f"Target structure created:")
        self.log(f"  {self.target_patterns}/")
        self.log(f"  {self.target_factors}/")
        self.log(f"  {self.target_clustering}/")
        self.log(f"  {self.target_relevance}/")


def main():
    parser = argparse.ArgumentParser(description="Migrate research framework")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without executing")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                       help="Run specific migration phase")
    parser.add_argument("--workspace", default="/workspace",
                       help="Workspace root directory")
    
    args = parser.parse_args()
    
    migrator = ResearchFrameworkMigrator(
        workspace_root=args.workspace,
        dry_run=args.dry_run
    )
    
    print("Research Framework Migration")
    print("=" * 50)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    migrator.run_migration(args.phase)
    migrator.generate_migration_report()
    
    print("\nMigration completed!")
    
    if not args.dry_run:
        print("\nNext steps:")
        print("1. Update import statements in migrated files")
        print("2. Run tests to ensure functionality")
        print("3. Update documentation")
        print("4. Remove old directories after validation")


if __name__ == "__main__":
    main()