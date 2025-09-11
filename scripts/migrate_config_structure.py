#!/usr/bin/env python3
"""
Configuration Structure Migration Script

This script helps migrate from the old config/ and configs/ structure to the new
unified config/ structure while maintaining backward compatibility.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Any
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigMigrator:
    """Handles migration of configuration structure."""
    
    def __init__(self, workspace_path: Path = None):
        """Initialize the migrator."""
        self.workspace_path = workspace_path or Path("/workspace")
        self.config_path = self.workspace_path / "config"
        self.legacy_configs_path = self.workspace_path / "configs"
        
        # Migration mapping
        self.migration_map = {
            # Environment configs
            "configs/development_config.json": "config/environments/development.json",
            "configs/production_config.json": "config/environments/production.json",
            "configs/testing_config.json": "config/environments/testing.json",
            
            # Feature configs (move to features subdirectory)
            "config/enhanced_reporting_config.yaml": "config/features/enhanced_reporting_config.yaml",
            "config/explainability_config.yaml": "config/features/explainability_config.yaml",
            "config/probabilistic_optimization.yaml": "config/features/probabilistic_optimization.yaml",
            "config/sr_levels_config.yaml": "config/features/sr_levels_config.yaml",
            "config/training_config.json": "config/features/training_config.json",
            "config/training_modes.yaml": "config/features/training_modes.yaml",
        }
    
    def check_current_structure(self) -> Dict[str, List[str]]:
        """Check the current configuration structure."""
        structure = {
            'config_files': [],
            'configs_files': [],
            'missing_files': []
        }
        
        # Check config/ directory
        if self.config_path.exists():
            for file_path in self.config_path.rglob("*"):
                if file_path.is_file() and file_path.name != "README.md":
                    structure['config_files'].append(str(file_path.relative_to(self.workspace_path)))
        
        # Check configs/ directory
        if self.legacy_configs_path.exists():
            for file_path in self.legacy_configs_path.rglob("*"):
                if file_path.is_file():
                    structure['configs_files'].append(str(file_path.relative_to(self.workspace_path)))
        
        # Check for missing files in new structure
        for old_path, new_path in self.migration_map.items():
            new_file_path = self.workspace_path / new_path
            if not new_file_path.exists():
                structure['missing_files'].append(new_path)
        
        return structure
    
    def create_directories(self) -> None:
        """Create necessary directories for the new structure."""
        directories = [
            self.config_path / "environments",
            self.config_path / "features"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def migrate_files(self, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate files to the new structure."""
        results = {
            'migrated': [],
            'skipped': [],
            'errors': []
        }
        
        for old_path, new_path in self.migration_map.items():
            old_file_path = self.workspace_path / old_path
            new_file_path = self.workspace_path / new_path
            
            try:
                if old_file_path.exists():
                    if dry_run:
                        logger.info(f"[DRY RUN] Would migrate: {old_path} -> {new_path}")
                        results['migrated'].append(f"{old_path} -> {new_path}")
                    else:
                        # Ensure target directory exists
                        new_file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Copy file to new location
                        shutil.copy2(old_file_path, new_file_path)
                        logger.info(f"Migrated: {old_path} -> {new_path}")
                        results['migrated'].append(f"{old_path} -> {new_path}")
                else:
                    logger.warning(f"Source file not found: {old_path}")
                    results['skipped'].append(old_path)
                    
            except Exception as e:
                error_msg = f"Error migrating {old_path}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def create_symlinks(self, dry_run: bool = False) -> Dict[str, Any]:
        """Create symlinks for backward compatibility."""
        results = {
            'created': [],
            'skipped': [],
            'errors': []
        }
        
        for old_path, new_path in self.migration_map.items():
            old_file_path = self.workspace_path / old_path
            new_file_path = self.workspace_path / new_path
            
            try:
                if new_file_path.exists() and not old_file_path.exists():
                    if dry_run:
                        logger.info(f"[DRY RUN] Would create symlink: {old_path} -> {new_path}")
                        results['created'].append(f"{old_path} -> {new_path}")
                    else:
                        # Create symlink for backward compatibility
                        old_file_path.symlink_to(new_file_path.relative_to(old_file_path.parent))
                        logger.info(f"Created symlink: {old_path} -> {new_path}")
                        results['created'].append(f"{old_path} -> {new_path}")
                else:
                    results['skipped'].append(old_path)
                    
            except Exception as e:
                error_msg = f"Error creating symlink {old_path}: {e}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def validate_migration(self) -> bool:
        """Validate that the migration was successful."""
        logger.info("Validating migration...")
        
        all_valid = True
        
        for old_path, new_path in self.migration_map.items():
            old_file_path = self.workspace_path / old_path
            new_file_path = self.workspace_path / new_path
            
            if new_file_path.exists():
                logger.info(f"✅ {new_path} exists")
            else:
                logger.error(f"❌ {new_path} missing")
                all_valid = False
        
        return all_valid
    
    def generate_migration_report(self, results: Dict[str, Any]) -> str:
        """Generate a migration report."""
        report = []
        report.append("# Configuration Migration Report")
        report.append("")
        report.append(f"**Migration completed at:** {Path().cwd()}")
        report.append("")
        
        if results.get('migrated'):
            report.append("## Migrated Files")
            for item in results['migrated']:
                report.append(f"- {item}")
            report.append("")
        
        if results.get('created'):
            report.append("## Created Symlinks (Backward Compatibility)")
            for item in results['created']:
                report.append(f"- {item}")
            report.append("")
        
        if results.get('skipped'):
            report.append("## Skipped Files")
            for item in results['skipped']:
                report.append(f"- {item}")
            report.append("")
        
        if results.get('errors'):
            report.append("## Errors")
            for item in results['errors']:
                report.append(f"- {item}")
            report.append("")
        
        report.append("## Next Steps")
        report.append("1. Test your application to ensure configurations are loaded correctly")
        report.append("2. Update any hardcoded paths in your code to use the new structure")
        report.append("3. Consider removing the old configs/ directory after verification")
        report.append("4. Use the UnifiedConfigService for new configuration loading")
        
        return "\n".join(report)


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(description="Migrate configuration structure")
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"),
                       help="Workspace path (default: /workspace)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be migrated without making changes")
    parser.add_argument("--no-symlinks", action="store_true",
                       help="Don't create backward compatibility symlinks")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate current structure")
    
    args = parser.parse_args()
    
    migrator = ConfigMigrator(args.workspace)
    
    if args.validate_only:
        structure = migrator.check_current_structure()
        logger.info("Current structure:")
        logger.info(f"Config files: {structure['config_files']}")
        logger.info(f"Configs files: {structure['configs_files']}")
        logger.info(f"Missing in new structure: {structure['missing_files']}")
        return
    
    # Check current structure
    structure = migrator.check_current_structure()
    logger.info(f"Found {len(structure['config_files'])} files in config/")
    logger.info(f"Found {len(structure['configs_files'])} files in configs/")
    
    if not structure['config_files'] and not structure['configs_files']:
        logger.warning("No configuration files found to migrate")
        return
    
    # Create directories
    migrator.create_directories()
    
    # Migrate files
    migration_results = migrator.migrate_files(dry_run=args.dry_run)
    
    # Create symlinks for backward compatibility
    if not args.no_symlinks and not args.dry_run:
        symlink_results = migrator.create_symlinks(dry_run=args.dry_run)
        migration_results['symlinks'] = symlink_results
    
    # Validate migration
    if not args.dry_run:
        is_valid = migrator.validate_migration()
        if is_valid:
            logger.info("✅ Migration validation successful")
        else:
            logger.error("❌ Migration validation failed")
            sys.exit(1)
    
    # Generate report
    report = migrator.generate_migration_report(migration_results)
    print("\n" + "="*50)
    print(report)
    print("="*50)
    
    # Save report to file
    if not args.dry_run:
        report_path = args.workspace / "config_migration_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Migration report saved to: {report_path}")


if __name__ == "__main__":
    main()