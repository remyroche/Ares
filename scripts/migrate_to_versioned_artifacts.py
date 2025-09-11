#!/usr/bin/env python3
"""
Migration Script: Convert to Versioned Artifacts

This script helps migrate existing artifacts to the new versioned naming system
and updates pipeline configurations to use the enhanced artifact management.
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import system_logger
from src.utils.version_manager import get_version_manager, set_ares_version
from src.utils.enhanced_artifact_manager import get_artifact_manager


class ArtifactMigrationManager:
    """Manages migration of artifacts to versioned naming system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the migration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = system_logger.getChild("ArtifactMigrationManager")
        self.config_path = config_path or "config/migration_config.json"
        self.config = self._load_config()
        
        # Initialize managers
        self.version_manager = get_version_manager()
        self.artifact_manager = get_artifact_manager()
    
    def _load_config(self) -> Dict:
        """Load migration configuration."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config
                default_config = {
                    "ares_version": "v1",
                    "backup_original": True,
                    "dry_run": True,
                    "artifact_directories": [
                        "artifacts",
                        "models", 
                        "data_cache",
                        "output"
                    ],
                    "file_patterns": [
                        "*.pkl",
                        "*.parquet", 
                        "*.json",
                        "*.joblib"
                    ],
                    "exclude_patterns": [
                        "*_metadata.json",
                        "*_registry.json"
                    ]
                }
                self._save_config(default_config)
                return default_config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def _save_config(self, config: Dict) -> None:
        """Save migration configuration."""
        try:
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
    
    def scan_artifacts(self) -> Dict[str, List[str]]:
        """Scan for existing artifacts that need migration.
        
        Returns:
            Dictionary mapping directories to lists of artifact files
        """
        artifacts = {}
        
        for directory in self.config.get("artifact_directories", []):
            if directory in self.artifact_manager.base_paths:
                dir_path = self.artifact_manager.base_paths[directory]
                if dir_path.exists():
                    artifacts[directory] = []
                    
                    for pattern in self.config.get("file_patterns", []):
                        for file_path in dir_path.rglob(pattern):
                            if file_path.is_file():
                                # Check if already versioned
                                if not self._is_versioned_filename(file_path.name):
                                    # Check exclude patterns
                                    if not self._should_exclude(file_path.name):
                                        artifacts[directory].append(str(file_path))
        
        return artifacts
    
    def _is_versioned_filename(self, filename: str) -> bool:
        """Check if filename already follows versioned naming pattern.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if filename is already versioned
        """
        import re
        # Pattern: base_name_version_YYYYMMDD_HHMMSS.extension
        pattern = r'^.+_[^_]+_\d{8}_\d{6}\..+$'
        return bool(re.match(pattern, filename))
    
    def _should_exclude(self, filename: str) -> bool:
        """Check if filename should be excluded from migration.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if filename should be excluded
        """
        for pattern in self.config.get("exclude_patterns", []):
            if filename.endswith(pattern.replace("*", "")):
                return True
        return False
    
    def _extract_base_name(self, file_path: str) -> str:
        """Extract base name from file path.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Base name without extension
        """
        path_obj = Path(file_path)
        return path_obj.stem
    
    def migrate_artifacts(self, dry_run: Optional[bool] = None) -> Dict[str, List[str]]:
        """Migrate artifacts to versioned naming.
        
        Args:
            dry_run: Whether to perform a dry run (defaults to config setting)
            
        Returns:
            Dictionary with migration results
        """
        if dry_run is None:
            dry_run = self.config.get("dry_run", True)
        
        artifacts = self.scan_artifacts()
        results = {
            "migrated": [],
            "skipped": [],
            "errors": []
        }
        
        self.logger.info(f"🔄 Starting artifact migration (dry_run={dry_run})")
        
        for directory, file_list in artifacts.items():
            self.logger.info(f"📁 Processing directory: {directory} ({len(file_list)} files)")
            
            for file_path in file_list:
                try:
                    result = self._migrate_single_artifact(file_path, dry_run)
                    if result["success"]:
                        results["migrated"].append(file_path)
                    else:
                        results["skipped"].append(file_path)
                except Exception as e:
                    self.logger.error(f"❌ Error migrating {file_path}: {e}")
                    results["errors"].append(f"{file_path}: {str(e)}")
        
        self.logger.info(f"✅ Migration completed: {len(results['migrated'])} migrated, "
                        f"{len(results['skipped'])} skipped, {len(results['errors'])} errors")
        
        return results
    
    def _migrate_single_artifact(self, file_path: str, dry_run: bool) -> Dict:
        """Migrate a single artifact file.
        
        Args:
            file_path: Path to the artifact file
            dry_run: Whether to perform a dry run
            
        Returns:
            Dictionary with migration result
        """
        path_obj = Path(file_path)
        base_name = self._extract_base_name(file_path)
        extension = path_obj.suffix
        
        # Generate new versioned filename
        new_filename = self.artifact_manager.get_versioned_filename(base_name, extension)
        new_path = path_obj.parent / new_filename
        
        # Check if target already exists
        if new_path.exists():
            self.logger.warning(f"⚠️ Target already exists: {new_filename}")
            return {"success": False, "reason": "target_exists"}
        
        if dry_run:
            self.logger.info(f"🔍 [DRY RUN] Would migrate: {path_obj.name} -> {new_filename}")
            return {"success": True, "reason": "dry_run"}
        
        # Create backup if configured
        if self.config.get("backup_original", True):
            backup_path = path_obj.with_suffix(f"{path_obj.suffix}.backup")
            shutil.copy2(file_path, backup_path)
            self.logger.debug(f"📋 Created backup: {backup_path.name}")
        
        # Rename the file
        path_obj.rename(new_path)
        self.logger.info(f"✅ Migrated: {path_obj.name} -> {new_filename}")
        
        return {"success": True, "reason": "migrated"}
    
    def update_pipeline_configs(self) -> Dict[str, bool]:
        """Update pipeline configuration files to use new artifact system.
        
        Returns:
            Dictionary mapping config files to update success status
        """
        results = {}
        
        # List of configuration files to update
        config_files = [
            "configs/development_config.json",
            "configs/production_config.json", 
            "configs/testing_config.json"
        ]
        
        for config_file in config_files:
            try:
                if Path(config_file).exists():
                    success = self._update_single_config(config_file)
                    results[config_file] = success
                else:
                    self.logger.warning(f"⚠️ Config file not found: {config_file}")
                    results[config_file] = False
            except Exception as e:
                self.logger.error(f"❌ Error updating {config_file}: {e}")
                results[config_file] = False
        
        return results
    
    def _update_single_config(self, config_file: str) -> bool:
        """Update a single configuration file.
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            True if update was successful
        """
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Add artifact management configuration
            if "artifact_management" not in config:
                config["artifact_management"] = {
                    "enabled": True,
                    "versioning": True,
                    "auto_cleanup": False,
                    "keep_recent_count": 5
                }
            
            # Add version configuration
            if "ares_version" not in config:
                config["ares_version"] = self.version_manager.get_ares_version()
            
            # Save updated config
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"✅ Updated configuration: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update {config_file}: {e}")
            return False
    
    def generate_migration_report(self, results: Dict) -> str:
        """Generate a migration report.
        
        Args:
            results: Migration results dictionary
            
        Returns:
            Formatted migration report
        """
        report = []
        report.append("📊 Artifact Migration Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Ares Version: {self.version_manager.get_ares_version()}")
        report.append("")
        
        report.append("📈 Migration Summary:")
        report.append(f"  ✅ Migrated: {len(results.get('migrated', []))}")
        report.append(f"  ⏭️ Skipped: {len(results.get('skipped', []))}")
        report.append(f"  ❌ Errors: {len(results.get('errors', []))}")
        report.append("")
        
        if results.get('migrated'):
            report.append("✅ Successfully Migrated:")
            for file_path in results['migrated']:
                report.append(f"  - {file_path}")
            report.append("")
        
        if results.get('errors'):
            report.append("❌ Errors:")
            for error in results['errors']:
                report.append(f"  - {error}")
            report.append("")
        
        report.append("🔧 Next Steps:")
        report.append("1. Review migrated artifacts")
        report.append("2. Update pipeline code to use new artifact pickup utilities")
        report.append("3. Test pipeline execution with new artifact system")
        report.append("4. Remove backup files if migration was successful")
        
        return "\n".join(report)


def main():
    """Main migration function."""
    print("🚀 Ares Artifact Migration Tool")
    print("=" * 40)
    
    # Initialize migration manager
    migration_manager = ArtifactMigrationManager()
    
    # Scan for artifacts
    print("\n1. Scanning for artifacts to migrate...")
    artifacts = migration_manager.scan_artifacts()
    
    total_artifacts = sum(len(files) for files in artifacts.values())
    print(f"📊 Found {total_artifacts} artifacts across {len(artifacts)} directories")
    
    for directory, files in artifacts.items():
        print(f"   📁 {directory}: {len(files)} files")
    
    if total_artifacts == 0:
        print("✅ No artifacts need migration!")
        return
    
    # Perform migration
    print(f"\n2. Performing migration (dry_run={migration_manager.config.get('dry_run', True)})...")
    results = migration_manager.migrate_artifacts()
    
    # Update configurations
    print("\n3. Updating pipeline configurations...")
    config_results = migration_manager.update_pipeline_configs()
    
    for config_file, success in config_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {config_file}")
    
    # Generate report
    print("\n4. Generating migration report...")
    report = migration_manager.generate_migration_report(results)
    
    # Save report
    report_file = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📋 Report saved to: {report_file}")
    print("\n" + report)


if __name__ == "__main__":
    main()