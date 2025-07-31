# src/database/migration_utils.py

import asyncio
import os
import shutil
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.utils.logger import system_logger
from src.database.sqlite_manager import SQLiteManager


class DatabaseMigrationUtils:
    """
    Utility class for managing database migrations between computers.
    Handles export, import, validation, and backup operations.
    """

    def __init__(self, db_manager: SQLiteManager):
        self.db_manager = db_manager
        self.logger = system_logger.getChild("MigrationUtils")

    async def export_for_trading(self, export_name: str = None) -> str:
        """
        Exports database from backtesting computer for use on trading computer.
        Filters out backtest-specific data and keeps only essential trading data.
        """
        if not export_name:
            export_name = f"trading_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        export_path = os.path.join(
            self.db_manager.migration_dir, f"{export_name}.sqlite"
        )

        try:
            # Create a clean copy for trading export
            shutil.copy2(self.db_manager.db_path, export_path)

            # Create a temporary SQLite manager for the export
            temp_db = SQLiteManager(export_path)
            await temp_db.initialize()

            # Remove backtest-specific data that shouldn't be on trading computer
            await self._clean_for_trading(temp_db)

            # Calculate checksum
            with open(export_path, "rb") as f:
                checksum = hashlib.md5(f.read()).hexdigest()

            # Record export
            export_data = {
                "export_id": export_name,
                "source_computer": os.uname().nodename
                if hasattr(os, "uname")
                else "unknown",
                "export_type": "trading_export",
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(export_path),
                "checksum": checksum,
                "description": "Database export for trading computer",
            }

            await self.db_manager.set_document(
                "database_migrations", export_name, export_data
            )

            self.logger.info(
                f"Trading export created: {export_path} (checksum: {checksum})"
            )
            return export_path

        except Exception as e:
            self.logger.error(f"Failed to create trading export: {e}", exc_info=True)
            return ""

    async def _clean_for_trading(self, temp_db: SQLiteManager):
        """Removes backtest-specific data from the export."""
        try:
            # Remove backtest results (keep only the latest successful ones)
            backtest_results = await temp_db.get_collection("backtest_results")
            if len(backtest_results) > 1:
                # Keep only the most recent successful backtest
                sorted_results = sorted(
                    backtest_results,
                    key=lambda x: x.get("created_at", ""),
                    reverse=True,
                )
                for result in sorted_results[1:]:
                    await temp_db.delete_document(
                        "backtest_results", result.get("backtest_id", "")
                    )

            # Mark remaining backtest as migrated
            for result in await temp_db.get_collection("backtest_results"):
                result["is_migrated"] = 1
                await temp_db.set_document(
                    "backtest_results", result.get("backtest_id", ""), result
                )

            self.logger.info("Database cleaned for trading export")

        except Exception as e:
            self.logger.error(
                f"Error cleaning database for trading: {e}", exc_info=True
            )

    async def import_for_trading(self, import_path: str) -> bool:
        """
        Imports database on trading computer from backtesting computer export.
        """
        try:
            # Verify file exists
            if not os.path.exists(import_path):
                self.logger.error(f"Import file not found: {import_path}")
                return False

            # Calculate checksum of import file
            with open(import_path, "rb") as f:
                checksum = hashlib.md5(f.read()).hexdigest()

            # Create backup before import
            await self.db_manager.create_backup("pre_trading_import")

            # Import the database
            success = await self.db_manager.import_migration(import_path)

            if success:
                # Update import record
                import_name = os.path.basename(import_path).replace(".sqlite", "")
                import_data = {
                    "import_id": import_name,
                    "target_computer": os.uname().nodename
                    if hasattr(os, "uname")
                    else "unknown",
                    "import_type": "trading_import",
                    "status": "completed",
                    "created_at": datetime.now().isoformat(),
                    "completed_at": datetime.now().isoformat(),
                    "file_size": os.path.getsize(import_path),
                    "checksum": checksum,
                    "description": "Database import for trading computer",
                }

                await self.db_manager.set_document(
                    "database_migrations", import_name, import_data
                )

                self.logger.info(f"Trading import completed: {import_path}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to import for trading: {e}", exc_info=True)
            return False

    async def export_backtest_results(self, export_name: str = None) -> str:
        """
        Exports only backtest results for analysis on another computer.
        """
        if not export_name:
            export_name = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        export_path = os.path.join(self.db_manager.migration_dir, f"{export_name}.json")

        try:
            # Get all backtest results
            backtest_results = await self.db_manager.get_collection("backtest_results")

            # Export to JSON
            export_data = {
                "export_id": export_name,
                "export_type": "backtest_results",
                "created_at": datetime.now().isoformat(),
                "source_computer": os.uname().nodename
                if hasattr(os, "uname")
                else "unknown",
                "results_count": len(backtest_results),
                "results": backtest_results,
            }

            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            # Calculate checksum
            with open(export_path, "rb") as f:
                checksum = hashlib.md5(f.read()).hexdigest()

            self.logger.info(
                f"Backtest results exported: {export_path} (checksum: {checksum})"
            )
            return export_path

        except Exception as e:
            self.logger.error(f"Failed to export backtest results: {e}", exc_info=True)
            return ""

    async def validate_migration_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validates a migration file for integrity and compatibility.
        """
        validation_result = {
            "valid": False,
            "file_exists": False,
            "checksum_valid": False,
            "size_valid": False,
            "database_valid": False,
            "errors": [],
        }

        try:
            # Check if file exists
            if not os.path.exists(file_path):
                validation_result["errors"].append("File does not exist")
                return validation_result
            validation_result["file_exists"] = True

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                validation_result["errors"].append("File is empty")
                return validation_result
            validation_result["size_valid"] = True

            # Calculate checksum
            with open(file_path, "rb") as f:
                checksum = hashlib.md5(f.read()).hexdigest()

            # Try to open as SQLite database
            try:
                import sqlite3

                conn = sqlite3.connect(file_path)
                cursor = conn.cursor()

                # Check if required tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]

                required_tables = [
                    "ares_optimized_params",
                    "ares_live_metrics",
                    "ares_alerts",
                    "detailed_trade_logs",
                    "model_checkpoints",
                    "backtest_results",
                    "trading_state",
                    "database_migrations",
                ]

                missing_tables = [
                    table for table in required_tables if table not in tables
                ]
                if missing_tables:
                    validation_result["errors"].append(
                        f"Missing required tables: {missing_tables}"
                    )
                else:
                    validation_result["database_valid"] = True

                conn.close()

            except Exception as e:
                validation_result["errors"].append(
                    f"Database validation failed: {str(e)}"
                )

            validation_result["checksum_valid"] = True
            validation_result["checksum"] = checksum
            validation_result["file_size"] = file_size

            # Overall validation
            validation_result["valid"] = (
                validation_result["file_exists"]
                and validation_result["size_valid"]
                and validation_result["checksum_valid"]
                and validation_result["database_valid"]
            )

        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")

        return validation_result

    async def list_migrations(self) -> List[Dict[str, Any]]:
        """
        Lists all available migrations with their details.
        """
        try:
            migrations = await self.db_manager.get_collection("database_migrations")
            return sorted(
                migrations, key=lambda x: x.get("created_at", ""), reverse=True
            )
        except Exception as e:
            self.logger.error(f"Failed to list migrations: {e}", exc_info=True)
            return []

    async def cleanup_old_migrations(self, keep_days: int = 30):
        """
        Cleans up old migration files and records.
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)

            # Get old migrations
            migrations = await self.list_migrations()
            old_migrations = [
                m
                for m in migrations
                if datetime.fromisoformat(m.get("created_at", "1970-01-01"))
                < cutoff_date
            ]

            for migration in old_migrations:
                migration_id = migration.get("migration_id", "")

                # Remove migration file
                migration_file = os.path.join(
                    self.db_manager.migration_dir, f"{migration_id}.sqlite"
                )
                if os.path.exists(migration_file):
                    os.remove(migration_file)
                    self.logger.info(f"Removed old migration file: {migration_file}")

                # Remove migration record
                await self.db_manager.delete_document(
                    "database_migrations", migration_id
                )
                self.logger.info(f"Removed old migration record: {migration_id}")

            self.logger.info(f"Cleaned up {len(old_migrations)} old migrations")

        except Exception as e:
            self.logger.error(f"Failed to cleanup old migrations: {e}", exc_info=True)


# Utility functions for command-line operations
async def export_database_for_trading(
    db_path: str = "data/ares_local_db.sqlite",
) -> str:
    """Command-line function to export database for trading."""
    db_manager = SQLiteManager(db_path)
    await db_manager.initialize()

    migration_utils = DatabaseMigrationUtils(db_manager)
    export_path = await migration_utils.export_for_trading()

    await db_manager.close()
    return export_path


async def import_database_for_trading(
    import_path: str, db_path: str = "data/ares_local_db.sqlite"
) -> bool:
    """Command-line function to import database for trading."""
    db_manager = SQLiteManager(db_path)
    await db_manager.initialize()

    migration_utils = DatabaseMigrationUtils(db_manager)
    success = await migration_utils.import_for_trading(import_path)

    await db_manager.close()
    return success


async def validate_migration_file(file_path: str) -> Dict[str, Any]:
    """Command-line function to validate a migration file."""
    db_manager = SQLiteManager()
    await db_manager.initialize()

    migration_utils = DatabaseMigrationUtils(db_manager)
    validation_result = await migration_utils.validate_migration_file(file_path)

    await db_manager.close()
    return validation_result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python migration_utils.py export [db_path]")
        print("  python migration_utils.py import <import_path> [db_path]")
        print("  python migration_utils.py validate <file_path>")
        sys.exit(1)

    command = sys.argv[1]

    async def main():
        if command == "export":
            db_path = sys.argv[2] if len(sys.argv) > 2 else "data/ares_local_db.sqlite"
            export_path = await export_database_for_trading(db_path)
            if export_path:
                print(f"Export created: {export_path}")
            else:
                print("Export failed")

        elif command == "import":
            if len(sys.argv) < 3:
                print("Import path required")
                sys.exit(1)
            import_path = sys.argv[2]
            db_path = sys.argv[3] if len(sys.argv) > 3 else "data/ares_local_db.sqlite"
            success = await import_database_for_trading(import_path, db_path)
            if success:
                print("Import completed successfully")
            else:
                print("Import failed")

        elif command == "validate":
            if len(sys.argv) < 3:
                print("File path required")
                sys.exit(1)
            file_path = sys.argv[2]
            validation_result = await validate_migration_file(file_path)
            print(json.dumps(validation_result, indent=2))

        else:
            print(f"Unknown command: {command}")
            sys.exit(1)

    asyncio.run(main())
