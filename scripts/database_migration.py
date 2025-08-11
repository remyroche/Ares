#!/usr/bin/env python3
"""
Database Migration Script for Ares Trading Bot

This script manages database migrations between backtesting and trading computers.
It allows you to:
1. Export database from backtesting computer for trading
2. Import database on trading computer from backtesting
3. Validate migration files
4. Manage backups and cleanup

Usage:
    python scripts/database_migration.py export [db_path]
    python scripts/database_migration.py import <import_path> [db_path]
    python scripts/database_migration.py validate <file_path>
    python scripts/database_migration.py backup [db_path]
    python scripts/database_migration.py list-migrations [db_path]
    python scripts/database_migration.py cleanup [db_path]
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.migration_utils import DatabaseMigrationUtils
from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import setup_logging, system_logger
from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)


async def export_database(db_path: str = "data/ares_local_db.sqlite"):
    """Export database for trading computer."""
    # logger = system_logger.getChild("MigrationScript")  # Unused variable removed

    try:
        # Initialize database manager
        db_manager = SQLiteManager(db_path)
        await db_manager.initialize()

        # Create migration utils
        migration_utils = DatabaseMigrationUtils(db_manager)

        # Export for trading
        export_path = await migration_utils.export_for_trading()

        if export_path:
            print("✅ Database exported successfully!")
            print(f"📁 Export file: {export_path}")
            print(f"📊 File size: {os.path.getsize(export_path) / 1024 / 1024:.2f} MB")

            # Calculate checksum
            import hashlib

            with open(export_path, "rb") as f:
                checksum = hashlib.md5(f.read()).hexdigest()
            print(f"🔍 Checksum: {checksum}")
            print("\n📋 Next steps:")
            print("   1. Copy the export file to your trading computer")
            print(
                f"   2. Run: python scripts/database_migration.py import {export_path}",
            )
        else:
            print(failed("Database export failed!")))

        await db_manager.close()

    except Exception as e:
        system_print(failed("Export failed: {e}")), exc_info=True)
        print(failed("Export failed: {e}")))


async def import_database(import_path: str, db_path: str = "data/ares_local_db.sqlite"):
    """Import database on trading computer."""
    # logger = system_logger.getChild("MigrationScript")  # Unused variable removed

    try:
        # Validate the import file first
        db_manager = SQLiteManager(db_path)
        await db_manager.initialize()

        migration_utils = DatabaseMigrationUtils(db_manager)
        validation_result = await migration_utils.validate_migration_file(import_path)

        if not validation_result["valid"]:
            print(failed("Import file validation failed!")))
            print("Errors:")
            for error in validation_result["errors"]:
                print(f"   - {error}")
            return

        print("✅ Import file validation passed!")

        # Import the database
        success = await migration_utils.import_for_trading(import_path)

        if success:
            print("✅ Database imported successfully!")
            print(f"📁 Database location: {db_path}")
            print(f"📊 Database size: {os.path.getsize(db_path) / 1024 / 1024:.2f} MB")
            print("\n🚀 You can now start the trading bot!")
        else:
            print(failed("Database import failed!")))

        await db_manager.close()

    except Exception as e:
        system_print(failed("Import failed: {e}")), exc_info=True)
        print(failed("Import failed: {e}")))


async def validate_file(file_path: str):
    """Validate a migration file."""
    # logger = system_logger.getChild("MigrationScript")  # Unused variable removed

    try:
        db_manager = SQLiteManager()
        await db_manager.initialize()

        migration_utils = DatabaseMigrationUtils(db_manager)
        validation_result = await migration_utils.validate_migration_file(file_path)

        print("🔍 File Validation Results:")
        print(f"   File exists: {'✅' if validation_result['file_exists'] else '❌'}")
        print(f"   Size valid: {'✅' if validation_result['size_valid'] else '❌'}")
        print(
            f"   Checksum valid: {'✅' if validation_result['checksum_valid'] else '❌'}",
        )
        print(
            f"   Database valid: {'✅' if validation_result['database_valid'] else '❌'}",
        )

        if validation_result["checksum"]:
            print(f"   Checksum: {validation_result['checksum']}")
        if validation_result["file_size"]:
            print(
                f"   File size: {validation_result['file_size'] / 1024 / 1024:.2f} MB",
            )

        if validation_result["errors"]:
            print("\n❌ Errors found:")
            for error in validation_result["errors"]:
                print(f"   - {error}")
        else:
            print("\n✅ File is valid and ready for import!")

        await db_manager.close()

    except Exception as e:
        system_print(failed("Validation failed: {e}")), exc_info=True)
        print(failed("Validation failed: {e}")))


async def create_backup(db_path: str = "data/ares_local_db.sqlite"):
    """Create a manual backup of the database."""
    # logger = system_logger.getChild("MigrationScript")  # Unused variable removed

    try:
        db_manager = SQLiteManager(db_path)
        await db_manager.initialize()

        backup_path = await db_manager.create_backup()

        if backup_path:
            print("✅ Backup created successfully!")
            print(f"📁 Backup file: {backup_path}")
            print(f"📊 File size: {os.path.getsize(backup_path) / 1024 / 1024:.2f} MB")
        else:
            print(failed("Backup creation failed!")))

        await db_manager.close()

    except Exception as e:
        system_print(failed("Backup failed: {e}")), exc_info=True)
        print(failed("Backup failed: {e}")))


async def list_migrations(db_path: str = "data/ares_local_db.sqlite"):
    """List all available migrations."""
    # logger = system_logger.getChild("MigrationScript")  # Unused variable removed

    try:
        db_manager = SQLiteManager(db_path)
        await db_manager.initialize()

        migration_utils = DatabaseMigrationUtils(db_manager)
        migrations = await migration_utils.list_migrations()

        if not migrations:
            print("📋 No migrations found.")
            return

        print("📋 Available Migrations:")
        print("-" * 80)

        for migration in migrations:
            migration_id = migration.get("migration_id", "Unknown")
            migration_type = migration.get("migration_type", "Unknown")
            status = migration.get("status", "Unknown")
            created_at = migration.get("created_at", "Unknown")
            file_size = migration.get("file_size", 0)

            print(f"🆔 ID: {migration_id}")
            print(f"📝 Type: {migration_type}")
            print(f"📊 Status: {status}")
            print(f"📅 Created: {created_at}")
            print(f"📁 Size: {file_size / 1024 / 1024:.2f} MB")
            print("-" * 80)

        await db_manager.close()

    except Exception as e:
        system_print(failed("Failed to list migrations: {e}")), exc_info=True)
        print(failed("Failed to list migrations: {e}")))


async def cleanup_migrations(db_path: str = "data/ares_local_db.sqlite"):
    """Clean up old migrations."""
    # logger = system_logger.getChild("MigrationScript")  # Unused variable removed

    try:
        db_manager = SQLiteManager(db_path)
        await db_manager.initialize()

        migration_utils = DatabaseMigrationUtils(db_manager)
        await migration_utils.cleanup_old_migrations()

        print("✅ Cleanup completed!")

        await db_manager.close()

    except Exception as e:
        system_print(failed("Cleanup failed: {e}")), exc_info=True)
        print(failed("Cleanup failed: {e}")))


def print_usage():
    """Print usage information."""
    print(__doc__)
    print("\nExamples:")
    print("  # Export database from backtesting computer")
    print("  python scripts/database_migration.py export")
    print()
    print("  # Import database on trading computer")
    print(
        "  python scripts/database_migration.py import data/migrations/trading_export_20231201_143022.sqlite",
    )
    print()
    print("  # Validate a migration file")
    print(
        "  python scripts/database_migration.py validate data/migrations/trading_export_20231201_143022.sqlite",
    )
    print()
    print("  # Create a backup")
    print("  python scripts/database_migration.py backup")
    print()
    print("  # List all migrations")
    print("  python scripts/database_migration.py list-migrations")
    print()
    print("  # Clean up old migrations")
    print("  python scripts/database_migration.py cleanup")


async def main():
    """Main function."""
    # Setup logging
    setup_logging()

    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]

    if command == "export":
        db_path = sys.argv[2] if len(sys.argv) > 2 else "data/ares_local_db.sqlite"
        await export_database(db_path)

    elif command == "import":
        if len(sys.argv) < 3:
            print(warning("Import path required")))
            print_usage()
            sys.exit(1)
        import_path = sys.argv[2]
        db_path = sys.argv[3] if len(sys.argv) > 3 else "data/ares_local_db.sqlite"
        await import_database(import_path, db_path)

    elif command == "validate":
        if len(sys.argv) < 3:
            print(warning("File path required")))
            print_usage()
            sys.exit(1)
        file_path = sys.argv[2]
        await validate_file(file_path)

    elif command == "backup":
        db_path = sys.argv[2] if len(sys.argv) > 2 else "data/ares_local_db.sqlite"
        await create_backup(db_path)

    elif command == "list-migrations":
        db_path = sys.argv[2] if len(sys.argv) > 2 else "data/ares_local_db.sqlite"
        await list_migrations(db_path)

    elif command == "cleanup":
        db_path = sys.argv[2] if len(sys.argv) > 2 else "data/ares_local_db.sqlite"
        await cleanup_migrations(db_path)

    else:
        print(warning("Unknown command: {command}")))
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
