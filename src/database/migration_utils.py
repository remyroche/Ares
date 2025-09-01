# src/database/migration_utils.py

import hashlib
import os
import shutil
from datetime import datetime, timedelta

from src.database.sqlite_manager import SQLiteManager
from src.utils.logger import system_logger


class DatabaseMigrationUtils:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="databasemigrationutils initialization",
    )
    async def initialize(self) -> bool:
        """Initialize DatabaseMigrationUtils."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    pass"""
    Utility class for managing database migrations between computers.
    Handles export, import, validation, and backup operations.
    """

    def __init__(...):
    passpassself.db_manager = db_manager
        self.logger = system_logger.getChild("MigrationUtils")

    async def export_for_trading(...) -> ...:
    """..."""
    passif not export_name:
    passexport_name = f"trading_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        export_path = os.path.join(
            self.db_manager.migration_dir,
            f"{export_name}.sqlite",
        )

        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            # Create a clean copy for trading export
            shutil.copy2(self.db_manager.db_path, export_path)

            # Create a temporary SQLite manager for the export
            temp_db = SQLiteManager(export_path)
            await temp_db.initialize()

            # Remove backtest-specific data that shouldn't be on trading computer
            await self._clean_for_trading(temp_db)

            # Calculate checksum
            with open(export_path, "rb") as f:
    passpasschecksum = hashlib.md5(f.read()).hexdigest()

            # Record export
            export_data = {
                "export_id": export_name,
                "source_computer": os.uname().nodename if hasattr(os, "uname") else "unknown",
                "export_type": "trading_export",
                "status": "created",
                "created_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(export_path),
                "checksum": checksum,
                "description": "Database export for trading computer",
            }

            await self.db_manager.set_document(
                "database_migrations",
                export_name,
                export_data,
            )

            self.logger.info(
                f"Trading export created: {export_path} (checksum: {checksum})",
            )
            return export_path

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Failed to create trading export: {e}")
            return ""

    async def _clean_for_trading(...):
    pass"""Removes backtest-specific data from the export."""
        try:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            # Remove backtest results (keep only the latest successful ones)
            backtest_results = await temp_db.get_collection("backtest_results")
            if len(backtest_results) > 1:
    pass# Keep only the most recent successful backtest
                sorted_results = sorted(
                    backtest_results.items(),
                    key=lambda x: x[1].get("created_at", ""),
                    reverse=True
                )
                
                # Remove all but the most recent
                for result_id, _ in sorted_results[1:]:
                    await temp_db.delete_document("backtest_results", result_id)
                    
        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error cleaning trading export: {e}")
