# Database Migration System

This document explains how to use the new SQLite-based database system for running backtesting on one computer and trading on another.

## Overview

The Ares Trading Bot now uses SQLite as its primary database, with built-in migration capabilities to transfer data between computers. This allows you to:

1. **Run backtesting** on one computer (e.g., a powerful desktop)
2. **Export the database** with optimized parameters and models
3. **Import the database** on another computer (e.g., a dedicated trading server)
4. **Run live trading** with the optimized parameters

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐
│ Backtesting     │    │ Trading         │
│ Computer        │    │ Computer        │
│                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ SQLite DB   │ │    │ │ SQLite DB   │ │
│ │ (Full Data) │ │    │ │ (Trading    │ │
│ │             │ │    │ │  Data Only) │ │
│ └─────────────┘ │    │ └─────────────┘ │
│         │       │    │         │       │
│         ▼       │    │         │       │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Export      │ │    │ │ Import      │ │
│ │ Script      │ │    │ │ Script      │ │
│ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘
         │                       ▲
         └───────────────────────┘
                    Migration File
```

## Database Structure

The SQLite database contains the following tables:

- **`ares_optimized_params`** - Optimized trading parameters from backtesting
- **`ares_live_metrics`** - Live trading metrics and performance data
- **`ares_alerts`** - System alerts and notifications
- **`detailed_trade_logs`** - Complete trade history and analysis
- **`model_checkpoints`** - Trained model checkpoints and versions
- **`backtest_results`** - Backtest results and performance metrics
- **`trading_state`** - Current trading state and configuration
- **`database_migrations`** - Migration history and metadata

## Workflow

### 1. Backtesting Computer Setup

1. **Install and configure** the Ares bot on your backtesting computer
2. **Run backtesting** to optimize parameters and train models
3. **Export the database** when ready for live trading

```bash
# Export database for trading computer
python scripts/database_migration.py export
```

This creates a migration file in `data/migrations/` that contains:
- Optimized trading parameters
- Trained model checkpoints
- Historical performance data
- Trading configuration

### 2. Trading Computer Setup

1. **Install the Ares bot** on your trading computer
2. **Copy the migration file** from the backtesting computer
3. **Import the database**

```bash
# Import database on trading computer
python scripts/database_migration.py import data/migrations/trading_export_20231201_143022.sqlite
```

### 3. Validation

Before importing, you can validate the migration file:

```bash
# Validate migration file
python scripts/database_migration.py validate data/migrations/trading_export_20231201_143022.sqlite
```

## Database Backup and Persistence

### Automatic Backups

The system automatically creates backups every 24 hours (configurable):

- **Location**: `data/backups/`
- **Format**: `backup_YYYYMMDD_HHMMSS.sqlite`
- **Retention**: 30 days (configurable)

### Manual Backups

Create manual backups when needed:

```bash
# Create manual backup
python scripts/database_migration.py backup
```

### Crash Recovery

The system is designed to handle crashes gracefully:

1. **Automatic recovery** - Database connections are restored on restart
2. **Backup restoration** - Can restore from any backup point
3. **Data integrity** - SQLite transactions ensure data consistency

## Migration Scripts

### Export Database

```bash
# Export from backtesting computer
python scripts/database_migration.py export [db_path]
```

**Output**: Migration file in `data/migrations/`

### Import Database

```bash
# Import on trading computer
python scripts/database_migration.py import <import_path> [db_path]
```

**Input**: Migration file from backtesting computer

### Validate Migration File

```bash
# Validate before import
python scripts/database_migration.py validate <file_path>
```

**Checks**:
- File exists and is not empty
- Valid SQLite database format
- Required tables present
- Checksum verification

### List Migrations

```bash
# List all migrations
python scripts/database_migration.py list-migrations [db_path]
```

### Cleanup Old Migrations

```bash
# Clean up old migration files
python scripts/database_migration.py cleanup [db_path]
```

## Configuration

### Database Settings

Edit `src/config.py` to configure:

```python
CONFIG = {
    # Database Configuration
    "DATABASE_TYPE": "sqlite",
    "SQLITE_DB_PATH": "data/ares_local_db.sqlite",
    "BACKUP_INTERVAL_HOURS": 24,
    "BACKUP_RETENTION_DAYS": 30,
    
    # SQLite Configuration
    "sqlite": {
        "enabled": True,
        "backup_enabled": True,
        "migration_enabled": True,
        # ... other settings
    }
}
```

### Directory Structure

```
data/
├── ares_local_db.sqlite          # Main database
├── backups/                      # Automatic backups
│   ├── backup_20231201_120000.sqlite
│   └── ...
└── migrations/                   # Migration files
    ├── trading_export_20231201_143022.sqlite
    └── ...
```

## Security Considerations

### File Permissions

Ensure proper file permissions:

```bash
# Set secure permissions
chmod 600 data/ares_local_db.sqlite
chmod 700 data/backups/
chmod 700 data/migrations/
```

### Network Security

When transferring migration files:

1. **Use secure transfer** (SCP, SFTP, or encrypted USB)
2. **Verify checksums** before import
3. **Validate files** before use

### Backup Security

- **Encrypt backups** if containing sensitive data
- **Store backups** in secure location
- **Test restoration** periodically

## Troubleshooting

### Common Issues

1. **Import fails**
   - Check file permissions
   - Verify file integrity
   - Ensure sufficient disk space

2. **Database corruption**
   - Restore from latest backup
   - Check disk health
   - Verify SQLite installation

3. **Migration validation fails**
   - Check file format
   - Verify file size
   - Ensure all required tables present

### Logs

Check logs for detailed error information:

```bash
# View recent logs
tail -f logs/ares.log
```

### Recovery Procedures

1. **Database corruption**:
   ```bash
   # Restore from backup
   python scripts/database_migration.py restore backup_file.sqlite
   ```

2. **Migration failure**:
   ```bash
   # Validate migration file
   python scripts/database_migration.py validate migration_file.sqlite
   ```

3. **Import issues**:
   ```bash
   # Check database status
   python scripts/database_migration.py list-migrations
   ```

## Performance Considerations

### Database Size

- **Typical size**: 10-100 MB
- **Backup size**: Similar to main database
- **Migration size**: Optimized for transfer

### Optimization

- **Regular cleanup** of old migrations
- **Compression** for large databases
- **Incremental backups** for efficiency

## Migration Best Practices

1. **Test migrations** on non-production systems first
2. **Keep multiple backups** before major changes
3. **Validate all imports** before use
4. **Document migration history** for troubleshooting
5. **Monitor disk space** for backups and migrations

## Integration with Trading Bot

The migration system integrates seamlessly with the trading bot:

1. **Automatic initialization** on startup
2. **Background backup scheduling**
3. **State persistence** across restarts
4. **Error recovery** and logging

The bot automatically uses the imported database for:
- Trading parameters
- Model predictions
- Performance tracking
- Risk management

This ensures consistent behavior between backtesting and live trading environments. 