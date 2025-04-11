"""
Migration utilities for schema evolution
"""
from typing import Dict, List, Any, Optional
import os
import time
import json
from datetime import datetime
import asyncio
from .repository import DuckDbRepository


class Migration:
    """Base migration class"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.created_at = datetime.now().isoformat()
    
    async def up(self, db: DuckDbRepository):
        """Apply migration"""
        raise NotImplementedError("Subclasses must implement up()")
    
    async def down(self, db: DuckDbRepository):
        """Revert migration"""
        raise NotImplementedError("Subclasses must implement down()")


class MigrationManager:
    """Manager for handling database migrations"""
    
    def __init__(self, db: DuckDbRepository):
        self.db = db
        self.migrations_table = "_migrations"
        self.initialized = False
    
    async def init(self):
        """Initialize migrations table"""
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.migrations_table} (
            id INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL,
            version VARCHAR NOT NULL,
            applied_at TIMESTAMP NOT NULL
        )
        """
        self.db.execute(sql)
        self.initialized = True
    
    async def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Get list of already applied migrations"""
        if not self.initialized:
            await self.init()
        
        sql = f"SELECT id, name, version, applied_at FROM {self.migrations_table} ORDER BY id"
        results = self.db.execute_and_fetch(sql)
        
        return [
            {
                "id": row[0],
                "name": row[1],
                "version": row[2],
                "applied_at": row[3]
            }
            for row in results
        ]
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration"""
        if not self.initialized:
            await self.init()
        
        # Check if migration was already applied
        sql = f"SELECT 1 FROM {self.migrations_table} WHERE name = :name AND version = :version"
        params = {"name": migration.name, "version": migration.version}
        result = self.db.execute_and_fetch(sql, params)
        
        if result:
            print(f"Migration {migration.name} (v{migration.version}) already applied")
            return False
        
        # Run migration
        try:
            self.db.begin_transaction()
            
            # Apply migration
            await migration.up(self.db)
            
            # Record migration
            sql = f"""
            INSERT INTO {self.migrations_table} (name, version, applied_at)
            VALUES (:name, :version, CURRENT_TIMESTAMP)
            """
            params = {"name": migration.name, "version": migration.version}
            self.db.execute(sql, params)
            
            self.db.commit()
            print(f"Applied migration {migration.name} (v{migration.version})")
            return True
        except Exception as e:
            self.db.rollback()
            print(f"Migration {migration.name} (v{migration.version}) failed: {str(e)}")
            raise
    
    async def revert_migration(self, migration: Migration) -> bool:
        """Revert a single migration"""
        if not self.initialized:
            await self.init()
        
        # Check if migration was applied
        sql = f"SELECT 1 FROM {self.migrations_table} WHERE name = :name AND version = :version"
        params = {"name": migration.name, "version": migration.version}
        result = self.db.execute_and_fetch(sql, params)
        
        if not result:
            print(f"Migration {migration.name} (v{migration.version}) was not applied")
            return False
        
        # Revert migration
        try:
            self.db.begin_transaction()
            
            # Apply down migration
            await migration.down(self.db)
            
            # Remove migration record
            sql = f"""
            DELETE FROM {self.migrations_table} 
            WHERE name = :name AND version = :version
            """
            params = {"name": migration.name, "version": migration.version}
            self.db.execute(sql, params)
            
            self.db.commit()
            print(f"Reverted migration {migration.name} (v{migration.version})")
            return True
        except Exception as e:
            self.db.rollback()
            print(f"Revert of migration {migration.name} (v{migration.version}) failed: {str(e)}")
            raise
    
    async def apply_migrations(self, migrations: List[Migration]) -> int:
        """Apply multiple migrations in order"""
        count = 0
        for migration in migrations:
            if await self.apply_migration(migration):
                count += 1
        return count
    
    async def revert_migrations(self, migrations: List[Migration]) -> int:
        """Revert multiple migrations in reverse order"""
        count = 0
        for migration in reversed(migrations):
            if await self.revert_migration(migration):
                count += 1
        return count
    
    @staticmethod
    def create_migration_file(name: str, directory: str = "migrations") -> str:
        """Create a new migration file"""
        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)
        
        # Generate timestamp and filename
        timestamp = int(time.time())
        version = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{version}_{name}.py"
        filepath = os.path.join(directory, filename)
        
        # Create migration file
        template = f'''"""
Migration: {name}
Version: {version}
Created: {datetime.now().isoformat()}
"""
from duckdb_tinyorm.migration import Migration
from duckdb_tinyorm.repository import DuckDbRepository


class {name.title().replace("_", "")}Migration(Migration):
    
    def __init__(self):
        super().__init__("{name}", "{version}")
    
    async def up(self, db: DuckDbRepository):
        """Apply migration"""
        # TODO: Implement migration
        db.execute("""
        -- Your migration SQL here
        """)
    
    async def down(self, db: DuckDbRepository):
        """Revert migration"""
        # TODO: Implement revert logic
        db.execute("""
        -- Your rollback SQL here
        """)

'''
        
        with open(filepath, "w") as f:
            f.write(template)
        
        print(f"Created migration file: {filepath}")
        return filepath