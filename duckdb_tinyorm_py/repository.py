"""
Repository module for database operations with extended ORM features
"""
from typing import TypeVar, Generic, Dict, Any, List, Optional, Type, Union, Callable
import duckdb
import pandas as pd
import numpy as np
import json
from datetime import datetime, date
import asyncio
from .decorators import get_entity_metadata
from .config import DuckDbLocation, DuckDbConfig
from .exceptions import EntityNotFoundError, InvalidQueryError

T = TypeVar('T')
K = TypeVar('K')


class DuckDbRepository:
    """DuckDB repository for managing database connections"""
    
    _instances = {}
    
    @classmethod
    def get_instance(cls, config: DuckDbConfig = None, **kwargs):
        """Get or create a DuckDB repository instance"""
        if config is None:
            name = kwargs.get('name', 'default')
            location = kwargs.get('location', DuckDbLocation.MEMORY)
            filename = kwargs.get('filename', None)
            config = DuckDbConfig(name=name, location=location, filename=filename)
        
        if config.name not in cls._instances:
            cls._instances[config.name] = DuckDbRepository(config)
        
        return cls._instances[config.name]
    
    def __init__(self, config: DuckDbConfig):
        """Initialize the repository with configuration"""
        self.config = config
        self.con = None
        self._init_connection()
    
    def _init_connection(self):
        """Initialize the DuckDB connection"""
        if self.config.location == DuckDbLocation.MEMORY:
            self.con = duckdb.connect(database=':memory:')
        else:
            self.con = duckdb.connect(database=self.config.filename)
        
        # Apply any custom settings
        if hasattr(self.config, 'settings') and self.config.settings:
            for setting, value in self.config.settings.items():
                self.con.execute(f"SET {setting}={value}")
    
    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None):
        """Execute a SQL query"""
        try:
            if params:
                return self.con.execute(sql, params)
            return self.con.execute(sql)
        except Exception as e:
            raise InvalidQueryError(f"Error executing query: {sql}. Error: {str(e)}")
    
    def execute_and_fetch(self, sql: str, params: Optional[Dict[str, Any]] = None):
        """Execute a SQL query and fetch results"""
        try:
            if params:
                return self.con.execute(sql, params).fetchall()
            return self.con.execute(sql).fetchall()
        except Exception as e:
            raise InvalidQueryError(f"Error executing query: {sql}. Error: {str(e)}")
    
    def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Run a query and return a pandas DataFrame"""
        try:
            if params:
                return self.con.execute(sql, params).df()
            return self.con.execute(sql).df()
        except Exception as e:
            raise InvalidQueryError(f"Error executing query: {sql}. Error: {str(e)}")
    
    def query_arrow(self, sql: str, params: Optional[Dict[str, Any]] = None):
        """Run a query and return an Arrow table"""
        try:
            if params:
                return self.con.execute(sql, params).arrow()
            return self.con.execute(sql).arrow()
        except Exception as e:
            raise InvalidQueryError(f"Error executing query: {sql}. Error: {str(e)}")
    
    def close(self):
        """Close the database connection"""
        if self.con:
            self.con.close()
            self.con = None
    
    def begin_transaction(self):
        """Begin a transaction"""
        self.con.execute("BEGIN TRANSACTION")
    
    def commit(self):
        """Commit the current transaction"""
        self.con.execute("COMMIT")
    
    def rollback(self):
        """Rollback the current transaction"""
        self.con.execute("ROLLBACK")
    
    def __enter__(self):
        """Context manager entry"""
        self.begin_transaction()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is not None:
            self.rollback()
        else:
            try:
                self.commit()
            except:
                self.rollback()
                raise


class QueryBuilder:
    """SQL query builder to simplify complex query construction"""
    
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.select_fields = ["*"]
        self.where_clauses = []
        self.order_by_clauses = []
        self.group_by_clauses = []
        self.having_clauses = []
        self.limit_value = None
        self.offset_value = None
        self.params = {}
        self.join_clauses = []
    
    def select(self, *fields):
        """Select specific fields"""
        if fields and fields[0] != "*":
            self.select_fields = list(fields)
        return self
    
    def where(self, field, operator, value=None):
        """Add a WHERE condition"""
        if value is None:
            value = operator
            operator = "="
        
        # Use ? placeholder instead of named parameter
        self.where_clauses.append(f"{field} {operator} ?")
        # Still collect values in order
        self.params[f"param_{len(self.params)}"] = value
        return self
    
    def and_where(self, field, operator, value=None):
        """Add an AND WHERE condition"""
        return self.where(field, operator, value)
    
    def or_where(self, field, operator, value=None):
        """Add an OR WHERE condition"""
        if not self.where_clauses:
            return self.where(field, operator, value)
        
        param_name = f"param_{len(self.params)}"
        
        if value is None:
            value = operator
            operator = "="
        
        # Convert the last condition to use OR
        last_clause = self.where_clauses.pop()
        self.where_clauses.append(f"({last_clause} OR {field} {operator} :{param_name})")
        self.params[param_name] = value
        return self
    
    def where_in(self, field, values):
        """Add a WHERE IN condition"""
        if not values:
            self.where_clauses.append("1 = 0") # Always false
            return self
        
        param_names = []
        for value in values:
            param_name = f"param_{len(self.params)}"
            param_names.append(f":{param_name}")
            self.params[param_name] = value
        
        placeholders = ", ".join(param_names)
        self.where_clauses.append(f"{field} IN ({placeholders})")
        return self
    
    def order_by(self, field, direction="ASC"):
        """Add ORDER BY clause"""
        self.order_by_clauses.append(f"{field} {direction}")
        return self
    
    def group_by(self, *fields):
        """Add GROUP BY clause"""
        self.group_by_clauses.extend(fields)
        return self
    
    def having(self, condition):
        """Add HAVING clause"""
        self.having_clauses.append(condition)
        return self
    
    def limit(self, limit):
        """Add LIMIT clause"""
        self.limit_value = limit
        return self
    
    def offset(self, offset):
        """Add OFFSET clause"""
        self.offset_value = offset
        return self
    
    def join(self, table, condition, join_type="INNER"):
        """Add JOIN clause"""
        self.join_clauses.append(f"{join_type} JOIN {table} ON {condition}")
        return self
    
    def left_join(self, table, condition):
        """Add LEFT JOIN clause"""
        return self.join(table, condition, "LEFT")
    
    def right_join(self, table, condition):
        """Add RIGHT JOIN clause"""
        return self.join(table, condition, "RIGHT")
    
    def build(self):
        """Build the SQL query"""
        select_clause = f"SELECT {', '.join(self.select_fields)}"
        from_clause = f"FROM {self.table_name}"
        
        # Add JOIN clauses
        join_clause = " ".join(self.join_clauses) if self.join_clauses else ""
        
        # Add WHERE clause
        where_clause = ""
        if self.where_clauses:
            where_clause = "WHERE " + " AND ".join(self.where_clauses)
        
        # Add GROUP BY clause
        group_by_clause = ""
        if self.group_by_clauses:
            group_by_clause = "GROUP BY " + ", ".join(self.group_by_clauses)
        
        # Add HAVING clause
        having_clause = ""
        if self.having_clauses:
            having_clause = "HAVING " + " AND ".join(self.having_clauses)
        
        # Add ORDER BY clause
        order_by_clause = ""
        if self.order_by_clauses:
            order_by_clause = "ORDER BY " + ", ".join(self.order_by_clauses)
        
        # Add LIMIT and OFFSET
        limit_clause = f"LIMIT {self.limit_value}" if self.limit_value is not None else ""
        offset_clause = f"OFFSET {self.offset_value}" if self.offset_value is not None else ""
        
        # Build the final query
        clauses = [
            select_clause,
            from_clause,
            join_clause,
            where_clause,
            group_by_clause,
            having_clause,
            order_by_clause,
            limit_clause,
            offset_clause
        ]
        
        query = " ".join(clause for clause in clauses if clause)
        
        # Return ordered parameter values instead of the param dict
        param_values = [self.params[f"param_{i}"] for i in range(len(self.params))]
        return query, param_values


class BaseRepository(Generic[T, K]):
    """Base repository for entity operations with enhanced functionality"""
    
    def __init__(self, db_repository: DuckDbRepository = None):
        """Initialize the repository with an optional database connection"""
        if not hasattr(self.__class__, '_entity_class'):
            raise ValueError("Repository must be decorated with @repository(EntityClass)")
        
        self.entity_class = self.__class__._entity_class
        self.entity_name = self.entity_class.__name__
        self.db = db_repository or DuckDbRepository.get_instance()
        self.entity_meta = get_entity_metadata(self.entity_class)
        
        if not self.entity_meta:
            raise ValueError(f"Entity {self.entity_name} is not properly decorated with @entity")
        
        self.table_name = self.entity_meta['table_name']
        self.id_field = self.entity_meta['id_field']
        
        if not self.id_field:
            raise ValueError(f"Entity {self.entity_name} must have an @id_field decorator")
    
    async def init(self, drop_if_exists: bool = False):
        """Initialize the repository - create tables if they don't exist"""
        if drop_if_exists:
            # If there's a sequence, drop it first
            self.db.execute(f"DROP SEQUENCE IF EXISTS seq_{self.table_name}")
            drop_sql = f"DROP TABLE IF EXISTS {self.table_name}"
            self.db.execute(drop_sql)
        
        # Create a sequence for auto-incrementing IDs
        if any(field_meta.get('auto_increment', False) for field_meta in self.entity_meta['fields'].values()):
            self.db.execute(f"CREATE SEQUENCE IF NOT EXISTS seq_{self.table_name}")
        
        fields = []
        for field_name, field_meta in self.entity_meta['fields'].items():
            field_type = field_meta['type']
            field_def = f"{field_name} {field_type}"
            
            is_id = field_meta.get('is_id', False)
            is_auto_increment = field_meta.get('auto_increment', False)

            if is_id:
                # For DuckDB, just use PRIMARY KEY for auto-incrementing fields
                # DuckDB will automatically make INTEGER PRIMARY KEY auto-incrementing
                field_def += " PRIMARY KEY"
                
                # Add NOT NULL constraint to primary key if not already implied
                if not "NOT NULL" in field_def:
                    field_def += " NOT NULL"
            else: # Add other constraints only if not the primary key
                if field_meta.get('not_null', False):
                    field_def += " NOT NULL"
                if field_meta.get('unique', False):
                    field_def += " UNIQUE"
                if field_meta.get('default') is not None:
                    default_val = field_meta['default']
                    if isinstance(default_val, str):
                        # Ensure proper quoting for string defaults
                        escaped = default_val.replace("'", "''")
                        default_val = f"'{escaped}'"
                    elif isinstance(default_val, bool):
                        # Use TRUE/FALSE for boolean defaults
                        default_val = 'TRUE' if default_val else 'FALSE'
                    # Add other type handling for defaults if necessary
                    field_def += f" DEFAULT {default_val}"
            
            fields.append(field_def)
        
        fields_str = ', '.join(fields)
        sql = f"CREATE TABLE IF NOT EXISTS {self.table_name} ({fields_str})"
        self.db.execute(sql)
        
        # Create any indexes defined for this entity
        if 'indexes' in self.entity_meta:
            for index in self.entity_meta['indexes']:
                index_name = index.get('name', f"idx_{self.table_name}_{'_'.join(index['fields'])}") # Fixed index name generation
                index_fields = ", ".join(index['fields'])
                index_type = index.get('type', '') # Removed extra space
                unique = "UNIQUE " if index.get('unique', False) else ""
                
                # Corrected index creation syntax (removed index_type from main part)
                index_sql = f"CREATE {unique}INDEX IF NOT EXISTS {index_name} ON {self.table_name} ({index_fields})"
                # If index_type like USING is needed, adjust syntax: e.g., USING {index_type} ({index_fields})
                self.db.execute(index_sql)
    
    def _entity_to_dict(self, entity: T) -> Dict[str, Any]:
        """Convert an entity to a dictionary for database operations"""
        result = {}
        for field_name in self.entity_meta['fields']:
            value = getattr(entity, field_name, None)
            # Handle serialization of complex types
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif isinstance(value, (datetime, date)):
                value = value.isoformat()
            
            result[field_name] = value
        return result
    
    def _row_to_entity(self, row, field_names=None) -> T:
        """Convert a database row to an entity"""
        if not field_names:
            field_names = list(self.entity_meta['fields'].keys())
        
        entity = self.entity_class()
        
        if isinstance(row, tuple):
            for i, field_name in enumerate(field_names):
                if i >= len(row):
                    continue
                    
                value = row[i]
                field_meta = self.entity_meta['fields'].get(field_name, {})
                field_type = field_meta.get('python_type')
                
                # Convert to appropriate Python type if needed
                if field_type == 'dict' and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except:
                        pass
                
                setattr(entity, field_name, value)
        elif isinstance(row, dict):
            for field_name in field_names:
                if field_name in row:
                    value = row[field_name]
                    field_meta = self.entity_meta['fields'].get(field_name, {})
                    field_type = field_meta.get('python_type')
                    
                    # Convert to appropriate Python type if needed
                    if field_type == 'dict' and isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except:
                            pass
                    
                    setattr(entity, field_name, value)
        
        return entity
    
    async def save(self, entity: T) -> T:
        """Save an entity - insert or update if it exists"""
        id_value = getattr(entity, self.id_field, None)
        id_field_meta = self.entity_meta['fields'].get(self.id_field, {})
        is_auto_increment = id_field_meta.get('auto_increment', False)

        # If ID is None AND it's an auto-increment field, perform an insert directly
        if id_value is None and is_auto_increment:
            # Note: The current insert doesn't retrieve the generated ID.
            # For auto-increment, we assume insert is the correct action.
            return await self.insert(entity)
        
        # If ID is None but NOT auto-increment, it's an error (as before)
        if id_value is None and not is_auto_increment:
             raise ValueError(f"Entity {self.entity_name} must have a value for non-auto-increment ID field {self.id_field}")

        # If ID has a value, check if it exists to decide between update and insert
        exists = await self.exists_by_id(id_value)
        
        if exists:
            return await self.update(entity)
        else:
            # Insert even if ID is provided but doesn't exist yet
            return await self.insert(entity)

    async def save_all(self, entities: List[T]) -> List[T]:
        """Save multiple entities in a single transaction"""
        if not entities:
            return []
        
        results = []
        
        try:
            self.db.begin_transaction()
            
            for entity in entities:
                result = await self.save(entity)
                results.append(result)
            
            self.db.commit()
            return results
        except Exception as e:
            self.db.rollback()
            raise e
    
    async def insert(self, entity: T) -> T:
        """Insert a new entity"""
        entity_dict = self._entity_to_dict(entity)
        
        # Check if we need to generate an ID
        id_field_meta = self.entity_meta['fields'].get(self.id_field, {})
        is_auto_increment = id_field_meta.get('auto_increment', False)
        id_value = entity_dict.get(self.id_field)
        
        # If auto-increment and no ID provided, get one from sequence
        if is_auto_increment and (id_value is None or id_value == 0):
            seq_result = self.db.execute_and_fetch(f"SELECT nextval('seq_{self.table_name}')")
            new_id = seq_result[0][0]
            entity_dict[self.id_field] = new_id
            # Also update the entity object
            setattr(entity, self.id_field, new_id)
        
        # Now continue with the insert operation as before
        field_names = []
        field_placeholders = []
        param_values = []
        
        for field_name in self.entity_meta['fields']:
            if field_name in entity_dict:
                field_names.append(field_name)
                field_placeholders.append("?")
                param_values.append(entity_dict[field_name])
        
        fields_str = ', '.join(field_names)
        placeholders_str = ', '.join(field_placeholders)
        
        if not field_names:
            raise ValueError("Cannot insert entity with no fields specified")
        
        sql = f"INSERT INTO {self.table_name} ({fields_str}) VALUES ({placeholders_str})"
        self.db.execute(sql, param_values)
        
        return entity
    
    async def bulk_insert(self, entities: List[T]) -> List[T]:
        """Insert multiple entities in a single transaction"""
        if not entities:
            return []
        
        try:
            self.db.begin_transaction()
            
            for entity in entities:
                await self.insert(entity)
            
            self.db.commit()
            return entities
        except Exception as e:
            self.db.rollback()
            raise e
    
    async def update(self, entity: T) -> T:
        """Update an existing entity"""
        entity_dict = self._entity_to_dict(entity)
        id_value = entity_dict[self.id_field]
        
        field_updates = []
        field_values = {}
        
        for field_name, value in entity_dict.items():
            if field_name != self.id_field:
                field_updates.append(f"{field_name} = :{field_name}")
                field_values[field_name] = value
        
        field_values[self.id_field] = id_value
        updates_str = ', '.join(field_updates)
        
        sql = f"UPDATE {self.table_name} SET {updates_str} WHERE {self.id_field} = :{self.id_field}"
        self.db.execute(sql, field_values)
        
        return entity
    
    async def exists_by_id(self, id_value: K) -> bool:
        """Check if an entity exists by ID"""
        sql = f"SELECT 1 FROM {self.table_name} WHERE {self.id_field} = :{self.id_field} LIMIT 1"
        result = self.db.execute_and_fetch(sql, {self.id_field: id_value})
        return len(result) > 0
    
    async def find_by_id(self, id_value: K) -> Optional[T]:
        """Find an entity by ID"""
        sql = f"SELECT * FROM {self.table_name} WHERE {self.id_field} = :{self.id_field} LIMIT 1"
        result = self.db.execute_and_fetch(sql, {self.id_field: id_value})
        
        if not result:
            return None
        
        return self._row_to_entity(result[0])
    
    async def find_by_id_or_error(self, id_value: K) -> T:
        """Find an entity by ID or throw an error if not found"""
        entity = await self.find_by_id(id_value)
        if entity is None:
            raise EntityNotFoundError(f"Entity {self.entity_name} with ID {id_value} not found")
        return entity
    
    async def find_all(self) -> List[T]:
        """Find all entities"""
        sql = f"SELECT * FROM {self.table_name}"
        results = self.db.execute_and_fetch(sql)
        
        return [self._row_to_entity(row) for row in results]
    
    async def find_all_paged(self, page: int = 1, page_size: int = 10) -> Dict[str, Any]:
        """Find all entities with pagination"""
        offset = (page - 1) * page_size
        
        count_sql = f"SELECT COUNT(*) FROM {self.table_name}"
        total = self.db.execute_and_fetch(count_sql)[0][0]
        
        sql = f"SELECT * FROM {self.table_name} LIMIT {page_size} OFFSET {offset}"
        results = self.db.execute_and_fetch(sql)
        
        entities = [self._row_to_entity(row) for row in results]
        
        return {
            "data": entities,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size,
            "has_next": page * page_size < total,
            "has_prev": page > 1
        }
    
    async def find_by(self, criteria: Dict[str, Any], fields: Optional[List[str]] = None) -> List[T]:
        """Find entities by criteria"""
        query_builder = QueryBuilder(self.table_name)
        
        if fields:
            query_builder.select(*fields)
        
        for field_name, value in criteria.items():
            query_builder.where(field_name, "=", value)
        
        sql, params = query_builder.build()
        results = self.db.execute_and_fetch(sql, params)
        
        field_names = fields or list(self.entity_meta['fields'].keys())
        return [self._row_to_entity(row, field_names) for row in results]
    
    async def find_one_by(self, criteria: Dict[str, Any]) -> Optional[T]:
        """Find a single entity by criteria"""
        query_builder = QueryBuilder(self.table_name)
        
        for field_name, value in criteria.items():
            query_builder.where(field_name, "=", value)
        
        query_builder.limit(1)
        sql, params = query_builder.build()
        
        results = self.db.execute_and_fetch(sql, params)
        
        if not results:
            return None
        
        return self._row_to_entity(results[0])
    
    def query(self) -> QueryBuilder:
        """Get a query builder for complex queries"""
        return QueryBuilder(self.table_name)
    
    async def execute_query(self, query: 'QueryBuilder') -> List[T]:
        """Execute a custom query and return entities"""
        sql, params = query.build()
        results = self.db.execute_and_fetch(sql, params)
        
        # Extract field names from the query's select fields if specified, else use all fields
        field_names = query.select_fields if query.select_fields and query.select_fields[0] != '*' else list(self.entity_meta['fields'].keys())
        
        # Convert rows to entities
        return [self._row_to_entity(row, field_names) for row in results]
    
    async def execute_raw_query(self, sql: str, params: Dict[str, Any] = None) -> List[T]:
        """Execute a raw SQL query and return entities"""
        results = self.db.execute_and_fetch(sql, params)
        return [self._row_to_entity(row) for row in results]
    
    async def remove_by_id(self, id_value: K) -> bool:
        """Remove an entity by ID"""
        sql = f"DELETE FROM {self.table_name} WHERE {self.id_field} = :{self.id_field}"
        self.db.execute(sql, {self.id_field: id_value})
        return True
    
    async def remove(self, entity: T) -> bool:
        """Remove an entity"""
        id_value = getattr(entity, self.id_field)
        return await self.remove_by_id(id_value)
    
    async def remove_all(self, criteria: Dict[str, Any] = None) -> int:
        """Remove entities by criteria or all if no criteria is provided"""
        if criteria:
            query_builder = QueryBuilder(self.table_name)
            
            for field_name, value in criteria.items():
                query_builder.where(field_name, "=", value)
            
            # Get the WHERE clause without the SELECT part
            sql_parts = query_builder.build()[0].split("WHERE")
            where_clause = f"WHERE {sql_parts[1]}" if len(sql_parts) > 1 else ""
            
            sql = f"DELETE FROM {self.table_name} {where_clause}"
            self.db.execute(sql, query_builder.params)
            # DuckDB doesn't have a way to return the number of deleted rows directly
            return 1  # Assuming success
        else:
            sql = f"DELETE FROM {self.table_name}"
            self.db.execute(sql)
            return 1  # Assuming success
    
    async def count(self, criteria: Dict[str, Any] = None) -> int:
        """Count entities by criteria or all if no criteria is provided"""
        if criteria:
            query_builder = QueryBuilder(self.table_name).select("COUNT(*) as count")
            
            for field_name, value in criteria.items():
                query_builder.where(field_name, "=", value)
            
            sql, params = query_builder.build()
        else:
            sql = f"SELECT COUNT(*) as count FROM {self.table_name}"
            params = {}
        
        result = self.db.execute_and_fetch(sql, params)
        return result[0][0]
    
    async def exists(self, criteria: Dict[str, Any]) -> bool:
        """Check if entities exist by criteria"""
        count = await self.count(criteria)
        return count > 0
    
    # Data export methods
    def to_dataframe(self, query_builder: Optional[QueryBuilder] = None) -> pd.DataFrame:
        """Export query results to a pandas DataFrame"""
        if query_builder:
            sql, params = query_builder.build()
        else:
            sql = f"SELECT * FROM {self.table_name}"
            params = {}
        
        return self.db.query(sql, params)
    
    def to_arrow(self, query_builder: Optional[QueryBuilder] = None):
        """Export query results to an Arrow table"""
        if query_builder:
            sql, params = query_builder.build()
        else:
            sql = f"SELECT * FROM {self.table_name}"
            params = {}
        
        return self.db.query_arrow(sql, params)
    
    async def to_json(self, path=None, query=None, orient='records', **kwargs):
        """Export entities to JSON"""
        # The to_dataframe method is synchronous, so don't use await
        df = self.to_dataframe(query) 
        
        # Convert DataFrame to JSON
        json_data = df.to_json(orient=orient, **kwargs)
        
        # If path is provided, save to file
        if path:
            with open(path, 'w') as f:
                f.write(json_data)
        
        return json_data
    
    async def to_dict_list(self, query_builder: Optional[QueryBuilder] = None) -> List[Dict[str, Any]]:
        """Export query results to a list of dictionaries"""
        if query_builder:
            sql, params = query_builder.build()
        else:
            sql = f"SELECT * FROM {self.table_name}"
            params = {}
        
        df = self.db.query(sql, params)
        return df.to_dict(orient="records")
    
    async def to_csv(self, query_builder: Optional[QueryBuilder] = None, 
                    path_or_buffer: Optional[str] = None, **kwargs) -> Optional[str]:
        """Export query results to CSV"""
        if query_builder:
            sql, params = query_builder.build()
        else:
            sql = f"SELECT * FROM {self.table_name}"
            params = {}
        
        df = self.db.query(sql, params)
        return df.to_csv(path_or_buffer=path_or_buffer, index=False, **kwargs)
    
    async def to_parquet(self, query_builder: Optional[QueryBuilder] = None, 
                        path: Optional[str] = None, **kwargs) -> Optional[bytes]:
        """Export query results to Parquet format"""
        if query_builder:
            sql, params = query_builder.build()
        else:
            sql = f"SELECT * FROM {self.table_name}"
            params = {}
        
        df = self.db.query(sql, params)
        if path:
            df.to_parquet(path, **kwargs)
            return None
        else:
            import io
            buffer = io.BytesIO()
            df.to_parquet(buffer, **kwargs)
            return buffer.getvalue()
    
    # Batch processing methods
    async def batch_process(self, 
                           batch_size: int = 1000, 
                           processor: Callable[[List[T]], Any] = None, 
                           criteria: Dict[str, Any] = None) -> List[Any]:
        """
        Process entities in batches to avoid memory issues with large data sets
        """
        results = []
        offset = 0
        has_more = True
        
        query_builder = QueryBuilder(self.table_name)
        if criteria:
            for field_name, value in criteria.items():
                query_builder.where(field_name, "=", value)
        
        while has_more:
            query = query_builder.limit(batch_size).offset(offset)
            sql, params = query.build()
            
            batch = self.db.execute_and_fetch(sql, params)
            if not batch:
                has_more = False
                break
            
            entities = [self._row_to_entity(row) for row in batch]
            
            if processor:
                result = processor(entities)
                results.append(result)
            
            offset += batch_size
            if len(batch) < batch_size:
                has_more = False
        
        return results
    
    # Schema inspection methods
    async def get_schema(self) -> Dict[str, Any]:
        """Get table schema information"""
        sql = f"PRAGMA table_info({self.table_name})"
        results = self.db.execute_and_fetch(sql)
        
        schema = []
        for row in results:
            schema.append({
                "cid": row[0],
                "name": row[1],
                "type": row[2],
                "notnull": row[3],
                "default_value": row[4],
                "pk": row[5]
            })
        
        return schema
    
    async def get_indexes(self) -> List[Dict[str, Any]]:
        """Get table index information"""
        sql = f"PRAGMA index_list({self.table_name})"
        results = self.db.execute_and_fetch(sql)
        
        indexes = []
        for row in results:
            index_name = row[1]
            index_info_sql = f"PRAGMA index_info({index_name})"
            index_info = self.db.execute_and_fetch(index_info_sql)
            
            columns = [info[2] for info in index_info]
            
            indexes.append({
                "seq": row[0],
                "name": index_name,
                "unique": row[2],
                "origin": row[3],
                "partial": row[4],
                "columns": columns
            })
        
        return indexes
    
    # Utility methods for data validation and transformation
    def validate_entity(self, entity: T) -> bool:
        """
        Validate entity against schema rules
        Returns True if valid, raises ValidationError if invalid
        """
        for field_name, field_meta in self.entity_meta['fields'].items():
            value = getattr(entity, field_name, None)
            
            # Required field check
            if field_meta.get('not_null', False) and value is None:
                raise ValueError(f"Field '{field_name}' cannot be null")
            
            # Type validation
            if value is not None:
                field_type = field_meta.get('type', '').upper()
                python_type = field_meta.get('python_type')
                
                # Basic type validation based on DuckDB types
                if field_type.startswith('VARCHAR') or field_type.startswith('TEXT'):
                    if not isinstance(value, str):
                        raise ValueError(f"Field '{field_name}' must be a string")
                
                elif field_type == 'INTEGER' or field_type == 'INT':
                    if not isinstance(value, int):
                        raise ValueError(f"Field '{field_name}' must be an integer")
                
                elif field_type == 'DOUBLE' or field_type == 'FLOAT':
                    if not isinstance(value, (float, int)):
                        raise ValueError(f"Field '{field_name}' must be a number")
                
                elif field_type == 'BOOLEAN':
                    if not isinstance(value, bool):
                        raise ValueError(f"Field '{field_name}' must be a boolean")
                
                elif field_type == 'DATE':
                    if not isinstance(value, (date, datetime, str)):
                        raise ValueError(f"Field '{field_name}' must be a date")
                
                elif field_type == 'TIMESTAMP':
                    if not isinstance(value, (datetime, str)):
                        raise ValueError(f"Field '{field_name}' must be a timestamp")
        
        # Custom validation if provided
        if hasattr(entity, 'validate'):
            entity.validate()
        
        return True