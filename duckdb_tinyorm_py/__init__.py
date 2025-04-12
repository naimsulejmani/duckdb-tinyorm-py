"""
duckdb-tinyorm-py: A tiny ORM for DuckDB in Python
"""

from .repository import DuckDbRepository, BaseRepository, QueryBuilder
from .decorators import entity, id_field, field, repository, index, get_entity_metadata
from .config import DuckDbLocation, DuckDbConfig
from .exceptions import DuckDbOrmError, EntityNotFoundError, ValidationError, InvalidQueryError

__all__ = [
    # Repository classes
    'DuckDbRepository',
    'BaseRepository',
    'QueryBuilder',
    
    # Decorators
    'entity',
    'id_field',
    'field', 
    'repository',
    'index',
    'get_entity_metadata',
    
    # Configuration
    'DuckDbLocation',
    'DuckDbConfig',
    
    # Exceptions
    'DuckDbOrmError',
    'EntityNotFoundError',
    'ValidationError',
    'InvalidQueryError',
]

__version__ = '0.1.2'