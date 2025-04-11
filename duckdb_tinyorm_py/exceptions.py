"""
Custom exceptions for DuckDB ORM
"""

class DuckDbOrmError(Exception):
    """Base exception for DuckDB ORM"""
    pass


class EntityNotFoundError(DuckDbOrmError):
    """Entity not found exception"""
    pass


class ValidationError(DuckDbOrmError):
    """Entity validation error"""
    pass


class InvalidQueryError(DuckDbOrmError):
    """Invalid query error"""
    pass


class ConnectionError(DuckDbOrmError):
    """Database connection error"""
    pass


class TransactionError(DuckDbOrmError):
    """Transaction error"""
    pass


class MigrationError(DuckDbOrmError):
    """Migration error"""
    pass