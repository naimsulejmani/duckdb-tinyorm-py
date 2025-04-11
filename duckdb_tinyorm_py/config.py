"""
Configuration module for DuckDB connections with advanced features
"""
from enum import Enum, auto
from typing import Optional, Dict, Any


class DuckDbLocation(Enum):
    """Enum for DuckDB database location types"""
    MEMORY = auto()
    FILE = auto()


class DuckDbConfig:
    """Configuration class for DuckDB connections"""
    
    def __init__(self, 
                 name: str = "default", 
                 location: DuckDbLocation = DuckDbLocation.MEMORY, 
                 filename: Optional[str] = None,
                 settings: Dict[str, Any] = None,
                 load_extensions: bool = False,
                 extensions: Optional[list] = None):
        """
        Initialize DuckDB configuration
        
        Args:
            name: Name of the connection (used as identifier for instance management)
            location: Database location (memory or file)
            filename: Database filename (required if location is FILE)
            settings: DuckDB configuration settings
            load_extensions: Whether to load extensions
            extensions: List of extensions to load
        """
        self.name = name
        self.location = location
        self.filename = filename
        self.settings = settings or {}
        self.load_extensions = load_extensions
        self.extensions = extensions or []
        
        # Validate config
        if self.location == DuckDbLocation.FILE and not self.filename:
            raise ValueError("Filename must be provided for FILE location")


class Index:
    """Configuration class for database indexes"""
    
    def __init__(self, 
                 fields: list, 
                 name: Optional[str] = None,
                 unique: bool = False,
                 type: str = ""):
        """
        Initialize an index configuration
        
        Args:
            fields: List of fields to index
            name: Name of the index
            unique: Whether the index is unique
            type: Index type (e.g. HASH)
        """
        self.fields = fields
        self.name = name
        self.unique = unique
        self.type = type
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            "fields": self.fields,
            "name": self.name,
            "unique": self.unique,
            "type": self.type
        }