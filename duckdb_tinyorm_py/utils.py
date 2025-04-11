"""
Utility classes and functions
"""
import os
import json
import yaml
from typing import Dict, Any, Optional
from .config import DuckDbConfig, DuckDbLocation


class ConfigLoader:
    """Utility for loading configuration from files"""
    
    @staticmethod
    def from_json(filepath: str) -> Dict[str, Any]:
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def from_yaml(filepath: str) -> Dict[str, Any]:
        """Load config from YAML file"""
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def from_env() -> Dict[str, Any]:
        """Load config from environment variables"""
        prefix = "DUCKDB_"
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                config[config_key] = value
        
        return config
    
    @staticmethod
    def create_db_config(config: Dict[str, Any]) -> DuckDbConfig:
        """Create DuckDbConfig from config dict"""
        name = config.get('name', 'default')
        
        # Determine location
        location_str = config.get('location', 'memory').upper()
        location = DuckDbLocation.FILE if location_str == 'FILE' else DuckDbLocation.MEMORY
        
        filename = config.get('filename')
        settings = config.get('settings', {})
        load_extensions = config.get('load_extensions', False)
        extensions = config.get('extensions', [])
        
        return DuckDbConfig(
            name=name,
            location=location,
            filename=filename,
            settings=settings,
            load_extensions=load_extensions,
            extensions=extensions
        )