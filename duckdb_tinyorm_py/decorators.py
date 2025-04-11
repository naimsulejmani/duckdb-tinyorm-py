"""
Decorator module for entity and field definitions with additional features
"""
from typing import Type, TypeVar, Generic, Dict, Any, List, Optional, get_type_hints, Union
import inspect
from enum import Enum

# Entity and field registry
_ENTITY_REGISTRY = {}
_REPOSITORY_REGISTRY = {}

T = TypeVar('T')


def entity(cls=None, *, table_name: Optional[str] = None):
    """
    Decorator to mark a class as a database entity.

    Args:
        cls: The class to decorate.
        table_name: Optional custom table name (defaults to lowercase class name).

    Returns:
        Decorated class.
    """
    def wrap(cls):
        cls_name = cls.__name__
        actual_table_name = table_name or cls_name.lower()

        # Initialize or update entity metadata in the registry
        if cls_name not in _ENTITY_REGISTRY:
            _ENTITY_REGISTRY[cls_name] = {
                'class': cls,
                'table_name': actual_table_name,
                'fields': {},
                'id_field': None,
                'indexes': [],
                '_fields_processed': False # Flag to track if fields were scanned
            }
        else:
            # Update existing entry, ensuring essential keys exist
            _ENTITY_REGISTRY[cls_name]['class'] = cls
            _ENTITY_REGISTRY[cls_name]['table_name'] = actual_table_name
            _ENTITY_REGISTRY[cls_name].setdefault('fields', {})
            _ENTITY_REGISTRY[cls_name].setdefault('id_field', None)
            _ENTITY_REGISTRY[cls_name].setdefault('indexes', [])
            _ENTITY_REGISTRY[cls_name].setdefault('_fields_processed', False)


        # Add a convenient method to access entity metadata
        # This will trigger the scan in get_entity_metadata if needed
        cls._get_entity_meta = lambda: get_entity_metadata(cls)

        return cls

    if cls is None:
        return wrap
    return wrap(cls)


def _get_sql_type_from_python_type(py_type):
    """Helper function to map Python types to SQL types, handling Optional."""
    origin = getattr(py_type, '__origin__', None)
    args = getattr(py_type, '__args__', ())

    # Handle Optional[T] -> T or Union[T, None] -> T
    if origin is Union and type(None) in args:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            py_type = non_none_args[0] # Use the underlying type

    # Type mappings
    if py_type is str: return 'VARCHAR'
    if py_type is int: return 'INTEGER'
    if py_type is float: return 'DOUBLE'
    if py_type is bool: return 'BOOLEAN'
    if py_type is bytes: return 'BLOB'
    if py_type is list or py_type is dict: return 'JSON'
    # Consider adding datetime/date if used
    # from datetime import date, datetime
    # if py_type is date: return 'DATE'
    # if py_type is datetime: return 'TIMESTAMP'

    # Default fallback
    return 'VARCHAR'


def field(data_type: str = None, *,
          not_null: bool = False,
          unique: bool = False,
          default: Any = None,
          comment: str = None):
    """
    Decorator to define a field's data type and constraints.
    Should be applied to property getter methods.
    """
    def decorator(method):
        # Store metadata on the method itself for later collection by get_entity_metadata
        method._field_meta = {
            'decorator': 'field',
            'data_type': data_type,
            'not_null': not_null,
            'unique': unique,
            'default': default,
            'comment': comment
        }
        # Return the original method, @property will handle the rest
        return method
    return decorator


def id_field(data_type: str = 'INTEGER', *, auto_increment: bool = False):
    """
    Decorator to mark a field as the primary key.
    Should be applied to the property getter method for the ID.
    """
    def decorator(method):
        # Store metadata on the method itself for later collection by get_entity_metadata
        method._field_meta = {
            'decorator': 'id_field',
            'data_type': data_type, # Store specified type
            'auto_increment': auto_increment
        }
        # Return the original method, @property will handle the rest
        return method
    return decorator


def index(fields: Union[str, List[str]], *,
          name: Optional[str] = None,
          unique: bool = False,
          index_type: str = ""):
    """
    Decorator to add an index to an entity. Applied to the class.
    """
    def decorator(cls):
        cls_name = cls.__name__

        # Ensure entity metadata exists
        if cls_name not in _ENTITY_REGISTRY:
             _ENTITY_REGISTRY[cls_name] = {
                'class': cls, 'table_name': cls_name.lower(), 'fields': {},
                'id_field': None, 'indexes': [], '_fields_processed': False
            }
        _ENTITY_REGISTRY[cls_name].setdefault('indexes', [])

        field_list = [fields] if isinstance(fields, str) else fields
        idx_name = name or f"idx_{cls_name.lower()}_{'_'.join(field_list)}"

        _ENTITY_REGISTRY[cls_name]['indexes'].append({
            'name': idx_name, 'fields': field_list, 'unique': unique, 'type': index_type
        })
        return cls
    return decorator


def repository(entity_cls):
    """
    Decorator to associate a repository with an entity. Applied to the repository class.
    """
    def decorator(repo_cls):
        entity_key = entity_cls.__name__
        _REPOSITORY_REGISTRY[repo_cls.__name__] = {
            'class': repo_cls, 'entity': entity_key
        }
        repo_cls._entity_class = entity_cls
        repo_cls._get_registry_info = lambda: _REPOSITORY_REGISTRY[repo_cls.__name__]
        return repo_cls
    return decorator


def get_entity_metadata(cls_or_name):
    """
    Get entity metadata from registry, processing field decorators if necessary.
    """
    key = cls_or_name.__name__ if isinstance(cls_or_name, type) else cls_or_name
    cls = cls_or_name if isinstance(cls_or_name, type) else None

    if key not in _ENTITY_REGISTRY:
        if cls: # If class object exists, try creating basic metadata entry
             _ENTITY_REGISTRY[key] = {
                'class': cls, 'table_name': key.lower(), 'fields': {},
                'id_field': None, 'indexes': [], '_fields_processed': False
            }
        else: # Cannot proceed without class or existing entry
            return None

    meta = _ENTITY_REGISTRY[key]

    # Scan class members for decorated fields only if not already processed
    if cls and not meta.get('_fields_processed', False):
         meta['fields'] = {} # Reset fields for fresh scan
         meta['id_field'] = None # Reset id_field
         all_hints = {}
         try:
             # Get type hints for the class itself
             all_hints = get_type_hints(cls)
         except Exception:
             # Handle cases where getting class hints might fail
             pass

         for name, member in inspect.getmembers(cls):
             # Check if the member is a property getter with our temporary metadata
             if hasattr(member, 'fget') and hasattr(member.fget, '_field_meta'):
                 method = member.fget # Get the original decorated method from property
                 field_meta_info = getattr(method, '_field_meta', None)
                 if not field_meta_info: continue # Skip if no metadata found

                 decorator_type = field_meta_info['decorator']
                 field_name = name # The name of the property

                 # Get Python type hint from the method's return annotation or class hint
                 python_type = None
                 try:
                     method_hints = get_type_hints(method)
                     python_type = method_hints.get('return')
                 except Exception: # Fallback to class hints if method hints fail
                     python_type = all_hints.get(field_name)

                 python_type_name = None
                 if python_type:
                     origin = getattr(python_type, '__origin__', None)
                     args = getattr(python_type, '__args__', ())
                     if origin is Union and type(None) in args:
                         non_none = [a for a in args if a is not type(None)]
                         if len(non_none) == 1:
                             base_type = non_none[0]
                             python_type_name = getattr(base_type, '__name__', str(base_type))
                         else: python_type_name = str(python_type) # Handle complex Unions
                     else:
                         python_type_name = getattr(python_type, '__name__', str(python_type))

                 # Determine SQL type
                 sql_type = field_meta_info.get('data_type')
                 if sql_type is None and python_type:
                     sql_type = _get_sql_type_from_python_type(python_type)
                 elif sql_type is None: # Default SQL type if not specified and hint unavailable
                     sql_type = 'INTEGER' if decorator_type == 'id_field' else 'VARCHAR'

                 # Build the metadata dictionary for this field
                 current_field_meta = {
                     'name': field_name,
                     'type': sql_type.upper(),
                     'python_type': python_type_name,
                     'is_id': False, 'not_null': False, 'unique': False,
                     'default': None, 'comment': None, 'auto_increment': False
                 }

                 if decorator_type == 'id_field':
                     current_field_meta.update({
                         'is_id': True, 'not_null': True, 'unique': True,
                         'auto_increment': field_meta_info.get('auto_increment', False),
                         'type': (field_meta_info.get('data_type') or 'INTEGER').upper() # Use specified or default ID type
                     })
                     meta['id_field'] = field_name # Set the primary key field name
                 elif decorator_type == 'field':
                     current_field_meta.update({
                         'not_null': field_meta_info.get('not_null', False),
                         'unique': field_meta_info.get('unique', False),
                         'default': field_meta_info.get('default'),
                         'comment': field_meta_info.get('comment'),
                         'type': sql_type.upper() # Ensure type is stored
                     })

                 meta['fields'][field_name] = current_field_meta

         meta['_fields_processed'] = True # Mark fields as processed

    return meta