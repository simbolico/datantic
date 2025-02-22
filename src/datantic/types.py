# ./src/datantic/types.py
# ./src/datantic/types.py
from typing import Annotated, Any, Callable, TypeVar, Union

import pandera as pa
from pydantic import BeforeValidator, ValidationError

try:
    import pandas as pd
    _PANDAS_INSTALLED = True
except ImportError:
    pd = None
    _PANDAS_INSTALLED = False

try:
    import polars as pl
    _POLARS_INSTALLED = True
except ImportError:
    _POLARS_INSTALLED = False


TypeT = TypeVar("TypeT")


def _coerce_nan_to_none(x: Any) -> Any:
    """Coerces Pandas NaN values to None for Pydantic compatibility."""
    if not _PANDAS_INSTALLED:
        raise RuntimeError("Pandas is required for NaN coercion. Please install it.")
    if pd is None: # Additional check in case pd is somehow None despite _PANDAS_INSTALLED
        raise RuntimeError("Pandas library import failed unexpectedly.")
    if pd.isna(x):
        return None
    return x

def is_polars_dataframe(obj: Any) -> bool:
    """Checks if an object is a Polars DataFrame (robust to import errors)."""
    if not _POLARS_INSTALLED: # Check flag directly for performance
        return False
    try:
        import polars as pl # Import locally to avoid top-level dependency
        return isinstance(obj, pl.DataFrame)
    except ImportError: # Redundant, but kept for safety. Flag should prevent reaching here if not installed
        return False

# Define a custom Optional type using Annotated and BeforeValidator.
Optional = Annotated[TypeT | None, BeforeValidator(_coerce_nan_to_none)]
"""Datantic's `Optional` type, handling NaNs correctly."""

# Define the type alias for the error handler.  It's a callable that
# takes a ValidationError or SchemaErrors and returns nothing.
ErrorHandler = Callable[[Union[ValidationError, pa.errors.SchemaErrors]], None]
"""Type alias for custom error handlers."""