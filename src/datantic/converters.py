# src/datantic/converters.py
import logging
from typing import Any

from .types import _PANDAS_INSTALLED, _POLARS_INSTALLED

logger = logging.getLogger(__name__)


def to_polars(pandas_df: Any) -> Any: # Any to avoid circular import, but should be pd.DataFrame
    """Converts a Pandas DataFrame to a Polars DataFrame."""
    if not _POLARS_INSTALLED:
        raise RuntimeError("Polars is not installed, but conversion to Polars DataFrame was requested.")
    if not _PANDAS_INSTALLED:
        logger.warning("Pandas is not installed, conversion may not be optimized.")

    try:
        import polars as pl

        return pl.from_pandas(pandas_df)
    except ImportError: # Redundant, but kept for extra safety
        raise RuntimeError("Polars import failed during conversion.") from None
    except Exception as e:
        logger.error(f"Error during Pandas to Polars conversion: {e}")
        return pandas_df # Fallback to original Pandas DataFrame (conversion failed)


def to_pandas(polars_df: Any) -> Any: # Any to avoid circular import, but should be pl.DataFrame
    """Converts a Polars DataFrame to a Pandas DataFrame."""
    try:
        import pandas as pd
        import polars as pl
        if isinstance(polars_df, pd.DataFrame):
            return polars_df
        if not isinstance(polars_df, pl.DataFrame):
            raise TypeError(f"Expected Polars or Pandas DataFrame, got {type(polars_df)}")
        return polars_df.to_pandas()
    except ImportError as e:
        if "pandas" in str(e):
            raise RuntimeError("Pandas is not installed. Please install pandas to use this functionality.") from e
        elif "polars" in str(e):
            raise RuntimeError("Polars is not installed. Please install polars to use this functionality.") from e
        # ADD THIS: Specifically check for pyarrow
        elif "pyarrow" in str(e):
            raise RuntimeError("pyarrow is not installed.  It is required for Polars -> Pandas conversion.") from e
        else:
            raise RuntimeError(f"Import error during conversion: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error during conversion: {e}") from e