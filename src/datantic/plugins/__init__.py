"""
`datantic.plugins`: Extending `datantic` with DataFrame Library Support

This module provides plugins that extend `datantic` with functionality
specific to particular DataFrame libraries (Pandas and Polars).

The plugins are loaded *conditionally*, based on whether the corresponding
library is installed.  This ensures that `datantic` remains lightweight
and doesn't have unnecessary dependencies.

If a plugin is successfully loaded, it typically adds a DataFrame accessor
(e.g., `.datantic`) to the DataFrame class, providing convenient methods
for validation.
"""

import logging

logger = logging.getLogger(__name__)

# Conditional import and accessor registration for Pandas
try:
    import pandas as pd
    from .pandas_plugin import DataFrameAccessor  # Import from the module

    pd.api.extensions.register_dataframe_accessor("datantic")(DataFrameAccessor)
    logger.info("Pandas plugin loaded successfully.")
except ImportError:
    logger.warning(
        "Pandas is not installed. The datantic Pandas plugin will not be available."
    )
    DataFrameAccessor = None

# Conditional import and accessor registration for Polars
try:
    import polars as pl
    from .polars_plugin import PolarsDataFrameAccessor  # Import from the module

    pl.api.extensions.register_dataframe_accessor("datantic")(
        PolarsDataFrameAccessor
    )
    logger.info("Polars plugin loaded successfully.")
except ImportError:
    logger.warning(
        "Polars is not installed. The datantic Polars plugin will not be available."
    )
    PolarsDataFrameAccessor = None


__all__ = []  # Start with an empty __all__
if DataFrameAccessor is not None:
    __all__.append("DataFrameAccessor")  # Add Pandas accessor if loaded
if PolarsDataFrameAccessor is not None:
    __all__.append("PolarsDataFrameAccessor")  # Add Polars accessor if loaded