# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
import logging
import pandera as pa
from pandera import Column, Int, String

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_pandas_dataframe():
    """Returns a basic Pandas DataFrame for testing."""
    data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
    return pd.DataFrame(data)

@pytest.fixture
def sample_pandas_dataframe_with_nans():
    """Returns a Pandas DataFrame with NaNs for testing optional handling."""
    data = {'col1': [1, 2, np.nan], 'col2': ['a', np.nan, 'c']}
    return pd.DataFrame(data)

@pytest.fixture
def sample_pandera_schema():
    """Returns a basic Pandera schema."""
    schema = pa.DataFrameSchema(
        {
            "col1": Column(Int, coerce=False),
            "col2": Column(String, coerce=False),
        }
    )
    return schema

@pytest.fixture
def sample_polars_dataframe():
    """Returns a basic Polars DataFrame for testing."""
    if not HAS_POLARS:
        pytest.skip("Polars is not installed")
    data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}
    return pl.DataFrame(data) if HAS_POLARS else None

@pytest.fixture
def sample_polars_dataframe_with_nans():
    """Returns a Polars DataFrame with null values for testing optional handling."""
    if not HAS_POLARS:
        pytest.skip("Polars is not installed")
    data = {'col1': [1, 2, None], 'col2': ['a', None, 'c']}
    return pl.DataFrame(data) if HAS_POLARS else None

@pytest.fixture
def complex_pandas_dataframe():
    """Returns a more complex Pandas DataFrame for advanced testing scenarios."""
    data = {
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, np.nan, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e'],
        'bool_col': [True, False, True, False, True],
        'date_col': pd.date_range('2024-01-01', periods=5)
    }
    return pd.DataFrame(data)

@pytest.fixture
def complex_polars_dataframe():
    """Returns a more complex Polars DataFrame for advanced testing scenarios."""
    if not HAS_POLARS:
        pytest.skip("Polars is not installed")
    data = {
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, None, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e'],
        'bool_col': [True, False, True, False, True],
        'date_col': pd.date_range('2024-01-01', periods=5).tolist()
    }
    return pl.DataFrame(data) if HAS_POLARS else None