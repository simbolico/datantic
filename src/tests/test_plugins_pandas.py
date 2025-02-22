# tests/test_plugins_pandas.py
import pytest
import pandas as pd
import pandera as pa
from typing import List, Type
from pydantic import BaseModel

try:
    from src.datantic.plugins.pandas import DataFrameAccessor
    pandas_installed = True
    # Register the accessor globally for all tests
    pd.api.extensions.register_dataframe_accessor("datantic")(DataFrameAccessor)
except ImportError:
    pandas_installed = False

@pytest.mark.skipif(not pandas_installed, reason="Pandas is not installed")
def test_dataframe_accessor_creation(sample_pandas_dataframe):
    """Test DataFrameAccessor creation."""
    accessor = sample_pandas_dataframe.datantic
    assert isinstance(accessor, DataFrameAccessor)

@pytest.mark.skipif(not pandas_installed, reason="Pandas is not installed")
def test_dataframe_accessor_validate(sample_pandas_dataframe, sample_pandera_schema):
    """Test validate method."""
    validated_df = sample_pandas_dataframe.datantic.validate(sample_pandera_schema)
    assert isinstance(validated_df, pd.DataFrame)

@pytest.mark.skipif(not pandas_installed, reason="Pandas is not installed")
def test_dataframe_accessor_is_valid(sample_pandas_dataframe, sample_pandera_schema):
    """Test is_valid method."""
    is_valid = sample_pandas_dataframe.datantic.is_valid(sample_pandera_schema)
    assert isinstance(is_valid, bool)