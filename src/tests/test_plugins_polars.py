# tests/test_plugins_polars.py
import pytest
import pandera as pa
from typing import List, Type
from pydantic import BaseModel

try:
    import polars as pl
    from src.datantic.plugins.polars import PolarsDataFrameAccessor
    polars_installed = True
    # Register the accessor globally for all tests using the correct method
    pl.api.register_dataframe_namespace("datantic")(PolarsDataFrameAccessor)
except ImportError:
    polars_installed = False

@pytest.mark.skipif(not polars_installed, reason="Polars is not installed")
def test_polars_dataframe_accessor_creation():
    """Test PolarsDataFrameAccessor creation."""
    df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    accessor = df.datantic
    assert isinstance(accessor, PolarsDataFrameAccessor)

@pytest.mark.skipif(not polars_installed, reason="Polars is not installed")
def test_polars_dataframe_accessor_validate():
    """Test validate method."""
    from src.datantic.fields import DataFrameField
    from src.datantic.model import DataFrameModel
    
    class MySchema(DataFrameModel):
        col1: int = DataFrameField()
        col2: str = DataFrameField()

    df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    validated_df = df.datantic.validate(MySchema)
    assert isinstance(validated_df, pl.DataFrame)

@pytest.mark.skipif(not polars_installed, reason="Polars is not installed")
def test_polars_dataframe_accessor_is_valid():
    """Test is_valid method."""
    from src.datantic.fields import DataFrameField
    from src.datantic.model import DataFrameModel
    
    class MySchema(DataFrameModel):
        col1: int = DataFrameField()
        col2: str = DataFrameField()

    df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    is_valid = df.datantic.is_valid(MySchema)
    assert isinstance(is_valid, bool)